#include <stdio.h>
#include <vector>
using namespace std;

#include <opencv2\opencv.hpp>
using namespace cv;

#ifdef _DEBUG
#pragma comment(lib,"opencv_world341d.lib")
#else
#pragma comment(lib,"opencv_world341.lib")
#endif

Point pStart, pEnd;
bool pause;

Point getCenter(Rect rt)
{
	return Point(rt.x + rt.width / 2, rt.y + rt.height / 2);
}

Rect getRect(Point p, int w, int h)
{
	return Rect(p.x - w / 2, p.y - h / 2, w, h);
}

vector<float> getFeature(Mat img)
{
	vector<float> feature(3,0);

	float total_R=0;
	float total_G=0;
	float total_B=0;

	for (int x = 0; x < img.cols; x++)
	{
		for (int y = 0; y < img.rows; y++)
		{
			feature[0] += img.ptr<uchar>(y)[3 * x + 0];
			feature[1] += img.ptr<uchar>(y)[3 * x + 1];
			feature[2] += img.ptr<uchar>(y)[3 * x + 2];
		}
	}

	float total = feature[0] + feature[1] + feature[2];
	feature[0] /= total;
	feature[1] /= total;
	feature[2] /= total;

	return feature;
}

float featureDist(vector<float> f1, vector<float> f2)
{
	float dot = 0;
	float sum_f1 = 0;
	float sum_f2 = 0;
	for (int i = 0; i < f1.size(); i++)
	{
		dot += f1[i] * f2[i];
		sum_f1 += f1[i] * f1[i];
		sum_f2 += f2[i] * f2[i];
	}

	return dot / (sqrt(sum_f1)*sqrt(sum_f2));
}

Rect FindObjectInNextFrame(Mat frame, Mat prev_frame, Rect prev_frame_rect)
{
	//得到前一帧目标的向量
	vector<float> prev_feature = getFeature(prev_frame(prev_frame_rect));

	//定义h的长度
	int h = 10;

	//
	Point center = getCenter(prev_frame_rect);

	//获得半径为h的圆内所有的备选框
	vector<Rect> prepare_rects;
	vector<Point> prepare_points;
	for (int x = center.x - h; x < center.x + 2 * h; x++)
	{
		for (int y = center.y - h; y < center.y + 2 * h; y++)
		{
			float dist = sqrt((x - center.x)*(x - center.x) + (y - center.y)*(y - center.y));
			if (dist <= h)
			{
				Rect obj_rect = getRect(Point(x, y), prev_frame_rect.width,
					prev_frame_rect.height);
				if (obj_rect.x < 0 || obj_rect.y < 0 || obj_rect.width < 0 || obj_rect.height < 0 ||
					obj_rect.x + obj_rect.width > frame.cols ||
					obj_rect.y + obj_rect.height > frame.rows)
				{
					continue;
				}

				prepare_rects.push_back(obj_rect);
				prepare_points.push_back(getCenter(obj_rect));
			}
		}
	}

	//得到所有备选框的向量
	vector<vector<float>> prepare_feature;
	for (int i = 0; i < prepare_rects.size(); i++)
	{
		prepare_feature.push_back(getFeature(frame(prepare_rects[i])));
	}

	//定义阈值
	float theashold_score = 0.9999;

	//计算所有备选框的向量是否符合标准
	vector<bool> prepare_flag; bool flag = false;
	for (int i = 0; i < prepare_rects.size(); i++)
	{
		/*char str[100];
		sprintf_s(str, "D:\\%d_%f.jpg", i, featureDist(prev_feature, prepare_feature[i]));
		imwrite(str,frame(prepare_rects[i]));*/

		if (featureDist(prev_feature, prepare_feature[i]) > theashold_score)
		{
			flag = true;
			prepare_flag.push_back(true);
		}
		else
		{
			prepare_flag.push_back(false);
		}
	}

	if (flag == false)
	{
		return Rect(-1, -1, 0, 0);
	}

	//计算所有符合标准向量的均值坐标
	Point mh = Point(0,0);
	int cnt = 0;
	for (int i = 0; i < prepare_flag.size(); i++)
	{
		Point pt = getCenter(prepare_rects[i]);
		if (prepare_flag[i])
		{
			mh.x += (pt.x - center.x);
			mh.y += (pt.y - center.y);
			cnt++;
		}
	}
	Point mean_point = Point(center.x + mh.x / cnt, center.y + mh.y / cnt);

	//漂移
	Rect rt = getRect(mean_point, prev_frame_rect.width,
		prev_frame_rect.height);

	return rt;
}

void onMouse(int event, int x, int y, int flags, void *param)
{
	Mat* img = (Mat*)param;
	
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
		pStart = Point(x, y);
		pause = true;
		break;
	}
	case CV_EVENT_LBUTTONUP:
	{
		pEnd = Point(x, y);
		pause = false;
		break;
	}
	case CV_EVENT_MOUSEMOVE:
	{
		if (pause)
		{
			pEnd = Point(x, y);
			
			Mat temp;
			img->copyTo(temp);
			rectangle(temp, pStart, pEnd, Scalar(0, 0, 255), 2);
			imshow("show", temp);
		}
		break;
	}
	}

	return;
}

int main()
{
	float scale = 2;
	vector<Point> record;

	//加载视频
	VideoCapture video;
	video.open("E:\\课程\\音视频信号处理\\box.mp4");

	//处理第一帧
	Mat img, prev_img;
	video >> img;
	resize(img, img, Size(img.cols / scale, img.rows / scale));
	img.copyTo(prev_img);

	namedWindow("show");
	setMouseCallback("show", onMouse, &img);
	imshow("show", img);
	waitKey();

	int frame_counter = 0;

	Rect obj_rect = Rect(pStart, pEnd);
	while (true)
	{
		video >> img;
		if (img.empty())
		{
			break;
		}
		resize(img, img, Size(img.cols / scale, img.rows / scale));

		//mean-shift
		obj_rect = FindObjectInNextFrame(img, prev_img, obj_rect);
		if (obj_rect.x < 0 || obj_rect.y < 0 || obj_rect.width < 0 || obj_rect.height < 0)
		{
			printf("obj miss!");
			break;
		}

		img.copyTo(prev_img);

		record.push_back(getCenter(obj_rect));
		for (int i = 1; i < record.size(); i++)
		{
			line(img, record[i-1], record[i], Scalar(255, 0, 0));
		}
		rectangle(img, obj_rect, Scalar(0, 0, 255), 2);

		imshow("show", img);
		if (27 == waitKey(0))
		{
			break;
		}
	}

	return 0;
}