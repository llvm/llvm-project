#include <memory>

struct Point {
  int x, y;
};

int main() {
  Point pt = { 1, 2 };
  Point points[] = {{1010,2020}, {3030,4040}, {5050,6060}};
  Point *pt_ptr = &points[1];
  Point &pt_ref = pt;
  std::shared_ptr<Point> pt_sp(new Point{111,222});
  return 0; // Set a breakpoint here
}

