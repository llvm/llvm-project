// RUN: %clang_cc1 -ast-print -verify -triple=x86_64-pc-win32 -fms-compatibility %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fms-compatibility -emit-pch -o %t %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fms-compatibility -include-pch %t -verify %s -ast-print -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

class Test1 {
private:
  int x_;

public:
  Test1(int x) : x_(x) {}
  __declspec(property(get = get_x)) int X;
  int get_x() const { return x_; }
  static Test1 *GetTest1() { return new Test1(10); }
};

class S {
public:
  __declspec(property(get=GetX,put=PutX)) int x[];
  int GetX(int i, int j) { return i+j; }
  void PutX(int i, int j, int k) { j = i = k; }
  __declspec(property(get=GetY,put=PutY)) int y[][];
  int GetY(int i, int j) { return i+j; }
  void PutY(int i, int j, int k) { j = i = k; }
  __declspec(property(get=GetZ,put=PutZ)) int z[][][];
  int GetZ(int i, int j, int k);
  void PutZ(int i, int j, int k, int val);
};

template <typename T>
class St {
public:
  __declspec(property(get=GetX,put=PutX)) T x[];
  T GetX(T i, T j) { return i+j; }
  T PutX(T i, T j, T k) { return j = i = k; }
  __declspec(property(get=GetY,put=PutY)) T y[][];
  T GetY(T i, T j) { return i+j; }
  T PutY(T i, T j, T k) { return j = i = k; }
  __declspec(property(get=GetZ,put=PutZ)) T z[][][];
  T GetZ(T i, T j, T k) { return i+j+k; }
  T PutZ(T i, T j, T k, T v) { return j = i = k = v; }
  ~St() { x[0][0] = x[1][1]; y[0][0] = x[1][1]; z[0][1][2] = z[2][1][0]; }
};

// CHECK: this->x[0][0] = this->x[1][1];
// CHECK: this->y[0][0] = this->x[1][1];
// CHECK: this->z[0][1][2] = this->z[2][1][0];
// CHECK: this->x[0][0] = this->x[1][1];
// CHECK: this->y[0][0] = this->x[1][1];
// CHECK: this->z[0][1][2] = this->z[2][1][0];

// CHECK-LABEL: main
int main(int argc, char **argv) {
  S *p1 = 0;
  St<float> *p2 = 0;
  // CHECK: St<int> a;
  St<int> a;
  // CHECK-NEXT: int j = (p1->x)[223][11];
  int j = (p1->x)[223][11];
  // CHECK-NEXT: (p1->x[23])[1] = j;
  (p1->x[23])[1] = j;
  // CHECK-NEXT: int k = (p1->y)[223][11];
  int k = (p1->y)[223][11];
  // CHECK-NEXT: (p1->y[23])[1] = k;
  (p1->y[23])[1] = k;
  // CHECK-NEXT: int k3 = p1->z[1][2][3];
  int k3 = p1->z[1][2][3];
  // CHECK-NEXT: p1->z[0][2][1] = k3;
  p1->z[0][2][1] = k3;
  // CHECK-NEXT: float j1 = (p2->x[223][11]);
  float j1 = (p2->x[223][11]);
  // CHECK-NEXT: ((p2->x)[23])[1] = j1;
  ((p2->x)[23])[1] = j1;
  // CHECK-NEXT: float k1 = (p2->y[223][11]);
  float k1 = (p2->y[223][11]);
  // CHECK-NEXT: ((p2->y)[23])[1] = k1;
  ((p2->y)[23])[1] = k1;
  // CHECK-NEXT: ++(((p2->x)[23])[1]);
  ++(((p2->x)[23])[1]);
  // CHECK-NEXT: j1 = ((p2->x)[23])[1] = j1;
  j1 = ((p2->x)[23])[1] = j1;
  // CHECK-NEXT: return Test1::GetTest1()->X;
  return Test1::GetTest1()->X;
}
#endif // HEADER
