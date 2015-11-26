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
};

template <typename T>
class St {
public:
  __declspec(property(get=GetX,put=PutX)) T x[];
  T GetX(T i, T j) { return i+j; }
  void PutX(T i, T j, T k) { j = i = k; }
  ~St() { x[0][0] = x[1][1]; }
};

// CHECK: this->x[0][0] = this->x[1][1];
// CHECK: this->x[0][0] = this->x[1][1];

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
  // CHECK-NEXT: float j1 = (p2->x[223][11]);
  float j1 = (p2->x[223][11]);
  // CHECK-NEXT: ((p2->x)[23])[1] = j1;
  ((p2->x)[23])[1] = j1;
  // CHECK-NEXT: ++(((p2->x)[23])[1]);
  ++(((p2->x)[23])[1]);
  // CHECK-NEXT: return Test1::GetTest1()->X;
  return Test1::GetTest1()->X;
}
#endif // HEADER
