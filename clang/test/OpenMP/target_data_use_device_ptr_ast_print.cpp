// RxUN: %clang_cc1 -verify -fopenmp -std=c++11 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct ST {
  int *a;
};
struct SA {
  int i, j;
  int *k = &j;
  int *&z = k;
  void func(int arg) {
#pragma omp target data map(tofrom: i) use_device_ptr(k)
    {}
#pragma omp target data map(tofrom: i) use_device_ptr(z)
    {}
  return;
 }
};
// CHECK: struct SA
// CHECK: void func(
// CHECK: #pragma omp target data map(tofrom: this->i) use_device_ptr(this->k){{$}}
// CHECK: #pragma omp target data map(tofrom: this->i) use_device_ptr(this->z)
struct SB {
  unsigned A;
  unsigned B;
  float Arr[100];
  float *Ptr;
  float *foo() {
    return &Arr[0];
  }
};

struct SC {
  unsigned A : 2;
  unsigned B : 3;
  unsigned C;
  unsigned D;
  float Arr[100];
  SB S;
  SB ArrS[100];
  SB *PtrS;
  SB *&RPtrS;
  float *Ptr;

  SC(SB *&_RPtrS) : RPtrS(_RPtrS) {}
};

union SD {
  unsigned A;
  float B;
};

struct S1;
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(S2 &s2):a(s2.a) { }
  static float S2s;
  static const float S2sc;
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
  S3(S3 &s3):a(s3.a) { }
};
const S3 c;
const S3 ca[5];
extern const int f;
class S4 {
  int a;
  S4();
  S4(const S4 &s4);
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {}
  S5(const S5 &s5):a(s5.a) { }
public:
  S5(int v):a(v) { }
};

S3 h;
#pragma omp threadprivate(h)

typedef int from;

template <typename T>
T tmain(T argc) {
  T i;
  T &j = i;
  T *k = &j;
  T *&z = k;
#pragma omp target data map(tofrom: i) use_device_ptr(k)
  {}
#pragma omp target data map(tofrom: i) use_device_ptr(z)
  {}
  return 0;
}

// CHECK: template<> int tmain<int>(int argc) {
// CHECK-NEXT: int i;
// CHECK-NEXT: int &j = i;
// CHECK-NEXT: int *k = &j;
// CHECK-NEXT: int *&z = k;
// CHECK-NEXT: #pragma omp target data map(tofrom: i) use_device_ptr(k)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target data map(tofrom: i) use_device_ptr(z)

// CHECK: template<> int *tmain<int *>(int *argc) {
// CHECK-NEXT: int *i;
// CHECK-NEXT: int *&j = i;
// CHECK-NEXT: int **k = &j;
// CHECK-NEXT: int **&z = k;
// CHECK-NEXT: #pragma omp target data map(tofrom: i) use_device_ptr(k)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target data map(tofrom: i) use_device_ptr(z)

// CHECK-LABEL: int main(int argc, char **argv) {
int main(int argc, char **argv) {
  int i;
  int &j = i;
  int *k = &j;
  int *&z = k;
// CHECK-NEXT: int i;
// CHECK-NEXT: int &j = i;
// CHECK-NEXT: int *k = &j;
// CHECK-NEXT: int *&z = k;
#pragma omp target data map(tofrom: i) use_device_ptr(k)
// CHECK-NEXT: #pragma omp target data map(tofrom: i) use_device_ptr(k)
  {}
// CHECK-NEXT: {
// CHECK-NEXT: }
#pragma omp target data map(tofrom: i) use_device_ptr(z)
// CHECK-NEXT: #pragma omp target data map(tofrom: i) use_device_ptr(z)
  {}
  return tmain<int>(argc) + (*tmain<int*>(&argc));
}

#endif
