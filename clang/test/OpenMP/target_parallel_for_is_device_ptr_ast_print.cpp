// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

struct ST {
  int *a;
};
typedef int arr[10];
typedef ST STarr[10];
struct SA {
  const int da[5] = { 0 };
  ST g[10];
  STarr &rg = g;
  int i;
  int &j = i;
  int *k = &j;
  int *&z = k;
  int aa[10];
  arr &raa = aa;
  void func(int arg) {
#pragma omp target parallel for is_device_ptr(k)
    for (int i=0; i<100; i++) foo();

#pragma omp target parallel for is_device_ptr(z)
    for (int i=0; i<100; i++) foo();

#pragma omp target parallel for is_device_ptr(aa) // OK
    for (int i=0; i<100; i++) foo();

#pragma omp target parallel for is_device_ptr(raa) // OK
    for (int i=0; i<100; i++) foo();

#pragma omp target parallel for is_device_ptr(g) // OK
    for (int i=0; i<100; i++) foo();

#pragma omp target parallel for is_device_ptr(rg) // OK
    for (int i=0; i<100; i++) foo();

#pragma omp target parallel for is_device_ptr(da) // OK
    for (int i=0; i<100; i++) foo();

    return;
  }
};
// CHECK: struct SA
// CHECK-NEXT: const int da[5] = {0};
// CHECK-NEXT: ST g[10];
// CHECK-NEXT: STarr &rg = this->g;
// CHECK-NEXT: int i;
// CHECK-NEXT: int &j = this->i;
// CHECK-NEXT: int *k = &this->j;
// CHECK-NEXT: int *&z = this->k;
// CHECK-NEXT: int aa[10];
// CHECK-NEXT: arr &raa = this->aa;
// CHECK-NEXT: func(
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(this->k){{$}}
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(this->z)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(this->aa)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(this->raa)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(this->g)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(this->rg)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(this->da)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

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

typedef struct {
  int a;
} S6;

template <typename T>
T tmain(T argc) {
  const T da[5] = { 0 };
  S6 h[10];
  auto &rh = h;
  T i;
  T &j = i;
  T *k = &j;
  T *&z = k;
  T aa[10];
  auto &raa = aa;
#pragma omp target parallel for is_device_ptr(k)
  for (int i=0; i<100; i++) foo();
#pragma omp target parallel for is_device_ptr(z)
  for (int i=0; i<100; i++) foo();
#pragma omp target parallel for is_device_ptr(aa)
  for (int i=0; i<100; i++) foo();
#pragma omp target parallel for is_device_ptr(raa)
  for (int i=0; i<100; i++) foo();
#pragma omp target parallel for is_device_ptr(h)
  for (int i=0; i<100; i++) foo();
#pragma omp target parallel for is_device_ptr(rh)
  for (int i=0; i<100; i++) foo();
#pragma omp target parallel for is_device_ptr(da)
  for (int i=0; i<100; i++) foo();
  return 0;
}

// CHECK: template<> int tmain<int>(int argc) {
// CHECK-NEXT: const int da[5] = {0};
// CHECK-NEXT: S6 h[10];
// CHECK-NEXT: auto &rh = h;
// CHECK-NEXT: int i;
// CHECK-NEXT: int &j = i;
// CHECK-NEXT: int *k = &j;
// CHECK-NEXT: int *&z = k;
// CHECK-NEXT: int aa[10];
// CHECK-NEXT: auto &raa = aa;
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(k)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(z)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(aa)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(raa)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(h)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(rh)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(da)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

// CHECK: template<> int *tmain<int *>(int *argc) {
// CHECK-NEXT: int *const da[5] = {0};
// CHECK-NEXT: S6 h[10];
// CHECK-NEXT: auto &rh = h;
// CHECK-NEXT: int *i;
// CHECK-NEXT: int *&j = i;
// CHECK-NEXT: int **k = &j;
// CHECK-NEXT: int **&z = k;
// CHECK-NEXT: int *aa[10];
// CHECK-NEXT: auto &raa = aa;
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(k)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(z)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(aa)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(raa)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(h)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(rh)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(da)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

// CHECK-LABEL: int main(int argc, char **argv) {
int main(int argc, char **argv) {
  const int da[5] = { 0 };
  S6 h[10];
  auto &rh = h;
  int i;
  int &j = i;
  int *k = &j;
  int *&z = k;
  int aa[10];
  auto &raa = aa;
// CHECK-NEXT: const int da[5] = {0};
// CHECK-NEXT: S6 h[10];
// CHECK-NEXT: auto &rh = h;
// CHECK-NEXT: int i;
// CHECK-NEXT: int &j = i;
// CHECK-NEXT: int *k = &j;
// CHECK-NEXT: int *&z = k;
// CHECK-NEXT: int aa[10];
// CHECK-NEXT: auto &raa = aa;
#pragma omp target parallel for is_device_ptr(k)
  for (int i=0; i<100; i++) foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(k)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

#pragma omp target parallel for is_device_ptr(z)
  for (int i=0; i<100; i++) foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(z)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

#pragma omp target parallel for is_device_ptr(aa)
  for (int i=0; i<100; i++) foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(aa)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

#pragma omp target parallel for is_device_ptr(raa)
  for (int i=0; i<100; i++) foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(raa)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

#pragma omp target parallel for is_device_ptr(h)
  for (int i=0; i<100; i++) foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(h)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

#pragma omp target parallel for is_device_ptr(rh)
  for (int i=0; i<100; i++) foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(rh)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();

#pragma omp target parallel for is_device_ptr(da)
  for (int i=0; i<100; i++) foo();
// CHECK-NEXT: #pragma omp target parallel for is_device_ptr(da)
// CHECK-NEXT: for (int i = 0; i < 100; i++)
// CHECK-NEXT: foo();
  return tmain<int>(argc) + *tmain<int *>(&argc);
}


#endif
