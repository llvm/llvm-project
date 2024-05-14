// RUN: %clang_cc1 -verify -fopenmp -std=c++11 \
// RUN:  -ast-print %s | FileCheck %s

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 \
// RUN:  -emit-pch -o %t %s

// RUN: %clang_cc1 -fopenmp -std=c++11 \
// RUN:  -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd \
// RUN:  -std=c++11 -ast-print %s | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 \
// RUN:  -emit-pch -o %t %s

// RUN: %clang_cc1 -fopenmp-simd -std=c++11 \
// RUN:  -include-pch %t -verify %s -ast-print | FileCheck %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

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
#pragma omp target has_device_addr(k)
    {}
#pragma omp target has_device_addr(z)
    {}
#pragma omp target has_device_addr(aa) // OK
    {}
#pragma omp target has_device_addr(raa) // OK
    {}
#pragma omp target has_device_addr(g) // OK
    {}
#pragma omp target has_device_addr(rg) // OK
    {}
#pragma omp target has_device_addr(da) // OK
    {}
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
// CHECK-NEXT: #pragma omp target has_device_addr(this->k)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(this->z)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(this->aa)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(this->raa)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(this->g)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(this->rg)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(this->da)

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
#pragma omp target has_device_addr(k)
  {}
#pragma omp target has_device_addr(z)
  {}
#pragma omp target has_device_addr(aa)
  {}
#pragma omp target has_device_addr(raa)
  {}
#pragma omp target has_device_addr(h)
  {}
#pragma omp target has_device_addr(rh)
  {}
#pragma omp target has_device_addr(da)
  {}
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
// CHECK-NEXT: #pragma omp target has_device_addr(k)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(z)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(aa)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(raa)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(h)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(rh)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(da)

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
// CHECK-NEXT: #pragma omp target has_device_addr(k)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(z)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(aa)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(raa)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(h)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(rh)
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(da)

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
#pragma omp target has_device_addr(k)
// CHECK-NEXT: #pragma omp target has_device_addr(k)
  {}
// CHECK-NEXT: {
// CHECK-NEXT: }
#pragma omp target has_device_addr(z)
// CHECK-NEXT: #pragma omp target has_device_addr(z)
  {}
// CHECK-NEXT: {
// CHECK-NEXT: }
#pragma omp target has_device_addr(aa)
// CHECK-NEXT: #pragma omp target has_device_addr(aa)
  {}
// CHECK-NEXT: {
// CHECK-NEXT: }
#pragma omp target has_device_addr(raa)
// CHECK-NEXT: #pragma omp target has_device_addr(raa)
  {}
// CHECK-NEXT: {
// CHECK-NEXT: }
#pragma omp target has_device_addr(h)
// CHECK-NEXT: #pragma omp target has_device_addr(h)
  {}
// CHECK-NEXT: {
// CHECK-NEXT: }
#pragma omp target has_device_addr(rh)
// CHECK-NEXT: #pragma omp target has_device_addr(rh)
  {}
// CHECK-NEXT: {
// CHECK-NEXT: }
#pragma omp target has_device_addr(da)
// CHECK-NEXT: #pragma omp target has_device_addr(da)
  {}
// CHECK-NEXT: {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp target has_device_addr(da[1:3])
// CHECK-NEXT: {
// CHECK-NEXT: }
#pragma omp target has_device_addr(da[1:3])
  {}
  return tmain<int>(argc) + *tmain<int *>(&argc);
}

struct SomeKernel {
  int targetDev;
  float devPtr;
  SomeKernel();
  ~SomeKernel();

  template<unsigned int nRHS>
  void apply() {
    #pragma omp target has_device_addr(devPtr) device(targetDev)
    {
    }
// CHECK:  #pragma omp target has_device_addr(this->devPtr) device(this->targetDev)
// CHECK-NEXT: {
// CHECK-NEXT: }
  }
// CHECK: template<> void apply<32U>() {
// CHECK: #pragma omp target has_device_addr(this->devPtr) device(this->targetDev)
// CHECK-NEXT: {
// CHECK-NEXT: }
};

void use_template() {
  SomeKernel aKern;
  aKern.apply<32>();
}
#endif
