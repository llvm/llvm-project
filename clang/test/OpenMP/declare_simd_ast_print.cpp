// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp declare simd
#pragma omp declare simd inbranch
#pragma omp declare simd notinbranch
void add_1(float *d) __attribute__((cold));

// CHECK: #pragma omp declare simd notinbranch
// CHECK-NEXT: #pragma omp declare simd inbranch
// CHECK-NEXT: #pragma omp declare simd
// CHECK-NEXT: void add_1(float *d) __attribute__((cold));
//

#pragma omp declare simd
template <class C> void h(C *hp, C *hp2, C *hq, C *lin) {
}

// CHECK: #pragma omp declare simd
// CHECK-NEXT: template <class C = int> void h(int *hp, int *hp2, int *hq, int *lin) {
// CHECK-NEXT: h((float *)hp, (float *)hp2, (float *)hq, (float *)lin);
// CHECK-NEXT: }

// CHECK: #pragma omp declare simd
// CHECK-NEXT: template <class C = float> void h(float *hp, float *hp2, float *hq, float *lin) {
// CHECK-NEXT: }

// CHECK: #pragma omp declare simd
// CHECK: template <class C> void h(C *hp, C *hp2, C *hq, C *lin) {
// CHECK-NEXT: }
//

// Explicit specialization with <C=int>.
// Pragmas need to be same, otherwise standard says that's undefined behavior.
#pragma omp declare simd
template <>
void h(int *hp, int *hp2, int *hq, int *lin)
{
  // Implicit specialization with <C=float>.
  // This is special case where the directive is stored by Sema and is
  // generated together with the (pending) function instatiation.
  h((float*) hp, (float*) hp2, (float*) hq, (float*) lin);
}

class VV {
  // CHECK: #pragma omp declare simd
  // CHECK-NEXT: int add(int a, int b) __attribute__((cold))    {
  // CHECK-NEXT: return a + b;
  // CHECK-NEXT: }
  #pragma omp declare simd
  int add(int a, int b) __attribute__((cold)) { return a + b; }

  // CHECK: #pragma omp declare simd
  // CHECK-NEXT: float taddpf(float *a, float *b)     {
  // CHECK-NEXT: return *a + *b;
  // CHECK-NEXT: }
  #pragma omp declare simd
  float taddpf(float *a, float *b) { return *a + *b; }

// CHECK: #pragma omp declare simd
// CHECK-NEXT: #pragma omp declare simd
// CHECK-NEXT: int tadd(int b) {
// CHECK-NEXT: return this->x[b] + b;
// CHECK-NEXT: }
  #pragma omp declare simd
  #pragma omp declare simd
  int tadd(int b) { return x[b] + b; }

private:
  int x[10];
};

// CHECK: template <int X = 16> class TVV {
// CHECK: #pragma omp declare simd
// CHECK-NEXT: int tadd(int a, int b);
// CHECK: #pragma omp declare simd
// CHECK-NEXT: float taddpf(float *a, float *b) {
// CHECK-NEXT: return *a + *b;
// CHECK-NEXT: }
// CHECK: #pragma omp declare simd
// CHECK-NEXT: #pragma omp declare simd
// CHECK-NEXT: int tadd(int b) {
// CHECK-NEXT: return this->x[b] + b;
// CHECK-NEXT: }
// CHECK: }
template <int X>
class TVV {
public:
// CHECK: template <int X> class TVV {
  #pragma omp declare simd
  int tadd(int a, int b) { return a + b; }

// CHECK: #pragma omp declare simd
// CHECK-NEXT: int tadd(int a, int b) {
// CHECK-NEXT: return a + b;
// CHECK-NEXT: }

  #pragma omp declare simd
  float taddpf(float *a, float *b) { return *a + *b; }

// CHECK: #pragma omp declare simd
// CHECK-NEXT: float taddpf(float *a, float *b) {
// CHECK-NEXT: return *a + *b;
// CHECK-NEXT: }

  #pragma omp declare simd
  #pragma omp declare simd
  int tadd(int b) { return x[b] + b; }

// CHECK: #pragma omp declare simd
// CHECK-NEXT: #pragma omp declare simd
// CHECK-NEXT: int tadd(int b) {
// CHECK-NEXT: return this->x[b] + b;
// CHECK-NEXT: }

private:
  int x[X];
};
// CHECK: };

// CHECK: TVV<16> t16;
TVV<16> t16;

void f() {
  float a = 1.0f, b = 2.0f;
  float r = t16.taddpf(&a, &b);
  int res = t16.tadd(b);
}

#endif
