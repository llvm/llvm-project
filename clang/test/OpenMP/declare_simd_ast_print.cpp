// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -std=c++11 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp declare simd linear(d: 8)
#pragma omp declare simd inbranch simdlen(32)
#pragma omp declare simd notinbranch
void add_1(float *d) __attribute__((cold));

// CHECK: #pragma omp declare simd notinbranch
// CHECK-NEXT: #pragma omp declare simd inbranch simdlen(32)
// CHECK-NEXT: #pragma omp declare simd linear(val(d): 8)
// CHECK-NEXT: void add_1(float *d) __attribute__((cold));
//

#pragma omp declare simd aligned(hp, hp2:V)
#pragma omp declare simd aligned(hp, hp2:V)
template <class C, int V> void h(C *hp, C *hp2, C *hq, C *lin) {
}
// CHECK-NEXT: #pragma omp declare simd aligned(hp: V) aligned(hp2: V)
// CHECK-NEXT: #pragma omp declare simd aligned(hp: V) aligned(hp2: V)
// CHECK-NEXT: template <class C, int V> void h(C *hp, C *hp2, C *hq, C *lin) {
// CHECK-NEXT: }

#pragma omp declare simd aligned(hp, hp2)
template <class C> void h(C *hp, C *hp2, C *hq, C *lin) {
}

// CHECK: #pragma omp declare simd aligned(hp) aligned(hp2)
// CHECK-NEXT: template <class C> void h(C *hp, C *hp2, C *hq, C *lin) {
// CHECK-NEXT: }

// CHECK: #pragma omp declare simd aligned(hp) aligned(hp2)
// CHECK-NEXT: template<> void h<float>(float *hp, float *hp2, float *hq, float *lin) {
// CHECK-NEXT: }

// CHECK-NEXT: template<> void h<int>(int *hp, int *hp2, int *hq, int *lin) {
// CHECK-NEXT: h((float *)hp, (float *)hp2, (float *)hq, (float *)lin);
// CHECK-NEXT: }

// Explicit specialization with <C=int>.
// Pragmas need to be same, otherwise standard says that's undefined behavior.
#pragma omp declare simd aligned(hp, hp2)
template <>
void h(int *hp, int *hp2, int *hq, int *lin)
{
  // Implicit specialization with <C=float>.
  // This is special case where the directive is stored by Sema and is
  // generated together with the (pending) function instatiation.
  h((float*) hp, (float*) hp2, (float*) hq, (float*) lin);
}

class VV {
  // CHECK: #pragma omp declare simd uniform(this, a) linear(val(b): a)
  // CHECK-NEXT: int add(int a, int b) __attribute__((cold))    {
  // CHECK-NEXT: return a + b;
  // CHECK-NEXT: }
  #pragma omp declare simd uniform(this, a) linear(val(b): a)
  int add(int a, int b) __attribute__((cold)) { return a + b; }

  // CHECK: #pragma omp declare simd aligned(b: 4) aligned(a) linear(ref(b): 4) linear(val(this)) linear(val(a))
  // CHECK-NEXT: float taddpf(float *a, float *&b)     {
  // CHECK-NEXT: return *a + *b;
  // CHECK-NEXT: }
  #pragma omp declare simd aligned (b: 4) aligned(a) linear(ref(b): 4) linear(this, a)
  float taddpf(float *a, float *&b) { return *a + *b; }

// CHECK: #pragma omp declare simd aligned(b: 8)
// CHECK-NEXT: #pragma omp declare simd linear(uval(c): 8)
// CHECK-NEXT: int tadd(int (&b)[], int &c) {
// CHECK-NEXT: return this->x[b[0]] + b[0];
// CHECK-NEXT: }
  #pragma omp declare simd linear(uval(c): 8)
  #pragma omp declare simd aligned(b : 8)
  int tadd(int (&b)[], int &c) { return x[b[0]] + b[0]; }

private:
  int x[10];
};

// CHECK: template <int X, typename T> class TVV {
// CHECK: #pragma omp declare simd simdlen(X)
// CHECK-NEXT: int tadd(int a, int b) {
// CHECK: #pragma omp declare simd aligned(a: X * 2) aligned(b) linear(ref(b): X)
// CHECK-NEXT: float taddpf(float *a, T *&b) {
// CHECK-NEXT: return *a + *b;
// CHECK-NEXT: }
// CHECK: #pragma omp declare simd uniform(this, b)
// CHECK-NEXT: #pragma omp declare simd{{$}}
// CHECK-NEXT: int tadd(int b) {
// CHECK-NEXT: return this->x[b] + b;
// CHECK-NEXT: }
// CHECK: }
template <int X, typename T>
class TVV {
public:
// CHECK: template<> class TVV<16, float> {
  #pragma omp declare simd simdlen(X)
  int tadd(int a, int b) { return a + b; }

// CHECK: #pragma omp declare simd simdlen(16)
// CHECK-NEXT: int tadd(int a, int b);

  #pragma omp declare simd aligned(a : X * 2) aligned(b) linear(ref(b): X)
  float taddpf(float *a, T *&b) { return *a + *b; }

// CHECK: #pragma omp declare simd aligned(a: 16 * 2) aligned(b) linear(ref(b): 16)
// CHECK-NEXT: float taddpf(float *a, float *&b) {
// CHECK-NEXT: return *a + *b;
// CHECK-NEXT: }

  #pragma omp declare simd
  #pragma omp declare simd uniform(this, b)
  int tadd(int b) { return x[b] + b; }

// CHECK: #pragma omp declare simd uniform(this, b)
// CHECK-NEXT: #pragma omp declare simd
// CHECK-NEXT: int tadd(int b) {
// CHECK-NEXT: return this->x[b] + b;
// CHECK-NEXT: }

private:
  int x[X];
};
// CHECK: };

// CHECK: #pragma omp declare simd simdlen(N) aligned(b: N * 2) linear(uval(c): N)
// CHECK: template <int N> void foo(int (&b)[N], float *&c)
// CHECK: #pragma omp declare simd simdlen(64) aligned(b: 64 * 2) linear(uval(c): 64)
// CHECK: template<> void foo<64>(int (&b)[64], float *&c)
#pragma omp declare simd simdlen(N) aligned(b : N * 2) linear(uval(c): N)
template <int N>
void foo(int (&b)[N], float *&c);

// CHECK: TVV<16, float> t16;
TVV<16, float> t16;

void f() {
  float a = 1.0f, b = 2.0f;
  float *p = &b;
  float r = t16.taddpf(&a, p);
  int res = t16.tadd(b);
  int c[64];
  foo(c, p);
}

#endif
