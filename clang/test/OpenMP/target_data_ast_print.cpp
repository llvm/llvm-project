// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i, j, b, c, d, e, x[20];

#pragma omp target data map(to: c)
  i = argc;

#pragma omp target data map(to: c) if (target data: j > 0)
  foo();

#pragma omp target data map(to: c) if (b)
  foo();

#pragma omp target data map(c)
  foo();

#pragma omp target data map(c) if(b>e)
  foo();

#pragma omp target data map(x[0:10], c)
  foo();

#pragma omp target data map(to: c) map(from: d)
  foo();

#pragma omp target data map(always,alloc: e)
  foo();

// nesting a target region
#pragma omp target data map(e)
{
  #pragma omp target map(always, alloc: e)
    foo();
}

  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T i, j, b, c, d, e, x[20];
// CHECK-NEXT: #pragma omp target data map(to: c)
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target data map(to: c) if(target data: j > 0)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(to: c) if(b)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: c)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: c) if(b > e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: x[0:10],c)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(to: c) map(from: d)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(always,alloc: e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: e)
// CHECK-NEXT: {
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
// CHECK-NEXT: foo();
// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int i, j, b, c, d, e, x[20];
// CHECK-NEXT: #pragma omp target data map(to: c)
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target data map(to: c) if(target data: j > 0)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(to: c) if(b)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: c)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: c) if(b > e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: x[0:10],c)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(to: c) map(from: d)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(always,alloc: e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: e)
// CHECK-NEXT: {
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
// CHECK-NEXT: foo();
// CHECK: template<> char tmain<char, 1>(char argc, char *argv) {
// CHECK-NEXT: char i, j, b, c, d, e, x[20];
// CHECK-NEXT: #pragma omp target data map(to: c)
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target data map(to: c) if(target data: j > 0)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(to: c) if(b)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: c)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: c) if(b > e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: x[0:10],c)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(to: c) map(from: d)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(always,alloc: e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: e)
// CHECK-NEXT: {
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
// CHECK-NEXT: foo();

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g, x[20];
  static int a;
// CHECK: static int a;

#pragma omp target data map(to: c)
// CHECK:      #pragma omp target data map(to: c)
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp target data map(to: c) if (target data: b)
// CHECK: #pragma omp target data map(to: c) if(target data: b)
  foo();
// CHECK-NEXT: foo();

#pragma omp target data map(to: c) if (b > g)
// CHECK: #pragma omp target data map(to: c) if(b > g)
  foo();
// CHECK-NEXT: foo();

#pragma omp target data map(c)
// CHECK-NEXT: #pragma omp target data map(tofrom: c)
  foo();
// CHECK-NEXT: foo();

#pragma omp target data map(c) if(b>g)
// CHECK-NEXT: #pragma omp target data map(tofrom: c) if(b > g)
  foo();
// CHECK-NEXT: foo();

#pragma omp target data map(x[0:10], c)
// CHECK-NEXT: #pragma omp target data map(tofrom: x[0:10],c)
  foo();
// CHECK-NEXT: foo();

#pragma omp target data map(to: c) map(from: d)
// CHECK-NEXT: #pragma omp target data map(to: c) map(from: d)
  foo();
// CHECK-NEXT: foo();

#pragma omp target data map(always,alloc: e)
// CHECK-NEXT: #pragma omp target data map(always,alloc: e)
  foo();
// CHECK-NEXT: foo();

// nesting a target region
#pragma omp target data map(e)
// CHECK-NEXT: #pragma omp target data map(tofrom: e)
{
// CHECK-NEXT: {
  #pragma omp target map(always, alloc: e)
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
    foo();
// CHECK-NEXT: foo();
}
  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
