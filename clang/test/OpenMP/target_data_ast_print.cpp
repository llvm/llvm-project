// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -DOMP51 -DOMPX -verify -fopenmp -fopenmp-extensions -ast-print %s | FileCheck -check-prefixes=CHECK,OMP51,OMPX %s
// RUN: %clang_cc1 -DOMP51 -DOMPX -fopenmp -fopenmp-extensions -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -DOMPX -fopenmp -fopenmp-extensions -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck -check-prefixes=CHECK,OMP51,OMPX %s

// RUN: %clang_cc1 -DOMP51 -DOMPX -verify -fopenmp-simd -fopenmp-extensions -ast-print %s | FileCheck -check-prefixes=CHECK,OMP51,OMPX %s
// RUN: %clang_cc1 -DOMP51 -DOMPX -fopenmp-simd -fopenmp-extensions -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -DOMPX -fopenmp-simd -fopenmp-extensions -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck -check-prefixes=CHECK,OMP51,OMPX %s
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

#pragma omp target data map(close,alloc: e)
  foo();

#ifdef OMP51
#pragma omp target data map(present,alloc: e)
  foo();
#endif

#ifdef OMPX
#pragma omp target data map(ompx_hold,alloc: e)
  foo();
#endif

// nesting a target region
#pragma omp target data map(e)
{
  #pragma omp target map(always, alloc: e)
    foo();
  #pragma omp target map(close, alloc: e)
    foo();
#ifdef OMP51
  #pragma omp target map(present, alloc: e)
    foo();
#endif
#ifdef OMPX
  #pragma omp target map(ompx_hold, alloc: e)
    foo();
#endif
}

  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T i, j, b, c, d, e, x[20];
// CHECK-NEXT: #pragma omp target data map(to: c){{$}}
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
// CHECK-NEXT: #pragma omp target data map(close,alloc: e)
// CHECK-NEXT: foo();
// OMP51-NEXT: #pragma omp target data map(present,alloc: e)
// OMP51-NEXT: foo();
//  OMPX-NEXT: #pragma omp target data map(ompx_hold,alloc: e)
//  OMPX-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: e)
// CHECK-NEXT: {
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target map(close,alloc: e)
// CHECK-NEXT: foo();
// OMP51-NEXT: #pragma omp target map(present,alloc: e)
// OMP51-NEXT: foo();
//  OMPX-NEXT: #pragma omp target map(ompx_hold,alloc: e)
//  OMPX-NEXT: foo();
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
// CHECK-NEXT: #pragma omp target data map(close,alloc: e)
// CHECK-NEXT: foo();
// OMP51-NEXT: #pragma omp target data map(present,alloc: e)
// OMP51-NEXT: foo();
//  OMPX-NEXT: #pragma omp target data map(ompx_hold,alloc: e)
//  OMPX-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: e)
// CHECK-NEXT: {
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target map(close,alloc: e)
// CHECK-NEXT: foo();
// OMP51-NEXT: #pragma omp target map(present,alloc: e)
// OMP51-NEXT: foo();
//  OMPX-NEXT: #pragma omp target map(ompx_hold,alloc: e)
//  OMPX-NEXT: foo();
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
// CHECK-NEXT: #pragma omp target data map(close,alloc: e)
// CHECK-NEXT: foo();
// OMP51-NEXT: #pragma omp target data map(present,alloc: e)
// OMP51-NEXT: foo();
//  OMPX-NEXT: #pragma omp target data map(ompx_hold,alloc: e)
//  OMPX-NEXT: foo();
// CHECK-NEXT: #pragma omp target data map(tofrom: e)
// CHECK-NEXT: {
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target map(close,alloc: e)
// CHECK-NEXT: foo();
// OMP51-NEXT: #pragma omp target map(present,alloc: e)
// OMP51-NEXT: foo();
//  OMPX-NEXT: #pragma omp target map(ompx_hold,alloc: e)
//  OMPX-NEXT: foo();

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g, x[20];
  static int a;
// CHECK: static int a;

#pragma omp target data map(to: ([argc][3][a])argv)
  // CHECK: #pragma omp target data map(to: ([argc][3][a])argv)
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

#pragma omp target data map(close,alloc: e)
// CHECK-NEXT: #pragma omp target data map(close,alloc: e)
  foo();
// CHECK-NEXT: foo();

// OMP51-NEXT: #pragma omp target data map(present,alloc: e)
// OMP51-NEXT: foo();
#ifdef OMP51
#pragma omp target data map(present,alloc: e)
  foo();
#endif

// OMPX-NEXT: #pragma omp target data map(ompx_hold,alloc: e)
// OMPX-NEXT: foo();
#ifdef OMPX
#pragma omp target data map(ompx_hold,alloc: e)
  foo();
#endif

// nesting a target region
#pragma omp target data map(e)
// CHECK-NEXT: #pragma omp target data map(tofrom: e)
{
// CHECK-NEXT: {
  #pragma omp target map(always, alloc: e)
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
    foo();
// CHECK-NEXT: foo();
#pragma omp target map(close, alloc: e)
// CHECK-NEXT: #pragma omp target map(close,alloc: e)
  foo();
// CHECK-NEXT: foo();
#pragma omp target map(always, alloc: e)
// CHECK-NEXT: #pragma omp target map(always,alloc: e)
  foo();
}

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
