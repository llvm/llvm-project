// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
// CHECK: static T a;
#pragma omp distribute dist_schedule(static,10)
// CHECK-NEXT: #pragma omp distribute dist_schedule(static, 10)
  for (int i=0; i < 2; ++i)a=2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp distribute dist_schedule(static,a)
// CHECK-NEXT: #pragma omp distribute dist_schedule(static, a)
  for (int i=0; i < 2; ++i)a=2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams
#pragma omp distribute dist_schedule(static,2)
  for (int i = 0; i < 10; ++i)
  for (int j = 0; j < 10; ++j)foo();
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams
// CHECK-NEXT: #pragma omp distribute dist_schedule(static, 2)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute dist_schedule(static,a)
  for (int i = 0; i < 10; ++i)
  for (int j = 0; j < 10; ++j)foo();
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams
// CHECK-NEXT: #pragma omp distribute dist_schedule(static, a)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: foo();
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp distribute
// CHECK: #pragma omp distribute
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();  
  return T();
}

int main (int argc, char **argv) {
// CHECK: int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp distribute dist_schedule(static,10)
// CHECK-NEXT: #pragma omp distribute dist_schedule(static, 10)
  for (int i=0; i < 2; ++i)a=2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp distribute dist_schedule(static,b)
// CHECK-NEXT: #pragma omp distribute dist_schedule(static, b)
  for (int i=0; i < 2; ++i)a=2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp target
#pragma omp teams
#pragma omp distribute dist_schedule(static,2)
  for (int i = 0; i < 10; ++i)
  for (int j = 0; j < 10; ++j)foo();
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams
// CHECK-NEXT: #pragma omp distribute dist_schedule(static, 2)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute dist_schedule(static,a)
  for (int i = 0; i < 10; ++i)
  for (int j = 0; j < 10; ++j)foo();
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: #pragma omp teams
// CHECK-NEXT: #pragma omp distribute dist_schedule(static, a)
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: for (int j = 0; j < 10; ++j)
// CHECK-NEXT: foo();
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
#pragma omp distribute
// CHECK: #pragma omp distribute
  for (int i = 0; i < 10; ++i)foo();
// CHECK-NEXT: for (int i = 0; i < 10; ++i)
// CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
