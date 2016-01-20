// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i, j, b, c, d, e, x[20];

  i = argc;
#pragma omp target exit data map(from: i)

#pragma omp target exit data map(from: i) if (target exit data: j > 0)

#pragma omp target exit data map(from: i) if (b)

#pragma omp target exit data map(from: c)

#pragma omp target exit data map(from: c) if(b>e)

#pragma omp target exit data map(release: x[0:10], c)

#pragma omp target exit data map(from: c) map(release: d)

#pragma omp target exit data map(always,release: e)

  return 0;
}

// CHECK: template <typename T = int, int C = 5> int tmain(int argc, int *argv) {
// CHECK-NEXT: int i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b)
// CHECK-NEXT: #pragma omp target exit data map(from: c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) if(b > e)
// CHECK-NEXT: #pragma omp target exit data map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data map(always,release: e)
// CHECK: template <typename T = char, int C = 1> char tmain(char argc, char *argv) {
// CHECK-NEXT: char i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b)
// CHECK-NEXT: #pragma omp target exit data map(from: c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) if(b > e)
// CHECK-NEXT: #pragma omp target exit data map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data map(always,release: e)
// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T i, j, b, c, d, e, x[20];
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(target exit data: j > 0)
// CHECK-NEXT: #pragma omp target exit data map(from: i) if(b)
// CHECK-NEXT: #pragma omp target exit data map(from: c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) if(b > e)
// CHECK-NEXT: #pragma omp target exit data map(release: x[0:10],c)
// CHECK-NEXT: #pragma omp target exit data map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data map(always,release: e)

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g, x[20];
  static int a;
// CHECK: static int a;

#pragma omp target exit data map(from: a)
// CHECK:      #pragma omp target exit data map(from: a)
  a=2;
// CHECK-NEXT: a = 2;
#pragma omp target exit data map(from: a) if (target exit data: b)
// CHECK: #pragma omp target exit data map(from: a) if(target exit data: b)

#pragma omp target exit data map(from: a) if (b > g)
// CHECK: #pragma omp target exit data map(from: a) if(b > g)

#pragma omp target exit data map(from: c)
// CHECK-NEXT: #pragma omp target exit data map(from: c)

#pragma omp target exit data map(release: c) if(b>g)
// CHECK-NEXT: #pragma omp target exit data map(release: c) if(b > g)

#pragma omp target exit data map(from: x[0:10], c)
// CHECK-NEXT: #pragma omp target exit data map(from: x[0:10],c)

#pragma omp target exit data map(from: c) map(release: d)
// CHECK-NEXT: #pragma omp target exit data map(from: c) map(release: d)

#pragma omp target exit data map(always,release: e)
// CHECK-NEXT: #pragma omp target exit data map(always,release: e)

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
