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

// CHECK: template <typename T, int N> int tmain(T argc, char **argv)
// CHECK: static int a;
// CHECK-NEXT: #pragma omp critical
// CHECK-NEXT: a = 2;
// CHECK-NEXT: ++a;
// CHECK-NEXT: #pragma omp critical (the_name) hint(N)
// CHECK-NEXT: foo();
// CHECK-NEXT: return N;
// CHECK: template<> int tmain<int, 4>(int argc, char **argv)
template <typename T, int N>
int tmain (T argc, char **argv) {
  T b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp critical
  a=2;
// CHECK-NEXT: #pragma omp critical
// CHECK-NEXT: a = 2;
// CHECK-NEXT: ++a;
  ++a;
#pragma omp critical  (the_name) hint(N)
  foo();
// CHECK-NEXT: #pragma omp critical (the_name) hint(4)
// CHECK-NEXT: foo();
// CHECK-NEXT: return 4;
  return N;
}

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp critical
  a=2;
// CHECK-NEXT: #pragma omp critical
// CHECK-NEXT: a = 2;
// CHECK-NEXT: ++a;
  ++a;
#pragma omp critical  (the_name1) hint(23)
  foo();
// CHECK-NEXT: #pragma omp critical (the_name1) hint(23)
// CHECK-NEXT: foo();
// CHECK-NEXT: return tmain<int, 4>(a, argv);
  return tmain<int, 4>(a, argv);
}

#endif
