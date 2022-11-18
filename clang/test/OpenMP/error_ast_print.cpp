// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -fopenmp-version=51 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-version=51 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -fopenmp-version=51 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -fopenmp-version=51 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}
// CHECK: template <typename T, int N> int tmain(T argc, char **argv)
// CHECK: static int a;
// CHECK-NEXT: #pragma omp error at(execution) severity(fatal)
// CHECK-NEXT: a = argv[0][0];
// CHECK-NEXT: ++a;
// CHECK-NEXT: #pragma omp error at(execution) severity(warning)
// CHECK-NEXT: {
// CHECK-NEXT: int b = 10;
// CHECK-NEXT: T c = 100;
// CHECK-NEXT: a = b + c;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp error at(execution) severity(fatal)
// CHECK-NEXT: foo();
// CHECK-NEXT: return N;

template <typename T, int N>
int tmain(T argc, char **argv) {
  T b = argc, c, d, e, f, g;
  static int a;
#pragma omp error at(execution) severity(fatal)
  a = argv[0][0];
  ++a;
#pragma omp error at(execution) severity(warning)
  {
    int b = 10;
    T c = 100;
    a = b + c;
  }
#pragma omp  error at(execution) severity(fatal)
  foo();
return N;
}

// CHECK: int main(int argc, char **argv)
// CHECK-NEXT: int b = argc, c, d, e, f, g;
// CHECK-NEXT: static int a;
// CHECK-NEXT: #pragma omp error at(execution) severity(fatal)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: #pragma omp error at(execution) severity(warning)
// CHECK-NEXT: foo();
int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
#pragma omp error at(execution) severity(fatal)
   a=2;
#pragma omp error at(execution) severity(warning)
  foo();
}
#endif
