// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <typename T, int C>
T tmain(T argc, T *argv) {
#pragma omp target
  foo();
#pragma omp target if (target:argc > 0)
  foo();
#pragma omp target if (C)
  foo();
  return 0;
}

// CHECK: template <typename T = int, int C = 5> int tmain(int argc, int *argv) {
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target if(target: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target if(5)
// CHECK-NEXT: foo()
// CHECK: template <typename T = char, int C = 1> char tmain(char argc, char *argv) {
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target if(target: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target if(1)
// CHECK-NEXT: foo()
// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: #pragma omp target
// CHECK-NEXT: foo();
// CHECK-NEXT: #pragma omp target if(target: argc > 0)
// CHECK-NEXT: foo()
// CHECK-NEXT: #pragma omp target if(C)
// CHECK-NEXT: foo()

// CHECK-LABEL: int main(int argc, char **argv) {
int main (int argc, char **argv) {
#pragma omp target
// CHECK-NEXT: #pragma omp target
  foo();
// CHECK-NEXT: foo();
#pragma omp target if (argc > 0)
// CHECK-NEXT: #pragma omp target if(argc > 0)
  foo();
// CHECK-NEXT: foo();
  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
