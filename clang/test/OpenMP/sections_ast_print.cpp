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
#pragma omp parallel
#pragma omp sections private(argc, b), firstprivate(c, d), lastprivate(d, f) reduction(task, - : g) nowait allocate(d)
  {
    foo();
  }
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp sections private(argc,b) firstprivate(c,d) lastprivate(d,f) reduction(task, -: g) nowait allocate(d){{$}}
  // CHECK-NEXT: {
  // CHECK-NEXT: foo();
  // CHECK-NEXT: }
  return T();
}

int main(int argc, char **argv) {
// CHECK: int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp parallel
#pragma omp sections allocate(c) private(argc, b), firstprivate(argv, c), lastprivate(d, f) reduction(+ : g) nowait
  {
#pragma omp section
    foo();
#pragma omp section
    foo();
  }
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp sections allocate(c) private(argc,b) firstprivate(argv,c) lastprivate(d,f) reduction(+: g) nowait
  // CHECK-NEXT: {
  // CHECK-NEXT: #pragma omp section{{$}}
  // CHECK-NEXT: foo();
  // CHECK-NEXT: #pragma omp section
  // CHECK-NEXT: foo();
  // CHECK-NEXT: }
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
