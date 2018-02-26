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

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp barrier
  switch (argc) {
  case 0:
#pragma omp barrier
    break;
  default:
#pragma omp barrier
#pragma omp barrier
    break;
  }
  return a + argc;
}
// CHECK:      static T a;
// CHECK-NEXT: #pragma omp barrier
// CHECK:      static int a;
// CHECK-NEXT: #pragma omp barrier
// CHECK:      static char a;
// CHECK-NEXT: #pragma omp barrier
// CHECK-NEXT: switch (argc) {
// CHECK-NEXT: case 0:
// CHECK-NEXT: #pragma omp barrier
// CHECK-NEXT: break;
// CHECK-NEXT: default:
// CHECK-NEXT: #pragma omp barrier
// CHECK-NEXT: #pragma omp barrier
// CHECK-NEXT: break;
// CHECK-NEXT: }

int main(int argc, char **argv) {
  static int a;
// CHECK: static int a;
#pragma omp barrier
  // CHECK-NEXT: #pragma omp barrier
  switch (argc) {
  case 0:
#pragma omp barrier
#pragma omp barrier
    break;
  default:
#pragma omp barrier
    break;
  }
// CHECK-NEXT: switch (argc) {
// CHECK-NEXT: case 0:
// CHECK-NEXT: #pragma omp barrier
// CHECK-NEXT: #pragma omp barrier
// CHECK-NEXT: break;
// CHECK-NEXT: default:
// CHECK-NEXT: #pragma omp barrier
// CHECK-NEXT: break;
// CHECK-NEXT: }
  return tmain(argc) + tmain(argv[0][0]) + a;
}

#endif
