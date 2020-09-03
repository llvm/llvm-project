// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp for reduction(inscan, +: a)
  for (int i = 0; i < 10; ++i) {
#pragma omp scan inclusive(a)
  }
  return a + argc;
}
// CHECK:      static T a;
// CHECK-NEXT: #pragma omp for reduction(inscan, +: a)
// CHECK-NEXT: for (int i = 0; i < 10; ++i) {
// CHECK-NEXT: #pragma omp scan inclusive(a){{$}}
// CHECK:      static int a;
// CHECK-NEXT: #pragma omp for reduction(inscan, +: a)
// CHECK-NEXT: for (int i = 0; i < 10; ++i) {
// CHECK-NEXT: #pragma omp scan inclusive(a)
// CHECK:      static char a;
// CHECK-NEXT: #pragma omp for reduction(inscan, +: a)
// CHECK-NEXT: for (int i = 0; i < 10; ++i) {
// CHECK-NEXT: #pragma omp scan inclusive(a)

int main(int argc, char **argv) {
  static int a;
// CHECK: static int a;
#pragma omp parallel
#pragma omp for simd reduction(inscan, ^: a, argc)
  for (int i = 0; i < 10; ++i) {
#pragma omp scan exclusive(a, argc)
  }
// CHECK-NEXT: #pragma omp parallel
// CHECK-NEXT: #pragma omp for simd reduction(inscan, ^: a,argc)
// CHECK-NEXT: for (int i = 0; i < 10; ++i) {
// CHECK-NEXT: #pragma omp scan exclusive(a,argc){{$}}
  return tmain(argc) + tmain(argv[0][0]) + a;
}

#endif
