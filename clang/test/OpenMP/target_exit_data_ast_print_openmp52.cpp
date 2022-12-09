// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i_def, i;

  i = argc;
#pragma omp target exit data map(i_def)

#pragma omp target exit data map(from: i)

  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T i_def, i;
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i_def){{$}}
// CHECK-NEXT: #pragma omp target exit data map(from: i){{$}}

// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int i_def, i;
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i_def)
// CHECK-NEXT: #pragma omp target exit data map(from: i)

// CHECK: template<> char tmain<char, 1>(char argc, char *argv) {
// CHECK-NEXT: char i_def, i;
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target exit data map(from: i_def)
// CHECK-NEXT: #pragma omp target exit data map(from: i)

int main (int argc, char **argv) {
  int b_def, b;
  static int a_def, a;
// CHECK: static int a_def, a;

#pragma omp target exit data map(a_def)
// CHECK:      #pragma omp target exit data map(from: a_def)
  a_def=2;
// CHECK-NEXT: a_def = 2;

#pragma omp target exit data map(from: a)
// CHECK:      #pragma omp target exit data map(from: a)
  a=2;
// CHECK-NEXT: a = 2;

#pragma omp target exit data map(b_def)
// CHECK-NEXT:      #pragma omp target exit data map(from: b_def)

#pragma omp target exit data map(from: b)
// CHECK-NEXT:      #pragma omp target exit data map(from: b)

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
