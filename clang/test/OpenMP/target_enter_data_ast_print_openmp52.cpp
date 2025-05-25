// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=52 -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck --check-prefix=CHECK --check-prefix=CHECK-52 %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i_def, i;

  i = argc;

#pragma omp target enter data map(i_def)

#pragma omp target enter data map(to: i)

  return 0;
}

// CHECK: template <typename T, int C> T tmain(T argc, T *argv) {
// CHECK-NEXT: T i_def, i;
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target enter data map(to: i_def){{$}}
// CHECK-NEXT: #pragma omp target enter data map(to: i){{$}}

// CHECK: template<> int tmain<int, 5>(int argc, int *argv) {
// CHECK-NEXT: int i_def, i;
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target enter data map(to: i_def){{$}}
// CHECK-NEXT: #pragma omp target enter data map(to: i)

// CHECK: template<> char tmain<char, 1>(char argc, char *argv) {
// CHECK-NEXT: char i_def, i;
// CHECK-NEXT: i = argc;
// CHECK-NEXT: #pragma omp target enter data map(to: i_def){{$}}
// CHECK-NEXT: #pragma omp target enter data map(to: i)

int main (int argc, char **argv) {
  int b_def, b;
  static int a_def, a;
// CHECK: static int a_def, a;

#pragma omp target enter data map(a_def)
// CHECK:      #pragma omp target enter data map(to: a_def)
  a_def=2;
// CHECK-NEXT: a_def = 2;

#pragma omp target enter data map(to: a)
// CHECK:      #pragma omp target enter data map(to: a)
  a=2;
// CHECK-NEXT: a = 2;

#pragma omp target enter data map(b_def)
// CHECK-NEXT:      #pragma omp target enter data map(to: b_def)

#pragma omp target enter data map(to: b)
// CHECK-NEXT:      #pragma omp target enter data map(to: b)

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
