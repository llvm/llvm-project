// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T, int N>
T tmain (T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
  #pragma omp for ordered
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered
  {
    a=2;
  }
  #pragma omp for ordered
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered threads
  {
    a=2;
  }
  #pragma omp simd
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered simd
  {
    a=2;
  }
  #pragma omp for simd
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered simd
  {
    a=2;
  }
  #pragma omp parallel for simd
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered simd
  {
    a=2;
  }
  #pragma omp parallel for ordered(1)
  for (int i =0 ; i < argc; ++i) {
  #pragma omp ordered depend(source)
  #pragma omp ordered depend(sink:i+N)
    a = 2;
  }
  return (0);
}

// CHECK: static int a;
// CHECK-NEXT: #pragma omp for ordered
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp for ordered
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered threads
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp for simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel for simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel for ordered(1)
// CHECK-NEXT: for (int i = 0; i < argc; ++i) {
// CHECK-NEXT: #pragma omp ordered depend(source)
// CHECK-NEXT: #pragma omp ordered depend(sink : i + 3)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK: static T a;
// CHECK-NEXT: #pragma omp for ordered
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp for ordered
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered threads
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp for simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel for simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel for ordered(1)
// CHECK-NEXT: for (int i = 0; i < argc; ++i) {
// CHECK-NEXT: #pragma omp ordered depend(source)
// CHECK-NEXT: #pragma omp ordered depend(sink : i + N)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }

// CHECK-LABEL: int main(
int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
  #pragma omp for ordered
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered
  {
    a=2;
  }
  #pragma omp for ordered
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered threads
  {
    a=2;
  }
  #pragma omp simd
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered simd
  {
    a=2;
  }
  #pragma omp for simd
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered simd
  {
    a=2;
  }
  #pragma omp parallel for simd
  for (int i =0 ; i < argc; ++i)
  #pragma omp ordered simd
  {
    a=2;
  }
  #pragma omp parallel for ordered(1)
  for (int i =0 ; i < argc; ++i) {
  #pragma omp ordered depend(source)
  #pragma omp ordered depend(sink: i - 5)
    a = 2;
  }
// CHECK-NEXT: #pragma omp for ordered
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp for ordered
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered threads
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp for simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel for simd
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered simd
// CHECK-NEXT: {
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp parallel for ordered(1)
// CHECK-NEXT: for (int i = 0; i < argc; ++i) {
// CHECK-NEXT: #pragma omp ordered depend(source)
// CHECK-NEXT: #pragma omp ordered depend(sink : i - 5)
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
  return tmain<int, 3>(argc);
}

#endif
