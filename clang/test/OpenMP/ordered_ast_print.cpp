// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck -check-prefixes CHECK,OMP51 %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck -check-prefixes CHECK,OMP51 %s

// RUN: %clang_cc1 -verify -fopenmp-version=52 -fopenmp -ast-print %s | FileCheck -check-prefixes CHECK,OMP52 %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck -check-prefixes CHECK,OMP52 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck -check-prefixes CHECK,OMP51 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck -check-prefixes CHECK,OMP51 %s
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
#if _OPENMP >= 202111
  #pragma omp ordered doacross(source:)
  #pragma omp ordered doacross(sink:i+N)
  #pragma omp ordered doacross(sink: omp_cur_iteration - 1)
  #pragma omp ordered doacross(source: omp_cur_iteration)
#else
  #pragma omp ordered depend(source)
  #pragma omp ordered depend(sink:i+N)
#endif
    a = 2;
  }
  return (0);
}

// CHECK: static T a;
// CHECK-NEXT: #pragma omp for ordered
// CHECK-NEXT: for (int i = 0; i < argc; ++i)
// CHECK-NEXT: #pragma omp ordered{{$}}
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
#if _OPENMP >= 202111
// OMP52: #pragma omp ordered doacross(source:)
// OMP52-NEXT: #pragma omp ordered doacross(sink: i + N)
// OMP52-NEXT: #pragma omp ordered doacross(sink: omp_cur_iteration - 1)
// OMP52-NEXT: #pragma omp ordered doacross(source: omp_cur_iteration)
#else
// OMP51: #pragma omp ordered depend(source)
// OMP51-NEXT: #pragma omp ordered depend(sink : i + N)
#endif
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
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
#if _OPENMP >= 202111
// OMP52: #pragma omp ordered doacross(source:)
// OMP52-NEXT: #pragma omp ordered doacross(sink: i + 3)
// OMP52-NEXT: #pragma omp ordered doacross(sink: omp_cur_iteration - 1)
// OMP52-NEXT: #pragma omp ordered doacross(source: omp_cur_iteration)
#else
// OMP51: #pragma omp ordered depend(source)
// OMP51-NEXT: #pragma omp ordered depend(sink : i + 3)
#endif
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
#if _OPENMP >= 202111
  #pragma omp ordered doacross(source:)
  #pragma omp ordered doacross(sink: i - 5)
  #pragma omp ordered doacross(sink: omp_cur_iteration - 1)
  #pragma omp ordered doacross(source: omp_cur_iteration)
#else
  #pragma omp ordered depend(source)
  #pragma omp ordered depend(sink: i - 5)
#endif
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
#if _OPENMP >= 202111
// OMP52: #pragma omp ordered doacross(source:)
// OMP52-NEXT: #pragma omp ordered doacross(sink: i - 5)
// OMP52-NEXT: #pragma omp ordered doacross(sink: omp_cur_iteration - 1)
// OMP52-NEXT: #pragma omp ordered doacross(source: omp_cur_iteration)
#else
// OMP51: #pragma omp ordered depend(source)
// OMP51-NEXT: #pragma omp ordered depend(sink : i - 5)
#endif
// CHECK-NEXT: a = 2;
// CHECK-NEXT: }
  return tmain<int, 3>(argc);
}

#endif
