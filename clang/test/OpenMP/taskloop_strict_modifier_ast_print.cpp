// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
// CHECK: static T a;
#pragma omp taskgroup
#pragma omp taskloop grainsize(strict: N)
  // CHECK-NEXT: #pragma omp taskgroup
  // CHECK-NEXT: #pragma omp taskloop grainsize(strict: N)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp taskloop grainsize(strict: N)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j) {
#pragma omp cancel taskgroup
#pragma omp cancellation point taskgroup
            foo();
          }
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp taskloop grainsize(strict: N)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j) {
  // CHECK-NEXT: #pragma omp cancel taskgroup
  // CHECK-NEXT: #pragma omp cancellation point taskgroup
  // CHECK-NEXT: foo();
  return T();
}

// CHECK-LABEL: int main(int argc, char **argv) {
int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  int tid = 0;
  static int a;
// CHECK: static int a;
#pragma omp taskloop grainsize(strict: argc)
  for (int i = 0; i < 10; ++i)
    foo();
  // CHECK-NEXT: #pragma omp taskloop grainsize(strict: argc)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: foo();

#pragma omp parallel
#pragma omp masked taskloop grainsize(strict: argc)
  for (int i = 0; i < 10; ++i)
    foo();
  // CHECK: #pragma omp parallel
  // CHECK-NEXT: #pragma omp masked taskloop grainsize(strict: argc)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: foo();

#pragma omp parallel
#pragma omp masked taskloop simd grainsize(strict: argc)
   for (int i = 0; i < 10; ++i)
     foo();
   // CHECK: #pragma omp parallel
   // CHECK-NEXT: #pragma omp masked taskloop simd grainsize(strict: argc)
   // CHECK-NEXT: for (int i = 0; i < 10; ++i)
   // CHECK-NEXT: foo();

#pragma omp parallel masked taskloop grainsize(strict: argc)
  for (int i = 0; i < 10; ++i)
    foo();
  // CHECK-NEXT: #pragma omp parallel masked taskloop grainsize(strict: argc)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: foo();

#pragma omp parallel masked taskloop simd grainsize(strict: argc)
   for (int i = 0; i < 10; ++i)
     foo();
   // CHECK-NEXT: #pragma omp parallel masked taskloop simd grainsize(strict: argc)
   // CHECK-NEXT: for (int i = 0; i < 10; ++i)
   // CHECK-NEXT: foo();

#pragma omp parallel
#pragma omp master taskloop grainsize(strict: argc)
  for (int i = 0; i < 10; ++i)
    foo();
  // CHECK: #pragma omp parallel
  // CHECK-NEXT: #pragma omp master taskloop grainsize(strict: argc)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: foo();

#pragma omp parallel
#pragma omp master taskloop simd grainsize(strict: argc)
   for (int i = 0; i < 10; ++i)
     foo();
   // CHECK: #pragma omp parallel
   // CHECK-NEXT: #pragma omp master taskloop simd grainsize(strict: argc)
   // CHECK-NEXT: for (int i = 0; i < 10; ++i)
   // CHECK-NEXT: foo();

#pragma omp parallel master taskloop grainsize(strict: argc)
  for (int i = 0; i < 10; ++i)
    foo();
  // CHECK-NEXT: #pragma omp parallel master taskloop grainsize(strict: argc)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: foo();

#pragma omp parallel master taskloop simd grainsize(strict: argc)
   for (int i = 0; i < 10; ++i)
     foo();
   // CHECK-NEXT: #pragma omp parallel master taskloop simd grainsize(strict: argc)
   // CHECK-NEXT: for (int i = 0; i < 10; ++i)
   // CHECK-NEXT: foo();

  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
