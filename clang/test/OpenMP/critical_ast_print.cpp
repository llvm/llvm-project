// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

int main (int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp critical
  a=2;
// CHECK-NEXT: #pragma omp critical
// CHECK-NEXT: a = 2;
// CHECK-NEXT: ++a;
  ++a;
#pragma omp critical  (the_name)
  foo();
// CHECK-NEXT: #pragma omp critical (the_name)
// CHECK-NEXT: foo();
// CHECK-NEXT: return 0;
  return 0;
}

#endif
