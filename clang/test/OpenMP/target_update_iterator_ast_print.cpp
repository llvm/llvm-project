// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ast-print %s | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void test() {
  int a[10];
  #pragma omp target update to(iterator(int it = 0:10): a[it]) 
  // CHECK:   int a[10];
  // CHECK: #pragma omp target update to(iterator(int it = 0:10): a[it])
  #pragma omp target update from(iterator(int it = 0:10): a[it]) 
  // CHECK: #pragma omp target update from(iterator(int it = 0:10): a[it])
}

#endif
