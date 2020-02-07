// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <class T>
T foo(T argc) {
  T b = T();
  T a = T();
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic capture
  a = b++;
#pragma omp atomic capture
  {
    a = b;
    b++;
  }
#pragma omp atomic seq_cst
  a++;
#pragma omp atomic read seq_cst
  a = argc;
#pragma omp atomic seq_cst write
  a = argc + argc;
#pragma omp atomic update seq_cst
  a = a + argc;
#pragma omp atomic seq_cst capture
  a = b++;
#pragma omp atomic capture seq_cst
  {
    a = b;
    b++;
  }
#pragma omp atomic acq_rel
  a++;
#pragma omp atomic read acq_rel
  a = argc;
#pragma omp atomic acq_rel write
  a = argc + argc;
#pragma omp atomic update acq_rel
  a = a + argc;
#pragma omp atomic acq_rel capture
  a = b++;
#pragma omp atomic capture acq_rel
  {
    a = b;
    b++;
  }
  return T();
}

// CHECK: T a = T();
// CHECK-NEXT: #pragma omp atomic{{$}}
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic seq_cst
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read seq_cst
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic seq_cst write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update seq_cst
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic seq_cst capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture seq_cst
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic acq_rel
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read acq_rel
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic acq_rel write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update acq_rel
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acq_rel capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acq_rel
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK: int a = int();
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic seq_cst
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read seq_cst
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic seq_cst write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update seq_cst
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic seq_cst capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture seq_cst
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic acq_rel
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read acq_rel
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic acq_rel write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update acq_rel
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acq_rel capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acq_rel
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }

int main(int argc, char **argv) {
  int b = 0;
  int a = 0;
// CHECK: int a = 0;
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic capture
  a = b++;
#pragma omp atomic capture
  {
    a = b;
    b++;
  }
#pragma omp atomic seq_cst
  a++;
#pragma omp atomic read seq_cst
  a = argc;
#pragma omp atomic seq_cst write
  a = argc + argc;
#pragma omp atomic update seq_cst
  a = a + argc;
#pragma omp atomic seq_cst capture
  a = b++;
#pragma omp atomic capture seq_cst
  {
    a = b;
    b++;
  }
#pragma omp atomic acq_rel
  a++;
#pragma omp atomic read acq_rel
  a = argc;
#pragma omp atomic acq_rel write
  a = argc + argc;
#pragma omp atomic update acq_rel
  a = a + argc;
#pragma omp atomic acq_rel capture
  a = b++;
#pragma omp atomic capture acq_rel
  {
    a = b;
    b++;
  }
  // CHECK-NEXT: #pragma omp atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma omp atomic seq_cst
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read seq_cst
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic seq_cst write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update seq_cst
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic seq_cst capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture seq_cst
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma omp atomic acq_rel
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read acq_rel
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic acq_rel write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update acq_rel
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic acq_rel capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture acq_rel
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  return foo(a);
}

#endif
