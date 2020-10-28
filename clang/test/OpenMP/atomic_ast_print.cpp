// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
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
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic acq_rel capture
  a = b++;
#pragma omp atomic capture acq_rel
  {
    a = b;
    b++;
  }
#pragma omp atomic
  a++;
#pragma omp atomic read acquire
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic acquire capture
  a = b++;
#pragma omp atomic capture acquire
  {
    a = b;
    b++;
  }
#pragma omp atomic release
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic release write
  a = argc + argc;
#pragma omp atomic update release
  a = a + argc;
#pragma omp atomic release capture
  a = b++;
#pragma omp atomic capture release
  {
    a = b;
    b++;
  }
#pragma omp atomic relaxed
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic relaxed write
  a = argc + argc;
#pragma omp atomic update relaxed
  a = a + argc;
#pragma omp atomic relaxed capture
  a = b++;
#pragma omp atomic capture relaxed
  {
    a = b;
    b++;
  }
#pragma omp atomic hint(6)
  a++;
#pragma omp atomic read hint(6)
  a = argc;
#pragma omp atomic hint(6) write
  a = argc + argc;
#pragma omp atomic update hint(6)
  a = a + argc;
#pragma omp atomic hint(6) capture
  a = b++;
#pragma omp atomic capture hint(6)
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
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acq_rel capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acq_rel
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read acquire
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acquire capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acquire
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic release
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic release write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update release
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic release capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture release
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic relaxed
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic relaxed write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update relaxed
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic relaxed capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture relaxed
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic hint(6)
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read hint(6)
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic hint(6) write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update hint(6)
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic hint(6) capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture hint(6)
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
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acq_rel capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acq_rel
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read acquire
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acquire capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acquire
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic release
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic release write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update release
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic release capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture release
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic relaxed
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic relaxed write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update relaxed
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic relaxed capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture relaxed
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp atomic hint(6)
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read hint(6)
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic hint(6) write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update hint(6)
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic hint(6) capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture hint(6)
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
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic acq_rel capture
  a = b++;
#pragma omp atomic capture acq_rel
  {
    a = b;
    b++;
  }
#pragma omp atomic
  a++;
#pragma omp atomic read acquire
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic acquire capture
  a = b++;
#pragma omp atomic capture acquire
  {
    a = b;
    b++;
  }
#pragma omp atomic release
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic release write
  a = argc + argc;
#pragma omp atomic update release
  a = a + argc;
#pragma omp atomic release capture
  a = b++;
#pragma omp atomic capture release
  {
    a = b;
    b++;
  }
#pragma omp atomic relaxed
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic relaxed write
  a = argc + argc;
#pragma omp atomic update relaxed
  a = a + argc;
#pragma omp atomic relaxed capture
  a = b++;
#pragma omp atomic capture relaxed
  {
    a = b;
    b++;
  }
#pragma omp atomic hint(6)
  a++;
#pragma omp atomic read hint(6)
  a = argc;
#pragma omp atomic hint(6) write
  a = argc + argc;
#pragma omp atomic update hint(6)
  a = a + argc;
#pragma omp atomic hint(6) capture
  a = b++;
#pragma omp atomic capture hint(6)
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
  // CHECK-NEXT: #pragma omp atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic acq_rel capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture acq_rel
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma omp atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read acquire
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic acquire capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture acquire
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma omp atomic release
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic release write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update release
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic release capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture release
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma omp atomic relaxed
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic relaxed write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update relaxed
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic relaxed capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture relaxed
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-NEXT: #pragma omp atomic hint(6)
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read hint(6)
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic hint(6) write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update hint(6)
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic hint(6) capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture hint(6)
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  return foo(a);
}

#endif
