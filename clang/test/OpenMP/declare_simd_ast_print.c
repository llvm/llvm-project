// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp declare simd
#pragma omp declare simd simdlen(32)
#pragma omp declare simd inbranch
#pragma omp declare simd notinbranch simdlen(2)
void add_1(float *d, float *s1, float *s2) __attribute__((cold));

// CHECK: #pragma omp declare simd notinbranch simdlen(2)
// CHECK-NEXT: #pragma omp declare simd inbranch
// CHECK-NEXT: #pragma omp declare simd simdlen(32)
// CHECK-NEXT: #pragma omp declare simd
// CHECK-NEXT: void add_1(float *d, float *s1, float *s2) __attribute__((cold))

#endif
