// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp declare simd aligned(b : 64)
#pragma omp declare simd simdlen(32) aligned(d, b)
#pragma omp declare simd inbranch, uniform(d) linear(val(s1, s2) : 32)
#pragma omp declare simd notinbranch simdlen(2), uniform(s1, s2) linear(d: s1)
void add_1(float *d, int s1, float *s2, double b[]) __attribute__((cold));

// CHECK: #pragma omp declare simd notinbranch simdlen(2) uniform(s1, s2) linear(val(d): s1)
// CHECK-NEXT: #pragma omp declare simd inbranch uniform(d) linear(val(s1): 32) linear(val(s2): 32)
// CHECK-NEXT: #pragma omp declare simd simdlen(32) aligned(d) aligned(b)
// CHECK-NEXT: #pragma omp declare simd aligned(b: 64)
// CHECK-NEXT: void add_1(float *d, int s1, float *s2, double b[]) __attribute__((cold))

#endif
