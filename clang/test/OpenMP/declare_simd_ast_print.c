// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp declare simd aligned(b : 64)
#pragma omp declare simd simdlen(32) aligned(d, s1)
#pragma omp declare simd inbranch, uniform(d)
#pragma omp declare simd notinbranch simdlen(2), uniform(s1, s2)
void add_1(float *d, float *s1, float *s2, double b[]) __attribute__((cold));

// CHECK: #pragma omp declare simd notinbranch simdlen(2) uniform(s1, s2)
// CHECK-NEXT: #pragma omp declare simd inbranch uniform(d)
// CHECK-NEXT: #pragma omp declare simd simdlen(32) aligned(d) aligned(s1)
// CHECK-NEXT: #pragma omp declare simd aligned(b: 64)
// CHECK-NEXT: void add_1(float *d, float *s1, float *s2, double b[]) __attribute__((cold))

#endif
