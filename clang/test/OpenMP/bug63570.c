// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-apple-darwin10 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c -triple x86_64-apple-darwin10 %s
// expected-no-diagnostics

void f(float *a, float *b) {
#pragma omp unroll
  for (int i = 0; i < 128; i++) {
    a[i] = b[i];
  }
}
