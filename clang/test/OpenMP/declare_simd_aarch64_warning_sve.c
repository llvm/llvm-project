// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve  -fopenmp  %s -S  -o %t -verify
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve  -fopenmp-simd  %s -S  -o %t -verify

#pragma omp declare simd simdlen(66)
double foo(float x);
//expected-warning@-2{{The clause simdlen must fit the 64-bit lanes in the architectural constraints for SVE (min is 128-bit, max is 2048-bit, by steps of 128-bit)}}

void foo_loop(double *x, float *y, int N) {
  for (int i = 0; i < N; ++i) {
    x[i] = foo(y[i]);
  }
}
