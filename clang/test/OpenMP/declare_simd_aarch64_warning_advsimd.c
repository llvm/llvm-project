// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -fopenmp  %s -S  -o %t -verify
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -fopenmp-simd  %s -S  -o %t -verify

#pragma omp declare simd simdlen(6)
double foo(float x);
// expected-warning@-2{{The value specified in simdlen must be a power of 2 when targeting Advanced SIMD.}}
#pragma omp declare simd simdlen(1)
float bar(double x);
// expected-warning@-2{{The clause simdlen(1) has no effect when targeting aarch64.}}

void foo_loop(double *x, float *y, int N) {
  for (int i = 0; i < N; ++i) {
    x[i] = foo(y[i]);
    y[i] = bar(x[i]);
  }
}
