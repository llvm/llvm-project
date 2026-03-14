// REQUIRES: aarch64-registered-target
// -fopemp and -fopenmp-simd behavior are expected to be the same.

// RUN: %clang_cc1 -verify -triple aarch64-linux-gnu -target-feature +neon -fopenmp -x c++ -emit-llvm %s -o - -femit-all-decls -verify| FileCheck %s --check-prefix=ADVSIMD
// RUN: %clang_cc1 -verify -triple aarch64-linux-gnu -target-feature +sve -fopenmp -x c++ -emit-llvm %s -o - -femit-all-decls -verify| FileCheck %s --check-prefix=SVE

// RUN: %clang_cc1 -verify -triple aarch64-linux-gnu -target-feature +neon -fopenmp-simd -x c++ -emit-llvm %s -o - -femit-all-decls -verify| FileCheck %s --check-prefix=ADVSIMD
// RUN: %clang_cc1 -verify -triple aarch64-linux-gnu -target-feature +sve -fopenmp-simd -x c++ -emit-llvm %s -o - -femit-all-decls -verify| FileCheck %s --check-prefix=SVE

// expected-no-diagnostics

#pragma omp declare simd
double f(double x);

#pragma omp declare simd
float f(float x);

void aaa(double *x, double *y, int N) {
  for (int i = 0; i < N; ++i) {
    x[i] = f(y[i]);
  }
}

void aaa(float *x, float *y, int N) {
  for (int i = 0; i < N; ++i) {
    x[i] = f(y[i]);
  }
}

// ADVSIMD: "_ZGVnN2v__Z1fd"
// ADVSIMD-NOT: _Z1fd
// ADVSIMD: "_ZGVnN4v__Z1ff"
// ADVSIMD-NOT: _Z1fF

// SVE: "_ZGVsMxv__Z1fd"
// SVE-NOT: _Z1fd
// SVE: "_ZGVsMxv__Z1ff"
// SVE-NOT: _Z1ff
