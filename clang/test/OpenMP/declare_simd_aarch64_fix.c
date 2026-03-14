// REQUIRES: aarch64-registered-target
// This test is making sure that no crash happens.

// RUN: %clang -o - -fno-fast-math -S -target aarch64-linux-gnu \
// RUN: -fopenmp -O3 -march=armv8-a  -c %s | FileCheck %s

// RUN: %clang -o - -fno-fast-math -S -target aarch64-linux-gnu \
// RUN: -fopenmp-simd -O3 -march=armv8-a  -c %s | FileCheck %s

// RUN: %clang -o - -fno-fast-math -S -target aarch64-linux-gnu \
// RUN: -fopenmp -O3 -march=armv8-a+sve  -c %s | FileCheck %s

// RUN: %clang -o - -fno-fast-math -S -target aarch64-linux-gnu \
// RUN: -fopenmp-simd -O3 -march=armv8-a+sve  -c %s | FileCheck %s

// loop in the user code, in user_code.c
#include "Inputs/declare-simd-fix.h"

// CHECK-LABEL: do_something:
void do_something(int *a, double *b, unsigned N) {
  for (unsigned i = 0; i < N; ++i) {
    a[i] = foo(b[0], b[0], 1);
  }
}

// CHECK-LABEL: do_something_else:
void do_something_else(int *a, double *b, unsigned N) {
  for (unsigned i = 0; i < N; ++i) {
    a[i] = foo(1.1, 1.2, 1);
  }
}

// CHECK-LABEL: do_something_more:
void do_something_more(int *a, double *b, unsigned N) {
  for (unsigned i = 0; i < N; ++i) {
    a[i] = foo(b[i], b[i], a[1]);
  }
}
