// REQUIRES: target=hexagon{{.*}} || target-x86_64 || target-aarch64
// RUN: %clang -Wsign-compare -S -emit-llvm -O2 -fenable-ripple -o - %s &>%t
// RUN: FileCheck %s --input-file=%t --check-prefix=ERRCHECK

#include <ripple.h>
void f(int start, int end, float *x, float *y, float *xpy) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  ripple_parallel(BS, 0);
  for (int i = start; i < end; ++i)
    xpy[i] = x[i] + y[i] + (float)i;
}

// ERRCHECK-NOT: warning: comparison of integers of different signs

