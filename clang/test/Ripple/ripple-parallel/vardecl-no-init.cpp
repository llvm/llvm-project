// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -fenable-ripple -S -emit-llvm %s -o - | FileCheck %s

#include <ripple.h>

void check(float *x,
           float *y, float *xpy) {
  auto BS = ripple_set_block_shape(0, 32);
  // CHECK: %ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
  ripple_parallel(BS, 0);
  for (int i = 0; i < 20; ++i) {
    [[maybe_unused]] int p;
    xpy[i] = x[i] + y[i];
  }
}
