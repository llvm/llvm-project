// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -fenable-ripple %s -o - | FileCheck %s

#include <ripple.h>

#ifndef HVX_LANE
#define HVX_LANE 0
#endif

void vecadd_subarray(int N, int start, int end, float x[__restrict 1][N],
                     float y[__restrict 1][N], float xpy[__restrict 1][N]) {
  ripple_block_t BS = ripple_set_block_shape(HVX_LANE, 32);
  int a = 3;
  // CHECK: %ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
  while (a-- > 0) {
    // while.body:
    // CHECK: store i{{[0-9]+}} %{{[0-9a-zA-Z_]+}}, ptr %ripple.loop.iters
    ripple_parallel(BS, 0);
    for (int i = start; i < end; ++i) {
      xpy[0][i] = x[0][i] + y[0][i];
    }
  }
}