// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -fenable-ripple %s -o - | FileCheck %s

#include <ripple.h>

#ifndef HVX_THREAD
#define HVX_THREAD 1
#endif

void vecadd_subarray(int N, int start, int end, float x[__restrict 1][N],
                     float y[__restrict 1][N], float xpy[__restrict 1][N]) {
  ripple_block_t BS = ripple_set_block_shape(HVX_THREAD, 32);
  // CHECK: %ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
  ripple_parallel(BS, 0);
  for (int i = start; i < end; ++i) {
    xpy[0][i] = x[0][i] + y[0][i];
  }
}