// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -fenable-ripple %s -o - | FileCheck %s
// XFAIL: *

// Test where a loop maps to 2 different PE dimensions (a thread, a vector)

#include <ripple.h>

#ifndef HVX_LANE
#define HVX_LANE 0
#endif
#ifndef HVX_THD
#define HVX_THD 1
#endif

void vecadd_subarray(int N, int start, int end, float x[__restrict 1][N],
                     float y[__restrict 1][N], float xpy[__restrict 1][N]) {
  ripple_block_t BS = ripple_set_block_shape(HVX_LANE, 32);
  ripple_block_t BS2 = ripple_set_block_shape(HVX_THD, 32);
  int i;
  ripple_parallel(BS, 0);
  ripple_parallel(BS2, 0);
  for (i = start; i < end; ++i)
    xpy[0][i] = x[0][i] + y[0][i];
}
