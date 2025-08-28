// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -fenable-ripple %s -o - | FileCheck %s
// Test case where a non-simple loop bound is used

#include <ripple.h>
#define HVX_LANE 0

#define min(a, b) (a < b ? a : b)

// CHECK: vecadd_subarray
void vecadd_subarray(int N, int start, int end, float x[__restrict 1][N],
                     float y[__restrict 1][N], float xpy[__restrict 1][N]) {
  ripple_block_t BS = ripple_set_block_shape(HVX_LANE, 32);
// CHECK: %ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
  ripple_parallel(BS, 0);
  for (int i = start; i < min(end, 129); ++i) {
    xpy[0][i] = x[0][i] + y[0][i];
  }
}
