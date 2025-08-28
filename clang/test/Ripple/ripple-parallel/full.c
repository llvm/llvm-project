// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -fenable-ripple %s -o - | FileCheck %s

#include <ripple.h>

#ifndef HVX_LANE
#define HVX_LANE 0
#endif

typedef float f32;

void vecadd_subarray(int N, int start, int end, f32 x[__restrict 1][N],
                     f32 y[__restrict 1][N], f32 xpy[__restrict 1][N]) {
  ripple_block_t BS = ripple_set_block_shape(HVX_LANE, 32);
  ripple_parallel_full(BS, 0);
// CHECK: %ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK-NOT: ripple.par.for.remainder.body{{[0-9]*}}
  for (int i = start; i < end; ++i)
    xpy[0][i] = x[0][i] + y[0][i];
}
