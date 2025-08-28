// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Xclang -disable-llvm-passes -S -emit-llvm -fenable-ripple %s 2>%t; FileCheck %s --input-file %t

#include <ripple.h>

#ifndef HVX_LANE
#define HVX_LANE 0
#endif

typedef float f32;

void vecadd_subarray(int N, int start, int end, f32 x[__restrict 1][N],
                     f32 y[__restrict 1][N], f32 xpy[__restrict 1][N]) {
  ripple_block_t BS = ripple_set_block_shape(HVX_LANE, 32);
// CHECK:{{.*}}/missing-arguments.c:16:25:  warning: expected identifier in '#pragma ripple parallel Block()'
  ripple_parallel_full();
  for (int i = start; i < end; ++i)
    xpy[0][i] = x[0][i] + y[0][i];
}
