// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -std=c99 -Wall -Wextra -Wpedantic -Xclang -disable-llvm-passes -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

// CHECK: check_nested_reduceadd
void check_nested_reduceadd(int a[128], float b[128], int *OutPtr) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t idx_x = ripple_id(BS, 0);
  int tmp = a[idx_x];
  float tmp2 = b[idx_x];
  // ripple_reduceadd uses _Generic
  // CHECK: @llvm.ripple.reduce.add.i[[IntSize:[0-9]+]](i64 1, i[[IntSize]]
  // CHECK: @llvm.ripple.reduce.fadd.f32(i64 1, float
  int out = ripple_reduceadd(0x1, tmp2 + (float)ripple_reduceadd(0x1, tmp));
  OutPtr[0] = out;
}
