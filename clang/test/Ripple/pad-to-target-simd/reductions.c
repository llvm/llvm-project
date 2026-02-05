// REQUIRES: hexagon-registered-target
// RUN: %clang -S --target=hexagon -mhvx -mv81 -mhvx-length=128B -O2 -fenable-ripple -fdisable-ripple-lib -mllvm -ripple-pad-to-target-simd -emit-llvm %s -o - 2>&1 | FileCheck %s

#include "../ripple_test.h"

int32_t check_that_reduceadd_is_padded1(int32_t a[20]) {
  ripple_block_t BS = ripple_set_block_shape(0, 20);
  size_t v0 = ripple_id(BS, 0);
  return ripple_reduceadd(0b1, a[v0]);
}

// CHECK: check_that_reduceadd_is_padded1
// CHECK: call i32 @llvm.vp.reduce.add.v32i32(i32 0, <32 x i32> [[REG:%.*]], <32 x i1> splat (i1 true), i32 20)
// CHECK: ret

float check_that_reduceadd_is_padded2(float a[20]) {
  ripple_block_t BS = ripple_set_block_shape(0, 20);
  size_t v0 = ripple_id(BS, 0);
  return ripple_reduceadd(0b1, a[v0]);
}

// CHECK: check_that_reduceadd_is_padded2
// CHECK: float @llvm.vp.reduce.fadd.v32f32(float -0.000000e+00, <32 x float> [[REG:%.*]], <32 x i1> splat (i1 true), i32 20)
// CHECK: ret


int32_t check_that_masked_reduceadd_is_padded1(int32_t a[18]) {
  ripple_block_t BS = ripple_set_block_shape(0, 21);
  size_t v0 = ripple_id(BS, 0);
  int result = 0;
  if (v0 < 18)
    result = ripple_reduceadd(0b1, a[v0]);
  return result;
}

// CHECK: check_that_masked_reduceadd_is_padded1
// CHECK: call i32 @llvm.vp.reduce.add.v32i32(i32 0, <32 x i32> [[REG:%.*]], <32 x i1> splat (i1 true), i32 21)
// CHECK: ret
