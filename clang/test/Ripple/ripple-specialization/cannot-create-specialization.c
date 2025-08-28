// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s 2>%t; FileCheck %s --input-file=%t

#include <ripple.h>
#include <stdint.h>

__attribute__((noinline)) int toBeSpecialized(int32_t n, int32_t m) {
  return ripple_reduceadd(0b10, n * m) * 32;
}

void test3(int32_t *in, int32_t *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 3);
  size_t idx = ripple_id(BS, 0);
  int32_t val_lhs = in[idx];
  ripple_block_t BS2 = ripple_set_block_shape(0, 2);
  idx = ripple_id(BS2, 0);
  int32_t val_rhs = in[idx];
  int x = 34;
  if (idx <= 2)
    x = toBeSpecialized(val_lhs, val_rhs);
  out[idx] = x;
}

// CHECK: cannot-create-specialization.c:20:{{.*}}The function toBeSpecialized does not follow the ripple broadcast rule:  the operands tensor shapes are not broadcast-compatible: [Tensor[3], Tensor[2]]
// CHECK: cannot-create-specialization.c:20:{{.*}}broadcast failure: cannot apply the broadcast rule between dimension size 3 and 2 at dimension index 0
// CHECK-NEXT: cannot-create-specialization.c:20:{{.*}}Ripple failed to broadcast the instruction of shape Tensor[3]
// CHECK-NEXT: cannot-create-specialization.c:17:{{.*}}with the shape coming from the operand 1 of shape Tensor[2]
