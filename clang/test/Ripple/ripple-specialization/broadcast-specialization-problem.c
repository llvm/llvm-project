// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s 2>%t; FileCheck %s --input-file=%t

#include <ripple.h>

__attribute__((noinline)) int toBeSpecialized(int n) {
  ripple_block_t BS = ripple_set_block_shape(0, 3);
  return n / ripple_id(BS, 0);
}

// CHECK: broadcast-specialization-problem.c:8:{{.*}}broadcast failure: cannot apply the broadcast rule between dimension size 4 and 3 at dimension index 0
// CHECK-NEXT: 8 |{{.*}}return n / ripple_id(BS, 0);

void test(int *in, int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized(in[idx]);
}

