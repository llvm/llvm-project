// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

// CHECK-NOT: @ripple.specialization.

__attribute__((noinline,weak)) int toBeSpecialized(int n, int m, int p) {
  return n * m * p * 32;
}

void test(int *in, int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized(in[idx], in[0], out[idx]);
}
