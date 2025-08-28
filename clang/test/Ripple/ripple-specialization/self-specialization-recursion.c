// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s 2> %t; FileCheck --implicit-check-not="warning:" %s --input-file=%t

#include <ripple.h>

__attribute__((noinline)) int toBeSpecialized(int n, int p) {
  if (p <= 0)
    return 42;
  return toBeSpecialized(n, p - 2) * toBeSpecialized(n, p - 1);
}

// CHECK: error: Ripple encountered a cycle with the following functions and cannot reliably propagate tensor shapes in this case: toBeSpecialized

void test(int *in, int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized(in[idx], 4);
}
