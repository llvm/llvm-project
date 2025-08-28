// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s 2> %t; FileCheck --implicit-check-not="warning:" %s --input-file=%t

#include <ripple.h>

__attribute__((noinline)) int toBeSpecialized1(int n);
__attribute__((noinline)) int toBeSpecialized2(int n);
__attribute__((noinline)) int toBeSpecialized3(int n);

int toBeSpecialized1(int n) {
  return n * toBeSpecialized2(n);
}

int toBeSpecialized2(int n) {
  return n * toBeSpecialized3(n);
}

int toBeSpecialized3(int n) {
  return n * toBeSpecialized1(n);
}

// CHECK: error: Ripple encountered a cycle with the following functions and cannot reliably propagate tensor shapes in this case: toBeSpecialized1, toBeSpecialized2, toBeSpecialized3

void test(int *in, int *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized1(in[idx]);
}

