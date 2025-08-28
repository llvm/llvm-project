// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

extern __attribute__((noinline)) __attribute__((visibility("hidden"))) int toBeSpecialized(int n) {
  return n * 32;
}

extern "C" {

// CHECK-LABEL: @test

void test(int in[128], int *out) {
  auto BS = ripple_set_block_shape(0, 128);
  size_t idx = ripple_id(BS, 0);
  out[idx] = toBeSpecialized(in[idx]);
}

}
