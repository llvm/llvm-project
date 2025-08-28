// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wpedantic -fenable-ripple -S -O2 -emit-llvm %s -o - | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

// Templates have LinkOnceODRType linkage

template <typename T>
__attribute__((noinline)) T toBeSpecialized(T n) {
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
