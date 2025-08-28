// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// REQUIRES: asserts
// RUN: %clang -S -fenable-ripple -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <stdint.h>
#include <stddef.h>
#include <ripple.h>
#define VEC 0

void baz(uint32_t N, const float A[N][20], float B[20][16]) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 8, 10);
  size_t v0 = ripple_id(BS, 0), v1 = ripple_id(BS, 1);
  float acc = A[0][v1];
  B[v1][v0] = acc;
}
