// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -O1 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>
#include <stdint.h>


void oneD(uint8_t x[35], uint8_t y[35]) {
  ripple_block_t BS = ripple_set_block_shape(0, 35);
  size_t v0 = ripple_id(BS, 0);
  uint8_t tmp = x[v0];
  y[v0] = tmp;
}

void twoD(uint8_t x[128], uint8_t y[128]) {
  ripple_block_t BS = ripple_set_block_shape(0, 32, 4);
  size_t v0 = ripple_id(BS, 0);
  size_t v1 = ripple_id(BS, 1);
  uint8_t tmp = x[v0 + 32 * v1];
  y[v0 + 4 * v1] = tmp;
}

void threeD(uint8_t x[128], uint8_t y[128]) {
  ripple_block_t BS = ripple_set_block_shape(0, 32, 2, 2);
  size_t v0 = ripple_id(BS, 0);
  size_t v1 = ripple_id(BS, 1);
  size_t v2 = ripple_id(BS, 2);
  uint8_t tmp = x[v0 + 32 * v1 + 32 * 2 * v2];
  y[v0 + 2 * v1 + v2] = tmp;
}
