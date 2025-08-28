// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -O1 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

void vecSwitch(unsigned size, int a[size], int *b) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  unsigned blockIdx = ripple_id(BS, 0);
  int val = 0;
  switch (blockIdx * size) {
    case 0:
      val = a[1];
      [[fallthrough]];
    case 1:
      val = a[0];
      break;
    case 2:
      val = -a[2];
      break;
    case 5:
      val = a[5] * a[6];
      break;
    case 6:
      break;
    default:
      val = a[blockIdx];
      break;
  }
  *b = ripple_reduceadd(0b1, val);
}
