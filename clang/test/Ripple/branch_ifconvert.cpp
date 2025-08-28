// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -O1 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

void vecBranch(unsigned size, int a[32][size], int b[32]) {
  auto BS = ripple_set_block_shape(0, 32);
  unsigned blockIdx = ripple_id(BS, 0);
  unsigned blockSize = ripple_get_block_size(BS, 0);
  for (unsigned i = 0; i < 32; ++i) {
    int red = 0;
    unsigned j;
    for (j = 0; j < size; j += blockSize) {
      red += a[i][j + blockIdx];
    }
    if (j + blockIdx < size)
      red += a[i][j + blockIdx];
    b[i] = ripple_reduceadd(0b1, red);
  }
}
