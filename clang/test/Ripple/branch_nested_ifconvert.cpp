// REQUIRES: asserts
// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -O1 -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

void vecBranch(unsigned size, int a[32][size][size], int b[32]) {
  auto BS = ripple_set_block_shape(0, 4, 8);
  unsigned blockIdx_0 = ripple_id(BS, 0);
  unsigned blockSize_0 = ripple_get_block_size(BS, 0);
  unsigned blockIdx_1 = ripple_id(BS, 1);
  unsigned blockSize_1 = ripple_get_block_size(BS, 1);
  for (unsigned i = 0; i < 32; ++i) {
    int red = 0;
    unsigned j;
    unsigned k;
    for (j = 0; j + blockSize_0 < size; j += blockSize_0) {
      for (k = 0; k + blockSize_1 < size; k += blockSize_1) {
        red += a[i][j + blockIdx_0][k + blockIdx_1];
      }
      if (k + blockIdx_1 < size)
        red += a[i][j + blockIdx_0][k + blockIdx_1];
    }
    if (j + blockIdx_0 < size)
      if (k + blockIdx_1 < size)
        red += a[i][j + blockIdx_0][k+blockIdx_1];

    b[i] = ripple_reduceadd(0b11, red);
  }
}
