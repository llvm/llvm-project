// REQUIRES: target=hexagon{{.*}} || target-aarch64 || target-x86_64
// RUN: %clang -g %s -O2 -fenable-ripple -S -emit-llvm 2>%t; FileCheck %s --input-file=%t

#include <ripple.h>

void check(float *A, float *B, float C[1][32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t i = ripple_id(BS, 10);
  size_t j = ripple_id(BS, 0);
  C[i][j] = A[i] + B[j];
  C[0][0] += ripple_get_block_size(BS, 1) + ripple_get_block_size(BS, 0);
}

// CHECK: exceeding-ripple-proc-dim-id.c:8{{.*}}: error: the requested dimension index (10) exceeds the number of dimensions supported by Ripple; supported values are in the range [0, 9] per block shape
// CHECK-NEXT: 8 |{{.*}}size_t i = ripple_id(BS, 10);
