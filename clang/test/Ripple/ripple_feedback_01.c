// REQUIRES: hexagon-registered-target
// RUN: %clang -S -Os -emit-llvm -fenable-ripple -target hexagon -mv75 -mhvx -mllvm -ripple-vectorize-analyze %s -o - 2> %t; FileCheck %s --input-file %t

#include <ripple.h>

#define VEC_SIZE 7

void AT_dot_B_perfect_tiled_variant(int Nk, const float (*A)[VEC_SIZE],  const float (*B)[10], float (*C)[10]) {
  ripple_block_t BS = ripple_set_block_shape(0, VEC_SIZE, VEC_SIZE);

  float A_tile[VEC_SIZE][VEC_SIZE];
  size_t v0 = ripple_id(BS, (0)), v1 = ripple_id(BS, (1));

  for (int j = 0; j < 10; j += VEC_SIZE) {
    float acc = 0;
    for (int k_inner = 0; k_inner < VEC_SIZE; k_inner++)
      acc += A_tile[k_inner][v1];
    if (v1 < Nk)
      A_tile[v1][v0] = A[v1][v0];
    C[v1][v0] = acc;
  }
}

// CHECK: Verify Tensor Shape: warning: Tensor shape[Tensor[7][7]] will require type(<49 x float>) and will be scalarized by the back end
