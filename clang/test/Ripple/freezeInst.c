// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o - -mllvm -print-after=ripple 2>&1 | FileCheck %s

#include <ripple.h>

#define VectorPE 0

// InstCombine removes the freeze instruction since there are no
// poison/undefined but ripple still has to handle it so we test for successful
// compilation
// CHECK: @ripple_function
// CHECK: <42 x float>

void ripple_function(size_t size, float*A, float* B, float*C) {
    ripple_block_t BS = ripple_set_block_shape(VectorPE, 42);
    size_t VecIdx = ripple_id(BS, 0);
    size_t VecSize = ripple_get_block_size(BS, 0);
    for (size_t i = 0; i < size; i += VecSize)
      if (i + VecIdx < size) {
        if (VecIdx < VecSize / 2)
          C[i + VecIdx] = A[i+VecIdx] * 2 + B[i+VecIdx] / 2;
        else
          C[i + VecIdx] = A[i+VecIdx] / 2 + B[i+VecIdx] * 2;
      }
}
