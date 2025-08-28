// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 %s -fenable-ripple -o - -emit-llvm
// XFAIL: *

#include <ripple.h>

void bar(int n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 42);
  size_t i = 32;;
  ripple_parallel(BS, 0);
  for (i = i - 32; i < n; i++)
    C[i] = A[i] + B[i];

}
