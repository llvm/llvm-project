<<<<<<< HEAD
// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 %s -fenable-ripple -o - -emit-llvm
=======
// REQUIRES: hexagon-registered-target || aarch64-registered-target || x86-registered-target
// RUN: %clang -S -Xclang -disable-llvm-passes %s -fenable-ripple -o - -emit-llvm
>>>>>>> f6954408c713 ([QTOOL-139164][Ripple] Move Ripple headers into a dedicated resource subdirectory)
// XFAIL: *

#include "../ripple_test.h"

void bar(int n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 42);
  size_t i = 32;;
  ripple_parallel(BS, 0);
  for (i = i - 32; i < n; i++)
    C[i] = A[i] + B[i];

}
