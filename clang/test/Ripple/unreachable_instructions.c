// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang %s -fenable-ripple -O0 -S -emit-llvm -o - | FileCheck %s

#include <ripple.h>
#include <stddef.h>

// CHECK: foo
void foo(int n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  size_t x = ripple_id(BS, 0);
  // i++ is unreachable from entry
  for (size_t i = 0; i < n; i++) {
    C[i + x] += A[i + x] + B[i + x];
    break;
  }
}
