// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang %s -std=c++20 -fenable-ripple -Xclang -disable-llvm-passes -Wripple -S -emit-llvm -o - | FileCheck %s --implicit-check-not="warning:"

#include <ripple.h>

// CHECK: foo
// CHECK-COUNT-21: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK-NOT: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
void foo(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);

  ripple_parallel(BS, 0, 1);             // 1
  for (int i = 0; i < n; i++) {
    ripple_parallel(BS, 0, 1);           // + 2
    for (int i = 0; i < n; i++) {
      ripple_parallel(BS, 0, 1);         // + 4
      for (int i = 0; i < n; i++) {
        C[i] += A[i] + B[i];
      }
      C[i] += A[i] + B[i];
      ripple_parallel(BS, 0, 1);         // + 4
      for (int i = 0; i < n; i++) {
        C[i] += A[i] + B[i];
      }
    }
    C[i] += A[i] + B[i];
    ripple_parallel(BS, 0, 1);           // + 2
    for (int i = 0; i < n; i++) {
      ripple_parallel(BS, 0, 1);         // + 4
      for (int i = 0; i < n; i++) {
        C[i] += A[i] + B[i];
      }
      C[i] += A[i] + B[i];
    }
    ripple_parallel(BS, 0, 1);           // + 2
    for (int i = 0; i < n; i++) {
      C[i] += A[i] + B[i];
    }
    ripple_parallel(BS, 0, 1);           // + 2
    for (int i = 0; i < n; i++) {
      C[i] += A[i] + B[i];
    }
  }
}
