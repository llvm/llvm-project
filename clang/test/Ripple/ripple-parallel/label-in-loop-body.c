// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -Wripple -fenable-ripple -o - %s -DUSE_PRAGMA 2>%t; FileCheck %s --input-file %t
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -Wripple -fenable-ripple -o - %s -DUSE_CALL 2>%t; FileCheck %s --input-file %t

#include <ripple.h>

// CHECK: label-in-loop-body.c:22:1: error: the Ripple parallel loop body cannot contain labels
// CHECK-NEXT:  22 | illegal_label:
// CHECK-NEXT:     | ^~~~~~~~~~~~~

void test1(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);

#ifdef USE_PRAGMA
  #pragma ripple parallel Block(BS) Dims(0, 1)
#elif USE_CALL
  ripple_parallel(BS, 0, 1)
#else
  #error "Should define USE_CALL or USE_PRAGMA to test this file"
#endif
  for (size_t i = 0; i <= n; i++) {
illegal_label:
    C[i] = A[i] + B[i];
  }

}