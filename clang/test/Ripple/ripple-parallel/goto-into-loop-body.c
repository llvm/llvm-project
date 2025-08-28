// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -Wripple -fenable-ripple -o - %s -DUSE_PRAGMA | FileCheck --implicit-check-not="warning:" %s
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -Wripple -fenable-ripple -o - %s -DUSE_CALL | FileCheck --implicit-check-not="warning:" %s

#include <ripple.h>

// CHECK: test1
void test1(size_t n, float * C, float * A, float * B) {
  ripple_block_t BS = ripple_set_block_shape(0, 8);

#ifdef USE_PRAGMA
  #pragma ripple parallel Block(BS) Dims(0)
#elif USE_CALL
  ripple_parallel(BS, 1)
#else
  #error "Should define USE_CALL or USE_PRAGMA to test this file"
#endif
// CHECK: for.body:
// CHECK: br label %EndLabel
// CHECK: ripple.par.for.remainder.body:                    ; preds = %ripple.par.for.remainder.cond
// CHECK: br label %EndLabel
  for (size_t i = 0; i <= n; i++) {
    C[i] = A[i] + B[i];
    goto EndLabel;
  }

EndLabel:
  return;
}
