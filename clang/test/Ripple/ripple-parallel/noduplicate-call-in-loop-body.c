// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -Wripple -fenable-ripple -o - %s -DUSE_PRAGMA 2>%t; FileCheck %s --input-file %t
// RUN: %clang_cc1 -disable-llvm-passes -emit-llvm -Wripple -fenable-ripple -o - %s -DUSE_CALL 2>%t; FileCheck %s --input-file %t

#include <ripple.h>

extern void nodupfunc() __attribute__((noduplicate));

// CHECK: noduplicate-call-in-loop-body.c:28:5: error: the Ripple parallel loop body cannot contain calls to function with the attribute 'noduplicate'
// CHECK-NEXT: 28 |     nodupfunc();
// CHECK-NEXT:    |     ^~~~~~~~~~~
// CHECK: noduplicate-call-in-loop-body.c:7:1: note: declared here
// CHECK-NEXT:  7 | extern void nodupfunc() __attribute__((noduplicate));
// CHECK-NEXT:    | ^

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
    C[i] = A[i] + B[i];
    nodupfunc();
  }

}