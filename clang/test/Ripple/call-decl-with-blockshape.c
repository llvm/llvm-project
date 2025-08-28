// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O0 -emit-llvm %s 2>%t; FileCheck %s --input-file=%t

#include <ripple.h>

extern float declWithBS(ripple_block_t BS, float *);

void check(float *In, float *Out) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);

  Out[ripple_id(BS, 0)] = declWithBS(BS, In);
}

// CHECK: call-decl-with-blockshape.c:{{.*}}: Passing a ripple block shape to a function call with no known definition is not allowed. Make sure that the function is available for ripple processing.
