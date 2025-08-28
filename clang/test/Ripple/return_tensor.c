// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - &> %t; FileCheck %s --input-file %t

#include <ripple.h>


// CHECK: Ripple does not allow vectorization of the return value

float check(float *Input) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  unsigned BlockIdx = ripple_id(BS, 0);
  return Input[ripple_id(BS, 0)];
}
