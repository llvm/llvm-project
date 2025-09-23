// REQUIRES: target=hexagon{{.*}} || target-x86_64 || target-aarch64
// RUN: %clang -S -emit-llvm -O1 -fenable-ripple -save-temps %s -o - | FileCheck %s

// CHECK-NOT: llvm.ripple
// CHECK: <4 x float>
// CHECK-NOT: llvm.ripple

#include <ripple.h>

void check(unsigned size, float *a, float *b, float *c) {
  ripple_block_t bs = ripple_set_block_shape(0, 4);

  for (unsigned i = 0; i < size; i+= ripple_get_block_size(bs, 0)) {
    unsigned rid = ripple_id(bs, 0);
    c[i + rid] = a[i + rid] * b[i + rid];
  }
}
