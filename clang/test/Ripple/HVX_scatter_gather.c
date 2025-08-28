// XFAIL: *
// REQUIRES: hexagon-registered-target
// RUN: %clang -g -S -fenable-ripple --target=hexagon -emit-llvm %s -o - -mllvm -ripple-disable-link  2>&1 | FileCheck %s

#include <ripple.h>
#include "HVX_scatter_gather.h"

#define HVX_32_i32 32
__attribute__((noinline)) void Ripple_scatter_32_library(unsigned int *destination, unsigned int *source, unsigned int *indexes)  {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  size_t v = ripple_id(BS, 0);
  int _ripple_block_0;
  int _ripple_ub_0 = ((HVX_32_i32 - 0) / ripple_get_block_size(BS, 0));
  for (_ripple_block_0 = 0; _ripple_block_0 < _ripple_ub_0; _ripple_block_0 += 1) {
    int i = ripple_get_block_size(BS, 0) * _ripple_block_0 + 0 + ripple_id(BS, 0);
    HVX_scatter_32(destination, source, indexes, 64, 8192);
  }
}
// CHECK: @Ripple_scatter_32_library
// CHECK: tail call void @ripple_HVX_scatter_32
// CHECK: ret void
