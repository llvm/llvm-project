// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -c -O2 -emit-llvm %S/external_library.c -o %t.rlib.bc
// RUN: %clang -c -O2 -fenable-ripple -emit-llvm -S -o - -fripple-lib %t.rlib.bc -mllvm -ripple-disable-link %s | FileCheck %s

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

// CHECK: masked_and_non_masked
// CHECK: ripple_ew_non_pure_ew_separate_mask
// CHECK: ripple_ew_mask_non_pure_ew_separate_mask
void masked_and_non_masked(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  size_t i;
  for (i = 0; i + BlockSizeX < size; i += BlockSizeX)
    output[i + BlockX] = non_pure_ew_separate_mask(input[i + BlockX]);
  if(i + BlockX < size)
    output[i + BlockX] = non_pure_ew_separate_mask(input[i + BlockX]);
}

// CHECK: masked_call_only
// CHECK: ripple_ew_mask_non_pure_ew_separate_mask
void masked_call_only(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  size_t i;
  for (i = 0; i + BlockSizeX < size; i += BlockSizeX)
    if(i + BlockX < size)
      output[i + BlockX] = non_pure_ew_separate_mask(input[i + BlockX]);
}
