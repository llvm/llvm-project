// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -c -O2 -emit-llvm %S/external_library.c -o %t
// RUN: %clang -c -O2 -fenable-ripple -emit-llvm -S -o - -ffast-math -fripple-lib %t %s | FileCheck %s --implicit-check-not="warning:"

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

// We are looking for 4 calls (2 inside the for loop and 2 masked)
// CHECK: ew_cos_external
// CHECK: ripple_ew_pure_cosf
// CHECK: ripple_ew_pure_cosf
// CHECK: ripple_ew_pure_cosf
// CHECK: ripple_ew_pure_cosf
void ew_cos_external(size_t size, float *input, float *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  size_t i;
  for (i = 0; i + BlockSizeX < size; i += BlockSizeX)
    output[i + BlockX] = cosf(input[i + BlockX]);
  if(i + BlockX < size)
    output[i + BlockX] = cosf(input[i + BlockX]);
}

