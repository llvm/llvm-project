// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -c -O2 -emit-llvm %S/external_library.c -o %t.rlib.bc
// RUN: %clang -c -O2 -fenable-ripple -fripple-lib %t.rlib.bc -emit-llvm -S -o - -mllvm -ripple-disable-link %s | FileCheck %s

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

extern _Float16 mysinf16scal(_Float16);

// CHECK: no_match_sinf16
// We expect loop w/ scalar call
// CHECK: call{{.*}}@mysinf16scal

void no_match_sinf16(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 63);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  for (size_t i = 0; i < size; i += BlockSizeX)
    output[i + BlockX] = mysinf16scal(input[i + BlockX]);
}

// CHECK: match_sinf16
// We expect the library call
// CHECK: call{{.*}}@ripple_mysinf16

void match_sinf16(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  for (size_t i = 0; i < size; i += BlockSizeX)
    output[i + BlockX] = mysinf16(input[i + BlockX]);
}

// CHECK: match_ripple_add_and_half
// We expect the library call
// CHECK: call{{.*}}@ripple_add_and_half

void match_ripple_add_and_half(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  for (size_t i = 0; i < size; i += BlockSizeX)
    output[i + BlockX] = add_and_half(input[i + BlockX], 2.f16);
}

// CHECK: no_match_ripple_add_and_half
// We expect loop w/ scalar call
// CHECK: call{{.*}}@add_and_half

void no_match_ripple_add_and_half(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  for (size_t i = 0; i < size; i += BlockSizeX)
    output[i + BlockX] = add_and_half(input[i + BlockX], input[i + BlockX]);
}
