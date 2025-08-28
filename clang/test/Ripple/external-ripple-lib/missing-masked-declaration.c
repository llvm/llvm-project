// REQUIRES: target-x86_64 || target=hexagon{{.*}}
// RUN: %clang -g -c -O2 -emit-llvm %S/external_library.c -o %t.lib.bc
// RUN: %clang -Wall -Wextra -Wpedantic -Wripple -g -O2 -fenable-ripple -emit-llvm -S -o - -mllvm -ripple-lib=%t.lib.bc %s 2> %t; FileCheck %s --input-file=%t

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

void missing_masked_version(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  size_t i;
  for (i = 0; i + BlockSizeX < size; i += BlockSizeX)
    output[i + BlockX] = add_and_half_ptr_has_no_mask_version(input[i + BlockX], input);
  if(i + BlockX < size)
    output[i + BlockX] = add_and_half_ptr_has_no_mask_version(input[i + BlockX], input);
}

// CHECK: missing-masked-declaration.c:19:{{.*}}: call to an external ripple function (ripple_add_and_half_ptr_has_no_mask_version) requires masking but no maskable declaration is available
// CHECK: missing-masked-declaration.c:11:{{.*}}: Ripple failed to vectorize this function
