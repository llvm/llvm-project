// REQUIRES: x86-registered-target
// RUN: %clang --target=x86_64-unknown-elf -c -O2 -emit-llvm %S/external_library.c -o %t.rlib.bc
// RUN: %clang --target=x86_64-unknown-elf -c -O2 -fenable-ripple -emit-llvm -S -o - -fripple-lib %t.rlib.bc %s | FileCheck %s

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

// CHECK-LABEL: call_masked_fn
// CHECK: br i1 %{{.*}}, label %[[FOR_BODY:[a-zA-Z0-9_.]+]], label %[[FOR_END:[a-zA-Z0-9_.]+]]
// CHECK: [[FOR_BODY]]:
// CHECK: br i1 %{{.*}}, label %[[FOR_BODY]], label %[[FOR_END]]
// CHECK: [[FOR_END]]:
// CHECK: %[[Mask:[a-zA-Z0-9_.]+]] = icmp ult <64 x i64>
// CHECK: %[[MaskExtend:[a-zA-Z0-9_.]+]] = zext <64 x i1> %[[Mask]] to <64 x i8>
// CHECK: store <64 x i8> %[[MaskExtend]], ptr %[[MaskByValPtr:[a-zA-Z0-9_.]+]]
// CHECK: ripple_mask_add_and_half_ptr({{.*}}, ptr nonnull %[[MaskByValPtr]])
void call_masked_fn(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  size_t i;
  for (i = 0; i + BlockSizeX < size; i += BlockSizeX)
    output[i + BlockX] = add_and_half_ptr(input[i + BlockX], input);
  if(i + BlockX < size)
    output[i + BlockX] = add_and_half_ptr(input[i + BlockX], input);
}
