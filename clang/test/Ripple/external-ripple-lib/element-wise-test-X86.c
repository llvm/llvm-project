// REQUIRES: x86-registered-target
// RUN: %clang --target=x86_64-unknown-elf -g -c -O2 -emit-llvm %S/external_library.c -o %t.rlib.bc
// RUN: %clang --target=x86_64-unknown-elf -c -O2 -fenable-ripple -fripple-lib=%t.rlib.bc -emit-llvm -S -o - %s | FileCheck %s

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

// CHECK: ew_smaller
// CHECK: %[[Load:[a-zA-Z0-9_.]+]] = load <16 x half>
// CHECK-NEXT: %[[Shuffle1:[a-zA-Z0-9_.]+]] = shufflevector <16 x half> %[[Load]]
// CHECK-NEXT: %[[Shuffle2:[a-zA-Z0-9_.]+]] = shufflevector <32 x half> %[[Shuffle1]]
// CHECK-NEXT: store <64 x half> %[[Shuffle2]], ptr %[[ArgOnStack:[a-zA-Z0-9_.]+]]
// CHECK-NEXT: %[[LoadRet:[a-zA-Z0-9_.]+]] = tail call <64 x half> @ripple_ew_pure_cosf16(ptr nonnull %[[ArgOnStack]])
// CHECK-NEXT: %[[Extract:[a-zA-Z0-9_.]+]] = shufflevector <64 x half> %[[LoadRet]]
// CHECK: store <16 x half> %[[Extract]]

void ew_smaller(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 16);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  for (size_t i = 0; i < size; i += BlockSizeX)
    output[i + BlockX] = cosf16(input[i + BlockX]);
}

// CHECK: ew_sizematch
// CHECK: %[[Load:[a-zA-Z0-9_.]+]] = load <64 x half>
// CHECK-NEXT: store <64 x half> %[[Load]], ptr %[[ArgOnStack:[a-zA-Z0-9_.]+]]
// CHECK-NEXT: %[[LoadRet:[a-zA-Z0-9_.]+]] = tail call <64 x half> @ripple_ew_pure_cosf16(ptr nonnull %[[ArgOnStack]])
// CHECK: store <64 x half> %[[LoadRet]]

void ew_sizematch(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 64);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  for (size_t i = 0; i < size; i += BlockSizeX)
    output[i + BlockX] = cosf16(input[i + BlockX]);
}

// CHECK: ew_larger
// CHECK: %[[Load:[a-zA-Z0-9_.]+]] = load <138 x half>
// CHECK-NEXT: %[[Slice1:[a-zA-Z0-9_.]+]] = shufflevector <138 x half> %[[Load]]
// CHECK-NEXT: store <64 x half> %[[Slice1]], ptr %[[ArgOnStack1:[a-zA-Z0-9_.]+]]
// CHECK-NEXT: %[[LoadRet1:[a-zA-Z0-9_.]+]] = tail call <64 x half> @ripple_ew_pure_cosf16(ptr nonnull %[[ArgOnStack1]])
// CHECK-NEXT: %[[Slice2:[a-zA-Z0-9_.]+]] = shufflevector <138 x half> %[[Load]]
// CHECK-NEXT: store <64 x half> %[[Slice2]], ptr %[[ArgOnStack2:[a-zA-Z0-9_.]+]]
// CHECK-NEXT: %[[LoadRet2:[a-zA-Z0-9_.]+]] = tail call <64 x half> @ripple_ew_pure_cosf16(ptr nonnull %[[ArgOnStack2]])
// CHECK-NEXT: %[[Slice3:[a-zA-Z0-9_.]+]] = shufflevector <138 x half> %[[Load]]
// CHECK-NEXT: store <64 x half> %[[Slice3]], ptr %[[ArgOnStack3:[a-zA-Z0-9_.]+]]
// CHECK-NEXT: %[[LoadRet3:[a-zA-Z0-9_.]+]] = tail call <64 x half> @ripple_ew_pure_cosf16(ptr nonnull %[[ArgOnStack3]])
// CHECK-NEXT: %[[Fuse1:[a-zA-Z0-9_.]+]] = shufflevector <64 x half> %[[LoadRet1]], <64 x half> %[[LoadRet2]]
// CHECK-NEXT: %[[Fuse2:[a-zA-Z0-9_.]+]] = shufflevector <64 x half> %[[LoadRet3]]
// CHECK-NEXT: %[[FuseFinal:[a-zA-Z0-9_.]+]] = shufflevector <128 x half> %[[Fuse1]], <128 x half> %[[Fuse2]]
// CHECK: store <138 x half> %[[FuseFinal]]

void ew_larger(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 138);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  for (size_t i = 0; i < size; i += BlockSizeX)
    output[i + BlockX] = cosf16(input[i + BlockX]);
}


// CHECK: ew_multidim
// CHECK: %[[Slice1:[a-zA-Z0-9_.]+]] = shufflevector <128 x half> %[[Load:[a-zA-Z0-9_.]+]]
// CHECK-NEXT: store <64 x half> %[[Slice1]], ptr %[[ArgOnStack1:[a-zA-Z0-9_.]+]]
// CHECK-NEXT: %[[LoadRet1:[a-zA-Z0-9_.]+]] = tail call <64 x half> @ripple_ew_pure_cosf16(ptr nonnull %[[ArgOnStack1]])
// CHECK-NEXT: %[[Slice2:[a-zA-Z0-9_.]+]] = shufflevector <128 x half> %[[Load]]
// CHECK-NEXT: store <64 x half> %[[Slice2]], ptr %[[ArgOnStack2:[a-zA-Z0-9_.]+]]
// CHECK-NEXT: %[[LoadRet2:[a-zA-Z0-9_.]+]] = tail call <64 x half> @ripple_ew_pure_cosf16(ptr nonnull %[[ArgOnStack2]])
// CHECK-NEXT: %[[FuseFinal:[a-zA-Z0-9_.]+]] = shufflevector <64 x half> %[[LoadRet1]], <64 x half> %[[LoadRet2]]

void ew_multidim(size_t size, _Float16 *input, _Float16 *output) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 4, 32);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockY = ripple_id(BS, 1);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  size_t BlockSizeY = ripple_get_block_size(BS, 1);
  for (size_t i = 0; i < size; i += BlockSizeX + BlockSizeY)
    output[i + BlockX + BlockY] = cosf16(input[i + BlockX + BlockY]);
}
