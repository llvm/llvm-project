// REQUIRES: hexagon-registered-target
// RUN: %clang --target=hexagon-unknown-elf -g -c -O2 -emit-llvm %S/external_library.c -o %t.rlib.bc
// RUN: %clang --target=hexagon-unknown-elf -c -O2 -fenable-ripple -fripple-lib=%t.rlib.bc -emit-llvm -S -o - %s | FileCheck %s

#include "external_library.h"
#include <stddef.h>
#include <ripple.h>

#define VEC 0

// CHECK: ew_smaller
// CHECK: %[[Load:[a-zA-Z0-9_.]+]] = load <16 x half>
// CHECK-NEXT: %[[Shuffle1:[a-zA-Z0-9_.]+]] = shufflevector <16 x half> %[[Load]]
// CHECK-NEXT: %[[Shuffle2:[a-zA-Z0-9_.]+]] = shufflevector <32 x half> %[[Shuffle1]]
// CHECK-NEXT: call void @ripple_ew_pure_cosf16(ptr nonnull %[[RetBuffer:[a-zA-Z0-9_.]+]], <64 x half> %[[Shuffle2]])
// CHECK-NEXT: %[[LoadRet:[a-zA-Z0-9_.]+]] = load <64 x half>, ptr %[[RetBuffer]]
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
// CHECK-NEXT: call void @ripple_ew_pure_cosf16(ptr nonnull %[[RetBuffer:[a-zA-Z0-9_.]+]], <64 x half> %[[Load]])
// CHECK-NEXT: %[[LoadRet:[a-zA-Z0-9_.]+]] = load <64 x half>, ptr %[[RetBuffer]]
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
// CHECK-NEXT: call void @ripple_ew_pure_cosf16(ptr nonnull %[[RetBuffer:[a-zA-Z0-9_.]+]], <64 x half> %[[Slice1]])
// CHECK-NEXT: %[[LoadRet1:[a-zA-Z0-9_.]+]] = load <64 x half>, ptr %[[RetBuffer]]
// CHECK-NEXT: %[[Slice2:[a-zA-Z0-9_.]+]] = shufflevector <138 x half> %[[Load]]
// CHECK-NEXT: call void @ripple_ew_pure_cosf16(ptr nonnull %[[RetBuffer:[a-zA-Z0-9_.]+]], <64 x half> %[[Slice2]])
// CHECK-NEXT: %[[LoadRet2:[a-zA-Z0-9_.]+]] = load <64 x half>, ptr %[[RetBuffer]]
// CHECK-NEXT: %[[Slice3:[a-zA-Z0-9_.]+]] = shufflevector <138 x half> %[[Load]]
// CHECK-NEXT: call void @ripple_ew_pure_cosf16(ptr nonnull %[[RetBuffer:[a-zA-Z0-9_.]+]], <64 x half> %[[Slice3]])
// CHECK-NEXT: %[[LoadRet3:[a-zA-Z0-9_.]+]] = load <64 x half>, ptr %[[RetBuffer]]
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
// CHECK-NEXT: call void @ripple_ew_pure_cosf16(ptr nonnull %[[RetBuffer:[a-zA-Z0-9_.]+]], <64 x half> %[[Slice1]])
// CHECK-NEXT: %[[LoadRet1:[a-zA-Z0-9_.]+]] = load <64 x half>, ptr %[[RetBuffer]]
// CHECK-NEXT: %[[Slice2:[a-zA-Z0-9_.]+]] = shufflevector <128 x half> %[[Load]]
// CHECK-NEXT: call void @ripple_ew_pure_cosf16(ptr nonnull %[[RetBuffer:[a-zA-Z0-9_.]+]], <64 x half> %[[Slice2]])
// CHECK-NEXT: %[[LoadRet2:[a-zA-Z0-9_.]+]] = load <64 x half>, ptr %[[RetBuffer]]
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

// CHECK: ew_multidim_bcast_void
// CHECK: for.body:
// CHECK: [[Input1Load:%.*]] = load <32 x float>, ptr
// CHECK: [[Input1Bcast:%.*]] = shufflevector <32 x float> [[Input1Load]], <32 x float> poison, <128 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 3, i32 4, i32 4, i32 4, i32 4, i32 5, i32 5, i32 5, i32 5, i32 6, i32 6, i32 6, i32 6, i32 7, i32 7, i32 7, i32 7, i32 8, i32 8, i32 8, i32 8, i32 9, i32 9, i32 9, i32 9, i32 10, i32 10, i32 10, i32 10, i32 11, i32 11, i32 11, i32 11, i32 12, i32 12, i32 12, i32 12, i32 13, i32 13, i32 13, i32 13, i32 14, i32 14, i32 14, i32 14, i32 15, i32 15, i32 15, i32 15, i32 16, i32 16, i32 16, i32 16, i32 17, i32 17, i32 17, i32 17, i32 18, i32 18, i32 18, i32 18, i32 19, i32 19, i32 19, i32 19, i32 20, i32 20, i32 20, i32 20, i32 21, i32 21, i32 21, i32 21, i32 22, i32 22, i32 22, i32 22, i32 23, i32 23, i32 23, i32 23, i32 24, i32 24, i32 24, i32 24, i32 25, i32 25, i32 25, i32 25, i32 26, i32 26, i32 26, i32 26, i32 27, i32 27, i32 27, i32 27, i32 28, i32 28, i32 28, i32 28, i32 29, i32 29, i32 29, i32 29, i32 30, i32 30, i32 30, i32 30, i32 31, i32 31, i32 31, i32 31>
// CHECK: [[Input2Load:%.*]] = load <4 x float>, ptr
// CHECK: [[Input2Bcast:%.*]] = shufflevector <4 x float> [[Input2Load]], <4 x float> poison, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
// CHECK: [[Extract1_quarter_1:%.*]] = shufflevector <128 x float> [[Input1Bcast]], <128 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
// CHECK: [[Extract2_quarter_1:%.*]] = shufflevector <128 x float> [[Input2Bcast]], <128 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
// CHECK: tail call void @ripple_ew_side_effect_no_return(<32 x float> [[Extract1_quarter_1]], <32 x float> [[Extract2_quarter_1]])
// CHECK: [[Extract1_quarter_2:%.*]] = shufflevector <128 x float> [[Input1Bcast]], <128 x float> poison, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
// CHECK: [[Extract2_quarter_2:%.*]] = shufflevector <128 x float> [[Input2Bcast]], <128 x float> poison, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
// CHECK: tail call void @ripple_ew_side_effect_no_return(<32 x float> [[Extract1_quarter_2]], <32 x float> [[Extract2_quarter_2]])
// CHECK: [[Extract1_quarter_3:%.*]] = shufflevector <128 x float> [[Input1Bcast]], <128 x float> poison, <32 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>
// CHECK: [[Extract2_quarter_3:%.*]] = shufflevector <128 x float> [[Input2Bcast]], <128 x float> poison, <32 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>
// CHECK: tail call void @ripple_ew_side_effect_no_return(<32 x float> [[Extract1_quarter_3]], <32 x float> [[Extract2_quarter_3]])
// CHECK: [[Extract1_quarter_4:%.*]] = shufflevector <128 x float> [[Input1Bcast]], <128 x float> poison, <32 x i32> <i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
// CHECK: [[Extract2_quarter_4:%.*]] = shufflevector <128 x float> [[Input2Bcast]], <128 x float> poison, <32 x i32> <i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>
// CHECK: tail call void @ripple_ew_side_effect_no_return(<32 x float> [[Extract1_quarter_4]], <32 x float> [[Extract2_quarter_4]])

void ew_multidim_bcast_void(size_t size, float *input1, float *input2) {
  ripple_block_t BS = ripple_set_block_shape(VEC, 4, 32);
  size_t BlockX = ripple_id(BS, 0);
  size_t BlockY = ripple_id(BS, 1);
  size_t BlockSizeX = ripple_get_block_size(BS, 0);
  size_t BlockSizeY = ripple_get_block_size(BS, 1);
  for (size_t i = 0; i < size; i += BlockSizeX + BlockSizeY)
    side_effect_no_return(input1[i + BlockY], input2[i + BlockX]);
}
