// RUN: %clang_cc1 -triple thumbv7-apple-darwin \
// RUN:   -target-abi apcs-gnu \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -mfloat-abi soft \
// RUN:   -target-feature +soft-float-abi \
// RUN:   -ffreestanding \
// RUN:   -emit-llvm -w -o - %s | opt -S -mem2reg | FileCheck %s

#include <arm_neon.h>

// Check that the vget_low/vget_high intrinsics generate a single shuffle
// without any bitcasting.
int8x8_t low_s8(int8x16_t a) {
// CHECK: shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return vget_low_s8(a);
}

uint8x8_t low_u8 (uint8x16_t a) {
// CHECK: shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return vget_low_u8(a);
}

int16x4_t low_s16( int16x8_t a) {
// CHECK: shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return vget_low_s16(a);
}

uint16x4_t low_u16(uint16x8_t a) {
// CHECK: shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return vget_low_u16(a);
}

int32x2_t low_s32( int32x4_t a) {
// CHECK: shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 0, i32 1>
  return vget_low_s32(a);
}

uint32x2_t low_u32(uint32x4_t a) {
// CHECK: shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 0, i32 1>
  return vget_low_u32(a);
}

int64x1_t low_s64( int64x2_t a) {
// CHECK: shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> zeroinitializer
  return vget_low_s64(a);
}

uint64x1_t low_u64(uint64x2_t a) {
// CHECK: shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> zeroinitializer
  return vget_low_u64(a);
}

poly8x8_t low_p8 (poly8x16_t a) {
// CHECK: shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return vget_low_p8(a);
}

poly16x4_t low_p16(poly16x8_t a) {
// CHECK: shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  return vget_low_p16(a);
}

float32x2_t low_f32(float32x4_t a) {
// CHECK: shufflevector <4 x float> %a, <4 x float> %a, <2 x i32> <i32 0, i32 1>
  return vget_low_f32(a);
}


int8x8_t high_s8(int8x16_t a) {
// CHECK: shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return vget_high_s8(a);
}

uint8x8_t high_u8 (uint8x16_t a) {
// CHECK: shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return vget_high_u8(a);
}

int16x4_t high_s16( int16x8_t a) {
// CHECK: shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return vget_high_s16(a);
}

uint16x4_t high_u16(uint16x8_t a) {
// CHECK: shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return vget_high_u16(a);
}

int32x2_t high_s32( int32x4_t a) {
// CHECK: shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
  return vget_high_s32(a);
}

uint32x2_t high_u32(uint32x4_t a) {
// CHECK: shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
  return vget_high_u32(a);
}

int64x1_t high_s64( int64x2_t a) {
// CHECK: shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> <i32 1>
  return vget_high_s64(a);
}

uint64x1_t high_u64(uint64x2_t a) {
// CHECK: shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> <i32 1>
  return vget_high_u64(a);
}

poly8x8_t high_p8 (poly8x16_t a) {
// CHECK: shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return vget_high_p8(a);
}

poly16x4_t high_p16(poly16x8_t a) {
// CHECK: shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  return vget_high_p16(a);
}

float32x2_t high_f32(float32x4_t a) {
// CHECK: shufflevector <4 x float> %a, <4 x float> %a, <2 x i32> <i32 2, i32 3>
  return vget_high_f32(a);
}

