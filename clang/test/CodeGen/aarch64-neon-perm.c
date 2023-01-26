// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN: -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: @test_vuzp1_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vuzp1_s8(int8x8_t a, int8x8_t b) {
  return vuzp1_s8(a, b);
}

// CHECK-LABEL: @test_vuzp1q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vuzp1q_s8(int8x16_t a, int8x16_t b) {
  return vuzp1q_s8(a, b);
}

// CHECK-LABEL: @test_vuzp1_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vuzp1_s16(int16x4_t a, int16x4_t b) {
  return vuzp1_s16(a, b);
}

// CHECK-LABEL: @test_vuzp1q_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vuzp1q_s16(int16x8_t a, int16x8_t b) {
  return vuzp1q_s16(a, b);
}

// CHECK-LABEL: @test_vuzp1_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vuzp1_s32(int32x2_t a, int32x2_t b) {
  return vuzp1_s32(a, b);
}

// CHECK-LABEL: @test_vuzp1q_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vuzp1q_s32(int32x4_t a, int32x4_t b) {
  return vuzp1q_s32(a, b);
}

// CHECK-LABEL: @test_vuzp1q_s64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vuzp1q_s64(int64x2_t a, int64x2_t b) {
  return vuzp1q_s64(a, b);
}

// CHECK-LABEL: @test_vuzp1_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vuzp1_u8(uint8x8_t a, uint8x8_t b) {
  return vuzp1_u8(a, b);
}

// CHECK-LABEL: @test_vuzp1q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vuzp1q_u8(uint8x16_t a, uint8x16_t b) {
  return vuzp1q_u8(a, b);
}

// CHECK-LABEL: @test_vuzp1_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vuzp1_u16(uint16x4_t a, uint16x4_t b) {
  return vuzp1_u16(a, b);
}

// CHECK-LABEL: @test_vuzp1q_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vuzp1q_u16(uint16x8_t a, uint16x8_t b) {
  return vuzp1q_u16(a, b);
}

// CHECK-LABEL: @test_vuzp1_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vuzp1_u32(uint32x2_t a, uint32x2_t b) {
  return vuzp1_u32(a, b);
}

// CHECK-LABEL: @test_vuzp1q_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vuzp1q_u32(uint32x4_t a, uint32x4_t b) {
  return vuzp1q_u32(a, b);
}

// CHECK-LABEL: @test_vuzp1q_u64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vuzp1q_u64(uint64x2_t a, uint64x2_t b) {
  return vuzp1q_u64(a, b);
}

// CHECK-LABEL: @test_vuzp1_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vuzp1_f32(float32x2_t a, float32x2_t b) {
  return vuzp1_f32(a, b);
}

// CHECK-LABEL: @test_vuzp1q_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vuzp1q_f32(float32x4_t a, float32x4_t b) {
  return vuzp1q_f32(a, b);
}

// CHECK-LABEL: @test_vuzp1q_f64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vuzp1q_f64(float64x2_t a, float64x2_t b) {
  return vuzp1q_f64(a, b);
}

// CHECK-LABEL: @test_vuzp1_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vuzp1_p8(poly8x8_t a, poly8x8_t b) {
  return vuzp1_p8(a, b);
}

// CHECK-LABEL: @test_vuzp1q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vuzp1q_p8(poly8x16_t a, poly8x16_t b) {
  return vuzp1q_p8(a, b);
}

// CHECK-LABEL: @test_vuzp1_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vuzp1_p16(poly16x4_t a, poly16x4_t b) {
  return vuzp1_p16(a, b);
}

// CHECK-LABEL: @test_vuzp1q_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vuzp1q_p16(poly16x8_t a, poly16x8_t b) {
  return vuzp1q_p16(a, b);
}

// CHECK-LABEL: @test_vuzp2_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vuzp2_s8(int8x8_t a, int8x8_t b) {
  return vuzp2_s8(a, b);
}

// CHECK-LABEL: @test_vuzp2q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vuzp2q_s8(int8x16_t a, int8x16_t b) {
  return vuzp2q_s8(a, b);
}

// CHECK-LABEL: @test_vuzp2_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vuzp2_s16(int16x4_t a, int16x4_t b) {
  return vuzp2_s16(a, b);
}

// CHECK-LABEL: @test_vuzp2q_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vuzp2q_s16(int16x8_t a, int16x8_t b) {
  return vuzp2q_s16(a, b);
}

// CHECK-LABEL: @test_vuzp2_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vuzp2_s32(int32x2_t a, int32x2_t b) {
  return vuzp2_s32(a, b);
}

// CHECK-LABEL: @test_vuzp2q_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vuzp2q_s32(int32x4_t a, int32x4_t b) {
  return vuzp2q_s32(a, b);
}

// CHECK-LABEL: @test_vuzp2q_s64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vuzp2q_s64(int64x2_t a, int64x2_t b) {
  return vuzp2q_s64(a, b);
}

// CHECK-LABEL: @test_vuzp2_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vuzp2_u8(uint8x8_t a, uint8x8_t b) {
  return vuzp2_u8(a, b);
}

// CHECK-LABEL: @test_vuzp2q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vuzp2q_u8(uint8x16_t a, uint8x16_t b) {
  return vuzp2q_u8(a, b);
}

// CHECK-LABEL: @test_vuzp2_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vuzp2_u16(uint16x4_t a, uint16x4_t b) {
  return vuzp2_u16(a, b);
}

// CHECK-LABEL: @test_vuzp2q_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vuzp2q_u16(uint16x8_t a, uint16x8_t b) {
  return vuzp2q_u16(a, b);
}

// CHECK-LABEL: @test_vuzp2_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vuzp2_u32(uint32x2_t a, uint32x2_t b) {
  return vuzp2_u32(a, b);
}

// CHECK-LABEL: @test_vuzp2q_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vuzp2q_u32(uint32x4_t a, uint32x4_t b) {
  return vuzp2q_u32(a, b);
}

// CHECK-LABEL: @test_vuzp2q_u64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vuzp2q_u64(uint64x2_t a, uint64x2_t b) {
  return vuzp2q_u64(a, b);
}

// CHECK-LABEL: @test_vuzp2_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vuzp2_f32(float32x2_t a, float32x2_t b) {
  return vuzp2_f32(a, b);
}

// CHECK-LABEL: @test_vuzp2q_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vuzp2q_f32(float32x4_t a, float32x4_t b) {
  return vuzp2q_f32(a, b);
}

// CHECK-LABEL: @test_vuzp2q_f64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vuzp2q_f64(float64x2_t a, float64x2_t b) {
  return vuzp2q_f64(a, b);
}

// CHECK-LABEL: @test_vuzp2_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vuzp2_p8(poly8x8_t a, poly8x8_t b) {
  return vuzp2_p8(a, b);
}

// CHECK-LABEL: @test_vuzp2q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vuzp2q_p8(poly8x16_t a, poly8x16_t b) {
  return vuzp2q_p8(a, b);
}

// CHECK-LABEL: @test_vuzp2_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vuzp2_p16(poly16x4_t a, poly16x4_t b) {
  return vuzp2_p16(a, b);
}

// CHECK-LABEL: @test_vuzp2q_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vuzp2q_p16(poly16x8_t a, poly16x8_t b) {
  return vuzp2q_p16(a, b);
}

// CHECK-LABEL: @test_vzip1_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vzip1_s8(int8x8_t a, int8x8_t b) {
  return vzip1_s8(a, b);
}

// CHECK-LABEL: @test_vzip1q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vzip1q_s8(int8x16_t a, int8x16_t b) {
  return vzip1q_s8(a, b);
}

// CHECK-LABEL: @test_vzip1_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vzip1_s16(int16x4_t a, int16x4_t b) {
  return vzip1_s16(a, b);
}

// CHECK-LABEL: @test_vzip1q_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vzip1q_s16(int16x8_t a, int16x8_t b) {
  return vzip1q_s16(a, b);
}

// CHECK-LABEL: @test_vzip1_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vzip1_s32(int32x2_t a, int32x2_t b) {
  return vzip1_s32(a, b);
}

// CHECK-LABEL: @test_vzip1q_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vzip1q_s32(int32x4_t a, int32x4_t b) {
  return vzip1q_s32(a, b);
}

// CHECK-LABEL: @test_vzip1q_s64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vzip1q_s64(int64x2_t a, int64x2_t b) {
  return vzip1q_s64(a, b);
}

// CHECK-LABEL: @test_vzip1_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vzip1_u8(uint8x8_t a, uint8x8_t b) {
  return vzip1_u8(a, b);
}

// CHECK-LABEL: @test_vzip1q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vzip1q_u8(uint8x16_t a, uint8x16_t b) {
  return vzip1q_u8(a, b);
}

// CHECK-LABEL: @test_vzip1_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vzip1_u16(uint16x4_t a, uint16x4_t b) {
  return vzip1_u16(a, b);
}

// CHECK-LABEL: @test_vzip1q_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vzip1q_u16(uint16x8_t a, uint16x8_t b) {
  return vzip1q_u16(a, b);
}

// CHECK-LABEL: @test_vzip1_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vzip1_u32(uint32x2_t a, uint32x2_t b) {
  return vzip1_u32(a, b);
}

// CHECK-LABEL: @test_vzip1q_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vzip1q_u32(uint32x4_t a, uint32x4_t b) {
  return vzip1q_u32(a, b);
}

// CHECK-LABEL: @test_vzip1q_u64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vzip1q_u64(uint64x2_t a, uint64x2_t b) {
  return vzip1q_u64(a, b);
}

// CHECK-LABEL: @test_vzip1_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vzip1_f32(float32x2_t a, float32x2_t b) {
  return vzip1_f32(a, b);
}

// CHECK-LABEL: @test_vzip1q_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vzip1q_f32(float32x4_t a, float32x4_t b) {
  return vzip1q_f32(a, b);
}

// CHECK-LABEL: @test_vzip1q_f64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vzip1q_f64(float64x2_t a, float64x2_t b) {
  return vzip1q_f64(a, b);
}

// CHECK-LABEL: @test_vzip1_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vzip1_p8(poly8x8_t a, poly8x8_t b) {
  return vzip1_p8(a, b);
}

// CHECK-LABEL: @test_vzip1q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vzip1q_p8(poly8x16_t a, poly8x16_t b) {
  return vzip1q_p8(a, b);
}

// CHECK-LABEL: @test_vzip1_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vzip1_p16(poly16x4_t a, poly16x4_t b) {
  return vzip1_p16(a, b);
}

// CHECK-LABEL: @test_vzip1q_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vzip1q_p16(poly16x8_t a, poly16x8_t b) {
  return vzip1q_p16(a, b);
}

// CHECK-LABEL: @test_vzip2_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vzip2_s8(int8x8_t a, int8x8_t b) {
  return vzip2_s8(a, b);
}

// CHECK-LABEL: @test_vzip2q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vzip2q_s8(int8x16_t a, int8x16_t b) {
  return vzip2q_s8(a, b);
}

// CHECK-LABEL: @test_vzip2_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vzip2_s16(int16x4_t a, int16x4_t b) {
  return vzip2_s16(a, b);
}

// CHECK-LABEL: @test_vzip2q_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vzip2q_s16(int16x8_t a, int16x8_t b) {
  return vzip2q_s16(a, b);
}

// CHECK-LABEL: @test_vzip2_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vzip2_s32(int32x2_t a, int32x2_t b) {
  return vzip2_s32(a, b);
}

// CHECK-LABEL: @test_vzip2q_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vzip2q_s32(int32x4_t a, int32x4_t b) {
  return vzip2q_s32(a, b);
}

// CHECK-LABEL: @test_vzip2q_s64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vzip2q_s64(int64x2_t a, int64x2_t b) {
  return vzip2q_s64(a, b);
}

// CHECK-LABEL: @test_vzip2_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vzip2_u8(uint8x8_t a, uint8x8_t b) {
  return vzip2_u8(a, b);
}

// CHECK-LABEL: @test_vzip2q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vzip2q_u8(uint8x16_t a, uint8x16_t b) {
  return vzip2q_u8(a, b);
}

// CHECK-LABEL: @test_vzip2_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vzip2_u16(uint16x4_t a, uint16x4_t b) {
  return vzip2_u16(a, b);
}

// CHECK-LABEL: @test_vzip2q_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vzip2q_u16(uint16x8_t a, uint16x8_t b) {
  return vzip2q_u16(a, b);
}

// CHECK-LABEL: @test_vzip2_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vzip2_u32(uint32x2_t a, uint32x2_t b) {
  return vzip2_u32(a, b);
}

// CHECK-LABEL: @test_vzip2q_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vzip2q_u32(uint32x4_t a, uint32x4_t b) {
  return vzip2q_u32(a, b);
}

// CHECK-LABEL: @test_vzip2q_u64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vzip2q_u64(uint64x2_t a, uint64x2_t b) {
  return vzip2q_u64(a, b);
}

// CHECK-LABEL: @test_vzip2_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vzip2_f32(float32x2_t a, float32x2_t b) {
  return vzip2_f32(a, b);
}

// CHECK-LABEL: @test_vzip2q_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vzip2q_f32(float32x4_t a, float32x4_t b) {
  return vzip2q_f32(a, b);
}

// CHECK-LABEL: @test_vzip2q_f64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vzip2q_f64(float64x2_t a, float64x2_t b) {
  return vzip2q_f64(a, b);
}

// CHECK-LABEL: @test_vzip2_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vzip2_p8(poly8x8_t a, poly8x8_t b) {
  return vzip2_p8(a, b);
}

// CHECK-LABEL: @test_vzip2q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vzip2q_p8(poly8x16_t a, poly8x16_t b) {
  return vzip2q_p8(a, b);
}

// CHECK-LABEL: @test_vzip2_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vzip2_p16(poly16x4_t a, poly16x4_t b) {
  return vzip2_p16(a, b);
}

// CHECK-LABEL: @test_vzip2q_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vzip2q_p16(poly16x8_t a, poly16x8_t b) {
  return vzip2q_p16(a, b);
}

// CHECK-LABEL: @test_vtrn1_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vtrn1_s8(int8x8_t a, int8x8_t b) {
  return vtrn1_s8(a, b);
}

// CHECK-LABEL: @test_vtrn1q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vtrn1q_s8(int8x16_t a, int8x16_t b) {
  return vtrn1q_s8(a, b);
}

// CHECK-LABEL: @test_vtrn1_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vtrn1_s16(int16x4_t a, int16x4_t b) {
  return vtrn1_s16(a, b);
}

// CHECK-LABEL: @test_vtrn1q_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vtrn1q_s16(int16x8_t a, int16x8_t b) {
  return vtrn1q_s16(a, b);
}

// CHECK-LABEL: @test_vtrn1_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vtrn1_s32(int32x2_t a, int32x2_t b) {
  return vtrn1_s32(a, b);
}

// CHECK-LABEL: @test_vtrn1q_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vtrn1q_s32(int32x4_t a, int32x4_t b) {
  return vtrn1q_s32(a, b);
}

// CHECK-LABEL: @test_vtrn1q_s64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vtrn1q_s64(int64x2_t a, int64x2_t b) {
  return vtrn1q_s64(a, b);
}

// CHECK-LABEL: @test_vtrn1_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vtrn1_u8(uint8x8_t a, uint8x8_t b) {
  return vtrn1_u8(a, b);
}

// CHECK-LABEL: @test_vtrn1q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vtrn1q_u8(uint8x16_t a, uint8x16_t b) {
  return vtrn1q_u8(a, b);
}

// CHECK-LABEL: @test_vtrn1_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vtrn1_u16(uint16x4_t a, uint16x4_t b) {
  return vtrn1_u16(a, b);
}

// CHECK-LABEL: @test_vtrn1q_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vtrn1q_u16(uint16x8_t a, uint16x8_t b) {
  return vtrn1q_u16(a, b);
}

// CHECK-LABEL: @test_vtrn1_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vtrn1_u32(uint32x2_t a, uint32x2_t b) {
  return vtrn1_u32(a, b);
}

// CHECK-LABEL: @test_vtrn1q_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vtrn1q_u32(uint32x4_t a, uint32x4_t b) {
  return vtrn1q_u32(a, b);
}

// CHECK-LABEL: @test_vtrn1q_u64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vtrn1q_u64(uint64x2_t a, uint64x2_t b) {
  return vtrn1q_u64(a, b);
}

// CHECK-LABEL: @test_vtrn1_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vtrn1_f32(float32x2_t a, float32x2_t b) {
  return vtrn1_f32(a, b);
}

// CHECK-LABEL: @test_vtrn1q_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vtrn1q_f32(float32x4_t a, float32x4_t b) {
  return vtrn1q_f32(a, b);
}

// CHECK-LABEL: @test_vtrn1q_f64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vtrn1q_f64(float64x2_t a, float64x2_t b) {
  return vtrn1q_f64(a, b);
}

// CHECK-LABEL: @test_vtrn1_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vtrn1_p8(poly8x8_t a, poly8x8_t b) {
  return vtrn1_p8(a, b);
}

// CHECK-LABEL: @test_vtrn1q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vtrn1q_p8(poly8x16_t a, poly8x16_t b) {
  return vtrn1q_p8(a, b);
}

// CHECK-LABEL: @test_vtrn1_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vtrn1_p16(poly16x4_t a, poly16x4_t b) {
  return vtrn1_p16(a, b);
}

// CHECK-LABEL: @test_vtrn1q_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vtrn1q_p16(poly16x8_t a, poly16x8_t b) {
  return vtrn1q_p16(a, b);
}

// CHECK-LABEL: @test_vtrn2_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vtrn2_s8(int8x8_t a, int8x8_t b) {
  return vtrn2_s8(a, b);
}

// CHECK-LABEL: @test_vtrn2q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vtrn2q_s8(int8x16_t a, int8x16_t b) {
  return vtrn2q_s8(a, b);
}

// CHECK-LABEL: @test_vtrn2_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vtrn2_s16(int16x4_t a, int16x4_t b) {
  return vtrn2_s16(a, b);
}

// CHECK-LABEL: @test_vtrn2q_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vtrn2q_s16(int16x8_t a, int16x8_t b) {
  return vtrn2q_s16(a, b);
}

// CHECK-LABEL: @test_vtrn2_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vtrn2_s32(int32x2_t a, int32x2_t b) {
  return vtrn2_s32(a, b);
}

// CHECK-LABEL: @test_vtrn2q_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vtrn2q_s32(int32x4_t a, int32x4_t b) {
  return vtrn2q_s32(a, b);
}

// CHECK-LABEL: @test_vtrn2q_s64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vtrn2q_s64(int64x2_t a, int64x2_t b) {
  return vtrn2q_s64(a, b);
}

// CHECK-LABEL: @test_vtrn2_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vtrn2_u8(uint8x8_t a, uint8x8_t b) {
  return vtrn2_u8(a, b);
}

// CHECK-LABEL: @test_vtrn2q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vtrn2q_u8(uint8x16_t a, uint8x16_t b) {
  return vtrn2q_u8(a, b);
}

// CHECK-LABEL: @test_vtrn2_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vtrn2_u16(uint16x4_t a, uint16x4_t b) {
  return vtrn2_u16(a, b);
}

// CHECK-LABEL: @test_vtrn2q_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vtrn2q_u16(uint16x8_t a, uint16x8_t b) {
  return vtrn2q_u16(a, b);
}

// CHECK-LABEL: @test_vtrn2_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vtrn2_u32(uint32x2_t a, uint32x2_t b) {
  return vtrn2_u32(a, b);
}

// CHECK-LABEL: @test_vtrn2q_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vtrn2q_u32(uint32x4_t a, uint32x4_t b) {
  return vtrn2q_u32(a, b);
}

// CHECK-LABEL: @test_vtrn2q_u64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vtrn2q_u64(uint64x2_t a, uint64x2_t b) {
  return vtrn2q_u64(a, b);
}

// CHECK-LABEL: @test_vtrn2_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vtrn2_f32(float32x2_t a, float32x2_t b) {
  return vtrn2_f32(a, b);
}

// CHECK-LABEL: @test_vtrn2q_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vtrn2q_f32(float32x4_t a, float32x4_t b) {
  return vtrn2q_f32(a, b);
}

// CHECK-LABEL: @test_vtrn2q_f64(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vtrn2q_f64(float64x2_t a, float64x2_t b) {
  return vtrn2q_f64(a, b);
}

// CHECK-LABEL: @test_vtrn2_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vtrn2_p8(poly8x8_t a, poly8x8_t b) {
  return vtrn2_p8(a, b);
}

// CHECK-LABEL: @test_vtrn2q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vtrn2q_p8(poly8x16_t a, poly8x16_t b) {
  return vtrn2q_p8(a, b);
}

// CHECK-LABEL: @test_vtrn2_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vtrn2_p16(poly16x4_t a, poly16x4_t b) {
  return vtrn2_p16(a, b);
}

// CHECK-LABEL: @test_vtrn2q_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vtrn2q_p16(poly16x8_t a, poly16x8_t b) {
  return vtrn2q_p16(a, b);
}

// CHECK-LABEL: @test_vuzp_s8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i8> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i8> [[VUZP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.int8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x2_t [[TMP8]]
int8x8x2_t test_vuzp_s8(int8x8_t a, int8x8_t b) {
  return vuzp_s8(a, b);
}

// CHECK-LABEL: @test_vuzp_s16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i16> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i16> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.int16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x2_t [[TMP10]]
int16x4x2_t test_vuzp_s16(int16x4_t a, int16x4_t b) {
  return vuzp_s16(a, b);
}

// CHECK-LABEL: @test_vuzp_s32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.int32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x2_t [[TMP10]]
int32x2x2_t test_vuzp_s32(int32x2_t a, int32x2_t b) {
  return vuzp_s32(a, b);
}

// CHECK-LABEL: @test_vuzp_u8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i8> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i8> [[VUZP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x2_t [[TMP8]]
uint8x8x2_t test_vuzp_u8(uint8x8_t a, uint8x8_t b) {
  return vuzp_u8(a, b);
}

// CHECK-LABEL: @test_vuzp_u16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i16> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i16> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.uint16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x2_t [[TMP10]]
uint16x4x2_t test_vuzp_u16(uint16x4_t a, uint16x4_t b) {
  return vuzp_u16(a, b);
}

// CHECK-LABEL: @test_vuzp_u32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.uint32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x2_t [[TMP10]]
uint32x2x2_t test_vuzp_u32(uint32x2_t a, uint32x2_t b) {
  return vuzp_u32(a, b);
}

// CHECK-LABEL: @test_vuzp_f32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x float> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x float>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x float> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.float32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.float32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.float32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x float>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.float32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x2_t [[TMP10]]
float32x2x2_t test_vuzp_f32(float32x2_t a, float32x2_t b) {
  return vuzp_f32(a, b);
}

// CHECK-LABEL: @test_vuzp_p8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i8> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i8> [[VUZP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x2_t [[TMP8]]
poly8x8x2_t test_vuzp_p8(poly8x8_t a, poly8x8_t b) {
  return vuzp_p8(a, b);
}

// CHECK-LABEL: @test_vuzp_p16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i16> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i16> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.poly16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.poly16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.poly16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x2_t [[TMP10]]
poly16x4x2_t test_vuzp_p16(poly16x4_t a, poly16x4_t b) {
  return vuzp_p16(a, b);
}

// CHECK-LABEL: @test_vuzpq_s8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[VUZP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   store <16 x i8> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   store <16 x i8> [[VUZP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.int8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x2_t [[TMP8]]
int8x16x2_t test_vuzpq_s8(int8x16_t a, int8x16_t b) {
  return vuzpq_s8(a, b);
}

// CHECK-LABEL: @test_vuzpq_s16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i16> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i16> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.int16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x2_t [[TMP10]]
int16x8x2_t test_vuzpq_s16(int16x8_t a, int16x8_t b) {
  return vuzpq_s16(a, b);
}

// CHECK-LABEL: @test_vuzpq_s32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i32> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i32> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.int32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x2_t [[TMP10]]
int32x4x2_t test_vuzpq_s32(int32x4_t a, int32x4_t b) {
  return vuzpq_s32(a, b);
}

// CHECK-LABEL: @test_vuzpq_u8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[VUZP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   store <16 x i8> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   store <16 x i8> [[VUZP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x2_t [[TMP8]]
uint8x16x2_t test_vuzpq_u8(uint8x16_t a, uint8x16_t b) {
  return vuzpq_u8(a, b);
}

// CHECK-LABEL: @test_vuzpq_u16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i16> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i16> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.uint16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x2_t [[TMP10]]
uint16x8x2_t test_vuzpq_u16(uint16x8_t a, uint16x8_t b) {
  return vuzpq_u16(a, b);
}

// CHECK-LABEL: @test_vuzpq_u32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i32> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i32> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.uint32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x2_t [[TMP10]]
uint32x4x2_t test_vuzpq_u32(uint32x4_t a, uint32x4_t b) {
  return vuzpq_u32(a, b);
}

// CHECK-LABEL: @test_vuzpq_f32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x float> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x float>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x float> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.float32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.float32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.float32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x float>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.float32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x2_t [[TMP10]]
float32x4x2_t test_vuzpq_f32(float32x4_t a, float32x4_t b) {
  return vuzpq_f32(a, b);
}

// CHECK-LABEL: @test_vuzpq_p8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[VUZP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   store <16 x i8> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   store <16 x i8> [[VUZP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x2_t [[TMP8]]
poly8x16x2_t test_vuzpq_p8(poly8x16_t a, poly8x16_t b) {
  return vuzpq_p8(a, b);
}

// CHECK-LABEL: @test_vuzpq_p16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i16> [[VUZP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i16> [[VUZP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.poly16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.poly16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.poly16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x2_t [[TMP10]]
poly16x8x2_t test_vuzpq_p16(poly16x8_t a, poly16x8_t b) {
  return vuzpq_p16(a, b);
}

// CHECK-LABEL: @test_vzip_s8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i8> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VZIP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.int8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x2_t [[TMP8]]
int8x8x2_t test_vzip_s8(int8x8_t a, int8x8_t b) {
  return vzip_s8(a, b);
}

// CHECK-LABEL: @test_vzip_s16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i16> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.int16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x2_t [[TMP10]]
int16x4x2_t test_vzip_s16(int16x4_t a, int16x4_t b) {
  return vzip_s16(a, b);
}

// CHECK-LABEL: @test_vzip_s32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.int32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x2_t [[TMP10]]
int32x2x2_t test_vzip_s32(int32x2_t a, int32x2_t b) {
  return vzip_s32(a, b);
}

// CHECK-LABEL: @test_vzip_u8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i8> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VZIP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x2_t [[TMP8]]
uint8x8x2_t test_vzip_u8(uint8x8_t a, uint8x8_t b) {
  return vzip_u8(a, b);
}

// CHECK-LABEL: @test_vzip_u16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i16> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.uint16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x2_t [[TMP10]]
uint16x4x2_t test_vzip_u16(uint16x4_t a, uint16x4_t b) {
  return vzip_u16(a, b);
}

// CHECK-LABEL: @test_vzip_u32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.uint32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x2_t [[TMP10]]
uint32x2x2_t test_vzip_u32(uint32x2_t a, uint32x2_t b) {
  return vzip_u32(a, b);
}

// CHECK-LABEL: @test_vzip_f32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x float> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x float>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x float> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.float32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.float32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.float32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x float>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.float32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x2_t [[TMP10]]
float32x2x2_t test_vzip_f32(float32x2_t a, float32x2_t b) {
  return vzip_f32(a, b);
}

// CHECK-LABEL: @test_vzip_p8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i8> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VZIP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x2_t [[TMP8]]
poly8x8x2_t test_vzip_p8(poly8x8_t a, poly8x8_t b) {
  return vzip_p8(a, b);
}

// CHECK-LABEL: @test_vzip_p16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i16> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.poly16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.poly16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.poly16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x2_t [[TMP10]]
poly16x4x2_t test_vzip_p16(poly16x4_t a, poly16x4_t b) {
  return vzip_p16(a, b);
}

// CHECK-LABEL: @test_vzipq_s8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[VZIP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   store <16 x i8> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VZIP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.int8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x2_t [[TMP8]]
int8x16x2_t test_vzipq_s8(int8x16_t a, int8x16_t b) {
  return vzipq_s8(a, b);
}

// CHECK-LABEL: @test_vzipq_s16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i16> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.int16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x2_t [[TMP10]]
int16x8x2_t test_vzipq_s16(int16x8_t a, int16x8_t b) {
  return vzipq_s16(a, b);
}

// CHECK-LABEL: @test_vzipq_s32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i32> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i32> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.int32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x2_t [[TMP10]]
int32x4x2_t test_vzipq_s32(int32x4_t a, int32x4_t b) {
  return vzipq_s32(a, b);
}

// CHECK-LABEL: @test_vzipq_u8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[VZIP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   store <16 x i8> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VZIP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x2_t [[TMP8]]
uint8x16x2_t test_vzipq_u8(uint8x16_t a, uint8x16_t b) {
  return vzipq_u8(a, b);
}

// CHECK-LABEL: @test_vzipq_u16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i16> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.uint16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x2_t [[TMP10]]
uint16x8x2_t test_vzipq_u16(uint16x8_t a, uint16x8_t b) {
  return vzipq_u16(a, b);
}

// CHECK-LABEL: @test_vzipq_u32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i32> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i32> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.uint32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x2_t [[TMP10]]
uint32x4x2_t test_vzipq_u32(uint32x4_t a, uint32x4_t b) {
  return vzipq_u32(a, b);
}

// CHECK-LABEL: @test_vzipq_f32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x float> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x float>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x float> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.float32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.float32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.float32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x float>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.float32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x2_t [[TMP10]]
float32x4x2_t test_vzipq_f32(float32x4_t a, float32x4_t b) {
  return vzipq_f32(a, b);
}

// CHECK-LABEL: @test_vzipq_p8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[VZIP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   store <16 x i8> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VZIP1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x2_t [[TMP8]]
poly8x16x2_t test_vzipq_p8(poly8x16_t a, poly8x16_t b) {
  return vzipq_p8(a, b);
}

// CHECK-LABEL: @test_vzipq_p16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i16> [[VZIP_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VZIP1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.poly16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.poly16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.poly16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x2_t [[TMP10]]
poly16x8x2_t test_vzipq_p16(poly16x8_t a, poly16x8_t b) {
  return vzipq_p16(a, b);
}

// CHECK-LABEL: @test_vtrn_s8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i8> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VTRN1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.int8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x2_t [[TMP8]]
int8x8x2_t test_vtrn_s8(int8x8_t a, int8x8_t b) {
  return vtrn_s8(a, b);
}

// CHECK-LABEL: @test_vtrn_s16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i16> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.int16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x2_t [[TMP10]]
int16x4x2_t test_vtrn_s16(int16x4_t a, int16x4_t b) {
  return vtrn_s16(a, b);
}

// CHECK-LABEL: @test_vtrn_s32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.int32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x2_t [[TMP10]]
int32x2x2_t test_vtrn_s32(int32x2_t a, int32x2_t b) {
  return vtrn_s32(a, b);
}

// CHECK-LABEL: @test_vtrn_u8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i8> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VTRN1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x2_t [[TMP8]]
uint8x8x2_t test_vtrn_u8(uint8x8_t a, uint8x8_t b) {
  return vtrn_u8(a, b);
}

// CHECK-LABEL: @test_vtrn_u16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i16> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.uint16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x2_t [[TMP10]]
uint16x4x2_t test_vtrn_u16(uint16x4_t a, uint16x4_t b) {
  return vtrn_u16(a, b);
}

// CHECK-LABEL: @test_vtrn_u32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.uint32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x2_t [[TMP10]]
uint32x2x2_t test_vtrn_u32(uint32x2_t a, uint32x2_t b) {
  return vtrn_u32(a, b);
}

// CHECK-LABEL: @test_vtrn_f32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x float> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <2 x float>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x float> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.float32x2x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.float32x2x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.float32x2x2_t [[TMP7]], 0
// CHECK:   store [2 x <2 x float>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.float32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x2_t [[TMP10]]
float32x2x2_t test_vtrn_f32(float32x2_t a, float32x2_t b) {
  return vtrn_f32(a, b);
}

// CHECK-LABEL: @test_vtrn_p8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i8> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VTRN1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], ptr [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x2_t [[TMP8]]
poly8x8x2_t test_vtrn_p8(poly8x8_t a, poly8x8_t b) {
  return vtrn_p8(a, b);
}

// CHECK-LABEL: @test_vtrn_p16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i16> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.poly16x4x2_t, ptr [[RETVAL_I]], align 8
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.poly16x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP9]], ptr [[TMP8]], align 8
// CHECK:   [[TMP10:%.*]] = load %struct.poly16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x2_t [[TMP10]]
poly16x4x2_t test_vtrn_p16(poly16x4_t a, poly16x4_t b) {
  return vtrn_p16(a, b);
}

// CHECK-LABEL: @test_vtrnq_s8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[VTRN_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   store <16 x i8> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VTRN1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.int8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x2_t [[TMP8]]
int8x16x2_t test_vtrnq_s8(int8x16_t a, int8x16_t b) {
  return vtrnq_s8(a, b);
}

// CHECK-LABEL: @test_vtrnq_s16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i16> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.int16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x2_t [[TMP10]]
int16x8x2_t test_vtrnq_s16(int16x8_t a, int16x8_t b) {
  return vtrnq_s16(a, b);
}

// CHECK-LABEL: @test_vtrnq_s32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i32> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i32> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.int32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.int32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.int32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.int32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x2_t [[TMP10]]
int32x4x2_t test_vtrnq_s32(int32x4_t a, int32x4_t b) {
  return vtrnq_s32(a, b);
}

// CHECK-LABEL: @test_vtrnq_u8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[VTRN_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   store <16 x i8> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VTRN1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x2_t [[TMP8]]
uint8x16x2_t test_vtrnq_u8(uint8x16_t a, uint8x16_t b) {
  return vtrnq_u8(a, b);
}

// CHECK-LABEL: @test_vtrnq_u16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i16> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.uint16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x2_t [[TMP10]]
uint16x8x2_t test_vtrnq_u16(uint16x8_t a, uint16x8_t b) {
  return vtrnq_u16(a, b);
}

// CHECK-LABEL: @test_vtrnq_u32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i32> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x i32>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i32> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.uint32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.uint32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.uint32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x2_t [[TMP10]]
uint32x4x2_t test_vtrnq_u32(uint32x4_t a, uint32x4_t b) {
  return vtrnq_u32(a, b);
}

// CHECK-LABEL: @test_vtrnq_f32(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x float> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <4 x float>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x float> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.float32x4x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.float32x4x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.float32x4x2_t [[TMP7]], 0
// CHECK:   store [2 x <4 x float>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.float32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x2_t [[TMP10]]
float32x4x2_t test_vtrnq_f32(float32x4_t a, float32x4_t b) {
  return vtrnq_f32(a, b);
}

// CHECK-LABEL: @test_vtrnq_p8(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[VTRN_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   store <16 x i8> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VTRN1_I]], ptr [[TMP2]]
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], ptr [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x2_t [[TMP8]]
poly8x16x2_t test_vtrnq_p8(poly8x16_t a, poly8x16_t b) {
  return vtrnq_p8(a, b);
}

// CHECK-LABEL: @test_vtrnq_p16(
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i16> [[VTRN_I]], ptr [[RETVAL_I]]
// CHECK:   [[TMP4:%.*]] = getelementptr inbounds <8 x i16>, ptr [[RETVAL_I]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VTRN1_I]], ptr [[TMP4]]
// CHECK:   [[TMP7:%.*]] = load %struct.poly16x8x2_t, ptr [[RETVAL_I]], align 16
// CHECK:   [[TMP8:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, ptr [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP9:%.*]] = extractvalue %struct.poly16x8x2_t [[TMP7]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP9]], ptr [[TMP8]], align 16
// CHECK:   [[TMP10:%.*]] = load %struct.poly16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x2_t [[TMP10]]
poly16x8x2_t test_vtrnq_p16(poly16x8_t a, poly16x8_t b) {
  return vtrnq_p16(a, b);
}
