// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// Test new aarch64 intrinsics and types
#include <arm_neon.h>

// CHECK-LABEL: define <8 x i8> @test_vuzp1_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vuzp1_s8(int8x8_t a, int8x8_t b) {
  return vuzp1_s8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vuzp1q_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vuzp1q_s8(int8x16_t a, int8x16_t b) {
  return vuzp1q_s8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vuzp1_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vuzp1_s16(int16x4_t a, int16x4_t b) {
  return vuzp1_s16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vuzp1q_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vuzp1q_s16(int16x8_t a, int16x8_t b) {
  return vuzp1q_s16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vuzp1_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vuzp1_s32(int32x2_t a, int32x2_t b) {
  return vuzp1_s32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vuzp1q_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vuzp1q_s32(int32x4_t a, int32x4_t b) {
  return vuzp1q_s32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vuzp1q_s64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vuzp1q_s64(int64x2_t a, int64x2_t b) {
  return vuzp1q_s64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vuzp1_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vuzp1_u8(uint8x8_t a, uint8x8_t b) {
  return vuzp1_u8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vuzp1q_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vuzp1q_u8(uint8x16_t a, uint8x16_t b) {
  return vuzp1q_u8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vuzp1_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vuzp1_u16(uint16x4_t a, uint16x4_t b) {
  return vuzp1_u16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vuzp1q_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vuzp1q_u16(uint16x8_t a, uint16x8_t b) {
  return vuzp1q_u16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vuzp1_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vuzp1_u32(uint32x2_t a, uint32x2_t b) {
  return vuzp1_u32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vuzp1q_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vuzp1q_u32(uint32x4_t a, uint32x4_t b) {
  return vuzp1q_u32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vuzp1q_u64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vuzp1q_u64(uint64x2_t a, uint64x2_t b) {
  return vuzp1q_u64(a, b);
}

// CHECK-LABEL: define <2 x float> @test_vuzp1_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vuzp1_f32(float32x2_t a, float32x2_t b) {
  return vuzp1_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vuzp1q_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vuzp1q_f32(float32x4_t a, float32x4_t b) {
  return vuzp1q_f32(a, b);
}

// CHECK-LABEL: define <2 x double> @test_vuzp1q_f64(<2 x double> %a, <2 x double> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vuzp1q_f64(float64x2_t a, float64x2_t b) {
  return vuzp1q_f64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vuzp1_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vuzp1_p8(poly8x8_t a, poly8x8_t b) {
  return vuzp1_p8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vuzp1q_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vuzp1q_p8(poly8x16_t a, poly8x16_t b) {
  return vuzp1q_p8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vuzp1_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vuzp1_p16(poly16x4_t a, poly16x4_t b) {
  return vuzp1_p16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vuzp1q_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vuzp1q_p16(poly16x8_t a, poly16x8_t b) {
  return vuzp1q_p16(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vuzp2_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vuzp2_s8(int8x8_t a, int8x8_t b) {
  return vuzp2_s8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vuzp2q_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vuzp2q_s8(int8x16_t a, int8x16_t b) {
  return vuzp2q_s8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vuzp2_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vuzp2_s16(int16x4_t a, int16x4_t b) {
  return vuzp2_s16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vuzp2q_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vuzp2q_s16(int16x8_t a, int16x8_t b) {
  return vuzp2q_s16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vuzp2_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vuzp2_s32(int32x2_t a, int32x2_t b) {
  return vuzp2_s32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vuzp2q_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vuzp2q_s32(int32x4_t a, int32x4_t b) {
  return vuzp2q_s32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vuzp2q_s64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vuzp2q_s64(int64x2_t a, int64x2_t b) {
  return vuzp2q_s64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vuzp2_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vuzp2_u8(uint8x8_t a, uint8x8_t b) {
  return vuzp2_u8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vuzp2q_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vuzp2q_u8(uint8x16_t a, uint8x16_t b) {
  return vuzp2q_u8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vuzp2_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vuzp2_u16(uint16x4_t a, uint16x4_t b) {
  return vuzp2_u16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vuzp2q_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vuzp2q_u16(uint16x8_t a, uint16x8_t b) {
  return vuzp2q_u16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vuzp2_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vuzp2_u32(uint32x2_t a, uint32x2_t b) {
  return vuzp2_u32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vuzp2q_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vuzp2q_u32(uint32x4_t a, uint32x4_t b) {
  return vuzp2q_u32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vuzp2q_u64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vuzp2q_u64(uint64x2_t a, uint64x2_t b) {
  return vuzp2q_u64(a, b);
}

// CHECK-LABEL: define <2 x float> @test_vuzp2_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vuzp2_f32(float32x2_t a, float32x2_t b) {
  return vuzp2_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vuzp2q_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vuzp2q_f32(float32x4_t a, float32x4_t b) {
  return vuzp2q_f32(a, b);
}

// CHECK-LABEL: define <2 x double> @test_vuzp2q_f64(<2 x double> %a, <2 x double> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vuzp2q_f64(float64x2_t a, float64x2_t b) {
  return vuzp2q_f64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vuzp2_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vuzp2_p8(poly8x8_t a, poly8x8_t b) {
  return vuzp2_p8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vuzp2q_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vuzp2q_p8(poly8x16_t a, poly8x16_t b) {
  return vuzp2q_p8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vuzp2_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vuzp2_p16(poly16x4_t a, poly16x4_t b) {
  return vuzp2_p16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vuzp2q_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vuzp2q_p16(poly16x8_t a, poly16x8_t b) {
  return vuzp2q_p16(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vzip1_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vzip1_s8(int8x8_t a, int8x8_t b) {
  return vzip1_s8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vzip1q_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vzip1q_s8(int8x16_t a, int8x16_t b) {
  return vzip1q_s8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vzip1_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vzip1_s16(int16x4_t a, int16x4_t b) {
  return vzip1_s16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vzip1q_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vzip1q_s16(int16x8_t a, int16x8_t b) {
  return vzip1q_s16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vzip1_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vzip1_s32(int32x2_t a, int32x2_t b) {
  return vzip1_s32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vzip1q_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vzip1q_s32(int32x4_t a, int32x4_t b) {
  return vzip1q_s32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vzip1q_s64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vzip1q_s64(int64x2_t a, int64x2_t b) {
  return vzip1q_s64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vzip1_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vzip1_u8(uint8x8_t a, uint8x8_t b) {
  return vzip1_u8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vzip1q_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vzip1q_u8(uint8x16_t a, uint8x16_t b) {
  return vzip1q_u8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vzip1_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vzip1_u16(uint16x4_t a, uint16x4_t b) {
  return vzip1_u16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vzip1q_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vzip1q_u16(uint16x8_t a, uint16x8_t b) {
  return vzip1q_u16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vzip1_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vzip1_u32(uint32x2_t a, uint32x2_t b) {
  return vzip1_u32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vzip1q_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vzip1q_u32(uint32x4_t a, uint32x4_t b) {
  return vzip1q_u32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vzip1q_u64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vzip1q_u64(uint64x2_t a, uint64x2_t b) {
  return vzip1q_u64(a, b);
}

// CHECK-LABEL: define <2 x float> @test_vzip1_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vzip1_f32(float32x2_t a, float32x2_t b) {
  return vzip1_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vzip1q_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vzip1q_f32(float32x4_t a, float32x4_t b) {
  return vzip1q_f32(a, b);
}

// CHECK-LABEL: define <2 x double> @test_vzip1q_f64(<2 x double> %a, <2 x double> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vzip1q_f64(float64x2_t a, float64x2_t b) {
  return vzip1q_f64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vzip1_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vzip1_p8(poly8x8_t a, poly8x8_t b) {
  return vzip1_p8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vzip1q_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vzip1q_p8(poly8x16_t a, poly8x16_t b) {
  return vzip1q_p8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vzip1_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vzip1_p16(poly16x4_t a, poly16x4_t b) {
  return vzip1_p16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vzip1q_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vzip1q_p16(poly16x8_t a, poly16x8_t b) {
  return vzip1q_p16(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vzip2_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vzip2_s8(int8x8_t a, int8x8_t b) {
  return vzip2_s8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vzip2q_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vzip2q_s8(int8x16_t a, int8x16_t b) {
  return vzip2q_s8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vzip2_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vzip2_s16(int16x4_t a, int16x4_t b) {
  return vzip2_s16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vzip2q_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vzip2q_s16(int16x8_t a, int16x8_t b) {
  return vzip2q_s16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vzip2_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vzip2_s32(int32x2_t a, int32x2_t b) {
  return vzip2_s32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vzip2q_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vzip2q_s32(int32x4_t a, int32x4_t b) {
  return vzip2q_s32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vzip2q_s64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vzip2q_s64(int64x2_t a, int64x2_t b) {
  return vzip2q_s64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vzip2_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vzip2_u8(uint8x8_t a, uint8x8_t b) {
  return vzip2_u8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vzip2q_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vzip2q_u8(uint8x16_t a, uint8x16_t b) {
  return vzip2q_u8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vzip2_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vzip2_u16(uint16x4_t a, uint16x4_t b) {
  return vzip2_u16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vzip2q_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vzip2q_u16(uint16x8_t a, uint16x8_t b) {
  return vzip2q_u16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vzip2_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vzip2_u32(uint32x2_t a, uint32x2_t b) {
  return vzip2_u32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vzip2q_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vzip2q_u32(uint32x4_t a, uint32x4_t b) {
  return vzip2q_u32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vzip2q_u64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vzip2q_u64(uint64x2_t a, uint64x2_t b) {
  return vzip2q_u64(a, b);
}

// CHECK-LABEL: define <2 x float> @test_vzip2_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vzip2_f32(float32x2_t a, float32x2_t b) {
  return vzip2_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vzip2q_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vzip2q_f32(float32x4_t a, float32x4_t b) {
  return vzip2q_f32(a, b);
}

// CHECK-LABEL: define <2 x double> @test_vzip2q_f64(<2 x double> %a, <2 x double> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vzip2q_f64(float64x2_t a, float64x2_t b) {
  return vzip2q_f64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vzip2_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vzip2_p8(poly8x8_t a, poly8x8_t b) {
  return vzip2_p8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vzip2q_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vzip2q_p8(poly8x16_t a, poly8x16_t b) {
  return vzip2q_p8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vzip2_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vzip2_p16(poly16x4_t a, poly16x4_t b) {
  return vzip2_p16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vzip2q_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vzip2q_p16(poly16x8_t a, poly16x8_t b) {
  return vzip2q_p16(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vtrn1_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vtrn1_s8(int8x8_t a, int8x8_t b) {
  return vtrn1_s8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vtrn1q_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vtrn1q_s8(int8x16_t a, int8x16_t b) {
  return vtrn1q_s8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vtrn1_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vtrn1_s16(int16x4_t a, int16x4_t b) {
  return vtrn1_s16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vtrn1q_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vtrn1q_s16(int16x8_t a, int16x8_t b) {
  return vtrn1q_s16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vtrn1_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vtrn1_s32(int32x2_t a, int32x2_t b) {
  return vtrn1_s32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vtrn1q_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vtrn1q_s32(int32x4_t a, int32x4_t b) {
  return vtrn1q_s32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vtrn1q_s64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vtrn1q_s64(int64x2_t a, int64x2_t b) {
  return vtrn1q_s64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vtrn1_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vtrn1_u8(uint8x8_t a, uint8x8_t b) {
  return vtrn1_u8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vtrn1q_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vtrn1q_u8(uint8x16_t a, uint8x16_t b) {
  return vtrn1q_u8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vtrn1_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vtrn1_u16(uint16x4_t a, uint16x4_t b) {
  return vtrn1_u16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vtrn1q_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vtrn1q_u16(uint16x8_t a, uint16x8_t b) {
  return vtrn1q_u16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vtrn1_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vtrn1_u32(uint32x2_t a, uint32x2_t b) {
  return vtrn1_u32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vtrn1q_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vtrn1q_u32(uint32x4_t a, uint32x4_t b) {
  return vtrn1q_u32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vtrn1q_u64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vtrn1q_u64(uint64x2_t a, uint64x2_t b) {
  return vtrn1q_u64(a, b);
}

// CHECK-LABEL: define <2 x float> @test_vtrn1_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vtrn1_f32(float32x2_t a, float32x2_t b) {
  return vtrn1_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vtrn1q_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vtrn1q_f32(float32x4_t a, float32x4_t b) {
  return vtrn1q_f32(a, b);
}

// CHECK-LABEL: define <2 x double> @test_vtrn1q_f64(<2 x double> %a, <2 x double> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vtrn1q_f64(float64x2_t a, float64x2_t b) {
  return vtrn1q_f64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vtrn1_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vtrn1_p8(poly8x8_t a, poly8x8_t b) {
  return vtrn1_p8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vtrn1q_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vtrn1q_p8(poly8x16_t a, poly8x16_t b) {
  return vtrn1q_p8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vtrn1_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vtrn1_p16(poly16x4_t a, poly16x4_t b) {
  return vtrn1_p16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vtrn1q_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vtrn1q_p16(poly16x8_t a, poly16x8_t b) {
  return vtrn1q_p16(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vtrn2_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vtrn2_s8(int8x8_t a, int8x8_t b) {
  return vtrn2_s8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vtrn2q_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vtrn2q_s8(int8x16_t a, int8x16_t b) {
  return vtrn2q_s8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vtrn2_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vtrn2_s16(int16x4_t a, int16x4_t b) {
  return vtrn2_s16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vtrn2q_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vtrn2q_s16(int16x8_t a, int16x8_t b) {
  return vtrn2q_s16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vtrn2_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vtrn2_s32(int32x2_t a, int32x2_t b) {
  return vtrn2_s32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vtrn2q_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vtrn2q_s32(int32x4_t a, int32x4_t b) {
  return vtrn2q_s32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vtrn2q_s64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vtrn2q_s64(int64x2_t a, int64x2_t b) {
  return vtrn2q_s64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vtrn2_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vtrn2_u8(uint8x8_t a, uint8x8_t b) {
  return vtrn2_u8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vtrn2q_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vtrn2q_u8(uint8x16_t a, uint8x16_t b) {
  return vtrn2q_u8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vtrn2_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vtrn2_u16(uint16x4_t a, uint16x4_t b) {
  return vtrn2_u16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vtrn2q_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vtrn2q_u16(uint16x8_t a, uint16x8_t b) {
  return vtrn2q_u16(a, b);
}

// CHECK-LABEL: define <2 x i32> @test_vtrn2_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vtrn2_u32(uint32x2_t a, uint32x2_t b) {
  return vtrn2_u32(a, b);
}

// CHECK-LABEL: define <4 x i32> @test_vtrn2q_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vtrn2q_u32(uint32x4_t a, uint32x4_t b) {
  return vtrn2q_u32(a, b);
}

// CHECK-LABEL: define <2 x i64> @test_vtrn2q_u64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vtrn2q_u64(uint64x2_t a, uint64x2_t b) {
  return vtrn2q_u64(a, b);
}

// CHECK-LABEL: define <2 x float> @test_vtrn2_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vtrn2_f32(float32x2_t a, float32x2_t b) {
  return vtrn2_f32(a, b);
}

// CHECK-LABEL: define <4 x float> @test_vtrn2q_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vtrn2q_f32(float32x4_t a, float32x4_t b) {
  return vtrn2q_f32(a, b);
}

// CHECK-LABEL: define <2 x double> @test_vtrn2q_f64(<2 x double> %a, <2 x double> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vtrn2q_f64(float64x2_t a, float64x2_t b) {
  return vtrn2q_f64(a, b);
}

// CHECK-LABEL: define <8 x i8> @test_vtrn2_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vtrn2_p8(poly8x8_t a, poly8x8_t b) {
  return vtrn2_p8(a, b);
}

// CHECK-LABEL: define <16 x i8> @test_vtrn2q_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vtrn2q_p8(poly8x16_t a, poly8x16_t b) {
  return vtrn2q_p8(a, b);
}

// CHECK-LABEL: define <4 x i16> @test_vtrn2_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vtrn2_p16(poly16x4_t a, poly16x4_t b) {
  return vtrn2_p16(a, b);
}

// CHECK-LABEL: define <8 x i16> @test_vtrn2q_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vtrn2q_p16(poly16x8_t a, poly16x8_t b) {
  return vtrn2q_p16(a, b);
}

// CHECK-LABEL: define %struct.int8x8x2_t @test_vuzp_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i8> [[VUZP_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i8> [[VUZP1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.int8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x2_t [[TMP8]]
int8x8x2_t test_vuzp_s8(int8x8_t a, int8x8_t b) {
  return vuzp_s8(a, b);
}

// CHECK-LABEL: define %struct.int16x4x2_t @test_vuzp_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i16> [[VUZP_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i16> [[VUZP1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x2_t [[TMP12]]
int16x4x2_t test_vuzp_s16(int16x4_t a, int16x4_t b) {
  return vuzp_s16(a, b);
}
// CHECK-LABEL: define %struct.int32x2x2_t @test_vuzp_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VUZP_I]], <2 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x i32>, <2 x i32>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VUZP1_I]], <2 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP11]], [2 x <2 x i32>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x2_t [[TMP12]]
int32x2x2_t test_vuzp_s32(int32x2_t a, int32x2_t b) {
  return vuzp_s32(a, b);
}
// CHECK-LABEL: define %struct.uint8x8x2_t @test_vuzp_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i8> [[VUZP_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i8> [[VUZP1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.uint8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x2_t [[TMP8]]
uint8x8x2_t test_vuzp_u8(uint8x8_t a, uint8x8_t b) {
  return vuzp_u8(a, b);
}
// CHECK-LABEL: define %struct.uint16x4x2_t @test_vuzp_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i16> [[VUZP_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i16> [[VUZP1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x2_t [[TMP12]]
uint16x4x2_t test_vuzp_u16(uint16x4_t a, uint16x4_t b) {
  return vuzp_u16(a, b);
}
// CHECK-LABEL: define %struct.uint32x2x2_t @test_vuzp_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VUZP_I]], <2 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x i32>, <2 x i32>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VUZP1_I]], <2 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP11]], [2 x <2 x i32>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x2_t [[TMP12]]
uint32x2x2_t test_vuzp_u32(uint32x2_t a, uint32x2_t b) {
  return vuzp_u32(a, b);
}
// CHECK-LABEL: define %struct.float32x2x2_t @test_vuzp_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x float>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <2 x float> [[TMP4]], <2 x float> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x float> [[VUZP_I]], <2 x float>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x float>, <2 x float>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <2 x float> [[TMP4]], <2 x float> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x float> [[VUZP1_I]], <2 x float>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.float32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.float32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.float32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x float>] [[TMP11]], [2 x <2 x float>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x2_t [[TMP12]]
float32x2x2_t test_vuzp_f32(float32x2_t a, float32x2_t b) {
  return vuzp_f32(a, b);
}
// CHECK-LABEL: define %struct.poly8x8x2_t @test_vuzp_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i8> [[VUZP_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i8> [[VUZP1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.poly8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x2_t [[TMP8]]
poly8x8x2_t test_vuzp_p8(poly8x8_t a, poly8x8_t b) {
  return vuzp_p8(a, b);
}
// CHECK-LABEL: define %struct.poly16x4x2_t @test_vuzp_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i16> [[VUZP_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i16> [[VUZP1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.poly16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x2_t [[TMP12]]
poly16x4x2_t test_vuzp_p16(poly16x4_t a, poly16x4_t b) {
  return vuzp_p16(a, b);
}
// CHECK-LABEL: define %struct.int8x16x2_t @test_vuzpq_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VUZP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   store <16 x i8> [[VUZP_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   store <16 x i8> [[VUZP1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.int8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x2_t [[TMP8]]
int8x16x2_t test_vuzpq_s8(int8x16_t a, int8x16_t b) {
  return vuzpq_s8(a, b);
}
// CHECK-LABEL: define %struct.int16x8x2_t @test_vuzpq_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i16> [[VUZP_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i16> [[VUZP1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x2_t [[TMP12]]
int16x8x2_t test_vuzpq_s16(int16x8_t a, int16x8_t b) {
  return vuzpq_s16(a, b);
}
// CHECK-LABEL: define %struct.int32x4x2_t @test_vuzpq_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i32> [[VUZP_I]], <4 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i32>, <4 x i32>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i32> [[VUZP1_I]], <4 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP11]], [2 x <4 x i32>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x2_t [[TMP12]]
int32x4x2_t test_vuzpq_s32(int32x4_t a, int32x4_t b) {
  return vuzpq_s32(a, b);
}
// CHECK-LABEL: define %struct.uint8x16x2_t @test_vuzpq_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VUZP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   store <16 x i8> [[VUZP_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   store <16 x i8> [[VUZP1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.uint8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x2_t [[TMP8]]
uint8x16x2_t test_vuzpq_u8(uint8x16_t a, uint8x16_t b) {
  return vuzpq_u8(a, b);
}
// CHECK-LABEL: define %struct.uint16x8x2_t @test_vuzpq_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i16> [[VUZP_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i16> [[VUZP1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x2_t [[TMP12]]
uint16x8x2_t test_vuzpq_u16(uint16x8_t a, uint16x8_t b) {
  return vuzpq_u16(a, b);
}
// CHECK-LABEL: define %struct.uint32x4x2_t @test_vuzpq_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x i32> [[VUZP_I]], <4 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i32>, <4 x i32>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x i32> [[VUZP1_I]], <4 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP11]], [2 x <4 x i32>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x2_t [[TMP12]]
uint32x4x2_t test_vuzpq_u32(uint32x4_t a, uint32x4_t b) {
  return vuzpq_u32(a, b);
}
// CHECK-LABEL: define %struct.float32x4x2_t @test_vuzpq_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x float>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <4 x float> [[TMP4]], <4 x float> [[TMP5]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x float> [[VUZP_I]], <4 x float>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x float>, <4 x float>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <4 x float> [[TMP4]], <4 x float> [[TMP5]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x float> [[VUZP1_I]], <4 x float>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.float32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.float32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.float32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x float>] [[TMP11]], [2 x <4 x float>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x2_t [[TMP12]]
float32x4x2_t test_vuzpq_f32(float32x4_t a, float32x4_t b) {
  return vuzpq_f32(a, b);
}
// CHECK-LABEL: define %struct.poly8x16x2_t @test_vuzpq_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VUZP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// CHECK:   store <16 x i8> [[VUZP_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
// CHECK:   store <16 x i8> [[VUZP1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.poly8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x2_t [[TMP8]]
poly8x16x2_t test_vuzpq_p8(poly8x16_t a, poly8x16_t b) {
  return vuzpq_p8(a, b);
}
// CHECK-LABEL: define %struct.poly16x8x2_t @test_vuzpq_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VUZP_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x i16> [[VUZP_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VUZP1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x i16> [[VUZP1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.poly16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x2_t [[TMP12]]
poly16x8x2_t test_vuzpq_p16(poly16x8_t a, poly16x8_t b) {
  return vuzpq_p16(a, b);
}

// CHECK-LABEL: define %struct.int8x8x2_t @test_vzip_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i8> [[VZIP_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VZIP1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.int8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x2_t [[TMP8]]
int8x8x2_t test_vzip_s8(int8x8_t a, int8x8_t b) {
  return vzip_s8(a, b);
}

// CHECK-LABEL: define %struct.int16x4x2_t @test_vzip_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i16> [[VZIP_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VZIP1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x2_t [[TMP12]]
int16x4x2_t test_vzip_s16(int16x4_t a, int16x4_t b) {
  return vzip_s16(a, b);
}
// CHECK-LABEL: define %struct.int32x2x2_t @test_vzip_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VZIP_I]], <2 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x i32>, <2 x i32>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VZIP1_I]], <2 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP11]], [2 x <2 x i32>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x2_t [[TMP12]]
int32x2x2_t test_vzip_s32(int32x2_t a, int32x2_t b) {
  return vzip_s32(a, b);
}
// CHECK-LABEL: define %struct.uint8x8x2_t @test_vzip_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i8> [[VZIP_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VZIP1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.uint8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x2_t [[TMP8]]
uint8x8x2_t test_vzip_u8(uint8x8_t a, uint8x8_t b) {
  return vzip_u8(a, b);
}
// CHECK-LABEL: define %struct.uint16x4x2_t @test_vzip_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i16> [[VZIP_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VZIP1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x2_t [[TMP12]]
uint16x4x2_t test_vzip_u16(uint16x4_t a, uint16x4_t b) {
  return vzip_u16(a, b);
}
// CHECK-LABEL: define %struct.uint32x2x2_t @test_vzip_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VZIP_I]], <2 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x i32>, <2 x i32>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VZIP1_I]], <2 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP11]], [2 x <2 x i32>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x2_t [[TMP12]]
uint32x2x2_t test_vzip_u32(uint32x2_t a, uint32x2_t b) {
  return vzip_u32(a, b);
}
// CHECK-LABEL: define %struct.float32x2x2_t @test_vzip_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x float>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <2 x float> [[TMP4]], <2 x float> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x float> [[VZIP_I]], <2 x float>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x float>, <2 x float>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <2 x float> [[TMP4]], <2 x float> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x float> [[VZIP1_I]], <2 x float>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.float32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.float32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.float32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x float>] [[TMP11]], [2 x <2 x float>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x2_t [[TMP12]]
float32x2x2_t test_vzip_f32(float32x2_t a, float32x2_t b) {
  return vzip_f32(a, b);
}
// CHECK-LABEL: define %struct.poly8x8x2_t @test_vzip_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i8> [[VZIP_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VZIP1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.poly8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x2_t [[TMP8]]
poly8x8x2_t test_vzip_p8(poly8x8_t a, poly8x8_t b) {
  return vzip_p8(a, b);
}
// CHECK-LABEL: define %struct.poly16x4x2_t @test_vzip_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i16> [[VZIP_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VZIP1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.poly16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x2_t [[TMP12]]
poly16x4x2_t test_vzip_p16(poly16x4_t a, poly16x4_t b) {
  return vzip_p16(a, b);
}
// CHECK-LABEL: define %struct.int8x16x2_t @test_vzipq_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VZIP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   store <16 x i8> [[VZIP_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VZIP1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.int8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x2_t [[TMP8]]
int8x16x2_t test_vzipq_s8(int8x16_t a, int8x16_t b) {
  return vzipq_s8(a, b);
}
// CHECK-LABEL: define %struct.int16x8x2_t @test_vzipq_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i16> [[VZIP_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VZIP1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x2_t [[TMP12]]
int16x8x2_t test_vzipq_s16(int16x8_t a, int16x8_t b) {
  return vzipq_s16(a, b);
}
// CHECK-LABEL: define %struct.int32x4x2_t @test_vzipq_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i32> [[VZIP_I]], <4 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i32>, <4 x i32>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i32> [[VZIP1_I]], <4 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP11]], [2 x <4 x i32>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x2_t [[TMP12]]
int32x4x2_t test_vzipq_s32(int32x4_t a, int32x4_t b) {
  return vzipq_s32(a, b);
}
// CHECK-LABEL: define %struct.uint8x16x2_t @test_vzipq_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VZIP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   store <16 x i8> [[VZIP_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VZIP1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.uint8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x2_t [[TMP8]]
uint8x16x2_t test_vzipq_u8(uint8x16_t a, uint8x16_t b) {
  return vzipq_u8(a, b);
}
// CHECK-LABEL: define %struct.uint16x8x2_t @test_vzipq_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i16> [[VZIP_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VZIP1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x2_t [[TMP12]]
uint16x8x2_t test_vzipq_u16(uint16x8_t a, uint16x8_t b) {
  return vzipq_u16(a, b);
}
// CHECK-LABEL: define %struct.uint32x4x2_t @test_vzipq_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x i32> [[VZIP_I]], <4 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i32>, <4 x i32>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x i32> [[VZIP1_I]], <4 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP11]], [2 x <4 x i32>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x2_t [[TMP12]]
uint32x4x2_t test_vzipq_u32(uint32x4_t a, uint32x4_t b) {
  return vzipq_u32(a, b);
}
// CHECK-LABEL: define %struct.float32x4x2_t @test_vzipq_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x float>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <4 x float> [[TMP4]], <4 x float> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x float> [[VZIP_I]], <4 x float>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x float>, <4 x float>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x float> [[TMP4]], <4 x float> [[TMP5]], <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x float> [[VZIP1_I]], <4 x float>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.float32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.float32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.float32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x float>] [[TMP11]], [2 x <4 x float>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x2_t [[TMP12]]
float32x4x2_t test_vzipq_f32(float32x4_t a, float32x4_t b) {
  return vzipq_f32(a, b);
}
// CHECK-LABEL: define %struct.poly8x16x2_t @test_vzipq_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VZIP_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
// CHECK:   store <16 x i8> [[VZIP_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VZIP1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.poly8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x2_t [[TMP8]]
poly8x16x2_t test_vzipq_p8(poly8x16_t a, poly8x16_t b) {
  return vzipq_p8(a, b);
}
// CHECK-LABEL: define %struct.poly16x8x2_t @test_vzipq_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VZIP_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x i16> [[VZIP_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VZIP1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.poly16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x2_t [[TMP12]]
poly16x8x2_t test_vzipq_p16(poly16x8_t a, poly16x8_t b) {
  return vzipq_p16(a, b);
}

// CHECK-LABEL: define %struct.int8x8x2_t @test_vtrn_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i8> [[VTRN_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VTRN1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.int8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.int8x8x2_t, %struct.int8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x2_t [[TMP8]]
int8x8x2_t test_vtrn_s8(int8x8_t a, int8x8_t b) {
  return vtrn_s8(a, b);
}

// CHECK-LABEL: define %struct.int16x4x2_t @test_vtrn_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i16> [[VTRN_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VTRN1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.int16x4x2_t, %struct.int16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x2_t [[TMP12]]
int16x4x2_t test_vtrn_s16(int16x4_t a, int16x4_t b) {
  return vtrn_s16(a, b);
}
// CHECK-LABEL: define %struct.int32x2x2_t @test_vtrn_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VTRN_I]], <2 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x i32>, <2 x i32>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VTRN1_I]], <2 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP11]], [2 x <2 x i32>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.int32x2x2_t, %struct.int32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x2_t [[TMP12]]
int32x2x2_t test_vtrn_s32(int32x2_t a, int32x2_t b) {
  return vtrn_s32(a, b);
}
// CHECK-LABEL: define %struct.uint8x8x2_t @test_vtrn_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i8> [[VTRN_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VTRN1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.uint8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x2_t [[TMP8]]
uint8x8x2_t test_vtrn_u8(uint8x8_t a, uint8x8_t b) {
  return vtrn_u8(a, b);
}
// CHECK-LABEL: define %struct.uint16x4x2_t @test_vtrn_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i16> [[VTRN_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VTRN1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.uint16x4x2_t, %struct.uint16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x2_t [[TMP12]]
uint16x4x2_t test_vtrn_u16(uint16x4_t a, uint16x4_t b) {
  return vtrn_u16(a, b);
}
// CHECK-LABEL: define %struct.uint32x2x2_t @test_vtrn_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x i32> [[VTRN_I]], <2 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x i32>, <2 x i32>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x i32> [[VTRN1_I]], <2 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x i32>] [[TMP11]], [2 x <2 x i32>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.uint32x2x2_t, %struct.uint32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x2_t [[TMP12]]
uint32x2x2_t test_vtrn_u32(uint32x2_t a, uint32x2_t b) {
  return vtrn_u32(a, b);
}
// CHECK-LABEL: define %struct.float32x2x2_t @test_vtrn_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x2x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <2 x float>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <2 x float> [[TMP4]], <2 x float> [[TMP5]], <2 x i32> <i32 0, i32 2>
// CHECK:   store <2 x float> [[VTRN_I]], <2 x float>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <2 x float>, <2 x float>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <2 x float> [[TMP4]], <2 x float> [[TMP5]], <2 x i32> <i32 1, i32 3>
// CHECK:   store <2 x float> [[VTRN1_I]], <2 x float>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.float32x2x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.float32x2x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.float32x2x2_t [[TMP9]], 0
// CHECK:   store [2 x <2 x float>] [[TMP11]], [2 x <2 x float>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.float32x2x2_t, %struct.float32x2x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x2_t [[TMP12]]
float32x2x2_t test_vtrn_f32(float32x2_t a, float32x2_t b) {
  return vtrn_f32(a, b);
}
// CHECK-LABEL: define %struct.poly8x8x2_t @test_vtrn_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <8 x i8>*
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i8> [[VTRN_I]], <8 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x i8>, <8 x i8>* [[TMP1]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i8> [[VTRN1_I]], <8 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.poly8x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly8x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x8x2_t [[TMP5]], 0
// CHECK:   store [2 x <8 x i8>] [[TMP7]], [2 x <8 x i8>]* [[TMP6]], align 8
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x2_t [[TMP8]]
poly8x8x2_t test_vtrn_p8(poly8x8_t a, poly8x8_t b) {
  return vtrn_p8(a, b);
}
// CHECK-LABEL: define %struct.poly16x4x2_t @test_vtrn_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i16> [[VTRN_I]], <4 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i16>, <4 x i16>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i16> [[TMP4]], <4 x i16> [[TMP5]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i16> [[VTRN1_I]], <4 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly16x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly16x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 16, i32 8, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL_I]], align 8
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.poly16x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i16>] [[TMP11]], [2 x <4 x i16>]* [[TMP10]], align 8
// CHECK:   [[TMP12:%.*]] = load %struct.poly16x4x2_t, %struct.poly16x4x2_t* [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x2_t [[TMP12]]
poly16x4x2_t test_vtrn_p16(poly16x4_t a, poly16x4_t b) {
  return vtrn_p16(a, b);
}
// CHECK-LABEL: define %struct.int8x16x2_t @test_vtrnq_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VTRN_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   store <16 x i8> [[VTRN_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VTRN1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.int8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.int8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.int8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.int8x16x2_t, %struct.int8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x2_t [[TMP8]]
int8x16x2_t test_vtrnq_s8(int8x16_t a, int8x16_t b) {
  return vtrnq_s8(a, b);
}
// CHECK-LABEL: define %struct.int16x8x2_t @test_vtrnq_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i16> [[VTRN_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VTRN1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.int16x8x2_t, %struct.int16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x2_t [[TMP12]]
int16x8x2_t test_vtrnq_s16(int16x8_t a, int16x8_t b) {
  return vtrnq_s16(a, b);
}
// CHECK-LABEL: define %struct.int32x4x2_t @test_vtrnq_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.int32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i32> [[VTRN_I]], <4 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i32>, <4 x i32>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i32> [[VTRN1_I]], <4 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.int32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.int32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.int32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP11]], [2 x <4 x i32>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.int32x4x2_t, %struct.int32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x2_t [[TMP12]]
int32x4x2_t test_vtrnq_s32(int32x4_t a, int32x4_t b) {
  return vtrnq_s32(a, b);
}
// CHECK-LABEL: define %struct.uint8x16x2_t @test_vtrnq_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VTRN_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   store <16 x i8> [[VTRN_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VTRN1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.uint8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.uint8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.uint8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x2_t [[TMP8]]
uint8x16x2_t test_vtrnq_u8(uint8x16_t a, uint8x16_t b) {
  return vtrnq_u8(a, b);
}
// CHECK-LABEL: define %struct.uint16x8x2_t @test_vtrnq_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i16> [[VTRN_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VTRN1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.uint16x8x2_t, %struct.uint16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x2_t [[TMP12]]
uint16x8x2_t test_vtrnq_u16(uint16x8_t a, uint16x8_t b) {
  return vtrnq_u16(a, b);
}
// CHECK-LABEL: define %struct.uint32x4x2_t @test_vtrnq_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x i32>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x i32> [[VTRN_I]], <4 x i32>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x i32>, <4 x i32>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x i32> [[VTRN1_I]], <4 x i32>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.uint32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.uint32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.uint32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x i32>] [[TMP11]], [2 x <4 x i32>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.uint32x4x2_t, %struct.uint32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x2_t [[TMP12]]
uint32x4x2_t test_vtrnq_u32(uint32x4_t a, uint32x4_t b) {
  return vtrnq_u32(a, b);
}
// CHECK-LABEL: define %struct.float32x4x2_t @test_vtrnq_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.float32x4x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <4 x float>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <4 x float> [[TMP4]], <4 x float> [[TMP5]], <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x float> [[VTRN_I]], <4 x float>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <4 x float>, <4 x float>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <4 x float> [[TMP4]], <4 x float> [[TMP5]], <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x float> [[VTRN1_I]], <4 x float>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.float32x4x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.float32x4x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.float32x4x2_t [[TMP9]], 0
// CHECK:   store [2 x <4 x float>] [[TMP11]], [2 x <4 x float>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.float32x4x2_t, %struct.float32x4x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x2_t [[TMP12]]
float32x4x2_t test_vtrnq_f32(float32x4_t a, float32x4_t b) {
  return vtrnq_f32(a, b);
}
// CHECK-LABEL: define %struct.poly8x16x2_t @test_vtrnq_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to <16 x i8>*
// CHECK:   [[VTRN_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
// CHECK:   store <16 x i8> [[VTRN_I]], <16 x i8>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <16 x i8>, <16 x i8>* [[TMP1]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
// CHECK:   store <16 x i8> [[VTRN1_I]], <16 x i8>* [[TMP2]]
// CHECK:   [[TMP3:%.*]] = bitcast %struct.poly8x16x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast %struct.poly8x16x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP3]], i8* [[TMP4]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP7:%.*]] = extractvalue %struct.poly8x16x2_t [[TMP5]], 0
// CHECK:   store [2 x <16 x i8>] [[TMP7]], [2 x <16 x i8>]* [[TMP6]], align 16
// CHECK:   [[TMP8:%.*]] = load %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x2_t [[TMP8]]
poly8x16x2_t test_vtrnq_p8(poly8x16_t a, poly8x16_t b) {
  return vtrnq_p8(a, b);
}
// CHECK-LABEL: define %struct.poly16x8x2_t @test_vtrnq_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[RETVAL_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET_I]] to i8*
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP0]] to <8 x i16>*
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK:   [[VTRN_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x i16> [[VTRN_I]], <8 x i16>* [[TMP3]]
// CHECK:   [[TMP6:%.*]] = getelementptr inbounds <8 x i16>, <8 x i16>* [[TMP3]], i32 1
// CHECK:   [[VTRN1_I:%.*]] = shufflevector <8 x i16> [[TMP4]], <8 x i16> [[TMP5]], <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x i16> [[VTRN1_I]], <8 x i16>* [[TMP6]]
// CHECK:   [[TMP7:%.*]] = bitcast %struct.poly16x8x2_t* [[RETVAL_I]] to i8*
// CHECK:   [[TMP8:%.*]] = bitcast %struct.poly16x8x2_t* [[__RET_I]] to i8*
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[TMP7]], i8* [[TMP8]], i64 32, i32 16, i1 false) #2
// CHECK:   [[TMP9:%.*]] = load %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL_I]], align 16
// CHECK:   [[TMP10:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL]], i32 0, i32 0
// CHECK:   [[TMP11:%.*]] = extractvalue %struct.poly16x8x2_t [[TMP9]], 0
// CHECK:   store [2 x <8 x i16>] [[TMP11]], [2 x <8 x i16>]* [[TMP10]], align 16
// CHECK:   [[TMP12:%.*]] = load %struct.poly16x8x2_t, %struct.poly16x8x2_t* [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x2_t [[TMP12]]
poly16x8x2_t test_vtrnq_p16(poly16x8_t a, poly16x8_t b) {
  return vtrnq_p16(a, b);
}
