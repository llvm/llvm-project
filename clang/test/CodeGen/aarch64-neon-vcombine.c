// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -fallow-half-arguments-and-returns -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vcombine_s8(<8 x i8> %low, <8 x i8> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %low, <8 x i8> %high, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vcombine_s8(int8x8_t low, int8x8_t high) {
  return vcombine_s8(low, high);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vcombine_s16(<4 x i16> %low, <4 x i16> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %low, <4 x i16> %high, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vcombine_s16(int16x4_t low, int16x4_t high) {
  return vcombine_s16(low, high);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcombine_s32(<2 x i32> %low, <2 x i32> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %low, <2 x i32> %high, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vcombine_s32(int32x2_t low, int32x2_t high) {
  return vcombine_s32(low, high);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vcombine_s64(<1 x i64> %low, <1 x i64> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <1 x i64> %low, <1 x i64> %high, <2 x i32> <i32 0, i32 1>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
int64x2_t test_vcombine_s64(int64x1_t low, int64x1_t high) {
  return vcombine_s64(low, high);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vcombine_u8(<8 x i8> %low, <8 x i8> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %low, <8 x i8> %high, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vcombine_u8(uint8x8_t low, uint8x8_t high) {
  return vcombine_u8(low, high);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vcombine_u16(<4 x i16> %low, <4 x i16> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %low, <4 x i16> %high, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vcombine_u16(uint16x4_t low, uint16x4_t high) {
  return vcombine_u16(low, high);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcombine_u32(<2 x i32> %low, <2 x i32> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %low, <2 x i32> %high, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vcombine_u32(uint32x2_t low, uint32x2_t high) {
  return vcombine_u32(low, high);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vcombine_u64(<1 x i64> %low, <1 x i64> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <1 x i64> %low, <1 x i64> %high, <2 x i32> <i32 0, i32 1>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
uint64x2_t test_vcombine_u64(uint64x1_t low, uint64x1_t high) {
  return vcombine_u64(low, high);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vcombine_p64(<1 x i64> %low, <1 x i64> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <1 x i64> %low, <1 x i64> %high, <2 x i32> <i32 0, i32 1>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
poly64x2_t test_vcombine_p64(poly64x1_t low, poly64x1_t high) {
  return vcombine_p64(low, high);
}

// CHECK-LABEL: define{{.*}} <8 x half> @test_vcombine_f16(<4 x half> %low, <4 x half> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x half> %low, <4 x half> %high, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x half> [[SHUFFLE_I]]
float16x8_t test_vcombine_f16(float16x4_t low, float16x4_t high) {
  return vcombine_f16(low, high);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vcombine_f32(<2 x float> %low, <2 x float> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %low, <2 x float> %high, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vcombine_f32(float32x2_t low, float32x2_t high) {
  return vcombine_f32(low, high);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vcombine_p8(<8 x i8> %low, <8 x i8> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %low, <8 x i8> %high, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vcombine_p8(poly8x8_t low, poly8x8_t high) {
  return vcombine_p8(low, high);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vcombine_p16(<4 x i16> %low, <4 x i16> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %low, <4 x i16> %high, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vcombine_p16(poly16x4_t low, poly16x4_t high) {
  return vcombine_p16(low, high);
}

// CHECK-LABEL: define{{.*}} <2 x double> @test_vcombine_f64(<1 x double> %low, <1 x double> %high) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <1 x double> %low, <1 x double> %high, <2 x i32> <i32 0, i32 1>
// CHECK:   ret <2 x double> [[SHUFFLE_I]]
float64x2_t test_vcombine_f64(float64x1_t low, float64x1_t high) {
  return vcombine_f64(low, high);
}
