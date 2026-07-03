// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vget_high_s8(<16 x i8> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vget_high_s8(int8x16_t a) {
  return vget_high_s8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vget_high_s16(<8 x i16> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vget_high_s16(int16x8_t a) {
  return vget_high_s16(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vget_high_s32(<4 x i32> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vget_high_s32(int32x4_t a) {
  return vget_high_s32(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vget_high_s64(<2 x i64> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> <i32 1>
// CHECK:   ret <1 x i64> [[SHUFFLE_I]]
int64x1_t test_vget_high_s64(int64x2_t a) {
  return vget_high_s64(a);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vget_high_u8(<16 x i8> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vget_high_u8(uint8x16_t a) {
  return vget_high_u8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vget_high_u16(<8 x i16> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vget_high_u16(uint16x8_t a) {
  return vget_high_u16(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vget_high_u32(<4 x i32> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vget_high_u32(uint32x4_t a) {
  return vget_high_u32(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vget_high_u64(<2 x i64> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> <i32 1>
// CHECK:   ret <1 x i64> [[SHUFFLE_I]]
uint64x1_t test_vget_high_u64(uint64x2_t a) {
  return vget_high_u64(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vget_high_p64(<2 x i64> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> <i32 1>
// CHECK:   ret <1 x i64> [[SHUFFLE_I]]
poly64x1_t test_vget_high_p64(poly64x2_t a) {
  return vget_high_p64(a);
}

// CHECK-LABEL: define{{.*}} <4 x half> @test_vget_high_f16(<8 x half> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <4 x half> [[SHUFFLE_I]]
float16x4_t test_vget_high_f16(float16x8_t a) {
  return vget_high_f16(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vget_high_f32(<4 x float> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vget_high_f32(float32x4_t a) {
  return vget_high_f32(a);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vget_high_p8(<16 x i8> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vget_high_p8(poly8x16_t a) {
  return vget_high_p8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vget_high_p16(<8 x i16> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vget_high_p16(poly16x8_t a) {
  return vget_high_p16(a);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vget_high_f64(<2 x double> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %a, <1 x i32> <i32 1>
// CHECK:   ret <1 x double> [[SHUFFLE_I]]
float64x1_t test_vget_high_f64(float64x2_t a) {
  return vget_high_f64(a);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vget_low_s8(<16 x i8> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vget_low_s8(int8x16_t a) {
  return vget_low_s8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vget_low_s16(<8 x i16> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vget_low_s16(int16x8_t a) {
  return vget_low_s16(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vget_low_s32(<4 x i32> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 0, i32 1>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vget_low_s32(int32x4_t a) {
  return vget_low_s32(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vget_low_s64(<2 x i64> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[SHUFFLE_I]]
int64x1_t test_vget_low_s64(int64x2_t a) {
  return vget_low_s64(a);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vget_low_u8(<16 x i8> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vget_low_u8(uint8x16_t a) {
  return vget_low_u8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vget_low_u16(<8 x i16> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vget_low_u16(uint16x8_t a) {
  return vget_low_u16(a);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vget_low_u32(<4 x i32> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 0, i32 1>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vget_low_u32(uint32x4_t a) {
  return vget_low_u32(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vget_low_u64(<2 x i64> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[SHUFFLE_I]]
uint64x1_t test_vget_low_u64(uint64x2_t a) {
  return vget_low_u64(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vget_low_p64(<2 x i64> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[SHUFFLE_I]]
poly64x1_t test_vget_low_p64(poly64x2_t a) {
  return vget_low_p64(a);
}

// CHECK-LABEL: define{{.*}} <4 x half> @test_vget_low_f16(<8 x half> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x half> [[SHUFFLE_I]]
float16x4_t test_vget_low_f16(float16x8_t a) {
  return vget_low_f16(a);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vget_low_f32(<4 x float> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %a, <2 x i32> <i32 0, i32 1>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vget_low_f32(float32x4_t a) {
  return vget_low_f32(a);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vget_low_p8(<16 x i8> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vget_low_p8(poly8x16_t a) {
  return vget_low_p8(a);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vget_low_p16(<8 x i16> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vget_low_p16(poly16x8_t a) {
  return vget_low_p16(a);
}

// CHECK-LABEL: define{{.*}} <1 x double> @test_vget_low_f64(<2 x double> noundef %a) #0 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x double> %a, <2 x double> %a, <1 x i32> zeroinitializer
// CHECK:   ret <1 x double> [[SHUFFLE_I]]
float64x1_t test_vget_low_f64(float64x2_t a) {
  return vget_low_f64(a);
}

