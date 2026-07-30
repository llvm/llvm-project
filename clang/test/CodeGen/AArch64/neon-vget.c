// RUN: %clang_cc1 -triple arm64-apple-darwin -target-feature +neon -flax-vector-conversions=none \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vset_lane_u8(i8 noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i8> %b, i8 %a, i32 7
// CHECK:   ret <8 x i8> [[VSET_LANE]]
uint8x8_t test_vset_lane_u8(uint8_t a, uint8x8_t b) {
  return vset_lane_u8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vset_lane_u16(i16 noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x i16> %b, i16 %a, i32 3
// CHECK:   ret <4 x i16> [[VSET_LANE]]
uint16x4_t test_vset_lane_u16(uint16_t a, uint16x4_t b) {
  return vset_lane_u16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vset_lane_u32(i32 noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i32> %b, i32 %a, i32 1
// CHECK:   ret <2 x i32> [[VSET_LANE]]
uint32x2_t test_vset_lane_u32(uint32_t a, uint32x2_t b) {
  return vset_lane_u32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vset_lane_s8(i8 noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i8> %b, i8 %a, i32 7
// CHECK:   ret <8 x i8> [[VSET_LANE]]
int8x8_t test_vset_lane_s8(int8_t a, int8x8_t b) {
  return vset_lane_s8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vset_lane_s16(i16 noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x i16> %b, i16 %a, i32 3
// CHECK:   ret <4 x i16> [[VSET_LANE]]
int16x4_t test_vset_lane_s16(int16_t a, int16x4_t b) {
  return vset_lane_s16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x i32> @test_vset_lane_s32(i32 noundef %a, <2 x i32> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i32> %b, i32 %a, i32 1
// CHECK:   ret <2 x i32> [[VSET_LANE]]
int32x2_t test_vset_lane_s32(int32_t a, int32x2_t b) {
  return vset_lane_s32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vset_lane_p8(i8 noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i8> %b, i8 %a, i32 7
// CHECK:   ret <8 x i8> [[VSET_LANE]]
poly8x8_t test_vset_lane_p8(poly8_t a, poly8x8_t b) {
  return vset_lane_p8(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i16> @test_vset_lane_p16(i16 noundef %a, <4 x i16> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x i16> %b, i16 %a, i32 3
// CHECK:   ret <4 x i16> [[VSET_LANE]]
poly16x4_t test_vset_lane_p16(poly16_t a, poly16x4_t b) {
  return vset_lane_p16(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <2 x float> @test_vset_lane_f32(float noundef %a, <2 x float> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x float> %b, float %a, i32 1
// CHECK:   ret <2 x float> [[VSET_LANE]]
float32x2_t test_vset_lane_f32(float32_t a, float32x2_t b) {
  return vset_lane_f32(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <4 x half> @test_vset_lane_f16(ptr noundef %a, <4 x half> noundef %b) #0 {
// CHECK:   [[__REINT_246:%.*]] = alloca half, align 2
// CHECK:   [[__REINT1_246:%.*]] = alloca <4 x half>, align 8
// CHECK:   [[__REINT2_246:%.*]] = alloca <4 x i16>, align 8
// CHECK:   [[TMP0:%.*]] = load half, ptr %a, align 2
// CHECK:   store half [[TMP0]], ptr [[__REINT_246]], align 2
// CHECK:   store <4 x half> %b, ptr [[__REINT1_246]], align 8
// CHECK:   [[TMP2:%.*]] = load i16, ptr [[__REINT_246]], align 2
// CHECK:   [[TMP4:%.*]] = load <4 x i16>, ptr [[__REINT1_246]], align 8
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x i16> [[TMP4]], i16 [[TMP2]], i32 3
// CHECK:   store <4 x i16> [[VSET_LANE]], ptr [[__REINT2_246]], align 8
// CHECK:   [[TMP8:%.*]] = load <4 x half>, ptr [[__REINT2_246]], align 8
// CHECK:   ret <4 x half> [[TMP8]]
float16x4_t test_vset_lane_f16(float16_t *a, float16x4_t b) {
  return vset_lane_f16(*a, b, 3);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vsetq_lane_u8(i8 noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <16 x i8> %b, i8 %a, i32 15
// CHECK:   ret <16 x i8> [[VSET_LANE]]
uint8x16_t test_vsetq_lane_u8(uint8_t a, uint8x16_t b) {
  return vsetq_lane_u8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vsetq_lane_u16(i16 noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i16> %b, i16 %a, i32 7
// CHECK:   ret <8 x i16> [[VSET_LANE]]
uint16x8_t test_vsetq_lane_u16(uint16_t a, uint16x8_t b) {
  return vsetq_lane_u16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vsetq_lane_u32(i32 noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x i32> %b, i32 %a, i32 3
// CHECK:   ret <4 x i32> [[VSET_LANE]]
uint32x4_t test_vsetq_lane_u32(uint32_t a, uint32x4_t b) {
  return vsetq_lane_u32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vsetq_lane_s8(i8 noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <16 x i8> %b, i8 %a, i32 15
// CHECK:   ret <16 x i8> [[VSET_LANE]]
int8x16_t test_vsetq_lane_s8(int8_t a, int8x16_t b) {
  return vsetq_lane_s8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vsetq_lane_s16(i16 noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i16> %b, i16 %a, i32 7
// CHECK:   ret <8 x i16> [[VSET_LANE]]
int16x8_t test_vsetq_lane_s16(int16_t a, int16x8_t b) {
  return vsetq_lane_s16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vsetq_lane_s32(i32 noundef %a, <4 x i32> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x i32> %b, i32 %a, i32 3
// CHECK:   ret <4 x i32> [[VSET_LANE]]
int32x4_t test_vsetq_lane_s32(int32_t a, int32x4_t b) {
  return vsetq_lane_s32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vsetq_lane_p8(i8 noundef %a, <16 x i8> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <16 x i8> %b, i8 %a, i32 15
// CHECK:   ret <16 x i8> [[VSET_LANE]]
poly8x16_t test_vsetq_lane_p8(poly8_t a, poly8x16_t b) {
  return vsetq_lane_p8(a, b, 15);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vsetq_lane_p16(i16 noundef %a, <8 x i16> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i16> %b, i16 %a, i32 7
// CHECK:   ret <8 x i16> [[VSET_LANE]]
poly16x8_t test_vsetq_lane_p16(poly16_t a, poly16x8_t b) {
  return vsetq_lane_p16(a, b, 7);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vsetq_lane_f32(float noundef %a, <4 x float> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x float> %b, float %a, i32 3
// CHECK:   ret <4 x float> [[VSET_LANE]]
float32x4_t test_vsetq_lane_f32(float32_t a, float32x4_t b) {
  return vsetq_lane_f32(a, b, 3);
}

// CHECK-LABEL: define{{.*}} <8 x half> @test_vsetq_lane_f16(ptr noundef %a, <8 x half> noundef %b) #0 {
// CHECK:   [[__REINT_248:%.*]] = alloca half, align 2
// CHECK:   [[__REINT1_248:%.*]] = alloca <8 x half>, align 16
// CHECK:   [[__REINT2_248:%.*]] = alloca <8 x i16>, align 16
// CHECK:   [[TMP0:%.*]] = load half, ptr %a, align 2
// CHECK:   store half [[TMP0]], ptr [[__REINT_248]], align 2
// CHECK:   store <8 x half> %b, ptr [[__REINT1_248]], align 16
// CHECK:   [[TMP2:%.*]] = load i16, ptr [[__REINT_248]], align 2
// CHECK:   [[TMP4:%.*]] = load <8 x i16>, ptr [[__REINT1_248]], align 16
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i16> [[TMP4]], i16 [[TMP2]], i32 7
// CHECK:   store <8 x i16> [[VSET_LANE]], ptr [[__REINT2_248]], align 16
// CHECK:   [[TMP8:%.*]] = load <8 x half>, ptr [[__REINT2_248]], align 16
// CHECK:   ret <8 x half> [[TMP8]]
float16x8_t test_vsetq_lane_f16(float16_t *a, float16x8_t b) {
  return vsetq_lane_f16(*a, b, 7);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vset_lane_s64(i64 noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x i64> %b, i64 %a, i32 0
// CHECK:   ret <1 x i64> [[VSET_LANE]]
int64x1_t test_vset_lane_s64(int64_t a, int64x1_t b) {
  return vset_lane_s64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vset_lane_u64(i64 noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x i64> %b, i64 %a, i32 0
// CHECK:   ret <1 x i64> [[VSET_LANE]]
uint64x1_t test_vset_lane_u64(uint64_t a, uint64x1_t b) {
  return vset_lane_u64(a, b, 0);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vsetq_lane_s64(i64 noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i64> %b, i64 %a, i32 1
// CHECK:   ret <2 x i64> [[VSET_LANE]]
int64x2_t test_vsetq_lane_s64(int64_t a, int64x2_t b) {
  return vsetq_lane_s64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vsetq_lane_u64(i64 noundef %a, <2 x i64> noundef %b) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i64> %b, i64 %a, i32 1
// CHECK:   ret <2 x i64> [[VSET_LANE]]
uint64x2_t test_vsetq_lane_u64(uint64_t a, uint64x2_t b) {
  return vsetq_lane_u64(a, b, 1);
}
