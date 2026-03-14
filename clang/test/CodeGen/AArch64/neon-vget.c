// RUN: %clang_cc1 -triple arm64-apple-darwin -target-feature +neon -flax-vector-conversions=none \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} i8 @test_vget_lane_u8(<8 x i8> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <8 x i8> %a, i32 7
// CHECK:   ret i8 [[VGET_LANE]]
uint8_t test_vget_lane_u8(uint8x8_t a) {
  return vget_lane_u8(a, 7);
}

// CHECK-LABEL: define{{.*}} i16 @test_vget_lane_u16(<4 x i16> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %a, i32 3
// CHECK:   ret i16 [[VGET_LANE]]
uint16_t test_vget_lane_u16(uint16x4_t a) {
  return vget_lane_u16(a, 3);
}

// CHECK-LABEL: define{{.*}} i32 @test_vget_lane_u32(<2 x i32> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x i32> %a, i32 1
// CHECK:   ret i32 [[VGET_LANE]]
uint32_t test_vget_lane_u32(uint32x2_t a) {
  return vget_lane_u32(a, 1);
}

// CHECK-LABEL: define{{.*}} i8 @test_vget_lane_s8(<8 x i8> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <8 x i8> %a, i32 7
// CHECK:   ret i8 [[VGET_LANE]]
int8_t test_vget_lane_s8(int8x8_t a) {
  return vget_lane_s8(a, 7);
}

// CHECK-LABEL: define{{.*}} i16 @test_vget_lane_s16(<4 x i16> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %a, i32 3
// CHECK:   ret i16 [[VGET_LANE]]
int16_t test_vget_lane_s16(int16x4_t a) {
  return vget_lane_s16(a, 3);
}

// CHECK-LABEL: define{{.*}} i32 @test_vget_lane_s32(<2 x i32> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x i32> %a, i32 1
// CHECK:   ret i32 [[VGET_LANE]]
int32_t test_vget_lane_s32(int32x2_t a) {
  return vget_lane_s32(a, 1);
}

// CHECK-LABEL: define{{.*}} i8 @test_vget_lane_p8(<8 x i8> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <8 x i8> %a, i32 7
// CHECK:   ret i8 [[VGET_LANE]]
poly8_t test_vget_lane_p8(poly8x8_t a) {
  return vget_lane_p8(a, 7);
}

// CHECK-LABEL: define{{.*}} i16 @test_vget_lane_p16(<4 x i16> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %a, i32 3
// CHECK:   ret i16 [[VGET_LANE]]
poly16_t test_vget_lane_p16(poly16x4_t a) {
  return vget_lane_p16(a, 3);
}

// CHECK-LABEL: define{{.*}} float @test_vget_lane_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x float> %a, i32 1
// CHECK:   ret float [[VGET_LANE]]
float32_t test_vget_lane_f32(float32x2_t a) {
  return vget_lane_f32(a, 1);
}

// CHECK-LABEL: define{{.*}} float @test_vget_lane_f16(<4 x half> noundef %a) #0 {
// CHECK:   [[__REINT_242:%.*]] = alloca <4 x half>, align 8
// CHECK:   [[__REINT1_242:%.*]] = alloca i16, align 2
// CHECK:   store <4 x half> %a, ptr [[__REINT_242]], align 8
// CHECK:   [[TMP1:%.*]] = load <4 x i16>, ptr [[__REINT_242]], align 8
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> [[TMP1]], i32 1
// CHECK:   store i16 [[VGET_LANE]], ptr [[__REINT1_242]], align 2
// CHECK:   [[TMP5:%.*]] = load half, ptr [[__REINT1_242]], align 2
// CHECK:   [[CONV:%.*]] = fpext half [[TMP5]] to float
// CHECK:   ret float [[CONV]]
float32_t test_vget_lane_f16(float16x4_t a) {
  return vget_lane_f16(a, 1);
}

// CHECK-LABEL: define{{.*}} i8 @test_vgetq_lane_u8(<16 x i8> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %a, i32 15
// CHECK:   ret i8 [[VGETQ_LANE]]
uint8_t test_vgetq_lane_u8(uint8x16_t a) {
  return vgetq_lane_u8(a, 15);
}

// CHECK-LABEL: define{{.*}} i16 @test_vgetq_lane_u16(<8 x i16> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %a, i32 7
// CHECK:   ret i16 [[VGETQ_LANE]]
uint16_t test_vgetq_lane_u16(uint16x8_t a) {
  return vgetq_lane_u16(a, 7);
}

// CHECK-LABEL: define{{.*}} i32 @test_vgetq_lane_u32(<4 x i32> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %a, i32 3
// CHECK:   ret i32 [[VGETQ_LANE]]
uint32_t test_vgetq_lane_u32(uint32x4_t a) {
  return vgetq_lane_u32(a, 3);
}

// CHECK-LABEL: define{{.*}} i8 @test_vgetq_lane_s8(<16 x i8> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %a, i32 15
// CHECK:   ret i8 [[VGETQ_LANE]]
int8_t test_vgetq_lane_s8(int8x16_t a) {
  return vgetq_lane_s8(a, 15);
}

// CHECK-LABEL: define{{.*}} i16 @test_vgetq_lane_s16(<8 x i16> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %a, i32 7
// CHECK:   ret i16 [[VGETQ_LANE]]
int16_t test_vgetq_lane_s16(int16x8_t a) {
  return vgetq_lane_s16(a, 7);
}

// CHECK-LABEL: define{{.*}} i32 @test_vgetq_lane_s32(<4 x i32> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %a, i32 3
// CHECK:   ret i32 [[VGETQ_LANE]]
int32_t test_vgetq_lane_s32(int32x4_t a) {
  return vgetq_lane_s32(a, 3);
}

// CHECK-LABEL: define{{.*}} i8 @test_vgetq_lane_p8(<16 x i8> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %a, i32 15
// CHECK:   ret i8 [[VGETQ_LANE]]
poly8_t test_vgetq_lane_p8(poly8x16_t a) {
  return vgetq_lane_p8(a, 15);
}

// CHECK-LABEL: define{{.*}} i16 @test_vgetq_lane_p16(<8 x i16> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %a, i32 7
// CHECK:   ret i16 [[VGETQ_LANE]]
poly16_t test_vgetq_lane_p16(poly16x8_t a) {
  return vgetq_lane_p16(a, 7);
}

// CHECK-LABEL: define{{.*}} float @test_vgetq_lane_f32(<4 x float> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x float> %a, i32 3
// CHECK:   ret float [[VGETQ_LANE]]
float32_t test_vgetq_lane_f32(float32x4_t a) {
  return vgetq_lane_f32(a, 3);
}

// CHECK-LABEL: define{{.*}} float @test_vgetq_lane_f16(<8 x half> noundef %a) #0 {
// CHECK:   [[__REINT_244:%.*]] = alloca <8 x half>, align 16
// CHECK:   [[__REINT1_244:%.*]] = alloca i16, align 2
// CHECK:   store <8 x half> %a, ptr [[__REINT_244]], align 16
// CHECK:   [[TMP1:%.*]] = load <8 x i16>, ptr [[__REINT_244]], align 16
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> [[TMP1]], i32 3
// CHECK:   store i16 [[VGETQ_LANE]], ptr [[__REINT1_244]], align 2
// CHECK:   [[TMP5:%.*]] = load half, ptr [[__REINT1_244]], align 2
// CHECK:   [[CONV:%.*]] = fpext half [[TMP5]] to float
// CHECK:   ret float [[CONV]]
float32_t test_vgetq_lane_f16(float16x8_t a) {
  return vgetq_lane_f16(a, 3);
}

// CHECK-LABEL: define{{.*}} i64 @test_vget_lane_s64(<1 x i64> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x i64> %a, i32 0
// CHECK:   ret i64 [[VGET_LANE]]
int64_t test_vget_lane_s64(int64x1_t a) {
  return vget_lane_s64(a, 0);
}

// CHECK-LABEL: define{{.*}} i64 @test_vget_lane_u64(<1 x i64> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x i64> %a, i32 0
// CHECK:   ret i64 [[VGET_LANE]]
uint64_t test_vget_lane_u64(uint64x1_t a) {
  return vget_lane_u64(a, 0);
}

// CHECK-LABEL: define{{.*}} i64 @test_vgetq_lane_s64(<2 x i64> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %a, i32 1
// CHECK:   ret i64 [[VGETQ_LANE]]
int64_t test_vgetq_lane_s64(int64x2_t a) {
  return vgetq_lane_s64(a, 1);
}

// CHECK-LABEL: define{{.*}} i64 @test_vgetq_lane_u64(<2 x i64> noundef %a) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %a, i32 1
// CHECK:   ret i64 [[VGETQ_LANE]]
uint64_t test_vgetq_lane_u64(uint64x2_t a) {
  return vgetq_lane_u64(a, 1);
}
