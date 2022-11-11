// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN: -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} float @test_vdups_lane_f32(<2 x float> noundef %a) #0 {
// CHECK:   [[VDUPS_LANE:%.*]] = extractelement <2 x float> %a, i32 1
// CHECK:   ret float [[VDUPS_LANE]]
float32_t test_vdups_lane_f32(float32x2_t a) {
  return vdups_lane_f32(a, 1);
}


// CHECK-LABEL: define{{.*}} double @test_vdupd_lane_f64(<1 x double> noundef %a) #0 {
// CHECK:   [[VDUPD_LANE:%.*]] = extractelement <1 x double> %a, i32 0
// CHECK:   ret double [[VDUPD_LANE]]
float64_t test_vdupd_lane_f64(float64x1_t a) {
  return vdupd_lane_f64(a, 0);
}


// CHECK-LABEL: define{{.*}} float @test_vdups_laneq_f32(<4 x float> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x float> %a, i32 3
// CHECK:   ret float [[VGETQ_LANE]]
float32_t test_vdups_laneq_f32(float32x4_t a) {
  return vdups_laneq_f32(a, 3);
}


// CHECK-LABEL: define{{.*}} double @test_vdupd_laneq_f64(<2 x double> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x double> %a, i32 1
// CHECK:   ret double [[VGETQ_LANE]]
float64_t test_vdupd_laneq_f64(float64x2_t a) {
  return vdupd_laneq_f64(a, 1);
}


// CHECK-LABEL: define{{.*}} i8 @test_vdupb_lane_s8(<8 x i8> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <8 x i8> %a, i32 7
// CHECK:   ret i8 [[VGET_LANE]]
int8_t test_vdupb_lane_s8(int8x8_t a) {
  return vdupb_lane_s8(a, 7);
}


// CHECK-LABEL: define{{.*}} i16 @test_vduph_lane_s16(<4 x i16> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %a, i32 3
// CHECK:   ret i16 [[VGET_LANE]]
int16_t test_vduph_lane_s16(int16x4_t a) {
  return vduph_lane_s16(a, 3);
}


// CHECK-LABEL: define{{.*}} i32 @test_vdups_lane_s32(<2 x i32> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x i32> %a, i32 1
// CHECK:   ret i32 [[VGET_LANE]]
int32_t test_vdups_lane_s32(int32x2_t a) {
  return vdups_lane_s32(a, 1);
}


// CHECK-LABEL: define{{.*}} i64 @test_vdupd_lane_s64(<1 x i64> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x i64> %a, i32 0
// CHECK:   ret i64 [[VGET_LANE]]
int64_t test_vdupd_lane_s64(int64x1_t a) {
  return vdupd_lane_s64(a, 0);
}


// CHECK-LABEL: define{{.*}} i8 @test_vdupb_lane_u8(<8 x i8> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <8 x i8> %a, i32 7
// CHECK:   ret i8 [[VGET_LANE]]
uint8_t test_vdupb_lane_u8(uint8x8_t a) {
  return vdupb_lane_u8(a, 7);
}


// CHECK-LABEL: define{{.*}} i16 @test_vduph_lane_u16(<4 x i16> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %a, i32 3
// CHECK:   ret i16 [[VGET_LANE]]
uint16_t test_vduph_lane_u16(uint16x4_t a) {
  return vduph_lane_u16(a, 3);
}


// CHECK-LABEL: define{{.*}} i32 @test_vdups_lane_u32(<2 x i32> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <2 x i32> %a, i32 1
// CHECK:   ret i32 [[VGET_LANE]]
uint32_t test_vdups_lane_u32(uint32x2_t a) {
  return vdups_lane_u32(a, 1);
}


// CHECK-LABEL: define{{.*}} i64 @test_vdupd_lane_u64(<1 x i64> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x i64> %a, i32 0
// CHECK:   ret i64 [[VGET_LANE]]
uint64_t test_vdupd_lane_u64(uint64x1_t a) {
  return vdupd_lane_u64(a, 0);
}

// CHECK-LABEL: define{{.*}} i8 @test_vdupb_laneq_s8(<16 x i8> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %a, i32 15
// CHECK:   ret i8 [[VGETQ_LANE]]
int8_t test_vdupb_laneq_s8(int8x16_t a) {
  return vdupb_laneq_s8(a, 15);
}


// CHECK-LABEL: define{{.*}} i16 @test_vduph_laneq_s16(<8 x i16> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %a, i32 7
// CHECK:   ret i16 [[VGETQ_LANE]]
int16_t test_vduph_laneq_s16(int16x8_t a) {
  return vduph_laneq_s16(a, 7);
}


// CHECK-LABEL: define{{.*}} i32 @test_vdups_laneq_s32(<4 x i32> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %a, i32 3
// CHECK:   ret i32 [[VGETQ_LANE]]
int32_t test_vdups_laneq_s32(int32x4_t a) {
  return vdups_laneq_s32(a, 3);
}


// CHECK-LABEL: define{{.*}} i64 @test_vdupd_laneq_s64(<2 x i64> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %a, i32 1
// CHECK:   ret i64 [[VGETQ_LANE]]
int64_t test_vdupd_laneq_s64(int64x2_t a) {
  return vdupd_laneq_s64(a, 1);
}


// CHECK-LABEL: define{{.*}} i8 @test_vdupb_laneq_u8(<16 x i8> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %a, i32 15
// CHECK:   ret i8 [[VGETQ_LANE]]
uint8_t test_vdupb_laneq_u8(uint8x16_t a) {
  return vdupb_laneq_u8(a, 15);
}


// CHECK-LABEL: define{{.*}} i16 @test_vduph_laneq_u16(<8 x i16> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %a, i32 7
// CHECK:   ret i16 [[VGETQ_LANE]]
uint16_t test_vduph_laneq_u16(uint16x8_t a) {
  return vduph_laneq_u16(a, 7);
}


// CHECK-LABEL: define{{.*}} i32 @test_vdups_laneq_u32(<4 x i32> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %a, i32 3
// CHECK:   ret i32 [[VGETQ_LANE]]
uint32_t test_vdups_laneq_u32(uint32x4_t a) {
  return vdups_laneq_u32(a, 3);
}


// CHECK-LABEL: define{{.*}} i64 @test_vdupd_laneq_u64(<2 x i64> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %a, i32 1
// CHECK:   ret i64 [[VGETQ_LANE]]
uint64_t test_vdupd_laneq_u64(uint64x2_t a) {
  return vdupd_laneq_u64(a, 1);
}

// CHECK-LABEL: define{{.*}} i8 @test_vdupb_lane_p8(<8 x i8> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <8 x i8> %a, i32 7
// CHECK:   ret i8 [[VGET_LANE]]
poly8_t test_vdupb_lane_p8(poly8x8_t a) {
  return vdupb_lane_p8(a, 7);
}

// CHECK-LABEL: define{{.*}} i16 @test_vduph_lane_p16(<4 x i16> noundef %a) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <4 x i16> %a, i32 3
// CHECK:   ret i16 [[VGET_LANE]]
poly16_t test_vduph_lane_p16(poly16x4_t a) {
  return vduph_lane_p16(a, 3);
}

// CHECK-LABEL: define{{.*}} i8 @test_vdupb_laneq_p8(<16 x i8> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %a, i32 15
// CHECK:   ret i8 [[VGETQ_LANE]]
poly8_t test_vdupb_laneq_p8(poly8x16_t a) {
  return vdupb_laneq_p8(a, 15);
}

// CHECK-LABEL: define{{.*}} i16 @test_vduph_laneq_p16(<8 x i16> noundef %a) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %a, i32 7
// CHECK:   ret i16 [[VGETQ_LANE]]
poly16_t test_vduph_laneq_p16(poly16x8_t a) {
  return vduph_laneq_p16(a, 7);
}

// CHECK: attributes #0 ={{.*}}"min-legal-vector-width"="64"
// CHECK: attributes #1 ={{.*}}"min-legal-vector-width"="128"
