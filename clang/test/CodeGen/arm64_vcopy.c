// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -disable-O0-optnone -emit-llvm %s | opt -S -mem2reg | FileCheck %s

// Test ARM64 SIMD copy vector element to vector element: vcopyq_lane*

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vcopyq_laneq_s8(<16 x i8> %a1, <16 x i8> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %a2, i32 13
// CHECK:   [[VSET_LANE:%.*]] = insertelement <16 x i8> %a1, i8 [[VGETQ_LANE]], i32 3
// CHECK:   ret <16 x i8> [[VSET_LANE]]
int8x16_t test_vcopyq_laneq_s8(int8x16_t a1, int8x16_t a2) {
  return vcopyq_laneq_s8(a1, (int64_t) 3, a2, (int64_t) 13);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vcopyq_laneq_u8(<16 x i8> %a1, <16 x i8> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %a2, i32 13
// CHECK:   [[VSET_LANE:%.*]] = insertelement <16 x i8> %a1, i8 [[VGETQ_LANE]], i32 3
// CHECK:   ret <16 x i8> [[VSET_LANE]]
uint8x16_t test_vcopyq_laneq_u8(uint8x16_t a1, uint8x16_t a2) {
  return vcopyq_laneq_u8(a1, (int64_t) 3, a2, (int64_t) 13);

}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vcopyq_laneq_s16(<8 x i16> %a1, <8 x i16> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %a2, i32 7
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i16> %a1, i16 [[VGETQ_LANE]], i32 3
// CHECK:   ret <8 x i16> [[VSET_LANE]]
int16x8_t test_vcopyq_laneq_s16(int16x8_t a1, int16x8_t a2) {
  return vcopyq_laneq_s16(a1, (int64_t) 3, a2, (int64_t) 7);

}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vcopyq_laneq_u16(<8 x i16> %a1, <8 x i16> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %a2, i32 7
// CHECK:   [[VSET_LANE:%.*]] = insertelement <8 x i16> %a1, i16 [[VGETQ_LANE]], i32 3
// CHECK:   ret <8 x i16> [[VSET_LANE]]
uint16x8_t test_vcopyq_laneq_u16(uint16x8_t a1, uint16x8_t a2) {
  return vcopyq_laneq_u16(a1, (int64_t) 3, a2, (int64_t) 7);

}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcopyq_laneq_s32(<4 x i32> %a1, <4 x i32> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %a2, i32 3
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x i32> %a1, i32 [[VGETQ_LANE]], i32 3
// CHECK:   ret <4 x i32> [[VSET_LANE]]
int32x4_t test_vcopyq_laneq_s32(int32x4_t a1, int32x4_t a2) {
  return vcopyq_laneq_s32(a1, (int64_t) 3, a2, (int64_t) 3);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vcopyq_laneq_u32(<4 x i32> %a1, <4 x i32> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %a2, i32 3
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x i32> %a1, i32 [[VGETQ_LANE]], i32 3
// CHECK:   ret <4 x i32> [[VSET_LANE]]
uint32x4_t test_vcopyq_laneq_u32(uint32x4_t a1, uint32x4_t a2) {
  return vcopyq_laneq_u32(a1, (int64_t) 3, a2, (int64_t) 3);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vcopyq_laneq_s64(<2 x i64> %a1, <2 x i64> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %a2, i32 1
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i64> %a1, i64 [[VGETQ_LANE]], i32 0
// CHECK:   ret <2 x i64> [[VSET_LANE]]
int64x2_t test_vcopyq_laneq_s64(int64x2_t a1, int64x2_t a2) {
  return vcopyq_laneq_s64(a1, (int64_t) 0, a2, (int64_t) 1);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vcopyq_laneq_u64(<2 x i64> %a1, <2 x i64> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %a2, i32 1
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i64> %a1, i64 [[VGETQ_LANE]], i32 0
// CHECK:   ret <2 x i64> [[VSET_LANE]]
uint64x2_t test_vcopyq_laneq_u64(uint64x2_t a1, uint64x2_t a2) {
  return vcopyq_laneq_u64(a1, (int64_t) 0, a2, (int64_t) 1);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vcopyq_laneq_f32(<4 x float> %a1, <4 x float> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <4 x float> %a2, i32 3
// CHECK:   [[VSET_LANE:%.*]] = insertelement <4 x float> %a1, float [[VGETQ_LANE]], i32 0
// CHECK:   ret <4 x float> [[VSET_LANE]]
float32x4_t test_vcopyq_laneq_f32(float32x4_t a1, float32x4_t a2) {
  return vcopyq_laneq_f32(a1, 0, a2, 3);
}

// CHECK-LABEL: define{{.*}} <2 x double> @test_vcopyq_laneq_f64(<2 x double> %a1, <2 x double> %a2) #0 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x double> %a2, i32 1
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x double> %a1, double [[VGETQ_LANE]], i32 0
// CHECK:   ret <2 x double> [[VSET_LANE]]
float64x2_t test_vcopyq_laneq_f64(float64x2_t a1, float64x2_t a2) {
  return vcopyq_laneq_f64(a1, 0, a2, 1);
}

