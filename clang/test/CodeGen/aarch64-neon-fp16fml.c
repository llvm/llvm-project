// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +v8.2a -target-feature +neon -target-feature +fp16fml \
// RUN: -fallow-half-arguments-and-returns -disable-O0-optnone -emit-llvm -o - %s | opt -S -instcombine | FileCheck %s

// REQUIRES: aarch64-registered-target

// Test AArch64 Armv8.2-A FP16 Fused Multiply-Add Long intrinsics

#include <arm_neon.h>

// Vector form

float32x2_t test_vfmlal_low_u32(float32x2_t a, float16x4_t b, float16x4_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlal_low_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlal.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlal_low_u32(a, b, c);
}

float32x2_t test_vfmlsl_low_u32(float32x2_t a, float16x4_t b, float16x4_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlsl_low_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlsl.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlsl_low_u32(a, b, c);
}

float32x2_t test_vfmlal_high_u32(float32x2_t a, float16x4_t b, float16x4_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlal_high_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlal2.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlal_high_u32(a, b, c);
}

float32x2_t test_vfmlsl_high_u32(float32x2_t a, float16x4_t b, float16x4_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlsl_high_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlsl2.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlsl_high_u32(a, b, c);
}

float32x4_t test_vfmlalq_low_u32(float32x4_t a, float16x8_t b, float16x8_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlalq_low_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlal.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlalq_low_u32(a, b, c);
}

float32x4_t test_vfmlslq_low_u32(float32x4_t a, float16x8_t b, float16x8_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlslq_low_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlsl.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlslq_low_u32(a, b, c);
}

float32x4_t test_vfmlalq_high_u32(float32x4_t a, float16x8_t b, float16x8_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlalq_high_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlal2.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlalq_high_u32(a, b, c);
}

float32x4_t test_vfmlslq_high_u32(float32x4_t a, float16x8_t b, float16x8_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlslq_high_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlsl2.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlslq_high_u32(a, b, c);
}

// Indexed form

float32x2_t test_vfmlal_lane_low_u32(float32x2_t a, float16x4_t b, float16x4_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlal_lane_low_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x half> %c, <4 x half> undef, <4 x i32> zeroinitializer
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlal.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> [[SHUFFLE]])
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlal_lane_low_u32(a, b, c, 0);
}

float32x2_t test_vfmlal_lane_high_u32(float32x2_t a, float16x4_t b, float16x4_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlal_lane_high_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x half> %c, <4 x half> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlal2.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> [[SHUFFLE]])
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlal_lane_high_u32(a, b, c, 1);
}

float32x4_t test_vfmlalq_lane_low_u32(float32x4_t a, float16x8_t b, float16x4_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlalq_lane_low_u32(<4 x float> %a, <8 x half> %b, <4 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x half> %c, <4 x half> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlal.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> [[SHUFFLE]])
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlalq_lane_low_u32(a, b, c, 2);
}

float32x4_t test_vfmlalq_lane_high_u32(float32x4_t a, float16x8_t b, float16x4_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlalq_lane_high_u32(<4 x float> %a, <8 x half> %b, <4 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x half> %c, <4 x half> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlal2.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> [[SHUFFLE]])
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlalq_lane_high_u32(a, b, c, 3);
}

float32x2_t test_vfmlal_laneq_low_u32(float32x2_t a, float16x4_t b, float16x8_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlal_laneq_low_u32(<2 x float> %a, <4 x half> %b, <8 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <8 x half> %c, <8 x half> undef, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlal.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> [[SHUFFLE]])
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlal_laneq_low_u32(a, b, c, 4);
}

float32x2_t test_vfmlal_laneq_high_u32(float32x2_t a, float16x4_t b, float16x8_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlal_laneq_high_u32(<2 x float> %a, <4 x half> %b, <8 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <8 x half> %c, <8 x half> undef, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlal2.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> [[SHUFFLE]])
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlal_laneq_high_u32(a, b, c, 5);
}

float32x4_t test_vfmlalq_laneq_low_u32(float32x4_t a, float16x8_t b, float16x8_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlalq_laneq_low_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <8 x half> %c, <8 x half> undef, <8 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlal.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> [[SHUFFLE]])
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlalq_laneq_low_u32(a, b, c, 6);
}

float32x4_t test_vfmlalq_laneq_high_u32(float32x4_t a, float16x8_t b, float16x8_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlalq_laneq_high_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <8 x half> %c, <8 x half> undef, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlal2.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> [[SHUFFLE]])
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlalq_laneq_high_u32(a, b, c, 7);
}

float32x2_t test_vfmlsl_lane_low_u32(float32x2_t a, float16x4_t b, float16x4_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlsl_lane_low_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x half> %c, <4 x half> undef, <4 x i32> zeroinitializer
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlsl.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> [[SHUFFLE]])
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlsl_lane_low_u32(a, b, c, 0);
}

float32x2_t test_vfmlsl_lane_high_u32(float32x2_t a, float16x4_t b, float16x4_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlsl_lane_high_u32(<2 x float> %a, <4 x half> %b, <4 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x half> %c, <4 x half> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlsl2.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> [[SHUFFLE]])
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlsl_lane_high_u32(a, b, c, 1);
}

float32x4_t test_vfmlslq_lane_low_u32(float32x4_t a, float16x8_t b, float16x4_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlslq_lane_low_u32(<4 x float> %a, <8 x half> %b, <4 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x half> %c, <4 x half> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlsl.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> [[SHUFFLE]])
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlslq_lane_low_u32(a, b, c, 2);
}

float32x4_t test_vfmlslq_lane_high_u32(float32x4_t a, float16x8_t b, float16x4_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlslq_lane_high_u32(<4 x float> %a, <8 x half> %b, <4 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x half> %c, <4 x half> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlsl2.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> [[SHUFFLE]])
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlslq_lane_high_u32(a, b, c, 3);
}

float32x2_t test_vfmlsl_laneq_low_u32(float32x2_t a, float16x4_t b, float16x8_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlsl_laneq_low_u32(<2 x float> %a, <4 x half> %b, <8 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <8 x half> %c, <8 x half> undef, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlsl.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> [[SHUFFLE]])
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlsl_laneq_low_u32(a, b, c, 4);
}

float32x2_t test_vfmlsl_laneq_high_u32(float32x2_t a, float16x4_t b, float16x8_t c) {
// CHECK-LABEL: define <2 x float> @test_vfmlsl_laneq_high_u32(<2 x float> %a, <4 x half> %b, <8 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <8 x half> %c, <8 x half> undef, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
// CHECK: [[RESULT:%.*]] = call <2 x float> @llvm.aarch64.neon.fmlsl2.v2f32.v4f16(<2 x float> %a, <4 x half> %b, <4 x half> [[SHUFFLE]])
// CHECK: ret <2 x float> [[RESULT]]
  return vfmlsl_laneq_high_u32(a, b, c, 5);
}

float32x4_t test_vfmlslq_laneq_low_u32(float32x4_t a, float16x8_t b, float16x8_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlslq_laneq_low_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <8 x half> %c, <8 x half> undef, <8 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlsl.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> [[SHUFFLE]])
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlslq_laneq_low_u32(a, b, c, 6);
}

float32x4_t test_vfmlslq_laneq_high_u32(float32x4_t a, float16x8_t b, float16x8_t c) {
// CHECK-LABEL: define <4 x float> @test_vfmlslq_laneq_high_u32(<4 x float> %a, <8 x half> %b, <8 x half> %c)
// CHECK: [[SHUFFLE:%.*]] = shufflevector <8 x half> %c, <8 x half> undef, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.aarch64.neon.fmlsl2.v4f32.v8f16(<4 x float> %a, <8 x half> %b, <8 x half> [[SHUFFLE]])
// CHECK: ret <4 x float> [[RESULT]]
  return vfmlslq_laneq_high_u32(a, b, c, 7);
}
