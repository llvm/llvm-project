// RUN: %clang_cc1 -triple arm64-apple-ios -target-feature +neon \
// RUN:        -target-feature +v8.3a \
// RUN:        -target-feature +fullfp16 \
// RUN:        -disable-O0-optnone -emit-llvm -o - %s | opt -S -O1 | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

// CHECK-LABEL: @test_vcmla_f16(
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot0.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> %rhs)
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_f32(
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot0.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_f16(
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot0.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> %rhs)
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_f32(
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot0.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> %rhs)
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_f64(
// CHECK: [[RES:%.*]] = tail call <2 x double> @llvm.aarch64.neon.vcmla.rot0.v2f64(<2 x double> %acc, <2 x double> %lhs, <2 x double> %rhs)
// CHECK: ret <2 x double> [[RES]]
float64x2_t test_vcmlaq_f64(float64x2_t acc, float64x2_t lhs, float64x2_t rhs) {
  return vcmlaq_f64(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot90_f16(
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot90.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> %rhs)
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot90_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot90_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot90_f32(
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot90.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot90_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot90_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot90_f16(
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot90.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> %rhs)
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot90_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot90_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot90_f32(
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot90.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> %rhs)
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot90_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot90_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot90_f64(
// CHECK: [[RES:%.*]] = tail call <2 x double> @llvm.aarch64.neon.vcmla.rot90.v2f64(<2 x double> %acc, <2 x double> %lhs, <2 x double> %rhs)
// CHECK: ret <2 x double> [[RES]]
float64x2_t test_vcmlaq_rot90_f64(float64x2_t acc, float64x2_t lhs, float64x2_t rhs) {
  return vcmlaq_rot90_f64(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot180_f16(
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot180.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> %rhs)
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot180_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot180_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot180_f32(
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot180.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot180_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot180_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot180_f16(
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot180.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> %rhs)
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot180_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot180_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot180_f32(
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot180.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> %rhs)
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot180_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot180_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot180_f64(
// CHECK: [[RES:%.*]] = tail call <2 x double> @llvm.aarch64.neon.vcmla.rot180.v2f64(<2 x double> %acc, <2 x double> %lhs, <2 x double> %rhs)
// CHECK: ret <2 x double> [[RES]]
float64x2_t test_vcmlaq_rot180_f64(float64x2_t acc, float64x2_t lhs, float64x2_t rhs) {
  return vcmlaq_rot180_f64(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot270_f16(
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot270.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> %rhs)
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot270_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot270_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot270_f32(
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot270.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot270_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot270_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot270_f16(
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot270.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> %rhs)
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot270_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot270_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot270_f32(
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot270.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> %rhs)
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot270_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot270_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot270_f64(
// CHECK: [[RES:%.*]] = tail call <2 x double> @llvm.aarch64.neon.vcmla.rot270.v2f64(<2 x double> %acc, <2 x double> %lhs, <2 x double> %rhs)
// CHECK: ret <2 x double> [[RES]]
float64x2_t test_vcmlaq_rot270_f64(float64x2_t acc, float64x2_t lhs, float64x2_t rhs) {
  return vcmlaq_rot270_f64(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_lane_f16(
// CHECK: [[DUP:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <4 x i32> <i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot0.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> [[DUP]])
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_lane_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_lane_f16(acc, lhs, rhs, 1);
}

// ACLE says this exists, but it won't map to a single instruction if lane > 1.
// CHECK-LABEL: @test_vcmla_laneq_f16(
// CHECK: [[CPLX:%.*]] = bitcast <8 x half> %rhs to <4 x i32>
// CHECK: [[DUP:%.*]] = shufflevector <4 x i32> [[CPLX]], <4 x i32> poison, <2 x i32> <i32 3, i32 3>
// CHECK: [[DUP_FLT:%.*]] = bitcast <2 x i32> [[DUP]] to <4 x half>
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot0.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> [[DUP_FLT]])
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_laneq_f16(float16x4_t acc, float16x4_t lhs, float16x8_t rhs) {
  return vcmla_laneq_f16(acc, lhs, rhs, 3);
}

// CHECK-LABEL: @test_vcmlaq_lane_f16(
// CHECK: [[DUP:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <8 x i32> <i32 2, i32 3, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3> 
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot0.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> [[DUP]])
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_lane_f16(float16x8_t acc, float16x8_t lhs, float16x4_t rhs) {
  return vcmlaq_lane_f16(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmlaq_laneq_f16(
// CHECK: [[CPLX:%.*]] = bitcast <8 x half> %rhs to <4 x i32>
// CHECK: [[DUP:%.*]] = shufflevector <4 x i32> [[CPLX]], <4 x i32> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[DUP_FLT:%.*]] = bitcast <4 x i32> [[DUP]] to <8 x half>
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot0.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> [[DUP_FLT]])
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_laneq_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_laneq_f16(acc, lhs, rhs, 3);
}

// CHECK-LABEL: @test_vcmla_lane_f32(
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot0.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_lane_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_lane_f32(acc, lhs, rhs, 0);
}

// ACLE says this exists, but it won't map to a single instruction if lane > 1.
// CHECK-LABEL: @test_vcmla_laneq_f32(
// CHECK: [[CPLX:%.*]] = bitcast <4 x float> %rhs to <2 x i64>
// CHECK: [[DUP:%.*]] = shufflevector <2 x i64> [[CPLX]], <2 x i64> poison, <1 x i32> <i32 1>
// CHECK: [[DUP_FLT:%.*]] = bitcast <1 x i64> [[DUP]] to <2 x float>
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot0.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> [[DUP_FLT]])
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_laneq_f32(float32x2_t acc, float32x2_t lhs, float32x4_t rhs) {
  return vcmla_laneq_f32(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmlaq_lane_f32(
// CHECK: [[CPLX:%.*]] = bitcast <2 x float> %rhs to i64
// CHECK: [[CPLX_VEC:%.*]] = insertelement <2 x i64> poison, i64 [[CPLX]], i64 0
// CHECK: [[CPLX2:%.*]] = bitcast <2 x i64> [[CPLX_VEC]] to <4 x float>
// CHECK: [[DUP:%.*]] = shufflevector <4 x float> [[CPLX2]], <4 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot0.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> [[DUP]])
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_lane_f32(float32x4_t acc, float32x4_t lhs, float32x2_t rhs) {
  return vcmlaq_lane_f32(acc, lhs, rhs, 0);
}

// CHECK-LABEL: @test_vcmlaq_laneq_f32(
// CHECK: [[DUP:%.*]] = shufflevector <4 x float> %rhs, <4 x float> poison, <4 x i32> <i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot0.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> [[DUP]])
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_laneq_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_laneq_f32(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmla_rot90_lane_f16(
// CHECK: [[DUP:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <4 x i32> <i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot90.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> [[DUP]])
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot90_lane_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot90_lane_f16(acc, lhs, rhs, 1);
}

// ACLE says this exists, but it won't map to a single instruction if lane > 1.
// CHECK-LABEL: @test_vcmla_rot90_laneq_f16(
// CHECK: [[CPLX:%.*]] = bitcast <8 x half> %rhs to <4 x i32>
// CHECK: [[DUP:%.*]] = shufflevector <4 x i32> [[CPLX]], <4 x i32> poison, <2 x i32> <i32 3, i32 3>
// CHECK: [[DUP_FLT:%.*]] = bitcast <2 x i32> [[DUP]] to <4 x half>
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot90.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> [[DUP_FLT]])
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot90_laneq_f16(float16x4_t acc, float16x4_t lhs, float16x8_t rhs) {
  return vcmla_rot90_laneq_f16(acc, lhs, rhs, 3);
}

// CHECK-LABEL: @test_vcmlaq_rot90_lane_f16(
// CHECK: [[DUP:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <8 x i32> <i32 2, i32 3, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot90.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> [[DUP]])
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot90_lane_f16(float16x8_t acc, float16x8_t lhs, float16x4_t rhs) {
  return vcmlaq_rot90_lane_f16(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmlaq_rot90_laneq_f16(
// CHECK: [[CPLX:%.*]] = bitcast <8 x half> %rhs to <4 x i32>
// CHECK: [[DUP:%.*]] = shufflevector <4 x i32> [[CPLX]], <4 x i32> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[DUP_FLT:%.*]] = bitcast <4 x i32> [[DUP]] to <8 x half>
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot90.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> [[DUP_FLT]])
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot90_laneq_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot90_laneq_f16(acc, lhs, rhs, 3);
}

// CHECK-LABEL: @test_vcmla_rot90_lane_f32(
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot90.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot90_lane_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot90_lane_f32(acc, lhs, rhs, 0);
}

// ACLE says this exists, but it won't map to a single instruction if lane > 1.
// CHECK-LABEL: @test_vcmla_rot90_laneq_f32(
// CHECK: [[CPLX:%.*]] = bitcast <4 x float> %rhs to <2 x i64>
// CHECK: [[DUP:%.*]] = shufflevector <2 x i64> [[CPLX]], <2 x i64> poison, <1 x i32> <i32 1>
// CHECK: [[DUP_FLT:%.*]] = bitcast <1 x i64> [[DUP]] to <2 x float>
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot90.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> [[DUP_FLT]])
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot90_laneq_f32(float32x2_t acc, float32x2_t lhs, float32x4_t rhs) {
  return vcmla_rot90_laneq_f32(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmlaq_rot90_lane_f32(
// CHECK: [[CPLX:%.*]] = bitcast <2 x float> %rhs to i64
// CHECK: [[CPLX_VEC:%.*]] = insertelement <2 x i64> poison, i64 [[CPLX]], i64 0
// CHECK: [[CPLX2:%.*]] = bitcast <2 x i64> [[CPLX_VEC]] to <4 x float>
// CHECK: [[DUP:%.*]] = shufflevector <4 x float> [[CPLX2]], <4 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot90.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> [[DUP]])
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot90_lane_f32(float32x4_t acc, float32x4_t lhs, float32x2_t rhs) {
  return vcmlaq_rot90_lane_f32(acc, lhs, rhs, 0);
}

// CHECK-LABEL: @test_vcmlaq_rot90_laneq_f32(
// CHECK: [[DUP:%.*]] = shufflevector <4 x float> %rhs, <4 x float> poison, <4 x i32> <i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot90.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> [[DUP]])
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot90_laneq_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot90_laneq_f32(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmla_rot180_lane_f16(
// CHECK: [[DUP:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <4 x i32> <i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot180.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> [[DUP]])
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot180_lane_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot180_lane_f16(acc, lhs, rhs, 1);
}

// ACLE says this exists, but it won't map to a single instruction if lane > 1.
// CHECK-LABEL: @test_vcmla_rot180_laneq_f16(
// CHECK: [[CPLX:%.*]] = bitcast <8 x half> %rhs to <4 x i32>
// CHECK: [[DUP:%.*]] = shufflevector <4 x i32> [[CPLX]], <4 x i32> poison, <2 x i32> <i32 3, i32 3>
// CHECK: [[DUP_FLT:%.*]] = bitcast <2 x i32> [[DUP]] to <4 x half>
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot180.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> [[DUP_FLT]])
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot180_laneq_f16(float16x4_t acc, float16x4_t lhs, float16x8_t rhs) {
  return vcmla_rot180_laneq_f16(acc, lhs, rhs, 3);
}

// CHECK-LABEL: @test_vcmlaq_rot180_lane_f16(
// CHECK: [[DUP:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <8 x i32> <i32 2, i32 3, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot180.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> [[DUP]])
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot180_lane_f16(float16x8_t acc, float16x8_t lhs, float16x4_t rhs) {
  return vcmlaq_rot180_lane_f16(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmlaq_rot180_laneq_f16(
// CHECK: [[CPLX:%.*]] = bitcast <8 x half> %rhs to <4 x i32>
// CHECK: [[DUP:%.*]] = shufflevector <4 x i32> [[CPLX]], <4 x i32> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[DUP_FLT:%.*]] = bitcast <4 x i32> [[DUP]] to <8 x half>
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot180.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> [[DUP_FLT]])
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot180_laneq_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot180_laneq_f16(acc, lhs, rhs, 3);
}

// CHECK-LABEL: @test_vcmla_rot180_lane_f32(
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot180.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot180_lane_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot180_lane_f32(acc, lhs, rhs, 0);
}

// ACLE says this exists, but it won't map to a single instruction if lane > 1.
// CHECK-LABEL: @test_vcmla_rot180_laneq_f32(
// CHECK: [[CPLX:%.*]] = bitcast <4 x float> %rhs to <2 x i64>
// CHECK: [[DUP:%.*]] = shufflevector <2 x i64> [[CPLX]], <2 x i64> poison, <1 x i32> <i32 1>
// CHECK: [[DUP_FLT:%.*]] = bitcast <1 x i64> [[DUP]] to <2 x float>
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot180.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> [[DUP_FLT]])
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot180_laneq_f32(float32x2_t acc, float32x2_t lhs, float32x4_t rhs) {
  return vcmla_rot180_laneq_f32(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmlaq_rot180_lane_f32(
// CHECK: [[CPLX:%.*]] = bitcast <2 x float> %rhs to i64
// CHECK: [[CPLX_VEC:%.*]] = insertelement <2 x i64> poison, i64 [[CPLX]], i64 0
// CHECK: [[CPLX2:%.*]] = bitcast <2 x i64> [[CPLX_VEC]] to <4 x float>
// CHECK: [[DUP:%.*]] = shufflevector <4 x float> [[CPLX2]], <4 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot180.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> [[DUP]])
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot180_lane_f32(float32x4_t acc, float32x4_t lhs, float32x2_t rhs) {
  return vcmlaq_rot180_lane_f32(acc, lhs, rhs, 0);
}

// CHECK-LABEL: @test_vcmlaq_rot180_laneq_f32(
// CHECK: [[DUP:%.*]] = shufflevector <4 x float> %rhs, <4 x float> poison, <4 x i32> <i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot180.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> [[DUP]])
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot180_laneq_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot180_laneq_f32(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmla_rot270_lane_f16(
// CHECK: [[DUP:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <4 x i32> <i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot270.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> [[DUP]])
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot270_lane_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot270_lane_f16(acc, lhs, rhs, 1);
}

// ACLE says this exists, but it won't map to a single instruction if lane > 1.
// CHECK-LABEL: @test_vcmla_rot270_laneq_f16(
// CHECK: [[CPLX:%.*]] = bitcast <8 x half> %rhs to <4 x i32>
// CHECK: [[DUP:%.*]] = shufflevector <4 x i32> [[CPLX]], <4 x i32> poison, <2 x i32> <i32 3, i32 3>
// CHECK: [[DUP_FLT:%.*]] = bitcast <2 x i32> [[DUP]] to <4 x half>
// CHECK: [[RES:%.*]] = tail call <4 x half> @llvm.aarch64.neon.vcmla.rot270.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> [[DUP_FLT]])
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot270_laneq_f16(float16x4_t acc, float16x4_t lhs, float16x8_t rhs) {
  return vcmla_rot270_laneq_f16(acc, lhs, rhs, 3);
}

// CHECK-LABEL: @test_vcmlaq_rot270_lane_f16(
// CHECK: [[DUP:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <8 x i32> <i32 2, i32 3, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot270.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> [[DUP]])
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot270_lane_f16(float16x8_t acc, float16x8_t lhs, float16x4_t rhs) {
  return vcmlaq_rot270_lane_f16(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmlaq_rot270_laneq_f16(
// CHECK: [[CPLX:%.*]] = bitcast <8 x half> %rhs to <4 x i32>
// CHECK: [[DUP:%.*]] = shufflevector <4 x i32> [[CPLX]], <4 x i32> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[DUP_FLT:%.*]] = bitcast <4 x i32> [[DUP]] to <8 x half>
// CHECK: [[RES:%.*]] = tail call <8 x half> @llvm.aarch64.neon.vcmla.rot270.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> [[DUP_FLT]])
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot270_laneq_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot270_laneq_f16(acc, lhs, rhs, 3);
}

// CHECK-LABEL: @test_vcmla_rot270_lane_f32(
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot270.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot270_lane_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot270_lane_f32(acc, lhs, rhs, 0);
}

// ACLE says this exists, but it won't map to a single instruction if lane > 1.
// CHECK-LABEL: @test_vcmla_rot270_laneq_f32(
// CHECK: [[CPLX:%.*]] = bitcast <4 x float> %rhs to <2 x i64>
// CHECK: [[DUP:%.*]] = shufflevector <2 x i64> [[CPLX]], <2 x i64> poison, <1 x i32> <i32 1>
// CHECK: [[DUP_FLT:%.*]] = bitcast <1 x i64> [[DUP]] to <2 x float>
// CHECK: [[RES:%.*]] = tail call <2 x float> @llvm.aarch64.neon.vcmla.rot270.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> [[DUP_FLT]])
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot270_laneq_f32(float32x2_t acc, float32x2_t lhs, float32x4_t rhs) {
  return vcmla_rot270_laneq_f32(acc, lhs, rhs, 1);
}

// CHECK-LABEL: @test_vcmlaq_rot270_lane_f32(
// CHECK: [[CPLX:%.*]] = bitcast <2 x float> %rhs to i64
// CHECK: [[CPLX_VEC:%.*]] = insertelement <2 x i64> poison, i64 [[CPLX]], i64 0
// CHECK: [[CPLX2:%.*]] = bitcast <2 x i64> [[DUP]] to <4 x float>
// CHECK: [[DUP:%.*]] = shufflevector <4 x float> [[CPLX2]], <4 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot270.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> [[DUP]])
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot270_lane_f32(float32x4_t acc, float32x4_t lhs, float32x2_t rhs) {
  return vcmlaq_rot270_lane_f32(acc, lhs, rhs, 0);
}

// CHECK-LABEL: @test_vcmlaq_rot270_laneq_f32(
// CHECK: [[DUP:%.*]] = shufflevector <4 x float> %rhs, <4 x float> poison, <4 x i32> <i32 2, i32 3, i32 2, i32 3>
// CHECK: [[RES:%.*]] = tail call <4 x float> @llvm.aarch64.neon.vcmla.rot270.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> [[DUP]])
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot270_laneq_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot270_laneq_f32(acc, lhs, rhs, 1);
}
