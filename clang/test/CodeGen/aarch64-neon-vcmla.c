// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-apple-ios -target-feature +neon \
// RUN:        -target-feature +v8.3a \
// RUN:        -target-feature +fullfp16 \
// RUN:        -disable-O0-optnone -emit-llvm -o - %s | opt -S -O1 | FileCheck %s
#include <arm_neon.h>

// CHECK-LABEL: @test_vcmla_f16(
// CHECK: [[RES:%.*]] = call <4 x half> @llvm.aarch64.neon.vcmla.rot0.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> %rhs)
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_f32(
// CHECK: [[RES:%.*]] = call <2 x float> @llvm.aarch64.neon.vcmla.rot0.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_f16(
// CHECK: [[RES:%.*]] = call <8 x half> @llvm.aarch64.neon.vcmla.rot0.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> %rhs)
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_f32(
// CHECK: [[RES:%.*]] = call <4 x float> @llvm.aarch64.neon.vcmla.rot0.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> %rhs)
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_f64(
// CHECK: [[RES:%.*]] = call <2 x double> @llvm.aarch64.neon.vcmla.rot0.v2f64(<2 x double> %acc, <2 x double> %lhs, <2 x double> %rhs)
// CHECK: ret <2 x double> [[RES]]
float64x2_t test_vcmlaq_f64(float64x2_t acc, float64x2_t lhs, float64x2_t rhs) {
  return vcmlaq_f64(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot90_f16(
// CHECK: [[RES:%.*]] = call <4 x half> @llvm.aarch64.neon.vcmla.rot90.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> %rhs)
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot90_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot90_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot90_f32(
// CHECK: [[RES:%.*]] = call <2 x float> @llvm.aarch64.neon.vcmla.rot90.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot90_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot90_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot90_f16(
// CHECK: [[RES:%.*]] = call <8 x half> @llvm.aarch64.neon.vcmla.rot90.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> %rhs)
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot90_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot90_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot90_f32(
// CHECK: [[RES:%.*]] = call <4 x float> @llvm.aarch64.neon.vcmla.rot90.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> %rhs)
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot90_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot90_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot90_f64(
// CHECK: [[RES:%.*]] = call <2 x double> @llvm.aarch64.neon.vcmla.rot90.v2f64(<2 x double> %acc, <2 x double> %lhs, <2 x double> %rhs)
// CHECK: ret <2 x double> [[RES]]
float64x2_t test_vcmlaq_rot90_f64(float64x2_t acc, float64x2_t lhs, float64x2_t rhs) {
  return vcmlaq_rot90_f64(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot180_f16(
// CHECK: [[RES:%.*]] = call <4 x half> @llvm.aarch64.neon.vcmla.rot180.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> %rhs)
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot180_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot180_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot180_f32(
// CHECK: [[RES:%.*]] = call <2 x float> @llvm.aarch64.neon.vcmla.rot180.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot180_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot180_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot180_f16(
// CHECK: [[RES:%.*]] = call <8 x half> @llvm.aarch64.neon.vcmla.rot180.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> %rhs)
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot180_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot180_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot180_f32(
// CHECK: [[RES:%.*]] = call <4 x float> @llvm.aarch64.neon.vcmla.rot180.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> %rhs)
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot180_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot180_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot180_f64(
// CHECK: [[RES:%.*]] = call <2 x double> @llvm.aarch64.neon.vcmla.rot180.v2f64(<2 x double> %acc, <2 x double> %lhs, <2 x double> %rhs)
// CHECK: ret <2 x double> [[RES]]
float64x2_t test_vcmlaq_rot180_f64(float64x2_t acc, float64x2_t lhs, float64x2_t rhs) {
  return vcmlaq_rot180_f64(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot270_f16(
// CHECK: [[RES:%.*]] = call <4 x half> @llvm.aarch64.neon.vcmla.rot270.v4f16(<4 x half> %acc, <4 x half> %lhs, <4 x half> %rhs)
// CHECK: ret <4 x half> [[RES]]
float16x4_t test_vcmla_rot270_f16(float16x4_t acc, float16x4_t lhs, float16x4_t rhs) {
  return vcmla_rot270_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmla_rot270_f32(
// CHECK: [[RES:%.*]] = call <2 x float> @llvm.aarch64.neon.vcmla.rot270.v2f32(<2 x float> %acc, <2 x float> %lhs, <2 x float> %rhs)
// CHECK: ret <2 x float> [[RES]]
float32x2_t test_vcmla_rot270_f32(float32x2_t acc, float32x2_t lhs, float32x2_t rhs) {
  return vcmla_rot270_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot270_f16(
// CHECK: [[RES:%.*]] = call <8 x half> @llvm.aarch64.neon.vcmla.rot270.v8f16(<8 x half> %acc, <8 x half> %lhs, <8 x half> %rhs)
// CHECK: ret <8 x half> [[RES]]
float16x8_t test_vcmlaq_rot270_f16(float16x8_t acc, float16x8_t lhs, float16x8_t rhs) {
  return vcmlaq_rot270_f16(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot270_f32(
// CHECK: [[RES:%.*]] = call <4 x float> @llvm.aarch64.neon.vcmla.rot270.v4f32(<4 x float> %acc, <4 x float> %lhs, <4 x float> %rhs)
// CHECK: ret <4 x float> [[RES]]
float32x4_t test_vcmlaq_rot270_f32(float32x4_t acc, float32x4_t lhs, float32x4_t rhs) {
  return vcmlaq_rot270_f32(acc, lhs, rhs);
}

// CHECK-LABEL: @test_vcmlaq_rot270_f64(
// CHECK: [[RES:%.*]] = call <2 x double> @llvm.aarch64.neon.vcmla.rot270.v2f64(<2 x double> %acc, <2 x double> %lhs, <2 x double> %rhs)
// CHECK: ret <2 x double> [[RES]]
float64x2_t test_vcmlaq_rot270_f64(float64x2_t acc, float64x2_t lhs, float64x2_t rhs) {
  return vcmlaq_rot270_f64(acc, lhs, rhs);
}
