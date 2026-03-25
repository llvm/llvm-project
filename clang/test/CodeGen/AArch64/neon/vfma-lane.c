// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

// LLVM-LABEL: @test_vfma_lane_f16(
// LLVM: shufflevector <4 x half>
// LLVM: call <4 x half> @llvm.fma.v4f16(
// CIR-LABEL: @test_vfma_lane_f16(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float16x4_t test_vfma_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfma_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: @test_vfmaq_lane_f16(
// LLVM: shufflevector <4 x half>
// LLVM: call <8 x half> @llvm.fma.v8f16(
// CIR-LABEL: @test_vfmaq_lane_f16(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float16x8_t test_vfmaq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c) {
  return vfmaq_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: @test_vfma_laneq_f16(
// LLVM: shufflevector <8 x half>
// LLVM: call <4 x half> @llvm.fma.v4f16(
// CIR-LABEL: @test_vfma_laneq_f16(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float16x4_t test_vfma_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c) {
  return vfma_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: @test_vfmaq_laneq_f16(
// LLVM: shufflevector <8 x half>
// LLVM: call <8 x half> @llvm.fma.v8f16(
// CIR-LABEL: @test_vfmaq_laneq_f16(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float16x8_t test_vfmaq_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmaq_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: @test_vfma_lane_f32(
// LLVM: shufflevector <2 x float>
// LLVM: call <2 x float> @llvm.fma.v2f32(
// CIR-LABEL: @test_vfma_lane_f32(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float32x2_t test_vfma_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vfma_lane_f32(a, b, v, 1);
}

// LLVM-LABEL: @test_vfmaq_lane_f32(
// LLVM: shufflevector <2 x float>
// LLVM: call <4 x float> @llvm.fma.v4f32(
// CIR-LABEL: @test_vfmaq_lane_f32(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float32x4_t test_vfmaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vfmaq_lane_f32(a, b, v, 1);
}

// LLVM-LABEL: @test_vfma_laneq_f32(
// LLVM: shufflevector <4 x float>
// LLVM: call <2 x float> @llvm.fma.v2f32(
// CIR-LABEL: @test_vfma_laneq_f32(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float32x2_t test_vfma_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vfma_laneq_f32(a, b, v, 3);
}

// LLVM-LABEL: @test_vfmaq_laneq_f32(
// LLVM: shufflevector <4 x float>
// LLVM: call <4 x float> @llvm.fma.v4f32(
// CIR-LABEL: @test_vfmaq_laneq_f32(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float32x4_t test_vfmaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vfmaq_laneq_f32(a, b, v, 3);
}

// LLVM-LABEL: @test_vfma_lane_f64(
// LLVM: shufflevector <1 x double>
// LLVM: call <1 x double> @llvm.fma.v1f64(
// CIR-LABEL: @test_vfma_lane_f64(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfma_lane_f64(a, b, v, 0);
}

// LLVM-LABEL: @test_vfmaq_lane_f64(
// LLVM: shufflevector <1 x double>
// LLVM: call <2 x double> @llvm.fma.v2f64(
// CIR-LABEL: @test_vfmaq_lane_f64(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float64x2_t test_vfmaq_lane_f64(float64x2_t a, float64x2_t b, float64x1_t v) {
  return vfmaq_lane_f64(a, b, v, 0);
}

// LLVM-LABEL: @test_vfma_laneq_f64(
// LLVM: @llvm.fma
// CIR-LABEL: @test_vfma_laneq_f64(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfma_laneq_f64(a, b, v, 0);
}

// LLVM-LABEL: @test_vfmaq_laneq_f64(
// LLVM: shufflevector <2 x double>
// LLVM: call <2 x double> @llvm.fma.v2f64(
// CIR-LABEL: @test_vfmaq_laneq_f64(
// CIR: cir.vec.shuffle
// CIR: cir.call_llvm_intrinsic "fma"
float64x2_t test_vfmaq_laneq_f64(float64x2_t a, float64x2_t b, float64x2_t v) {
  return vfmaq_laneq_f64(a, b, v, 1);
}
