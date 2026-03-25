// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

// LLVM-LABEL: @test_vfmah_lane_f16(
// LLVM: extractelement <4 x half>
// LLVM: call half @llvm.fma.f16(
// CIR-LABEL: @test_vfmah_lane_f16(
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"
float16_t test_vfmah_lane_f16(float16_t a, float16_t b, float16x4_t c) {
  return vfmah_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: @test_vfmah_laneq_f16(
// LLVM: extractelement <8 x half>
// LLVM: call half @llvm.fma.f16(
// CIR-LABEL: @test_vfmah_laneq_f16(
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"
float16_t test_vfmah_laneq_f16(float16_t a, float16_t b, float16x8_t c) {
  return vfmah_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: @test_vfmas_lane_f32(
// LLVM: extractelement <2 x float>
// LLVM: call float @llvm.fma.f32(
// CIR-LABEL: @test_vfmas_lane_f32(
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmas_lane_f32(a, b, c, 1);
}

// LLVM-LABEL: @test_vfmas_laneq_f32(
// LLVM: extractelement <4 x float>
// LLVM: call float @llvm.fma.f32(
// CIR-LABEL: @test_vfmas_laneq_f32(
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"
float32_t test_vfmas_laneq_f32(float32_t a, float32_t b, float32x4_t c) {
  return vfmas_laneq_f32(a, b, c, 3);
}

// LLVM-LABEL: @test_vfmad_lane_f64(
// LLVM: extractelement <1 x double>
// LLVM: call double @llvm.fma.f64(
// CIR-LABEL: @test_vfmad_lane_f64(
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
  return vfmad_lane_f64(a, b, c, 0);
}

// LLVM-LABEL: @test_vfmad_laneq_f64(
// LLVM: extractelement <2 x double>
// LLVM: call double @llvm.fma.f64(
// CIR-LABEL: @test_vfmad_laneq_f64(
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
  return vfmad_laneq_f64(a, b, c, 1);
}
