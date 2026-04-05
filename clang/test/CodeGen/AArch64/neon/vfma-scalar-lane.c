// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

// LLVM-LABEL: define {{[^@]+}}@test_vfmah_lane_f16
// LLVM-SAME: (half noundef [[A:%.*]], half noundef [[B:%.*]], <4 x half> noundef [[C:%.*]]) #[[ATTR0:[0-9]+]] {
// CIR-LABEL: @test_vfmah_lane_f16(
float16_t test_vfmah_lane_f16(float16_t a, float16_t b, float16x4_t c) {
  // CIR: cir.vec.extract
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[EXTRACT:%.*]] = extractelement <4 x half> [[C]], i32 3
  // LLVM: [[TMP0:%.*]] = call half @llvm.fma.f16(half [[B]], half [[EXTRACT]], half [[A]])
  // LLVM: ret half [[TMP0]]
  return vfmah_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmah_laneq_f16
// LLVM-SAME: (half noundef [[A:%.*]], half noundef [[B:%.*]], <8 x half> noundef [[C:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmah_laneq_f16(
float16_t test_vfmah_laneq_f16(float16_t a, float16_t b, float16x8_t c) {
  // CIR: cir.vec.extract
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[EXTRACT:%.*]] = extractelement <8 x half> [[C]], i32 7
  // LLVM: [[TMP0:%.*]] = call half @llvm.fma.f16(half [[B]], half [[EXTRACT]], half [[A]])
  // LLVM: ret half [[TMP0]]
  return vfmah_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: define dso_local float @test_vfmas_lane_f32(
// LLVM-SAME: float noundef [[A:%.*]], float noundef [[B:%.*]], <2 x float> noundef [[C:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmas_lane_f32(
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  // CIR: cir.vec.extract
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[EXTRACT:%.*]] = extractelement <2 x float> [[C]], i32 1
  // LLVM: [[TMP0:%.*]] = call float @llvm.fma.f32(float [[B]], float [[EXTRACT]], float [[A]])
  // LLVM: ret float [[TMP0]]
  return vfmas_lane_f32(a, b, c, 1);
}

// LLVM-LABEL: define dso_local float @test_vfmas_laneq_f32(
// LLVM-SAME: float noundef [[A:%.*]], float noundef [[B:%.*]], <4 x float> noundef [[C:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmas_laneq_f32(
float32_t test_vfmas_laneq_f32(float32_t a, float32_t b, float32x4_t c) {
  // CIR: cir.vec.extract
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[EXTRACT:%.*]] = extractelement <4 x float> [[C]], i32 3
  // LLVM: [[TMP0:%.*]] = call float @llvm.fma.f32(float [[B]], float [[EXTRACT]], float [[A]])
  // LLVM: ret float [[TMP0]]
  return vfmas_laneq_f32(a, b, c, 3);
}

// LLVM-LABEL: define dso_local double @test_vfmad_lane_f64(
// LLVM-SAME: double noundef [[A:%.*]], double noundef [[B:%.*]], <1 x double> noundef [[C:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmad_lane_f64(
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
  // CIR: cir.vec.extract
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[EXTRACT:%.*]] = extractelement <1 x double> [[C]], i32 0
  // LLVM: [[TMP0:%.*]] = call double @llvm.fma.f64(double [[B]], double [[EXTRACT]], double [[A]])
  // LLVM: ret double [[TMP0]]
  return vfmad_lane_f64(a, b, c, 0);
}

// LLVM-LABEL: define dso_local double @test_vfmad_laneq_f64(
// LLVM-SAME: double noundef [[A:%.*]], double noundef [[B:%.*]], <2 x double> noundef [[C:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmad_laneq_f64(
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
  // CIR: cir.vec.extract
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[EXTRACT:%.*]] = extractelement <2 x double> [[C]], i32 1
  // LLVM: [[TMP0:%.*]] = call double @llvm.fma.f64(double [[B]], double [[EXTRACT]], double [[A]])
  // LLVM: ret double [[TMP0]]
  return vfmad_laneq_f64(a, b, c, 1);
}
