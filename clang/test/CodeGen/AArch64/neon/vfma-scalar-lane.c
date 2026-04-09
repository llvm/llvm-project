// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -DCIR_SCALAR_F32_F64_ONLY -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=CIRLLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -DCIR_SCALAR_F32_F64_ONLY -fclangir -emit-cir -o - %s | FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

#ifndef CIR_SCALAR_F32_F64_ONLY
// LLVM-LABEL: define {{[^@]+}}@test_vfmah_lane_f16
float16_t test_vfmah_lane_f16(float16_t a, float16_t b, float16x4_t c) {

// LLVM-SAME: (half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  [[ENTRY:.*:]]
// LLVM-NEXT:    [[EXTRACT:%.*]] = extractelement <4 x half> [[C]], i32 3
// LLVM-NEXT:    [[TMP0:%.*]] = call half @llvm.fma.f16(half [[B]], half [[EXTRACT]], half [[A]])
// LLVM-NEXT:    ret half [[TMP0]]
  return vfmah_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmah_laneq_f16
float16_t test_vfmah_laneq_f16(float16_t a, float16_t b, float16x8_t c) {

// LLVM-SAME: (half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  [[ENTRY:.*:]]
// LLVM-NEXT:    [[EXTRACT:%.*]] = extractelement <8 x half> [[C]], i32 7
// LLVM-NEXT:    [[TMP0:%.*]] = call half @llvm.fma.f16(half [[B]], half [[EXTRACT]], half [[A]])
// LLVM-NEXT:    ret half [[TMP0]]
  return vfmah_laneq_f16(a, b, c, 7);
}
#endif

// LLVM-LABEL: define dso_local float @test_vfmas_lane_f32(
// CIRLLVM-LABEL: define dso_local float @test_vfmas_lane_f32(
// CIR-LABEL: @test_vfmas_lane_f32(
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      [[EXTRACT:%.*]] = extractelement <2 x float> {{.*}}, i32 1
// CIRLLVM-NEXT: [[RES:%.*]] = call float @llvm.fma.f32(float {{.*}}, float [[EXTRACT]], float {{.*}})

// LLVM-SAME: float {{.*}} [[A:%.*]], float {{.*}} [[B:%.*]], <2 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  [[ENTRY:.*:]]
// LLVM-NEXT:    [[EXTRACT:%.*]] = extractelement <2 x float> [[C]], i32 1
// LLVM-NEXT:    [[TMP0:%.*]] = call float @llvm.fma.f32(float [[B]], float [[EXTRACT]], float [[A]])
// LLVM-NEXT:    ret float [[TMP0]]
  return vfmas_lane_f32(a, b, c, 1);
}

// LLVM-LABEL: define dso_local float @test_vfmas_laneq_f32(
// CIRLLVM-LABEL: define dso_local float @test_vfmas_laneq_f32(
// CIR-LABEL: @test_vfmas_laneq_f32(
float32_t test_vfmas_laneq_f32(float32_t a, float32_t b, float32x4_t c) {
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      [[EXTRACT:%.*]] = extractelement <4 x float> {{.*}}, i32 3
// CIRLLVM-NEXT: [[RES:%.*]] = call float @llvm.fma.f32(float {{.*}}, float [[EXTRACT]], float {{.*}})

// LLVM-SAME: float {{.*}} [[A:%.*]], float {{.*}} [[B:%.*]], <4 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  [[ENTRY:.*:]]
// LLVM-NEXT:    [[EXTRACT:%.*]] = extractelement <4 x float> [[C]], i32 3
// LLVM-NEXT:    [[TMP0:%.*]] = call float @llvm.fma.f32(float [[B]], float [[EXTRACT]], float [[A]])
// LLVM-NEXT:    ret float [[TMP0]]
  return vfmas_laneq_f32(a, b, c, 3);
}

// LLVM-LABEL: define dso_local double @test_vfmad_lane_f64(
// CIRLLVM-LABEL: define dso_local double @test_vfmad_lane_f64(
// CIR-LABEL: @test_vfmad_lane_f64(
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      [[EXTRACT:%.*]] = extractelement <1 x double> {{.*}}, i32 0
// CIRLLVM-NEXT: [[RES:%.*]] = call double @llvm.fma.f64(double {{.*}}, double [[EXTRACT]], double {{.*}})

// LLVM-SAME: double {{.*}} [[A:%.*]], double {{.*}} [[B:%.*]], <1 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  [[ENTRY:.*:]]
// LLVM-NEXT:    [[EXTRACT:%.*]] = extractelement <1 x double> [[C]], i32 0
// LLVM-NEXT:    [[TMP0:%.*]] = call double @llvm.fma.f64(double [[B]], double [[EXTRACT]], double [[A]])
// LLVM-NEXT:    ret double [[TMP0]]
  return vfmad_lane_f64(a, b, c, 0);
}

// LLVM-LABEL: define dso_local double @test_vfmad_laneq_f64(
// CIRLLVM-LABEL: define dso_local double @test_vfmad_laneq_f64(
// CIR-LABEL: @test_vfmad_laneq_f64(
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
// CIR: cir.vec.extract
// CIR: cir.call_llvm_intrinsic "fma"

// CIRLLVM:      [[EXTRACT:%.*]] = extractelement <2 x double> {{.*}}, i32 1
// CIRLLVM-NEXT: [[RES:%.*]] = call double @llvm.fma.f64(double {{.*}}, double [[EXTRACT]], double {{.*}})

// LLVM-SAME: double {{.*}} [[A:%.*]], double {{.*}} [[B:%.*]], <2 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM-NEXT:  [[ENTRY:.*:]]
// LLVM-NEXT:    [[EXTRACT:%.*]] = extractelement <2 x double> [[C]], i32 1
// LLVM-NEXT:    [[TMP0:%.*]] = call double @llvm.fma.f64(double [[B]], double [[EXTRACT]], double [[A]])
// LLVM-NEXT:    ret double [[TMP0]]
  return vfmad_laneq_f64(a, b, c, 1);
}
