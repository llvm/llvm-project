// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM,PLAINLLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM,CIRLLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

// LLVM-LABEL: define {{[^@]+}}@test_vfma_lane_f16
// LLVM-SAME: (<4 x half> noundef [[A:%.*]], <4 x half> noundef [[B:%.*]], <4 x half> noundef [[C:%.*]]) #[[ATTR0:[0-9]+]] {
// CIR-LABEL: @test_vfma_lane_f16(
float16x4_t test_vfma_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
  // LLVM: [[TMP5:%.*]] = bitcast <4 x i16> [[TMP2]] to <8 x i8>
  // LLVM: [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x half>
  // LLVM: shufflevector <4 x half> [[TMP6]], <4 x half> {{[^,]+}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  // LLVM: call <4 x half> @llvm.fma.v4f16(
  return vfma_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmaq_lane_f16
// LLVM-SAME: (<8 x half> noundef [[A:%.*]], <8 x half> noundef [[B:%.*]], <4 x half> noundef [[C:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmaq_lane_f16(
float16x8_t test_vfmaq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
  // LLVM: [[TMP5:%.*]] = bitcast <4 x i16> [[TMP2]] to <8 x i8>
  // LLVM: [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <4 x half>
  // LLVM: shufflevector <4 x half> [[TMP6]], <4 x half> {{[^,]+}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  // LLVM: call <8 x half> @llvm.fma.v8f16(
  return vfmaq_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfma_laneq_f16
// LLVM-SAME: (<4 x half> noundef [[A:%.*]], <4 x half> noundef [[B:%.*]], <8 x half> noundef [[C:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfma_laneq_f16(
float16x4_t test_vfma_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
  // LLVM: [[TMP5:%.*]] = bitcast <8 x i16> [[TMP2]] to <16 x i8>
  // LLVM: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x half>
  // LLVM: shufflevector <8 x half> [[TMP8]], <8 x half> {{[^,]+}}, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  // LLVM: call <4 x half> @llvm.fma.v4f16(
  return vfma_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmaq_laneq_f16
// LLVM-SAME: (<8 x half> noundef [[A:%.*]], <8 x half> noundef [[B:%.*]], <8 x half> noundef [[C:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmaq_laneq_f16(
float16x8_t test_vfmaq_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
  // LLVM: [[TMP5:%.*]] = bitcast <8 x i16> [[TMP2]] to <16 x i8>
  // LLVM: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <8 x half>
  // LLVM: shufflevector <8 x half> [[TMP8]], <8 x half> {{[^,]+}}, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  // LLVM: call <8 x half> @llvm.fma.v8f16(
  return vfmaq_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfma_lane_f32(
// LLVM-SAME: <2 x float> noundef [[A:%.*]], <2 x float> noundef [[B:%.*]], <2 x float> noundef [[V:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfma_lane_f32(
float32x2_t test_vfma_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x float> [[B]] to <2 x i32>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
  // LLVM: [[TMP5:%.*]] = bitcast <2 x i32> [[TMP2]] to <8 x i8>
  // LLVM: [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x float>
  // LLVM: shufflevector <2 x float> [[TMP6]], <2 x float> {{[^,]+}}, <2 x i32> <i32 1, i32 1>
  // LLVM: call <2 x float> @llvm.fma.v2f32(
  return vfma_lane_f32(a, b, v, 1);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmaq_lane_f32(
// LLVM-SAME: <4 x float> noundef [[A:%.*]], <4 x float> noundef [[B:%.*]], <2 x float> noundef [[V:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmaq_lane_f32(
float32x4_t test_vfmaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x float> [[B]] to <4 x i32>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
  // LLVM: [[TMP5:%.*]] = bitcast <2 x i32> [[TMP2]] to <8 x i8>
  // LLVM: [[TMP6:%.*]] = bitcast <8 x i8> [[TMP5]] to <2 x float>
  // LLVM: shufflevector <2 x float> [[TMP6]], <2 x float> {{[^,]+}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  // LLVM: call <4 x float> @llvm.fma.v4f32(
  return vfmaq_lane_f32(a, b, v, 1);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfma_laneq_f32(
// LLVM-SAME: <2 x float> noundef [[A:%.*]], <2 x float> noundef [[B:%.*]], <4 x float> noundef [[V:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfma_laneq_f32(
float32x2_t test_vfma_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x float> [[B]] to <2 x i32>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
  // LLVM: [[TMP5:%.*]] = bitcast <4 x i32> [[TMP2]] to <16 x i8>
  // LLVM: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x float>
  // LLVM: shufflevector <4 x float> [[TMP8]], <4 x float> {{[^,]+}}, <2 x i32> <i32 3, i32 3>
  // LLVM: call <2 x float> @llvm.fma.v2f32(
  return vfma_laneq_f32(a, b, v, 3);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmaq_laneq_f32(
// LLVM-SAME: <4 x float> noundef [[A:%.*]], <4 x float> noundef [[B:%.*]], <4 x float> noundef [[V:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmaq_laneq_f32(
float32x4_t test_vfmaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x float> [[B]] to <4 x i32>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
  // LLVM: [[TMP5:%.*]] = bitcast <4 x i32> [[TMP2]] to <16 x i8>
  // LLVM: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <4 x float>
  // LLVM: shufflevector <4 x float> [[TMP8]], <4 x float> {{[^,]+}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  // LLVM: call <4 x float> @llvm.fma.v4f32(
  return vfmaq_laneq_f32(a, b, v, 3);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfma_lane_f64(
// LLVM-SAME: <1 x double> noundef [[A:%.*]], <1 x double> noundef [[B:%.*]], <1 x double> noundef [[V:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfma_lane_f64(
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
  // LLVM: [[TMP1:%.*]] = bitcast <1 x double> [[B]] to i64
  // LLVM: [[TMP2:%.*]] = bitcast <1 x double> [[V]] to i64
  // LLVM: [[INS:%.*]] = insertelement <1 x i64> undef, i64 [[TMP2]], i32 0
  // LLVM: shufflevector <1 x double> {{[^,]+}}, <1 x double> {{[^,]+}}, <1 x i32> zeroinitializer
  // LLVM: call <1 x double> @llvm.fma.v1f64(
  return vfma_lane_f64(a, b, v, 0);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmaq_lane_f64(
// LLVM-SAME: <2 x double> noundef [[A:%.*]], <2 x double> noundef [[B:%.*]], <1 x double> noundef [[V:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmaq_lane_f64(
float64x2_t test_vfmaq_lane_f64(float64x2_t a, float64x2_t b, float64x1_t v) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
  // LLVM: [[TMP2:%.*]] = bitcast <1 x double> [[V]] to i64
  // LLVM: [[INS:%.*]] = insertelement <1 x i64> undef, i64 [[TMP2]], i32 0
  // LLVM: shufflevector <1 x double> {{[^,]+}}, <1 x double> {{[^,]+}}, <2 x i32> zeroinitializer
  // LLVM: call <2 x double> @llvm.fma.v2f64(
  return vfmaq_lane_f64(a, b, v, 0);
}

// PLAINLLVM-LABEL: define {{[^@]+}}@test_vfma_laneq_f64(
// PLAINLLVM-SAME: <1 x double> noundef [[A:%.*]], <1 x double> noundef [[B:%.*]], <2 x double> noundef [[V:%.*]]) #[[ATTR0]] {
// CIRLLVM-LABEL: define {{[^@]+}}@test_vfma_laneq_f64(
// CIRLLVM-SAME: <1 x double> noundef [[A:%.*]], <1 x double> noundef [[B:%.*]], <2 x double> noundef [[V:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfma_laneq_f64(
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // PLAINLLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
  // PLAINLLVM: [[TMP1:%.*]] = bitcast <1 x double> [[B]] to i64
  // PLAINLLVM: [[TMP2:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
  // PLAINLLVM: extractelement <2 x double>{{.*}}, i32 1
  // PLAINLLVM: call double @llvm.fma.f64(
  // CIRLLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
  // CIRLLVM: [[TMP1:%.*]] = bitcast <1 x double> [[B]] to i64
  // CIRLLVM: [[TMP2:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
  // CIRLLVM: shufflevector <2 x double> {{[^,]+}}, <2 x double> {{[^,]+}}, <1 x i32> <i32 1>
  // CIRLLVM: call <1 x double> @llvm.fma.v1f64(
  return vfma_laneq_f64(a, b, v, 1);
}

// LLVM-LABEL: define {{[^@]+}}@test_vfmaq_laneq_f64(
// LLVM-SAME: <2 x double> noundef [[A:%.*]], <2 x double> noundef [[B:%.*]], <2 x double> noundef [[V:%.*]]) #[[ATTR0]] {
// CIR-LABEL: @test_vfmaq_laneq_f64(
float64x2_t test_vfmaq_laneq_f64(float64x2_t a, float64x2_t b, float64x2_t v) {
  // CIR: cir.vec.shuffle
  // CIR: cir.call_llvm_intrinsic "fma"

  // LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
  // LLVM: [[TMP5:%.*]] = bitcast <2 x i64> [[TMP2]] to <16 x i8>
  // LLVM: [[TMP8:%.*]] = bitcast <16 x i8> [[TMP5]] to <2 x double>
  // LLVM: shufflevector <2 x double> [[TMP8]], <2 x double> {{[^,]+}}, <2 x i32> <i32 1, i32 1>
  // LLVM: call <2 x double> @llvm.fma.v2f64(
  return vfmaq_laneq_f64(a, b, v, 1);
}
