// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.1.1.9. Rounding
//https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#rounding
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vrnd_f32(
// CIR-LABEL: @vrnd_f32(
float32x2_t test_vrnd_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.trunc [[CAST]] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRND_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRND1_I:%.*]] = call <2 x float> @llvm.trunc.v2f32(<2 x float> [[VRND_I]])
// LLVM: ret <2 x float> [[VRND1_I]]
  return vrnd_f32(a);
}

// LLVM-LABEL: @test_vrndq_f32(
// CIR-LABEL: @vrndq_f32(
float32x4_t test_vrndq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.trunc [[CAST]] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRND_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRND1_I:%.*]] = call <4 x float> @llvm.trunc.v4f32(<4 x float> [[VRND_I]])
// LLVM: ret <4 x float> [[VRND1_I]]
  return vrndq_f32(a);
}

// LLVM-LABEL: @test_vrnd_f64(
// CIR-LABEL: @vrnd_f64(
float64x1_t test_vrnd_f64(float64x1_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR: cir.trunc [[CAST]] : !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i64 0
// LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[TMP1]] to <8 x i8>
// LLVM: [[VRND_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// LLVM: [[VRND1_I:%.*]] = call <1 x double> @llvm.trunc.v1f64(<1 x double> [[VRND_I]])
// LLVM: ret <1 x double> [[VRND1_I]]
  return vrnd_f64(a);
}

// LLVM-LABEL: @test_vrndq_f64(
// CIR-LABEL: @vrndq_f64(
float64x2_t test_vrndq_f64(float64x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR: cir.trunc [[CAST]] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM: [[VRND_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// LLVM: [[VRND1_I:%.*]] = call <2 x double> @llvm.trunc.v2f64(<2 x double> [[VRND_I]])
// LLVM: ret <2 x double> [[VRND1_I]]
  return vrndq_f64(a);
}

// LLVM-LABEL: @test_vrndn_f32(
// CIR-LABEL: @vrndn_f32(
float32x2_t test_vrndn_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.roundeven [[CAST]] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRNDN_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRNDN1_I:%.*]] = call <2 x float> @llvm.roundeven.v2f32(<2 x float> [[VRNDN_I]])
// LLVM: ret <2 x float> [[VRNDN1_I]]
  return vrndn_f32(a);
}

// LLVM-LABEL: @test_vrndnq_f32(
// CIR-LABEL: @vrndnq_f32(
float32x4_t test_vrndnq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.roundeven [[CAST]] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDN_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRNDN1_I:%.*]] = call <4 x float> @llvm.roundeven.v4f32(<4 x float> [[VRNDN_I]])
// LLVM: ret <4 x float> [[VRNDN1_I]]
  return vrndnq_f32(a);
}

// LLVM-LABEL: @test_vrndn_f64(
// CIR-LABEL: @vrndn_f64(
float64x1_t test_vrndn_f64(float64x1_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR: cir.roundeven [[CAST]] : !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i64 0
// LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[TMP1]] to <8 x i8>
// LLVM: [[VRNDN_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// LLVM: [[VRNDN1_I:%.*]] = call <1 x double> @llvm.roundeven.v1f64(<1 x double> [[VRNDN_I]])
// LLVM: ret <1 x double> [[VRNDN1_I]]
  return vrndn_f64(a);
}

// LLVM-LABEL: @test_vrndnq_f64(
// CIR-LABEL: @vrndnq_f64(
float64x2_t test_vrndnq_f64(float64x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR: cir.roundeven [[CAST]] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDN_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// LLVM: [[VRNDN1_I:%.*]] = call <2 x double> @llvm.roundeven.v2f64(<2 x double> [[VRNDN_I]])
// LLVM: ret <2 x double> [[VRNDN1_I]]
  return vrndnq_f64(a);
}

// LLVM-LABEL: @test_vrndns_f32(
// CIR-LABEL: @vrndns_f32(
float32_t test_vrndns_f32(float32_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: cir.roundeven [[LOAD]] : !cir.float

// LLVM-SAME: float {{.*}} [[A:%.*]])
// LLVM: [[VRNDN_I:%.*]] = call float @llvm.roundeven.f32(float [[A]])
// LLVM: ret float [[VRNDN_I]]
  return vrndns_f32(a);
}

// LLVM-LABEL: @test_vrndm_f32(
// CIR-LABEL: @vrndm_f32(
float32x2_t test_vrndm_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.floor [[CAST]] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRNDM_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRNDM1_I:%.*]] = call <2 x float> @llvm.floor.v2f32(<2 x float> [[VRNDM_I]])
// LLVM: ret <2 x float> [[VRNDM1_I]]
  return vrndm_f32(a);
}

// LLVM-LABEL: @test_vrndmq_f32(
// CIR-LABEL: @vrndmq_f32(
float32x4_t test_vrndmq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.floor [[CAST]] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDM_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRNDM1_I:%.*]] = call <4 x float> @llvm.floor.v4f32(<4 x float> [[VRNDM_I]])
// LLVM: ret <4 x float> [[VRNDM1_I]]
  return vrndmq_f32(a);
}

// LLVM-LABEL: @test_vrndm_f64(
// CIR-LABEL: @vrndm_f64(
float64x1_t test_vrndm_f64(float64x1_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR: cir.floor [[CAST]] : !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i64 0
// LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[TMP1]] to <8 x i8>
// LLVM: [[VRNDM_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// LLVM: [[VRNDM1_I:%.*]] = call <1 x double> @llvm.floor.v1f64(<1 x double> [[VRNDM_I]])
// LLVM: ret <1 x double> [[VRNDM1_I]]
  return vrndm_f64(a);
}

// LLVM-LABEL: @test_vrndmq_f64(
// CIR-LABEL: @vrndmq_f64(
float64x2_t test_vrndmq_f64(float64x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR: cir.floor [[CAST]] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDM_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// LLVM: [[VRNDM1_I:%.*]] = call <2 x double> @llvm.floor.v2f64(<2 x double> [[VRNDM_I]])
// LLVM: ret <2 x double> [[VRNDM1_I]]
  return vrndmq_f64(a);
}

// LLVM-LABEL: @test_vrndp_f32(
// CIR-LABEL: @vrndp_f32(
float32x2_t test_vrndp_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.ceil [[CAST]] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRNDP_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRNDP1_I:%.*]] = call <2 x float> @llvm.ceil.v2f32(<2 x float> [[VRNDP_I]])
// LLVM: ret <2 x float> [[VRNDP1_I]]
  return vrndp_f32(a);
}

// LLVM-LABEL: @test_vrndpq_f32(
// CIR-LABEL: @vrndpq_f32(
float32x4_t test_vrndpq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.ceil [[CAST]] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDP_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRNDP1_I:%.*]] = call <4 x float> @llvm.ceil.v4f32(<4 x float> [[VRNDP_I]])
// LLVM: ret <4 x float> [[VRNDP1_I]]
  return vrndpq_f32(a);
}

// LLVM-LABEL: @test_vrndp_f64(
// CIR-LABEL: @vrndp_f64(
float64x1_t test_vrndp_f64(float64x1_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR: cir.ceil [[CAST]] : !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i64 0
// LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[TMP1]] to <8 x i8>
// LLVM: [[VRNDP_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// LLVM: [[VRNDP1_I:%.*]] = call <1 x double> @llvm.ceil.v1f64(<1 x double> [[VRNDP_I]])
// LLVM: ret <1 x double> [[VRNDP1_I]]
  return vrndp_f64(a);
}

// LLVM-LABEL: @test_vrndpq_f64(
// CIR-LABEL: @vrndpq_f64(
float64x2_t test_vrndpq_f64(float64x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR: cir.ceil [[CAST]] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDP_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// LLVM: [[VRNDP1_I:%.*]] = call <2 x double> @llvm.ceil.v2f64(<2 x double> [[VRNDP_I]])
// LLVM: ret <2 x double> [[VRNDP1_I]]
  return vrndpq_f64(a);
}

// LLVM-LABEL: @test_vrnda_f32(
// CIR-LABEL: @vrnda_f32(
float32x2_t test_vrnda_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.round [[CAST]] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRNDA_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRNDA1_I:%.*]] = call <2 x float> @llvm.round.v2f32(<2 x float> [[VRNDA_I]])
// LLVM: ret <2 x float> [[VRNDA1_I]]
  return vrnda_f32(a);
}

// LLVM-LABEL: @test_vrndaq_f32(
// CIR-LABEL: @vrndaq_f32(
float32x4_t test_vrndaq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.round [[CAST]] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDA_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRNDA1_I:%.*]] = call <4 x float> @llvm.round.v4f32(<4 x float> [[VRNDA_I]])
// LLVM: ret <4 x float> [[VRNDA1_I]]
  return vrndaq_f32(a);
}

// LLVM-LABEL: @test_vrnda_f64(
// CIR-LABEL: @vrnda_f64(
float64x1_t test_vrnda_f64(float64x1_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR: cir.round [[CAST]] : !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i64 0
// LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[TMP1]] to <8 x i8>
// LLVM: [[VRNDA_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// LLVM: [[VRNDA1_I:%.*]] = call <1 x double> @llvm.round.v1f64(<1 x double> [[VRNDA_I]])
// LLVM: ret <1 x double> [[VRNDA1_I]]
  return vrnda_f64(a);
}

// LLVM-LABEL: @test_vrndaq_f64(
// CIR-LABEL: @vrndaq_f64(
float64x2_t test_vrndaq_f64(float64x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR: cir.round [[CAST]] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDA_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// LLVM: [[VRNDA1_I:%.*]] = call <2 x double> @llvm.round.v2f64(<2 x double> [[VRNDA_I]])
// LLVM: ret <2 x double> [[VRNDA1_I]]
  return vrndaq_f64(a);
}

// LLVM-LABEL: @test_vrndi_f32(
// CIR-LABEL: @vrndi_f32(
float32x2_t test_vrndi_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.nearbyint [[CAST]] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRNDI_V_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRNDI_V1_I:%.*]] = call <2 x float> @llvm.nearbyint.v2f32(<2 x float> [[VRNDI_V_I]])
// LLVM: ret <2 x float> [[VRNDI_V1_I]]
  return vrndi_f32(a);
}

// LLVM-LABEL: @test_vrndiq_f32(
// CIR-LABEL: @vrndiq_f32(
float32x4_t test_vrndiq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.nearbyint [[CAST]] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDIQ_V_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRNDIQ_V1_I:%.*]] = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> [[VRNDIQ_V_I]])
// LLVM: ret <4 x float> [[VRNDIQ_V1_I]]
  return vrndiq_f32(a);
}

// LLVM-LABEL: @test_vrndi_f64(
// CIR-LABEL: @vrndi_f64(
float64x1_t test_vrndi_f64(float64x1_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR: cir.nearbyint [[CAST]] : !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i64 0
// LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[TMP1]] to <8 x i8>
// LLVM: [[VRNDI_V_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// LLVM: [[VRNDI_V1_I:%.*]] = call <1 x double> @llvm.nearbyint.v1f64(<1 x double> [[VRNDI_V_I]])
// LLVM: ret <1 x double> [[VRNDI_V1_I]]
  return vrndi_f64(a);
}

// LLVM-LABEL: @test_vrndiq_f64(
// CIR-LABEL: @vrndiq_f64(
float64x2_t test_vrndiq_f64(float64x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR: cir.nearbyint [[CAST]] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDI_V_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// LLVM: [[VRNDI_V1_I:%.*]] = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> [[VRNDI_V_I]])
// LLVM: ret <2 x double> [[VRNDI_V1_I]]
  return vrndiq_f64(a);
}

// LLVM-LABEL: @test_vrndx_f32(
// CIR-LABEL: @vrndx_f32(
float32x2_t test_vrndx_f32(float32x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR: cir.rint [[CAST]] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM: [[VRNDX_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM: [[VRNDX1_I:%.*]] = call <2 x float> @llvm.rint.v2f32(<2 x float> [[VRNDX_I]])
// LLVM: ret <2 x float> [[VRNDX1_I]]
  return vrndx_f32(a);
}

// LLVM-LABEL: @test_vrndxq_f32(
// CIR-LABEL: @vrndxq_f32(
float32x4_t test_vrndxq_f32(float32x4_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR: cir.rint [[CAST]] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDX_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM: [[VRNDX1_I:%.*]] = call <4 x float> @llvm.rint.v4f32(<4 x float> [[VRNDX_I]])
// LLVM: ret <4 x float> [[VRNDX1_I]]
  return vrndxq_f32(a);
}

// LLVM-LABEL: @test_vrndx_f64(
// CIR-LABEL: @vrndx_f64(
float64x1_t test_vrndx_f64(float64x1_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR: cir.rint [[CAST]] : !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[TMP1:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i64 0
// LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[TMP1]] to <8 x i8>
// LLVM: [[VRNDX_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// LLVM: [[VRNDX1_I:%.*]] = call <1 x double> @llvm.rint.v1f64(<1 x double> [[VRNDX_I]])
// LLVM: ret <1 x double> [[VRNDX1_I]]
  return vrndx_f64(a);
}

// LLVM-LABEL: @test_vrndxq_f64(
// CIR-LABEL: @vrndxq_f64(
float64x2_t test_vrndxq_f64(float64x2_t a) {
// CIR: [[LOAD:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[CAST:%.*]] = cir.cast bitcast [[LOAD]] : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR: cir.rint [[CAST]] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM: [[VRNDX_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// LLVM: [[VRNDX1_I:%.*]] = call <2 x double> @llvm.rint.v2f64(<2 x double> [[VRNDX_I]])
// LLVM: ret <2 x double> [[VRNDX1_I]]
  return vrndxq_f64(a);
}
