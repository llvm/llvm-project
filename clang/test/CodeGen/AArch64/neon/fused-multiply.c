// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefixes=ALL,CIR %}

// ALL: {{[Mm]}}odule

//=============================================================================
// NOTES
//
// This file contains tests that were originally located in:
//  * clang/test/CodeGen/AArch64/neon-intrinsics.c
//  * clang/test/CodeGen/AArch64/neon-2velem.c
//  * clang/test/CodeGen/AArch64/neon-scalar-x-indexed-elem.c
// The main difference is the use of RUN lines that enable ClangIR lowering.
// This file currently covers the f32/f64 fused multiply-accumulate and fused
// multiply-subtract wrappers, including the vfma/vfmaq/vfmas/vfmad and
// vfms/vfmsq/vfmss/vfmsd vector, lane, laneq, and scalar-lane forms.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#fused-multiply-accumulate
//
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.1.1.2.5 Fused multiply-accumulate, vector quad forms
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vfma_f32(
// CIR-LABEL: @vfma_f32(
float32x2_t test_vfma_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[B]] to <2 x i32>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <2 x float> [[C]] to <2 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <2 x i32> [[C_I]] to <8 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <2 x float>
// LLVM-NEXT: [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[B_CAST]], <2 x float> [[C_CAST]], <2 x float> [[A_CAST]])
// LLVM-NEXT: ret <2 x float> [[FMA]]
  return vfma_f32(a, b, c);
}

// LLVM-LABEL: @test_vfma_f64(
// CIR-LABEL: @vfma_f64(
float64x1_t test_vfma_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>) -> !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT: [[A_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[A_I]], i32 0
// LLVM-NEXT: [[B_I:%.*]] = bitcast <1 x double> [[B]] to i64
// LLVM-NEXT: [[B_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[B_I]], i32 0
// LLVM-NEXT: [[C_I:%.*]] = bitcast <1 x double> [[C]] to i64
// LLVM-NEXT: [[C_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[C_I]], i32 0
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <1 x i64> [[A_INSERT]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <1 x i64> [[B_INSERT]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <1 x i64> [[C_INSERT]] to <8 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <1 x double>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <1 x double>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <1 x double>
// LLVM-NEXT: [[FMA:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[B_CAST]], <1 x double> [[C_CAST]], <1 x double> [[A_CAST]])
// LLVM-NEXT: ret <1 x double> [[FMA]]
  return vfma_f64(a, b, c);
}

// LLVM-LABEL: @test_vfmaq_f32(
// CIR-LABEL: @vfmaq_f32(
float32x4_t test_vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[B]] to <4 x i32>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x float> [[C]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i32> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <4 x float>
// LLVM-NEXT: [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[B_CAST]], <4 x float> [[C_CAST]], <4 x float> [[A_CAST]])
// LLVM-NEXT: ret <4 x float> [[FMA]]
  return vfmaq_f32(a, b, c);
}

// LLVM-LABEL: @test_vfmaq_f64(
// CIR-LABEL: @vfmaq_f64(
float64x2_t test_vfmaq_f64(float64x2_t a, float64x2_t b, float64x2_t c) {
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <2 x double> [[C]] to <2 x i64>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <2 x i64> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <2 x double>
// LLVM-NEXT: [[FMA:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[B_CAST]], <2 x double> [[C_CAST]], <2 x double> [[A_CAST]])
// LLVM-NEXT: ret <2 x double> [[FMA]]
  return vfmaq_f64(a, b, c);
}

// ALL-LABEL: @test_vfma_lane_f32(
float32x2_t test_vfma_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[B]] to <2 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i32> [[V_I]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <2 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <2 x float> [[V_CAST]], <2 x float> {{.*}}, <2 x i32> <i32 1, i32 1>
// LLVM:      [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[B_CAST]], <2 x float> [[LANE]], <2 x float> [[A_CAST]])
// LLVM:      ret <2 x float> [[FMA]]
  return vfma_lane_f32(a, b, v, 1);
}

// ALL-LABEL: @test_vfma_lane_f64(
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<1 x !cir.double>) [#cir.int<0> : !s32i] : !cir.vector<1 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>) -> !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT: [[A_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[A_I]], i32 0
// LLVM-NEXT: [[B_I:%.*]] = bitcast <1 x double> [[B]] to i64
// LLVM-NEXT: [[B_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[B_I]], i32 0
// LLVM-NEXT: [[V_I:%.*]] = bitcast <1 x double> [[V]] to i64
// LLVM-NEXT: [[V_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[V_I]], i32 0
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <1 x i64> [[A_INSERT]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <1 x i64> [[B_INSERT]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <1 x i64> [[V_INSERT]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <1 x double>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <1 x double>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <1 x double>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <1 x double> [[V_CAST]], <1 x double> {{.*}}, <1 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[B_CAST]], <1 x double> [[LANE]], <1 x double> [[A_CAST]])
// LLVM:      ret <1 x double> [[FMA]]
  return vfma_lane_f64(a, b, v, 0);
}

// ALL-LABEL: @test_vfmaq_lane_f32(
float32x4_t test_vfmaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[B]] to <4 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i32> [[V_I]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <2 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <2 x float> [[V_CAST]], <2 x float> {{.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// LLVM:      [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[B_CAST]], <4 x float> [[LANE]], <4 x float> [[A_CAST]])
// LLVM:      ret <4 x float> [[FMA]]
  return vfmaq_lane_f32(a, b, v, 1);
}

// ALL-LABEL: @test_vfmaq_lane_f64(
float64x2_t test_vfmaq_lane_f64(float64x2_t a, float64x2_t b, float64x1_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<1 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <1 x double> [[V]] to i64
// LLVM-NEXT: [[V_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[V_I]], i32 0
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <1 x i64> [[V_INSERT]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <1 x double>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <1 x double> [[V_CAST]], <1 x double> {{.*}}, <2 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[B_CAST]], <2 x double> [[LANE]], <2 x double> [[A_CAST]])
// LLVM:      ret <2 x double> [[FMA]]
  return vfmaq_lane_f64(a, b, v, 0);
}

// ALL-LABEL: @test_vfma_laneq_f32(
float32x2_t test_vfma_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[B]] to <2 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <4 x i32> [[V_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM:      [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <4 x float>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <4 x float> [[V_CAST]], <4 x float> {{.*}}, <2 x i32> <i32 3, i32 3>
// LLVM:      [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[B_CAST]], <2 x float> [[A_CAST]])
// LLVM:      ret <2 x float> [[FMA]]
  return vfma_laneq_f32(a, b, v, 3);
}

// ALL-LABEL: @test_vfma_laneq_f64(
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b,
                                 float64x2_t v) {
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT: [[A_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[A_I]], i32 0
// LLVM-NEXT: [[B_I:%.*]] = bitcast <1 x double> [[B]] to i64
// LLVM-NEXT: [[B_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[B_I]], i32 0
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <1 x i64> [[A_INSERT]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <1 x i64> [[B_INSERT]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i64> [[V_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to double
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to double
// LLVM:      [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <2 x double>
// LLVM-NEXT: [[LANE:%.*]] = extractelement <2 x double> [[V_CAST]], i{{32|64}} 0
// LLVM:      [[FMA:%.*]] = call double @llvm.fma.f64(double [[B_CAST]], double [[LANE]], double [[A_CAST]])
// LLVM:      [[RESULT:%.*]] = bitcast double [[FMA]] to <1 x double>
// LLVM:      ret <1 x double> [[RESULT]]
  return vfma_laneq_f64(a, b, v, 0);
}

// ALL-LABEL: @test_vfma_laneq_f32_0(
float32x2_t test_vfma_laneq_f32_0(float32x2_t a, float32x2_t b,
                                   float32x4_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[B]] to <2 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <4 x i32> [[V_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM:      [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <4 x float>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <4 x float> [[V_CAST]], <4 x float> {{.*}}, <2 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[B_CAST]], <2 x float> [[A_CAST]])
// LLVM:      ret <2 x float> [[FMA]]
  return vfma_laneq_f32(a, b, v, 0);
}

// ALL-LABEL: @test_vfmaq_laneq_f32(
float32x4_t test_vfmaq_laneq_f32(float32x4_t a, float32x4_t b,
                                  float32x4_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[B]] to <4 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <4 x i32> [[V_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-NEXT: [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <4 x float>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <4 x float> [[V_CAST]], <4 x float> {{.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// LLVM-NEXT: [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[B_CAST]], <4 x float> [[A_CAST]])
// LLVM:      ret <4 x float> [[FMA]]
  return vfmaq_laneq_f32(a, b, v, 3);
}

// ALL-LABEL: @test_vfmaq_laneq_f64(
float64x2_t test_vfmaq_laneq_f64(float64x2_t a, float64x2_t b,
                                  float64x2_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i64> [[V_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM-NEXT: [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <2 x double>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <2 x double> [[V_CAST]], <2 x double> {{.*}}, <2 x i32> <i32 1, i32 1>
// LLVM-NEXT: [[FMA:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[B_CAST]], <2 x double> [[A_CAST]])
// LLVM:      ret <2 x double> [[FMA]]
  return vfmaq_laneq_f64(a, b, v, 1);
}

// ALL-LABEL: @test_vfmaq_laneq_f32_0(
float32x4_t test_vfmaq_laneq_f32_0(float32x4_t a, float32x4_t b,
                                    float32x4_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[B]] to <4 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <4 x i32> [[V_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-NEXT: [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <4 x float>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <4 x float> [[V_CAST]], <4 x float> {{.*}}, <4 x i32> zeroinitializer
// LLVM-NEXT: [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[B_CAST]], <4 x float> [[A_CAST]])
// LLVM:      ret <4 x float> [[FMA]]
  return vfmaq_laneq_f32(a, b, v, 0);
}

// ALL-LABEL: @test_vfmaq_laneq_f64_0(
float64x2_t test_vfmaq_laneq_f64_0(float64x2_t a, float64x2_t b,
                                    float64x2_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i64> [[V_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM-NEXT: [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <2 x double>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <2 x double> [[V_CAST]], <2 x double> {{.*}}, <2 x i32> zeroinitializer
// LLVM-NEXT: [[FMA:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[B_CAST]], <2 x double> [[A_CAST]])
// LLVM:      ret <2 x double> [[FMA]]
  return vfmaq_laneq_f64(a, b, v, 0);
}

// ALL-LABEL: @test_vfmas_lane_f32(
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.float, !cir.float, !cir.float) -> !cir.float

// LLVM-SAME: float {{.*}} [[A:%.*]], float {{.*}} [[B:%.*]], <2 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[LANE:%.*]] = extractelement <2 x float> [[C]], i{{32|64}} 1
// LLVM:      [[FMA:%.*]] = call float @llvm.fma.f32(float [[B]], float [[LANE]], float [[A]])
// LLVM:      ret float [[FMA]]
  return vfmas_lane_f32(a, b, c, 1);
}

// ALL-LABEL: @test_vfmas_laneq_f32(
float32_t test_vfmas_laneq_f32(float32_t a, float32_t b, float32x4_t c) {
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.float, !cir.float, !cir.float) -> !cir.float

// LLVM-SAME: float {{.*}} [[A:%.*]], float {{.*}} [[B:%.*]], <4 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[LANE:%.*]] = extractelement <4 x float> [[C]], i{{32|64}} 3
// LLVM:      [[FMA:%.*]] = call float @llvm.fma.f32(float [[B]], float [[LANE]], float [[A]])
// LLVM:      ret float [[FMA]]
  return vfmas_laneq_f32(a, b, c, 3);
}

// ALL-LABEL: @test_vfmad_lane_f64(
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<1 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double

// LLVM-SAME: double {{.*}} [[A:%.*]], double {{.*}} [[B:%.*]], <1 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[LANE:%.*]] = extractelement <1 x double> [[C]], i{{32|64}} 0
// LLVM:      [[FMA:%.*]] = call double @llvm.fma.f64(double [[B]], double [[LANE]], double [[A]])
// LLVM:      ret double [[FMA]]
  return vfmad_lane_f64(a, b, c, 0);
}

// ALL-LABEL: @test_vfmad_laneq_f64(
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double

// LLVM-SAME: double {{.*}} [[A:%.*]], double {{.*}} [[B:%.*]], <2 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[LANE:%.*]] = extractelement <2 x double> [[C]], i{{32|64}} 1
// LLVM:      [[FMA:%.*]] = call double @llvm.fma.f64(double [[B]], double [[LANE]], double [[A]])
// LLVM:      ret double [[FMA]]
  return vfmad_laneq_f64(a, b, c, 1);
}

//===------------------------------------------------------===//
// 2.1.1.2.5 Fused multiply-subtract forms
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vfms_f32(
// CIR-LABEL: @vfms_f32(
float32x2_t test_vfms_f32(float32x2_t a, float32x2_t b, float32x2_t c) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.float>
// CIR: cir.call @vfma_f32(%{{.*}}, [[NEG]], %{{.*}}) :

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[NEG:%.*]] = fneg <2 x float> [[B]]
// LLVM-NEXT: [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[NEG]] to <2 x i32>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <2 x float> [[C]] to <2 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <2 x i32> [[C_I]] to <8 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <2 x float>
// LLVM-NEXT: [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[B_CAST]], <2 x float> [[C_CAST]], <2 x float> [[A_CAST]])
// LLVM-NEXT: ret <2 x float> [[FMA]]
  return vfms_f32(a, b, c);
}

// LLVM-LABEL: @test_vfms_f64(
// CIR-LABEL: @vfms_f64(
float64x1_t test_vfms_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<1 x !cir.double>
// CIR: cir.call @vfma_f64(%{{.*}}, [[NEG]], %{{.*}}) :

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[NEG:%.*]] = fneg <1 x double> [[B]]
// LLVM-NEXT: [[A_I:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT: [[A_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[A_I]], i32 0
// LLVM-NEXT: [[B_I:%.*]] = bitcast <1 x double> [[NEG]] to i64
// LLVM-NEXT: [[B_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[B_I]], i32 0
// LLVM-NEXT: [[C_I:%.*]] = bitcast <1 x double> [[C]] to i64
// LLVM-NEXT: [[C_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[C_I]], i32 0
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <1 x i64> [[A_INSERT]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <1 x i64> [[B_INSERT]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <1 x i64> [[C_INSERT]] to <8 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <1 x double>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <1 x double>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <1 x double>
// LLVM-NEXT: [[FMA:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[B_CAST]], <1 x double> [[C_CAST]], <1 x double> [[A_CAST]])
// LLVM-NEXT: ret <1 x double> [[FMA]]
  return vfms_f64(a, b, c);
}

// LLVM-LABEL: @test_vfmsq_f32(
// CIR-LABEL: @vfmsq_f32(
float32x4_t test_vfmsq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.float>
// CIR: cir.call @vfmaq_f32(%{{.*}}, [[NEG]], %{{.*}}) :

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[NEG:%.*]] = fneg <4 x float> [[B]]
// LLVM-NEXT: [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[NEG]] to <4 x i32>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x float> [[C]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i32> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <4 x float>
// LLVM-NEXT: [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[B_CAST]], <4 x float> [[C_CAST]], <4 x float> [[A_CAST]])
// LLVM-NEXT: ret <4 x float> [[FMA]]
  return vfmsq_f32(a, b, c);
}

// LLVM-LABEL: @test_vfmsq_f64(
// CIR-LABEL: @vfmsq_f64(
float64x2_t test_vfmsq_f64(float64x2_t a, float64x2_t b, float64x2_t c) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.double>
// CIR: cir.call @vfmaq_f64(%{{.*}}, [[NEG]], %{{.*}}) :

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[NEG:%.*]] = fneg <2 x double> [[B]]
// LLVM-NEXT: [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x double> [[NEG]] to <2 x i64>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <2 x double> [[C]] to <2 x i64>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <2 x i64> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <2 x double>
// LLVM-NEXT: [[FMA:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[B_CAST]], <2 x double> [[C_CAST]], <2 x double> [[A_CAST]])
// LLVM-NEXT: ret <2 x double> [[FMA]]
  return vfmsq_f64(a, b, c);
}

// ALL-LABEL: @test_vfms_lane_f32(
float32x2_t test_vfms_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.float>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[NEG:%.*]] = fneg <2 x float> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[NEG]] to <2 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i32> [[V_I]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <2 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <2 x float> [[V_CAST]], <2 x float> {{.*}}, <2 x i32> <i32 1, i32 1>
// LLVM:      [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[B_CAST]], <2 x float> [[LANE]], <2 x float> [[A_CAST]])
// LLVM:      ret <2 x float> [[FMA]]
  return vfms_lane_f32(a, b, v, 1);
}

// ALL-LABEL: @test_vfms_lane_f64(
float64x1_t test_vfms_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<1 x !cir.double>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<1 x !cir.double>) [#cir.int<0> : !s32i] : !cir.vector<1 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>) -> !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT: [[A_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[A_I]], i32 0
// LLVM-NEXT: [[NEG:%.*]] = fneg <1 x double> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <1 x double> [[NEG]] to i64
// LLVM-NEXT: [[B_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[B_I]], i32 0
// LLVM-NEXT: [[V_I:%.*]] = bitcast <1 x double> [[V]] to i64
// LLVM-NEXT: [[V_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[V_I]], i32 0
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <1 x i64> [[A_INSERT]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <1 x i64> [[B_INSERT]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <1 x i64> [[V_INSERT]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <1 x double>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <1 x double>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <1 x double>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <1 x double> [[V_CAST]], <1 x double> {{.*}}, <1 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[B_CAST]], <1 x double> [[LANE]], <1 x double> [[A_CAST]])
// LLVM:      ret <1 x double> [[FMA]]
  return vfms_lane_f64(a, b, v, 0);
}

// ALL-LABEL: @test_vfms_lane_f32_0(
float32x2_t test_vfms_lane_f32_0(float32x2_t a, float32x2_t b,
                                  float32x2_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.float>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[NEG:%.*]] = fneg <2 x float> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[NEG]] to <2 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i32> [[V_I]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <2 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <2 x float> [[V_CAST]], <2 x float> {{.*}}, <2 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[B_CAST]], <2 x float> [[LANE]], <2 x float> [[A_CAST]])
// LLVM:      ret <2 x float> [[FMA]]
  return vfms_lane_f32(a, b, v, 0);
}

// ALL-LABEL: @test_vfmsq_lane_f32(
float32x4_t test_vfmsq_lane_f32(float32x4_t a, float32x4_t b,
                                 float32x2_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.float>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[NEG:%.*]] = fneg <4 x float> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[NEG]] to <4 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i32> [[V_I]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <2 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <2 x float> [[V_CAST]], <2 x float> {{.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// LLVM:      [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[B_CAST]], <4 x float> [[LANE]], <4 x float> [[A_CAST]])
// LLVM:      ret <4 x float> [[FMA]]
  return vfmsq_lane_f32(a, b, v, 1);
}

// ALL-LABEL: @test_vfmsq_lane_f64(
float64x2_t test_vfmsq_lane_f64(float64x2_t a, float64x2_t b,
                                 float64x1_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.double>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<1 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT: [[NEG:%.*]] = fneg <2 x double> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x double> [[NEG]] to <2 x i64>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <1 x double> [[V]] to i64
// LLVM-NEXT: [[V_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[V_I]], i32 0
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <1 x i64> [[V_INSERT]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <1 x double>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <1 x double> [[V_CAST]], <1 x double> {{.*}}, <2 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[B_CAST]], <2 x double> [[LANE]], <2 x double> [[A_CAST]])
// LLVM:      ret <2 x double> [[FMA]]
  return vfmsq_lane_f64(a, b, v, 0);
}

// ALL-LABEL: @test_vfmsq_lane_f32_0(
float32x4_t test_vfmsq_lane_f32_0(float32x4_t a, float32x4_t b,
                                   float32x2_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.float>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <2 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[NEG:%.*]] = fneg <4 x float> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[NEG]] to <4 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x float> [[V]] to <2 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i32> [[V_I]] to <8 x i8>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <8 x i8> [[V_BYTES]] to <2 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <2 x float> [[V_CAST]], <2 x float> {{.*}}, <4 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[B_CAST]], <4 x float> [[LANE]], <4 x float> [[A_CAST]])
// LLVM:      ret <4 x float> [[FMA]]
  return vfmsq_lane_f32(a, b, v, 0);
}

// ALL-LABEL: @test_vfms_laneq_f32(
float32x2_t test_vfms_laneq_f32(float32x2_t a, float32x2_t b,
                                 float32x4_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.float>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[NEG:%.*]] = fneg <2 x float> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[NEG]] to <2 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <4 x i32> [[V_I]] to <16 x i8>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <4 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <4 x float> [[V_CAST]], <4 x float> {{.*}}, <2 x i32> <i32 3, i32 3>
// LLVM:      [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[B_CAST]], <2 x float> [[A_CAST]])
// LLVM:      ret <2 x float> [[FMA]]
  return vfms_laneq_f32(a, b, v, 3);
}

// ALL-LABEL: @test_vfms_laneq_f64(
float64x1_t test_vfms_laneq_f64(float64x1_t a, float64x1_t b,
                                 float64x2_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<1 x !cir.double>
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT: [[A_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[A_I]], i32 0
// LLVM-NEXT: [[NEG:%.*]] = fneg <1 x double> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <1 x double> [[NEG]] to i64
// LLVM-NEXT: [[B_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[B_I]], i32 0
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <1 x i64> [[A_INSERT]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <1 x i64> [[B_INSERT]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i64> [[V_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to double
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to double
// LLVM-NEXT: [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <2 x double>
// LLVM-NEXT: [[LANE:%.*]] = extractelement <2 x double> [[V_CAST]], i{{32|64}} 0
// LLVM-NEXT: [[FMA:%.*]] = call double @llvm.fma.f64(double [[B_CAST]], double [[LANE]], double [[A_CAST]])
// LLVM-NEXT: [[RET:%.*]] = bitcast double [[FMA]] to <1 x double>
// LLVM:      ret <1 x double> [[RET]]
  return vfms_laneq_f64(a, b, v, 0);
}

// ALL-LABEL: @test_vfms_laneq_f32_0(
float32x2_t test_vfms_laneq_f32_0(float32x2_t a, float32x2_t b,
                                   float32x4_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.float>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT: [[NEG:%.*]] = fneg <2 x float> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x float> [[NEG]] to <2 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i32> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i32> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <4 x i32> [[V_I]] to <16 x i8>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <2 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <2 x float>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <4 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <4 x float> [[V_CAST]], <4 x float> {{.*}}, <2 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[B_CAST]], <2 x float> [[A_CAST]])
// LLVM:      ret <2 x float> [[FMA]]
  return vfms_laneq_f32(a, b, v, 0);
}

// ALL-LABEL: @test_vfmsq_laneq_f32(
float32x4_t test_vfmsq_laneq_f32(float32x4_t a, float32x4_t b,
                                  float32x4_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.float>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[NEG:%.*]] = fneg <4 x float> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[NEG]] to <4 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <4 x i32> [[V_I]] to <16 x i8>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <4 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <4 x float> [[V_CAST]], <4 x float> {{.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// LLVM:      [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[B_CAST]], <4 x float> [[A_CAST]])
// LLVM:      ret <4 x float> [[FMA]]
  return vfmsq_laneq_f32(a, b, v, 3);
}

// ALL-LABEL: @test_vfmsq_laneq_f64(
float64x2_t test_vfmsq_laneq_f64(float64x2_t a, float64x2_t b,
                                  float64x2_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.double>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT: [[NEG:%.*]] = fneg <2 x double> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x double> [[NEG]] to <2 x i64>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i64> [[V_I]] to <16 x i8>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <2 x double>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <2 x double> [[V_CAST]], <2 x double> {{.*}}, <2 x i32> <i32 1, i32 1>
// LLVM:      [[FMA:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[B_CAST]], <2 x double> [[A_CAST]])
// LLVM:      ret <2 x double> [[FMA]]
  return vfmsq_laneq_f64(a, b, v, 1);
}

// ALL-LABEL: @test_vfmsq_laneq_f32_0(
float32x4_t test_vfmsq_laneq_f32_0(float32x4_t a, float32x4_t b,
                                    float32x4_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.float>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]], <4 x float> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT: [[NEG:%.*]] = fneg <4 x float> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x float> [[NEG]] to <4 x i32>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <4 x float> [[V]] to <4 x i32>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i32> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i32> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <4 x i32> [[V_I]] to <16 x i8>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <4 x float>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <4 x float>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <4 x float>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <4 x float> [[V_CAST]], <4 x float> {{.*}}, <4 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[B_CAST]], <4 x float> [[A_CAST]])
// LLVM:      ret <4 x float> [[FMA]]
  return vfmsq_laneq_f32(a, b, v, 0);
}

// ALL-LABEL: @test_vfmsq_laneq_f64_0(
float64x2_t test_vfmsq_laneq_f64_0(float64x2_t a, float64x2_t b,
                                    float64x2_t v) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<2 x !cir.double>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT: [[NEG:%.*]] = fneg <2 x double> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <2 x double> [[NEG]] to <2 x i64>
// LLVM-NEXT: [[V_I:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[V_BYTES:%.*]] = bitcast <2 x i64> [[V_I]] to <16 x i8>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM-DAG:  [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <2 x double>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <2 x double> [[V_CAST]], <2 x double> {{.*}}, <2 x i32> zeroinitializer
// LLVM:      [[FMA:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[B_CAST]], <2 x double> [[A_CAST]])
// LLVM:      ret <2 x double> [[FMA]]
  return vfmsq_laneq_f64(a, b, v, 0);
}

// ALL-LABEL: @test_vfmss_lane_f32(
float32_t test_vfmss_lane_f32(float32_t a, float32_t b, float32x2_t c) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.float
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<2 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.float, !cir.float, !cir.float) -> !cir.float

// LLVM-SAME: float {{.*}} [[A:%.*]], float {{.*}} [[B:%.*]], <2 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[NEG:%.*]] = fneg float [[B]]
// LLVM:      [[LANE:%.*]] = extractelement <2 x float> [[C]], i{{32|64}} 1
// LLVM:      [[FMA:%.*]] = call float @llvm.fma.f32(float [[NEG]], float [[LANE]], float [[A]])
// LLVM:      ret float [[FMA]]
  return vfmss_lane_f32(a, b, c, 1);
}

// ALL-LABEL: @test_vfmss_laneq_f32(
float32_t test_vfmss_laneq_f32(float32_t a, float32_t b, float32x4_t c) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.float
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !cir.float>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.float, !cir.float, !cir.float) -> !cir.float

// LLVM-SAME: float {{.*}} [[A:%.*]], float {{.*}} [[B:%.*]], <4 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[NEG:%.*]] = fneg float [[B]]
// LLVM:      [[LANE:%.*]] = extractelement <4 x float> [[C]], i{{32|64}} 3
// LLVM:      [[FMA:%.*]] = call float @llvm.fma.f32(float [[NEG]], float [[LANE]], float [[A]])
// LLVM:      ret float [[FMA]]
  return vfmss_laneq_f32(a, b, c, 3);
}

// ALL-LABEL: @test_vfmsd_lane_f64(
float64_t test_vfmsd_lane_f64(float64_t a, float64_t b, float64x1_t c) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.double
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<1 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double

// LLVM-SAME: double {{.*}} [[A:%.*]], double {{.*}} [[B:%.*]], <1 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[NEG:%.*]] = fneg double [[B]]
// LLVM:      [[LANE:%.*]] = extractelement <1 x double> [[C]], i{{32|64}} 0
// LLVM:      [[FMA:%.*]] = call double @llvm.fma.f64(double [[NEG]], double [[LANE]], double [[A]])
// LLVM:      ret double [[FMA]]
  return vfmsd_lane_f64(a, b, c, 0);
}

// ALL-LABEL: @test_vfmsd_laneq_f64(
float64_t test_vfmsd_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
// CIR: [[NEG:%.*]] = cir.fneg %{{.*}} : !cir.double
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<2 x !cir.double>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.double, !cir.double, !cir.double) -> !cir.double

// LLVM-SAME: double {{.*}} [[A:%.*]], double {{.*}} [[B:%.*]], <2 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[NEG:%.*]] = fneg double [[B]]
// LLVM:      [[LANE:%.*]] = extractelement <2 x double> [[C]], i{{32|64}} 1
// LLVM:      [[FMA:%.*]] = call double @llvm.fma.f64(double [[NEG]], double [[LANE]], double [[A]])
// LLVM:      ret double [[FMA]]
  return vfmsd_laneq_f64(a, b, c, 1);
}
