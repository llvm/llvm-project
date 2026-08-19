// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon -target-feature +fullfp16           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefixes=ALL,CIR %}

// ALL: {{[Mm]}}odule

//=============================================================================
// NOTES
//
// This file contains fullfp16 tests that were originally located in:
//  * clang/test/CodeGen/AArch64/v8.2a-neon-intrinsics.c
// The main difference is the use of RUN lines that enable ClangIR lowering.
// This file currently covers the f16 ACLE wrappers from the
// fused-multiply-accumulate section. Several wrappers are header-level forms
// over the existing lowered vfma/vfmah builtins, such as vfma_n_f16,
// vfms_f16, vfms_lane_f16, vfms_n_f16, and vfmsh_lane_f16.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#fused-multiply-accumulate-2
//
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.6.1.9.3 Fused multiply-accumulate, vector quad forms
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vfma_f16(
// CIR-LABEL: @vfma_f16(
float16x4_t test_vfma_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
// CIR: cir.fma %{{.*}}, %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-NEXT: [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[B_CAST]], <4 x half> [[C_CAST]], <4 x half> [[A_CAST]])
// LLVM-NEXT: ret <4 x half> [[FMA]]
  return vfma_f16(a, b, c);
}

// LLVM-LABEL: @test_vfmaq_f16(
// CIR-LABEL: @vfmaq_f16(
float16x8_t test_vfmaq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
// CIR: cir.fma %{{.*}}, %{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM-NEXT: [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[B_CAST]], <8 x half> [[C_CAST]], <8 x half> [[A_CAST]])
// LLVM-NEXT: ret <8 x half> [[FMA]]
  return vfmaq_f16(a, b, c);
}

// LLVM-LABEL: @test_vfms_f16(
// CIR-LABEL: @vfms_f16(
float16x4_t test_vfms_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
// CIR: [[FNEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.f16>
// CIR: cir.call @vfma_f16(%{{.*}}, [[FNEG]], %{{.*}}) :

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[FNEG:%.*]] = fneg <4 x half> [[B]]
// LLVM-NEXT: [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x half> [[FNEG]] to <4 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-NEXT: [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[B_CAST]], <4 x half> [[C_CAST]], <4 x half> [[A_CAST]])
// LLVM-NEXT: ret <4 x half> [[FMA]]
  return vfms_f16(a, b, c);
}

// LLVM-LABEL: @test_vfmsq_f16(
// CIR-LABEL: @vfmsq_f16(
float16x8_t test_vfmsq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
// CIR: [[FNEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<8 x !cir.f16>
// CIR: cir.call @vfmaq_f16(%{{.*}}, [[FNEG]], %{{.*}}) :

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[FNEG:%.*]] = fneg <8 x half> [[B]]
// LLVM-NEXT: [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[FNEG]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM-NEXT: [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[B_CAST]], <8 x half> [[C_CAST]], <8 x half> [[A_CAST]])
// LLVM-NEXT: ret <8 x half> [[FMA]]
  return vfmsq_f16(a, b, c);
}

// ALL-LABEL: @test_vfma_lane_f16(
float16x4_t test_vfma_lane_f16(float16x4_t a, float16x4_t b,
                                float16x4_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.vector<4 x !cir.f16>

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM-DAG:  [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <4 x half> [[C_CAST]], <4 x half> {{.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// LLVM:      [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[B_CAST]], <4 x half> [[LANE]], <4 x half> [[A_CAST]])
// LLVM:      ret <4 x half> [[FMA]]
  return vfma_lane_f16(a, b, c, 3);
}

// ALL-LABEL: @test_vfmaq_lane_f16(
float16x8_t test_vfmaq_lane_f16(float16x8_t a, float16x8_t b,
                                 float16x4_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<8 x !cir.f16>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.vector<8 x !cir.f16>

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM-DAG:  [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <4 x half> [[C_CAST]], <4 x half> {{.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// LLVM:      [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[B_CAST]], <8 x half> [[LANE]], <8 x half> [[A_CAST]])
// LLVM:      ret <8 x half> [[FMA]]
  return vfmaq_lane_f16(a, b, c, 3);
}

// ALL-LABEL: @test_vfma_laneq_f16(
float16x4_t test_vfma_laneq_f16(float16x4_t a, float16x4_t b,
                                 float16x8_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma [[LANE]], %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM:      [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <8 x half> [[C_CAST]], <8 x half> {{.*}}, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// LLVM:      [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[B_CAST]], <4 x half> [[A_CAST]])
// LLVM:      ret <4 x half> [[FMA]]
  return vfma_laneq_f16(a, b, c, 7);
}

// ALL-LABEL: @test_vfmaq_laneq_f16(
float16x8_t test_vfmaq_laneq_f16(float16x8_t a, float16x8_t b,
                                  float16x8_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.f16>
// CIR: cir.fma [[LANE]], %{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <8 x half> [[C_CAST]], <8 x half> {{.*}}, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// LLVM-NEXT: [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[B_CAST]], <8 x half> [[A_CAST]])
// LLVM:      ret <8 x half> [[FMA]]
  return vfmaq_laneq_f16(a, b, c, 7);
}

// ALL-LABEL: @test_vfma_n_f16(
float16x4_t test_vfma_n_f16(float16x4_t a, float16x4_t b, float16_t c) {
// CIR: {{%.*}} = cir.vec.create({{.*}}) : !cir.vector<4 x !cir.f16>
// CIR: cir.call @vfma_f16(%{{.*}}, %{{.*}}, %{{.*}}) :

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], half {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[SPLAT:%.*]] = insertelement <4 x half> {{.*}}, half [[C]], i{{32|64}} 3
// LLVM-NEXT: [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[SPLAT]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-NEXT: [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[B_CAST]], <4 x half> [[C_CAST]], <4 x half> [[A_CAST]])
// LLVM:      ret <4 x half> [[FMA]]
  return vfma_n_f16(a, b, c);
}

// ALL-LABEL: @test_vfmaq_n_f16(
float16x8_t test_vfmaq_n_f16(float16x8_t a, float16x8_t b, float16_t c) {
// CIR: {{%.*}} = cir.vec.create({{.*}}) : !cir.vector<8 x !cir.f16>
// CIR: cir.call @vfmaq_f16(%{{.*}}, %{{.*}}, %{{.*}}) :

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], half {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[SPLAT:%.*]] = insertelement <8 x half> {{.*}}, half [[C]], i{{32|64}} 7
// LLVM-NEXT: [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <8 x half> [[SPLAT]] to <8 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM-NEXT: [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[B_CAST]], <8 x half> [[C_CAST]], <8 x half> [[A_CAST]])
// LLVM:      ret <8 x half> [[FMA]]
  return vfmaq_n_f16(a, b, c);
}

// ALL-LABEL: @test_vfmah_lane_f16(
float16_t test_vfmah_lane_f16(float16_t a, float16_t b, float16x4_t c) {
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[LANE:%.*]] = extractelement <4 x half> [[C]], i{{32|64}} 3
// LLVM:      [[FMA:%.*]] = call half @llvm.fma.f16(half [[B]], half [[LANE]], half [[A]])
// LLVM:      ret half [[FMA]]
  return vfmah_lane_f16(a, b, c, 3);
}

// ALL-LABEL: @test_vfmah_laneq_f16(
float16_t test_vfmah_laneq_f16(float16_t a, float16_t b, float16x8_t c) {
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<8 x !cir.f16>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[LANE:%.*]] = extractelement <8 x half> [[C]], i{{32|64}} 7
// LLVM:      [[FMA:%.*]] = call half @llvm.fma.f16(half [[B]], half [[LANE]], half [[A]])
// LLVM:      ret half [[FMA]]
  return vfmah_laneq_f16(a, b, c, 7);
}

// ALL-LABEL: @test_vfms_lane_f16(
float16x4_t test_vfms_lane_f16(float16x4_t a, float16x4_t b,
                                float16x4_t c) {
// CIR: [[FNEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.f16>
// CIR: cir.store align(8) [[FNEG]], [[NEG_SLOT:%.*]] : !cir.vector<4 x !cir.f16>, !cir.ptr<!cir.vector<4 x !cir.f16>>
// CIR: [[NEG_PTR:%.*]] = cir.cast bitcast [[NEG_SLOT]] : !cir.ptr<!cir.vector<4 x !cir.f16>> -> !cir.ptr<!cir.vector<8 x !s8i>>
// CIR: [[NEG_BYTES:%.*]] = cir.load align(8) [[NEG_PTR]] : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[NEG_CAST:%.*]] = cir.cast bitcast [[NEG_BYTES]] : !cir.vector<8 x !s8i> -> !cir.vector<4 x !cir.f16>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma [[NEG_CAST]], [[LANE]], %{{.*}} : !cir.vector<4 x !cir.f16>

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT: [[FNEG:%.*]] = fneg <4 x half> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x half> [[FNEG]] to <4 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM-DAG:  [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <4 x half> [[C_CAST]], <4 x half> {{.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// LLVM:      [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[B_CAST]], <4 x half> [[LANE]], <4 x half> [[A_CAST]])
// LLVM:      ret <4 x half> [[FMA]]
  return vfms_lane_f16(a, b, c, 3);
}

// ALL-LABEL: @test_vfmsq_lane_f16(
float16x8_t test_vfmsq_lane_f16(float16x8_t a, float16x8_t b,
                                 float16x4_t c) {
// CIR: [[FNEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<8 x !cir.f16>
// CIR: cir.store align(16) [[FNEG]], [[NEG_SLOT:%.*]] : !cir.vector<8 x !cir.f16>, !cir.ptr<!cir.vector<8 x !cir.f16>>
// CIR: [[NEG_PTR:%.*]] = cir.cast bitcast [[NEG_SLOT]] : !cir.ptr<!cir.vector<8 x !cir.f16>> -> !cir.ptr<!cir.vector<16 x !s8i>>
// CIR: [[NEG_BYTES:%.*]] = cir.load align(16) [[NEG_PTR]] : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[NEG_CAST:%.*]] = cir.cast bitcast [[NEG_BYTES]] : !cir.vector<16 x !s8i> -> !cir.vector<8 x !cir.f16>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<8 x !cir.f16>
// CIR: cir.fma [[NEG_CAST]], [[LANE]], %{{.*}} : !cir.vector<8 x !cir.f16>

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[FNEG:%.*]] = fneg <8 x half> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[FNEG]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM-DAG:  [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-DAG:  [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM-DAG:  [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM-DAG:  [[LANE:%.*]] = shufflevector <4 x half> [[C_CAST]], <4 x half> {{.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// LLVM:      [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[B_CAST]], <8 x half> [[LANE]], <8 x half> [[A_CAST]])
// LLVM:      ret <8 x half> [[FMA]]
  return vfmsq_lane_f16(a, b, c, 3);
}

// ALL-LABEL: @test_vfms_laneq_f16(
float16x4_t test_vfms_laneq_f16(float16x4_t a, float16x4_t b,
                                 float16x8_t c) {
// CIR: [[FNEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.f16>
// CIR: cir.store align(8) [[FNEG]], [[NEG_SLOT:%.*]] : !cir.vector<4 x !cir.f16>, !cir.ptr<!cir.vector<4 x !cir.f16>>
// CIR: [[NEG_PTR:%.*]] = cir.cast bitcast [[NEG_SLOT]] : !cir.ptr<!cir.vector<4 x !cir.f16>> -> !cir.ptr<!cir.vector<8 x !s8i>>
// CIR: [[NEG_BYTES:%.*]] = cir.load align(8) [[NEG_PTR]] : !cir.ptr<!cir.vector<8 x !s8i>>, !cir.vector<8 x !s8i>
// CIR: [[NEG_CAST:%.*]] = cir.cast bitcast [[NEG_BYTES]] : !cir.vector<8 x !s8i> -> !cir.vector<4 x !cir.f16>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma [[LANE]], [[NEG_CAST]], %{{.*}} : !cir.vector<4 x !cir.f16>

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT: [[FNEG:%.*]] = fneg <4 x half> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x half> [[FNEG]] to <4 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <8 x half> [[C_CAST]], <8 x half> {{.*}}, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// LLVM-NEXT: [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[B_CAST]], <4 x half> [[A_CAST]])
// LLVM:      ret <4 x half> [[FMA]]
  return vfms_laneq_f16(a, b, c, 7);
}

// ALL-LABEL: @test_vfmsq_laneq_f16(
float16x8_t test_vfmsq_laneq_f16(float16x8_t a, float16x8_t b,
                                  float16x8_t c) {
// CIR: [[FNEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<8 x !cir.f16>
// CIR: cir.store align(16) [[FNEG]], [[NEG_SLOT:%.*]] : !cir.vector<8 x !cir.f16>, !cir.ptr<!cir.vector<8 x !cir.f16>>
// CIR: [[NEG_PTR:%.*]] = cir.cast bitcast [[NEG_SLOT]] : !cir.ptr<!cir.vector<8 x !cir.f16>> -> !cir.ptr<!cir.vector<16 x !s8i>>
// CIR: [[NEG_BYTES:%.*]] = cir.load align(16) [[NEG_PTR]] : !cir.ptr<!cir.vector<16 x !s8i>>, !cir.vector<16 x !s8i>
// CIR: [[NEG_CAST:%.*]] = cir.cast bitcast [[NEG_BYTES]] : !cir.vector<16 x !s8i> -> !cir.vector<8 x !cir.f16>
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.f16>
// CIR: cir.fma [[LANE]], [[NEG_CAST]], %{{.*}} : !cir.vector<8 x !cir.f16>

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[FNEG:%.*]] = fneg <8 x half> [[B]]
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[FNEG]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <8 x half> [[C_CAST]], <8 x half> {{.*}}, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// LLVM-NEXT: [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[B_CAST]], <8 x half> [[A_CAST]])
// LLVM:      ret <8 x half> [[FMA]]
  return vfmsq_laneq_f16(a, b, c, 7);
}

// ALL-LABEL: @test_vfms_n_f16(
float16x4_t test_vfms_n_f16(float16x4_t a, float16x4_t b, float16_t c) {
// CIR: [[FNEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<4 x !cir.f16>
// CIR: {{%.*}} = cir.vec.create({{.*}}) : !cir.vector<4 x !cir.f16>
// CIR: cir.call @vfma_f16(%{{.*}}, [[FNEG]], %{{.*}}) :

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], half {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[FNEG:%.*]] = fneg <4 x half> [[B]]
// LLVM:      [[SPLAT:%.*]] = insertelement <4 x half> {{.*}}, half [[C]], i{{32|64}} 3
// LLVM-NEXT: [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <4 x half> [[FNEG]] to <4 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[SPLAT]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-NEXT: [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[B_CAST]], <4 x half> [[C_CAST]], <4 x half> [[A_CAST]])
// LLVM:      ret <4 x half> [[FMA]]
  return vfms_n_f16(a, b, c);
}

// ALL-LABEL: @test_vfmsq_n_f16(
float16x8_t test_vfmsq_n_f16(float16x8_t a, float16x8_t b, float16_t c) {
// CIR: [[FNEG:%.*]] = cir.fneg %{{.*}} : !cir.vector<8 x !cir.f16>
// CIR: {{%.*}} = cir.vec.create({{.*}}) : !cir.vector<8 x !cir.f16>
// CIR: cir.call @vfmaq_f16(%{{.*}}, [[FNEG]], %{{.*}}) :

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], half {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[FNEG:%.*]] = fneg <8 x half> [[B]]
// LLVM:      [[SPLAT:%.*]] = insertelement <8 x half> {{.*}}, half [[C]], i{{32|64}} 7
// LLVM-NEXT: [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[FNEG]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <8 x half> [[SPLAT]] to <8 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM-NEXT: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM-NEXT: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM-NEXT: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM-NEXT: [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[B_CAST]], <8 x half> [[C_CAST]], <8 x half> [[A_CAST]])
// LLVM:      ret <8 x half> [[FMA]]
  return vfmsq_n_f16(a, b, c);
}

// ALL-LABEL: @test_vfmsh_lane_f16(
float16_t test_vfmsh_lane_f16(float16_t a, float16_t b, float16x4_t c) {
// CIR: [[CONV:%.*]] = cir.cast floating %{{.*}} : !cir.f16 -> !cir.float
// CIR: [[FNEG:%.*]] = cir.fneg [[CONV]] : !cir.float
// CIR: [[NEG:%.*]] = cir.cast floating [[FNEG]] : !cir.float -> !cir.f16
// CIR: cir.store align(2) [[NEG]], [[NEG_SLOT:%.*]] : !cir.f16, !cir.ptr<!cir.f16>
// CIR: [[NEG_ARG:%.*]] = cir.load align(2) [[NEG_SLOT]] : !cir.ptr<!cir.f16>, !cir.f16
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma [[NEG_ARG]], [[LANE]], %{{.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[CONV:%.*]] = fpext half [[B]] to float
// LLVM-NEXT: [[FNEG:%.*]] = fneg float [[CONV]]
// LLVM-NEXT: [[NEG:%.*]] = fptrunc float [[FNEG]] to half
// LLVM-NEXT: [[LANE:%.*]] = extractelement <4 x half> [[C]], i{{32|64}} 3
// LLVM-NEXT: [[FMA:%.*]] = call half @llvm.fma.f16(half [[NEG]], half [[LANE]], half [[A]])
// LLVM:      ret half [[FMA]]
  return vfmsh_lane_f16(a, b, c, 3);
}

// ALL-LABEL: @test_vfmsh_laneq_f16(
float16_t test_vfmsh_laneq_f16(float16_t a, float16_t b, float16x8_t c) {
// CIR: [[CONV:%.*]] = cir.cast floating %{{.*}} : !cir.f16 -> !cir.float
// CIR: [[FNEG:%.*]] = cir.fneg [[CONV]] : !cir.float
// CIR: [[NEG:%.*]] = cir.cast floating [[FNEG]] : !cir.float -> !cir.f16
// CIR: cir.store align(2) [[NEG]], [[NEG_SLOT:%.*]] : !cir.f16, !cir.ptr<!cir.f16>
// CIR: [[NEG_ARG:%.*]] = cir.load align(2) [[NEG_SLOT]] : !cir.ptr<!cir.f16>, !cir.f16
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<8 x !cir.f16>
// CIR: cir.fma [[NEG_ARG]], [[LANE]], %{{.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[CONV:%.*]] = fpext half [[B]] to float
// LLVM-NEXT: [[FNEG:%.*]] = fneg float [[CONV]]
// LLVM-NEXT: [[NEG:%.*]] = fptrunc float [[FNEG]] to half
// LLVM-NEXT: [[LANE:%.*]] = extractelement <8 x half> [[C]], i{{32|64}} 7
// LLVM-NEXT: [[FMA:%.*]] = call half @llvm.fma.f16(half [[NEG]], half [[LANE]], half [[A]])
// LLVM:      ret half [[FMA]]
  return vfmsh_laneq_f16(a, b, c, 7);
}
