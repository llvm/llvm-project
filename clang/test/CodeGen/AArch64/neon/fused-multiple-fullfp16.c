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
// This file currently covers the f16 wrappers that lower through
// BI__builtin_neon_vfmaq_v, BI__builtin_neon_vfmaq_lane_v, and
// BI__builtin_neon_vfma_laneq_v.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#fused-multiply-accumulate-2
//
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.6.1.9.3 Fused multiply-accumulate, vector quad forms
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vfmaq_f16(
// CIR-LABEL: @vfmaq_f16(
float16x8_t test_vfmaq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>) -> !cir.vector<8 x !cir.f16>

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

// ALL-LABEL: @test_vfmaq_lane_f16(
float16x8_t test_vfmaq_lane_f16(float16x8_t a, float16x8_t b,
                                 float16x4_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<8 x !cir.f16>
// CIR: cir.call_llvm_intrinsic "fma" %{{.*}}, [[LANE]], %{{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>) -> !cir.vector<8 x !cir.f16>

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM:      [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT: [[B_I:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM-NEXT: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM-NEXT: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM-NEXT: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM-NEXT: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM:      [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM-NEXT: [[LANE:%.*]] = shufflevector <4 x half> [[C_CAST]], <4 x half> {{.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// LLVM:      [[FMA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[B_CAST:%.*]], <8 x half> [[LANE]], <8 x half> [[A_CAST:%.*]])
// LLVM:      ret <8 x half> [[FMA]]
  return vfmaq_lane_f16(a, b, c, 3);
}

// ALL-LABEL: @test_vfma_laneq_f16(
float16x4_t test_vfma_laneq_f16(float16x4_t a, float16x4_t b,
                                 float16x8_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.f16>
// CIR: cir.call_llvm_intrinsic "fma" [[LANE]], %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>) -> !cir.vector<4 x !cir.f16>

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
