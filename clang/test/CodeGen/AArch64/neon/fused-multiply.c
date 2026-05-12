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
// The main difference is the use of RUN lines that enable ClangIR lowering.
// This file currently covers the f32/f64 wrappers that lower through
// BI__builtin_neon_vfmaq_v.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#fused-multiply-accumulate
//
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.1.1.2.5 Fused multiply-accumulate, vector quad forms
//===------------------------------------------------------===//

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
