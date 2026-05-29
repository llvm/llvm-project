// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon -target-feature +fullfp16           -emit-llvm  %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa  | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fclangir -emit-llvm  %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa  | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fclangir -emit-cir   %s -disable-O0-optnone |                                FileCheck %s --check-prefixes=CIR %}

//=============================================================================
// NOTES
//
// Tests for unconstrained conversion intrinsics that require the fullfp16 extension.
//
// This file contains FP16 tests that were originally located in
//  *  clang/test/CodeGen/AArch64/v8.2a-neon-intrinsics.c
// The main difference is the use of RUN lines that enable ClangIR lowering;
// therefore only builtins currently supported by ClangIR are tested here.
// Once ClangIR support is complete, this file is intended to replace the
// original test file.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#bitwise-equal-to-zero
//
//=============================================================================

#include <arm_fp16.h>
#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.6.3.1 Convearions
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#conversions-2
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vcvt_s16_f16
// CIR-LABEL: @vcvt_s16_f16
int16x4_t test_vcvt_s16_f16 (float16x4_t a) {
// CIR: cir.call_llvm_intrinsic "fptosi.sat"

// LLVM-SAME: (<4 x half> {{.*}} [[A:%.*]])
// LLVM:         [[TMP0:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// LLVM-NEXT:    [[VCVTZ_I:%.*]] = call <4 x i16> @llvm.fptosi.sat.v4i16.v4f16(<4 x half> [[TMP2]])
// LLVM-NEXT:    ret <4 x i16> [[VCVTZ_I]]
  return vcvt_s16_f16(a);
}

// LLVM-LABEL: @test_vcvtq_s16_f16
// CIR-LABEL: @vcvtq_s16_f16
int16x8_t test_vcvtq_s16_f16 (float16x8_t a) {
// CIR: cir.call_llvm_intrinsic "fptosi.sat"

// LLVM-SAME: (<8 x half> {{.*}} [[A:%.*]])
// LLVM:         [[TMP0:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// LLVM-NEXT:    [[VCVTZ_I:%.*]] = call <8 x i16> @llvm.fptosi.sat.v8i16.v8f16(<8 x half> [[TMP2]])
// LLVM-NEXT:    ret <8 x i16> [[VCVTZ_I]]
  return vcvtq_s16_f16(a);
}

// LLVM-LABEL: @test_vcvt_u16_f16
// CIR-LABEL: @vcvt_u16_f16
uint16x4_t test_vcvt_u16_f16 (float16x4_t a) {
// CIR: cir.call_llvm_intrinsic "fptoui.sat"

// LLVM-SAME: (<4 x half> {{.*}} [[A:%.*]])
// LLVM:         [[TMP0:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// LLVM-NEXT:    [[VCVTZ_I:%.*]] = call <4 x i16> @llvm.fptoui.sat.v4i16.v4f16(<4 x half> [[TMP2]])
// LLVM-NEXT:    ret <4 x i16> [[VCVTZ_I]]
  return vcvt_u16_f16(a);
}

// LLVM-LABEL: @test_vcvtq_u16_f16
// CIR-LABEL: @vcvtq_u16_f16
uint16x8_t test_vcvtq_u16_f16 (float16x8_t a) {
// CIR: cir.call_llvm_intrinsic "fptoui.sat"

// LLVM: (<8 x half> {{.*}} [[A:%.*]])
// LLVM:         [[TMP0:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// LLVM-NEXT:    [[VCVTZ_I:%.*]] = call <8 x i16> @llvm.fptoui.sat.v8i16.v8f16(<8 x half> [[TMP2]])
// LLVM-NEXT:    ret <8 x i16> [[VCVTZ_I]]
  return vcvtq_u16_f16(a);
}

