// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefixes=CIR %}

//=============================================================================
// NOTES
//
// Tests for vector Widen intrinsics
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#widen
//
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// 5.1.5.2. Widen
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#widen
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vmovl_s8(
// CIR-LABEL: @vmovl_s8(
int16x8_t test_vmovl_s8(int8x8_t a) {
// CIR: [[VMOVL_I:%.*]] = cir.cast integral {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]])
// LLVM: [[VMOVL_I:%.*]] = sext <8 x i8> [[A]] to <8 x i16>
// LLVM: ret <8 x i16> [[VMOVL_I]]
  return vmovl_s8(a);
}

// LLVM-LABEL: @test_vmovl_s16(
// CIR-LABEL: @vmovl_s16(
int32x4_t test_vmovl_s16(int16x4_t a) {
// CIR: [[VMOVL_I:%.*]] = cir.cast integral {{%.*}} : !cir.vector<4 x !s16i> -> !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL_I:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: ret <4 x i32> [[VMOVL_I]]
  return vmovl_s16(a);
}

// LLVM-LABEL: @test_vmovl_s32(
// CIR-LABEL: @vmovl_s32(
int64x2_t test_vmovl_s32(int32x2_t a) {
// CIR: [[VMOVL_I:%.*]] = cir.cast integral {{%.*}} : !cir.vector<2 x !s32i> -> !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL_I:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: ret <2 x i64> [[VMOVL_I]]
  return vmovl_s32(a);
}

// LLVM-LABEL: @test_vmovl_u8(
// CIR-LABEL: @vmovl_u8(
uint16x8_t test_vmovl_u8(uint8x8_t a) {
// CIR: [[VMOVL_I:%.*]] = cir.cast integral {{%.*}} : !cir.vector<8 x !u8i> -> !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]])
// LLVM: [[VMOVL_I:%.*]] = zext <8 x i8> [[A]] to <8 x i16>
// LLVM: ret <8 x i16> [[VMOVL_I]]
  return vmovl_u8(a);
}

// LLVM-LABEL: @test_vmovl_u16(
// CIR-LABEL: @vmovl_u16(
uint32x4_t test_vmovl_u16(uint16x4_t a) {
// CIR: [[VMOVL_I:%.*]] = cir.cast integral {{%.*}} : !cir.vector<4 x !u16i> -> !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL_I:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: ret <4 x i32> [[VMOVL_I]]
  return vmovl_u16(a);
}

// LLVM-LABEL: @test_vmovl_u32(
// CIR-LABEL: @vmovl_u32(
uint64x2_t test_vmovl_u32(uint32x2_t a) {
// CIR: [[VMOVL_I:%.*]] = cir.cast integral {{%.*}} : !cir.vector<2 x !u32i> -> !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL_I:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: ret <2 x i64> [[VMOVL_I]]
  return vmovl_u32(a);
}
// LLVM-LABEL: @test_vmovl_high_s8(
// CIR-LABEL: @vmovl_high_s8(
int16x8_t test_vmovl_high_s8(int8x16_t a) {
// CIR: cir.call @vget_high_s8
// CIR: cir.cast integral {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !s16i>
// CIR: cir.shift(left, %{{.*}} : !cir.vector<8 x !s16i>, %{{.*}} : !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM: [[VMOVL_I:%.*]] = sext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// LLVM: ret <8 x i16> [[VMOVL_I]]
  return vmovl_high_s8(a);
}

// LLVM-LABEL: @test_vmovl_high_s16(
// CIR-LABEL: @vmovl_high_s16(
int32x4_t test_vmovl_high_s16(int16x8_t a) {
// CIR: cir.call @vget_high_s16
// CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR: cir.cast integral %{{.*}} : !cir.vector<4 x !s16i> -> !cir.vector<4 x !s32i>
// CIR: cir.shift(left, %{{.*}} : !cir.vector<4 x !s32i>, %{{.*}} : !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL_I:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: ret <4 x i32> [[VMOVL_I]]
  return vmovl_high_s16(a);
}

// LLVM-LABEL: @test_vmovl_high_s32(
// CIR-LABEL: @vmovl_high_s32(
int64x2_t test_vmovl_high_s32(int32x4_t a) {
// CIR: cir.call @vget_high_s32
// CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
// CIR: cir.cast integral %{{.*}} : !cir.vector<2 x !s32i> -> !cir.vector<2 x !s64i>
// CIR: cir.shift(left, %{{.*}} : !cir.vector<2 x !s64i>, %{{.*}} : !cir.vector<2 x !s64i>) -> !cir.vector<2 x !s64i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[A]], <2 x i32> <i32 2, i32 3>
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL_I:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: ret <2 x i64> [[VMOVL_I]]
  return vmovl_high_s32(a);
}

// LLVM-LABEL: @test_vmovl_high_u8(
// CIR-LABEL: @vmovl_high_u8(
uint16x8_t test_vmovl_high_u8(uint8x16_t a) {
// CIR: cir.call @vget_high_u8
// CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
// CIR: cir.cast integral %{{.*}} : !cir.vector<8 x !u8i> -> !cir.vector<8 x !u16i>
// CIR: cir.shift(left, %{{.*}} : !cir.vector<8 x !u16i>, %{{.*}} : !cir.vector<8 x !u16i>) -> !cir.vector<8 x !u16i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM: [[VMOVL_I:%.*]] = zext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// LLVM: ret <8 x i16> [[VMOVL_I]]
  return vmovl_high_u8(a);
}

// LLVM-LABEL: @test_vmovl_high_u16(
// CIR-LABEL: @vmovl_high_u16(
uint32x4_t test_vmovl_high_u16(uint16x8_t a) {
// CIR: cir.call @vget_high_u16
// CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
// CIR: cir.cast integral %{{.*}} : !cir.vector<4 x !u16i> -> !cir.vector<4 x !u32i>
// CIR: cir.shift(left, %{{.*}} : !cir.vector<4 x !u32i>, %{{.*}} : !cir.vector<4 x !u32i>) -> !cir.vector<4 x !u32i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL_I:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: ret <4 x i32> [[VMOVL_I]]
  return vmovl_high_u16(a);
}

// LLVM-LABEL: @test_vmovl_high_u32(
// CIR-LABEL: @vmovl_high_u32(
uint64x2_t test_vmovl_high_u32(uint32x4_t a) {
// CIR: cir.call @vget_high_u32
// CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
// CIR: cir.cast integral %{{.*}} : !cir.vector<2 x !u32i> -> !cir.vector<2 x !u64i>
// CIR: cir.shift(left, %{{.*}} : !cir.vector<2 x !u64i>, %{{.*}} : !cir.vector<2 x !u64i>) -> !cir.vector<2 x !u64i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[A]], <2 x i32> <i32 2, i32 3>
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL_I:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: ret <2 x i64> [[VMOVL_I]]
  return vmovl_high_u32(a);
}

