// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=ALL,CIR %}

//=============================================================================
// NOTES
//
// This file contains tests that were originally located in
//  *  clang/test/CodeGen/AArch64/neon-intrinsics.c.
// The main difference is the use of RUN lines that enable ClangIR lowering;
// therefore only builtins currently supported by ClangIR are tested here.
// Once ClangIR support is complete, this file is intended to replace the
// original test file.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#bitwise-equal-to-zero
//
// Different labels for CIR stem from an additional function call that is
// present at the AST and CIR levels, but is inlined at the LLVM IR level.
//
//=============================================================================

#include <arm_neon.h>

// LLVM-LABEL: @test_vnegd_s64
// CIR-LABEL: @vnegd_s64
int64_t test_vnegd_s64(int64_t a) {
// CIR: cir.minus {{.*}} : !s64i

// LLVM-SAME: i64 {{.*}} [[A:%.*]])
// LLVM:          [[VNEGD_I:%.*]] = sub i64 0, [[A]]
// LLVM-NEXT:     ret i64 [[VNEGD_I]]
  return (int64_t)vnegd_s64(a);
}

//===------------------------------------------------------===//
// 2.1.2.2 Bitwise equal to zero
//===------------------------------------------------------===//
// LLVM-LABEL: @test_vceqz_s8(
// CIR-LABEL: @vceqz_s8(
uint8x8_t test_vceqz_s8(int8x8_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<8 x !s8i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]]) #[[ATTR0:[0-9]+]] {
// LLVM:    [[TMP0:%.*]] = icmp eq <8 x i8> [[A]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    ret <8 x i8> [[VCEQZ_I]]
  return vceqz_s8(a);
}

// LLVM-LABEL: @test_vceqz_s16(
// CIR-LABEL: @vceqz_s16(
uint16x4_t test_vceqz_s16(int16x4_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<4 x !s16i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <4 x i16> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP2]] to <4 x i16>
// LLVM-NEXT:    ret <4 x i16> [[VCEQZ_I]]
  return vceqz_s16(a);
}

// LLVM-LABEL: @test_vceqz_s32(
// CIR-LABEL: @vceqz_s32(
uint32x2_t test_vceqz_s32(int32x2_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<2 x !s32i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <2 x i32> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP2]] to <2 x i32>
// LLVM-NEXT:    ret <2 x i32> [[VCEQZ_I]]
  return vceqz_s32(a);
}

// LLVM-LABEL: @test_vceqz_s64(
// CIR-LABEL: @vceqz_s64(
uint64x1_t test_vceqz_s64(int64x1_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<1 x !s64i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>

// LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <1 x i64> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <1 x i1> [[TMP2]] to <1 x i64>
// LLVM-NEXT:    ret <1 x i64> [[VCEQZ_I]]
  return vceqz_s64(a);
}

// LLVM-LABEL: @test_vceqz_p64(
// CIR-LABEL: @vceqz_p64(
uint64x1_t test_vceqz_p64(poly64x1_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<1 x !s64i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>

// LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <1 x i64> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <1 x i1> [[TMP2]] to <1 x i64>
// LLVM-NEXT:    ret <1 x i64> [[VCEQZ_I]]
  return vceqz_p64(a);
}

// LLVM-LABEL: @test_vceqzq_s8(
// CIR-LABEL: @vceqzq_s8(
uint8x16_t test_vceqzq_s8(int8x16_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<16 x !s8i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = icmp eq <16 x i8> [[A]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    ret <16 x i8> [[VCEQZ_I]]
  return vceqzq_s8(a);
}

// LLVM-LABEL: @test_vceqzq_s16(
// CIR-LABEL: @vceqzq_s16(
uint16x8_t test_vceqzq_s16(int16x8_t a) {
// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <8 x i16> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP2]] to <8 x i16>
// LLVM-NEXT:    ret <8 x i16> [[VCEQZ_I]]
  return vceqzq_s16(a);
}

// LLVM-LABEL: @test_vceqzq_s32(
// CIR-LABEL: @vceqzq_s32(
uint32x4_t test_vceqzq_s32(int32x4_t a) {
// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <4 x i32> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP2]] to <4 x i32>
// LLVM-NEXT:    ret <4 x i32> [[VCEQZ_I]]
  return vceqzq_s32(a);
}

// LLVM-LABEL: @test_vceqzq_s64(
// CIR-LABEL: @vceqzq_s64(
uint64x2_t test_vceqzq_s64(int64x2_t a) {
// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <2 x i64> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP2]] to <2 x i64>
// LLVM-NEXT:    ret <2 x i64> [[VCEQZ_I]]
  return vceqzq_s64(a);
}

// LLVM-LABEL: @test_vceqz_u8(
// CIR-LABEL: @vceqz_u8(
uint8x8_t test_vceqz_u8(uint8x8_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<8 x !u8i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = icmp eq <8 x i8> [[A]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    ret <8 x i8> [[VCEQZ_I]]
  return vceqz_u8(a);
}

// LLVM-LABEL: @test_vceqz_u16(
// CIR-LABEL: @vceqz_u16(
uint16x4_t test_vceqz_u16(uint16x4_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<4 x !u16i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<4 x !u16i>, !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <4 x i16> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP2]] to <4 x i16>
// LLVM-NEXT:    ret <4 x i16> [[VCEQZ_I]]
  return vceqz_u16(a);
}

// LLVM-LABEL: @test_vceqz_u32(
// CIR-LABEL: @vceqz_u32(
uint32x2_t test_vceqz_u32(uint32x2_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<2 x !u32i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<2 x !u32i>, !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <2 x i32> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP2]] to <2 x i32>
// LLVM-NEXT:    ret <2 x i32> [[VCEQZ_I]]
  return vceqz_u32(a);
}

// LLVM-LABEL: @test_vceqz_u64(
// CIR-LABEL: @vceqz_u64(
uint64x1_t test_vceqz_u64(uint64x1_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !u64i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<1 x !u64i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<1 x !u64i>, !cir.vector<1 x !s64i>

// LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <1 x i64> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <1 x i1> [[TMP2]] to <1 x i64>
// LLVM-NEXT:    ret <1 x i64> [[VCEQZ_I]]
  return vceqz_u64(a);
}

// LLVM-LABEL: @test_vceqzq_u8(
// CIR-LABEL: @vceqzq_u8(
uint8x16_t test_vceqzq_u8(uint8x16_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<16 x !u8i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = icmp eq <16 x i8> [[A]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    ret <16 x i8> [[VCEQZ_I]]
  return vceqzq_u8(a);
}

// LLVM-LABEL: @test_vceqzq_u16(
// CIR-LABEL: @vceqzq_u16(
uint16x8_t test_vceqzq_u16(uint16x8_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<8 x !u16i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<8 x !u16i>, !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <8 x i16> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP2]] to <8 x i16>
// LLVM-NEXT:    ret <8 x i16> [[VCEQZ_I]]
  return vceqzq_u16(a);
}

// LLVM-LABEL: @test_vceqzq_u32(
// CIR-LABEL: @vceqzq_u32(
uint32x4_t test_vceqzq_u32(uint32x4_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<4 x !u32i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<4 x !u32i>, !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <4 x i32> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP2]] to <4 x i32>
// LLVM-NEXT:    ret <4 x i32> [[VCEQZ_I]]
  return vceqzq_u32(a);
}

// LLVM-LABEL: @test_vceqzq_u64(
// CIR-LABEL: @vceqzq_u64(
uint64x2_t test_vceqzq_u64(uint64x2_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !u64i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<2 x !u64i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<2 x !u64i>, !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <2 x i64> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP2]] to <2 x i64>
// LLVM-NEXT:    ret <2 x i64> [[VCEQZ_I]]
  return vceqzq_u64(a);
}

// LLVM-LABEL: @test_vceqz_f32(
// CIR-LABEL: @vceqz_f32(
uint32x2_t test_vceqz_f32(float32x2_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.float>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<2 x !cir.float>, !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x float> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <2 x i32>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// LLVM-NEXT:    [[TMP3:%.*]] = fcmp oeq <2 x float> [[TMP2]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i32>
// LLVM-NEXT:    ret <2 x i32> [[VCEQZ_I]]
  return vceqz_f32(a);
}

// LLVM-LABEL: @test_vceqz_f64(
// CIR-LABEL: @vceqz_f64(
uint64x1_t test_vceqz_f64(float64x1_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<1 x !cir.double>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<1 x !cir.double>, !cir.vector<1 x !s64i>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM-NEXT:    [[__P0_ADDR_I_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i32 0
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <1 x i64> [[__P0_ADDR_I_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// LLVM-NEXT:    [[TMP3:%.*]] = fcmp oeq <1 x double> [[TMP2]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <1 x i1> [[TMP3]] to <1 x i64>
// LLVM-NEXT:    ret <1 x i64> [[VCEQZ_I]]
  return vceqz_f64(a);
}

// LLVM-LABEL: @test_vceqzq_f32(
// CIR-LABEL: @vceqzq_f32(
uint32x4_t test_vceqzq_f32(float32x4_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.float>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x float> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x float> [[A]] to <4 x i32>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// LLVM-NEXT:    [[TMP3:%.*]] = fcmp oeq <4 x float> [[TMP2]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i32>
// LLVM-NEXT:    ret <4 x i32> [[VCEQZ_I]]
  return vceqzq_f32(a);
}

// LLVM-LABEL: @test_vceqzq_f64(
// CIR-LABEL: @vceqzq_f64(
uint64x2_t test_vceqzq_f64(float64x2_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<2 x !cir.double>, !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// LLVM-NEXT:    [[TMP3:%.*]] = fcmp oeq <2 x double> [[TMP2]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i64>
// LLVM-NEXT:    ret <2 x i64> [[VCEQZ_I]]
  return vceqzq_f64(a);
}

// LLVM-LABEL: @test_vceqz_p8(
// CIR-LABEL: @vceqz_p8(
uint8x8_t test_vceqz_p8(poly8x8_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<8 x !s8i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = icmp eq <8 x i8> [[A]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    ret <8 x i8> [[VCEQZ_I]]
  return vceqz_p8(a);
}

// LLVM-LABEL: @test_vceqzq_p8(
// CIR-LABEL: @vceqzq_p8(
uint8x16_t test_vceqzq_p8(poly8x16_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<16 x !s8i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = icmp eq <16 x i8> [[A]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    ret <16 x i8> [[VCEQZ_I]]
  return vceqzq_p8(a);
}

// LLVM-LABEL: @test_vceqzq_p64(
// CIR-LABEL: @vceqzq_p64(
uint64x2_t test_vceqzq_p64(poly64x2_t a) {
// CIR:   cir.cast bitcast {{%.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
// CIR:   [[C_0:%.*]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
// CIR:   cir.vec.cmp(eq, {{%.*}}, [[C_0]]) : !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]]) #[[ATTR0]] {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq <2 x i64> [[TMP1]], zeroinitializer
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP2]] to <2 x i64>
// LLVM-NEXT:    ret <2 x i64> [[VCEQZ_I]]
  return vceqzq_p64(a);
}

// LLVM-LABEL: @test_vceqzd_s64
// CIR-LABEL: @vceqzd_s64
uint64_t test_vceqzd_s64(int64_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.int<0>
// CIR:   [[CMP:%.*]] = cir.cmp eq %{{.*}}, [[C_0]] : !s64i
// CIR:   [[RES:%.*]] = cir.cast bool_to_int [[CMP]] : !cir.bool -> !cir.int<s, 1>
// CIR:   cir.cast integral [[RES]] : !cir.int<s, 1> -> !u64i

// LLVM-SAME: i64{{.*}} [[A:%.*]])
// LLVM:          [[TMP0:%.*]] = icmp eq i64 [[A]], 0
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// LLVM-NEXT:    ret i64 [[VCEQZ_I]]
  return (uint64_t)vceqzd_s64(a);
}

// LLVM-LABEL: @test_vceqzd_u64(
// CIR-LABEL: @vceqzd_u64(
int64_t test_vceqzd_u64(int64_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.int<0>
// CIR:   [[CMP:%.*]] = cir.cmp eq %{{.*}}, [[C_0]] : !u64i
// CIR:   [[RES:%.*]] = cir.cast bool_to_int [[CMP]] : !cir.bool -> !cir.int<s, 1>
// CIR:   cir.cast integral [[RES]] : !cir.int<s, 1> -> !u64i

// LLVM-SAME: i64 {{.*}} [[A:%.*]])
// LLVM:    [[TMP0:%.*]] = icmp eq i64 [[A]], 0
// LLVM-NEXT:    [[VCEQZD_I:%.*]] = sext i1 [[TMP0]] to i64
// LLVM-NEXT:    ret i64 [[VCEQZD_I]]
  return (int64_t)vceqzd_u64(a);
}

// LLVM-LABEL: @test_vceqzs_f32(
// CIR-LABEL: @vceqzs_f32(
uint32_t test_vceqzs_f32(float32_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.fp<0.000000e+00>
// CIR:   [[CMP:%.*]] = cir.cmp eq %{{.*}}, [[C_0]] : !cir.float
// CIR:   [[RES:%.*]] = cir.cast bool_to_int [[CMP]] : !cir.bool -> !cir.int<s, 1>
// CIR:   cir.cast integral [[RES]] : !cir.int<s, 1> -> !u32i

// LLVM-SAME: float {{.*}} [[A:%.*]])
// LLVM:    [[TMP0:%.*]] = fcmp oeq float [[A]], 0.000000e+00
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i32
// LLVM-NEXT:    ret i32 [[VCEQZ_I]]
  return (uint32_t)vceqzs_f32(a);
}

// LLVM-LABEL: @test_vceqzd_f64(
// CIR-LABEL: @vceqzd_f64(
uint64_t test_vceqzd_f64(float64_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.fp<0.000000e+00>
// CIR:   [[CMP:%.*]] = cir.cmp eq %{{.*}}, [[C_0]] : !cir.double
// CIR:   [[RES:%.*]] = cir.cast bool_to_int [[CMP]] : !cir.bool -> !cir.int<s, 1>
// CIR:   cir.cast integral [[RES]] : !cir.int<s, 1> -> !u64i


// LLVM-SAME: double {{.*}} [[A:%.*]])
// LLVM:    [[TMP0:%.*]] = fcmp oeq double [[A]], 0.000000e+00
// LLVM-NEXT:    [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// LLVM-NEXT:    ret i64 [[VCEQZ_I]]
  return (uint64_t)vceqzd_f64(a);
}


//===------------------------------------------------------===//
// 2.1.1.6.1. Absolute difference
//===------------------------------------------------------===//
// LLVM-LABEL: @test_vabd_s8(
// CIR-LABEL: @vabd_s8(
int8x8_t test_vabd_s8(int8x8_t v1, int8x8_t v2) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.sabd" %{{.*}}, %{{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>) -> !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]]) 
// LLVM:         [[VABD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
// LLVM-NEXT:    ret <8 x i8> [[VABD_I]]
  return vabd_s8(v1, v2);
}

// LLVM-LABEL: @test_vabd_s16(
// CIR-LABEL: @vabd_s16(
int16x4_t test_vabd_s16(int16x4_t v1, int16x4_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.sabd" [[V1]], [[V2]]

// LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> [[VABD_I]], <4 x i16> [[VABD1_I]])
// LLVM-NEXT:    ret <4 x i16> [[VABD2_I]]
  return vabd_s16(v1, v2);
}

// LLVM-LABEL: @test_vabd_s32(
// CIR-LABEL: @vabd_s32(
int32x2_t test_vabd_s32(int32x2_t v1, int32x2_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.sabd" [[V1]], [[V2]]

// LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> [[VABD_I]], <2 x i32> [[VABD1_I]])
// LLVM-NEXT:    ret <2 x i32> [[VABD2_I]]
  return vabd_s32(v1, v2);
}

// LLVM-LABEL: @test_vabd_u8(
// CIR-LABEL: @vabd_u8(
uint8x8_t test_vabd_u8(uint8x8_t v1, uint8x8_t v2) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.uabd" %{{.*}}, %{{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>) -> !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]]) 
// LLVM:         [[VABD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
// LLVM-NEXT:    ret <8 x i8> [[VABD_I]]
  return vabd_u8(v1, v2);
}

// LLVM-LABEL: @test_vabd_u16(
// CIR-LABEL: @vabd_u16(
uint16x4_t test_vabd_u16(uint16x4_t v1, uint16x4_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.uabd" [[V1]], [[V2]]

// LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> [[VABD_I]], <4 x i16> [[VABD1_I]])
// LLVM-NEXT:    ret <4 x i16> [[VABD2_I]]
  return vabd_u16(v1, v2);
}

// LLVM-LABEL: @test_vabd_u32(
// CIR-LABEL: @vabd_u32(
uint32x2_t test_vabd_u32(uint32x2_t v1, uint32x2_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.uabd" [[V1]], [[V2]]

// LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> [[VABD_I]], <2 x i32> [[VABD1_I]])
// LLVM-NEXT:    ret <2 x i32> [[VABD2_I]]
  return vabd_u32(v1, v2);
}

// LLVM-LABEL: @test_vabd_f32(
// CIR-LABEL: @vabd_f32(
float32x2_t test_vabd_f32(float32x2_t v1, float32x2_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !cir.float>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.fabd" [[V1]], [[V2]]

// LLVM-SAME: <2 x float> {{.*}} [[V1:%.*]], <2 x float> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <2 x float> [[V1]] to <2 x i32>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x float> [[V2]] to <2 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <2 x i32> [[TMP1]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x float>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fabd.v2f32(<2 x float> [[VABD_I]], <2 x float> [[VABD1_I]])
// LLVM-NEXT:    ret <2 x float> [[VABD2_I]]
  return vabd_f32(v1, v2);
}

// LLVM-LABEL: @test_vabd_f64(
// CIR-LABEL: @vabd_f64(
float64x1_t test_vabd_f64(float64x1_t v1, float64x1_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !cir.double>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.fabd" [[V1]], [[V2]]

// LLVM-SAME: <1 x double> {{.*}} [[V1:%.*]], <1 x double> {{.*}} [[V2:%.*]])
// LLVM:         [[TMP0:%.*]] = bitcast <1 x double> [[V1]] to i64
// LLVM-NEXT:    [[__P0_ADDR_I_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i32 0
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <1 x double> [[V2]] to i64
// LLVM-NEXT:    [[__P1_ADDR_I_SROA_0_0_VEC_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[TMP1]], i32 0
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <1 x i64> [[__P0_ADDR_I_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <1 x i64> [[__P1_ADDR_I_SROA_0_0_VEC_INSERT]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP3]] to <1 x double>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fabd.v1f64(<1 x double> [[VABD_I]], <1 x double> [[VABD1_I]])
// LLVM-NEXT:    ret <1 x double> [[VABD2_I]]
  return vabd_f64(v1, v2);
}

// LLVM-LABEL: @test_vabdq_s8(
// CIR-LABEL: @vabdq_s8(
int8x16_t test_vabdq_s8(int8x16_t v1, int8x16_t v2) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.sabd" %{{.*}}, %{{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>) -> !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]]) 
// LLVM:    [[VABD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sabd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
// LLVM-NEXT:    ret <16 x i8> [[VABD_I]]
  return vabdq_s8(v1, v2);
}

// LLVM-LABEL: @test_vabdq_s16(
// CIR-LABEL: @vabdq_s16(
int16x8_t test_vabdq_s16(int16x8_t v1, int16x8_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.sabd" [[V1]], [[V2]]

// LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sabd.v8i16(<8 x i16> [[VABD_I]], <8 x i16> [[VABD1_I]])
// LLVM-NEXT:    ret <8 x i16> [[VABD2_I]]
  return vabdq_s16(v1, v2);
}

// LLVM-LABEL: @test_vabdq_s32(
// CIR-LABEL: @vabdq_s32(
int32x4_t test_vabdq_s32(int32x4_t v1, int32x4_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.sabd" [[V1]], [[V2]]

// LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sabd.v4i32(<4 x i32> [[VABD_I]], <4 x i32> [[VABD1_I]])
// LLVM-NEXT:    ret <4 x i32> [[VABD2_I]]
  return vabdq_s32(v1, v2);
}

// LLVM-LABEL: @test_vabdq_u8(
// CIR-LABEL: @vabdq_u8(
uint8x16_t test_vabdq_u8(uint8x16_t v1, uint8x16_t v2) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.uabd" %{{.*}}, %{{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>) -> !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]]) 
// LLVM:    [[VABD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uabd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
// LLVM-NEXT:    ret <16 x i8> [[VABD_I]]
  return vabdq_u8(v1, v2);
}

// LLVM-LABEL: @test_vabdq_u16(
// CIR-LABEL: @vabdq_u16(
uint16x8_t test_vabdq_u16(uint16x8_t v1, uint16x8_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.uabd" [[V1]], [[V2]]

// LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> [[VABD_I]], <8 x i16> [[VABD1_I]])
// LLVM-NEXT:    ret <8 x i16> [[VABD2_I]]
  return vabdq_u16(v1, v2);
}

// LLVM-LABEL: @test_vabdq_u32(
// CIR-LABEL: @vabdq_u32(
uint32x4_t test_vabdq_u32(uint32x4_t v1, uint32x4_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.uabd" [[V1]], [[V2]]

// LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uabd.v4i32(<4 x i32> [[VABD_I]], <4 x i32> [[VABD1_I]])
// LLVM-NEXT:    ret <4 x i32> [[VABD2_I]]
  return vabdq_u32(v1, v2);
}

// LLVM-LABEL: @test_vabdq_f32(
// CIR-LABEL: @vabdq_f32(
float32x4_t test_vabdq_f32(float32x4_t v1, float32x4_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !cir.float>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.fabd" [[V1]], [[V2]]

// LLVM-SAME: <4 x float> {{.*}} [[V1:%.*]], <4 x float> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <4 x float> [[V1]] to <4 x i32>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x float> [[V2]] to <4 x i32>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <4 x i32> [[TMP1]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <4 x float>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fabd.v4f32(<4 x float> [[VABD_I]], <4 x float> [[VABD1_I]])
// LLVM-NEXT:    ret <4 x float> [[VABD2_I]]
  return vabdq_f32(v1, v2);
}

// LLVM-LABEL: @test_vabdq_f64(
// CIR-LABEL: @vabdq_f64(
float64x2_t test_vabdq_f64(float64x2_t v1, float64x2_t v2) {
// CIR:   [[V1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR:   [[V2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !cir.double>
// CIR:   cir.call_llvm_intrinsic "aarch64.neon.fabd" [[V1]], [[V2]]

// LLVM-SAME: <2 x double> {{.*}} [[V1:%.*]], <2 x double> {{.*}} [[V2:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <2 x double> [[V1]] to <2 x i64>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x double> [[V2]] to <2 x i64>
// LLVM-NEXT:    [[TMP2:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
// LLVM-NEXT:    [[TMP3:%.*]] = bitcast <2 x i64> [[TMP1]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <2 x double>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fabd.v2f64(<2 x double> [[VABD_I]], <2 x double> [[VABD1_I]])
// LLVM-NEXT:    ret <2 x double> [[VABD2_I]]
  return vabdq_f64(v1, v2);
}

// LLVM-LABEL: @test_vabds_f32(
// CIR-LABEL: @vabds_f32(
float32_t test_vabds_f32(float32_t a, float32_t b) {
// CIR:   cir.call_llvm_intrinsic "aarch64.sisd.fabd"

// LLVM-SAME: float {{.*}} [[A:%.*]], float noundef [[B:%.*]])
// LLVM:    [[VABDS_F32_I:%.*]] = call float @llvm.aarch64.sisd.fabd.f32(float [[A]], float [[B]])
// LLVM-NEXT:    ret float [[VABDS_F32_I]]
  return vabds_f32(a, b);
}

// LLVM-LABEL: @test_vabdd_f64(
// CIR-LABEL: @vabdd_f64(
float64_t test_vabdd_f64(float64_t a, float64_t b) {
// CIR:   cir.call_llvm_intrinsic "aarch64.sisd.fabd"

// LLVM-SAME: double {{.*}} [[A:%.*]], double noundef [[B:%.*]])
// LLVM:    [[VABDD_F64_I:%.*]] = call double @llvm.aarch64.sisd.fabd.f64(double [[A]], double [[B]])
// LLVM-NEXT:    ret double [[VABDD_F64_I]]
  return vabdd_f64(a, b);
}

//===------------------------------------------------------===//
// 2.1.1.6.3. Absolute difference and accumulate
//
// The following builtins expand to a call to vabd_{} builtins,
// which is reflected in the CIR output.
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vaba_u8(
// CIR-LABEL: @vaba_u8(
uint8x8_t test_vaba_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabd_u8
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]], <8 x i8> {{.*}} [[V3:%.*]])
// LLVM:         [[VABD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> [[V2]], <8 x i8> [[V3]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i8> [[V1]], [[VABD_I]]
// LLVM-NEXT:    ret <8 x i8> [[ADD_I]]
  return vaba_u8(v1, v2, v3);
}

// LLVM-LABEL: @test_vaba_u16(
// CIR-LABEL: @vaba_u16(
uint16x4_t test_vaba_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabd_u16
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]], <4 x i16> {{.*}} [[V3:%.*]])
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[V3]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> [[VABD_I]], <4 x i16> [[VABD1_I]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i16> [[V1]], [[VABD2_I]]
// LLVM-NEXT:    ret <4 x i16> [[ADD_I]]
  return vaba_u16(v1, v2, v3);
}

// LLVM-LABEL: @test_vaba_u32(
// CIR-LABEL: @vaba_u32(
uint32x2_t test_vaba_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabd_u32
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> {{.*}} [[V2:%.*]], <2 x i32> {{.*}} [[V3:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[V3]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> [[VABD_I]], <2 x i32> [[VABD1_I]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i32> [[V1]], [[VABD2_I]]
// LLVM-NEXT:    ret <2 x i32> [[ADD_I]]
  return vaba_u32(v1, v2, v3);
}

// LLVM-LABEL: @test_vaba_s8(
// CIR-LABEL: @vaba_s8(
int8x8_t test_vaba_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabd_s8
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]], <8 x i8> {{.*}} [[V3:%.*]]) 
// LLVM:         [[VABD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> [[V2]], <8 x i8> [[V3]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i8> [[V1]], [[VABD_I]]
// LLVM-NEXT:    ret <8 x i8> [[ADD_I]]
  return vaba_s8(v1, v2, v3);
}

// LLVM-LABEL: @test_vaba_s16(
// CIR-LABEL: @vaba_s16(
int16x4_t test_vaba_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabd_s16
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]], <4 x i16> {{.*}} [[V3:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[V3]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> [[VABD_I]], <4 x i16> [[VABD1_I]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i16> [[V1]], [[VABD2_I]]
// LLVM-NEXT:    ret <4 x i16> [[ADD_I]]
  return vaba_s16(v1, v2, v3);
}

// LLVM-LABEL: @test_vaba_s32(
// CIR-LABEL: @vaba_s32(
int32x2_t test_vaba_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabd_s32
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> {{.*}} [[V2:%.*]], <2 x i32> {{.*}} [[V3:%.*]])
// LLVM:         [[TMP0:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[V3]] to <8 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> [[VABD_I]], <2 x i32> [[VABD1_I]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i32> [[V1]], [[VABD2_I]]
// LLVM-NEXT:    ret <2 x i32> [[ADD_I]]
  return vaba_s32(v1, v2, v3);
}

// LLVM-LABEL: @test_vabaq_s8(
// CIR-LABEL: @vabaq_s8(
int8x16_t test_vabaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabdq_s8
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]], <16 x i8> {{.*}} [[V3:%.*]]) 
// LLVM:         [[VABD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sabd.v16i8(<16 x i8> [[V2]], <16 x i8> [[V3]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <16 x i8> [[V1]], [[VABD_I]]
// LLVM-NEXT:    ret <16 x i8> [[ADD_I]]
  return vabaq_s8(v1, v2, v3);
}

// LLVM-LABEL: @test_vabaq_s16(
// CIR-LABEL: @vabaq_s16(
int16x8_t test_vabaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabdq_s16
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]], <8 x i16> {{.*}} [[V3:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[V3]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sabd.v8i16(<8 x i16> [[VABD_I]], <8 x i16> [[VABD1_I]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[V1]], [[VABD2_I]]
// LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vabaq_s16(v1, v2, v3);
}

// LLVM-LABEL: @test_vabaq_s32(
// CIR-LABEL: @vabaq_s32(
int32x4_t test_vabaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabdq_s32
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> {{.*}} [[V2:%.*]], <4 x i32> {{.*}} [[V3:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32> [[V3]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sabd.v4i32(<4 x i32> [[VABD_I]], <4 x i32> [[VABD1_I]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[V1]], [[VABD2_I]]
// LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vabaq_s32(v1, v2, v3);
}

// LLVM-LABEL: @test_vabaq_u8(
// CIR-LABEL: @vabaq_u8(
uint8x16_t test_vabaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabdq_u8
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]], <16 x i8> {{.*}} [[V3:%.*]]) 
// LLVM:         [[VABD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uabd.v16i8(<16 x i8> [[V2]], <16 x i8> [[V3]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <16 x i8> [[V1]], [[VABD_I]]
// LLVM-NEXT:    ret <16 x i8> [[ADD_I]]
  return vabaq_u8(v1, v2, v3);
}

// LLVM-LABEL: @test_vabaq_u16(
// CIR-LABEL: @vabaq_u16(
uint16x8_t test_vabaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabdq_u16
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]], <8 x i16> {{.*}} [[V3:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[V3]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> [[VABD_I]], <8 x i16> [[VABD1_I]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[V1]], [[VABD2_I]]
// LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vabaq_u16(v1, v2, v3);
}

// LLVM-LABEL: @test_vabaq_u32(
// CIR-LABEL: @vabaq_u32(
uint32x4_t test_vabaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
// CIR: [[ABD:%.*]] = cir.call @vabdq_u32
// CIR: [[RES:%.*]] = cir.add {{.*}}, [[ABD]]

// LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> {{.*}} [[V2:%.*]], <4 x i32> {{.*}} [[V3:%.*]]) 
// LLVM:         [[TMP0:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i32> [[V3]] to <16 x i8>
// LLVM-NEXT:    [[VABD_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM-NEXT:    [[VABD1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// LLVM-NEXT:    [[VABD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uabd.v4i32(<4 x i32> [[VABD_I]], <4 x i32> [[VABD1_I]])
// LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[V1]], [[VABD2_I]]
// LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vabaq_u32(v1, v2, v3);
}

//===----------------------------------------------------------------------===//
// 2.1.1.7. Maximum
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#maximum
//===----------------------------------------------------------------------===//

// LLVM-LABEL: @test_vmax_s8
// CIR-LABEL: @vmax_s8(
int8x8_t test_vmax_s8(int8x8_t v1, int8x8_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.smax" %{{.*}}, %{{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>) -> !cir.vector<8 x !s8i>

 // LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[VMAX_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.smax.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
 // LLVM: ret <8 x i8> [[VMAX_V_I]]
 return vmax_s8(v1, v2);
}

// LLVM-LABEL: @test_vmax_s16
// CIR-LABEL: @vmax_s16(
int16x4_t test_vmax_s16(int16x4_t v1, int16x4_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.smax" %{{.*}}, %{{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>) -> !cir.vector<4 x !s16i>

 // LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
 // LLVM: [[VMAX_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
 // LLVM: [[VMAX_V1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
 // LLVM: [[VMAX_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.smax.v4i16(<4 x i16> [[VMAX_V_I]], <4 x i16> [[VMAX_V1_I]])
 // LLVM: ret <4 x i16> [[VMAX_V2_I]]
 return vmax_s16(v1, v2);
}

// LLVM-LABEL: @test_vmax_s32
// CIR-LABEL: @vmax_s32(
int32x2_t test_vmax_s32(int32x2_t v1, int32x2_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.smax" %{{.*}}, %{{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>) -> !cir.vector<2 x !s32i>

 // LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
 // LLVM: [[VMAX_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
 // LLVM: [[VMAX_V1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
 // LLVM: [[VMAX_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.smax.v2i32(<2 x i32> [[VMAX_V_I]], <2 x i32> [[VMAX_V1_I]])
 // LLVM: ret <2 x i32> [[VMAX_V2_I]]
 return vmax_s32(v1, v2);
}

// LLVM-LABEL: @test_vmax_u8
// CIR-LABEL: @vmax_u8(
uint8x8_t test_vmax_u8(uint8x8_t v1, uint8x8_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.umax" %{{.*}}, %{{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>) -> !cir.vector<8 x !u8i>

 // LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[VMAX_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.umax.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
 // LLVM: ret <8 x i8> [[VMAX_V_I]]
 return vmax_u8(v1, v2);
}

// LLVM-LABEL: @test_vmax_u16
// CIR-LABEL: @vmax_u16(
uint16x4_t test_vmax_u16(uint16x4_t v1, uint16x4_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.umax" %{{.*}}, %{{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>) -> !cir.vector<4 x !u16i>

 // LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
 // LLVM: [[VMAX_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
 // LLVM: [[VMAX_V1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
 // LLVM: [[VMAX_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.umax.v4i16(<4 x i16> [[VMAX_V_I]], <4 x i16> [[VMAX_V1_I]])
 // LLVM: ret <4 x i16> [[VMAX_V2_I]]
 return vmax_u16(v1, v2);
}

// LLVM-LABEL: @test_vmax_u32
// CIR-LABEL: @vmax_u32(
uint32x2_t test_vmax_u32(uint32x2_t v1, uint32x2_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.umax" %{{.*}}, %{{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>) -> !cir.vector<2 x !u32i>

 // LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
 // LLVM: [[VMAX_V_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
 // LLVM: [[VMAX_V1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
 // LLVM: [[VMAX_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.umax.v2i32(<2 x i32> [[VMAX_V_I]], <2 x i32> [[VMAX_V1_I]])
 // LLVM: ret <2 x i32> [[VMAX_V2_I]]
 return vmax_u32(v1, v2);
}

// LLVM-LABEL: @test_vmaxq_s8
// CIR-LABEL: @vmaxq_s8(
int8x16_t test_vmaxq_s8(int8x16_t v1, int8x16_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.smax" %{{.*}}, %{{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>) -> !cir.vector<16 x !s8i>

 // LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[VMAXQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.smax.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
 // LLVM: ret <16 x i8> [[VMAXQ_V_I]]
 return vmaxq_s8(v1, v2);
}

// LLVM-LABEL: @test_vmaxq_s16
// CIR-LABEL: @vmaxq_s16(
int16x8_t test_vmaxq_s16(int16x8_t v1, int16x8_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.smax" %{{.*}}, %{{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>

 // LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
 // LLVM: [[VMAXQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
 // LLVM: [[VMAXQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
 // LLVM: [[VMAXQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smax.v8i16(<8 x i16> [[VMAXQ_V_I]], <8 x i16> [[VMAXQ_V1_I]])
 // LLVM: ret <8 x i16> [[VMAXQ_V2_I]]
 return vmaxq_s16(v1, v2);
}

// LLVM-LABEL: @test_vmaxq_s32
// CIR-LABEL: @vmaxq_s32(
int32x4_t test_vmaxq_s32(int32x4_t v1, int32x4_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.smax" %{{.*}}, %{{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>

 // LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
 // LLVM: [[VMAXQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
 // LLVM: [[VMAXQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
 // LLVM: [[VMAXQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smax.v4i32(<4 x i32> [[VMAXQ_V_I]], <4 x i32> [[VMAXQ_V1_I]])
 // LLVM: ret <4 x i32> [[VMAXQ_V2_I]]
 return vmaxq_s32(v1, v2);
}

// LLVM-LABEL: @test_vmaxq_u8
// CIR-LABEL: @vmaxq_u8(
uint8x16_t test_vmaxq_u8(uint8x16_t v1, uint8x16_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.umax" %{{.*}}, %{{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>) -> !cir.vector<16 x !u8i>

 // LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[VMAXQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.umax.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
 // LLVM: ret <16 x i8> [[VMAXQ_V_I]]
 return vmaxq_u8(v1, v2);
}

// LLVM-LABEL: @test_vmaxq_u16
// CIR-LABEL: @vmaxq_u16(
uint16x8_t test_vmaxq_u16(uint16x8_t v1, uint16x8_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.umax" %{{.*}}, %{{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>) -> !cir.vector<8 x !u16i>

 // LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
 // LLVM: [[VMAXQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
 // LLVM: [[VMAXQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
 // LLVM: [[VMAXQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umax.v8i16(<8 x i16> [[VMAXQ_V_I]], <8 x i16> [[VMAXQ_V1_I]])
 // LLVM: ret <8 x i16> [[VMAXQ_V2_I]]
 return vmaxq_u16(v1, v2);
}

// LLVM-LABEL: @test_vmaxq_u32
// CIR-LABEL: @vmaxq_u32
uint32x4_t test_vmaxq_u32(uint32x4_t v1, uint32x4_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.umax" %{{.*}}, %{{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>) -> !cir.vector<4 x !u32i>

 // LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
 // LLVM: [[VMAXQ_V_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
 // LLVM: [[VMAXQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
 // LLVM: [[VMAXQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umax.v4i32(<4 x i32> [[VMAXQ_V_I]], <4 x i32> [[VMAXQ_V1_I]])
 // LLVM: ret <4 x i32> [[VMAXQ_V2_I]]
 return vmaxq_u32(v1, v2);
}

// LLVM-LABEL: @test_vmax_f32
// CIR-LABEL: @vmax_f32
float32x2_t test_vmax_f32(float32x2_t v1, float32x2_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.fmax" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

 // LLVM-SAME: <2 x float> {{.*}} [[V1:%.*]], <2 x float> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[V1]] to <2 x i32>
 // LLVM: [[TMP1:%.*]] = bitcast <2 x float> [[V2]] to <2 x i32>
 // LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
 // LLVM: [[TMP3:%.*]] = bitcast <2 x i32> [[TMP1]] to <8 x i8>
 // LLVM: [[VMAX_V_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
 // LLVM: [[VMAX_V1_I:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x float>
 // LLVM: [[VMAX_V2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmax.v2f32(<2 x float> [[VMAX_V_I]], <2 x float> [[VMAX_V1_I]])
 // LLVM: ret <2 x float> [[VMAX_V2_I]]
 return vmax_f32(v1, v2);
}

// LLVM-LABEL: @test_vmax_f64
// CIR-LABEL: @vmax_f64
float64x1_t test_vmax_f64(float64x1_t v1, float64x1_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.fmax" %{{.*}}, %{{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>) -> !cir.vector<1 x !cir.double>

 // LLVM-SAME: <1 x double> {{.*}} [[V1:%.*]], <1 x double> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[VMAX_V_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmax.v1f64(<1 x double> {{.*}}, <1 x double> {{.*}})
 // LLVM: ret <1 x double> [[VMAX_V_I]]
 return vmax_f64(v1, v2);
}

// LLVM-LABEL: @test_vmaxq_f32
// CIR-LABEL: @vmaxq_f32
float32x4_t test_vmaxq_f32(float32x4_t v1, float32x4_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.fmax" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

 // LLVM-SAME: <4 x float> {{.*}} [[V1:%.*]], <4 x float> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[V1]] to <4 x i32>
 // LLVM: [[TMP1:%.*]] = bitcast <4 x float> [[V2]] to <4 x i32>
 // LLVM: [[TMP2:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
 // LLVM: [[TMP3:%.*]] = bitcast <4 x i32> [[TMP1]] to <16 x i8>
 // LLVM: [[VMAXQ_V_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
 // LLVM: [[VMAXQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <4 x float>
 // LLVM: [[VMAXQ_V2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmax.v4f32(<4 x float> [[VMAXQ_V_I]], <4 x float> [[VMAXQ_V1_I]])
 // LLVM: ret <4 x float> [[VMAXQ_V2_I]]
 return vmaxq_f32(v1, v2);
}

// LLVM-LABEL: @test_vmaxq_f64
// CIR-LABEL: @vmaxq_f64
float64x2_t test_vmaxq_f64(float64x2_t v1, float64x2_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.fmax" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

 // LLVM-SAME: <2 x double> {{.*}} [[V1:%.*]], <2 x double> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[V1]] to <2 x i64>
 // LLVM: [[TMP1:%.*]] = bitcast <2 x double> [[V2]] to <2 x i64>
 // LLVM: [[TMP2:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
 // LLVM: [[TMP3:%.*]] = bitcast <2 x i64> [[TMP1]] to <16 x i8>
 // LLVM: [[VMAXQ_V_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
 // LLVM: [[VMAXQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <2 x double>
 // LLVM: [[VMAXQ_V2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmax.v2f64(<2 x double> [[VMAXQ_V_I]], <2 x double> [[VMAXQ_V1_I]])
 // LLVM: ret <2 x double> [[VMAXQ_V2_I]]
 return vmaxq_f64(v1, v2);
}

// LLVM-LABEL: @test_vmaxnm_f32
// CIR-LABEL: @vmaxnm_f32
float32x2_t test_vmaxnm_f32(float32x2_t v1, float32x2_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.fmaxnm" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>) -> !cir.vector<2 x !cir.float>

 // LLVM-SAME: <2 x float> {{.*}} [[V1:%.*]], <2 x float> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[V1]] to <2 x i32>
 // LLVM: [[TMP1:%.*]] = bitcast <2 x float> [[V2]] to <2 x i32>
 // LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
 // LLVM: [[TMP3:%.*]] = bitcast <2 x i32> [[TMP1]] to <8 x i8>
 // LLVM: [[VMAXNM_V_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
 // LLVM: [[VMAXNM_V1_I:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x float>
 // LLVM: [[VMAXNM_V2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmaxnm.v2f32(<2 x float> [[VMAXNM_V_I]], <2 x float> [[VMAXNM_V1_I]])
 // LLVM: ret <2 x float> [[VMAXNM_V2_I]]
 return vmaxnm_f32(v1, v2);
}

// LLVM-LABEL: @test_vmaxnm_f64
// CIR-LABEL: @vmaxnm_f64
float64x1_t test_vmaxnm_f64(float64x1_t v1, float64x1_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.fmaxnm" %{{.*}}, %{{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>) -> !cir.vector<1 x !cir.double>

 // LLVM-SAME: <1 x double> {{.*}} [[V1:%.*]], <1 x double> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[VMAXNM_V_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmaxnm.v1f64(<1 x double> {{.*}}, <1 x double> {{.*}})
 // LLVM: ret <1 x double> [[VMAXNM_V_I]]
 return vmaxnm_f64(v1, v2);
}

// LLVM-LABEL: @test_vmaxnmq_f32
// CIR-LABEL: @vmaxnmq_f32
float32x4_t test_vmaxnmq_f32(float32x4_t v1, float32x4_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.fmaxnm" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

 // LLVM-SAME: <4 x float> {{.*}} [[V1:%.*]], <4 x float> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <4 x float> [[V1]] to <4 x i32>
 // LLVM: [[TMP1:%.*]] = bitcast <4 x float> [[V2]] to <4 x i32>
 // LLVM: [[TMP2:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
 // LLVM: [[TMP3:%.*]] = bitcast <4 x i32> [[TMP1]] to <16 x i8>
 // LLVM: [[VMAXNMQ_V_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
 // LLVM: [[VMAXNMQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <4 x float>
 // LLVM: [[VMAXNMQ_V2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmaxnm.v4f32(<4 x float> [[VMAXNMQ_V_I]], <4 x float> [[VMAXNMQ_V1_I]])
 // LLVM: ret <4 x float> [[VMAXNMQ_V2_I]]
 return vmaxnmq_f32(v1, v2);
}

// LLVM-LABEL: @test_vmaxnmq_f64
// CIR-LABEL: @vmaxnmq_f64
float64x2_t test_vmaxnmq_f64(float64x2_t v1, float64x2_t v2) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.fmaxnm" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

 // LLVM-SAME: <2 x double> {{.*}} [[V1:%.*]], <2 x double> noundef [[V2:%.*]]) {{.*}} {
 // LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[V1]] to <2 x i64>
 // LLVM: [[TMP1:%.*]] = bitcast <2 x double> [[V2]] to <2 x i64>
 // LLVM: [[TMP2:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
 // LLVM: [[TMP3:%.*]] = bitcast <2 x i64> [[TMP1]] to <16 x i8>
 // LLVM: [[VMAXNMQ_V_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
 // LLVM: [[VMAXNMQ_V1_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <2 x double>
 // LLVM: [[VMAXNMQ_V2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmaxnm.v2f64(<2 x double> [[VMAXNMQ_V_I]], <2 x double> [[VMAXNMQ_V1_I]])
 // LLVM: ret <2 x double> [[VMAXNMQ_V2_I]]
 return vmaxnmq_f64(v1, v2);
}

//===------------------------------------------------------===//
// 2.1.1.2.8. Widening Multiplication
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vmull_s8(
// CIR-LABEL: @vmull_s8(
int16x8_t test_vmull_s8(int8x8_t a, int8x8_t b) {
 // CIR: cir.call_llvm_intrinsic "aarch64.neon.smull" %{{.*}}, %{{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>) -> !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM:    [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[A]], <8 x i8> [[B]])
// LLVM-NEXT:    ret <8 x i16> [[VMULL_I]]
  return vmull_s8(a, b);
}

// LLVM-LABEL: @test_vmull_s16(
// CIR-LABEL: @vmull_s16(
int32x4_t test_vmull_s16(int16x4_t a, int16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.smull" %{{.*}}, %{{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>) -> !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
// LLVM-NEXT:    [[VMULL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VMULL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[VMULL_I]], <4 x i16> [[VMULL1_I]])
// LLVM-NEXT:    ret <4 x i32> [[VMULL2_I]]
  return vmull_s16(a, b);
}

// LLVM-LABEL: @test_vmull_s32(
// CIR-LABEL: @vmull_s32(
int64x2_t test_vmull_s32(int32x2_t a, int32x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.smull" %{{.*}}, %{{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>) -> !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
// LLVM-NEXT:    [[VMULL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VMULL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[VMULL_I]], <2 x i32> [[VMULL1_I]])
// LLVM-NEXT:    ret <2 x i64> [[VMULL2_I]]
 return vmull_s32(a, b);
}

// LLVM-LABEL: @test_vmull_u8(
// CIR-LABEL: @vmull_u8(
uint16x8_t test_vmull_u8(uint8x8_t a, uint8x8_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.umull" %{{.*}}, %{{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>) -> !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM:    [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[A]], <8 x i8> [[B]])
// LLVM-NEXT:    ret <8 x i16> [[VMULL_I]]
  return vmull_u8(a, b);
}

// LLVM-LABEL: @test_vmull_u16(
// CIR-LABEL: @vmull_u16(
uint32x4_t test_vmull_u16(uint16x4_t a, uint16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.umull" %{{.*}}, %{{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>) -> !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
// LLVM-NEXT:    [[VMULL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VMULL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[VMULL_I]], <4 x i16> [[VMULL1_I]])
// LLVM-NEXT:    ret <4 x i32> [[VMULL2_I]]
  return vmull_u16(a, b);
}

// LLVM-LABEL: @test_vmull_u32(
// CIR-LABEL: @vmull_u32(
uint64x2_t test_vmull_u32(uint32x2_t a, uint32x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.umull" %{{.*}}, %{{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>) -> !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
// LLVM-NEXT:    [[VMULL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VMULL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[VMULL_I]], <2 x i32> [[VMULL1_I]])
// LLVM-NEXT:    ret <2 x i64> [[VMULL2_I]]
  return vmull_u32(a, b);
}

// LLVM-LABEL: @test_vmull_high_s8(
// CIR-LABEL: @vmull_high_s8(
int16x8_t test_vmull_high_s8(int8x16_t a, int8x16_t b) {
// CIR: [[HIGH_A:%.*]] = cir.call @vget_high_s8
// CIR: [[HIGH_B:%.*]] = cir.call @vget_high_s8
// CIR: cir.call @vmull_s8([[HIGH_A]], [[HIGH_B]])

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM:    [[SHUFFLE_I5_I:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM-NEXT:    [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM-NEXT:    [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[SHUFFLE_I5_I]], <8 x i8> [[SHUFFLE_I_I]])
// LLVM-NEXT:    ret <8 x i16> [[VMULL_I_I]]
  return vmull_high_s8(a, b);
}

// LLVM-LABEL: @test_vmull_high_s16(
// CIR-LABEL: @vmull_high_s16(
int32x4_t test_vmull_high_s16(int16x8_t a, int16x8_t b) {
// CIR: [[HIGH_A:%.*]] = cir.call @vget_high_s16
// CIR: [[HIGH_B:%.*]] = cir.call @vget_high_s16
// CIR: {{%.*}} = cir.call @vmull_s16([[HIGH_A]], [[HIGH_B]])

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM:    [[SHUFFLE_I5_I:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM-NEXT:    [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I5_I]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// LLVM-NEXT:    [[VMULL_I_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VMULL1_I_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[VMULL_I_I]], <4 x i16> [[VMULL1_I_I]])
// LLVM-NEXT:    ret <4 x i32> [[VMULL2_I_I]]
  return vmull_high_s16(a, b);
}

// LLVM-LABEL: @test_vmull_high_s32(
// CIR-LABEL: @vmull_high_s32(
int64x2_t test_vmull_high_s32(int32x4_t a, int32x4_t b) {
// CIR: [[HIGH_A:%.*]] = cir.call @vget_high_s32
// CIR: [[HIGH_B:%.*]] = cir.call @vget_high_s32
// CIR: {{%.*}} = cir.call @vmull_s32([[HIGH_A]], [[HIGH_B]])

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM:    [[SHUFFLE_I5_I:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[A]], <2 x i32> <i32 2, i32 3>
// LLVM-NEXT:    [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I5_I]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// LLVM-NEXT:    [[VMULL_I_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VMULL1_I_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[VMULL_I_I]], <2 x i32> [[VMULL1_I_I]])
// LLVM-NEXT:    ret <2 x i64> [[VMULL2_I_I]]
  return vmull_high_s32(a, b);
}

// LLVM-LABEL: @test_vmull_high_u8(
// CIR-LABEL: @vmull_high_u8(
uint16x8_t test_vmull_high_u8(uint8x16_t a, uint8x16_t b) {
// CIR: [[HIGH_A:%.*]] = cir.call @vget_high_u8
// CIR: [[HIGH_B:%.*]] = cir.call @vget_high_u8
// CIR: {{%.*}} = cir.call @vmull_u8([[HIGH_A]], [[HIGH_B]])

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM:    [[SHUFFLE_I5_I:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM-NEXT:    [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM-NEXT:    [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[SHUFFLE_I5_I]], <8 x i8> [[SHUFFLE_I_I]])
// LLVM-NEXT:    ret <8 x i16> [[VMULL_I_I]]
  return vmull_high_u8(a, b);
}

// LLVM-LABEL: @test_vmull_high_u16(
// CIR-LABEL: @vmull_high_u16(
uint32x4_t test_vmull_high_u16(uint16x8_t a, uint16x8_t b) {
// CIR: [[HIGH_A:%.*]] = cir.call @vget_high_u16
// CIR: [[HIGH_B:%.*]] = cir.call @vget_high_u16
// CIR: {{%.*}} = cir.call @vmull_u16([[HIGH_A]], [[HIGH_B]])

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM:    [[SHUFFLE_I5_I:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM-NEXT:    [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I5_I]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// LLVM-NEXT:    [[VMULL_I_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM-NEXT:    [[VMULL1_I_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// LLVM-NEXT:    [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[VMULL_I_I]], <4 x i16> [[VMULL1_I_I]])
// LLVM-NEXT:    ret <4 x i32> [[VMULL2_I_I]]
  return vmull_high_u16(a, b);
}

// LLVM-LABEL: @test_vmull_high_u32(
// CIR-LABEL: @vmull_high_u32(
uint64x2_t test_vmull_high_u32(uint32x4_t a, uint32x4_t b) {
// CIR: [[HIGH_A:%.*]] = cir.call @vget_high_u32
// CIR: [[HIGH_B:%.*]] = cir.call @vget_high_u32
// CIR: {{%.*}} = cir.call @vmull_u32([[HIGH_A]], [[HIGH_B]])

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM:    [[SHUFFLE_I5_I:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[A]], <2 x i32> <i32 2, i32 3>
// LLVM-NEXT:    [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
// LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I5_I]] to <8 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// LLVM-NEXT:    [[VMULL_I_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM-NEXT:    [[VMULL1_I_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// LLVM-NEXT:    [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[VMULL_I_I]], <2 x i32> [[VMULL1_I_I]])
// LLVM-NEXT:    ret <2 x i64> [[VMULL2_I_I]]
  return vmull_high_u32(a, b);
}

//===------------------------------------------------------===//
// 2.1.1.3.1. Polynomial Multiply
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vmul_p8(
// CIR-LABEL: @vmul_p8(
poly8x8_t test_vmul_p8(poly8x8_t v1, poly8x8_t v2) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.pmul" %{{.*}}, %{{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>) -> !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]])
// LLVM:    [[VMUL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.pmul.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
// LLVM-NEXT:    ret <8 x i8> [[VMUL_V_I]]
  return vmul_p8(v1, v2);
}

// LLVM-LABEL: @test_vmulq_p8(
// CIR-LABEL: @vmulq_p8(
poly8x16_t test_vmulq_p8(poly8x16_t v1, poly8x16_t v2) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.pmul" %{{.*}}, %{{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>) -> !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]])
// LLVM:    [[VMULQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.pmul.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
// LLVM-NEXT:    ret <16 x i8> [[VMULQ_V_I]]
  return vmulq_p8(v1, v2);
}

// LLVM-LABEL: @test_vmull_p8(
// CIR-LABEL: @vmull_p8(
poly16x8_t test_vmull_p8(poly8x8_t a, poly8x8_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.pmull" %{{.*}}, %{{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>) -> !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM:    [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8> [[A]], <8 x i8> [[B]])
// LLVM-NEXT:    ret <8 x i16> [[VMULL_I]]
  return vmull_p8(a, b);
}

// LLVM-LABEL: @test_vmull_high_p8(
// CIR-LABEL: @vmull_high_p8(
poly16x8_t test_vmull_high_p8(poly8x16_t a, poly8x16_t b) {
// CIR: [[HIGH_A:%.*]] = cir.call @vget_high_p8
// CIR: [[HIGH_B:%.*]] = cir.call @vget_high_p8
// CIR: {{%.*}} = cir.call @vmull_p8([[HIGH_A]], [[HIGH_B]])

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM:    [[SHUFFLE_I5:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM-NEXT:    [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8> [[SHUFFLE_I5]], <8 x i8> [[SHUFFLE_I]])
// LLVM-NEXT:    ret <8 x i16> [[VMULL_I_I]]
  return vmull_high_p8(a, b);
}

//===------------------------------------------------------===//
// 2.1.3.1.1. Vector Shift Left
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#vector-shift-left
//===------------------------------------------------------===//

// ALL-LABEL: test_vshld_n_s64
int64_t test_vshld_n_s64(int64_t a) {
  // CIR: cir.shift(left, {{.*}})

  // LLVM-SAME: i64 {{.*}} [[A:%.*]])
  // LLVM: [[SHL_N:%.*]] = shl i64 [[A]], 1
  // LLVM: ret i64 [[SHL_N]]
  return (int64_t)vshld_n_s64(a, 1);
}

// ALL-LABEL: test_vshld_n_u64
int64_t test_vshld_n_u64(int64_t a) {
  // CIR: cir.shift(left, {{.*}})

  // LLVM-SAME: i64 {{.*}} [[A:%.*]])
  // LLVM: [[SHL_N:%.*]] = shl i64 [[A]], 1
  // LLVM: ret i64 [[SHL_N]]
  return (int64_t)vshld_n_u64(a, 1);
}

// LLVM-LABEL: test_vshld_s64
// CIR-LABEL: vshld_s64
int64_t test_vshld_s64(int64_t a,int64_t b) {
 // CIR:  cir.call_llvm_intrinsic "aarch64.neon.sshl" %{{.*}}, %{{.*}} : (!s64i, !s64i) -> !s64i

 // LLVM-SAME: i64 {{.*}} [[A:%.*]], i64 {{.*}} [[B:%.*]]) #[[ATTR0:[0-9]+]] {
 // LLVM:    [[VSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sshl.i64(i64 [[A]], i64 [[B]])
 // LLVM:    ret i64 [[VSHLD_S64_I]]
  return (int64_t)vshld_s64(a, b);
}

// LLVM-LABEL: test_vshld_u64
// CIR-LABEL: vshld_u64
int64_t test_vshld_u64(int64_t a,int64_t b) {
 // CIR:  cir.call_llvm_intrinsic "aarch64.neon.ushl" %{{.*}}, %{{.*}} : (!u64i, !s64i) -> !u64i

 // LLVM-SAME: i64 {{.*}} [[A:%.*]], i64 {{.*}} [[B:%.*]]) #[[ATTR0:[0-9]+]] {
 // LLVM:    [[VSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.ushl.i64(i64 [[A]], i64 [[B]])
 // LLVM:    ret i64 [[VSHLD_S64_I]]
  return (int64_t)vshld_u64(a, b);
}

// ALL-LABEL: test_vshlq_n_s8
int8x16_t test_vshlq_n_s8(int8x16_t a) {
// CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<16 x !s8i>, %{{.*}} : !cir.vector<16 x !s8i>) -> !cir.vector<16 x !s8i>
  
// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[VSHL_N:%.*]] = shl <16 x i8> [[A]], splat (i8 3)
// LLVM:    ret <16 x i8> [[VSHL_N]]
//
 return vshlq_n_s8(a, 3);
}

// ALL-LABEL: test_vshlq_n_s16
int16x8_t test_vshlq_n_s16(int16x8_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<8 x !s16i>, %{{.*}} : !cir.vector<8 x !s16i>) -> !cir.vector<8 x !s16i>
  
// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
// LLVM-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM:    [[VSHL_N:%.*]] = shl <8 x i16> [[TMP1]], splat (i16 3)
// LLVM:    ret <8 x i16> [[VSHL_N]]
 return vshlq_n_s16(a, 3);
}

// ALL-LABEL: test_vshlq_n_s32
int32x4_t test_vshlq_n_s32(int32x4_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<4 x !s32i>, %{{.*}} : !cir.vector<4 x !s32i>) -> !cir.vector<4 x !s32i>
  
// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM:    [[VSHL_N:%.*]] = shl <4 x i32> [[TMP1]], splat (i32 3)
// LLVM:    ret <4 x i32> [[VSHL_N]]
 return vshlq_n_s32(a, 3);
}

// ALL-LABEL: test_vshlq_n_s64
int64x2_t test_vshlq_n_s64(int64x2_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<2 x !s64i>, %{{.*}} : !cir.vector<2 x !s64i>) -> !cir.vector<2 x !s64i>
  
// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// LLVM:    [[VSHL_N:%.*]] = shl <2 x i64> [[TMP1]], splat (i64 3)
// LLVM:    ret <2 x i64> [[VSHL_N]]
 return vshlq_n_s64(a, 3);
}

// ALL-LABEL: test_vshlq_n_u8
uint8x16_t test_vshlq_n_u8(uint8x16_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<16 x !u8i>, %{{.*}} : !cir.vector<16 x !u8i>) -> !cir.vector<16 x !u8i>
  
// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[VSHL_N:%.*]] = shl <16 x i8> [[A]], splat (i8 3)
// LLVM:    ret <16 x i8> [[VSHL_N]]
 return vshlq_n_u8(a, 3);
}

// ALL-LABEL: test_vshlq_n_u16
uint16x8_t test_vshlq_n_u16(uint16x8_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<8 x !u16i>, %{{.*}} : !cir.vector<8 x !u16i>) -> !cir.vector<8 x !u16i>
  
// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// LLVM:    [[VSHL_N:%.*]] = shl <8 x i16> [[TMP1]], splat (i16 3)
// LLVM:    ret <8 x i16> [[VSHL_N]]
 return vshlq_n_u16(a, 3);
}

// ALL-LABEL: test_vshlq_n_u32
uint32x4_t test_vshlq_n_u32(uint32x4_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<4 x !u32i>, %{{.*}} : !cir.vector<4 x !u32i>) -> !cir.vector<4 x !u32i>
  
// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// LLVM:    [[VSHL_N:%.*]] = shl <4 x i32> [[TMP1]], splat (i32 3)
// LLVM:    ret <4 x i32> [[VSHL_N]]
 return vshlq_n_u32(a, 3);
}

// ALL-LABEL: test_vshlq_n_u64
uint64x2_t test_vshlq_n_u64(uint64x2_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<2 x !u64i>, %{{.*}} : !cir.vector<2 x !u64i>) -> !cir.vector<2 x !u64i>
  
// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// LLVM:    [[VSHL_N:%.*]] = shl <2 x i64> [[TMP1]], splat (i64 3)
// LLVM:    ret <2 x i64> [[VSHL_N]]
 return vshlq_n_u64(a, 3);
}

// ALL-LABEL: test_vshl_n_s8
int8x8_t test_vshl_n_s8(int8x8_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<8 x !s8i>, %{{.*}} : !cir.vector<8 x !s8i>) -> !cir.vector<8 x !s8i>
  
// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[VSHL_N:%.*]] = shl <8 x i8> [[A]], splat (i8 3)
// LLVM:    ret <8 x i8> [[VSHL_N]]
 return vshl_n_s8(a, 3);
}

// ALL-LABEL: test_vshl_n_s16
int16x4_t test_vshl_n_s16(int16x4_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<4 x !s16i>, %{{.*}} : !cir.vector<4 x !s16i>) -> !cir.vector<4 x !s16i>
  
// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM:    [[VSHL_N:%.*]] = shl <4 x i16> [[TMP1]], splat (i16 3)
// LLVM:    ret <4 x i16> [[VSHL_N]]
 return vshl_n_s16(a, 3);
}

// ALL-LABEL: test_vshl_n_s32
int32x2_t test_vshl_n_s32(int32x2_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<2 x !s32i>, %{{.*}} : !cir.vector<2 x !s32i>) -> !cir.vector<2 x !s32i>
  
// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM:    [[VSHL_N:%.*]] = shl <2 x i32> [[TMP1]], splat (i32 3)
// LLVM:    ret <2 x i32> [[VSHL_N]]
 return vshl_n_s32(a, 3);
}

// ALL-LABEL: test_vshl_n_s64
int64x1_t test_vshl_n_s64(int64x1_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<1 x !s64i>, %{{.*}} : !cir.vector<1 x !s64i>) -> !cir.vector<1 x !s64i>
  
// LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// LLVM:    [[VSHL_N:%.*]] = shl <1 x i64> [[TMP1]], splat (i64 1)
// LLVM:    ret <1 x i64> [[VSHL_N]]
 return vshl_n_s64(a, 1);
}

// ALL-LABEL: test_vshl_n_u8
uint8x8_t test_vshl_n_u8(uint8x8_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<8 x !u8i>, %{{.*}} : !cir.vector<8 x !u8i>) -> !cir.vector<8 x !u8i>
  
// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[VSHL_N:%.*]] = shl <8 x i8> [[A]], splat (i8 3)
// LLVM:    ret <8 x i8> [[VSHL_N]]
 return vshl_n_u8(a, 3);
}

// ALL-LABEL: test_vshl_n_u16
uint16x4_t test_vshl_n_u16(uint16x4_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<4 x !u16i>, %{{.*}} : !cir.vector<4 x !u16i>) -> !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM:    [[VSHL_N:%.*]] = shl <4 x i16> [[TMP1]], splat (i16 3)
// LLVM:    ret <4 x i16> [[VSHL_N]]
 return vshl_n_u16(a, 3);
}

// ALL-LABEL: test_vshl_n_u32
uint32x2_t test_vshl_n_u32(uint32x2_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<2 x !u32i>, %{{.*}} : !cir.vector<2 x !u32i>) -> !cir.vector<2 x !u32i>
  
// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM:    [[VSHL_N:%.*]] = shl <2 x i32> [[TMP1]], splat (i32 3)
// LLVM:    ret <2 x i32> [[VSHL_N]]
 return vshl_n_u32(a, 3);
}

// ALL-LABEL: test_vshl_n_u64
uint64x1_t test_vshl_n_u64(uint64x1_t a) {
 // CIR: [[RES:%.*]] = cir.shift(left, %{{.*}} : !cir.vector<1 x !u64i>, %{{.*}} : !cir.vector<1 x !u64i>) -> !cir.vector<1 x !u64i>
  
// LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM:    [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
// LLVM:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// LLVM:    [[VSHL_N:%.*]] = shl <1 x i64> [[TMP1]], splat (i64 1)
// LLVM:    ret <1 x i64> [[VSHL_N]]
 return vshl_n_u64(a, 1);
}

//===------------------------------------------------------===//
// 2.1.8.5 Bitwise select 
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#markdown-toc-bitwise-select
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vbsl_s8(
// CIR-LABEL: @vbsl_s8(
int8x8_t test_vbsl_s8(uint8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CIR: [[MASK_PTR:%.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[AND:%.*]] = cir.and %{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>
  // CIR: [[NOT:%.*]] = cir.not %{{.*}} : !cir.vector<8 x !s8i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], %{{.*}} : !cir.vector<8 x !s8i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<8 x !s8i>

  // LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]], <8 x i8> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[VBSL_I:%.*]] = and <8 x i8> [[V1]], [[V2]]
  // LLVM: [[TMP0:%.*]] = xor <8 x i8> [[V1]], splat (i8 -1)
  // LLVM: [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], [[V3]]
  // LLVM: [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: ret <8 x i8> [[VBSL2_I]]
  return vbsl_s8(v1, v2, v3);
}
  
// LLVM-LABEL: @test_vbslq_s8(
// CIR-LABEL: @vbslq_s8(
int8x16_t test_vbslq_s8(uint8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[AND:%.*]] = cir.and %{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>
  // CIR: [[NOT:%.*]] = cir.not %{{.*}} : !cir.vector<16 x !s8i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], %{{.*}} : !cir.vector<16 x !s8i>
  // CIR: cir.or [[AND]], [[AND2]] : !cir.vector<16 x !s8i>

  // LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]], <16 x i8> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[VBSL_I:%.*]] = and <16 x i8> [[V1]], [[V2]]
  // LLVM: [[TMP0:%.*]] = xor <16 x i8> [[V1]], splat (i8 -1)
  // LLVM: [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], [[V3]]
  // LLVM: [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: ret <16 x i8> [[VBSL2_I]]
  return vbslq_s8(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_s16(
// CIR-LABEL: @vbsl_s16(
int8x8_t test_vbsl_s16(uint16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !s16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !s16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<4 x !s16i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<4 x !s16i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<4 x !s16i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<4 x !s16i>

  // LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]], <4 x i16> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x i16> [[V3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
  // LLVM: [[VBSL3_I:%.*]] = and <4 x i16> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <4 x i16> [[VBSL_I]], splat (i16 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: [[TMP4:%.*]] = bitcast <4 x i16> [[VBSL5_I]] to <8 x i8>
  // LLVM: ret <8 x i8> [[TMP4]]
  return (int8x8_t)vbsl_s16(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_s16(
// CIR-LABEL: @vbslq_s16(
int16x8_t test_vbslq_s16(uint16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !s16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !s16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<8 x !s16i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<8 x !s16i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<8 x !s16i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<8 x !s16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]], <8 x i16> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i16> [[V3]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
  // LLVM: [[VBSL3_I:%.*]] = and <8 x i16> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <8 x i16> [[VBSL_I]], splat (i16 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <8 x i16> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <8 x i16> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <8 x i16> [[VBSL5_I]]
  return vbslq_s16(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_s32(
// CIR-LABEL: @vbsl_s32(
int32x2_t test_vbsl_s32(uint32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u32i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !s32i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !s32i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<2 x !s32i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<2 x !s32i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<2 x !s32i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<2 x !s32i>

  // LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> {{.*}} [[V2:%.*]], <2 x i32> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[V3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
  // LLVM: [[VBSL3_I:%.*]] = and <2 x i32> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <2 x i32> [[VBSL_I]], splat (i32 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <2 x i32> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <2 x i32> [[VBSL5_I]]
  return vbsl_s32(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_s32(
// CIR-LABEL: @vbslq_s32(
int32x4_t test_vbslq_s32(uint32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u32i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !s32i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !s32i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<4 x !s32i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<4 x !s32i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<4 x !s32i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<4 x !s32i>

  // LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> {{.*}} [[V2:%.*]], <4 x i32> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x i32> [[V3]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
  // LLVM: [[VBSL3_I:%.*]] = and <4 x i32> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <4 x i32> [[VBSL_I]], splat (i32 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <4 x i32> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <4 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <4 x i32> [[VBSL5_I]]
  return vbslq_s32(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_s64(
// CIR-LABEL: @vbsl_s64(
int64x1_t test_vbsl_s64(uint64x1_t v1, int64x1_t v2, int64x1_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !u64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !s64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !s64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<1 x !s64i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<1 x !s64i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<1 x !s64i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<1 x !s64i>

  // LLVM-SAME: <1 x i64> {{.*}} [[V1:%.*]], <1 x i64> {{.*}} [[V2:%.*]], <1 x i64> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <1 x i64> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <1 x i64> [[V2]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[V3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x i64>
  // LLVM: [[VBSL3_I:%.*]] = and <1 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <1 x i64> [[VBSL_I]], splat (i64 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <1 x i64> [[VBSL5_I]]
  return vbsl_s64(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_s64(
// CIR-LABEL: @vbslq_s64(
int64x2_t test_vbslq_s64(uint64x2_t v1, int64x2_t v2, int64x2_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !s64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !s64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<2 x !s64i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<2 x !s64i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<2 x !s64i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<2 x !s64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[V1:%.*]], <2 x i64> {{.*}} [[V2:%.*]], <2 x i64> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <2 x i64> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[V2]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x i64> [[V3]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x i64>
  // LLVM: [[VBSL3_I:%.*]] = and <2 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <2 x i64> [[VBSL_I]], splat (i64 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <2 x i64> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <2 x i64> [[VBSL5_I]]
  return vbslq_s64(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_u8(
// CIR-LABEL: @vbsl_u8(
uint8x8_t test_vbsl_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<8 x !u8i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<8 x !u8i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<8 x !u8i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<8 x !u8i>

  // LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]], <8 x i8> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[VBSL_I:%.*]] = and <8 x i8> [[V1]], [[V2]]
  // LLVM: [[TMP0:%.*]] = xor <8 x i8> [[V1]], splat (i8 -1)
  // LLVM: [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], [[V3]]
  // LLVM: [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: ret <8 x i8> [[VBSL2_I]]
  return vbsl_u8(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_u8(
// CIR-LABEL: @vbslq_u8(
uint8x16_t test_vbslq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<16 x !u8i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<16 x !u8i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<16 x !u8i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<16 x !u8i>

  // LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]], <16 x i8> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[VBSL_I:%.*]] = and <16 x i8> [[V1]], [[V2]]
  // LLVM: [[TMP0:%.*]] = xor <16 x i8> [[V1]], splat (i8 -1)
  // LLVM: [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], [[V3]]
  // LLVM: [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: ret <16 x i8> [[VBSL2_I]]
  return vbslq_u8(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_u16(
// CIR-LABEL: @vbsl_u16(
uint16x4_t test_vbsl_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !u16i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<4 x !u16i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<4 x !u16i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<4 x !u16i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<4 x !u16i>

  // LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]], <4 x i16> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x i16> [[V3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
  // LLVM: [[VBSL3_I:%.*]] = and <4 x i16> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <4 x i16> [[VBSL_I]], splat (i16 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <4 x i16> [[VBSL5_I]]
  return vbsl_u16(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_u16(
// CIR-LABEL: @vbslq_u16(
uint16x8_t test_vbslq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !u16i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<8 x !u16i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<8 x !u16i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<8 x !u16i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<8 x !u16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]], <8 x i16> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i16> [[V3]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
  // LLVM: [[VBSL3_I:%.*]] = and <8 x i16> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <8 x i16> [[VBSL_I]], splat (i16 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <8 x i16> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <8 x i16> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <8 x i16> [[VBSL5_I]]
  return vbslq_u16(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_u32(
// CIR-LABEL: @vbsl_u32(
uint32x2_t test_vbsl_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u32i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u32i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u32i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !u32i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<2 x !u32i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<2 x !u32i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<2 x !u32i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<2 x !u32i>

  // LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> {{.*}} [[V2:%.*]], <2 x i32> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[V3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
  // LLVM: [[VBSL3_I:%.*]] = and <2 x i32> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <2 x i32> [[VBSL_I]], splat (i32 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <2 x i32> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <2 x i32> [[VBSL5_I]]
  return vbsl_u32(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_u32(
// CIR-LABEL: @vbslq_u32(
uint32x4_t test_vbslq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u32i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u32i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u32i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !u32i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<4 x !u32i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<4 x !u32i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<4 x !u32i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<4 x !u32i>

  // LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> {{.*}} [[V2:%.*]], <4 x i32> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x i32> [[V3]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
  // LLVM: [[VBSL3_I:%.*]] = and <4 x i32> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <4 x i32> [[VBSL_I]], splat (i32 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <4 x i32> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <4 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <4 x i32> [[VBSL5_I]]
  return vbslq_u32(v1, v2, v3);
}


// LLVM-LABEL: @test_vbsl_u64(
// CIR-LABEL: @vbsl_u64(
uint64x1_t test_vbsl_u64(uint64x1_t v1, uint64x1_t v2, uint64x1_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !u64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !u64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !u64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !u64i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !u64i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !u64i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<1 x !u64i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<1 x !u64i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<1 x !u64i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<1 x !u64i>

  // LLVM-SAME: <1 x i64> {{.*}} [[V1:%.*]], <1 x i64> {{.*}} [[V2:%.*]], <1 x i64> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <1 x i64> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <1 x i64> [[V2]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[V3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x i64>
  // LLVM: [[VBSL3_I:%.*]] = and <1 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <1 x i64> [[VBSL_I]], splat (i64 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <1 x i64> [[VBSL5_I]]
  return vbsl_u64(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_u64(
// CIR-LABEL: @vbslq_u64(
uint64x2_t test_vbslq_u64(uint64x2_t v1, uint64x2_t v2, uint64x2_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !u64i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !u64i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !u64i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<2 x !u64i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<2 x !u64i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<2 x !u64i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<2 x !u64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[V1:%.*]], <2 x i64> {{.*}} [[V2:%.*]], <2 x i64> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <2 x i64> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[V2]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x i64> [[V3]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x i64>
  // LLVM: [[VBSL3_I:%.*]] = and <2 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <2 x i64> [[VBSL_I]], splat (i64 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <2 x i64> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <2 x i64> [[VBSL5_I]]
  return vbslq_u64(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_f32(
// CIR-LABEL: @vbsl_f32(
float32x2_t test_vbsl_f32(uint32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u32i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !cir.float>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !cir.float>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<2 x !s32i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<2 x !s32i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<2 x !s32i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<2 x !s32i>
  // CIR: [[OR:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<2 x !s32i>
  // CIR: cir.cast bitcast [[OR]] : !cir.vector<2 x !s32i> -> !cir.vector<2 x !cir.float>

  // LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x float> {{.*}} [[V2:%.*]], <2 x float> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <2 x float> [[V2]] to <2 x i32>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x float> [[V3]] to <2 x i32>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
  // LLVM: [[TMP3:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
  // LLVM: [[TMP4:%.*]] = bitcast <2 x i32> [[TMP1]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x i32>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
  // LLVM: [[VBSL3_I:%.*]] = and <2 x i32> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP5:%.*]] = xor <2 x i32> [[VBSL_I]], splat (i32 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <2 x i32> [[TMP5]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: [[TMP6:%.*]] = bitcast <2 x i32> [[VBSL5_I]] to <2 x float>
  // LLVM: ret <2 x float> [[TMP6]]
  return vbsl_f32(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_f32(
// CIR-LABEL: @vbslq_f32(
float32x4_t test_vbslq_f32(uint32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u32i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !cir.float>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<4 x !s32i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<4 x !s32i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<4 x !s32i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<4 x !s32i>
  // CIR: [[OR:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast [[OR]] : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x float> {{.*}} [[V2:%.*]], <4 x float> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <4 x float> [[V2]] to <4 x i32>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x float> [[V3]] to <4 x i32>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
  // LLVM: [[TMP3:%.*]] = bitcast <4 x i32> [[TMP0]] to <16 x i8>
  // LLVM: [[TMP4:%.*]] = bitcast <4 x i32> [[TMP1]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <4 x i32>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
  // LLVM: [[VBSL3_I:%.*]] = and <4 x i32> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP5:%.*]] = xor <4 x i32> [[VBSL_I]], splat (i32 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <4 x i32> [[TMP5]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <4 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: [[TMP6:%.*]] = bitcast <4 x i32> [[VBSL5_I]] to <4 x float>
  // LLVM: ret <4 x float> [[TMP6]]
  return vbslq_f32(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_p8(
// CIR-LABEL: @vbsl_p8(
poly8x8_t test_vbsl_p8(uint8x8_t v1, poly8x8_t v2, poly8x8_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[AND:%.*]] = cir.and %{{.*}}, %{{.*}} : !cir.vector<8 x !s8i>
  // CIR: [[NOT:%.*]] = cir.not %{{.*}} : !cir.vector<8 x !s8i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], %{{.*}} : !cir.vector<8 x !s8i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<8 x !s8i>

  // LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]], <8 x i8> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[VBSL_I:%.*]] = and <8 x i8> [[V1]], [[V2]]
  // LLVM: [[TMP0:%.*]] = xor <8 x i8> [[V1]], splat (i8 -1)
  // LLVM: [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], [[V3]]
  // LLVM: [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: ret <8 x i8> [[VBSL2_I]]
  return vbsl_p8(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_p8(
// CIR-LABEL: @vbslq_p8(
poly8x16_t test_vbslq_p8(uint8x16_t v1, poly8x16_t v2, poly8x16_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[AND:%.*]] = cir.and %{{.*}}, %{{.*}} : !cir.vector<16 x !s8i>
  // CIR: [[NOT:%.*]] = cir.not %{{.*}} : !cir.vector<16 x !s8i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], %{{.*}} : !cir.vector<16 x !s8i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<16 x !s8i>

  // LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]], <16 x i8> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[VBSL_I:%.*]] = and <16 x i8> [[V1]], [[V2]]
  // LLVM: [[TMP0:%.*]] = xor <16 x i8> [[V1]], splat (i8 -1)
  // LLVM: [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], [[V3]]
  // LLVM: [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: ret <16 x i8> [[VBSL2_I]]
  return vbslq_p8(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_p16(
// CIR-LABEL: @vbsl_p16(
poly16x4_t test_vbsl_p16(uint16x4_t v1, poly16x4_t v2, poly16x4_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<4 x !u16i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<4 x !s16i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<4 x !s16i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<4 x !s16i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<4 x !s16i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<4 x !s16i>

  // LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]], <4 x i16> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <4 x i16> [[V3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
  // LLVM: [[VBSL3_I:%.*]] = and <4 x i16> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <4 x i16> [[VBSL_I]], splat (i16 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <4 x i16> [[VBSL5_I]]
  return vbsl_p16(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_p16(
// CIR-LABEL: @vbslq_p16(
poly16x8_t test_vbslq_p16(uint16x8_t v1, poly16x8_t v2, poly16x8_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u16i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<8 x !s16i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<8 x !s16i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<8 x !s16i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<8 x !s16i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<8 x !s16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]], <8 x i16> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i16> [[V3]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
  // LLVM: [[VBSL3_I:%.*]] = and <8 x i16> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <8 x i16> [[VBSL_I]], splat (i16 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <8 x i16> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <8 x i16> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <8 x i16> [[VBSL5_I]]
  return vbslq_p16(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_p64(
// CIR-LABEL: @vbsl_p64(
poly64x1_t test_vbsl_p64(poly64x1_t v1, poly64x1_t v2, poly64x1_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !u64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !u64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !u64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<1 x !s64i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<1 x !s64i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<1 x !s64i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<1 x !s64i>

  // LLVM-SAME: <1 x i64> {{.*}} [[V1:%.*]], <1 x i64> {{.*}} [[V2:%.*]], <1 x i64> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <1 x i64> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <1 x i64> [[V2]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <1 x i64> [[V3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x i64>
  // LLVM: [[VBSL3_I:%.*]] = and <1 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <1 x i64> [[VBSL_I]], splat (i64 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <1 x i64> [[VBSL5_I]]
  return vbsl_p64(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_p64(
// CIR-LABEL: @vbslq_p64(
poly64x2_t test_vbslq_p64(poly64x2_t v1, poly64x2_t v2, poly64x2_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<2 x !s64i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<2 x !s64i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<2 x !s64i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<2 x !s64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[V1:%.*]], <2 x i64> {{.*}} [[V2:%.*]], <2 x i64> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <2 x i64> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i64> [[V2]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x i64> [[V3]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x i64>
  // LLVM: [[VBSL3_I:%.*]] = and <2 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP3:%.*]] = xor <2 x i64> [[VBSL_I]], splat (i64 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <2 x i64> [[TMP3]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: ret <2 x i64> [[VBSL5_I]]
  return vbslq_p64(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_f64(
// CIR-LABEL: @vbsl_f64(
float64x1_t test_vbsl_f64(uint64x1_t v1, float64x1_t v2, float64x1_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !u64i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !cir.double>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<1 x !cir.double>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<1 x !s64i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<1 x !s64i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<1 x !s64i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<1 x !s64i>
  // CIR: [[OR:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<1 x !s64i>
  // CIR: cir.cast bitcast [[OR]] : !cir.vector<1 x !s64i> -> !cir.vector<1 x !cir.double>

  // LLVM-SAME: <1 x i64> {{.*}} [[V1:%.*]], <1 x double> {{.*}} [[V2:%.*]], <1 x double> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <1 x double> [[V2]] to i64
  // LLVM: [[TMP1:%.*]] = insertelement <1 x i64> undef, i64 [[TMP0]], i32 0
  // LLVM: [[TMP2:%.*]] = bitcast <1 x double> [[V3]] to i64
  // LLVM: [[TMP3:%.*]] = insertelement <1 x i64> undef, i64 [[TMP2]], i32 0
  // LLVM: [[TMP4:%.*]] = bitcast <1 x i64> [[V1]] to <8 x i8>
  // LLVM: [[TMP5:%.*]] = bitcast <1 x i64> [[TMP1]] to <8 x i8>
  // LLVM: [[TMP6:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP5]] to <1 x i64>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
  // LLVM: [[VBSL3_I:%.*]] = and <1 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP7:%.*]] = xor <1 x i64> [[VBSL_I]], splat (i64 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <1 x i64> [[TMP7]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: [[TMP8:%.*]] = bitcast <1 x i64> [[VBSL5_I]] to <1 x double>
  // LLVM: ret <1 x double> [[TMP8]]
  return vbsl_f64(v1, v2, v3);
}

// LLVM-LABEL: @test_vbslq_f64(
// CIR-LABEL: @vbslq_f64(
float64x2_t test_vbslq_f64(uint64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !u64i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !cir.double>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<2 x !cir.double>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<2 x !s64i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<2 x !s64i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<2 x !s64i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<2 x !s64i>
  // CIR: [[OR:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<2 x !s64i>
  // CIR: cir.cast bitcast [[OR]] : !cir.vector<2 x !s64i> -> !cir.vector<2 x !cir.double>

  // LLVM-SAME: <2 x i64> {{.*}} [[V1:%.*]], <2 x double> {{.*}} [[V2:%.*]], <2 x double> {{.*}} [[V3:%.*]]) {{.*}} {
  // LLVM:      [[TMP0:%.*]] = bitcast <2 x double> [[V2]] to <2 x i64>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x double> [[V3]] to <2 x i64>
  // LLVM: [[TMP2:%.*]] = bitcast <2 x i64> [[V1]] to <16 x i8>
  // LLVM: [[TMP3:%.*]] = bitcast <2 x i64> [[TMP0]] to <16 x i8>
  // LLVM: [[TMP4:%.*]] = bitcast <2 x i64> [[TMP1]] to <16 x i8>
  // LLVM: [[VBSL_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x i64>
  // LLVM: [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <2 x i64>
  // LLVM: [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
  // LLVM: [[VBSL3_I:%.*]] = and <2 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: [[TMP5:%.*]] = xor <2 x i64> [[VBSL_I]], splat (i64 -1)
  // LLVM: [[VBSL4_I:%.*]] = and <2 x i64> [[TMP5]], [[VBSL2_I]]
  // LLVM: [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM: [[TMP6:%.*]] = bitcast <2 x i64> [[VBSL5_I]] to <2 x double>
  // LLVM: ret <2 x double> [[TMP6]]
  return vbslq_f64(v1, v2, v3);
}

// LLVM-LABEL: @test_vbsl_mf8(
// CIR-LABEL: @vbsl_mf8(
mfloat8x8_t test_vbsl_mf8(uint8x8_t v1, mfloat8x8_t v2, mfloat8x8_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<8 x !u8i>> -> !cir.ptr<!cir.vector<8 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s8i> -> !cir.vector<8 x !u8i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<8 x !u8i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<8 x !u8i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<8 x !u8i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<8 x !u8i>

  // LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> [[V2:%.*]], <8 x i8> [[V3:%.*]]) {{.*}} {
  // LLVM:      [[VBSL_I:%.*]] = and <8 x i8> [[V1]], [[V2]]
  // LLVM: [[TMP0:%.*]] = xor <8 x i8> [[V1]], splat (i8 -1)
  // LLVM: [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], [[V3]]
  // LLVM: [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: ret <8 x i8> [[VBSL2_I]]
  return vbsl_mf8(v1, v2, v3);
}  

// LLVM-LABEL: @test_vbslq_mf8(
// CIR-LABEL: @vbslq_mf8(
mfloat8x16_t test_vbslq_mf8(uint8x16_t v1, mfloat8x16_t v2, mfloat8x16_t v3) {
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!cir.vector<16 x !u8i>> -> !cir.ptr<!cir.vector<16 x !s8i>>
  // CIR: [[MASK:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
  // CIR: [[VAL1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
  // CIR: [[VAL2:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<16 x !s8i> -> !cir.vector<16 x !u8i>
  // CIR: [[AND:%.*]] = cir.and [[MASK]], [[VAL1]] : !cir.vector<16 x !u8i>
  // CIR: [[NOT:%.*]] = cir.not [[MASK]] : !cir.vector<16 x !u8i>
  // CIR: [[AND2:%.*]] = cir.and [[NOT]], [[VAL2]] : !cir.vector<16 x !u8i>
  // CIR: [[RES:%.*]] = cir.or [[AND]], [[AND2]] : !cir.vector<16 x !u8i>

  // LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> [[V2:%.*]], <16 x i8> [[V3:%.*]]) {{.*}} {
  // LLVM:      [[VBSL_I:%.*]] = and <16 x i8> [[V1]], [[V2]]
  // LLVM: [[TMP0:%.*]] = xor <16 x i8> [[V1]], splat (i8 -1)
  // LLVM: [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], [[V3]]
  // LLVM: [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM: ret <16 x i8> [[VBSL2_I]]
  return vbslq_mf8(v1, v2, v3);
}
