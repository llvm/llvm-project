// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

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
// CIR: cir.unary(minus, {{.*}}) : !s64

// LLVM-SAME: i64 {{.*}} [[A:%.*]])
// LLVM:          [[VNEGD_I:%.*]] = sub i64 0, [[A]]
// LLVM-NEXT:     ret i64 [[VNEGD_I]]
  return (int64_t)vnegd_s64(a);
}

//===------------------------------------------------------===//
// 2.1.2.2 Bitwise equal to zero
//===------------------------------------------------------===//
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

// TODO SISD variants:
// vceqzd_u64, vceqzs_f32, vceqzd_f64


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

// TODO SISD variants:
// TODO @vabdd_f64(a, b);
// TODO @test_vabds_f32(

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
