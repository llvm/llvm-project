// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefixes=CIR %}
// Narrowing-subtraction checks use instcombine to fold no-op bitcasts (LLVM-IC prefix).
// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefixes=LLVM-IC
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefixes=LLVM-IC %}

//=============================================================================
// NOTES
//
// Tests for vector subtraction intrinsics: Subtraction, Widening subtraction, Narrowing subtraction and Saturating subtract elements.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#subtract
//
// TODO: Migrate Saturating subtract test cases.
//
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.1.1.5.1.  Subtraction
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#subtraction
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vsub_s8(
// CIR-LABEL: @vsub_s8(
int8x8_t test_vsub_s8(int8x8_t v1, int8x8_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<8 x !s8i>

// LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <8 x i8> [[V1]], [[V2]]
// LLVM: ret <8 x i8> [[SUB_I]]
  return vsub_s8(v1, v2);
}

// LLVM-LABEL: @test_vsubq_s8(
// CIR-LABEL: @vsubq_s8(
int8x16_t test_vsubq_s8(int8x16_t v1, int8x16_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<16 x !s8i>

// LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <16 x i8> [[V1]], [[V2]]
// LLVM: ret <16 x i8> [[SUB_I]]
  return vsubq_s8(v1, v2);
}

// LLVM-LABEL: @test_vsub_s16(
// CIR-LABEL: @vsub_s16(
int16x4_t test_vsub_s16(int16x4_t v1, int16x4_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<4 x !s16i>

// LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <4 x i16> [[V1]], [[V2]]
// LLVM: ret <4 x i16> [[SUB_I]]
  return vsub_s16(v1, v2);
}

// LLVM-LABEL: @test_vsubq_s16(
// CIR-LABEL: @vsubq_s16(
int16x8_t test_vsubq_s16(int16x8_t v1, int16x8_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[V1]], [[V2]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubq_s16(v1, v2);
}

// LLVM-LABEL: @test_vsub_s32(
// CIR-LABEL: @vsub_s32(
int32x2_t test_vsub_s32(int32x2_t v1, int32x2_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<2 x !s32i>

// LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <2 x i32> [[V1]], [[V2]]
// LLVM: ret <2 x i32> [[SUB_I]]
  return vsub_s32(v1, v2);
}

// LLVM-LABEL: @test_vsubq_s32(
// CIR-LABEL: @vsubq_s32(
int32x4_t test_vsubq_s32(int32x4_t v1, int32x4_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[V1]], [[V2]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubq_s32(v1, v2);
}

// LLVM-LABEL: @test_vsub_s64(
// CIR-LABEL: @vsub_s64(
int64x1_t test_vsub_s64(int64x1_t v1, int64x1_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<1 x !s64i>

// LLVM-SAME: <1 x i64> {{.*}} [[V1:%.*]], <1 x i64> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <1 x i64> [[V1]], [[V2]]
// LLVM: ret <1 x i64> [[SUB_I]]
  return vsub_s64(v1, v2);
}

// LLVM-LABEL: @test_vsubq_s64(
// CIR-LABEL: @vsubq_s64(
int64x2_t test_vsubq_s64(int64x2_t v1, int64x2_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64> {{.*}} [[V1:%.*]], <2 x i64> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[V1]], [[V2]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubq_s64(v1, v2);
}

// LLVM-LABEL: @test_vsub_u8(
// CIR-LABEL: @vsub_u8(
uint8x8_t test_vsub_u8(uint8x8_t v1, uint8x8_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<8 x !u8i>

// LLVM-SAME: <8 x i8> {{.*}} [[V1:%.*]], <8 x i8> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <8 x i8> [[V1]], [[V2]]
// LLVM: ret <8 x i8> [[SUB_I]]
  return vsub_u8(v1, v2);
}

// LLVM-LABEL: @test_vsub_u16(
// CIR-LABEL: @vsub_u16(
uint16x4_t test_vsub_u16(uint16x4_t v1, uint16x4_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<4 x !u16i>

// LLVM-SAME: <4 x i16> {{.*}} [[V1:%.*]], <4 x i16> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <4 x i16> [[V1]], [[V2]]
// LLVM: ret <4 x i16> [[SUB_I]]
  return vsub_u16(v1, v2);
}

// LLVM-LABEL: @test_vsub_u32(
// CIR-LABEL: @vsub_u32(
uint32x2_t test_vsub_u32(uint32x2_t v1, uint32x2_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<2 x !u32i>

// LLVM-SAME: <2 x i32> {{.*}} [[V1:%.*]], <2 x i32> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <2 x i32> [[V1]], [[V2]]
// LLVM: ret <2 x i32> [[SUB_I]]
  return vsub_u32(v1, v2);
}

// LLVM-LABEL: @test_vsub_u64(
// CIR-LABEL: @vsub_u64(
uint64x1_t test_vsub_u64(uint64x1_t v1, uint64x1_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<1 x !u64i>

// LLVM-SAME: <1 x i64> {{.*}} [[V1:%.*]], <1 x i64> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <1 x i64> [[V1]], [[V2]]
// LLVM: ret <1 x i64> [[SUB_I]]
  return vsub_u64(v1, v2);
}

// LLVM-LABEL: @test_vsubq_u8(
// CIR-LABEL: @vsubq_u8(
uint8x16_t test_vsubq_u8(uint8x16_t v1, uint8x16_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<16 x !u8i>

// LLVM-SAME: <16 x i8> {{.*}} [[V1:%.*]], <16 x i8> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <16 x i8> [[V1]], [[V2]]
// LLVM: ret <16 x i8> [[SUB_I]]
  return vsubq_u8(v1, v2);
}

// LLVM-LABEL: @test_vsubq_u16(
// CIR-LABEL: @vsubq_u16(
uint16x8_t test_vsubq_u16(uint16x8_t v1, uint16x8_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16> {{.*}} [[V1:%.*]], <8 x i16> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[V1]], [[V2]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubq_u16(v1, v2);
}

// LLVM-LABEL: @test_vsubq_u32(
// CIR-LABEL: @vsubq_u32(
uint32x4_t test_vsubq_u32(uint32x4_t v1, uint32x4_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32> {{.*}} [[V1:%.*]], <4 x i32> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[V1]], [[V2]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubq_u32(v1, v2);
}

// LLVM-LABEL: @test_vsubq_u64(
// CIR-LABEL: @vsubq_u64(
uint64x2_t test_vsubq_u64(uint64x2_t v1, uint64x2_t v2) {
// CIR: [[SUB_I:%.*]] = cir.sub [[V1:%.*]], [[V2:%.*]] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64> {{.*}} [[V1:%.*]], <2 x i64> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[V1]], [[V2]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubq_u64(v1, v2);
}

// LLVM-LABEL: @test_vsub_f32(
// CIR-LABEL: @vsub_f32(
float32x2_t test_vsub_f32(float32x2_t v1, float32x2_t v2) {
// CIR: [[SUB_I:%.*]] = cir.fsub [[V1:%.*]], [[V2:%.*]] : !cir.vector<2 x !cir.float>

// LLVM-SAME: <2 x float> {{.*}} [[V1:%.*]], <2 x float> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = fsub <2 x float> [[V1]], [[V2]]
// LLVM: ret <2 x float> [[SUB_I]]
  return vsub_f32(v1, v2);
}

// LLVM-LABEL: @test_vsubq_f32(
// CIR-LABEL: @vsubq_f32(
float32x4_t test_vsubq_f32(float32x4_t v1, float32x4_t v2) {
// CIR: [[SUB_I:%.*]] = cir.fsub [[V1:%.*]], [[V2:%.*]] : !cir.vector<4 x !cir.float>

// LLVM-SAME: <4 x float> {{.*}} [[V1:%.*]], <4 x float> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = fsub <4 x float> [[V1]], [[V2]]
// LLVM: ret <4 x float> [[SUB_I]]
  return vsubq_f32(v1, v2);
}

// LLVM-LABEL: @test_vsub_f64(
// CIR-LABEL: @vsub_f64(
float64x1_t test_vsub_f64(float64x1_t a, float64x1_t b) {
// CIR: [[SUB_I:%.*]] = cir.fsub [[A:%.*]], [[B:%.*]] : !cir.vector<1 x !cir.double>

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]])
// LLVM: [[SUB_I:%.*]] = fsub <1 x double> [[A]], [[B]]
// LLVM: ret <1 x double> [[SUB_I]]
  return vsub_f64(a, b);
}

// LLVM-LABEL: @test_vsubq_f64(
// CIR-LABEL: @vsubq_f64(
float64x2_t test_vsubq_f64(float64x2_t v1, float64x2_t v2) {
// CIR: [[SUB_I:%.*]] = cir.fsub [[V1:%.*]], [[V2:%.*]] : !cir.vector<2 x !cir.double>

// LLVM-SAME: <2 x double> {{.*}} [[V1:%.*]], <2 x double> {{.*}} [[V2:%.*]])
// LLVM: [[SUB_I:%.*]] = fsub <2 x double> [[V1]], [[V2]]
// LLVM: ret <2 x double> [[SUB_I]]
  return vsubq_f64(v1, v2);
}

// LLVM-LABEL: @test_vsubd_s64(
// CIR-LABEL: @vsubd_s64(
int64_t test_vsubd_s64(int64_t a, int64_t b) {
// CIR: [[VSUBD_I:%.*]] = cir.sub [[A:%.*]], [[B:%.*]] : !s64i

// LLVM-SAME: i64 {{.*}} [[A:%.*]], i64 {{.*}} [[B:%.*]])
// LLVM: [[VSUBD_I:%.*]] = sub i64 [[A]], [[B]]
// LLVM: ret i64 [[VSUBD_I]]
  return vsubd_s64(a, b);
}

// LLVM-LABEL: @test_vsubd_u64(
// CIR-LABEL: @vsubd_u64(
uint64_t test_vsubd_u64(uint64_t a, uint64_t b) {
// CIR: [[VSUBD_I:%.*]] = cir.sub [[A:%.*]], [[B:%.*]] : !u64i

// LLVM-SAME: i64 {{.*}} [[A:%.*]], i64 {{.*}} [[B:%.*]])
// LLVM: [[VSUBD_I:%.*]] = sub i64 [[A]], [[B]]
// LLVM: ret i64 [[VSUBD_I]]
  return vsubd_u64(a, b);
}

//===------------------------------------------------------===//
// 2.1.1.5.3.  Widening subtraction
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#widening-subtraction
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vsubl_s8(
// CIR-LABEL: @vsubl_s8(
int16x8_t test_vsubl_s8(int8x8_t a, int8x8_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_s8({{.*}}) : {{.*}} -> !cir.vector<8 x !s16i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_s8({{.*}}) : {{.*}} -> !cir.vector<8 x !s16i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[VMOVL0:%.*]] = sext <8 x i8> [[A]] to <8 x i16>
// LLVM: [[VMOVL1:%.*]] = sext <8 x i8> [[B]] to <8 x i16>
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubl_s8(a, b);
}

// LLVM-LABEL: @test_vsubl_s16(
// CIR-LABEL: @vsubl_s16(
int32x4_t test_vsubl_s16(int16x4_t a, int16x4_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_s16({{.*}}) : {{.*}} -> !cir.vector<4 x !s32i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_s16({{.*}}) : {{.*}} -> !cir.vector<4 x !s32i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL0:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: [[TMP2:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
// LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// LLVM: [[VMOVL1:%.*]] = sext <4 x i16> [[TMP3]] to <4 x i32>
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubl_s16(a, b);
}

// LLVM-LABEL: @test_vsubl_s32(
// CIR-LABEL: @vsubl_s32(
int64x2_t test_vsubl_s32(int32x2_t a, int32x2_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_s32({{.*}}) : {{.*}} -> !cir.vector<2 x !s64i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_s32({{.*}}) : {{.*}} -> !cir.vector<2 x !s64i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL0:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
// LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// LLVM: [[VMOVL1:%.*]] = sext <2 x i32> [[TMP3]] to <2 x i64>
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubl_s32(a, b);
}

// LLVM-LABEL: @test_vsubl_u8(
// CIR-LABEL: @vsubl_u8(
uint16x8_t test_vsubl_u8(uint8x8_t a, uint8x8_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_u8({{.*}}) : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_u8({{.*}}) : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[VMOVL0:%.*]] = zext <8 x i8> [[A]] to <8 x i16>
// LLVM: [[VMOVL1:%.*]] = zext <8 x i8> [[B]] to <8 x i16>
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubl_u8(a, b);
}

// LLVM-LABEL: @test_vsubl_u16(
// CIR-LABEL: @vsubl_u16(
uint32x4_t test_vsubl_u16(uint16x4_t a, uint16x4_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_u16({{.*}}) : {{.*}} -> !cir.vector<4 x !u32i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_u16({{.*}}) : {{.*}} -> !cir.vector<4 x !u32i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL0:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: [[TMP2:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
// LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// LLVM: [[VMOVL1:%.*]] = zext <4 x i16> [[TMP3]] to <4 x i32>
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubl_u16(a, b);
}

// LLVM-LABEL: @test_vsubl_u32(
// CIR-LABEL: @vsubl_u32(
uint64x2_t test_vsubl_u32(uint32x2_t a, uint32x2_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_u32({{.*}}) : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_u32({{.*}}) : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL0:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
// LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// LLVM: [[VMOVL1:%.*]] = zext <2 x i32> [[TMP3]] to <2 x i64>
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubl_u32(a, b);
}

// LLVM-LABEL: @test_vsubw_s8(
// CIR-LABEL: @vsubw_s8(
int16x8_t test_vsubw_s8(int16x8_t a, int8x8_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_s8({{.*}}) : {{.*}} -> !cir.vector<8 x !s16i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[VMOVL_I:%.*]] = sext <8 x i8> [[B]] to <8 x i16>
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[A]], [[VMOVL_I]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubw_s8(a, b);
}

// LLVM-LABEL: @test_vsubw_s16(
// CIR-LABEL: @vsubw_s16(
int32x4_t test_vsubw_s16(int32x4_t a, int16x4_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_s16({{.*}}) : {{.*}} -> !cir.vector<4 x !s32i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL_I:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[A]], [[VMOVL_I]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubw_s16(a, b);
}

// LLVM-LABEL: @test_vsubw_s32(
// CIR-LABEL: @vsubw_s32(
int64x2_t test_vsubw_s32(int64x2_t a, int32x2_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_s32({{.*}}) : {{.*}} -> !cir.vector<2 x !s64i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL_I:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[A]], [[VMOVL_I]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubw_s32(a, b);
}

// LLVM-LABEL: @test_vsubw_u8(
// CIR-LABEL: @vsubw_u8(
uint16x8_t test_vsubw_u8(uint16x8_t a, uint8x8_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_u8({{.*}}) : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[VMOVL_I:%.*]] = zext <8 x i8> [[B]] to <8 x i16>
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[A]], [[VMOVL_I]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubw_u8(a, b);
}

// LLVM-LABEL: @test_vsubw_u16(
// CIR-LABEL: @vsubw_u16(
uint32x4_t test_vsubw_u16(uint32x4_t a, uint16x4_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_u16({{.*}}) : {{.*}} -> !cir.vector<4 x !u32i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL_I:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[A]], [[VMOVL_I]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubw_u16(a, b);
}

// LLVM-LABEL: @test_vsubw_u32(
// CIR-LABEL: @vsubw_u32(
uint64x2_t test_vsubw_u32(uint64x2_t a, uint32x2_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_u32({{.*}}) : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL_I:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[A]], [[VMOVL_I]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubw_u32(a, b);
}

// LLVM-LABEL: @test_vsubl_high_s8(
// CIR-LABEL: @vsubl_high_s8(
int16x8_t test_vsubl_high_s8(int8x16_t a, int8x16_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_high_s8({{.*}}) : {{.*}} -> !cir.vector<8 x !s16i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_high_s8({{.*}}) : {{.*}} -> !cir.vector<8 x !s16i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<8 x !s16i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE0:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM: [[VMOVL0:%.*]] = sext <8 x i8> [[SHUFFLE0]] to <8 x i16>
// LLVM: [[SHUFFLE1:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM: [[VMOVL1:%.*]] = sext <8 x i8> [[SHUFFLE1]] to <8 x i16>
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubl_high_s8(a, b);
}

// LLVM-LABEL: @test_vsubl_high_s16(
// CIR-LABEL: @vsubl_high_s16(
int32x4_t test_vsubl_high_s16(int16x8_t a, int16x8_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_high_s16({{.*}}) : {{.*}} -> !cir.vector<4 x !s32i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_high_s16({{.*}}) : {{.*}} -> !cir.vector<4 x !s32i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<4 x !s32i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE0:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE0]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL0:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: [[SHUFFLE1:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM: [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE1]] to <8 x i8>
// LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// LLVM: [[VMOVL1:%.*]] = sext <4 x i16> [[TMP3]] to <4 x i32>
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubl_high_s16(a, b);
}

// LLVM-LABEL: @test_vsubl_high_s32(
// CIR-LABEL: @vsubl_high_s32(
int64x2_t test_vsubl_high_s32(int32x4_t a, int32x4_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_high_s32({{.*}}) : {{.*}} -> !cir.vector<2 x !s64i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_high_s32({{.*}}) : {{.*}} -> !cir.vector<2 x !s64i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<2 x !s64i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE0:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[A]], <2 x i32> <i32 2, i32 3>
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE0]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL0:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: [[SHUFFLE1:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
// LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE1]] to <8 x i8>
// LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// LLVM: [[VMOVL1:%.*]] = sext <2 x i32> [[TMP3]] to <2 x i64>
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubl_high_s32(a, b);
}

// LLVM-LABEL: @test_vsubl_high_u8(
// CIR-LABEL: @vsubl_high_u8(
uint16x8_t test_vsubl_high_u8(uint8x16_t a, uint8x16_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_high_u8({{.*}}) : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_high_u8({{.*}}) : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<8 x !u16i>

// LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE0:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM: [[VMOVL0:%.*]] = zext <8 x i8> [[SHUFFLE0]] to <8 x i16>
// LLVM: [[SHUFFLE1:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM: [[VMOVL1:%.*]] = zext <8 x i8> [[SHUFFLE1]] to <8 x i16>
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubl_high_u8(a, b);
}

// LLVM-LABEL: @test_vsubl_high_u16(
// CIR-LABEL: @vsubl_high_u16(
uint32x4_t test_vsubl_high_u16(uint16x8_t a, uint16x8_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_high_u16({{.*}}) : {{.*}} -> !cir.vector<4 x !u32i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_high_u16({{.*}}) : {{.*}} -> !cir.vector<4 x !u32i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<4 x !u32i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE0:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE0]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL0:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: [[SHUFFLE1:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM: [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE1]] to <8 x i8>
// LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// LLVM: [[VMOVL1:%.*]] = zext <4 x i16> [[TMP3]] to <4 x i32>
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubl_high_u16(a, b);
}

// LLVM-LABEL: @test_vsubl_high_u32(
// CIR-LABEL: @vsubl_high_u32(
uint64x2_t test_vsubl_high_u32(uint32x4_t a, uint32x4_t b) {
// CIR: [[VMOVL0:%.*]] = cir.call @vmovl_high_u32({{.*}}) : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: [[VMOVL1:%.*]] = cir.call @vmovl_high_u32({{.*}}) : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: {{%.*}} = cir.sub [[VMOVL0]], [[VMOVL1]] : !cir.vector<2 x !u64i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE0:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[A]], <2 x i32> <i32 2, i32 3>
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE0]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL0:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: [[SHUFFLE1:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
// LLVM: [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE1]] to <8 x i8>
// LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// LLVM: [[VMOVL1:%.*]] = zext <2 x i32> [[TMP3]] to <2 x i64>
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[VMOVL0]], [[VMOVL1]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubl_high_u32(a, b);
}

// LLVM-LABEL: @test_vsubw_high_s8(
// CIR-LABEL: @vsubw_high_s8(
int16x8_t test_vsubw_high_s8(int16x8_t a, int8x16_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_high_s8({{.*}}) : {{.*}} -> !cir.vector<8 x !s16i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<8 x !s16i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM: [[VMOVL_I:%.*]] = sext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[A]], [[VMOVL_I]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubw_high_s8(a, b);
}

// LLVM-LABEL: @test_vsubw_high_s16(
// CIR-LABEL: @vsubw_high_s16(
int32x4_t test_vsubw_high_s16(int32x4_t a, int16x8_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_high_s16({{.*}}) : {{.*}} -> !cir.vector<4 x !s32i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<4 x !s32i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL_I:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[A]], [[VMOVL_I]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubw_high_s16(a, b);
}

// LLVM-LABEL: @test_vsubw_high_s32(
// CIR-LABEL: @vsubw_high_s32(
int64x2_t test_vsubw_high_s32(int64x2_t a, int32x4_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_high_s32({{.*}}) : {{.*}} -> !cir.vector<2 x !s64i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<2 x !s64i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL_I:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[A]], [[VMOVL_I]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubw_high_s32(a, b);
}

// LLVM-LABEL: @test_vsubw_high_u8(
// CIR-LABEL: @vsubw_high_u8(
uint16x8_t test_vsubw_high_u8(uint16x8_t a, uint8x16_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_high_u8({{.*}}) : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<8 x !u16i>

// LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// LLVM: [[VMOVL_I:%.*]] = zext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// LLVM: [[SUB_I:%.*]] = sub <8 x i16> [[A]], [[VMOVL_I]]
// LLVM: ret <8 x i16> [[SUB_I]]
  return vsubw_high_u8(a, b);
}

// LLVM-LABEL: @test_vsubw_high_u16(
// CIR-LABEL: @vsubw_high_u16(
uint32x4_t test_vsubw_high_u16(uint32x4_t a, uint16x8_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_high_u16({{.*}}) : {{.*}} -> !cir.vector<4 x !u32i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<4 x !u32i>

// LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// LLVM: [[VMOVL_I:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// LLVM: [[SUB_I:%.*]] = sub <4 x i32> [[A]], [[VMOVL_I]]
// LLVM: ret <4 x i32> [[SUB_I]]
  return vsubw_high_u16(a, b);
}

// LLVM-LABEL: @test_vsubw_high_u32(
// CIR-LABEL: @vsubw_high_u32(
uint64x2_t test_vsubw_high_u32(uint64x2_t a, uint32x4_t b) {
// CIR: [[VMOVL_I:%.*]] = cir.call @vmovl_high_u32({{.*}}) : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: {{%.*}} = cir.sub {{%.*}}, [[VMOVL_I]] : !cir.vector<2 x !u64i>

// LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
// LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// LLVM: [[VMOVL_I:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// LLVM: [[SUB_I:%.*]] = sub <2 x i64> [[A]], [[VMOVL_I]]
// LLVM: ret <2 x i64> [[SUB_I]]
  return vsubw_high_u32(a, b);
}

//===----------------------------------------------------------------------===//
// 2.1.1.5.3. Narrowing subtraction
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#narrowing-subtraction
//===----------------------------------------------------------------------===//

// LLVM-IC-LABEL: @test_vhsub_s8(
// CIR-LABEL: @vhsub_s8(
int8x8_t test_vhsub_s8(int8x8_t v1, int8x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shsub"

  // LLVM-IC-SAME: <8 x i8> {{.*}}[[V1:%.*]], <8 x i8> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.shsub.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM-IC: ret <8 x i8> [[RES]]
  return vhsub_s8(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsub_s16(
// CIR-LABEL: @vhsub_s16(
int16x4_t test_vhsub_s16(int16x4_t v1, int16x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shsub"

  // LLVM-IC-SAME: <4 x i16> {{.*}}[[V1:%.*]], <4 x i16> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.shsub.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM-IC: ret <4 x i16> [[RES]]
  return vhsub_s16(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsub_s32(
// CIR-LABEL: @vhsub_s32(
int32x2_t test_vhsub_s32(int32x2_t v1, int32x2_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shsub"

  // LLVM-IC-SAME: <2 x i32> {{.*}}[[V1:%.*]], <2 x i32> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.shsub.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM-IC: ret <2 x i32> [[RES]]
  return vhsub_s32(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsub_u8(
// CIR-LABEL: @vhsub_u8(
uint8x8_t test_vhsub_u8(uint8x8_t v1, uint8x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhsub"

  // LLVM-IC-SAME: <8 x i8> {{.*}}[[V1:%.*]], <8 x i8> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.uhsub.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM-IC: ret <8 x i8> [[RES]]
  return vhsub_u8(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsub_u16(
// CIR-LABEL: @vhsub_u16(
uint16x4_t test_vhsub_u16(uint16x4_t v1, uint16x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhsub"

  // LLVM-IC-SAME: <4 x i16> {{.*}}[[V1:%.*]], <4 x i16> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.uhsub.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM-IC: ret <4 x i16> [[RES]]
  return vhsub_u16(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsub_u32(
// CIR-LABEL: @vhsub_u32(
uint32x2_t test_vhsub_u32(uint32x2_t v1, uint32x2_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhsub"

  // LLVM-IC-SAME: <2 x i32> {{.*}}[[V1:%.*]], <2 x i32> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.uhsub.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM-IC: ret <2 x i32> [[RES]]
  return vhsub_u32(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsubq_s8(
// CIR-LABEL: @vhsubq_s8(
int8x16_t test_vhsubq_s8(int8x16_t v1, int8x16_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shsub"

  // LLVM-IC-SAME: <16 x i8> {{.*}}[[V1:%.*]], <16 x i8> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.shsub.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM-IC: ret <16 x i8> [[RES]]
  return vhsubq_s8(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsubq_s16(
// CIR-LABEL: @vhsubq_s16(
int16x8_t test_vhsubq_s16(int16x8_t v1, int16x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shsub"

  // LLVM-IC-SAME: <8 x i16> {{.*}}[[V1:%.*]], <8 x i16> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.shsub.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM-IC: ret <8 x i16> [[RES]]
  return vhsubq_s16(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsubq_s32(
// CIR-LABEL: @vhsubq_s32(
int32x4_t test_vhsubq_s32(int32x4_t v1, int32x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shsub"

  // LLVM-IC-SAME: <4 x i32> {{.*}}[[V1:%.*]], <4 x i32> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.shsub.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM-IC: ret <4 x i32> [[RES]]
  return vhsubq_s32(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsubq_u8(
// CIR-LABEL: @vhsubq_u8(
uint8x16_t test_vhsubq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhsub"

  // LLVM-IC-SAME: <16 x i8> {{.*}}[[V1:%.*]], <16 x i8> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.uhsub.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM-IC: ret <16 x i8> [[RES]]
  return vhsubq_u8(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsubq_u16(
// CIR-LABEL: @vhsubq_u16(
uint16x8_t test_vhsubq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhsub"

  // LLVM-IC-SAME: <8 x i16> {{.*}}[[V1:%.*]], <8 x i16> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.uhsub.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM-IC: ret <8 x i16> [[RES]]
  return vhsubq_u16(v1, v2);
}

// LLVM-IC-LABEL: @test_vhsubq_u32(
// CIR-LABEL: @vhsubq_u32(
uint32x4_t test_vhsubq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhsub"

  // LLVM-IC-SAME: <4 x i32> {{.*}}[[V1:%.*]], <4 x i32> {{.*}}[[V2:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.uhsub.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM-IC: ret <4 x i32> [[RES]]
  return vhsubq_u32(v1, v2);
}

// LLVM-IC-LABEL: @test_vsubhn_s16(
// CIR-LABEL: @vsubhn_s16(
int8x8_t test_vsubhn_s16(int16x8_t a, int16x8_t b) {
  // CIR: cir.sub
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-IC-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <8 x i16> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <8 x i16> [[SUB]], splat (i16 8)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <8 x i16> [[SH]] to <8 x i8>
  // LLVM-IC: ret <8 x i8> [[TR]]
  return vsubhn_s16(a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_s32(
// CIR-LABEL: @vsubhn_s32(
int16x4_t test_vsubhn_s32(int32x4_t a, int32x4_t b) {
  // CIR: cir.sub
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-IC-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <4 x i32> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <4 x i32> [[SUB]], splat (i32 16)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <4 x i32> [[SH]] to <4 x i16>
  // LLVM-IC: ret <4 x i16> [[TR]]
  return vsubhn_s32(a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_s64(
// CIR-LABEL: @vsubhn_s64(
int32x2_t test_vsubhn_s64(int64x2_t a, int64x2_t b) {
  // CIR: cir.sub
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-IC-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <2 x i64> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <2 x i64> [[SUB]], splat (i64 32)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <2 x i64> [[SH]] to <2 x i32>
  // LLVM-IC: ret <2 x i32> [[TR]]
  return vsubhn_s64(a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_u16(
// CIR-LABEL: @vsubhn_u16(
uint8x8_t test_vsubhn_u16(uint16x8_t a, uint16x8_t b) {
  // CIR: cir.sub
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-IC-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <8 x i16> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <8 x i16> [[SUB]], splat (i16 8)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <8 x i16> [[SH]] to <8 x i8>
  // LLVM-IC: ret <8 x i8> [[TR]]
  return vsubhn_u16(a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_u32(
// CIR-LABEL: @vsubhn_u32(
uint16x4_t test_vsubhn_u32(uint32x4_t a, uint32x4_t b) {
  // CIR: cir.sub
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-IC-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <4 x i32> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <4 x i32> [[SUB]], splat (i32 16)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <4 x i32> [[SH]] to <4 x i16>
  // LLVM-IC: ret <4 x i16> [[TR]]
  return vsubhn_u32(a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_u64(
// CIR-LABEL: @vsubhn_u64(
uint32x2_t test_vsubhn_u64(uint64x2_t a, uint64x2_t b) {
  // CIR: cir.sub
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-IC-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <2 x i64> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <2 x i64> [[SUB]], splat (i64 32)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <2 x i64> [[SH]] to <2 x i32>
  // LLVM-IC: ret <2 x i32> [[TR]]
  return vsubhn_u64(a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_high_s16(
// CIR-LABEL: @vsubhn_high_s16(
int8x16_t test_vsubhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  // CIR: cir.call @vsubhn_s16(
  // CIR: cir.call @vcombine_s8(

  // LLVM-IC-SAME: <8 x i8> {{.*}}[[R:%.*]], <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <8 x i16> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <8 x i16> [[SUB]], splat (i16 8)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <8 x i16> [[SH]] to <8 x i8>
  // LLVM-IC: [[RES:%.*]] = shufflevector <8 x i8> [[R]], <8 x i8> [[TR]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-IC: ret <16 x i8> [[RES]]
  return vsubhn_high_s16(r, a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_high_s32(
// CIR-LABEL: @vsubhn_high_s32(
int16x8_t test_vsubhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  // CIR: cir.call @vsubhn_s32(
  // CIR: cir.call @vcombine_s16(

  // LLVM-IC-SAME: <4 x i16> {{.*}}[[R:%.*]], <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <4 x i32> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <4 x i32> [[SUB]], splat (i32 16)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <4 x i32> [[SH]] to <4 x i16>
  // LLVM-IC: [[RES:%.*]] = shufflevector <4 x i16> [[R]], <4 x i16> [[TR]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM-IC: ret <8 x i16> [[RES]]
  return vsubhn_high_s32(r, a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_high_s64(
// CIR-LABEL: @vsubhn_high_s64(
int32x4_t test_vsubhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  // CIR: cir.call @vsubhn_s64(
  // CIR: cir.call @vcombine_s32(

  // LLVM-IC-SAME: <2 x i32> {{.*}}[[R:%.*]], <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <2 x i64> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <2 x i64> [[SUB]], splat (i64 32)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <2 x i64> [[SH]] to <2 x i32>
  // LLVM-IC: [[RES:%.*]] = shufflevector <2 x i32> [[R]], <2 x i32> [[TR]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM-IC: ret <4 x i32> [[RES]]
  return vsubhn_high_s64(r, a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_high_u16(
// CIR-LABEL: @vsubhn_high_u16(
uint8x16_t test_vsubhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  // CIR: cir.call @vsubhn_u16(
  // CIR: cir.call @vcombine_u8(

  // LLVM-IC-SAME: <8 x i8> {{.*}}[[R:%.*]], <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <8 x i16> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <8 x i16> [[SUB]], splat (i16 8)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <8 x i16> [[SH]] to <8 x i8>
  // LLVM-IC: [[RES:%.*]] = shufflevector <8 x i8> [[R]], <8 x i8> [[TR]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-IC: ret <16 x i8> [[RES]]
  return vsubhn_high_u16(r, a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_high_u32(
// CIR-LABEL: @vsubhn_high_u32(
uint16x8_t test_vsubhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  // CIR: cir.call @vsubhn_u32(
  // CIR: cir.call @vcombine_u16(

  // LLVM-IC-SAME: <4 x i16> {{.*}}[[R:%.*]], <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <4 x i32> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <4 x i32> [[SUB]], splat (i32 16)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <4 x i32> [[SH]] to <4 x i16>
  // LLVM-IC: [[RES:%.*]] = shufflevector <4 x i16> [[R]], <4 x i16> [[TR]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM-IC: ret <8 x i16> [[RES]]
  return vsubhn_high_u32(r, a, b);
}

// LLVM-IC-LABEL: @test_vsubhn_high_u64(
// CIR-LABEL: @vsubhn_high_u64(
uint32x4_t test_vsubhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  // CIR: cir.call @vsubhn_u64(
  // CIR: cir.call @vcombine_u32(

  // LLVM-IC-SAME: <2 x i32> {{.*}}[[R:%.*]], <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM-IC: [[SUB:%.*]] = sub <2 x i64> [[A]], [[B]]
  // LLVM-IC: [[SH:%.*]] = lshr <2 x i64> [[SUB]], splat (i64 32)
  // LLVM-IC: [[TR:%.*]] = trunc nuw <2 x i64> [[SH]] to <2 x i32>
  // LLVM-IC: [[RES:%.*]] = shufflevector <2 x i32> [[R]], <2 x i32> [[TR]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM-IC: ret <4 x i32> [[RES]]
  return vsubhn_high_u64(r, a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_s16(
// CIR-LABEL: @vrsubhn_s16(
int8x8_t test_vrsubhn_s16(int16x8_t a, int16x8_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.rsubhn"

  // LLVM-IC-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM-IC: ret <8 x i8> [[RES]]
  return vrsubhn_s16(a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_s32(
// CIR-LABEL: @vrsubhn_s32(
int16x4_t test_vrsubhn_s32(int32x4_t a, int32x4_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.rsubhn"

  // LLVM-IC-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM-IC: ret <4 x i16> [[RES]]
  return vrsubhn_s32(a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_s64(
// CIR-LABEL: @vrsubhn_s64(
int32x2_t test_vrsubhn_s64(int64x2_t a, int64x2_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.rsubhn"

  // LLVM-IC-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> [[A]], <2 x i64> [[B]])
  // LLVM-IC: ret <2 x i32> [[RES]]
  return vrsubhn_s64(a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_u16(
// CIR-LABEL: @vrsubhn_u16(
uint8x8_t test_vrsubhn_u16(uint16x8_t a, uint16x8_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.rsubhn"

  // LLVM-IC-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM-IC: ret <8 x i8> [[RES]]
  return vrsubhn_u16(a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_u32(
// CIR-LABEL: @vrsubhn_u32(
uint16x4_t test_vrsubhn_u32(uint32x4_t a, uint32x4_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.rsubhn"

  // LLVM-IC-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM-IC: ret <4 x i16> [[RES]]
  return vrsubhn_u32(a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_u64(
// CIR-LABEL: @vrsubhn_u64(
uint32x2_t test_vrsubhn_u64(uint64x2_t a, uint64x2_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.rsubhn"

  // LLVM-IC-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM-IC: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> [[A]], <2 x i64> [[B]])
  // LLVM-IC: ret <2 x i32> [[RES]]
  return vrsubhn_u64(a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_high_s16(
// CIR-LABEL: @vrsubhn_high_s16(
int8x16_t test_vrsubhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  // CIR: cir.call @vrsubhn_s16(
  // CIR: cir.call @vcombine_s8(

  // LLVM-IC-SAME: <8 x i8> {{.*}}[[R:%.*]], <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM-IC: [[TMP:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM-IC: [[RES:%.*]] = shufflevector <8 x i8> [[R]], <8 x i8> [[TMP]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-IC: ret <16 x i8> [[RES]]
  return vrsubhn_high_s16(r, a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_high_s32(
// CIR-LABEL: @vrsubhn_high_s32(
int16x8_t test_vrsubhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  // CIR: cir.call @vrsubhn_s32(
  // CIR: cir.call @vcombine_s16(

  // LLVM-IC-SAME: <4 x i16> {{.*}}[[R:%.*]], <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM-IC: [[TMP:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM-IC: [[RES:%.*]] = shufflevector <4 x i16> [[R]], <4 x i16> [[TMP]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM-IC: ret <8 x i16> [[RES]]
  return vrsubhn_high_s32(r, a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_high_s64(
// CIR-LABEL: @vrsubhn_high_s64(
int32x4_t test_vrsubhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  // CIR: cir.call @vrsubhn_s64(
  // CIR: cir.call @vcombine_s32(

  // LLVM-IC-SAME: <2 x i32> {{.*}}[[R:%.*]], <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM-IC: [[TMP:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> [[A]], <2 x i64> [[B]])
  // LLVM-IC: [[RES:%.*]] = shufflevector <2 x i32> [[R]], <2 x i32> [[TMP]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM-IC: ret <4 x i32> [[RES]]
  return vrsubhn_high_s64(r, a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_high_u16(
// CIR-LABEL: @vrsubhn_high_u16(
uint8x16_t test_vrsubhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  // CIR: cir.call @vrsubhn_u16(
  // CIR: cir.call @vcombine_u8(

  // LLVM-IC-SAME: <8 x i8> {{.*}}[[R:%.*]], <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM-IC: [[TMP:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM-IC: [[RES:%.*]] = shufflevector <8 x i8> [[R]], <8 x i8> [[TMP]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-IC: ret <16 x i8> [[RES]]
  return vrsubhn_high_u16(r, a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_high_u32(
// CIR-LABEL: @vrsubhn_high_u32(
uint16x8_t test_vrsubhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  // CIR: cir.call @vrsubhn_u32(
  // CIR: cir.call @vcombine_u16(

  // LLVM-IC-SAME: <4 x i16> {{.*}}[[R:%.*]], <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM-IC: [[TMP:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM-IC: [[RES:%.*]] = shufflevector <4 x i16> [[R]], <4 x i16> [[TMP]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM-IC: ret <8 x i16> [[RES]]
  return vrsubhn_high_u32(r, a, b);
}

// LLVM-IC-LABEL: @test_vrsubhn_high_u64(
// CIR-LABEL: @vrsubhn_high_u64(
uint32x4_t test_vrsubhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  // CIR: cir.call @vrsubhn_u64(
  // CIR: cir.call @vcombine_u32(

  // LLVM-IC-SAME: <2 x i32> {{.*}}[[R:%.*]], <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM-IC: [[TMP:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> [[A]], <2 x i64> [[B]])
  // LLVM-IC: [[RES:%.*]] = shufflevector <2 x i32> [[R]], <2 x i32> [[TMP]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM-IC: ret <4 x i32> [[RES]]
  return vrsubhn_high_u64(r, a, b);
}
