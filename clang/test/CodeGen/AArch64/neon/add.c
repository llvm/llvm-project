// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefix=CIR %}

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.1.1.1.1 Addition
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#addition
//===------------------------------------------------------===//

// LLVM-LABEL: @test_vadd_s8(
// CIR-LABEL: @vadd_s8(
int8x8_t test_vadd_s8(int8x8_t a, int8x8_t b) {
// CIR: cir.add

// LLVM-SAME: <8 x i8> {{.*}}[[A:%.*]], <8 x i8> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <8 x i8> [[A]], [[B]]
// LLVM: ret <8 x i8> [[ADD_I]]
  return vadd_s8(a, b);
}

// LLVM-LABEL: @test_vadd_s16(
// CIR-LABEL: @vadd_s16(
int16x4_t test_vadd_s16(int16x4_t a, int16x4_t b) {
// CIR: cir.add

// LLVM-SAME: <4 x i16> {{.*}}[[A:%.*]], <4 x i16> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <4 x i16> [[A]], [[B]]
// LLVM: ret <4 x i16> [[ADD_I]]
  return vadd_s16(a, b);
}

// LLVM-LABEL: @test_vadd_s32(
// CIR-LABEL: @vadd_s32(
int32x2_t test_vadd_s32(int32x2_t a, int32x2_t b) {
// CIR: cir.add

// LLVM-SAME: <2 x i32> {{.*}}[[A:%.*]], <2 x i32> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <2 x i32> [[A]], [[B]]
// LLVM: ret <2 x i32> [[ADD_I]]
  return vadd_s32(a, b);
}

// LLVM-LABEL: @test_vadd_s64(
// CIR-LABEL: @vadd_s64(
int64x1_t test_vadd_s64(int64x1_t a, int64x1_t b) {
// CIR: cir.add

// LLVM-SAME: <1 x i64> {{.*}}[[A:%.*]], <1 x i64> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <1 x i64> [[A]], [[B]]
// LLVM: ret <1 x i64> [[ADD_I]]
  return vadd_s64(a, b);
}

// LLVM-LABEL: @test_vadd_f32(
// CIR-LABEL: @vadd_f32(
float32x2_t test_vadd_f32(float32x2_t a, float32x2_t b) {
// CIR: cir.fadd

// LLVM-SAME: <2 x float> {{.*}}[[A:%.*]], <2 x float> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = fadd <2 x float> [[A]], [[B]]
// LLVM: ret <2 x float> [[ADD_I]]
  return vadd_f32(a, b);
}

// LLVM-LABEL: @test_vadd_u8(
// CIR-LABEL: @vadd_u8(
uint8x8_t test_vadd_u8(uint8x8_t a, uint8x8_t b) {
// CIR: cir.add

// LLVM-SAME: <8 x i8> {{.*}}[[A:%.*]], <8 x i8> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <8 x i8> [[A]], [[B]]
// LLVM: ret <8 x i8> [[ADD_I]]
  return vadd_u8(a, b);
}

// LLVM-LABEL: @test_vadd_u16(
// CIR-LABEL: @vadd_u16(
uint16x4_t test_vadd_u16(uint16x4_t a, uint16x4_t b) {
// CIR: cir.add

// LLVM-SAME: <4 x i16> {{.*}}[[A:%.*]], <4 x i16> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <4 x i16> [[A]], [[B]]
// LLVM: ret <4 x i16> [[ADD_I]]
  return vadd_u16(a, b);
}

// LLVM-LABEL: @test_vadd_u32(
// CIR-LABEL: @vadd_u32(
uint32x2_t test_vadd_u32(uint32x2_t a, uint32x2_t b) {
// CIR: cir.add

// LLVM-SAME: <2 x i32> {{.*}}[[A:%.*]], <2 x i32> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <2 x i32> [[A]], [[B]]
// LLVM: ret <2 x i32> [[ADD_I]]
  return vadd_u32(a, b);
}

// LLVM-LABEL: @test_vadd_u64(
// CIR-LABEL: @vadd_u64(
uint64x1_t test_vadd_u64(uint64x1_t a, uint64x1_t b) {
// CIR: cir.add

// LLVM-SAME: <1 x i64> {{.*}}[[A:%.*]], <1 x i64> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <1 x i64> [[A]], [[B]]
// LLVM: ret <1 x i64> [[ADD_I]]
  return vadd_u64(a, b);
}

// LLVM-LABEL: @test_vaddq_s8(
// CIR-LABEL: @vaddq_s8(
int8x16_t test_vaddq_s8(int8x16_t a, int8x16_t b) {
// CIR: cir.add

// LLVM-SAME: <16 x i8> {{.*}}[[A:%.*]], <16 x i8> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <16 x i8> [[A]], [[B]]
// LLVM: ret <16 x i8> [[ADD_I]]
  return vaddq_s8(a, b);
}

// LLVM-LABEL: @test_vaddq_s16(
// CIR-LABEL: @vaddq_s16(
int16x8_t test_vaddq_s16(int16x8_t a, int16x8_t b) {
// CIR: cir.add

// LLVM-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <8 x i16> [[A]], [[B]]
// LLVM: ret <8 x i16> [[ADD_I]]
  return vaddq_s16(a, b);
}

// LLVM-LABEL: @test_vaddq_s32(
// CIR-LABEL: @vaddq_s32(
int32x4_t test_vaddq_s32(int32x4_t a, int32x4_t b) {
// CIR: cir.add

// LLVM-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <4 x i32> [[A]], [[B]]
// LLVM: ret <4 x i32> [[ADD_I]]
  return vaddq_s32(a, b);
}

// LLVM-LABEL: @test_vaddq_s64(
// CIR-LABEL: @vaddq_s64(
int64x2_t test_vaddq_s64(int64x2_t a, int64x2_t b) {
// CIR: cir.add

// LLVM-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <2 x i64> [[A]], [[B]]
// LLVM: ret <2 x i64> [[ADD_I]]
  return vaddq_s64(a, b);
}

// LLVM-LABEL: @test_vaddq_f32(
// CIR-LABEL: @vaddq_f32(
float32x4_t test_vaddq_f32(float32x4_t a, float32x4_t b) {
// CIR: cir.fadd

// LLVM-SAME: <4 x float> {{.*}}[[A:%.*]], <4 x float> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = fadd <4 x float> [[A]], [[B]]
// LLVM: ret <4 x float> [[ADD_I]]
  return vaddq_f32(a, b);
}

// LLVM-LABEL: @test_vaddq_f64(
// CIR-LABEL: @vaddq_f64(
float64x2_t test_vaddq_f64(float64x2_t a, float64x2_t b) {
// CIR: cir.fadd

// LLVM-SAME: <2 x double> {{.*}}[[A:%.*]], <2 x double> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = fadd <2 x double> [[A]], [[B]]
// LLVM: ret <2 x double> [[ADD_I]]
  return vaddq_f64(a, b);
}

// LLVM-LABEL: @test_vaddq_u8(
// CIR-LABEL: @vaddq_u8(
uint8x16_t test_vaddq_u8(uint8x16_t a, uint8x16_t b) {
// CIR: cir.add

// LLVM-SAME: <16 x i8> {{.*}}[[A:%.*]], <16 x i8> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <16 x i8> [[A]], [[B]]
// LLVM: ret <16 x i8> [[ADD_I]]
  return vaddq_u8(a, b);
}

// LLVM-LABEL: @test_vaddq_u16(
// CIR-LABEL: @vaddq_u16(
uint16x8_t test_vaddq_u16(uint16x8_t a, uint16x8_t b) {
// CIR: cir.add

// LLVM-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <8 x i16> [[A]], [[B]]
// LLVM: ret <8 x i16> [[ADD_I]]
  return vaddq_u16(a, b);
}

// LLVM-LABEL: @test_vaddq_u32(
// CIR-LABEL: @vaddq_u32(
uint32x4_t test_vaddq_u32(uint32x4_t a, uint32x4_t b) {
// CIR: cir.add

// LLVM-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <4 x i32> [[A]], [[B]]
// LLVM: ret <4 x i32> [[ADD_I]]
  return vaddq_u32(a, b);
}

// LLVM-LABEL: @test_vaddq_u64(
// CIR-LABEL: @vaddq_u64(
uint64x2_t test_vaddq_u64(uint64x2_t a, uint64x2_t b) {
// CIR: cir.add

// LLVM-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = add <2 x i64> [[A]], [[B]]
// LLVM: ret <2 x i64> [[ADD_I]]
  return vaddq_u64(a, b);
}

// LLVM-LABEL: @test_vadd_f64(
// CIR-LABEL: @vadd_f64(
float64x1_t test_vadd_f64(float64x1_t a, float64x1_t b) {
// CIR: cir.fadd

// LLVM-SAME: <1 x double> {{.*}}[[A:%.*]], <1 x double> {{.*}}[[B:%.*]])
// LLVM: [[ADD_I:%.*]] = fadd <1 x double> [[A]], [[B]]
// LLVM: ret <1 x double> [[ADD_I]]
  return vadd_f64(a, b);
}

// LLVM-LABEL: @test_vaddd_s64(
// CIR-LABEL: @vaddd_s64(
int64_t test_vaddd_s64(int64_t a, int64_t b) {
// CIR: cir.add

// LLVM-SAME: i64 {{.*}}[[A:%.*]], i64 {{.*}}[[B:%.*]])
// LLVM: [[VADDD_I:%.*]] = add i64 [[A]], [[B]]
// LLVM: ret i64 [[VADDD_I]]
  return vaddd_s64(a, b);
}

// LLVM-LABEL: @test_vaddd_u64(
// CIR-LABEL: @vaddd_u64(
uint64_t test_vaddd_u64(uint64_t a, uint64_t b) {
// CIR: cir.add

// LLVM-SAME: i64 {{.*}}[[A:%.*]], i64 {{.*}}[[B:%.*]])
// LLVM: [[VADDD_I:%.*]] = add i64 [[A]], [[B]]
// LLVM: ret i64 [[VADDD_I]]
  return vaddd_u64(a, b);
}

//===----------------------------------------------------------------------===//
// 2.2.2.1.2. Polynomial addition
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#polynomial-addition
//===----------------------------------------------------------------------===//

// LLVM-LABEL: @test_vadd_p8(
// CIR-LABEL: @vadd_p8(
poly8x8_t test_vadd_p8(poly8x8_t a, poly8x8_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<8 x !u8i>

  // LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = xor <8 x i8> [[A]], [[B]]
  // LLVM-NEXT:    ret <8 x i8> [[TMP0]]
  return vadd_p8(a, b);
}

// LLVM-LABEL: @test_vadd_p16(
// CIR-LABEL: @vadd_p16(
poly16x4_t test_vadd_p16(poly16x4_t a, poly16x4_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<8 x !u8i>

  // LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP2:%.*]] = xor <8 x i8> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
  // LLVM-NEXT:    ret <4 x i16> [[TMP3]]
  return vadd_p16(a, b);
}

// LLVM-LABEL: @test_vadd_p64(
// CIR-LABEL: @vadd_p64(
poly64x1_t test_vadd_p64(poly64x1_t a, poly64x1_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<8 x !u8i>

  // LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <1 x i64> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP2:%.*]] = xor <8 x i8> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x i64>
  // LLVM-NEXT:    ret <1 x i64> [[TMP3]]
  return vadd_p64(a, b);
}

// LLVM-LABEL: @test_vaddq_p8(
// CIR-LABEL: @vaddq_p8(
poly8x16_t test_vaddq_p8(poly8x16_t a, poly8x16_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<16 x !u8i>

  // LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = xor <16 x i8> [[A]], [[B]]
  // LLVM-NEXT:    ret <16 x i8> [[TMP0]]
  return vaddq_p8(a, b);
}

// LLVM-LABEL: @test_vaddq_p16(
// CIR-LABEL: @vaddq_p16(
poly16x8_t test_vaddq_p16(poly16x8_t a, poly16x8_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<16 x !u8i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
  // LLVM-NEXT:    [[TMP2:%.*]] = xor <16 x i8> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
  // LLVM-NEXT:    ret <8 x i16> [[TMP3]]
  return vaddq_p16(a, b);
}

// LLVM-LABEL: @test_vaddq_p64(
// CIR-LABEL: @vaddq_p64(
poly64x2_t test_vaddq_p64(poly64x2_t a, poly64x2_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<16 x !u8i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <2 x i64> [[B]] to <16 x i8>
  // LLVM-NEXT:    [[TMP2:%.*]] = xor <16 x i8> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x i64>
  // LLVM-NEXT:    ret <2 x i64> [[TMP3]]
  return vaddq_p64(a, b);
}

// LLVM-LABEL: @test_vaddq_p128(
// CIR-LABEL: @vaddq_p128(
poly128_t test_vaddq_p128(poly128_t a, poly128_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<16 x !u8i>

  // LLVM-SAME: i128 {{.*}} [[A:%.*]], i128 {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast i128 [[A]] to <16 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast i128 [[B]] to <16 x i8>
  // LLVM-NEXT:    [[TMP2:%.*]] = xor <16 x i8> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <16 x i8> [[TMP2]] to i128
  // LLVM-NEXT:    ret i128 [[TMP3]]
  return vaddq_p128(a, b);
}

//===----------------------------------------------------------------------===//
// 2.1.1.1.2. Widening addition
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#widening-addition
//===----------------------------------------------------------------------===//

// LLVM-LABEL: @test_vaddl_s8(
// CIR-LABEL: @vaddl_s8(
int16x8_t test_vaddl_s8(int8x8_t a, int8x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !s16i>

  // LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I5_I:%.*]] = sext <8 x i8> [[A]] to <8 x i16>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = sext <8 x i8> [[B]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddl_s8(a, b);
}

// LLVM-LABEL: @test_vaddl_s16(
// CIR-LABEL: @vaddl_s16(
int32x4_t test_vaddl_s16(int16x4_t a, int16x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !s32i>

  // LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM-NEXT:    [[VMOVL_I5_I:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM-NEXT:    [[TMP2:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = sext <4 x i16> [[TMP3]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddl_s16(a, b);
}

// LLVM-LABEL: @test_vaddl_s32(
// CIR-LABEL: @vaddl_s32(
int64x2_t test_vaddl_s32(int32x2_t a, int32x2_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !s64i>

  // LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM-NEXT:    [[VMOVL_I5_I:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM-NEXT:    [[TMP2:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = sext <2 x i32> [[TMP3]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddl_s32(a, b);
}

// LLVM-LABEL: @test_vaddl_u8(
// CIR-LABEL: @vaddl_u8(
uint16x8_t test_vaddl_u8(uint8x8_t a, uint8x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !u16i>

  // LLVM-SAME: <8 x i8> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I5_I:%.*]] = zext <8 x i8> [[A]] to <8 x i16>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = zext <8 x i8> [[B]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddl_u8(a, b);
}

// LLVM-LABEL: @test_vaddl_u16(
// CIR-LABEL: @vaddl_u16(
uint32x4_t test_vaddl_u16(uint16x4_t a, uint16x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !u32i>

  // LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM-NEXT:    [[VMOVL_I5_I:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM-NEXT:    [[TMP2:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = zext <4 x i16> [[TMP3]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddl_u16(a, b);
}

// LLVM-LABEL: @test_vaddl_u32(
// CIR-LABEL: @vaddl_u32(
uint64x2_t test_vaddl_u32(uint32x2_t a, uint32x2_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !u64i>

  // LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM-NEXT:    [[VMOVL_I5_I:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM-NEXT:    [[TMP2:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = zext <2 x i32> [[TMP3]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddl_u32(a, b);
}

// LLVM-LABEL: @test_vaddl_high_s8(
// CIR-LABEL: @vaddl_high_s8(
int16x8_t test_vaddl_high_s8(int8x16_t a, int8x16_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !s16i>

  // LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I12_I]] to <8 x i16>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP1:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddl_high_s8(a, b);
}

// LLVM-LABEL: @test_vaddl_high_s16(
// CIR-LABEL: @vaddl_high_s16(
int32x4_t test_vaddl_high_s16(int16x8_t a, int16x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !s32i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I12_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM-NEXT:    [[TMP2:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP4:%.*]] = bitcast <8 x i8> [[TMP3]] to <4 x i16>
  // LLVM-NEXT:    [[TMP5:%.*]] = sext <4 x i16> [[TMP4]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[TMP2]], [[TMP5]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddl_high_s16(a, b);
}

// LLVM-LABEL: @test_vaddl_high_s32(
// CIR-LABEL: @vaddl_high_s32(
int64x2_t test_vaddl_high_s32(int32x4_t a, int32x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !s64i>

  // LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[A]], <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I12_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM-NEXT:    [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP4:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x i32>
  // LLVM-NEXT:    [[TMP5:%.*]] = sext <2 x i32> [[TMP4]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[TMP2]], [[TMP5]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddl_high_s32(a, b);
}

// LLVM-LABEL: @test_vaddl_high_u8(
// CIR-LABEL: @vaddl_high_u8(
uint16x8_t test_vaddl_high_u8(uint8x16_t a, uint8x16_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !u16i>

  // LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> [[A]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I12_I]] to <8 x i16>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP1:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddl_high_u8(a, b);
}

// LLVM-LABEL: @test_vaddl_high_u16(
// CIR-LABEL: @vaddl_high_u16(
uint32x4_t test_vaddl_high_u16(uint16x8_t a, uint16x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !u32i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I12_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM-NEXT:    [[TMP2:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP4:%.*]] = bitcast <8 x i8> [[TMP3]] to <4 x i16>
  // LLVM-NEXT:    [[TMP5:%.*]] = zext <4 x i16> [[TMP4]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[TMP2]], [[TMP5]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddl_high_u16(a, b);
}

// LLVM-LABEL: @test_vaddl_high_u32(
// CIR-LABEL: @vaddl_high_u32(
uint64x2_t test_vaddl_high_u32(uint32x4_t a, uint32x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !u64i>

  // LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> [[A]], <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I12_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM-NEXT:    [[TMP2:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP3:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP4:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x i32>
  // LLVM-NEXT:    [[TMP5:%.*]] = zext <2 x i32> [[TMP4]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[TMP2]], [[TMP5]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddl_high_u32(a, b);
}

// LLVM-LABEL: @test_vaddw_s8(
// CIR-LABEL: @vaddw_s8(
int16x8_t test_vaddw_s8(int16x8_t a, int8x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !s16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I_I:%.*]] = sext <8 x i8> [[B]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddw_s8(a, b);
}

// LLVM-LABEL: @test_vaddw_s16(
// CIR-LABEL: @vaddw_s16(
int32x4_t test_vaddw_s16(int32x4_t a, int16x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !s32i>

  // LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddw_s16(a, b);
}

// LLVM-LABEL: @test_vaddw_s32(
// CIR-LABEL: @vaddw_s32(
int64x2_t test_vaddw_s32(int64x2_t a, int32x2_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !s64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddw_s32(a, b);
}

// LLVM-LABEL: @test_vaddw_u8(
// CIR-LABEL: @vaddw_u8(
uint16x8_t test_vaddw_u8(uint16x8_t a, uint8x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !u16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I_I:%.*]] = zext <8 x i8> [[B]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddw_u8(a, b);
}

// LLVM-LABEL: @test_vaddw_u16(
// CIR-LABEL: @vaddw_u16(
uint32x4_t test_vaddw_u16(uint32x4_t a, uint16x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !u32i>

  // LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddw_u16(a, b);
}

// LLVM-LABEL: @test_vaddw_u32(
// CIR-LABEL: @vaddw_u32(
uint64x2_t test_vaddw_u32(uint64x2_t a, uint32x2_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !u64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddw_u32(a, b);
}

// LLVM-LABEL: @test_vaddw_high_s8(
// CIR-LABEL: @vaddw_high_s8(
int16x8_t test_vaddw_high_s8(int16x8_t a, int8x16_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !s16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[A]], [[TMP0]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddw_high_s8(a, b);
}

// LLVM-LABEL: @test_vaddw_high_s16(
// CIR-LABEL: @vaddw_high_s16(
int32x4_t test_vaddw_high_s16(int32x4_t a, int16x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !s32i>

  // LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM-NEXT:    [[TMP2:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[A]], [[TMP2]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddw_high_s16(a, b);
}

// LLVM-LABEL: @test_vaddw_high_s32(
// CIR-LABEL: @vaddw_high_s32(
int64x2_t test_vaddw_high_s32(int64x2_t a, int32x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !s64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM-NEXT:    [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[A]], [[TMP2]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddw_high_s32(a, b);
}

// LLVM-LABEL: @test_vaddw_high_u8(
// CIR-LABEL: @vaddw_high_u8(
uint16x8_t test_vaddw_high_u8(uint16x8_t a, uint8x16_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !u16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> [[B]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <8 x i16> [[A]], [[TMP0]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddw_high_u8(a, b);
}

// LLVM-LABEL: @test_vaddw_high_u16(
// CIR-LABEL: @vaddw_high_u16(
uint32x4_t test_vaddw_high_u16(uint32x4_t a, uint16x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !u32i>

  // LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> [[B]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM-NEXT:    [[TMP2:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[A]], [[TMP2]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddw_high_u16(a, b);
}

// LLVM-LABEL: @test_vaddw_high_u32(
// CIR-LABEL: @vaddw_high_u32(
uint64x2_t test_vaddw_high_u32(uint64x2_t a, uint32x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !u64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> [[B]], <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
  // LLVM-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM-NEXT:    [[TMP2:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[A]], [[TMP2]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddw_high_u32(a, b);
}
