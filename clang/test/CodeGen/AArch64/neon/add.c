// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefix=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefix=LLVM %}
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
  // LLVM:    [[TMP0:%.*]] = xor <4 x i16> [[A]], [[B]]
  // LLVM-NEXT:    ret <4 x i16> [[TMP0]]
  return vadd_p16(a, b);
}

// LLVM-LABEL: @test_vadd_p64(
// CIR-LABEL: @vadd_p64(
poly64x1_t test_vadd_p64(poly64x1_t a, poly64x1_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<8 x !u8i>

  // LLVM-SAME: <1 x i64> {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = xor <1 x i64> [[A]], [[B]]
  // LLVM-NEXT:    ret <1 x i64> [[TMP0]]
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
  // LLVM:    [[TMP0:%.*]] = xor <8 x i16> [[A]], [[B]]
  // LLVM-NEXT:    ret <8 x i16> [[TMP0]]
  return vaddq_p16(a, b);
}

// LLVM-LABEL: @test_vaddq_p64(
// CIR-LABEL: @vaddq_p64(
poly64x2_t test_vaddq_p64(poly64x2_t a, poly64x2_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<16 x !u8i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = xor <2 x i64> [[A]], [[B]]
  // LLVM-NEXT:    ret <2 x i64> [[TMP0]]
  return vaddq_p64(a, b);
}

// LLVM-LABEL: @test_vaddq_p128(
// CIR-LABEL: @vaddq_p128(
poly128_t test_vaddq_p128(poly128_t a, poly128_t b) {
  // CIR: cir.xor {{.*}} : !cir.vector<16 x !u8i>

  // LLVM-SAME: i128 {{.*}} [[A:%.*]], i128 {{.*}} [[B:%.*]])
  // LLVM:    [[TMP0:%.*]] = xor i128 [[A]], [[B]]
  // LLVM-NEXT:    ret i128 [[TMP0]]
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
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nsw <8 x i16> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddl_s8(a, b);
}

// LLVM-LABEL: @test_vaddl_s16(
// CIR-LABEL: @vaddl_s16(
int32x4_t test_vaddl_s16(int16x4_t a, int16x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !s32i>

  // LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I5_I:%.*]] = sext <4 x i16> [[A]] to <4 x i32>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = sext <4 x i16> [[B]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nsw <4 x i32> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddl_s16(a, b);
}

// LLVM-LABEL: @test_vaddl_s32(
// CIR-LABEL: @vaddl_s32(
int64x2_t test_vaddl_s32(int32x2_t a, int32x2_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !s64i>

  // LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I5_I:%.*]] = sext <2 x i32> [[A]] to <2 x i64>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = sext <2 x i32> [[B]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nsw <2 x i64> [[VMOVL_I5_I]], [[VMOVL_I_I]]
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
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nuw nsw <8 x i16> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddl_u8(a, b);
}

// LLVM-LABEL: @test_vaddl_u16(
// CIR-LABEL: @vaddl_u16(
uint32x4_t test_vaddl_u16(uint16x4_t a, uint16x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !u32i>

  // LLVM-SAME: <4 x i16> {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I5_I:%.*]] = zext <4 x i16> [[A]] to <4 x i32>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = zext <4 x i16> [[B]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nuw nsw <4 x i32> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddl_u16(a, b);
}

// LLVM-LABEL: @test_vaddl_u32(
// CIR-LABEL: @vaddl_u32(
uint64x2_t test_vaddl_u32(uint32x2_t a, uint32x2_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !u64i>

  // LLVM-SAME: <2 x i32> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I5_I:%.*]] = zext <2 x i32> [[A]] to <2 x i64>
  // LLVM-NEXT:    [[VMOVL_I_I:%.*]] = zext <2 x i32> [[B]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nuw nsw <2 x i64> [[VMOVL_I5_I]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddl_u32(a, b);
}

// LLVM-LABEL: @test_vaddl_high_s8(
// CIR-LABEL: @vaddl_high_s8(
int16x8_t test_vaddl_high_s8(int8x16_t a, int8x16_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !s16i>

  // LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I12_I]] to <8 x i16>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP1:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nsw <8 x i16> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddl_high_s8(a, b);
}

// LLVM-LABEL: @test_vaddl_high_s16(
// CIR-LABEL: @vaddl_high_s16(
int32x4_t test_vaddl_high_s16(int16x8_t a, int16x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !s32i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP0:%.*]] = sext <4 x i16> [[SHUFFLE_I_I12_I]] to <4 x i32>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nsw <4 x i32> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddl_high_s16(a, b);
}

// LLVM-LABEL: @test_vaddl_high_s32(
// CIR-LABEL: @vaddl_high_s32(
int64x2_t test_vaddl_high_s32(int32x4_t a, int32x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !s64i>

  // LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> poison, <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP0:%.*]] = sext <2 x i32> [[SHUFFLE_I_I12_I]] to <2 x i64>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> poison, <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nsw <2 x i64> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddl_high_s32(a, b);
}

// LLVM-LABEL: @test_vaddl_high_u8(
// CIR-LABEL: @vaddl_high_u8(
uint16x8_t test_vaddl_high_u8(uint8x16_t a, uint8x16_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !u16i>

  // LLVM-SAME: <16 x i8> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <16 x i8> [[A]], <16 x i8> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I12_I]] to <8 x i16>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM-NEXT:    [[TMP1:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nuw nsw <8 x i16> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    ret <8 x i16> [[ADD_I]]
  return vaddl_high_u8(a, b);
}

// LLVM-LABEL: @test_vaddl_high_u16(
// CIR-LABEL: @vaddl_high_u16(
uint32x4_t test_vaddl_high_u16(uint16x8_t a, uint16x8_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<4 x !u32i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <8 x i16> [[A]], <8 x i16> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP0:%.*]] = zext <4 x i16> [[SHUFFLE_I_I12_I]] to <4 x i32>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nuw nsw <4 x i32> [[TMP0]], [[TMP1]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddl_high_u16(a, b);
}

// LLVM-LABEL: @test_vaddl_high_u32(
// CIR-LABEL: @vaddl_high_u32(
uint64x2_t test_vaddl_high_u32(uint32x4_t a, uint32x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !u64i>

  // LLVM-SAME: <4 x i32> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I12_I:%.*]] = shufflevector <4 x i32> [[A]], <4 x i32> poison, <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP0:%.*]] = zext <2 x i32> [[SHUFFLE_I_I12_I]] to <2 x i64>
  // LLVM-NEXT:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> poison, <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add nuw nsw <2 x i64> [[TMP0]], [[TMP1]]
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
  // LLVM:    [[VMOVL_I_I:%.*]] = sext <4 x i16> [[B]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddw_s16(a, b);
}

// LLVM-LABEL: @test_vaddw_s32(
// CIR-LABEL: @vaddw_s32(
int64x2_t test_vaddw_s32(int64x2_t a, int32x2_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !s64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I_I:%.*]] = sext <2 x i32> [[B]] to <2 x i64>
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
  // LLVM:    [[VMOVL_I_I:%.*]] = zext <4 x i16> [[B]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddw_u16(a, b);
}

// LLVM-LABEL: @test_vaddw_u32(
// CIR-LABEL: @vaddw_u32(
uint64x2_t test_vaddw_u32(uint64x2_t a, uint32x2_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !u64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[VMOVL_I_I:%.*]] = zext <2 x i32> [[B]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[A]], [[VMOVL_I_I]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddw_u32(a, b);
}

// LLVM-LABEL: @test_vaddw_high_s8(
// CIR-LABEL: @vaddw_high_s8(
int16x8_t test_vaddw_high_s8(int16x8_t a, int8x16_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !s16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
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
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP0:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[A]], [[TMP0]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddw_high_s16(a, b);
}

// LLVM-LABEL: @test_vaddw_high_s32(
// CIR-LABEL: @vaddw_high_s32(
int64x2_t test_vaddw_high_s32(int64x2_t a, int32x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !s64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> poison, <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP0:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[A]], [[TMP0]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddw_high_s32(a, b);
}

// LLVM-LABEL: @test_vaddw_high_u8(
// CIR-LABEL: @vaddw_high_u8(
uint16x8_t test_vaddw_high_u8(uint16x8_t a, uint8x16_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<8 x !u16i>

  // LLVM-SAME: <8 x i16> {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> [[B]], <16 x i8> poison, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
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
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> [[B]], <8 x i16> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[TMP0:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <4 x i32> [[A]], [[TMP0]]
  // LLVM-NEXT:    ret <4 x i32> [[ADD_I]]
  return vaddw_high_u16(a, b);
}

// LLVM-LABEL: @test_vaddw_high_u32(
// CIR-LABEL: @vaddw_high_u32(
uint64x2_t test_vaddw_high_u32(uint64x2_t a, uint32x4_t b) {
  // CIR: cir.add {{.*}} : !cir.vector<2 x !u64i>

  // LLVM-SAME: <2 x i64> {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
  // LLVM:    [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> [[B]], <4 x i32> poison, <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT:    [[TMP0:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
  // LLVM-NEXT:    [[ADD_I:%.*]] = add <2 x i64> [[A]], [[TMP0]]
  // LLVM-NEXT:    ret <2 x i64> [[ADD_I]]
  return vaddw_high_u32(a, b);
}

//===----------------------------------------------------------------------===//
// 2.1.1.1.3. Narrowing addition
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#narrowing-addition
//===----------------------------------------------------------------------===//

// LLVM-LABEL: @test_vhadd_s8(
// CIR-LABEL: @vhadd_s8(
int8x8_t test_vhadd_s8(int8x8_t v1, int8x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shadd"

  // LLVM-SAME: <8 x i8> {{.*}}[[V1:%.*]], <8 x i8> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.shadd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[RES]]
  return vhadd_s8(v1, v2);
}

// LLVM-LABEL: @test_vhadd_s16(
// CIR-LABEL: @vhadd_s16(
int16x4_t test_vhadd_s16(int16x4_t v1, int16x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shadd"

  // LLVM-SAME: <4 x i16> {{.*}}[[V1:%.*]], <4 x i16> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.shadd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM: ret <4 x i16> [[RES]]
  return vhadd_s16(v1, v2);
}

// LLVM-LABEL: @test_vhadd_s32(
// CIR-LABEL: @vhadd_s32(
int32x2_t test_vhadd_s32(int32x2_t v1, int32x2_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shadd"

  // LLVM-SAME: <2 x i32> {{.*}}[[V1:%.*]], <2 x i32> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.shadd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM: ret <2 x i32> [[RES]]
  return vhadd_s32(v1, v2);
}

// LLVM-LABEL: @test_vhadd_u8(
// CIR-LABEL: @vhadd_u8(
uint8x8_t test_vhadd_u8(uint8x8_t v1, uint8x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhadd"

  // LLVM-SAME: <8 x i8> {{.*}}[[V1:%.*]], <8 x i8> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.uhadd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[RES]]
  return vhadd_u8(v1, v2);
}

// LLVM-LABEL: @test_vhadd_u16(
// CIR-LABEL: @vhadd_u16(
uint16x4_t test_vhadd_u16(uint16x4_t v1, uint16x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhadd"

  // LLVM-SAME: <4 x i16> {{.*}}[[V1:%.*]], <4 x i16> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.uhadd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM: ret <4 x i16> [[RES]]
  return vhadd_u16(v1, v2);
}

// LLVM-LABEL: @test_vhadd_u32(
// CIR-LABEL: @vhadd_u32(
uint32x2_t test_vhadd_u32(uint32x2_t v1, uint32x2_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhadd"

  // LLVM-SAME: <2 x i32> {{.*}}[[V1:%.*]], <2 x i32> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.uhadd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM: ret <2 x i32> [[RES]]
  return vhadd_u32(v1, v2);
}

// LLVM-LABEL: @test_vhaddq_s8(
// CIR-LABEL: @vhaddq_s8(
int8x16_t test_vhaddq_s8(int8x16_t v1, int8x16_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shadd"

  // LLVM-SAME: <16 x i8> {{.*}}[[V1:%.*]], <16 x i8> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.shadd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[RES]]
  return vhaddq_s8(v1, v2);
}

// LLVM-LABEL: @test_vhaddq_s16(
// CIR-LABEL: @vhaddq_s16(
int16x8_t test_vhaddq_s16(int16x8_t v1, int16x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shadd"

  // LLVM-SAME: <8 x i16> {{.*}}[[V1:%.*]], <8 x i16> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.shadd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM: ret <8 x i16> [[RES]]
  return vhaddq_s16(v1, v2);
}

// LLVM-LABEL: @test_vhaddq_s32(
// CIR-LABEL: @vhaddq_s32(
int32x4_t test_vhaddq_s32(int32x4_t v1, int32x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.shadd"

  // LLVM-SAME: <4 x i32> {{.*}}[[V1:%.*]], <4 x i32> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.shadd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM: ret <4 x i32> [[RES]]
  return vhaddq_s32(v1, v2);
}

// LLVM-LABEL: @test_vhaddq_u8(
// CIR-LABEL: @vhaddq_u8(
uint8x16_t test_vhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhadd"

  // LLVM-SAME: <16 x i8> {{.*}}[[V1:%.*]], <16 x i8> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.uhadd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[RES]]
  return vhaddq_u8(v1, v2);
}

// LLVM-LABEL: @test_vhaddq_u16(
// CIR-LABEL: @vhaddq_u16(
uint16x8_t test_vhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhadd"

  // LLVM-SAME: <8 x i16> {{.*}}[[V1:%.*]], <8 x i16> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.uhadd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM: ret <8 x i16> [[RES]]
  return vhaddq_u16(v1, v2);
}

// LLVM-LABEL: @test_vhaddq_u32(
// CIR-LABEL: @vhaddq_u32(
uint32x4_t test_vhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.uhadd"

  // LLVM-SAME: <4 x i32> {{.*}}[[V1:%.*]], <4 x i32> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.uhadd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM: ret <4 x i32> [[RES]]
  return vhaddq_u32(v1, v2);
}

// LLVM-LABEL: @test_vrhadd_s8(
// CIR-LABEL: @vrhadd_s8(
int8x8_t test_vrhadd_s8(int8x8_t v1, int8x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.srhadd"

  // LLVM-SAME: <8 x i8> {{.*}}[[V1:%.*]], <8 x i8> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.srhadd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[RES]]
  return vrhadd_s8(v1, v2);
}

// LLVM-LABEL: @test_vrhadd_s16(
// CIR-LABEL: @vrhadd_s16(
int16x4_t test_vrhadd_s16(int16x4_t v1, int16x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.srhadd"

  // LLVM-SAME: <4 x i16> {{.*}}[[V1:%.*]], <4 x i16> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.srhadd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM: ret <4 x i16> [[RES]]
  return vrhadd_s16(v1, v2);
}

// LLVM-LABEL: @test_vrhadd_s32(
// CIR-LABEL: @vrhadd_s32(
int32x2_t test_vrhadd_s32(int32x2_t v1, int32x2_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.srhadd"

  // LLVM-SAME: <2 x i32> {{.*}}[[V1:%.*]], <2 x i32> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.srhadd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM: ret <2 x i32> [[RES]]
  return vrhadd_s32(v1, v2);
}

// LLVM-LABEL: @test_vrhadd_u8(
// CIR-LABEL: @vrhadd_u8(
uint8x8_t test_vrhadd_u8(uint8x8_t v1, uint8x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.urhadd"

  // LLVM-SAME: <8 x i8> {{.*}}[[V1:%.*]], <8 x i8> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.urhadd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[RES]]
  return vrhadd_u8(v1, v2);
}

// LLVM-LABEL: @test_vrhadd_u16(
// CIR-LABEL: @vrhadd_u16(
uint16x4_t test_vrhadd_u16(uint16x4_t v1, uint16x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.urhadd"

  // LLVM-SAME: <4 x i16> {{.*}}[[V1:%.*]], <4 x i16> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.urhadd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM: ret <4 x i16> [[RES]]
  return vrhadd_u16(v1, v2);
}

// LLVM-LABEL: @test_vrhadd_u32(
// CIR-LABEL: @vrhadd_u32(
uint32x2_t test_vrhadd_u32(uint32x2_t v1, uint32x2_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.urhadd"

  // LLVM-SAME: <2 x i32> {{.*}}[[V1:%.*]], <2 x i32> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.urhadd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM: ret <2 x i32> [[RES]]
  return vrhadd_u32(v1, v2);
}

// LLVM-LABEL: @test_vrhaddq_s8(
// CIR-LABEL: @vrhaddq_s8(
int8x16_t test_vrhaddq_s8(int8x16_t v1, int8x16_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.srhadd"

  // LLVM-SAME: <16 x i8> {{.*}}[[V1:%.*]], <16 x i8> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.srhadd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[RES]]
  return vrhaddq_s8(v1, v2);
}

// LLVM-LABEL: @test_vrhaddq_s16(
// CIR-LABEL: @vrhaddq_s16(
int16x8_t test_vrhaddq_s16(int16x8_t v1, int16x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.srhadd"

  // LLVM-SAME: <8 x i16> {{.*}}[[V1:%.*]], <8 x i16> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.srhadd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM: ret <8 x i16> [[RES]]
  return vrhaddq_s16(v1, v2);
}

// LLVM-LABEL: @test_vrhaddq_s32(
// CIR-LABEL: @vrhaddq_s32(
int32x4_t test_vrhaddq_s32(int32x4_t v1, int32x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.srhadd"

  // LLVM-SAME: <4 x i32> {{.*}}[[V1:%.*]], <4 x i32> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.srhadd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM: ret <4 x i32> [[RES]]
  return vrhaddq_s32(v1, v2);
}

// LLVM-LABEL: @test_vrhaddq_u8(
// CIR-LABEL: @vrhaddq_u8(
uint8x16_t test_vrhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.urhadd"

  // LLVM-SAME: <16 x i8> {{.*}}[[V1:%.*]], <16 x i8> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <16 x i8> @llvm.aarch64.neon.urhadd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[RES]]
  return vrhaddq_u8(v1, v2);
}

// LLVM-LABEL: @test_vrhaddq_u16(
// CIR-LABEL: @vrhaddq_u16(
uint16x8_t test_vrhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.urhadd"

  // LLVM-SAME: <8 x i16> {{.*}}[[V1:%.*]], <8 x i16> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i16> @llvm.aarch64.neon.urhadd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM: ret <8 x i16> [[RES]]
  return vrhaddq_u16(v1, v2);
}

// LLVM-LABEL: @test_vrhaddq_u32(
// CIR-LABEL: @vrhaddq_u32(
uint32x4_t test_vrhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.urhadd"

  // LLVM-SAME: <4 x i32> {{.*}}[[V1:%.*]], <4 x i32> {{.*}}[[V2:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i32> @llvm.aarch64.neon.urhadd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM: ret <4 x i32> [[RES]]
  return vrhaddq_u32(v1, v2);
}

// LLVM-LABEL: @test_vaddhn_s16(
// CIR-LABEL: @vaddhn_s16(
int8x8_t test_vaddhn_s16(int16x8_t a, int16x8_t b) {
  // CIR: cir.add
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <8 x i16> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <8 x i16> [[ADD]], splat (i16 8)
  // LLVM: [[TR:%.*]] = trunc nuw <8 x i16> [[SH]] to <8 x i8>
  // LLVM: ret <8 x i8> [[TR]]
  return vaddhn_s16(a, b);
}

// LLVM-LABEL: @test_vaddhn_s32(
// CIR-LABEL: @vaddhn_s32(
int16x4_t test_vaddhn_s32(int32x4_t a, int32x4_t b) {
  // CIR: cir.add
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <4 x i32> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <4 x i32> [[ADD]], splat (i32 16)
  // LLVM: [[TR:%.*]] = trunc nuw <4 x i32> [[SH]] to <4 x i16>
  // LLVM: ret <4 x i16> [[TR]]
  return vaddhn_s32(a, b);
}

// LLVM-LABEL: @test_vaddhn_s64(
// CIR-LABEL: @vaddhn_s64(
int32x2_t test_vaddhn_s64(int64x2_t a, int64x2_t b) {
  // CIR: cir.add
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <2 x i64> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <2 x i64> [[ADD]], splat (i64 32)
  // LLVM: [[TR:%.*]] = trunc nuw <2 x i64> [[SH]] to <2 x i32>
  // LLVM: ret <2 x i32> [[TR]]
  return vaddhn_s64(a, b);
}

// LLVM-LABEL: @test_vaddhn_u16(
// CIR-LABEL: @vaddhn_u16(
uint8x8_t test_vaddhn_u16(uint16x8_t a, uint16x8_t b) {
  // CIR: cir.add
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <8 x i16> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <8 x i16> [[ADD]], splat (i16 8)
  // LLVM: [[TR:%.*]] = trunc nuw <8 x i16> [[SH]] to <8 x i8>
  // LLVM: ret <8 x i8> [[TR]]
  return vaddhn_u16(a, b);
}

// LLVM-LABEL: @test_vaddhn_u32(
// CIR-LABEL: @vaddhn_u32(
uint16x4_t test_vaddhn_u32(uint32x4_t a, uint32x4_t b) {
  // CIR: cir.add
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <4 x i32> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <4 x i32> [[ADD]], splat (i32 16)
  // LLVM: [[TR:%.*]] = trunc nuw <4 x i32> [[SH]] to <4 x i16>
  // LLVM: ret <4 x i16> [[TR]]
  return vaddhn_u32(a, b);
}

// LLVM-LABEL: @test_vaddhn_u64(
// CIR-LABEL: @vaddhn_u64(
uint32x2_t test_vaddhn_u64(uint64x2_t a, uint64x2_t b) {
  // CIR: cir.add
  // CIR: cir.shift(right
  // CIR: cir.cast integral

  // LLVM-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <2 x i64> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <2 x i64> [[ADD]], splat (i64 32)
  // LLVM: [[TR:%.*]] = trunc nuw <2 x i64> [[SH]] to <2 x i32>
  // LLVM: ret <2 x i32> [[TR]]
  return vaddhn_u64(a, b);
}

// LLVM-LABEL: @test_vaddhn_high_s16(
// CIR-LABEL: @vaddhn_high_s16(
int8x16_t test_vaddhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  // CIR: cir.call @vaddhn_s16(
  // CIR: cir.call @vcombine_s8(

  // LLVM-SAME: <8 x i8> {{.*}}[[R:%.*]], <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <8 x i16> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <8 x i16> [[ADD]], splat (i16 8)
  // LLVM: [[TR:%.*]] = trunc nuw <8 x i16> [[SH]] to <8 x i8>
  // LLVM: [[RES:%.*]] = shufflevector <8 x i8> [[R]], <8 x i8> [[TR]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: ret <16 x i8> [[RES]]
  return vaddhn_high_s16(r, a, b);
}

// LLVM-LABEL: @test_vaddhn_high_s32(
// CIR-LABEL: @vaddhn_high_s32(
int16x8_t test_vaddhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  // CIR: cir.call @vaddhn_s32(
  // CIR: cir.call @vcombine_s16(

  // LLVM-SAME: <4 x i16> {{.*}}[[R:%.*]], <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <4 x i32> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <4 x i32> [[ADD]], splat (i32 16)
  // LLVM: [[TR:%.*]] = trunc nuw <4 x i32> [[SH]] to <4 x i16>
  // LLVM: [[RES:%.*]] = shufflevector <4 x i16> [[R]], <4 x i16> [[TR]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: ret <8 x i16> [[RES]]
  return vaddhn_high_s32(r, a, b);
}

// LLVM-LABEL: @test_vaddhn_high_s64(
// CIR-LABEL: @vaddhn_high_s64(
int32x4_t test_vaddhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  // CIR: cir.call @vaddhn_s64(
  // CIR: cir.call @vcombine_s32(

  // LLVM-SAME: <2 x i32> {{.*}}[[R:%.*]], <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <2 x i64> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <2 x i64> [[ADD]], splat (i64 32)
  // LLVM: [[TR:%.*]] = trunc nuw <2 x i64> [[SH]] to <2 x i32>
  // LLVM: [[RES:%.*]] = shufflevector <2 x i32> [[R]], <2 x i32> [[TR]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: ret <4 x i32> [[RES]]
  return vaddhn_high_s64(r, a, b);
}

// LLVM-LABEL: @test_vaddhn_high_u16(
// CIR-LABEL: @vaddhn_high_u16(
uint8x16_t test_vaddhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  // CIR: cir.call @vaddhn_u16(
  // CIR: cir.call @vcombine_u8(

  // LLVM-SAME: <8 x i8> {{.*}}[[R:%.*]], <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <8 x i16> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <8 x i16> [[ADD]], splat (i16 8)
  // LLVM: [[TR:%.*]] = trunc nuw <8 x i16> [[SH]] to <8 x i8>
  // LLVM: [[RES:%.*]] = shufflevector <8 x i8> [[R]], <8 x i8> [[TR]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: ret <16 x i8> [[RES]]
  return vaddhn_high_u16(r, a, b);
}

// LLVM-LABEL: @test_vaddhn_high_u32(
// CIR-LABEL: @vaddhn_high_u32(
uint16x8_t test_vaddhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  // CIR: cir.call @vaddhn_u32(
  // CIR: cir.call @vcombine_u16(

  // LLVM-SAME: <4 x i16> {{.*}}[[R:%.*]], <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <4 x i32> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <4 x i32> [[ADD]], splat (i32 16)
  // LLVM: [[TR:%.*]] = trunc nuw <4 x i32> [[SH]] to <4 x i16>
  // LLVM: [[RES:%.*]] = shufflevector <4 x i16> [[R]], <4 x i16> [[TR]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: ret <8 x i16> [[RES]]
  return vaddhn_high_u32(r, a, b);
}

// LLVM-LABEL: @test_vaddhn_high_u64(
// CIR-LABEL: @vaddhn_high_u64(
uint32x4_t test_vaddhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  // CIR: cir.call @vaddhn_u64(
  // CIR: cir.call @vcombine_u32(

  // LLVM-SAME: <2 x i32> {{.*}}[[R:%.*]], <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM: [[ADD:%.*]] = add <2 x i64> [[A]], [[B]]
  // LLVM: [[SH:%.*]] = lshr <2 x i64> [[ADD]], splat (i64 32)
  // LLVM: [[TR:%.*]] = trunc nuw <2 x i64> [[SH]] to <2 x i32>
  // LLVM: [[RES:%.*]] = shufflevector <2 x i32> [[R]], <2 x i32> [[TR]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: ret <4 x i32> [[RES]]
  return vaddhn_high_u64(r, a, b);
}

// LLVM-LABEL: @test_vraddhn_s16(
// CIR-LABEL: @vraddhn_s16(
int8x8_t test_vraddhn_s16(int16x8_t a, int16x8_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.raddhn"

  // LLVM-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM: ret <8 x i8> [[RES]]
  return vraddhn_s16(a, b);
}

// LLVM-LABEL: @test_vraddhn_s32(
// CIR-LABEL: @vraddhn_s32(
int16x4_t test_vraddhn_s32(int32x4_t a, int32x4_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.raddhn"

  // LLVM-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM: ret <4 x i16> [[RES]]
  return vraddhn_s32(a, b);
}

// LLVM-LABEL: @test_vraddhn_s64(
// CIR-LABEL: @vraddhn_s64(
int32x2_t test_vraddhn_s64(int64x2_t a, int64x2_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.raddhn"

  // LLVM-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> [[A]], <2 x i64> [[B]])
  // LLVM: ret <2 x i32> [[RES]]
  return vraddhn_s64(a, b);
}

// LLVM-LABEL: @test_vraddhn_u16(
// CIR-LABEL: @vraddhn_u16(
uint8x8_t test_vraddhn_u16(uint16x8_t a, uint16x8_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.raddhn"

  // LLVM-SAME: <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM: ret <8 x i8> [[RES]]
  return vraddhn_u16(a, b);
}

// LLVM-LABEL: @test_vraddhn_u32(
// CIR-LABEL: @vraddhn_u32(
uint16x4_t test_vraddhn_u32(uint32x4_t a, uint32x4_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.raddhn"

  // LLVM-SAME: <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM: ret <4 x i16> [[RES]]
  return vraddhn_u32(a, b);
}

// LLVM-LABEL: @test_vraddhn_u64(
// CIR-LABEL: @vraddhn_u64(
uint32x2_t test_vraddhn_u64(uint64x2_t a, uint64x2_t b) {
  // CIR: cir.call_llvm_intrinsic "aarch64.neon.raddhn"

  // LLVM-SAME: <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM: [[RES:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> [[A]], <2 x i64> [[B]])
  // LLVM: ret <2 x i32> [[RES]]
  return vraddhn_u64(a, b);
}

// LLVM-LABEL: @test_vraddhn_high_s16(
// CIR-LABEL: @vraddhn_high_s16(
int8x16_t test_vraddhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  // CIR: cir.call @vraddhn_s16(
  // CIR: cir.call @vcombine_s8(

  // LLVM-SAME: <8 x i8> {{.*}}[[R:%.*]], <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM: [[TMP:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM: [[RES:%.*]] = shufflevector <8 x i8> [[R]], <8 x i8> [[TMP]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: ret <16 x i8> [[RES]]
  return vraddhn_high_s16(r, a, b);
}

// LLVM-LABEL: @test_vraddhn_high_s32(
// CIR-LABEL: @vraddhn_high_s32(
int16x8_t test_vraddhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  // CIR: cir.call @vraddhn_s32(
  // CIR: cir.call @vcombine_s16(

  // LLVM-SAME: <4 x i16> {{.*}}[[R:%.*]], <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM: [[TMP:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM: [[RES:%.*]] = shufflevector <4 x i16> [[R]], <4 x i16> [[TMP]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: ret <8 x i16> [[RES]]
  return vraddhn_high_s32(r, a, b);
}

// LLVM-LABEL: @test_vraddhn_high_s64(
// CIR-LABEL: @vraddhn_high_s64(
int32x4_t test_vraddhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  // CIR: cir.call @vraddhn_s64(
  // CIR: cir.call @vcombine_s32(

  // LLVM-SAME: <2 x i32> {{.*}}[[R:%.*]], <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM: [[TMP:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> [[A]], <2 x i64> [[B]])
  // LLVM: [[RES:%.*]] = shufflevector <2 x i32> [[R]], <2 x i32> [[TMP]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: ret <4 x i32> [[RES]]
  return vraddhn_high_s64(r, a, b);
}

// LLVM-LABEL: @test_vraddhn_high_u16(
// CIR-LABEL: @vraddhn_high_u16(
uint8x16_t test_vraddhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  // CIR: cir.call @vraddhn_u16(
  // CIR: cir.call @vcombine_u8(

  // LLVM-SAME: <8 x i8> {{.*}}[[R:%.*]], <8 x i16> {{.*}}[[A:%.*]], <8 x i16> {{.*}}[[B:%.*]])
  // LLVM: [[TMP:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM: [[RES:%.*]] = shufflevector <8 x i8> [[R]], <8 x i8> [[TMP]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: ret <16 x i8> [[RES]]
  return vraddhn_high_u16(r, a, b);
}

// LLVM-LABEL: @test_vraddhn_high_u32(
// CIR-LABEL: @vraddhn_high_u32(
uint16x8_t test_vraddhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  // CIR: cir.call @vraddhn_u32(
  // CIR: cir.call @vcombine_u16(

  // LLVM-SAME: <4 x i16> {{.*}}[[R:%.*]], <4 x i32> {{.*}}[[A:%.*]], <4 x i32> {{.*}}[[B:%.*]])
  // LLVM: [[TMP:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM: [[RES:%.*]] = shufflevector <4 x i16> [[R]], <4 x i16> [[TMP]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM: ret <8 x i16> [[RES]]
  return vraddhn_high_u32(r, a, b);
}

// LLVM-LABEL: @test_vraddhn_high_u64(
// CIR-LABEL: @vraddhn_high_u64(
uint32x4_t test_vraddhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  // CIR: cir.call @vraddhn_u64(
  // CIR: cir.call @vcombine_u32(

  // LLVM-SAME: <2 x i32> {{.*}}[[R:%.*]], <2 x i64> {{.*}}[[A:%.*]], <2 x i64> {{.*}}[[B:%.*]])
  // LLVM: [[TMP:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> [[A]], <2 x i64> [[B]])
  // LLVM: [[RES:%.*]] = shufflevector <2 x i32> [[R]], <2 x i32> [[TMP]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: ret <4 x i32> [[RES]]
  return vraddhn_high_u64(r, a, b);
}
