// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefix=CIR %}

#include <arm_neon.h>

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

