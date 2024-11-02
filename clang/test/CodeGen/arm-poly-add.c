// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv8.2a-arm-none-eabi \
// RUN:   -target-feature +neon \
// RUN:   -mfloat-abi hard \
// RUN: -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg \
// RUN:  | FileCheck %s

#include <arm_neon.h>

// CHECK-LABEL: @test_vadd_p8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = xor <8 x i8> [[A:%.*]], [[B:%.*]]
// CHECK-NEXT:    ret <8 x i8> [[TMP0]]
//
poly8x8_t test_vadd_p8(poly8x8_t a, poly8x8_t b) {
  return vadd_p8 (a, b);
}

// CHECK-LABEL: @test_vadd_p16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[A:%.*]] to <8 x i8>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <4 x i16> [[B:%.*]] to <8 x i8>
// CHECK-NEXT:    [[TMP2:%.*]] = xor <8 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x i16>
// CHECK-NEXT:    ret <4 x i16> [[TMP3]]
//
poly16x4_t test_vadd_p16(poly16x4_t a, poly16x4_t b) {
  return vadd_p16 (a, b);
}

// CHECK-LABEL: @test_vadd_p64(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <1 x i64> [[A:%.*]] to <8 x i8>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <1 x i64> [[B:%.*]] to <8 x i8>
// CHECK-NEXT:    [[TMP2:%.*]] = xor <8 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x i64>
// CHECK-NEXT:    ret <1 x i64> [[TMP3]]
//
poly64x1_t test_vadd_p64(poly64x1_t a, poly64x1_t b) {
  return vadd_p64(a, b);
}

// CHECK-LABEL: @test_vaddq_p8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = xor <16 x i8> [[A:%.*]], [[B:%.*]]
// CHECK-NEXT:    ret <16 x i8> [[TMP0]]
//
poly8x16_t test_vaddq_p8(poly8x16_t a, poly8x16_t b){
  return vaddq_p8(a, b);
}

// CHECK-LABEL: @test_vaddq_p16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i16> [[A:%.*]] to <16 x i8>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <8 x i16> [[B:%.*]] to <16 x i8>
// CHECK-NEXT:    [[TMP2:%.*]] = xor <16 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x i16>
// CHECK-NEXT:    ret <8 x i16> [[TMP3]]
//
poly16x8_t test_vaddq_p16(poly16x8_t a, poly16x8_t b){
  return vaddq_p16(a, b);
}

// CHECK-LABEL: @test_vaddq_p64(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x i64> [[A:%.*]] to <16 x i8>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <2 x i64> [[B:%.*]] to <16 x i8>
// CHECK-NEXT:    [[TMP2:%.*]] = xor <16 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x i64>
// CHECK-NEXT:    ret <2 x i64> [[TMP3]]
//
poly64x2_t test_vaddq_p64(poly64x2_t a, poly64x2_t b){
  return vaddq_p64(a, b);
}

// TODO: poly128_t not implemented on aarch32
// CHCK-LABEL: @test_vaddq_p128(
// CHCK-NEXT:  entry:
// CHCK-NEXT:    [[TMP0:%.*]] = bitcast i128 [[A:%.*]] to <16 x i8>
// CHCK-NEXT:    [[TMP1:%.*]] = bitcast i128 [[B:%.*]] to <16 x i8>
// CHCK-NEXT:    [[TMP2:%.*]] = xor <16 x i8> [[TMP0]], [[TMP1]]
// CHCK-NEXT:    [[TMP3:%.*]] = bitcast <16 x i8> [[TMP2]] to i128
// CHCK-NEXT:    ret i128 [[TMP3]]
//
//poly128_t test_vaddq_p128 (poly128_t a, poly128_t b){
//  return vaddq_p128(a, b);
