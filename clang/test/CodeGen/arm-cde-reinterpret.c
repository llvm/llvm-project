// RUN: %clang_cc1 -triple thumbv8.1m.main-none-none-eabi \
// RUN:   -target-feature +cdecp0 -target-feature +mve.fp \
// RUN:   -mfloat-abi hard -O0 -disable-O0-optnone \
// RUN:   -S -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s --check-prefixes=CHECK,CHECK-LE
// RUN: %clang_cc1 -triple thumbebv8.1m.main-arm-none-eabi \
// RUN:   -target-feature +cdecp0 -target-feature +mve.fp \
// RUN:   -mfloat-abi hard -O0 -disable-O0-optnone \
// RUN:   -S -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s --check-prefixes=CHECK,CHECK-BE

#include <arm_cde.h>

// CHECK-LABEL: @test_s8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret <16 x i8> [[X:%.*]]
//
int8x16_t test_s8(uint8x16_t x) {
  return __arm_vreinterpretq_s8_u8(x);
}

// CHECK-LABEL: @test_u16(
// CHECK-NEXT:  entry:
// CHECK-LE-NEXT: [[TMP0:%.*]] = bitcast <16 x i8> [[X:%.*]] to <8 x i16>
// CHECK-BE-NEXT: [[TMP0:%.*]] = call <8 x i16> @llvm.arm.mve.vreinterpretq.v8i16.v16i8(<16 x i8> [[X:%.*]])
// CHECK-NEXT:    ret <8 x i16> [[TMP0]]
//
uint16x8_t test_u16(uint8x16_t x) {
  return __arm_vreinterpretq_u16_u8(x);
}

// CHECK-LABEL: @test_s32(
// CHECK-NEXT:  entry:
// CHECK-LE-NEXT: [[TMP0:%.*]] = bitcast <16 x i8> [[X:%.*]] to <4 x i32>
// CHECK-BE-NEXT: [[TMP0:%.*]] = call <4 x i32> @llvm.arm.mve.vreinterpretq.v4i32.v16i8(<16 x i8> [[X:%.*]])
// CHECK-NEXT:    ret <4 x i32> [[TMP0]]
//
int32x4_t test_s32(uint8x16_t x) {
  return __arm_vreinterpretq_s32_u8(x);
}

// CHECK-LABEL: @test_u32(
// CHECK-NEXT:  entry:
// CHECK-LE-NEXT: [[TMP0:%.*]] = bitcast <16 x i8> [[X:%.*]] to <4 x i32>
// CHECK-BE-NEXT: [[TMP0:%.*]] = call <4 x i32> @llvm.arm.mve.vreinterpretq.v4i32.v16i8(<16 x i8> [[X:%.*]])
// CHECK-NEXT:    ret <4 x i32> [[TMP0]]
//
uint32x4_t test_u32(uint8x16_t x) {
  return __arm_vreinterpretq_u32_u8(x);
}

// CHECK-LABEL: @test_s64(
// CHECK-NEXT:  entry:
// CHECK-LE-NEXT: [[TMP0:%.*]] = bitcast <16 x i8> [[X:%.*]] to <2 x i64>
// CHECK-BE-NEXT: [[TMP0:%.*]] = call <2 x i64> @llvm.arm.mve.vreinterpretq.v2i64.v16i8(<16 x i8> [[X:%.*]])
// CHECK-NEXT:    ret <2 x i64> [[TMP0]]
//
int64x2_t test_s64(uint8x16_t x) {
  return __arm_vreinterpretq_s64_u8(x);
}

// CHECK-LABEL: @test_u64(
// CHECK-NEXT:  entry:
// CHECK-LE-NEXT: [[TMP0:%.*]] = bitcast <16 x i8> [[X:%.*]] to <2 x i64>
// CHECK-BE-NEXT: [[TMP0:%.*]] = call <2 x i64> @llvm.arm.mve.vreinterpretq.v2i64.v16i8(<16 x i8> [[X:%.*]])
// CHECK-NEXT:    ret <2 x i64> [[TMP0]]
//
uint64x2_t test_u64(uint8x16_t x) {
  return __arm_vreinterpretq_u64_u8(x);
}

// CHECK-LABEL: @test_f16(
// CHECK-NEXT:  entry:
// CHECK-LE-NEXT: [[TMP0:%.*]] = bitcast <16 x i8> [[X:%.*]] to <8 x half>
// CHECK-BE-NEXT: [[TMP0:%.*]] = call <8 x half> @llvm.arm.mve.vreinterpretq.v8f16.v16i8(<16 x i8> [[X:%.*]])
// CHECK-NEXT:    ret <8 x half> [[TMP0]]
//
float16x8_t test_f16(uint8x16_t x) {
  return __arm_vreinterpretq_f16_u8(x);
}

// CHECK-LABEL: @test_f32(
// CHECK-NEXT:  entry:
// CHECK-LE-NEXT: [[TMP0:%.*]] = bitcast <16 x i8> [[X:%.*]] to <4 x float>
// CHECK-BE-NEXT: [[TMP0:%.*]] = call <4 x float> @llvm.arm.mve.vreinterpretq.v4f32.v16i8(<16 x i8> [[X:%.*]])
// CHECK-NEXT:    ret <4 x float> [[TMP0]]
//
float32x4_t test_f32(uint8x16_t x) {
  return __arm_vreinterpretq_f32_u8(x);
}
