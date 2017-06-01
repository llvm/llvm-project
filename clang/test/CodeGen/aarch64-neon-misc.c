// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -fallow-half-arguments-and-returns -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

// CHECK-LABEL: @test_vceqz_s8(
// CHECK:   [[TMP0:%.*]] = icmp eq <8 x i8> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VCEQZ_I]]
uint8x8_t test_vceqz_s8(int8x8_t a) {
  return vceqz_s8(a);
}

// CHECK-LABEL: @test_vceqz_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <4 x i16> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VCEQZ_I]]
uint16x4_t test_vceqz_s16(int16x4_t a) {
  return vceqz_s16(a);
}

// CHECK-LABEL: @test_vceqz_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <2 x i32> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCEQZ_I]]
uint32x2_t test_vceqz_s32(int32x2_t a) {
  return vceqz_s32(a);
}

// CHECK-LABEL: @test_vceqz_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <1 x i64> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCEQZ_I]]
uint64x1_t test_vceqz_s64(int64x1_t a) {
  return vceqz_s64(a);
}

// CHECK-LABEL: @test_vceqz_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <1 x i64> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCEQZ_I]]
uint64x1_t test_vceqz_u64(uint64x1_t a) {
  return vceqz_u64(a);
}

// CHECK-LABEL: @test_vceqz_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <1 x i64> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCEQZ_I]]
uint64x1_t test_vceqz_p64(poly64x1_t a) {
  return vceqz_p64(a);
}

// CHECK-LABEL: @test_vceqzq_s8(
// CHECK:   [[TMP0:%.*]] = icmp eq <16 x i8> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VCEQZ_I]]
uint8x16_t test_vceqzq_s8(int8x16_t a) {
  return vceqzq_s8(a);
}

// CHECK-LABEL: @test_vceqzq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <8 x i16> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VCEQZ_I]]
uint16x8_t test_vceqzq_s16(int16x8_t a) {
  return vceqzq_s16(a);
}

// CHECK-LABEL: @test_vceqzq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <4 x i32> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCEQZ_I]]
uint32x4_t test_vceqzq_s32(int32x4_t a) {
  return vceqzq_s32(a);
}

// CHECK-LABEL: @test_vceqzq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <2 x i64> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCEQZ_I]]
uint64x2_t test_vceqzq_s64(int64x2_t a) {
  return vceqzq_s64(a);
}

// CHECK-LABEL: @test_vceqz_u8(
// CHECK:   [[TMP0:%.*]] = icmp eq <8 x i8> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VCEQZ_I]]
uint8x8_t test_vceqz_u8(uint8x8_t a) {
  return vceqz_u8(a);
}

// CHECK-LABEL: @test_vceqz_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <4 x i16> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VCEQZ_I]]
uint16x4_t test_vceqz_u16(uint16x4_t a) {
  return vceqz_u16(a);
}

// CHECK-LABEL: @test_vceqz_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <2 x i32> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCEQZ_I]]
uint32x2_t test_vceqz_u32(uint32x2_t a) {
  return vceqz_u32(a);
}

// CHECK-LABEL: @test_vceqzq_u8(
// CHECK:   [[TMP0:%.*]] = icmp eq <16 x i8> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VCEQZ_I]]
uint8x16_t test_vceqzq_u8(uint8x16_t a) {
  return vceqzq_u8(a);
}

// CHECK-LABEL: @test_vceqzq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <8 x i16> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VCEQZ_I]]
uint16x8_t test_vceqzq_u16(uint16x8_t a) {
  return vceqzq_u16(a);
}

// CHECK-LABEL: @test_vceqzq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <4 x i32> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCEQZ_I]]
uint32x4_t test_vceqzq_u32(uint32x4_t a) {
  return vceqzq_u32(a);
}

// CHECK-LABEL: @test_vceqzq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <2 x i64> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCEQZ_I]]
uint64x2_t test_vceqzq_u64(uint64x2_t a) {
  return vceqzq_u64(a);
}

// CHECK-LABEL: @test_vceqz_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp oeq <2 x float> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCEQZ_I]]
uint32x2_t test_vceqz_f32(float32x2_t a) {
  return vceqz_f32(a);
}

// CHECK-LABEL: @test_vceqz_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp oeq <1 x double> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCEQZ_I]]
uint64x1_t test_vceqz_f64(float64x1_t a) {
  return vceqz_f64(a);
}

// CHECK-LABEL: @test_vceqzq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp oeq <4 x float> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCEQZ_I]]
uint32x4_t test_vceqzq_f32(float32x4_t a) {
  return vceqzq_f32(a);
}

// CHECK-LABEL: @test_vceqz_p8(
// CHECK:   [[TMP0:%.*]] = icmp eq <8 x i8> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VCEQZ_I]]
uint8x8_t test_vceqz_p8(poly8x8_t a) {
  return vceqz_p8(a);
}

// CHECK-LABEL: @test_vceqzq_p8(
// CHECK:   [[TMP0:%.*]] = icmp eq <16 x i8> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VCEQZ_I]]
uint8x16_t test_vceqzq_p8(poly8x16_t a) {
  return vceqzq_p8(a);
}

// CHECK-LABEL: @test_vceqz_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <4 x i16> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VCEQZ_I]]
uint16x4_t test_vceqz_p16(poly16x4_t a) {
  return vceqz_p16(a);
}

// CHECK-LABEL: @test_vceqzq_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <8 x i16> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VCEQZ_I]]
uint16x8_t test_vceqzq_p16(poly16x8_t a) {
  return vceqzq_p16(a);
}

// CHECK-LABEL: @test_vceqzq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp oeq <2 x double> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCEQZ_I]]
uint64x2_t test_vceqzq_f64(float64x2_t a) {
  return vceqzq_f64(a);
}

// CHECK-LABEL: @test_vceqzq_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp eq <2 x i64> %a, zeroinitializer
// CHECK:   [[VCEQZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCEQZ_I]]
uint64x2_t test_vceqzq_p64(poly64x2_t a) {
  return vceqzq_p64(a);
}

// CHECK-LABEL: @test_vcgez_s8(
// CHECK:   [[TMP0:%.*]] = icmp sge <8 x i8> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VCGEZ_I]]
uint8x8_t test_vcgez_s8(int8x8_t a) {
  return vcgez_s8(a);
}

// CHECK-LABEL: @test_vcgez_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sge <4 x i16> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VCGEZ_I]]
uint16x4_t test_vcgez_s16(int16x4_t a) {
  return vcgez_s16(a);
}

// CHECK-LABEL: @test_vcgez_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sge <2 x i32> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCGEZ_I]]
uint32x2_t test_vcgez_s32(int32x2_t a) {
  return vcgez_s32(a);
}

// CHECK-LABEL: @test_vcgez_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sge <1 x i64> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCGEZ_I]]
uint64x1_t test_vcgez_s64(int64x1_t a) {
  return vcgez_s64(a);
}

// CHECK-LABEL: @test_vcgezq_s8(
// CHECK:   [[TMP0:%.*]] = icmp sge <16 x i8> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VCGEZ_I]]
uint8x16_t test_vcgezq_s8(int8x16_t a) {
  return vcgezq_s8(a);
}

// CHECK-LABEL: @test_vcgezq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sge <8 x i16> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VCGEZ_I]]
uint16x8_t test_vcgezq_s16(int16x8_t a) {
  return vcgezq_s16(a);
}

// CHECK-LABEL: @test_vcgezq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sge <4 x i32> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCGEZ_I]]
uint32x4_t test_vcgezq_s32(int32x4_t a) {
  return vcgezq_s32(a);
}

// CHECK-LABEL: @test_vcgezq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sge <2 x i64> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCGEZ_I]]
uint64x2_t test_vcgezq_s64(int64x2_t a) {
  return vcgezq_s64(a);
}

// CHECK-LABEL: @test_vcgez_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp oge <2 x float> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCGEZ_I]]
uint32x2_t test_vcgez_f32(float32x2_t a) {
  return vcgez_f32(a);
}

// CHECK-LABEL: @test_vcgez_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp oge <1 x double> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCGEZ_I]]
uint64x1_t test_vcgez_f64(float64x1_t a) {
  return vcgez_f64(a);
}

// CHECK-LABEL: @test_vcgezq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp oge <4 x float> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCGEZ_I]]
uint32x4_t test_vcgezq_f32(float32x4_t a) {
  return vcgezq_f32(a);
}

// CHECK-LABEL: @test_vcgezq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp oge <2 x double> %a, zeroinitializer
// CHECK:   [[VCGEZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCGEZ_I]]
uint64x2_t test_vcgezq_f64(float64x2_t a) {
  return vcgezq_f64(a);
}

// CHECK-LABEL: @test_vclez_s8(
// CHECK:   [[TMP0:%.*]] = icmp sle <8 x i8> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VCLEZ_I]]
uint8x8_t test_vclez_s8(int8x8_t a) {
  return vclez_s8(a);
}

// CHECK-LABEL: @test_vclez_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sle <4 x i16> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VCLEZ_I]]
uint16x4_t test_vclez_s16(int16x4_t a) {
  return vclez_s16(a);
}

// CHECK-LABEL: @test_vclez_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sle <2 x i32> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCLEZ_I]]
uint32x2_t test_vclez_s32(int32x2_t a) {
  return vclez_s32(a);
}

// CHECK-LABEL: @test_vclez_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sle <1 x i64> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCLEZ_I]]
uint64x1_t test_vclez_s64(int64x1_t a) {
  return vclez_s64(a);
}

// CHECK-LABEL: @test_vclezq_s8(
// CHECK:   [[TMP0:%.*]] = icmp sle <16 x i8> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VCLEZ_I]]
uint8x16_t test_vclezq_s8(int8x16_t a) {
  return vclezq_s8(a);
}

// CHECK-LABEL: @test_vclezq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sle <8 x i16> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VCLEZ_I]]
uint16x8_t test_vclezq_s16(int16x8_t a) {
  return vclezq_s16(a);
}

// CHECK-LABEL: @test_vclezq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sle <4 x i32> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCLEZ_I]]
uint32x4_t test_vclezq_s32(int32x4_t a) {
  return vclezq_s32(a);
}

// CHECK-LABEL: @test_vclezq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sle <2 x i64> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCLEZ_I]]
uint64x2_t test_vclezq_s64(int64x2_t a) {
  return vclezq_s64(a);
}

// CHECK-LABEL: @test_vclez_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp ole <2 x float> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCLEZ_I]]
uint32x2_t test_vclez_f32(float32x2_t a) {
  return vclez_f32(a);
}

// CHECK-LABEL: @test_vclez_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp ole <1 x double> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCLEZ_I]]
uint64x1_t test_vclez_f64(float64x1_t a) {
  return vclez_f64(a);
}

// CHECK-LABEL: @test_vclezq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp ole <4 x float> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCLEZ_I]]
uint32x4_t test_vclezq_f32(float32x4_t a) {
  return vclezq_f32(a);
}

// CHECK-LABEL: @test_vclezq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp ole <2 x double> %a, zeroinitializer
// CHECK:   [[VCLEZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCLEZ_I]]
uint64x2_t test_vclezq_f64(float64x2_t a) {
  return vclezq_f64(a);
}

// CHECK-LABEL: @test_vcgtz_s8(
// CHECK:   [[TMP0:%.*]] = icmp sgt <8 x i8> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VCGTZ_I]]
uint8x8_t test_vcgtz_s8(int8x8_t a) {
  return vcgtz_s8(a);
}

// CHECK-LABEL: @test_vcgtz_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sgt <4 x i16> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VCGTZ_I]]
uint16x4_t test_vcgtz_s16(int16x4_t a) {
  return vcgtz_s16(a);
}

// CHECK-LABEL: @test_vcgtz_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sgt <2 x i32> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCGTZ_I]]
uint32x2_t test_vcgtz_s32(int32x2_t a) {
  return vcgtz_s32(a);
}

// CHECK-LABEL: @test_vcgtz_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sgt <1 x i64> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCGTZ_I]]
uint64x1_t test_vcgtz_s64(int64x1_t a) {
  return vcgtz_s64(a);
}

// CHECK-LABEL: @test_vcgtzq_s8(
// CHECK:   [[TMP0:%.*]] = icmp sgt <16 x i8> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VCGTZ_I]]
uint8x16_t test_vcgtzq_s8(int8x16_t a) {
  return vcgtzq_s8(a);
}

// CHECK-LABEL: @test_vcgtzq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sgt <8 x i16> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VCGTZ_I]]
uint16x8_t test_vcgtzq_s16(int16x8_t a) {
  return vcgtzq_s16(a);
}

// CHECK-LABEL: @test_vcgtzq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sgt <4 x i32> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCGTZ_I]]
uint32x4_t test_vcgtzq_s32(int32x4_t a) {
  return vcgtzq_s32(a);
}

// CHECK-LABEL: @test_vcgtzq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp sgt <2 x i64> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCGTZ_I]]
uint64x2_t test_vcgtzq_s64(int64x2_t a) {
  return vcgtzq_s64(a);
}

// CHECK-LABEL: @test_vcgtz_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp ogt <2 x float> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCGTZ_I]]
uint32x2_t test_vcgtz_f32(float32x2_t a) {
  return vcgtz_f32(a);
}

// CHECK-LABEL: @test_vcgtz_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp ogt <1 x double> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCGTZ_I]]
uint64x1_t test_vcgtz_f64(float64x1_t a) {
  return vcgtz_f64(a);
}

// CHECK-LABEL: @test_vcgtzq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp ogt <4 x float> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCGTZ_I]]
uint32x4_t test_vcgtzq_f32(float32x4_t a) {
  return vcgtzq_f32(a);
}

// CHECK-LABEL: @test_vcgtzq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp ogt <2 x double> %a, zeroinitializer
// CHECK:   [[VCGTZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCGTZ_I]]
uint64x2_t test_vcgtzq_f64(float64x2_t a) {
  return vcgtzq_f64(a);
}

// CHECK-LABEL: @test_vcltz_s8(
// CHECK:   [[TMP0:%.*]] = icmp slt <8 x i8> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VCLTZ_I]]
uint8x8_t test_vcltz_s8(int8x8_t a) {
  return vcltz_s8(a);
}

// CHECK-LABEL: @test_vcltz_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp slt <4 x i16> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VCLTZ_I]]
uint16x4_t test_vcltz_s16(int16x4_t a) {
  return vcltz_s16(a);
}

// CHECK-LABEL: @test_vcltz_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp slt <2 x i32> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCLTZ_I]]
uint32x2_t test_vcltz_s32(int32x2_t a) {
  return vcltz_s32(a);
}

// CHECK-LABEL: @test_vcltz_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = icmp slt <1 x i64> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCLTZ_I]]
uint64x1_t test_vcltz_s64(int64x1_t a) {
  return vcltz_s64(a);
}

// CHECK-LABEL: @test_vcltzq_s8(
// CHECK:   [[TMP0:%.*]] = icmp slt <16 x i8> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <16 x i1> [[TMP0]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VCLTZ_I]]
uint8x16_t test_vcltzq_s8(int8x16_t a) {
  return vcltzq_s8(a);
}

// CHECK-LABEL: @test_vcltzq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp slt <8 x i16> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VCLTZ_I]]
uint16x8_t test_vcltzq_s16(int16x8_t a) {
  return vcltzq_s16(a);
}

// CHECK-LABEL: @test_vcltzq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp slt <4 x i32> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCLTZ_I]]
uint32x4_t test_vcltzq_s32(int32x4_t a) {
  return vcltzq_s32(a);
}

// CHECK-LABEL: @test_vcltzq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = icmp slt <2 x i64> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCLTZ_I]]
uint64x2_t test_vcltzq_s64(int64x2_t a) {
  return vcltzq_s64(a);
}

// CHECK-LABEL: @test_vcltz_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp olt <2 x float> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VCLTZ_I]]
uint32x2_t test_vcltz_f32(float32x2_t a) {
  return vcltz_f32(a);
}

// CHECK-LABEL: @test_vcltz_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp olt <1 x double> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <1 x i1> [[TMP1]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VCLTZ_I]]
uint64x1_t test_vcltz_f64(float64x1_t a) {
  return vcltz_f64(a);
}

// CHECK-LABEL: @test_vcltzq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp olt <4 x float> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VCLTZ_I]]
uint32x4_t test_vcltzq_f32(float32x4_t a) {
  return vcltzq_f32(a);
}

// CHECK-LABEL: @test_vcltzq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fcmp olt <2 x double> %a, zeroinitializer
// CHECK:   [[VCLTZ_I:%.*]] = sext <2 x i1> [[TMP1]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VCLTZ_I]]
uint64x2_t test_vcltzq_f64(float64x2_t a) {
  return vcltzq_f64(a);
}

// CHECK-LABEL: @test_vrev16_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vrev16_s8(int8x8_t a) {
  return vrev16_s8(a);
}

// CHECK-LABEL: @test_vrev16_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vrev16_u8(uint8x8_t a) {
  return vrev16_u8(a);
}

// CHECK-LABEL: @test_vrev16_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vrev16_p8(poly8x8_t a) {
  return vrev16_p8(a);
}

// CHECK-LABEL: @test_vrev16q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vrev16q_s8(int8x16_t a) {
  return vrev16q_s8(a);
}

// CHECK-LABEL: @test_vrev16q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vrev16q_u8(uint8x16_t a) {
  return vrev16q_u8(a);
}

// CHECK-LABEL: @test_vrev16q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vrev16q_p8(poly8x16_t a) {
  return vrev16q_p8(a);
}

// CHECK-LABEL: @test_vrev32_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vrev32_s8(int8x8_t a) {
  return vrev32_s8(a);
}

// CHECK-LABEL: @test_vrev32_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %a, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vrev32_s16(int16x4_t a) {
  return vrev32_s16(a);
}

// CHECK-LABEL: @test_vrev32_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vrev32_u8(uint8x8_t a) {
  return vrev32_u8(a);
}

// CHECK-LABEL: @test_vrev32_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %a, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vrev32_u16(uint16x4_t a) {
  return vrev32_u16(a);
}

// CHECK-LABEL: @test_vrev32_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vrev32_p8(poly8x8_t a) {
  return vrev32_p8(a);
}

// CHECK-LABEL: @test_vrev32_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %a, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vrev32_p16(poly16x4_t a) {
  return vrev32_p16(a);
}

// CHECK-LABEL: @test_vrev32q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vrev32q_s8(int8x16_t a) {
  return vrev32q_s8(a);
}

// CHECK-LABEL: @test_vrev32q_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vrev32q_s16(int16x8_t a) {
  return vrev32q_s16(a);
}

// CHECK-LABEL: @test_vrev32q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vrev32q_u8(uint8x16_t a) {
  return vrev32q_u8(a);
}

// CHECK-LABEL: @test_vrev32q_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vrev32q_u16(uint16x8_t a) {
  return vrev32q_u16(a);
}

// CHECK-LABEL: @test_vrev32q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vrev32q_p8(poly8x16_t a) {
  return vrev32q_p8(a);
}

// CHECK-LABEL: @test_vrev32q_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vrev32q_p16(poly16x8_t a) {
  return vrev32q_p16(a);
}

// CHECK-LABEL: @test_vrev64_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
int8x8_t test_vrev64_s8(int8x8_t a) {
  return vrev64_s8(a);
}

// CHECK-LABEL: @test_vrev64_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %a, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
int16x4_t test_vrev64_s16(int16x4_t a) {
  return vrev64_s16(a);
}

// CHECK-LABEL: @test_vrev64_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %a, <2 x i32> <i32 1, i32 0>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
int32x2_t test_vrev64_s32(int32x2_t a) {
  return vrev64_s32(a);
}

// CHECK-LABEL: @test_vrev64_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
uint8x8_t test_vrev64_u8(uint8x8_t a) {
  return vrev64_u8(a);
}

// CHECK-LABEL: @test_vrev64_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %a, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
uint16x4_t test_vrev64_u16(uint16x4_t a) {
  return vrev64_u16(a);
}

// CHECK-LABEL: @test_vrev64_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> %a, <2 x i32> <i32 1, i32 0>
// CHECK:   ret <2 x i32> [[SHUFFLE_I]]
uint32x2_t test_vrev64_u32(uint32x2_t a) {
  return vrev64_u32(a);
}

// CHECK-LABEL: @test_vrev64_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %a, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
// CHECK:   ret <8 x i8> [[SHUFFLE_I]]
poly8x8_t test_vrev64_p8(poly8x8_t a) {
  return vrev64_p8(a);
}

// CHECK-LABEL: @test_vrev64_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> %a, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// CHECK:   ret <4 x i16> [[SHUFFLE_I]]
poly16x4_t test_vrev64_p16(poly16x4_t a) {
  return vrev64_p16(a);
}

// CHECK-LABEL: @test_vrev64_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x float> %a, <2 x float> %a, <2 x i32> <i32 1, i32 0>
// CHECK:   ret <2 x float> [[SHUFFLE_I]]
float32x2_t test_vrev64_f32(float32x2_t a) {
  return vrev64_f32(a);
}

// CHECK-LABEL: @test_vrev64q_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vrev64q_s8(int8x16_t a) {
  return vrev64q_s8(a);
}

// CHECK-LABEL: @test_vrev64q_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vrev64q_s16(int16x8_t a) {
  return vrev64q_s16(a);
}

// CHECK-LABEL: @test_vrev64q_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vrev64q_s32(int32x4_t a) {
  return vrev64q_s32(a);
}

// CHECK-LABEL: @test_vrev64q_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vrev64q_u8(uint8x16_t a) {
  return vrev64q_u8(a);
}

// CHECK-LABEL: @test_vrev64q_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vrev64q_u16(uint16x8_t a) {
  return vrev64q_u16(a);
}

// CHECK-LABEL: @test_vrev64q_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vrev64q_u32(uint32x4_t a) {
  return vrev64q_u32(a);
}

// CHECK-LABEL: @test_vrev64q_p8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
poly8x16_t test_vrev64q_p8(poly8x16_t a) {
  return vrev64q_p8(a);
}

// CHECK-LABEL: @test_vrev64q_p16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
poly16x8_t test_vrev64q_p16(poly16x8_t a) {
  return vrev64q_p16(a);
}

// CHECK-LABEL: @test_vrev64q_f32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:   ret <4 x float> [[SHUFFLE_I]]
float32x4_t test_vrev64q_f32(float32x4_t a) {
  return vrev64q_f32(a);
}

// CHECK-LABEL: @test_vpaddl_s8(
// CHECK:   [[VPADDL_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.saddlp.v4i16.v8i8(<8 x i8> %a) #2
// CHECK:   ret <4 x i16> [[VPADDL_I]]
int16x4_t test_vpaddl_s8(int8x8_t a) {
  return vpaddl_s8(a);
}

// CHECK-LABEL: @test_vpaddl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VPADDL1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.saddlp.v2i32.v4i16(<4 x i16> %a) #2
// CHECK:   ret <2 x i32> [[VPADDL1_I]]
int32x2_t test_vpaddl_s16(int16x4_t a) {
  return vpaddl_s16(a);
}

// CHECK-LABEL: @test_vpaddl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VPADDL1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.saddlp.v1i64.v2i32(<2 x i32> %a) #2
// CHECK:   ret <1 x i64> [[VPADDL1_I]]
int64x1_t test_vpaddl_s32(int32x2_t a) {
  return vpaddl_s32(a);
}

// CHECK-LABEL: @test_vpaddl_u8(
// CHECK:   [[VPADDL_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uaddlp.v4i16.v8i8(<8 x i8> %a) #2
// CHECK:   ret <4 x i16> [[VPADDL_I]]
uint16x4_t test_vpaddl_u8(uint8x8_t a) {
  return vpaddl_u8(a);
}

// CHECK-LABEL: @test_vpaddl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VPADDL1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uaddlp.v2i32.v4i16(<4 x i16> %a) #2
// CHECK:   ret <2 x i32> [[VPADDL1_I]]
uint32x2_t test_vpaddl_u16(uint16x4_t a) {
  return vpaddl_u16(a);
}

// CHECK-LABEL: @test_vpaddl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VPADDL1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uaddlp.v1i64.v2i32(<2 x i32> %a) #2
// CHECK:   ret <1 x i64> [[VPADDL1_I]]
uint64x1_t test_vpaddl_u32(uint32x2_t a) {
  return vpaddl_u32(a);
}

// CHECK-LABEL: @test_vpaddlq_s8(
// CHECK:   [[VPADDL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.saddlp.v8i16.v16i8(<16 x i8> %a) #2
// CHECK:   ret <8 x i16> [[VPADDL_I]]
int16x8_t test_vpaddlq_s8(int8x16_t a) {
  return vpaddlq_s8(a);
}

// CHECK-LABEL: @test_vpaddlq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VPADDL1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.saddlp.v4i32.v8i16(<8 x i16> %a) #2
// CHECK:   ret <4 x i32> [[VPADDL1_I]]
int32x4_t test_vpaddlq_s16(int16x8_t a) {
  return vpaddlq_s16(a);
}

// CHECK-LABEL: @test_vpaddlq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VPADDL1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.saddlp.v2i64.v4i32(<4 x i32> %a) #2
// CHECK:   ret <2 x i64> [[VPADDL1_I]]
int64x2_t test_vpaddlq_s32(int32x4_t a) {
  return vpaddlq_s32(a);
}

// CHECK-LABEL: @test_vpaddlq_u8(
// CHECK:   [[VPADDL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uaddlp.v8i16.v16i8(<16 x i8> %a) #2
// CHECK:   ret <8 x i16> [[VPADDL_I]]
uint16x8_t test_vpaddlq_u8(uint8x16_t a) {
  return vpaddlq_u8(a);
}

// CHECK-LABEL: @test_vpaddlq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VPADDL1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uaddlp.v4i32.v8i16(<8 x i16> %a) #2
// CHECK:   ret <4 x i32> [[VPADDL1_I]]
uint32x4_t test_vpaddlq_u16(uint16x8_t a) {
  return vpaddlq_u16(a);
}

// CHECK-LABEL: @test_vpaddlq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VPADDL1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uaddlp.v2i64.v4i32(<4 x i32> %a) #2
// CHECK:   ret <2 x i64> [[VPADDL1_I]]
uint64x2_t test_vpaddlq_u32(uint32x4_t a) {
  return vpaddlq_u32(a);
}

// CHECK-LABEL: @test_vpadal_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VPADAL_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.saddlp.v4i16.v8i8(<8 x i8> %b) #2
// CHECK:   [[TMP1:%.*]] = add <4 x i16> [[VPADAL_I]], %a
// CHECK:   ret <4 x i16> [[TMP1]]
int16x4_t test_vpadal_s8(int16x4_t a, int8x8_t b) {
  return vpadal_s8(a, b);
}

// CHECK-LABEL: @test_vpadal_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VPADAL1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.saddlp.v2i32.v4i16(<4 x i16> %b) #2
// CHECK:   [[TMP2:%.*]] = add <2 x i32> [[VPADAL1_I]], %a
// CHECK:   ret <2 x i32> [[TMP2]]
int32x2_t test_vpadal_s16(int32x2_t a, int16x4_t b) {
  return vpadal_s16(a, b);
}

// CHECK-LABEL: @test_vpadal_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VPADAL1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.saddlp.v1i64.v2i32(<2 x i32> %b) #2
// CHECK:   [[TMP2:%.*]] = add <1 x i64> [[VPADAL1_I]], %a
// CHECK:   ret <1 x i64> [[TMP2]]
int64x1_t test_vpadal_s32(int64x1_t a, int32x2_t b) {
  return vpadal_s32(a, b);
}

// CHECK-LABEL: @test_vpadal_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VPADAL_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uaddlp.v4i16.v8i8(<8 x i8> %b) #2
// CHECK:   [[TMP1:%.*]] = add <4 x i16> [[VPADAL_I]], %a
// CHECK:   ret <4 x i16> [[TMP1]]
uint16x4_t test_vpadal_u8(uint16x4_t a, uint8x8_t b) {
  return vpadal_u8(a, b);
}

// CHECK-LABEL: @test_vpadal_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VPADAL1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uaddlp.v2i32.v4i16(<4 x i16> %b) #2
// CHECK:   [[TMP2:%.*]] = add <2 x i32> [[VPADAL1_I]], %a
// CHECK:   ret <2 x i32> [[TMP2]]
uint32x2_t test_vpadal_u16(uint32x2_t a, uint16x4_t b) {
  return vpadal_u16(a, b);
}

// CHECK-LABEL: @test_vpadal_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VPADAL1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uaddlp.v1i64.v2i32(<2 x i32> %b) #2
// CHECK:   [[TMP2:%.*]] = add <1 x i64> [[VPADAL1_I]], %a
// CHECK:   ret <1 x i64> [[TMP2]]
uint64x1_t test_vpadal_u32(uint64x1_t a, uint32x2_t b) {
  return vpadal_u32(a, b);
}

// CHECK-LABEL: @test_vpadalq_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VPADAL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.saddlp.v8i16.v16i8(<16 x i8> %b) #2
// CHECK:   [[TMP1:%.*]] = add <8 x i16> [[VPADAL_I]], %a
// CHECK:   ret <8 x i16> [[TMP1]]
int16x8_t test_vpadalq_s8(int16x8_t a, int8x16_t b) {
  return vpadalq_s8(a, b);
}

// CHECK-LABEL: @test_vpadalq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VPADAL1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.saddlp.v4i32.v8i16(<8 x i16> %b) #2
// CHECK:   [[TMP2:%.*]] = add <4 x i32> [[VPADAL1_I]], %a
// CHECK:   ret <4 x i32> [[TMP2]]
int32x4_t test_vpadalq_s16(int32x4_t a, int16x8_t b) {
  return vpadalq_s16(a, b);
}

// CHECK-LABEL: @test_vpadalq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VPADAL1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.saddlp.v2i64.v4i32(<4 x i32> %b) #2
// CHECK:   [[TMP2:%.*]] = add <2 x i64> [[VPADAL1_I]], %a
// CHECK:   ret <2 x i64> [[TMP2]]
int64x2_t test_vpadalq_s32(int64x2_t a, int32x4_t b) {
  return vpadalq_s32(a, b);
}

// CHECK-LABEL: @test_vpadalq_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VPADAL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uaddlp.v8i16.v16i8(<16 x i8> %b) #2
// CHECK:   [[TMP1:%.*]] = add <8 x i16> [[VPADAL_I]], %a
// CHECK:   ret <8 x i16> [[TMP1]]
uint16x8_t test_vpadalq_u8(uint16x8_t a, uint8x16_t b) {
  return vpadalq_u8(a, b);
}

// CHECK-LABEL: @test_vpadalq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VPADAL1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uaddlp.v4i32.v8i16(<8 x i16> %b) #2
// CHECK:   [[TMP2:%.*]] = add <4 x i32> [[VPADAL1_I]], %a
// CHECK:   ret <4 x i32> [[TMP2]]
uint32x4_t test_vpadalq_u16(uint32x4_t a, uint16x8_t b) {
  return vpadalq_u16(a, b);
}

// CHECK-LABEL: @test_vpadalq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VPADAL1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uaddlp.v2i64.v4i32(<4 x i32> %b) #2
// CHECK:   [[TMP2:%.*]] = add <2 x i64> [[VPADAL1_I]], %a
// CHECK:   ret <2 x i64> [[TMP2]]
uint64x2_t test_vpadalq_u32(uint64x2_t a, uint32x4_t b) {
  return vpadalq_u32(a, b);
}

// CHECK-LABEL: @test_vqabs_s8(
// CHECK:   [[VQABS_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqabs.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VQABS_V_I]]
int8x8_t test_vqabs_s8(int8x8_t a) {
  return vqabs_s8(a);
}

// CHECK-LABEL: @test_vqabsq_s8(
// CHECK:   [[VQABSQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqabs.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VQABSQ_V_I]]
int8x16_t test_vqabsq_s8(int8x16_t a) {
  return vqabsq_s8(a);
}

// CHECK-LABEL: @test_vqabs_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VQABS_V1_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqabs.v4i16(<4 x i16> %a) #2
// CHECK:   [[VQABS_V2_I:%.*]] = bitcast <4 x i16> [[VQABS_V1_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQABS_V1_I]]
int16x4_t test_vqabs_s16(int16x4_t a) {
  return vqabs_s16(a);
}

// CHECK-LABEL: @test_vqabsq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQABSQ_V1_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqabs.v8i16(<8 x i16> %a) #2
// CHECK:   [[VQABSQ_V2_I:%.*]] = bitcast <8 x i16> [[VQABSQ_V1_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQABSQ_V1_I]]
int16x8_t test_vqabsq_s16(int16x8_t a) {
  return vqabsq_s16(a);
}

// CHECK-LABEL: @test_vqabs_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VQABS_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqabs.v2i32(<2 x i32> %a) #2
// CHECK:   [[VQABS_V2_I:%.*]] = bitcast <2 x i32> [[VQABS_V1_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQABS_V1_I]]
int32x2_t test_vqabs_s32(int32x2_t a) {
  return vqabs_s32(a);
}

// CHECK-LABEL: @test_vqabsq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQABSQ_V1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqabs.v4i32(<4 x i32> %a) #2
// CHECK:   [[VQABSQ_V2_I:%.*]] = bitcast <4 x i32> [[VQABSQ_V1_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQABSQ_V1_I]]
int32x4_t test_vqabsq_s32(int32x4_t a) {
  return vqabsq_s32(a);
}

// CHECK-LABEL: @test_vqabsq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQABSQ_V1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqabs.v2i64(<2 x i64> %a) #2
// CHECK:   [[VQABSQ_V2_I:%.*]] = bitcast <2 x i64> [[VQABSQ_V1_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQABSQ_V1_I]]
int64x2_t test_vqabsq_s64(int64x2_t a) {
  return vqabsq_s64(a);
}

// CHECK-LABEL: @test_vqneg_s8(
// CHECK:   [[VQNEG_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqneg.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VQNEG_V_I]]
int8x8_t test_vqneg_s8(int8x8_t a) {
  return vqneg_s8(a);
}

// CHECK-LABEL: @test_vqnegq_s8(
// CHECK:   [[VQNEGQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqneg.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VQNEGQ_V_I]]
int8x16_t test_vqnegq_s8(int8x16_t a) {
  return vqnegq_s8(a);
}

// CHECK-LABEL: @test_vqneg_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VQNEG_V1_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqneg.v4i16(<4 x i16> %a) #2
// CHECK:   [[VQNEG_V2_I:%.*]] = bitcast <4 x i16> [[VQNEG_V1_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQNEG_V1_I]]
int16x4_t test_vqneg_s16(int16x4_t a) {
  return vqneg_s16(a);
}

// CHECK-LABEL: @test_vqnegq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQNEGQ_V1_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqneg.v8i16(<8 x i16> %a) #2
// CHECK:   [[VQNEGQ_V2_I:%.*]] = bitcast <8 x i16> [[VQNEGQ_V1_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQNEGQ_V1_I]]
int16x8_t test_vqnegq_s16(int16x8_t a) {
  return vqnegq_s16(a);
}

// CHECK-LABEL: @test_vqneg_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VQNEG_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqneg.v2i32(<2 x i32> %a) #2
// CHECK:   [[VQNEG_V2_I:%.*]] = bitcast <2 x i32> [[VQNEG_V1_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQNEG_V1_I]]
int32x2_t test_vqneg_s32(int32x2_t a) {
  return vqneg_s32(a);
}

// CHECK-LABEL: @test_vqnegq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQNEGQ_V1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqneg.v4i32(<4 x i32> %a) #2
// CHECK:   [[VQNEGQ_V2_I:%.*]] = bitcast <4 x i32> [[VQNEGQ_V1_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQNEGQ_V1_I]]
int32x4_t test_vqnegq_s32(int32x4_t a) {
  return vqnegq_s32(a);
}

// CHECK-LABEL: @test_vqnegq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQNEGQ_V1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqneg.v2i64(<2 x i64> %a) #2
// CHECK:   [[VQNEGQ_V2_I:%.*]] = bitcast <2 x i64> [[VQNEGQ_V1_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQNEGQ_V1_I]]
int64x2_t test_vqnegq_s64(int64x2_t a) {
  return vqnegq_s64(a);
}

// CHECK-LABEL: @test_vneg_s8(
// CHECK:   [[SUB_I:%.*]] = sub <8 x i8> zeroinitializer, %a
// CHECK:   ret <8 x i8> [[SUB_I]]
int8x8_t test_vneg_s8(int8x8_t a) {
  return vneg_s8(a);
}

// CHECK-LABEL: @test_vnegq_s8(
// CHECK:   [[SUB_I:%.*]] = sub <16 x i8> zeroinitializer, %a
// CHECK:   ret <16 x i8> [[SUB_I]]
int8x16_t test_vnegq_s8(int8x16_t a) {
  return vnegq_s8(a);
}

// CHECK-LABEL: @test_vneg_s16(
// CHECK:   [[SUB_I:%.*]] = sub <4 x i16> zeroinitializer, %a
// CHECK:   ret <4 x i16> [[SUB_I]]
int16x4_t test_vneg_s16(int16x4_t a) {
  return vneg_s16(a);
}

// CHECK-LABEL: @test_vnegq_s16(
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> zeroinitializer, %a
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vnegq_s16(int16x8_t a) {
  return vnegq_s16(a);
}

// CHECK-LABEL: @test_vneg_s32(
// CHECK:   [[SUB_I:%.*]] = sub <2 x i32> zeroinitializer, %a
// CHECK:   ret <2 x i32> [[SUB_I]]
int32x2_t test_vneg_s32(int32x2_t a) {
  return vneg_s32(a);
}

// CHECK-LABEL: @test_vnegq_s32(
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> zeroinitializer, %a
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vnegq_s32(int32x4_t a) {
  return vnegq_s32(a);
}

// CHECK-LABEL: @test_vnegq_s64(
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> zeroinitializer, %a
// CHECK:   ret <2 x i64> [[SUB_I]]
int64x2_t test_vnegq_s64(int64x2_t a) {
  return vnegq_s64(a);
}

// CHECK-LABEL: @test_vneg_f32(
// CHECK:   [[SUB_I:%.*]] = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %a
// CHECK:   ret <2 x float> [[SUB_I]]
float32x2_t test_vneg_f32(float32x2_t a) {
  return vneg_f32(a);
}

// CHECK-LABEL: @test_vnegq_f32(
// CHECK:   [[SUB_I:%.*]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %a
// CHECK:   ret <4 x float> [[SUB_I]]
float32x4_t test_vnegq_f32(float32x4_t a) {
  return vnegq_f32(a);
}

// CHECK-LABEL: @test_vnegq_f64(
// CHECK:   [[SUB_I:%.*]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %a
// CHECK:   ret <2 x double> [[SUB_I]]
float64x2_t test_vnegq_f64(float64x2_t a) {
  return vnegq_f64(a);
}

// CHECK-LABEL: @test_vabs_s8(
// CHECK:   [[VABS_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.abs.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VABS_I]]
int8x8_t test_vabs_s8(int8x8_t a) {
  return vabs_s8(a);
}

// CHECK-LABEL: @test_vabsq_s8(
// CHECK:   [[VABS_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.abs.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VABS_I]]
int8x16_t test_vabsq_s8(int8x16_t a) {
  return vabsq_s8(a);
}

// CHECK-LABEL: @test_vabs_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.abs.v4i16(<4 x i16> %a) #2
// CHECK:   ret <4 x i16> [[VABS1_I]]
int16x4_t test_vabs_s16(int16x4_t a) {
  return vabs_s16(a);
}

// CHECK-LABEL: @test_vabsq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.abs.v8i16(<8 x i16> %a) #2
// CHECK:   ret <8 x i16> [[VABS1_I]]
int16x8_t test_vabsq_s16(int16x8_t a) {
  return vabsq_s16(a);
}

// CHECK-LABEL: @test_vabs_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.abs.v2i32(<2 x i32> %a) #2
// CHECK:   ret <2 x i32> [[VABS1_I]]
int32x2_t test_vabs_s32(int32x2_t a) {
  return vabs_s32(a);
}

// CHECK-LABEL: @test_vabsq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.abs.v4i32(<4 x i32> %a) #2
// CHECK:   ret <4 x i32> [[VABS1_I]]
int32x4_t test_vabsq_s32(int32x4_t a) {
  return vabsq_s32(a);
}

// CHECK-LABEL: @test_vabsq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.abs.v2i64(<2 x i64> %a) #2
// CHECK:   ret <2 x i64> [[VABS1_I]]
int64x2_t test_vabsq_s64(int64x2_t a) {
  return vabsq_s64(a);
}

// CHECK-LABEL: @test_vabs_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <2 x float> @llvm.fabs.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VABS1_I]]
float32x2_t test_vabs_f32(float32x2_t a) {
  return vabs_f32(a);
}

// CHECK-LABEL: @test_vabsq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <4 x float> @llvm.fabs.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VABS1_I]]
float32x4_t test_vabsq_f32(float32x4_t a) {
  return vabsq_f32(a);
}

// CHECK-LABEL: @test_vabsq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <2 x double> @llvm.fabs.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VABS1_I]]
float64x2_t test_vabsq_f64(float64x2_t a) {
  return vabsq_f64(a);
}

// CHECK-LABEL: @test_vuqadd_s8(
// CHECK:   [[VUQADD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.suqadd.v8i8(<8 x i8> %a, <8 x i8> %b) #2
// CHECK:   ret <8 x i8> [[VUQADD_I]]
int8x8_t test_vuqadd_s8(int8x8_t a, int8x8_t b) {
  return vuqadd_s8(a, b);
}

// CHECK-LABEL: @test_vuqaddq_s8(
// CHECK:   [[VUQADD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.suqadd.v16i8(<16 x i8> %a, <16 x i8> %b) #2
// CHECK:   ret <16 x i8> [[VUQADD_I]]
int8x16_t test_vuqaddq_s8(int8x16_t a, int8x16_t b) {
  return vuqaddq_s8(a, b);
}

// CHECK-LABEL: @test_vuqadd_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VUQADD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16> %a, <4 x i16> %b) #2
// CHECK:   ret <4 x i16> [[VUQADD2_I]]
int16x4_t test_vuqadd_s16(int16x4_t a, int16x4_t b) {
  return vuqadd_s16(a, b);
}

// CHECK-LABEL: @test_vuqaddq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VUQADD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.suqadd.v8i16(<8 x i16> %a, <8 x i16> %b) #2
// CHECK:   ret <8 x i16> [[VUQADD2_I]]
int16x8_t test_vuqaddq_s16(int16x8_t a, int16x8_t b) {
  return vuqaddq_s16(a, b);
}

// CHECK-LABEL: @test_vuqadd_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VUQADD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.suqadd.v2i32(<2 x i32> %a, <2 x i32> %b) #2
// CHECK:   ret <2 x i32> [[VUQADD2_I]]
int32x2_t test_vuqadd_s32(int32x2_t a, int32x2_t b) {
  return vuqadd_s32(a, b);
}

// CHECK-LABEL: @test_vuqaddq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VUQADD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.suqadd.v4i32(<4 x i32> %a, <4 x i32> %b) #2
// CHECK:   ret <4 x i32> [[VUQADD2_I]]
int32x4_t test_vuqaddq_s32(int32x4_t a, int32x4_t b) {
  return vuqaddq_s32(a, b);
}

// CHECK-LABEL: @test_vuqaddq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VUQADD2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.suqadd.v2i64(<2 x i64> %a, <2 x i64> %b) #2
// CHECK:   ret <2 x i64> [[VUQADD2_I]]
int64x2_t test_vuqaddq_s64(int64x2_t a, int64x2_t b) {
  return vuqaddq_s64(a, b);
}

// CHECK-LABEL: @test_vcls_s8(
// CHECK:   [[VCLS_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.cls.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VCLS_V_I]]
int8x8_t test_vcls_s8(int8x8_t a) {
  return vcls_s8(a);
}

// CHECK-LABEL: @test_vclsq_s8(
// CHECK:   [[VCLSQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.cls.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VCLSQ_V_I]]
int8x16_t test_vclsq_s8(int8x16_t a) {
  return vclsq_s8(a);
}

// CHECK-LABEL: @test_vcls_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VCLS_V1_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.cls.v4i16(<4 x i16> %a) #2
// CHECK:   [[VCLS_V2_I:%.*]] = bitcast <4 x i16> [[VCLS_V1_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VCLS_V1_I]]
int16x4_t test_vcls_s16(int16x4_t a) {
  return vcls_s16(a);
}

// CHECK-LABEL: @test_vclsq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VCLSQ_V1_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.cls.v8i16(<8 x i16> %a) #2
// CHECK:   [[VCLSQ_V2_I:%.*]] = bitcast <8 x i16> [[VCLSQ_V1_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VCLSQ_V1_I]]
int16x8_t test_vclsq_s16(int16x8_t a) {
  return vclsq_s16(a);
}

// CHECK-LABEL: @test_vcls_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VCLS_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.cls.v2i32(<2 x i32> %a) #2
// CHECK:   [[VCLS_V2_I:%.*]] = bitcast <2 x i32> [[VCLS_V1_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VCLS_V1_I]]
int32x2_t test_vcls_s32(int32x2_t a) {
  return vcls_s32(a);
}

// CHECK-LABEL: @test_vclsq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VCLSQ_V1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.cls.v4i32(<4 x i32> %a) #2
// CHECK:   [[VCLSQ_V2_I:%.*]] = bitcast <4 x i32> [[VCLSQ_V1_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VCLSQ_V1_I]]
int32x4_t test_vclsq_s32(int32x4_t a) {
  return vclsq_s32(a);
}

// CHECK-LABEL: @test_vclz_s8(
// CHECK:   [[VCLZ_V_I:%.*]] = call <8 x i8> @llvm.ctlz.v8i8(<8 x i8> %a, i1 false) #2
// CHECK:   ret <8 x i8> [[VCLZ_V_I]]
int8x8_t test_vclz_s8(int8x8_t a) {
  return vclz_s8(a);
}

// CHECK-LABEL: @test_vclzq_s8(
// CHECK:   [[VCLZQ_V_I:%.*]] = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 false) #2
// CHECK:   ret <16 x i8> [[VCLZQ_V_I]]
int8x16_t test_vclzq_s8(int8x16_t a) {
  return vclzq_s8(a);
}

// CHECK-LABEL: @test_vclz_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VCLZ_V1_I:%.*]] = call <4 x i16> @llvm.ctlz.v4i16(<4 x i16> %a, i1 false) #2
// CHECK:   [[VCLZ_V2_I:%.*]] = bitcast <4 x i16> [[VCLZ_V1_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VCLZ_V1_I]]
int16x4_t test_vclz_s16(int16x4_t a) {
  return vclz_s16(a);
}

// CHECK-LABEL: @test_vclzq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VCLZQ_V1_I:%.*]] = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 false) #2
// CHECK:   [[VCLZQ_V2_I:%.*]] = bitcast <8 x i16> [[VCLZQ_V1_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VCLZQ_V1_I]]
int16x8_t test_vclzq_s16(int16x8_t a) {
  return vclzq_s16(a);
}

// CHECK-LABEL: @test_vclz_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VCLZ_V1_I:%.*]] = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %a, i1 false) #2
// CHECK:   [[VCLZ_V2_I:%.*]] = bitcast <2 x i32> [[VCLZ_V1_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VCLZ_V1_I]]
int32x2_t test_vclz_s32(int32x2_t a) {
  return vclz_s32(a);
}

// CHECK-LABEL: @test_vclzq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VCLZQ_V1_I:%.*]] = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 false) #2
// CHECK:   [[VCLZQ_V2_I:%.*]] = bitcast <4 x i32> [[VCLZQ_V1_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VCLZQ_V1_I]]
int32x4_t test_vclzq_s32(int32x4_t a) {
  return vclzq_s32(a);
}

// CHECK-LABEL: @test_vclz_u8(
// CHECK:   [[VCLZ_V_I:%.*]] = call <8 x i8> @llvm.ctlz.v8i8(<8 x i8> %a, i1 false) #2
// CHECK:   ret <8 x i8> [[VCLZ_V_I]]
uint8x8_t test_vclz_u8(uint8x8_t a) {
  return vclz_u8(a);
}

// CHECK-LABEL: @test_vclzq_u8(
// CHECK:   [[VCLZQ_V_I:%.*]] = call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 false) #2
// CHECK:   ret <16 x i8> [[VCLZQ_V_I]]
uint8x16_t test_vclzq_u8(uint8x16_t a) {
  return vclzq_u8(a);
}

// CHECK-LABEL: @test_vclz_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VCLZ_V1_I:%.*]] = call <4 x i16> @llvm.ctlz.v4i16(<4 x i16> %a, i1 false) #2
// CHECK:   [[VCLZ_V2_I:%.*]] = bitcast <4 x i16> [[VCLZ_V1_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VCLZ_V1_I]]
uint16x4_t test_vclz_u16(uint16x4_t a) {
  return vclz_u16(a);
}

// CHECK-LABEL: @test_vclzq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VCLZQ_V1_I:%.*]] = call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 false) #2
// CHECK:   [[VCLZQ_V2_I:%.*]] = bitcast <8 x i16> [[VCLZQ_V1_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VCLZQ_V1_I]]
uint16x8_t test_vclzq_u16(uint16x8_t a) {
  return vclzq_u16(a);
}

// CHECK-LABEL: @test_vclz_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VCLZ_V1_I:%.*]] = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %a, i1 false) #2
// CHECK:   [[VCLZ_V2_I:%.*]] = bitcast <2 x i32> [[VCLZ_V1_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VCLZ_V1_I]]
uint32x2_t test_vclz_u32(uint32x2_t a) {
  return vclz_u32(a);
}

// CHECK-LABEL: @test_vclzq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VCLZQ_V1_I:%.*]] = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 false) #2
// CHECK:   [[VCLZQ_V2_I:%.*]] = bitcast <4 x i32> [[VCLZQ_V1_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VCLZQ_V1_I]]
uint32x4_t test_vclzq_u32(uint32x4_t a) {
  return vclzq_u32(a);
}

// CHECK-LABEL: @test_vcnt_s8(
// CHECK:   [[VCNT_V_I:%.*]] = call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VCNT_V_I]]
int8x8_t test_vcnt_s8(int8x8_t a) {
  return vcnt_s8(a);
}

// CHECK-LABEL: @test_vcntq_s8(
// CHECK:   [[VCNTQ_V_I:%.*]] = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VCNTQ_V_I]]
int8x16_t test_vcntq_s8(int8x16_t a) {
  return vcntq_s8(a);
}

// CHECK-LABEL: @test_vcnt_u8(
// CHECK:   [[VCNT_V_I:%.*]] = call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VCNT_V_I]]
uint8x8_t test_vcnt_u8(uint8x8_t a) {
  return vcnt_u8(a);
}

// CHECK-LABEL: @test_vcntq_u8(
// CHECK:   [[VCNTQ_V_I:%.*]] = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VCNTQ_V_I]]
uint8x16_t test_vcntq_u8(uint8x16_t a) {
  return vcntq_u8(a);
}

// CHECK-LABEL: @test_vcnt_p8(
// CHECK:   [[VCNT_V_I:%.*]] = call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VCNT_V_I]]
poly8x8_t test_vcnt_p8(poly8x8_t a) {
  return vcnt_p8(a);
}

// CHECK-LABEL: @test_vcntq_p8(
// CHECK:   [[VCNTQ_V_I:%.*]] = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VCNTQ_V_I]]
poly8x16_t test_vcntq_p8(poly8x16_t a) {
  return vcntq_p8(a);
}

// CHECK-LABEL: @test_vmvn_s8(
// CHECK:   [[NEG_I:%.*]] = xor <8 x i8> %a, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   ret <8 x i8> [[NEG_I]]
int8x8_t test_vmvn_s8(int8x8_t a) {
  return vmvn_s8(a);
}

// CHECK-LABEL: @test_vmvnq_s8(
// CHECK:   [[NEG_I:%.*]] = xor <16 x i8> %a, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   ret <16 x i8> [[NEG_I]]
int8x16_t test_vmvnq_s8(int8x16_t a) {
  return vmvnq_s8(a);
}

// CHECK-LABEL: @test_vmvn_s16(
// CHECK:   [[NEG_I:%.*]] = xor <4 x i16> %a, <i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   ret <4 x i16> [[NEG_I]]
int16x4_t test_vmvn_s16(int16x4_t a) {
  return vmvn_s16(a);
}

// CHECK-LABEL: @test_vmvnq_s16(
// CHECK:   [[NEG_I:%.*]] = xor <8 x i16> %a, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   ret <8 x i16> [[NEG_I]]
int16x8_t test_vmvnq_s16(int16x8_t a) {
  return vmvnq_s16(a);
}

// CHECK-LABEL: @test_vmvn_s32(
// CHECK:   [[NEG_I:%.*]] = xor <2 x i32> %a, <i32 -1, i32 -1>
// CHECK:   ret <2 x i32> [[NEG_I]]
int32x2_t test_vmvn_s32(int32x2_t a) {
  return vmvn_s32(a);
}

// CHECK-LABEL: @test_vmvnq_s32(
// CHECK:   [[NEG_I:%.*]] = xor <4 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   ret <4 x i32> [[NEG_I]]
int32x4_t test_vmvnq_s32(int32x4_t a) {
  return vmvnq_s32(a);
}

// CHECK-LABEL: @test_vmvn_u8(
// CHECK:   [[NEG_I:%.*]] = xor <8 x i8> %a, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   ret <8 x i8> [[NEG_I]]
uint8x8_t test_vmvn_u8(uint8x8_t a) {
  return vmvn_u8(a);
}

// CHECK-LABEL: @test_vmvnq_u8(
// CHECK:   [[NEG_I:%.*]] = xor <16 x i8> %a, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   ret <16 x i8> [[NEG_I]]
uint8x16_t test_vmvnq_u8(uint8x16_t a) {
  return vmvnq_u8(a);
}

// CHECK-LABEL: @test_vmvn_u16(
// CHECK:   [[NEG_I:%.*]] = xor <4 x i16> %a, <i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   ret <4 x i16> [[NEG_I]]
uint16x4_t test_vmvn_u16(uint16x4_t a) {
  return vmvn_u16(a);
}

// CHECK-LABEL: @test_vmvnq_u16(
// CHECK:   [[NEG_I:%.*]] = xor <8 x i16> %a, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:   ret <8 x i16> [[NEG_I]]
uint16x8_t test_vmvnq_u16(uint16x8_t a) {
  return vmvnq_u16(a);
}

// CHECK-LABEL: @test_vmvn_u32(
// CHECK:   [[NEG_I:%.*]] = xor <2 x i32> %a, <i32 -1, i32 -1>
// CHECK:   ret <2 x i32> [[NEG_I]]
uint32x2_t test_vmvn_u32(uint32x2_t a) {
  return vmvn_u32(a);
}

// CHECK-LABEL: @test_vmvnq_u32(
// CHECK:   [[NEG_I:%.*]] = xor <4 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1>
// CHECK:   ret <4 x i32> [[NEG_I]]
uint32x4_t test_vmvnq_u32(uint32x4_t a) {
  return vmvnq_u32(a);
}

// CHECK-LABEL: @test_vmvn_p8(
// CHECK:   [[NEG_I:%.*]] = xor <8 x i8> %a, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   ret <8 x i8> [[NEG_I]]
poly8x8_t test_vmvn_p8(poly8x8_t a) {
  return vmvn_p8(a);
}

// CHECK-LABEL: @test_vmvnq_p8(
// CHECK:   [[NEG_I:%.*]] = xor <16 x i8> %a, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   ret <16 x i8> [[NEG_I]]
poly8x16_t test_vmvnq_p8(poly8x16_t a) {
  return vmvnq_p8(a);
}

// CHECK-LABEL: @test_vrbit_s8(
// CHECK:   [[VRBIT_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rbit.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VRBIT_I]]
int8x8_t test_vrbit_s8(int8x8_t a) {
  return vrbit_s8(a);
}

// CHECK-LABEL: @test_vrbitq_s8(
// CHECK:   [[VRBIT_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VRBIT_I]]
int8x16_t test_vrbitq_s8(int8x16_t a) {
  return vrbitq_s8(a);
}

// CHECK-LABEL: @test_vrbit_u8(
// CHECK:   [[VRBIT_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rbit.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VRBIT_I]]
uint8x8_t test_vrbit_u8(uint8x8_t a) {
  return vrbit_u8(a);
}

// CHECK-LABEL: @test_vrbitq_u8(
// CHECK:   [[VRBIT_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VRBIT_I]]
uint8x16_t test_vrbitq_u8(uint8x16_t a) {
  return vrbitq_u8(a);
}

// CHECK-LABEL: @test_vrbit_p8(
// CHECK:   [[VRBIT_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rbit.v8i8(<8 x i8> %a) #2
// CHECK:   ret <8 x i8> [[VRBIT_I]]
poly8x8_t test_vrbit_p8(poly8x8_t a) {
  return vrbit_p8(a);
}

// CHECK-LABEL: @test_vrbitq_p8(
// CHECK:   [[VRBIT_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8> %a) #2
// CHECK:   ret <16 x i8> [[VRBIT_I]]
poly8x16_t test_vrbitq_p8(poly8x16_t a) {
  return vrbitq_p8(a);
}

// CHECK-LABEL: @test_vmovn_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VMOVN_I:%.*]] = trunc <8 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[VMOVN_I]]
int8x8_t test_vmovn_s16(int16x8_t a) {
  return vmovn_s16(a);
}

// CHECK-LABEL: @test_vmovn_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VMOVN_I:%.*]] = trunc <4 x i32> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[VMOVN_I]]
int16x4_t test_vmovn_s32(int32x4_t a) {
  return vmovn_s32(a);
}

// CHECK-LABEL: @test_vmovn_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VMOVN_I:%.*]] = trunc <2 x i64> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[VMOVN_I]]
int32x2_t test_vmovn_s64(int64x2_t a) {
  return vmovn_s64(a);
}

// CHECK-LABEL: @test_vmovn_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VMOVN_I:%.*]] = trunc <8 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[VMOVN_I]]
uint8x8_t test_vmovn_u16(uint16x8_t a) {
  return vmovn_u16(a);
}

// CHECK-LABEL: @test_vmovn_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VMOVN_I:%.*]] = trunc <4 x i32> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[VMOVN_I]]
uint16x4_t test_vmovn_u32(uint32x4_t a) {
  return vmovn_u32(a);
}

// CHECK-LABEL: @test_vmovn_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VMOVN_I:%.*]] = trunc <2 x i64> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[VMOVN_I]]
uint32x2_t test_vmovn_u64(uint64x2_t a) {
  return vmovn_u64(a);
}

// CHECK-LABEL: @test_vmovn_high_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VMOVN_I_I:%.*]] = trunc <8 x i16> %b to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VMOVN_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
int8x16_t test_vmovn_high_s16(int8x8_t a, int16x8_t b) {
  return vmovn_high_s16(a, b);
}

// CHECK-LABEL: @test_vmovn_high_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VMOVN_I_I:%.*]] = trunc <4 x i32> %b to <4 x i16>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VMOVN_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
int16x8_t test_vmovn_high_s32(int16x4_t a, int32x4_t b) {
  return vmovn_high_s32(a, b);
}

// CHECK-LABEL: @test_vmovn_high_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VMOVN_I_I:%.*]] = trunc <2 x i64> %b to <2 x i32>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VMOVN_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
int32x4_t test_vmovn_high_s64(int32x2_t a, int64x2_t b) {
  return vmovn_high_s64(a, b);
}

// CHECK-LABEL: @test_vmovn_high_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VMOVN_I_I:%.*]] = trunc <8 x i16> %b to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VMOVN_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
int8x16_t test_vmovn_high_u16(int8x8_t a, int16x8_t b) {
  return vmovn_high_u16(a, b);
}

// CHECK-LABEL: @test_vmovn_high_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VMOVN_I_I:%.*]] = trunc <4 x i32> %b to <4 x i16>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VMOVN_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
int16x8_t test_vmovn_high_u32(int16x4_t a, int32x4_t b) {
  return vmovn_high_u32(a, b);
}

// CHECK-LABEL: @test_vmovn_high_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VMOVN_I_I:%.*]] = trunc <2 x i64> %b to <2 x i32>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VMOVN_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
int32x4_t test_vmovn_high_u64(int32x2_t a, int64x2_t b) {
  return vmovn_high_u64(a, b);
}

// CHECK-LABEL: @test_vqmovun_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQMOVUN_V1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtun.v8i8(<8 x i16> %a) #2
// CHECK:   ret <8 x i8> [[VQMOVUN_V1_I]]
int8x8_t test_vqmovun_s16(int16x8_t a) {
  return vqmovun_s16(a);
}

// CHECK-LABEL: @test_vqmovun_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQMOVUN_V1_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32> %a) #2
// CHECK:   [[VQMOVUN_V2_I:%.*]] = bitcast <4 x i16> [[VQMOVUN_V1_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQMOVUN_V1_I]]
int16x4_t test_vqmovun_s32(int32x4_t a) {
  return vqmovun_s32(a);
}

// CHECK-LABEL: @test_vqmovun_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQMOVUN_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqxtun.v2i32(<2 x i64> %a) #2
// CHECK:   [[VQMOVUN_V2_I:%.*]] = bitcast <2 x i32> [[VQMOVUN_V1_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQMOVUN_V1_I]]
int32x2_t test_vqmovun_s64(int64x2_t a) {
  return vqmovun_s64(a);
}

// CHECK-LABEL: @test_vqmovun_high_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQMOVUN_V1_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtun.v8i8(<8 x i16> %b) #2
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQMOVUN_V1_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
int8x16_t test_vqmovun_high_s16(int8x8_t a, int16x8_t b) {
  return vqmovun_high_s16(a, b);
}

// CHECK-LABEL: @test_vqmovun_high_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQMOVUN_V1_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32> %b) #2
// CHECK:   [[VQMOVUN_V2_I_I:%.*]] = bitcast <4 x i16> [[VQMOVUN_V1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQMOVUN_V1_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
int16x8_t test_vqmovun_high_s32(int16x4_t a, int32x4_t b) {
  return vqmovun_high_s32(a, b);
}

// CHECK-LABEL: @test_vqmovun_high_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQMOVUN_V1_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqxtun.v2i32(<2 x i64> %b) #2
// CHECK:   [[VQMOVUN_V2_I_I:%.*]] = bitcast <2 x i32> [[VQMOVUN_V1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQMOVUN_V1_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
int32x4_t test_vqmovun_high_s64(int32x2_t a, int64x2_t b) {
  return vqmovun_high_s64(a, b);
}

// CHECK-LABEL: @test_vqmovn_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQMOVN_V1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtn.v8i8(<8 x i16> %a) #2
// CHECK:   ret <8 x i8> [[VQMOVN_V1_I]]
int8x8_t test_vqmovn_s16(int16x8_t a) {
  return vqmovn_s16(a);
}

// CHECK-LABEL: @test_vqmovn_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQMOVN_V1_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtn.v4i16(<4 x i32> %a) #2
// CHECK:   [[VQMOVN_V2_I:%.*]] = bitcast <4 x i16> [[VQMOVN_V1_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQMOVN_V1_I]]
int16x4_t test_vqmovn_s32(int32x4_t a) {
  return vqmovn_s32(a);
}

// CHECK-LABEL: @test_vqmovn_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQMOVN_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqxtn.v2i32(<2 x i64> %a) #2
// CHECK:   [[VQMOVN_V2_I:%.*]] = bitcast <2 x i32> [[VQMOVN_V1_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQMOVN_V1_I]]
int32x2_t test_vqmovn_s64(int64x2_t a) {
  return vqmovn_s64(a);
}

// CHECK-LABEL: @test_vqmovn_high_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQMOVN_V1_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtn.v8i8(<8 x i16> %b) #2
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQMOVN_V1_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
int8x16_t test_vqmovn_high_s16(int8x8_t a, int16x8_t b) {
  return vqmovn_high_s16(a, b);
}

// CHECK-LABEL: @test_vqmovn_high_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQMOVN_V1_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtn.v4i16(<4 x i32> %b) #2
// CHECK:   [[VQMOVN_V2_I_I:%.*]] = bitcast <4 x i16> [[VQMOVN_V1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQMOVN_V1_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
int16x8_t test_vqmovn_high_s32(int16x4_t a, int32x4_t b) {
  return vqmovn_high_s32(a, b);
}

// CHECK-LABEL: @test_vqmovn_high_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQMOVN_V1_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqxtn.v2i32(<2 x i64> %b) #2
// CHECK:   [[VQMOVN_V2_I_I:%.*]] = bitcast <2 x i32> [[VQMOVN_V1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQMOVN_V1_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
int32x4_t test_vqmovn_high_s64(int32x2_t a, int64x2_t b) {
  return vqmovn_high_s64(a, b);
}

// CHECK-LABEL: @test_vqmovn_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQMOVN_V1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqxtn.v8i8(<8 x i16> %a) #2
// CHECK:   ret <8 x i8> [[VQMOVN_V1_I]]
uint8x8_t test_vqmovn_u16(uint16x8_t a) {
  return vqmovn_u16(a);
}

// CHECK-LABEL: @test_vqmovn_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQMOVN_V1_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqxtn.v4i16(<4 x i32> %a) #2
// CHECK:   [[VQMOVN_V2_I:%.*]] = bitcast <4 x i16> [[VQMOVN_V1_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQMOVN_V1_I]]
uint16x4_t test_vqmovn_u32(uint32x4_t a) {
  return vqmovn_u32(a);
}

// CHECK-LABEL: @test_vqmovn_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQMOVN_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqxtn.v2i32(<2 x i64> %a) #2
// CHECK:   [[VQMOVN_V2_I:%.*]] = bitcast <2 x i32> [[VQMOVN_V1_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQMOVN_V1_I]]
uint32x2_t test_vqmovn_u64(uint64x2_t a) {
  return vqmovn_u64(a);
}

// CHECK-LABEL: @test_vqmovn_high_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQMOVN_V1_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqxtn.v8i8(<8 x i16> %b) #2
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQMOVN_V1_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
uint8x16_t test_vqmovn_high_u16(uint8x8_t a, uint16x8_t b) {
  return vqmovn_high_u16(a, b);
}

// CHECK-LABEL: @test_vqmovn_high_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQMOVN_V1_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqxtn.v4i16(<4 x i32> %b) #2
// CHECK:   [[VQMOVN_V2_I_I:%.*]] = bitcast <4 x i16> [[VQMOVN_V1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQMOVN_V1_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
uint16x8_t test_vqmovn_high_u32(uint16x4_t a, uint32x4_t b) {
  return vqmovn_high_u32(a, b);
}

// CHECK-LABEL: @test_vqmovn_high_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQMOVN_V1_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqxtn.v2i32(<2 x i64> %b) #2
// CHECK:   [[VQMOVN_V2_I_I:%.*]] = bitcast <2 x i32> [[VQMOVN_V1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQMOVN_V1_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
uint32x4_t test_vqmovn_high_u64(uint32x2_t a, uint64x2_t b) {
  return vqmovn_high_u64(a, b);
}

// CHECK-LABEL: @test_vshll_n_s8(
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> %a to <8 x i16>
// CHECK:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// CHECK:   ret <8 x i16> [[VSHLL_N]]
int16x8_t test_vshll_n_s8(int8x8_t a) {
  return vshll_n_s8(a, 8);
}

// CHECK-LABEL: @test_vshll_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP2:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// CHECK:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], <i32 16, i32 16, i32 16, i32 16>
// CHECK:   ret <4 x i32> [[VSHLL_N]]
int32x4_t test_vshll_n_s16(int16x4_t a) {
  return vshll_n_s16(a, 16);
}

// CHECK-LABEL: @test_vshll_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// CHECK:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], <i64 32, i64 32>
// CHECK:   ret <2 x i64> [[VSHLL_N]]
int64x2_t test_vshll_n_s32(int32x2_t a) {
  return vshll_n_s32(a, 32);
}

// CHECK-LABEL: @test_vshll_n_u8(
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> %a to <8 x i16>
// CHECK:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// CHECK:   ret <8 x i16> [[VSHLL_N]]
uint16x8_t test_vshll_n_u8(uint8x8_t a) {
  return vshll_n_u8(a, 8);
}

// CHECK-LABEL: @test_vshll_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP2:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// CHECK:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], <i32 16, i32 16, i32 16, i32 16>
// CHECK:   ret <4 x i32> [[VSHLL_N]]
uint32x4_t test_vshll_n_u16(uint16x4_t a) {
  return vshll_n_u16(a, 16);
}

// CHECK-LABEL: @test_vshll_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP2:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// CHECK:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], <i64 32, i64 32>
// CHECK:   ret <2 x i64> [[VSHLL_N]]
uint64x2_t test_vshll_n_u32(uint32x2_t a) {
  return vshll_n_u32(a, 32);
}

// CHECK-LABEL: @test_vshll_high_n_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// CHECK:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// CHECK:   ret <8 x i16> [[VSHLL_N]]
int16x8_t test_vshll_high_n_s8(int8x16_t a) {
  return vshll_high_n_s8(a, 8);
}

// CHECK-LABEL: @test_vshll_high_n_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP2:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// CHECK:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], <i32 16, i32 16, i32 16, i32 16>
// CHECK:   ret <4 x i32> [[VSHLL_N]]
int32x4_t test_vshll_high_n_s16(int16x8_t a) {
  return vshll_high_n_s16(a, 16);
}

// CHECK-LABEL: @test_vshll_high_n_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// CHECK:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], <i64 32, i64 32>
// CHECK:   ret <2 x i64> [[VSHLL_N]]
int64x2_t test_vshll_high_n_s32(int32x4_t a) {
  return vshll_high_n_s32(a, 32);
}

// CHECK-LABEL: @test_vshll_high_n_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// CHECK:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// CHECK:   ret <8 x i16> [[VSHLL_N]]
uint16x8_t test_vshll_high_n_u8(uint8x16_t a) {
  return vshll_high_n_u8(a, 8);
}

// CHECK-LABEL: @test_vshll_high_n_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP2:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// CHECK:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], <i32 16, i32 16, i32 16, i32 16>
// CHECK:   ret <4 x i32> [[VSHLL_N]]
uint32x4_t test_vshll_high_n_u16(uint16x8_t a) {
  return vshll_high_n_u16(a, 16);
}

// CHECK-LABEL: @test_vshll_high_n_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP2:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// CHECK:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], <i64 32, i64 32>
// CHECK:   ret <2 x i64> [[VSHLL_N]]
uint64x2_t test_vshll_high_n_u32(uint32x4_t a) {
  return vshll_high_n_u32(a, 32);
}

// CHECK-LABEL: @test_vcvt_f16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVT_F16_F321_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.vcvtfp2hf(<4 x float> %a) #2
// CHECK:   [[VCVT_F16_F322_I:%.*]] = bitcast <4 x i16> [[VCVT_F16_F321_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[VCVT_F16_F322_I]] to <4 x half>
// CHECK:   ret <4 x half> [[TMP1]]
float16x4_t test_vcvt_f16_f32(float32x4_t a) {
  return vcvt_f16_f32(a);
}

// CHECK-LABEL: @test_vcvt_high_f16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VCVT_F16_F321_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.vcvtfp2hf(<4 x float> %b) #2
// CHECK:   [[VCVT_F16_F322_I_I:%.*]] = bitcast <4 x i16> [[VCVT_F16_F321_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[VCVT_F16_F322_I_I]] to <4 x half>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x half> %a, <4 x half> [[TMP1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x half> [[SHUFFLE_I_I]]
float16x8_t test_vcvt_high_f16_f32(float16x4_t a, float32x4_t b) {
  return vcvt_high_f16_f32(a, b);
}

// CHECK-LABEL: @test_vcvt_f32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVT_I:%.*]] = fptrunc <2 x double> %a to <2 x float>
// CHECK:   ret <2 x float> [[VCVT_I]]
float32x2_t test_vcvt_f32_f64(float64x2_t a) {
  return vcvt_f32_f64(a);
}

// CHECK-LABEL: @test_vcvt_high_f32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VCVT_I_I:%.*]] = fptrunc <2 x double> %b to <2 x float>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x float> %a, <2 x float> [[VCVT_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x float> [[SHUFFLE_I_I]]
float32x4_t test_vcvt_high_f32_f64(float32x2_t a, float64x2_t b) {
  return vcvt_high_f32_f64(a, b);
}

// CHECK-LABEL: @test_vcvtx_f32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTX_F32_V1_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x float> [[VCVTX_F32_V1_I]]
float32x2_t test_vcvtx_f32_f64(float64x2_t a) {
  return vcvtx_f32_f64(a);
}

// CHECK-LABEL: @test_vcvtx_high_f32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VCVTX_F32_V1_I_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double> %b) #2
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x float> %a, <2 x float> [[VCVTX_F32_V1_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x float> [[SHUFFLE_I_I]]
float32x4_t test_vcvtx_high_f32_f64(float32x2_t a, float64x2_t b) {
  return vcvtx_high_f32_f64(a, b);
}

// CHECK-LABEL: @test_vcvt_f32_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK:   [[VCVT_F32_F16_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VCVT_F32_F161_I:%.*]] = call <4 x float> @llvm.aarch64.neon.vcvthf2fp(<4 x i16> [[VCVT_F32_F16_I]]) #2
// CHECK:   [[VCVT_F32_F162_I:%.*]] = bitcast <4 x float> [[VCVT_F32_F161_I]] to <16 x i8>
// CHECK:   ret <4 x float> [[VCVT_F32_F161_I]]
float32x4_t test_vcvt_f32_f16(float16x4_t a) {
  return vcvt_f32_f16(a);
}

// CHECK-LABEL: @test_vcvt_high_f32_f16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VCVT_F32_F16_I_I:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VCVT_F32_F161_I_I:%.*]] = call <4 x float> @llvm.aarch64.neon.vcvthf2fp(<4 x i16> [[VCVT_F32_F16_I_I]]) #2
// CHECK:   [[VCVT_F32_F162_I_I:%.*]] = bitcast <4 x float> [[VCVT_F32_F161_I_I]] to <16 x i8>
// CHECK:   ret <4 x float> [[VCVT_F32_F161_I_I]]
float32x4_t test_vcvt_high_f32_f16(float16x8_t a) {
  return vcvt_high_f32_f16(a);
}

// CHECK-LABEL: @test_vcvt_f64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVT_I:%.*]] = fpext <2 x float> %a to <2 x double>
// CHECK:   ret <2 x double> [[VCVT_I]]
float64x2_t test_vcvt_f64_f32(float32x2_t a) {
  return vcvt_f64_f32(a);
}

// CHECK-LABEL: @test_vcvt_high_f64_f32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x float> %a, <4 x float> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VCVT_I_I:%.*]] = fpext <2 x float> [[SHUFFLE_I_I]] to <2 x double>
// CHECK:   ret <2 x double> [[VCVT_I_I]]
float64x2_t test_vcvt_high_f64_f32(float32x4_t a) {
  return vcvt_high_f64_f32(a);
}

// CHECK-LABEL: @test_vrndn_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDN1_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frintn.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDN1_I]]
float32x2_t test_vrndn_f32(float32x2_t a) {
  return vrndn_f32(a);
}

// CHECK-LABEL: @test_vrndnq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDN1_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDN1_I]]
float32x4_t test_vrndnq_f32(float32x4_t a) {
  return vrndnq_f32(a);
}

// CHECK-LABEL: @test_vrndnq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRNDN1_I:%.*]] = call <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRNDN1_I]]
float64x2_t test_vrndnq_f64(float64x2_t a) {
  return vrndnq_f64(a);
}

// CHECK-LABEL: @test_vrnda_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDA1_I:%.*]] = call <2 x float> @llvm.round.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDA1_I]]
float32x2_t test_vrnda_f32(float32x2_t a) {
  return vrnda_f32(a);
}

// CHECK-LABEL: @test_vrndaq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDA1_I:%.*]] = call <4 x float> @llvm.round.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDA1_I]]
float32x4_t test_vrndaq_f32(float32x4_t a) {
  return vrndaq_f32(a);
}

// CHECK-LABEL: @test_vrndaq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRNDA1_I:%.*]] = call <2 x double> @llvm.round.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRNDA1_I]]
float64x2_t test_vrndaq_f64(float64x2_t a) {
  return vrndaq_f64(a);
}

// CHECK-LABEL: @test_vrndp_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDP1_I:%.*]] = call <2 x float> @llvm.ceil.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDP1_I]]
float32x2_t test_vrndp_f32(float32x2_t a) {
  return vrndp_f32(a);
}

// CHECK-LABEL: @test_vrndpq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDP1_I:%.*]] = call <4 x float> @llvm.ceil.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDP1_I]]
float32x4_t test_vrndpq_f32(float32x4_t a) {
  return vrndpq_f32(a);
}

// CHECK-LABEL: @test_vrndpq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRNDP1_I:%.*]] = call <2 x double> @llvm.ceil.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRNDP1_I]]
float64x2_t test_vrndpq_f64(float64x2_t a) {
  return vrndpq_f64(a);
}

// CHECK-LABEL: @test_vrndm_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDM1_I:%.*]] = call <2 x float> @llvm.floor.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDM1_I]]
float32x2_t test_vrndm_f32(float32x2_t a) {
  return vrndm_f32(a);
}

// CHECK-LABEL: @test_vrndmq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDM1_I:%.*]] = call <4 x float> @llvm.floor.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDM1_I]]
float32x4_t test_vrndmq_f32(float32x4_t a) {
  return vrndmq_f32(a);
}

// CHECK-LABEL: @test_vrndmq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRNDM1_I:%.*]] = call <2 x double> @llvm.floor.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRNDM1_I]]
float64x2_t test_vrndmq_f64(float64x2_t a) {
  return vrndmq_f64(a);
}

// CHECK-LABEL: @test_vrndx_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDX1_I:%.*]] = call <2 x float> @llvm.rint.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDX1_I]]
float32x2_t test_vrndx_f32(float32x2_t a) {
  return vrndx_f32(a);
}

// CHECK-LABEL: @test_vrndxq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDX1_I:%.*]] = call <4 x float> @llvm.rint.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDX1_I]]
float32x4_t test_vrndxq_f32(float32x4_t a) {
  return vrndxq_f32(a);
}

// CHECK-LABEL: @test_vrndxq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRNDX1_I:%.*]] = call <2 x double> @llvm.rint.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRNDX1_I]]
float64x2_t test_vrndxq_f64(float64x2_t a) {
  return vrndxq_f64(a);
}

// CHECK-LABEL: @test_vrnd_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDZ1_I:%.*]] = call <2 x float> @llvm.trunc.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDZ1_I]]
float32x2_t test_vrnd_f32(float32x2_t a) {
  return vrnd_f32(a);
}

// CHECK-LABEL: @test_vrndq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDZ1_I:%.*]] = call <4 x float> @llvm.trunc.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDZ1_I]]
float32x4_t test_vrndq_f32(float32x4_t a) {
  return vrndq_f32(a);
}

// CHECK-LABEL: @test_vrndq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRNDZ1_I:%.*]] = call <2 x double> @llvm.trunc.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRNDZ1_I]]
float64x2_t test_vrndq_f64(float64x2_t a) {
  return vrndq_f64(a);
}

// CHECK-LABEL: @test_vrndi_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRNDI1_I:%.*]] = call <2 x float> @llvm.nearbyint.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRNDI1_I]]
float32x2_t test_vrndi_f32(float32x2_t a) {
  return vrndi_f32(a);
}

// CHECK-LABEL: @test_vrndiq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRNDI1_I:%.*]] = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRNDI1_I]]
float32x4_t test_vrndiq_f32(float32x4_t a) {
  return vrndiq_f32(a);
}

// CHECK-LABEL: @test_vrndiq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRNDI1_I:%.*]] = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRNDI1_I]]
float64x2_t test_vrndiq_f64(float64x2_t a) {
  return vrndiq_f64(a);
}

// CHECK-LABEL: @test_vcvt_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fptosi <2 x float> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP1]]
int32x2_t test_vcvt_s32_f32(float32x2_t a) {
  return vcvt_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtq_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fptosi <4 x float> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP1]]
int32x4_t test_vcvtq_s32_f32(float32x4_t a) {
  return vcvtq_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtq_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fptosi <2 x double> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP1]]
int64x2_t test_vcvtq_s64_f64(float64x2_t a) {
  return vcvtq_s64_f64(a);
}

// CHECK-LABEL: @test_vcvt_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = fptoui <2 x float> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP1]]
uint32x2_t test_vcvt_u32_f32(float32x2_t a) {
  return vcvt_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtq_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fptoui <4 x float> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP1]]
uint32x4_t test_vcvtq_u32_f32(float32x4_t a) {
  return vcvtq_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtq_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = fptoui <2 x double> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP1]]
uint64x2_t test_vcvtq_u64_f64(float64x2_t a) {
  return vcvtq_u64_f64(a);
}

// CHECK-LABEL: @test_vcvtn_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVTN1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.fcvtns.v2i32.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x i32> [[VCVTN1_I]]
int32x2_t test_vcvtn_s32_f32(float32x2_t a) {
  return vcvtn_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtnq_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVTN1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.fcvtns.v4i32.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x i32> [[VCVTN1_I]]
int32x4_t test_vcvtnq_s32_f32(float32x4_t a) {
  return vcvtnq_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtnq_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTN1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.fcvtns.v2i64.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x i64> [[VCVTN1_I]]
int64x2_t test_vcvtnq_s64_f64(float64x2_t a) {
  return vcvtnq_s64_f64(a);
}

// CHECK-LABEL: @test_vcvtn_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVTN1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.fcvtnu.v2i32.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x i32> [[VCVTN1_I]]
uint32x2_t test_vcvtn_u32_f32(float32x2_t a) {
  return vcvtn_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtnq_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVTN1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.fcvtnu.v4i32.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x i32> [[VCVTN1_I]]
uint32x4_t test_vcvtnq_u32_f32(float32x4_t a) {
  return vcvtnq_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtnq_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTN1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.fcvtnu.v2i64.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x i64> [[VCVTN1_I]]
uint64x2_t test_vcvtnq_u64_f64(float64x2_t a) {
  return vcvtnq_u64_f64(a);
}

// CHECK-LABEL: @test_vcvtp_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVTP1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.fcvtps.v2i32.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x i32> [[VCVTP1_I]]
int32x2_t test_vcvtp_s32_f32(float32x2_t a) {
  return vcvtp_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtpq_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVTP1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.fcvtps.v4i32.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x i32> [[VCVTP1_I]]
int32x4_t test_vcvtpq_s32_f32(float32x4_t a) {
  return vcvtpq_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtpq_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTP1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.fcvtps.v2i64.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x i64> [[VCVTP1_I]]
int64x2_t test_vcvtpq_s64_f64(float64x2_t a) {
  return vcvtpq_s64_f64(a);
}

// CHECK-LABEL: @test_vcvtp_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVTP1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.fcvtpu.v2i32.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x i32> [[VCVTP1_I]]
uint32x2_t test_vcvtp_u32_f32(float32x2_t a) {
  return vcvtp_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtpq_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVTP1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.fcvtpu.v4i32.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x i32> [[VCVTP1_I]]
uint32x4_t test_vcvtpq_u32_f32(float32x4_t a) {
  return vcvtpq_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtpq_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTP1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.fcvtpu.v2i64.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x i64> [[VCVTP1_I]]
uint64x2_t test_vcvtpq_u64_f64(float64x2_t a) {
  return vcvtpq_u64_f64(a);
}

// CHECK-LABEL: @test_vcvtm_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVTM1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.fcvtms.v2i32.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x i32> [[VCVTM1_I]]
int32x2_t test_vcvtm_s32_f32(float32x2_t a) {
  return vcvtm_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtmq_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVTM1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.fcvtms.v4i32.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x i32> [[VCVTM1_I]]
int32x4_t test_vcvtmq_s32_f32(float32x4_t a) {
  return vcvtmq_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtmq_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTM1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.fcvtms.v2i64.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x i64> [[VCVTM1_I]]
int64x2_t test_vcvtmq_s64_f64(float64x2_t a) {
  return vcvtmq_s64_f64(a);
}

// CHECK-LABEL: @test_vcvtm_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVTM1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.fcvtmu.v2i32.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x i32> [[VCVTM1_I]]
uint32x2_t test_vcvtm_u32_f32(float32x2_t a) {
  return vcvtm_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtmq_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVTM1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.fcvtmu.v4i32.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x i32> [[VCVTM1_I]]
uint32x4_t test_vcvtmq_u32_f32(float32x4_t a) {
  return vcvtmq_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtmq_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTM1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.fcvtmu.v2i64.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x i64> [[VCVTM1_I]]
uint64x2_t test_vcvtmq_u64_f64(float64x2_t a) {
  return vcvtmq_u64_f64(a);
}

// CHECK-LABEL: @test_vcvta_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVTA1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.fcvtas.v2i32.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x i32> [[VCVTA1_I]]
int32x2_t test_vcvta_s32_f32(float32x2_t a) {
  return vcvta_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtaq_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVTA1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.fcvtas.v4i32.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x i32> [[VCVTA1_I]]
int32x4_t test_vcvtaq_s32_f32(float32x4_t a) {
  return vcvtaq_s32_f32(a);
}

// CHECK-LABEL: @test_vcvtaq_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTA1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.fcvtas.v2i64.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x i64> [[VCVTA1_I]]
int64x2_t test_vcvtaq_s64_f64(float64x2_t a) {
  return vcvtaq_s64_f64(a);
}

// CHECK-LABEL: @test_vcvta_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVTA1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.fcvtau.v2i32.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x i32> [[VCVTA1_I]]
uint32x2_t test_vcvta_u32_f32(float32x2_t a) {
  return vcvta_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtaq_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVTA1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.fcvtau.v4i32.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x i32> [[VCVTA1_I]]
uint32x4_t test_vcvtaq_u32_f32(float32x4_t a) {
  return vcvtaq_u32_f32(a);
}

// CHECK-LABEL: @test_vcvtaq_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVTA1_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.fcvtau.v2i64.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x i64> [[VCVTA1_I]]
uint64x2_t test_vcvtaq_u64_f64(float64x2_t a) {
  return vcvtaq_u64_f64(a);
}

// CHECK-LABEL: @test_vrsqrte_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRSQRTE_V1_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frsqrte.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRSQRTE_V1_I]]
float32x2_t test_vrsqrte_f32(float32x2_t a) {
  return vrsqrte_f32(a);
}

// CHECK-LABEL: @test_vrsqrteq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRSQRTEQ_V1_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frsqrte.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRSQRTEQ_V1_I]]
float32x4_t test_vrsqrteq_f32(float32x4_t a) {
  return vrsqrteq_f32(a);
}

// CHECK-LABEL: @test_vrsqrteq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRSQRTEQ_V1_I:%.*]] = call <2 x double> @llvm.aarch64.neon.frsqrte.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRSQRTEQ_V1_I]]
float64x2_t test_vrsqrteq_f64(float64x2_t a) {
  return vrsqrteq_f64(a);
}

// CHECK-LABEL: @test_vrecpe_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VRECPE_V1_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frecpe.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VRECPE_V1_I]]
float32x2_t test_vrecpe_f32(float32x2_t a) {
  return vrecpe_f32(a);
}

// CHECK-LABEL: @test_vrecpeq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VRECPEQ_V1_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frecpe.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VRECPEQ_V1_I]]
float32x4_t test_vrecpeq_f32(float32x4_t a) {
  return vrecpeq_f32(a);
}

// CHECK-LABEL: @test_vrecpeq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VRECPEQ_V1_I:%.*]] = call <2 x double> @llvm.aarch64.neon.frecpe.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VRECPEQ_V1_I]]
float64x2_t test_vrecpeq_f64(float64x2_t a) {
  return vrecpeq_f64(a);
}

// CHECK-LABEL: @test_vrecpe_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VRECPE_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.urecpe.v2i32(<2 x i32> %a) #2
// CHECK:   ret <2 x i32> [[VRECPE_V1_I]]
uint32x2_t test_vrecpe_u32(uint32x2_t a) {
  return vrecpe_u32(a);
}

// CHECK-LABEL: @test_vrecpeq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VRECPEQ_V1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.urecpe.v4i32(<4 x i32> %a) #2
// CHECK:   ret <4 x i32> [[VRECPEQ_V1_I]]
uint32x4_t test_vrecpeq_u32(uint32x4_t a) {
  return vrecpeq_u32(a);
}

// CHECK-LABEL: @test_vsqrt_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VSQRT_I:%.*]] = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %a) #2
// CHECK:   ret <2 x float> [[VSQRT_I]]
float32x2_t test_vsqrt_f32(float32x2_t a) {
  return vsqrt_f32(a);
}

// CHECK-LABEL: @test_vsqrtq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VSQRT_I:%.*]] = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a) #2
// CHECK:   ret <4 x float> [[VSQRT_I]]
float32x4_t test_vsqrtq_f32(float32x4_t a) {
  return vsqrtq_f32(a);
}

// CHECK-LABEL: @test_vsqrtq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VSQRT_I:%.*]] = call <2 x double> @llvm.sqrt.v2f64(<2 x double> %a) #2
// CHECK:   ret <2 x double> [[VSQRT_I]]
float64x2_t test_vsqrtq_f64(float64x2_t a) {
  return vsqrtq_f64(a);
}

// CHECK-LABEL: @test_vcvt_f32_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VCVT_I:%.*]] = sitofp <2 x i32> %a to <2 x float>
// CHECK:   ret <2 x float> [[VCVT_I]]
float32x2_t test_vcvt_f32_s32(int32x2_t a) {
  return vcvt_f32_s32(a);
}

// CHECK-LABEL: @test_vcvt_f32_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VCVT_I:%.*]] = uitofp <2 x i32> %a to <2 x float>
// CHECK:   ret <2 x float> [[VCVT_I]]
float32x2_t test_vcvt_f32_u32(uint32x2_t a) {
  return vcvt_f32_u32(a);
}

// CHECK-LABEL: @test_vcvtq_f32_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VCVT_I:%.*]] = sitofp <4 x i32> %a to <4 x float>
// CHECK:   ret <4 x float> [[VCVT_I]]
float32x4_t test_vcvtq_f32_s32(int32x4_t a) {
  return vcvtq_f32_s32(a);
}

// CHECK-LABEL: @test_vcvtq_f32_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VCVT_I:%.*]] = uitofp <4 x i32> %a to <4 x float>
// CHECK:   ret <4 x float> [[VCVT_I]]
float32x4_t test_vcvtq_f32_u32(uint32x4_t a) {
  return vcvtq_f32_u32(a);
}

// CHECK-LABEL: @test_vcvtq_f64_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VCVT_I:%.*]] = sitofp <2 x i64> %a to <2 x double>
// CHECK:   ret <2 x double> [[VCVT_I]]
float64x2_t test_vcvtq_f64_s64(int64x2_t a) {
  return vcvtq_f64_s64(a);
}

// CHECK-LABEL: @test_vcvtq_f64_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VCVT_I:%.*]] = uitofp <2 x i64> %a to <2 x double>
// CHECK:   ret <2 x double> [[VCVT_I]]
float64x2_t test_vcvtq_f64_u64(uint64x2_t a) {
  return vcvtq_f64_u64(a);
}
