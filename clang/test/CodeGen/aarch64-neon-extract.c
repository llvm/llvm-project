// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

// CHECK-LABEL: define <8 x i8> @test_vext_s8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[VEXT:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
// CHECK:   ret <8 x i8> [[VEXT]]
int8x8_t test_vext_s8(int8x8_t a, int8x8_t b) {
  return vext_s8(a, b, 2);
}

// CHECK-LABEL: define <4 x i16> @test_vext_s16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VEXT:%.*]] = shufflevector <4 x i16> [[TMP2]], <4 x i16> [[TMP3]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
// CHECK:   ret <4 x i16> [[VEXT]]
int16x4_t test_vext_s16(int16x4_t a, int16x4_t b) {
  return vext_s16(a, b, 3);
}

// CHECK-LABEL: define <2 x i32> @test_vext_s32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VEXT:%.*]] = shufflevector <2 x i32> [[TMP2]], <2 x i32> [[TMP3]], <2 x i32> <i32 1, i32 2>
// CHECK:   ret <2 x i32> [[VEXT]]
int32x2_t test_vext_s32(int32x2_t a, int32x2_t b) {
  return vext_s32(a, b, 1);
}

// CHECK-LABEL: define <1 x i64> @test_vext_s64(<1 x i64> %a, <1 x i64> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VEXT:%.*]] = shufflevector <1 x i64> [[TMP2]], <1 x i64> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[VEXT]]
int64x1_t test_vext_s64(int64x1_t a, int64x1_t b) {
  return vext_s64(a, b, 0);
}

// CHECK-LABEL: define <16 x i8> @test_vextq_s8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[VEXT:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
// CHECK:   ret <16 x i8> [[VEXT]]
int8x16_t test_vextq_s8(int8x16_t a, int8x16_t b) {
  return vextq_s8(a, b, 2);
}

// CHECK-LABEL: define <8 x i16> @test_vextq_s16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VEXT:%.*]] = shufflevector <8 x i16> [[TMP2]], <8 x i16> [[TMP3]], <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
// CHECK:   ret <8 x i16> [[VEXT]]
int16x8_t test_vextq_s16(int16x8_t a, int16x8_t b) {
  return vextq_s16(a, b, 3);
}

// CHECK-LABEL: define <4 x i32> @test_vextq_s32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VEXT:%.*]] = shufflevector <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> <i32 1, i32 2, i32 3, i32 4>
// CHECK:   ret <4 x i32> [[VEXT]]
int32x4_t test_vextq_s32(int32x4_t a, int32x4_t b) {
  return vextq_s32(a, b, 1);
}

// CHECK-LABEL: define <2 x i64> @test_vextq_s64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VEXT:%.*]] = shufflevector <2 x i64> [[TMP2]], <2 x i64> [[TMP3]], <2 x i32> <i32 1, i32 2>
// CHECK:   ret <2 x i64> [[VEXT]]
int64x2_t test_vextq_s64(int64x2_t a, int64x2_t b) {
  return vextq_s64(a, b, 1);
}

// CHECK-LABEL: define <8 x i8> @test_vext_u8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[VEXT:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
// CHECK:   ret <8 x i8> [[VEXT]]
uint8x8_t test_vext_u8(uint8x8_t a, uint8x8_t b) {
  return vext_u8(a, b, 2);
}

// CHECK-LABEL: define <4 x i16> @test_vext_u16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VEXT:%.*]] = shufflevector <4 x i16> [[TMP2]], <4 x i16> [[TMP3]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
// CHECK:   ret <4 x i16> [[VEXT]]
uint16x4_t test_vext_u16(uint16x4_t a, uint16x4_t b) {
  return vext_u16(a, b, 3);
}

// CHECK-LABEL: define <2 x i32> @test_vext_u32(<2 x i32> %a, <2 x i32> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VEXT:%.*]] = shufflevector <2 x i32> [[TMP2]], <2 x i32> [[TMP3]], <2 x i32> <i32 1, i32 2>
// CHECK:   ret <2 x i32> [[VEXT]]
uint32x2_t test_vext_u32(uint32x2_t a, uint32x2_t b) {
  return vext_u32(a, b, 1);
}

// CHECK-LABEL: define <1 x i64> @test_vext_u64(<1 x i64> %a, <1 x i64> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VEXT:%.*]] = shufflevector <1 x i64> [[TMP2]], <1 x i64> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[VEXT]]
uint64x1_t test_vext_u64(uint64x1_t a, uint64x1_t b) {
  return vext_u64(a, b, 0);
}

// CHECK-LABEL: define <16 x i8> @test_vextq_u8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[VEXT:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
// CHECK:   ret <16 x i8> [[VEXT]]
uint8x16_t test_vextq_u8(uint8x16_t a, uint8x16_t b) {
  return vextq_u8(a, b, 2);
}

// CHECK-LABEL: define <8 x i16> @test_vextq_u16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VEXT:%.*]] = shufflevector <8 x i16> [[TMP2]], <8 x i16> [[TMP3]], <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
// CHECK:   ret <8 x i16> [[VEXT]]
uint16x8_t test_vextq_u16(uint16x8_t a, uint16x8_t b) {
  return vextq_u16(a, b, 3);
}

// CHECK-LABEL: define <4 x i32> @test_vextq_u32(<4 x i32> %a, <4 x i32> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VEXT:%.*]] = shufflevector <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> <i32 1, i32 2, i32 3, i32 4>
// CHECK:   ret <4 x i32> [[VEXT]]
uint32x4_t test_vextq_u32(uint32x4_t a, uint32x4_t b) {
  return vextq_u32(a, b, 1);
}

// CHECK-LABEL: define <2 x i64> @test_vextq_u64(<2 x i64> %a, <2 x i64> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VEXT:%.*]] = shufflevector <2 x i64> [[TMP2]], <2 x i64> [[TMP3]], <2 x i32> <i32 1, i32 2>
// CHECK:   ret <2 x i64> [[VEXT]]
uint64x2_t test_vextq_u64(uint64x2_t a, uint64x2_t b) {
  return vextq_u64(a, b, 1);
}

// CHECK-LABEL: define <2 x float> @test_vext_f32(<2 x float> %a, <2 x float> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[VEXT:%.*]] = shufflevector <2 x float> [[TMP2]], <2 x float> [[TMP3]], <2 x i32> <i32 1, i32 2>
// CHECK:   ret <2 x float> [[VEXT]]
float32x2_t test_vext_f32(float32x2_t a, float32x2_t b) {
  return vext_f32(a, b, 1);
}

// CHECK-LABEL: define <1 x double> @test_vext_f64(<1 x double> %a, <1 x double> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// CHECK:   [[VEXT:%.*]] = shufflevector <1 x double> [[TMP2]], <1 x double> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   ret <1 x double> [[VEXT]]
float64x1_t test_vext_f64(float64x1_t a, float64x1_t b) {
  return vext_f64(a, b, 0);
}

// CHECK-LABEL: define <4 x float> @test_vextq_f32(<4 x float> %a, <4 x float> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[VEXT:%.*]] = shufflevector <4 x float> [[TMP2]], <4 x float> [[TMP3]], <4 x i32> <i32 1, i32 2, i32 3, i32 4>
// CHECK:   ret <4 x float> [[VEXT]]
float32x4_t test_vextq_f32(float32x4_t a, float32x4_t b) {
  return vextq_f32(a, b, 1);
}

// CHECK-LABEL: define <2 x double> @test_vextq_f64(<2 x double> %a, <2 x double> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[VEXT:%.*]] = shufflevector <2 x double> [[TMP2]], <2 x double> [[TMP3]], <2 x i32> <i32 1, i32 2>
// CHECK:   ret <2 x double> [[VEXT]]
float64x2_t test_vextq_f64(float64x2_t a, float64x2_t b) {
  return vextq_f64(a, b, 1);
}

// CHECK-LABEL: define <8 x i8> @test_vext_p8(<8 x i8> %a, <8 x i8> %b) #0 {
// CHECK:   [[VEXT:%.*]] = shufflevector <8 x i8> %a, <8 x i8> %b, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
// CHECK:   ret <8 x i8> [[VEXT]]
poly8x8_t test_vext_p8(poly8x8_t a, poly8x8_t b) {
  return vext_p8(a, b, 2);
}

// CHECK-LABEL: define <4 x i16> @test_vext_p16(<4 x i16> %a, <4 x i16> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VEXT:%.*]] = shufflevector <4 x i16> [[TMP2]], <4 x i16> [[TMP3]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
// CHECK:   ret <4 x i16> [[VEXT]]
poly16x4_t test_vext_p16(poly16x4_t a, poly16x4_t b) {
  return vext_p16(a, b, 3);
}

// CHECK-LABEL: define <16 x i8> @test_vextq_p8(<16 x i8> %a, <16 x i8> %b) #0 {
// CHECK:   [[VEXT:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
// CHECK:   ret <16 x i8> [[VEXT]]
poly8x16_t test_vextq_p8(poly8x16_t a, poly8x16_t b) {
  return vextq_p8(a, b, 2);
}

// CHECK-LABEL: define <8 x i16> @test_vextq_p16(<8 x i16> %a, <8 x i16> %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VEXT:%.*]] = shufflevector <8 x i16> [[TMP2]], <8 x i16> [[TMP3]], <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
// CHECK:   ret <8 x i16> [[VEXT]]
poly16x8_t test_vextq_p16(poly16x8_t a, poly16x8_t b) {
  return vextq_p16(a, b, 3);
}
