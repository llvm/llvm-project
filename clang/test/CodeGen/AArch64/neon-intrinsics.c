// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:     -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg \
// RUN: | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: @test_vadd_s8(
// CHECK:   [[ADD_I:%.*]] = add <8 x i8> %v1, %v2
// CHECK:   ret <8 x i8> [[ADD_I]]
int8x8_t test_vadd_s8(int8x8_t v1, int8x8_t v2) {
  return vadd_s8(v1, v2);
}

// CHECK-LABEL: @test_vadd_s16(
// CHECK:   [[ADD_I:%.*]] = add <4 x i16> %v1, %v2
// CHECK:   ret <4 x i16> [[ADD_I]]
int16x4_t test_vadd_s16(int16x4_t v1, int16x4_t v2) {
  return vadd_s16(v1, v2);
}

// CHECK-LABEL: @test_vadd_s32(
// CHECK:   [[ADD_I:%.*]] = add <2 x i32> %v1, %v2
// CHECK:   ret <2 x i32> [[ADD_I]]
int32x2_t test_vadd_s32(int32x2_t v1, int32x2_t v2) {
  return vadd_s32(v1, v2);
}

// CHECK-LABEL: @test_vadd_s64(
// CHECK:   [[ADD_I:%.*]] = add <1 x i64> %v1, %v2
// CHECK:   ret <1 x i64> [[ADD_I]]
int64x1_t test_vadd_s64(int64x1_t v1, int64x1_t v2) {
  return vadd_s64(v1, v2);
}

// CHECK-LABEL: @test_vadd_f32(
// CHECK:   [[ADD_I:%.*]] = fadd <2 x float> %v1, %v2
// CHECK:   ret <2 x float> [[ADD_I]]
float32x2_t test_vadd_f32(float32x2_t v1, float32x2_t v2) {
  return vadd_f32(v1, v2);
}

// CHECK-LABEL: @test_vadd_u8(
// CHECK:   [[ADD_I:%.*]] = add <8 x i8> %v1, %v2
// CHECK:   ret <8 x i8> [[ADD_I]]
uint8x8_t test_vadd_u8(uint8x8_t v1, uint8x8_t v2) {
  return vadd_u8(v1, v2);
}

// CHECK-LABEL: @test_vadd_u16(
// CHECK:   [[ADD_I:%.*]] = add <4 x i16> %v1, %v2
// CHECK:   ret <4 x i16> [[ADD_I]]
uint16x4_t test_vadd_u16(uint16x4_t v1, uint16x4_t v2) {
  return vadd_u16(v1, v2);
}

// CHECK-LABEL: @test_vadd_u32(
// CHECK:   [[ADD_I:%.*]] = add <2 x i32> %v1, %v2
// CHECK:   ret <2 x i32> [[ADD_I]]
uint32x2_t test_vadd_u32(uint32x2_t v1, uint32x2_t v2) {
  return vadd_u32(v1, v2);
}

// CHECK-LABEL: @test_vadd_u64(
// CHECK:   [[ADD_I:%.*]] = add <1 x i64> %v1, %v2
// CHECK:   ret <1 x i64> [[ADD_I]]
uint64x1_t test_vadd_u64(uint64x1_t v1, uint64x1_t v2) {
  return vadd_u64(v1, v2);
}

// CHECK-LABEL: @test_vaddq_s8(
// CHECK:   [[ADD_I:%.*]] = add <16 x i8> %v1, %v2
// CHECK:   ret <16 x i8> [[ADD_I]]
int8x16_t test_vaddq_s8(int8x16_t v1, int8x16_t v2) {
  return vaddq_s8(v1, v2);
}

// CHECK-LABEL: @test_vaddq_s16(
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %v1, %v2
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vaddq_s16(int16x8_t v1, int16x8_t v2) {
  return vaddq_s16(v1, v2);
}

// CHECK-LABEL: @test_vaddq_s32(
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %v1, %v2
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vaddq_s32(int32x4_t v1, int32x4_t v2) {
  return vaddq_s32(v1, v2);
}

// CHECK-LABEL: @test_vaddq_s64(
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %v1, %v2
// CHECK:   ret <2 x i64> [[ADD_I]]
int64x2_t test_vaddq_s64(int64x2_t v1, int64x2_t v2) {
  return vaddq_s64(v1, v2);
}

// CHECK-LABEL: @test_vaddq_f32(
// CHECK:   [[ADD_I:%.*]] = fadd <4 x float> %v1, %v2
// CHECK:   ret <4 x float> [[ADD_I]]
float32x4_t test_vaddq_f32(float32x4_t v1, float32x4_t v2) {
  return vaddq_f32(v1, v2);
}

// CHECK-LABEL: @test_vaddq_f64(
// CHECK:   [[ADD_I:%.*]] = fadd <2 x double> %v1, %v2
// CHECK:   ret <2 x double> [[ADD_I]]
float64x2_t test_vaddq_f64(float64x2_t v1, float64x2_t v2) {
  return vaddq_f64(v1, v2);
}

// CHECK-LABEL: @test_vaddq_u8(
// CHECK:   [[ADD_I:%.*]] = add <16 x i8> %v1, %v2
// CHECK:   ret <16 x i8> [[ADD_I]]
uint8x16_t test_vaddq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vaddq_u8(v1, v2);
}

// CHECK-LABEL: @test_vaddq_u16(
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %v1, %v2
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vaddq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vaddq_u16(v1, v2);
}

// CHECK-LABEL: @test_vaddq_u32(
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %v1, %v2
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vaddq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vaddq_u32(v1, v2);
}

// CHECK-LABEL: @test_vaddq_u64(
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %v1, %v2
// CHECK:   ret <2 x i64> [[ADD_I]]
uint64x2_t test_vaddq_u64(uint64x2_t v1, uint64x2_t v2) {
  return vaddq_u64(v1, v2);
}

// CHECK-LABEL: @test_vsub_s8(
// CHECK:   [[SUB_I:%.*]] = sub <8 x i8> %v1, %v2
// CHECK:   ret <8 x i8> [[SUB_I]]
int8x8_t test_vsub_s8(int8x8_t v1, int8x8_t v2) {
  return vsub_s8(v1, v2);
}

// CHECK-LABEL: @test_vsub_s16(
// CHECK:   [[SUB_I:%.*]] = sub <4 x i16> %v1, %v2
// CHECK:   ret <4 x i16> [[SUB_I]]
int16x4_t test_vsub_s16(int16x4_t v1, int16x4_t v2) {
  return vsub_s16(v1, v2);
}

// CHECK-LABEL: @test_vsub_s32(
// CHECK:   [[SUB_I:%.*]] = sub <2 x i32> %v1, %v2
// CHECK:   ret <2 x i32> [[SUB_I]]
int32x2_t test_vsub_s32(int32x2_t v1, int32x2_t v2) {
  return vsub_s32(v1, v2);
}

// CHECK-LABEL: @test_vsub_s64(
// CHECK:   [[SUB_I:%.*]] = sub <1 x i64> %v1, %v2
// CHECK:   ret <1 x i64> [[SUB_I]]
int64x1_t test_vsub_s64(int64x1_t v1, int64x1_t v2) {
  return vsub_s64(v1, v2);
}

// CHECK-LABEL: @test_vsub_f32(
// CHECK:   [[SUB_I:%.*]] = fsub <2 x float> %v1, %v2
// CHECK:   ret <2 x float> [[SUB_I]]
float32x2_t test_vsub_f32(float32x2_t v1, float32x2_t v2) {
  return vsub_f32(v1, v2);
}

// CHECK-LABEL: @test_vsub_u8(
// CHECK:   [[SUB_I:%.*]] = sub <8 x i8> %v1, %v2
// CHECK:   ret <8 x i8> [[SUB_I]]
uint8x8_t test_vsub_u8(uint8x8_t v1, uint8x8_t v2) {
  return vsub_u8(v1, v2);
}

// CHECK-LABEL: @test_vsub_u16(
// CHECK:   [[SUB_I:%.*]] = sub <4 x i16> %v1, %v2
// CHECK:   ret <4 x i16> [[SUB_I]]
uint16x4_t test_vsub_u16(uint16x4_t v1, uint16x4_t v2) {
  return vsub_u16(v1, v2);
}

// CHECK-LABEL: @test_vsub_u32(
// CHECK:   [[SUB_I:%.*]] = sub <2 x i32> %v1, %v2
// CHECK:   ret <2 x i32> [[SUB_I]]
uint32x2_t test_vsub_u32(uint32x2_t v1, uint32x2_t v2) {
  return vsub_u32(v1, v2);
}

// CHECK-LABEL: @test_vsub_u64(
// CHECK:   [[SUB_I:%.*]] = sub <1 x i64> %v1, %v2
// CHECK:   ret <1 x i64> [[SUB_I]]
uint64x1_t test_vsub_u64(uint64x1_t v1, uint64x1_t v2) {
  return vsub_u64(v1, v2);
}

// CHECK-LABEL: @test_vsubq_s8(
// CHECK:   [[SUB_I:%.*]] = sub <16 x i8> %v1, %v2
// CHECK:   ret <16 x i8> [[SUB_I]]
int8x16_t test_vsubq_s8(int8x16_t v1, int8x16_t v2) {
  return vsubq_s8(v1, v2);
}

// CHECK-LABEL: @test_vsubq_s16(
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %v1, %v2
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vsubq_s16(int16x8_t v1, int16x8_t v2) {
  return vsubq_s16(v1, v2);
}

// CHECK-LABEL: @test_vsubq_s32(
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %v1, %v2
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vsubq_s32(int32x4_t v1, int32x4_t v2) {
  return vsubq_s32(v1, v2);
}

// CHECK-LABEL: @test_vsubq_s64(
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %v1, %v2
// CHECK:   ret <2 x i64> [[SUB_I]]
int64x2_t test_vsubq_s64(int64x2_t v1, int64x2_t v2) {
  return vsubq_s64(v1, v2);
}

// CHECK-LABEL: @test_vsubq_f32(
// CHECK:   [[SUB_I:%.*]] = fsub <4 x float> %v1, %v2
// CHECK:   ret <4 x float> [[SUB_I]]
float32x4_t test_vsubq_f32(float32x4_t v1, float32x4_t v2) {
  return vsubq_f32(v1, v2);
}

// CHECK-LABEL: @test_vsubq_f64(
// CHECK:   [[SUB_I:%.*]] = fsub <2 x double> %v1, %v2
// CHECK:   ret <2 x double> [[SUB_I]]
float64x2_t test_vsubq_f64(float64x2_t v1, float64x2_t v2) {
  return vsubq_f64(v1, v2);
}

// CHECK-LABEL: @test_vsubq_u8(
// CHECK:   [[SUB_I:%.*]] = sub <16 x i8> %v1, %v2
// CHECK:   ret <16 x i8> [[SUB_I]]
uint8x16_t test_vsubq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vsubq_u8(v1, v2);
}

// CHECK-LABEL: @test_vsubq_u16(
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %v1, %v2
// CHECK:   ret <8 x i16> [[SUB_I]]
uint16x8_t test_vsubq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vsubq_u16(v1, v2);
}

// CHECK-LABEL: @test_vsubq_u32(
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %v1, %v2
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vsubq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vsubq_u32(v1, v2);
}

// CHECK-LABEL: @test_vsubq_u64(
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %v1, %v2
// CHECK:   ret <2 x i64> [[SUB_I]]
uint64x2_t test_vsubq_u64(uint64x2_t v1, uint64x2_t v2) {
  return vsubq_u64(v1, v2);
}

// CHECK-LABEL: @test_vmul_s8(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i8> %v1, %v2
// CHECK:   ret <8 x i8> [[MUL_I]]
int8x8_t test_vmul_s8(int8x8_t v1, int8x8_t v2) {
  return vmul_s8(v1, v2);
}

// CHECK-LABEL: @test_vmul_s16(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %v1, %v2
// CHECK:   ret <4 x i16> [[MUL_I]]
int16x4_t test_vmul_s16(int16x4_t v1, int16x4_t v2) {
  return vmul_s16(v1, v2);
}

// CHECK-LABEL: @test_vmul_s32(
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %v1, %v2
// CHECK:   ret <2 x i32> [[MUL_I]]
int32x2_t test_vmul_s32(int32x2_t v1, int32x2_t v2) {
  return vmul_s32(v1, v2);
}

// CHECK-LABEL: @test_vmul_f32(
// CHECK:   [[MUL_I:%.*]] = fmul <2 x float> %v1, %v2
// CHECK:   ret <2 x float> [[MUL_I]]
float32x2_t test_vmul_f32(float32x2_t v1, float32x2_t v2) {
  return vmul_f32(v1, v2);
}

// CHECK-LABEL: @test_vmul_u8(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i8> %v1, %v2
// CHECK:   ret <8 x i8> [[MUL_I]]
uint8x8_t test_vmul_u8(uint8x8_t v1, uint8x8_t v2) {
  return vmul_u8(v1, v2);
}

// CHECK-LABEL: @test_vmul_u16(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %v1, %v2
// CHECK:   ret <4 x i16> [[MUL_I]]
uint16x4_t test_vmul_u16(uint16x4_t v1, uint16x4_t v2) {
  return vmul_u16(v1, v2);
}

// CHECK-LABEL: @test_vmul_u32(
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %v1, %v2
// CHECK:   ret <2 x i32> [[MUL_I]]
uint32x2_t test_vmul_u32(uint32x2_t v1, uint32x2_t v2) {
  return vmul_u32(v1, v2);
}

// CHECK-LABEL: @test_vmulq_s8(
// CHECK:   [[MUL_I:%.*]] = mul <16 x i8> %v1, %v2
// CHECK:   ret <16 x i8> [[MUL_I]]
int8x16_t test_vmulq_s8(int8x16_t v1, int8x16_t v2) {
  return vmulq_s8(v1, v2);
}

// CHECK-LABEL: @test_vmulq_s16(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %v1, %v2
// CHECK:   ret <8 x i16> [[MUL_I]]
int16x8_t test_vmulq_s16(int16x8_t v1, int16x8_t v2) {
  return vmulq_s16(v1, v2);
}

// CHECK-LABEL: @test_vmulq_s32(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %v1, %v2
// CHECK:   ret <4 x i32> [[MUL_I]]
int32x4_t test_vmulq_s32(int32x4_t v1, int32x4_t v2) {
  return vmulq_s32(v1, v2);
}

// CHECK-LABEL: @test_vmulq_u8(
// CHECK:   [[MUL_I:%.*]] = mul <16 x i8> %v1, %v2
// CHECK:   ret <16 x i8> [[MUL_I]]
uint8x16_t test_vmulq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vmulq_u8(v1, v2);
}

// CHECK-LABEL: @test_vmulq_u16(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %v1, %v2
// CHECK:   ret <8 x i16> [[MUL_I]]
uint16x8_t test_vmulq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vmulq_u16(v1, v2);
}

// CHECK-LABEL: @test_vmulq_u32(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %v1, %v2
// CHECK:   ret <4 x i32> [[MUL_I]]
uint32x4_t test_vmulq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vmulq_u32(v1, v2);
}

// CHECK-LABEL: @test_vmulq_f32(
// CHECK:   [[MUL_I:%.*]] = fmul <4 x float> %v1, %v2
// CHECK:   ret <4 x float> [[MUL_I]]
float32x4_t test_vmulq_f32(float32x4_t v1, float32x4_t v2) {
  return vmulq_f32(v1, v2);
}

// CHECK-LABEL: @test_vmulq_f64(
// CHECK:   [[MUL_I:%.*]] = fmul <2 x double> %v1, %v2
// CHECK:   ret <2 x double> [[MUL_I]]
float64x2_t test_vmulq_f64(float64x2_t v1, float64x2_t v2) {
  return vmulq_f64(v1, v2);
}

// CHECK-LABEL: @test_vmul_p8(
// CHECK:   [[VMUL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.pmul.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VMUL_V_I]]
poly8x8_t test_vmul_p8(poly8x8_t v1, poly8x8_t v2) {
  return vmul_p8(v1, v2);
}

// CHECK-LABEL: @test_vmulq_p8(
// CHECK:   [[VMULQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.pmul.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VMULQ_V_I]]
poly8x16_t test_vmulq_p8(poly8x16_t v1, poly8x16_t v2) {
  return vmulq_p8(v1, v2);
}

// CHECK-LABEL: @test_vmla_s8(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i8> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <8 x i8> %v1, [[MUL_I]]
// CHECK:   ret <8 x i8> [[ADD_I]]
int8x8_t test_vmla_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  return vmla_s8(v1, v2, v3);
}

// CHECK-LABEL: @test_vmla_s16(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <4 x i16> %v1, [[MUL_I]]
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[ADD_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vmla_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  return (int8x8_t)vmla_s16(v1, v2, v3);
}

// CHECK-LABEL: @test_vmla_s32(
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <2 x i32> %v1, [[MUL_I]]
// CHECK:   ret <2 x i32> [[ADD_I]]
int32x2_t test_vmla_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  return vmla_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmla_f32(
// CHECK:   [[MUL_I:%.*]] = fmul <2 x float> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = fadd <2 x float> %v1, [[MUL_I]]
// CHECK:   ret <2 x float> [[ADD_I]]
float32x2_t test_vmla_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vmla_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmla_u8(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i8> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <8 x i8> %v1, [[MUL_I]]
// CHECK:   ret <8 x i8> [[ADD_I]]
uint8x8_t test_vmla_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  return vmla_u8(v1, v2, v3);
}

// CHECK-LABEL: @test_vmla_u16(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <4 x i16> %v1, [[MUL_I]]
// CHECK:   ret <4 x i16> [[ADD_I]]
uint16x4_t test_vmla_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  return vmla_u16(v1, v2, v3);
}

// CHECK-LABEL: @test_vmla_u32(
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <2 x i32> %v1, [[MUL_I]]
// CHECK:   ret <2 x i32> [[ADD_I]]
uint32x2_t test_vmla_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  return vmla_u32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlaq_s8(
// CHECK:   [[MUL_I:%.*]] = mul <16 x i8> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <16 x i8> %v1, [[MUL_I]]
// CHECK:   ret <16 x i8> [[ADD_I]]
int8x16_t test_vmlaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  return vmlaq_s8(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlaq_s16(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %v1, [[MUL_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vmlaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  return vmlaq_s16(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlaq_s32(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %v1, [[MUL_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vmlaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  return vmlaq_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlaq_f32(
// CHECK:   [[MUL_I:%.*]] = fmul <4 x float> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = fadd <4 x float> %v1, [[MUL_I]]
// CHECK:   ret <4 x float> [[ADD_I]]
float32x4_t test_vmlaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vmlaq_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlaq_u8(
// CHECK:   [[MUL_I:%.*]] = mul <16 x i8> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <16 x i8> %v1, [[MUL_I]]
// CHECK:   ret <16 x i8> [[ADD_I]]
uint8x16_t test_vmlaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  return vmlaq_u8(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlaq_u16(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %v1, [[MUL_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vmlaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  return vmlaq_u16(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlaq_u32(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %v1, [[MUL_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vmlaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  return vmlaq_u32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlaq_f64(
// CHECK:   [[MUL_I:%.*]] = fmul <2 x double> %v2, %v3
// CHECK:   [[ADD_I:%.*]] = fadd <2 x double> %v1, [[MUL_I]]
// CHECK:   ret <2 x double> [[ADD_I]]
float64x2_t test_vmlaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vmlaq_f64(v1, v2, v3);
}

// CHECK-LABEL: @test_vmls_s8(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i8> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <8 x i8> %v1, [[MUL_I]]
// CHECK:   ret <8 x i8> [[SUB_I]]
int8x8_t test_vmls_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  return vmls_s8(v1, v2, v3);
}

// CHECK-LABEL: @test_vmls_s16(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <4 x i16> %v1, [[MUL_I]]
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SUB_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vmls_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  return (int8x8_t)vmls_s16(v1, v2, v3);
}

// CHECK-LABEL: @test_vmls_s32(
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <2 x i32> %v1, [[MUL_I]]
// CHECK:   ret <2 x i32> [[SUB_I]]
int32x2_t test_vmls_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  return vmls_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmls_f32(
// CHECK:   [[MUL_I:%.*]] = fmul <2 x float> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = fsub <2 x float> %v1, [[MUL_I]]
// CHECK:   ret <2 x float> [[SUB_I]]
float32x2_t test_vmls_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vmls_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmls_u8(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i8> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <8 x i8> %v1, [[MUL_I]]
// CHECK:   ret <8 x i8> [[SUB_I]]
uint8x8_t test_vmls_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  return vmls_u8(v1, v2, v3);
}

// CHECK-LABEL: @test_vmls_u16(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <4 x i16> %v1, [[MUL_I]]
// CHECK:   ret <4 x i16> [[SUB_I]]
uint16x4_t test_vmls_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  return vmls_u16(v1, v2, v3);
}

// CHECK-LABEL: @test_vmls_u32(
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <2 x i32> %v1, [[MUL_I]]
// CHECK:   ret <2 x i32> [[SUB_I]]
uint32x2_t test_vmls_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  return vmls_u32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlsq_s8(
// CHECK:   [[MUL_I:%.*]] = mul <16 x i8> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <16 x i8> %v1, [[MUL_I]]
// CHECK:   ret <16 x i8> [[SUB_I]]
int8x16_t test_vmlsq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  return vmlsq_s8(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlsq_s16(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %v1, [[MUL_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vmlsq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  return vmlsq_s16(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlsq_s32(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %v1, [[MUL_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vmlsq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  return vmlsq_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlsq_f32(
// CHECK:   [[MUL_I:%.*]] = fmul <4 x float> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = fsub <4 x float> %v1, [[MUL_I]]
// CHECK:   ret <4 x float> [[SUB_I]]
float32x4_t test_vmlsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vmlsq_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlsq_u8(
// CHECK:   [[MUL_I:%.*]] = mul <16 x i8> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <16 x i8> %v1, [[MUL_I]]
// CHECK:   ret <16 x i8> [[SUB_I]]
uint8x16_t test_vmlsq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  return vmlsq_u8(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlsq_u16(
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %v1, [[MUL_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
uint16x8_t test_vmlsq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  return vmlsq_u16(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlsq_u32(
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %v1, [[MUL_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vmlsq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  return vmlsq_u32(v1, v2, v3);
}

// CHECK-LABEL: @test_vmlsq_f64(
// CHECK:   [[MUL_I:%.*]] = fmul <2 x double> %v2, %v3
// CHECK:   [[SUB_I:%.*]] = fsub <2 x double> %v1, [[MUL_I]]
// CHECK:   ret <2 x double> [[SUB_I]]
float64x2_t test_vmlsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vmlsq_f64(v1, v2, v3);
}

// CHECK-LABEL: @test_vfma_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v3 to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> %v2, <2 x float> %v3, <2 x float> %v1)
// CHECK:   ret <2 x float> [[TMP3]]
float32x2_t test_vfma_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vfma_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vfmaq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v3 to <16 x i8>
// CHECK:   [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %v2, <4 x float> %v3, <4 x float> %v1)
// CHECK:   ret <4 x float> [[TMP3]]
float32x4_t test_vfmaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vfmaq_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vfmaq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v3 to <16 x i8>
// CHECK:   [[TMP3:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %v2, <2 x double> %v3, <2 x double> %v1)
// CHECK:   ret <2 x double> [[TMP3]]
float64x2_t test_vfmaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vfmaq_f64(v1, v2, v3);
}

// CHECK-LABEL: @test_vfms_f32(
// CHECK:   [[SUB_I:%.*]] = fneg <2 x float> %v2
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SUB_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v3 to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[SUB_I]], <2 x float> %v3, <2 x float> %v1)
// CHECK:   ret <2 x float> [[TMP3]]
float32x2_t test_vfms_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vfms_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vfmsq_f32(
// CHECK:   [[SUB_I:%.*]] = fneg <4 x float> %v2
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SUB_I]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v3 to <16 x i8>
// CHECK:   [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[SUB_I]], <4 x float> %v3, <4 x float> %v1)
// CHECK:   ret <4 x float> [[TMP3]]
float32x4_t test_vfmsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vfmsq_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vfmsq_f64(
// CHECK:   [[SUB_I:%.*]] = fneg <2 x double> %v2
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> [[SUB_I]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v3 to <16 x i8>
// CHECK:   [[TMP3:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[SUB_I]], <2 x double> %v3, <2 x double> %v1)
// CHECK:   ret <2 x double> [[TMP3]]
float64x2_t test_vfmsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vfmsq_f64(v1, v2, v3);
}

// CHECK-LABEL: @test_vdivq_f64(
// CHECK:   [[DIV_I:%.*]] = fdiv <2 x double> %v1, %v2
// CHECK:   ret <2 x double> [[DIV_I]]
float64x2_t test_vdivq_f64(float64x2_t v1, float64x2_t v2) {
  return vdivq_f64(v1, v2);
}

// CHECK-LABEL: @test_vdivq_f32(
// CHECK:   [[DIV_I:%.*]] = fdiv <4 x float> %v1, %v2
// CHECK:   ret <4 x float> [[DIV_I]]
float32x4_t test_vdivq_f32(float32x4_t v1, float32x4_t v2) {
  return vdivq_f32(v1, v2);
}

// CHECK-LABEL: @test_vdiv_f32(
// CHECK:   [[DIV_I:%.*]] = fdiv <2 x float> %v1, %v2
// CHECK:   ret <2 x float> [[DIV_I]]
float32x2_t test_vdiv_f32(float32x2_t v1, float32x2_t v2) {
  return vdiv_f32(v1, v2);
}

// CHECK-LABEL: @test_vaba_s8(
// CHECK:   [[VABD_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %v2, <8 x i8> %v3)
// CHECK:   [[ADD_I:%.*]] = add <8 x i8> %v1, [[VABD_I_I]]
// CHECK:   ret <8 x i8> [[ADD_I]]
int8x8_t test_vaba_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  return vaba_s8(v1, v2, v3);
}

// CHECK-LABEL: @test_vaba_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v3 to <8 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %v2, <4 x i16> %v3)
// CHECK:   [[ADD_I:%.*]] = add <4 x i16> %v1, [[VABD2_I_I]]
// CHECK:   ret <4 x i16> [[ADD_I]]
int16x4_t test_vaba_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  return vaba_s16(v1, v2, v3);
}

// CHECK-LABEL: @test_vaba_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v3 to <8 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %v2, <2 x i32> %v3)
// CHECK:   [[ADD_I:%.*]] = add <2 x i32> %v1, [[VABD2_I_I]]
// CHECK:   ret <2 x i32> [[ADD_I]]
int32x2_t test_vaba_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  return vaba_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vaba_u8(
// CHECK:   [[VABD_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %v2, <8 x i8> %v3)
// CHECK:   [[ADD_I:%.*]] = add <8 x i8> %v1, [[VABD_I_I]]
// CHECK:   ret <8 x i8> [[ADD_I]]
uint8x8_t test_vaba_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  return vaba_u8(v1, v2, v3);
}

// CHECK-LABEL: @test_vaba_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v3 to <8 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %v2, <4 x i16> %v3)
// CHECK:   [[ADD_I:%.*]] = add <4 x i16> %v1, [[VABD2_I_I]]
// CHECK:   ret <4 x i16> [[ADD_I]]
uint16x4_t test_vaba_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  return vaba_u16(v1, v2, v3);
}

// CHECK-LABEL: @test_vaba_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v3 to <8 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %v2, <2 x i32> %v3)
// CHECK:   [[ADD_I:%.*]] = add <2 x i32> %v1, [[VABD2_I_I]]
// CHECK:   ret <2 x i32> [[ADD_I]]
uint32x2_t test_vaba_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  return vaba_u32(v1, v2, v3);
}

// CHECK-LABEL: @test_vabaq_s8(
// CHECK:   [[VABD_I_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sabd.v16i8(<16 x i8> %v2, <16 x i8> %v3)
// CHECK:   [[ADD_I:%.*]] = add <16 x i8> %v1, [[VABD_I_I]]
// CHECK:   ret <16 x i8> [[ADD_I]]
int8x16_t test_vabaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  return vabaq_s8(v1, v2, v3);
}

// CHECK-LABEL: @test_vabaq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v3 to <16 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sabd.v8i16(<8 x i16> %v2, <8 x i16> %v3)
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %v1, [[VABD2_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vabaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  return vabaq_s16(v1, v2, v3);
}

// CHECK-LABEL: @test_vabaq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v3 to <16 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sabd.v4i32(<4 x i32> %v2, <4 x i32> %v3)
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %v1, [[VABD2_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vabaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  return vabaq_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vabaq_u8(
// CHECK:   [[VABD_I_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uabd.v16i8(<16 x i8> %v2, <16 x i8> %v3)
// CHECK:   [[ADD_I:%.*]] = add <16 x i8> %v1, [[VABD_I_I]]
// CHECK:   ret <16 x i8> [[ADD_I]]
uint8x16_t test_vabaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  return vabaq_u8(v1, v2, v3);
}

// CHECK-LABEL: @test_vabaq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v3 to <16 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> %v2, <8 x i16> %v3)
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %v1, [[VABD2_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vabaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  return vabaq_u16(v1, v2, v3);
}

// CHECK-LABEL: @test_vabaq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v3 to <16 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uabd.v4i32(<4 x i32> %v2, <4 x i32> %v3)
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %v1, [[VABD2_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vabaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  return vabaq_u32(v1, v2, v3);
}

// CHECK-LABEL: @test_vabd_s8(
// CHECK:   [[VABD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VABD_I]]
int8x8_t test_vabd_s8(int8x8_t v1, int8x8_t v2) {
  return vabd_s8(v1, v2);
}

// CHECK-LABEL: @test_vabd_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %v1, <4 x i16> %v2)
// CHECK:   ret <4 x i16> [[VABD2_I]]
int16x4_t test_vabd_s16(int16x4_t v1, int16x4_t v2) {
  return vabd_s16(v1, v2);
}

// CHECK-LABEL: @test_vabd_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %v1, <2 x i32> %v2)
// CHECK:   ret <2 x i32> [[VABD2_I]]
int32x2_t test_vabd_s32(int32x2_t v1, int32x2_t v2) {
  return vabd_s32(v1, v2);
}

// CHECK-LABEL: @test_vabd_u8(
// CHECK:   [[VABD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VABD_I]]
uint8x8_t test_vabd_u8(uint8x8_t v1, uint8x8_t v2) {
  return vabd_u8(v1, v2);
}

// CHECK-LABEL: @test_vabd_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %v1, <4 x i16> %v2)
// CHECK:   ret <4 x i16> [[VABD2_I]]
uint16x4_t test_vabd_u16(uint16x4_t v1, uint16x4_t v2) {
  return vabd_u16(v1, v2);
}

// CHECK-LABEL: @test_vabd_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %v1, <2 x i32> %v2)
// CHECK:   ret <2 x i32> [[VABD2_I]]
uint32x2_t test_vabd_u32(uint32x2_t v1, uint32x2_t v2) {
  return vabd_u32(v1, v2);
}

// CHECK-LABEL: @test_vabd_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fabd.v2f32(<2 x float> %v1, <2 x float> %v2)
// CHECK:   ret <2 x float> [[VABD2_I]]
float32x2_t test_vabd_f32(float32x2_t v1, float32x2_t v2) {
  return vabd_f32(v1, v2);
}

// CHECK-LABEL: @test_vabdq_s8(
// CHECK:   [[VABD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sabd.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VABD_I]]
int8x16_t test_vabdq_s8(int8x16_t v1, int8x16_t v2) {
  return vabdq_s8(v1, v2);
}

// CHECK-LABEL: @test_vabdq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sabd.v8i16(<8 x i16> %v1, <8 x i16> %v2)
// CHECK:   ret <8 x i16> [[VABD2_I]]
int16x8_t test_vabdq_s16(int16x8_t v1, int16x8_t v2) {
  return vabdq_s16(v1, v2);
}

// CHECK-LABEL: @test_vabdq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sabd.v4i32(<4 x i32> %v1, <4 x i32> %v2)
// CHECK:   ret <4 x i32> [[VABD2_I]]
int32x4_t test_vabdq_s32(int32x4_t v1, int32x4_t v2) {
  return vabdq_s32(v1, v2);
}

// CHECK-LABEL: @test_vabdq_u8(
// CHECK:   [[VABD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uabd.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VABD_I]]
uint8x16_t test_vabdq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vabdq_u8(v1, v2);
}

// CHECK-LABEL: @test_vabdq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> %v1, <8 x i16> %v2)
// CHECK:   ret <8 x i16> [[VABD2_I]]
uint16x8_t test_vabdq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vabdq_u16(v1, v2);
}

// CHECK-LABEL: @test_vabdq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uabd.v4i32(<4 x i32> %v1, <4 x i32> %v2)
// CHECK:   ret <4 x i32> [[VABD2_I]]
uint32x4_t test_vabdq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vabdq_u32(v1, v2);
}

// CHECK-LABEL: @test_vabdq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fabd.v4f32(<4 x float> %v1, <4 x float> %v2)
// CHECK:   ret <4 x float> [[VABD2_I]]
float32x4_t test_vabdq_f32(float32x4_t v1, float32x4_t v2) {
  return vabdq_f32(v1, v2);
}

// CHECK-LABEL: @test_vabdq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fabd.v2f64(<2 x double> %v1, <2 x double> %v2)
// CHECK:   ret <2 x double> [[VABD2_I]]
float64x2_t test_vabdq_f64(float64x2_t v1, float64x2_t v2) {
  return vabdq_f64(v1, v2);
}

// CHECK-LABEL: @test_vbsl_s8(
// CHECK:   [[VBSL_I:%.*]] = and <8 x i8> %v1, %v2
// CHECK:   [[TMP0:%.*]] = xor <8 x i8> %v1, splat (i8 -1)
// CHECK:   [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], %v3
// CHECK:   [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
// CHECK:   ret <8 x i8> [[VBSL2_I]]
int8x8_t test_vbsl_s8(uint8x8_t v1, int8x8_t v2, int8x8_t v3) {
  return vbsl_s8(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %v3 to <8 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <4 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <4 x i16> %v1, splat (i16 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[VBSL5_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP4]]
int8x8_t test_vbsl_s16(uint16x4_t v1, int16x4_t v2, int16x4_t v3) {
  return (int8x8_t)vbsl_s16(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %v3 to <8 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <2 x i32> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <2 x i32> %v1, splat (i32 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <2 x i32> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <2 x i32> [[VBSL5_I]]
int32x2_t test_vbsl_s32(uint32x2_t v1, int32x2_t v2, int32x2_t v3) {
  return vbsl_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x i64> %v3 to <8 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <1 x i64> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <1 x i64> %v1, splat (i64 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <1 x i64> [[VBSL5_I]]
int64x1_t test_vbsl_s64(uint64x1_t v1, int64x1_t v2, int64x1_t v3) {
  return vbsl_s64(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_u8(
// CHECK:   [[VBSL_I:%.*]] = and <8 x i8> %v1, %v2
// CHECK:   [[TMP0:%.*]] = xor <8 x i8> %v1, splat (i8 -1)
// CHECK:   [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], %v3
// CHECK:   [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
// CHECK:   ret <8 x i8> [[VBSL2_I]]
uint8x8_t test_vbsl_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  return vbsl_u8(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %v3 to <8 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <4 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <4 x i16> %v1, splat (i16 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <4 x i16> [[VBSL5_I]]
uint16x4_t test_vbsl_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  return vbsl_u16(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %v3 to <8 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <2 x i32> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <2 x i32> %v1, splat (i32 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <2 x i32> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <2 x i32> [[VBSL5_I]]
uint32x2_t test_vbsl_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  return vbsl_u32(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x i64> %v3 to <8 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <1 x i64> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <1 x i64> %v1, splat (i64 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <1 x i64> [[VBSL5_I]]
uint64x1_t test_vbsl_u64(uint64x1_t v1, uint64x1_t v2, uint64x1_t v3) {
  return vbsl_u64(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_f32(
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <2 x float> %v3 to <8 x i8>
// CHECK:   [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
// CHECK:   [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x i32>
// CHECK:   [[VBSL3_I:%.*]] = and <2 x i32> %v1, [[VBSL1_I]]
// CHECK:   [[TMP4:%.*]] = xor <2 x i32> %v1, splat (i32 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <2 x i32> [[TMP4]], [[VBSL2_I]]
// CHECK:   [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   [[TMP5:%.*]] = bitcast <2 x i32> [[VBSL5_I]] to <2 x float>
// CHECK:   ret <2 x float> [[TMP5]]
float32x2_t test_vbsl_f32(uint32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vbsl_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> %v3 to <8 x i8>
// CHECK:   [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x i64>
// CHECK:   [[VBSL3_I:%.*]] = and <1 x i64> %v1, [[VBSL1_I]]
// CHECK:   [[TMP3:%.*]] = xor <1 x i64> %v1, splat (i64 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], [[VBSL2_I]]
// CHECK:   [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[VBSL5_I]] to <1 x double>
// CHECK:   ret <1 x double> [[TMP4]]
float64x1_t test_vbsl_f64(uint64x1_t v1, float64x1_t v2, float64x1_t v3) {
  return vbsl_f64(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_p8(
// CHECK:   [[VBSL_I:%.*]] = and <8 x i8> %v1, %v2
// CHECK:   [[TMP0:%.*]] = xor <8 x i8> %v1, splat (i8 -1)
// CHECK:   [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], %v3
// CHECK:   [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
// CHECK:   ret <8 x i8> [[VBSL2_I]]
poly8x8_t test_vbsl_p8(uint8x8_t v1, poly8x8_t v2, poly8x8_t v3) {
  return vbsl_p8(v1, v2, v3);
}

// CHECK-LABEL: @test_vbsl_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %v3 to <8 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <4 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <4 x i16> %v1, splat (i16 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <4 x i16> [[VBSL5_I]]
poly16x4_t test_vbsl_p16(uint16x4_t v1, poly16x4_t v2, poly16x4_t v3) {
  return vbsl_p16(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_s8(
// CHECK:   [[VBSL_I:%.*]] = and <16 x i8> %v1, %v2
// CHECK:   [[TMP0:%.*]] = xor <16 x i8> %v1, splat (i8 -1)
// CHECK:   [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], %v3
// CHECK:   [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
// CHECK:   ret <16 x i8> [[VBSL2_I]]
int8x16_t test_vbslq_s8(uint8x16_t v1, int8x16_t v2, int8x16_t v3) {
  return vbslq_s8(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %v3 to <16 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <8 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <8 x i16> %v1, splat (i16 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <8 x i16> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <8 x i16> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <8 x i16> [[VBSL5_I]]
int16x8_t test_vbslq_s16(uint16x8_t v1, int16x8_t v2, int16x8_t v3) {
  return vbslq_s16(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %v3 to <16 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <4 x i32> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <4 x i32> %v1, splat (i32 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <4 x i32> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <4 x i32> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <4 x i32> [[VBSL5_I]]
int32x4_t test_vbslq_s32(uint32x4_t v1, int32x4_t v2, int32x4_t v3) {
  return vbslq_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i64> %v3 to <16 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <2 x i64> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <2 x i64> %v1, splat (i64 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <2 x i64> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <2 x i64> [[VBSL5_I]]
int64x2_t test_vbslq_s64(uint64x2_t v1, int64x2_t v2, int64x2_t v3) {
  return vbslq_s64(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_u8(
// CHECK:   [[VBSL_I:%.*]] = and <16 x i8> %v1, %v2
// CHECK:   [[TMP0:%.*]] = xor <16 x i8> %v1, splat (i8 -1)
// CHECK:   [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], %v3
// CHECK:   [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
// CHECK:   ret <16 x i8> [[VBSL2_I]]
uint8x16_t test_vbslq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  return vbslq_u8(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %v3 to <16 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <8 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <8 x i16> %v1, splat (i16 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <8 x i16> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <8 x i16> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <8 x i16> [[VBSL5_I]]
uint16x8_t test_vbslq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  return vbslq_u16(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i32> %v3 to <16 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <4 x i32> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <4 x i32> %v1, splat (i32 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <4 x i32> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <4 x i32> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <4 x i32> [[VBSL5_I]]
int32x4_t test_vbslq_u32(uint32x4_t v1, int32x4_t v2, int32x4_t v3) {
  return vbslq_s32(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i64> %v3 to <16 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <2 x i64> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <2 x i64> %v1, splat (i64 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <2 x i64> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <2 x i64> [[VBSL5_I]]
uint64x2_t test_vbslq_u64(uint64x2_t v1, uint64x2_t v2, uint64x2_t v3) {
  return vbslq_u64(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v3 to <16 x i8>
// CHECK:   [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
// CHECK:   [[VBSL3_I:%.*]] = and <4 x i32> %v1, [[VBSL1_I]]
// CHECK:   [[TMP3:%.*]] = xor <4 x i32> %v1, splat (i32 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <4 x i32> [[TMP3]], [[VBSL2_I]]
// CHECK:   [[VBSL5_I:%.*]] = or <4 x i32> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[VBSL5_I]] to <4 x float>
// CHECK:   ret <4 x float> [[TMP4]]
float32x4_t test_vbslq_f32(uint32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vbslq_f32(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_p8(
// CHECK:   [[VBSL_I:%.*]] = and <16 x i8> %v1, %v2
// CHECK:   [[TMP0:%.*]] = xor <16 x i8> %v1, splat (i8 -1)
// CHECK:   [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], %v3
// CHECK:   [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
// CHECK:   ret <16 x i8> [[VBSL2_I]]
poly8x16_t test_vbslq_p8(uint8x16_t v1, poly8x16_t v2, poly8x16_t v3) {
  return vbslq_p8(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i16> %v3 to <16 x i8>
// CHECK:   [[VBSL3_I:%.*]] = and <8 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = xor <8 x i16> %v1, splat (i16 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <8 x i16> [[TMP3]], %v3
// CHECK:   [[VBSL5_I:%.*]] = or <8 x i16> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <8 x i16> [[VBSL5_I]]
poly16x8_t test_vbslq_p16(uint16x8_t v1, poly16x8_t v2, poly16x8_t v3) {
  return vbslq_p16(v1, v2, v3);
}

// CHECK-LABEL: @test_vbslq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v3 to <16 x i8>
// CHECK:   [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x i64>
// CHECK:   [[VBSL3_I:%.*]] = and <2 x i64> %v1, [[VBSL1_I]]
// CHECK:   [[TMP3:%.*]] = xor <2 x i64> %v1, splat (i64 -1)
// CHECK:   [[VBSL4_I:%.*]] = and <2 x i64> [[TMP3]], [[VBSL2_I]]
// CHECK:   [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[VBSL5_I]] to <2 x double>
// CHECK:   ret <2 x double> [[TMP4]]
float64x2_t test_vbslq_f64(uint64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vbslq_f64(v1, v2, v3);
}

// CHECK-LABEL: @test_vrecps_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[VRECPS_V2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frecps.v2f32(<2 x float> %v1, <2 x float> %v2)
// CHECK:   ret <2 x float> [[VRECPS_V2_I]]
float32x2_t test_vrecps_f32(float32x2_t v1, float32x2_t v2) {
  return vrecps_f32(v1, v2);
}

// CHECK-LABEL: @test_vrecpsq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[VRECPSQ_V2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frecps.v4f32(<4 x float> %v1, <4 x float> %v2)
// CHECK:   [[VRECPSQ_V3_I:%.*]] = bitcast <4 x float> [[VRECPSQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x float> [[VRECPSQ_V2_I]]
float32x4_t test_vrecpsq_f32(float32x4_t v1, float32x4_t v2) {
  return vrecpsq_f32(v1, v2);
}

// CHECK-LABEL: @test_vrecpsq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[VRECPSQ_V2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.frecps.v2f64(<2 x double> %v1, <2 x double> %v2)
// CHECK:   [[VRECPSQ_V3_I:%.*]] = bitcast <2 x double> [[VRECPSQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x double> [[VRECPSQ_V2_I]]
float64x2_t test_vrecpsq_f64(float64x2_t v1, float64x2_t v2) {
  return vrecpsq_f64(v1, v2);
}

// CHECK-LABEL: @test_vrsqrts_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[VRSQRTS_V2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frsqrts.v2f32(<2 x float> %v1, <2 x float> %v2)
// CHECK:   [[VRSQRTS_V3_I:%.*]] = bitcast <2 x float> [[VRSQRTS_V2_I]] to <8 x i8>
// CHECK:   ret <2 x float> [[VRSQRTS_V2_I]]
float32x2_t test_vrsqrts_f32(float32x2_t v1, float32x2_t v2) {
  return vrsqrts_f32(v1, v2);
}

// CHECK-LABEL: @test_vrsqrtsq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[VRSQRTSQ_V2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frsqrts.v4f32(<4 x float> %v1, <4 x float> %v2)
// CHECK:   [[VRSQRTSQ_V3_I:%.*]] = bitcast <4 x float> [[VRSQRTSQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x float> [[VRSQRTSQ_V2_I]]
float32x4_t test_vrsqrtsq_f32(float32x4_t v1, float32x4_t v2) {
  return vrsqrtsq_f32(v1, v2);
}

// CHECK-LABEL: @test_vrsqrtsq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[VRSQRTSQ_V2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.frsqrts.v2f64(<2 x double> %v1, <2 x double> %v2)
// CHECK:   [[VRSQRTSQ_V3_I:%.*]] = bitcast <2 x double> [[VRSQRTSQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x double> [[VRSQRTSQ_V2_I]]
float64x2_t test_vrsqrtsq_f64(float64x2_t v1, float64x2_t v2) {
  return vrsqrtsq_f64(v1, v2);
}

// CHECK-LABEL: @test_vcage_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[VCAGE_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.facge.v2i32.v2f32(<2 x float> %v1, <2 x float> %v2)
// CHECK:   ret <2 x i32> [[VCAGE_V2_I]]
uint32x2_t test_vcage_f32(float32x2_t v1, float32x2_t v2) {
  return vcage_f32(v1, v2);
}

// CHECK-LABEL: @test_vcage_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VCAGE_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.facge.v1i64.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x i64> [[VCAGE_V2_I]]
uint64x1_t test_vcage_f64(float64x1_t a, float64x1_t b) {
  return vcage_f64(a, b);
}

// CHECK-LABEL: @test_vcageq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[VCAGEQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.facge.v4i32.v4f32(<4 x float> %v1, <4 x float> %v2)
// CHECK:   ret <4 x i32> [[VCAGEQ_V2_I]]
uint32x4_t test_vcageq_f32(float32x4_t v1, float32x4_t v2) {
  return vcageq_f32(v1, v2);
}

// CHECK-LABEL: @test_vcageq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[VCAGEQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.facge.v2i64.v2f64(<2 x double> %v1, <2 x double> %v2)
// CHECK:   ret <2 x i64> [[VCAGEQ_V2_I]]
uint64x2_t test_vcageq_f64(float64x2_t v1, float64x2_t v2) {
  return vcageq_f64(v1, v2);
}

// CHECK-LABEL: @test_vcagt_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[VCAGT_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.facgt.v2i32.v2f32(<2 x float> %v1, <2 x float> %v2)
// CHECK:   ret <2 x i32> [[VCAGT_V2_I]]
uint32x2_t test_vcagt_f32(float32x2_t v1, float32x2_t v2) {
  return vcagt_f32(v1, v2);
}

// CHECK-LABEL: @test_vcagt_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VCAGT_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.facgt.v1i64.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x i64> [[VCAGT_V2_I]]
uint64x1_t test_vcagt_f64(float64x1_t a, float64x1_t b) {
  return vcagt_f64(a, b);
}

// CHECK-LABEL: @test_vcagtq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[VCAGTQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.facgt.v4i32.v4f32(<4 x float> %v1, <4 x float> %v2)
// CHECK:   ret <4 x i32> [[VCAGTQ_V2_I]]
uint32x4_t test_vcagtq_f32(float32x4_t v1, float32x4_t v2) {
  return vcagtq_f32(v1, v2);
}

// CHECK-LABEL: @test_vcagtq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[VCAGTQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.facgt.v2i64.v2f64(<2 x double> %v1, <2 x double> %v2)
// CHECK:   ret <2 x i64> [[VCAGTQ_V2_I]]
uint64x2_t test_vcagtq_f64(float64x2_t v1, float64x2_t v2) {
  return vcagtq_f64(v1, v2);
}

// CHECK-LABEL: @test_vcale_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[VCALE_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.facge.v2i32.v2f32(<2 x float> %v2, <2 x float> %v1)
// CHECK:   ret <2 x i32> [[VCALE_V2_I]]
uint32x2_t test_vcale_f32(float32x2_t v1, float32x2_t v2) {
  return vcale_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
}

// CHECK-LABEL: @test_vcale_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VCALE_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.facge.v1i64.v1f64(<1 x double> %b, <1 x double> %a)
// CHECK:   ret <1 x i64> [[VCALE_V2_I]]
uint64x1_t test_vcale_f64(float64x1_t a, float64x1_t b) {
  return vcale_f64(a, b);
}

// CHECK-LABEL: @test_vcaleq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[VCALEQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.facge.v4i32.v4f32(<4 x float> %v2, <4 x float> %v1)
// CHECK:   ret <4 x i32> [[VCALEQ_V2_I]]
uint32x4_t test_vcaleq_f32(float32x4_t v1, float32x4_t v2) {
  return vcaleq_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
}

// CHECK-LABEL: @test_vcaleq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[VCALEQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.facge.v2i64.v2f64(<2 x double> %v2, <2 x double> %v1)
// CHECK:   ret <2 x i64> [[VCALEQ_V2_I]]
uint64x2_t test_vcaleq_f64(float64x2_t v1, float64x2_t v2) {
  return vcaleq_f64(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
}

// CHECK-LABEL: @test_vcalt_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// CHECK:   [[VCALT_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.facgt.v2i32.v2f32(<2 x float> %v2, <2 x float> %v1)
// CHECK:   ret <2 x i32> [[VCALT_V2_I]]
uint32x2_t test_vcalt_f32(float32x2_t v1, float32x2_t v2) {
  return vcalt_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
}

// CHECK-LABEL: @test_vcalt_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VCALT_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.facgt.v1i64.v1f64(<1 x double> %b, <1 x double> %a)
// CHECK:   ret <1 x i64> [[VCALT_V2_I]]
uint64x1_t test_vcalt_f64(float64x1_t a, float64x1_t b) {
  return vcalt_f64(a, b);
}

// CHECK-LABEL: @test_vcaltq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// CHECK:   [[VCALTQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.facgt.v4i32.v4f32(<4 x float> %v2, <4 x float> %v1)
// CHECK:   ret <4 x i32> [[VCALTQ_V2_I]]
uint32x4_t test_vcaltq_f32(float32x4_t v1, float32x4_t v2) {
  return vcaltq_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
}

// CHECK-LABEL: @test_vcaltq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// CHECK:   [[VCALTQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.facgt.v2i64.v2f64(<2 x double> %v2, <2 x double> %v1)
// CHECK:   ret <2 x i64> [[VCALTQ_V2_I]]
uint64x2_t test_vcaltq_f64(float64x2_t v1, float64x2_t v2) {
  return vcaltq_f64(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
}

// CHECK-LABEL: @test_vtst_s8(
// CHECK:   [[TMP0:%.*]] = and <8 x i8> %v1, %v2
// CHECK:   [[TMP1:%.*]] = icmp ne <8 x i8> [[TMP0]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VTST_I]]
uint8x8_t test_vtst_s8(int8x8_t v1, int8x8_t v2) {
  return vtst_s8(v1, v2);
}

// CHECK-LABEL: @test_vtst_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <4 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <4 x i16> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VTST_I]]
uint16x4_t test_vtst_s16(int16x4_t v1, int16x4_t v2) {
  return vtst_s16(v1, v2);
}

// CHECK-LABEL: @test_vtst_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <2 x i32> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <2 x i32> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VTST_I]]
uint32x2_t test_vtst_s32(int32x2_t v1, int32x2_t v2) {
  return vtst_s32(v1, v2);
}

// CHECK-LABEL: @test_vtst_u8(
// CHECK:   [[TMP0:%.*]] = and <8 x i8> %v1, %v2
// CHECK:   [[TMP1:%.*]] = icmp ne <8 x i8> [[TMP0]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VTST_I]]
uint8x8_t test_vtst_u8(uint8x8_t v1, uint8x8_t v2) {
  return vtst_u8(v1, v2);
}

// CHECK-LABEL: @test_vtst_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <4 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <4 x i16> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VTST_I]]
uint16x4_t test_vtst_u16(uint16x4_t v1, uint16x4_t v2) {
  return vtst_u16(v1, v2);
}

// CHECK-LABEL: @test_vtst_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <2 x i32> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <2 x i32> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VTST_I]]
uint32x2_t test_vtst_u32(uint32x2_t v1, uint32x2_t v2) {
  return vtst_u32(v1, v2);
}

// CHECK-LABEL: @test_vtstq_s8(
// CHECK:   [[TMP0:%.*]] = and <16 x i8> %v1, %v2
// CHECK:   [[TMP1:%.*]] = icmp ne <16 x i8> [[TMP0]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <16 x i1> [[TMP1]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VTST_I]]
uint8x16_t test_vtstq_s8(int8x16_t v1, int8x16_t v2) {
  return vtstq_s8(v1, v2);
}

// CHECK-LABEL: @test_vtstq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = and <8 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <8 x i16> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP3]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VTST_I]]
uint16x8_t test_vtstq_s16(int16x8_t v1, int16x8_t v2) {
  return vtstq_s16(v1, v2);
}

// CHECK-LABEL: @test_vtstq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = and <4 x i32> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <4 x i32> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VTST_I]]
uint32x4_t test_vtstq_s32(int32x4_t v1, int32x4_t v2) {
  return vtstq_s32(v1, v2);
}

// CHECK-LABEL: @test_vtstq_u8(
// CHECK:   [[TMP0:%.*]] = and <16 x i8> %v1, %v2
// CHECK:   [[TMP1:%.*]] = icmp ne <16 x i8> [[TMP0]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <16 x i1> [[TMP1]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VTST_I]]
uint8x16_t test_vtstq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vtstq_u8(v1, v2);
}

// CHECK-LABEL: @test_vtstq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = and <8 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <8 x i16> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP3]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VTST_I]]
uint16x8_t test_vtstq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vtstq_u16(v1, v2);
}

// CHECK-LABEL: @test_vtstq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = and <4 x i32> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <4 x i32> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VTST_I]]
uint32x4_t test_vtstq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vtstq_u32(v1, v2);
}

// CHECK-LABEL: @test_vtstq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = and <2 x i64> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <2 x i64> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VTST_I]]
uint64x2_t test_vtstq_s64(int64x2_t v1, int64x2_t v2) {
  return vtstq_s64(v1, v2);
}

// CHECK-LABEL: @test_vtstq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = and <2 x i64> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <2 x i64> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VTST_I]]
uint64x2_t test_vtstq_u64(uint64x2_t v1, uint64x2_t v2) {
  return vtstq_u64(v1, v2);
}

// CHECK-LABEL: @test_vtst_p8(
// CHECK:   [[TMP0:%.*]] = and <8 x i8> %v1, %v2
// CHECK:   [[TMP1:%.*]] = icmp ne <8 x i8> [[TMP0]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VTST_I]]
uint8x8_t test_vtst_p8(poly8x8_t v1, poly8x8_t v2) {
  return vtst_p8(v1, v2);
}

// CHECK-LABEL: @test_vtst_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <4 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <4 x i16> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VTST_I]]
uint16x4_t test_vtst_p16(poly16x4_t v1, poly16x4_t v2) {
  return vtst_p16(v1, v2);
}

// CHECK-LABEL: @test_vtstq_p8(
// CHECK:   [[TMP0:%.*]] = and <16 x i8> %v1, %v2
// CHECK:   [[TMP1:%.*]] = icmp ne <16 x i8> [[TMP0]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <16 x i1> [[TMP1]] to <16 x i8>
// CHECK:   ret <16 x i8> [[VTST_I]]
uint8x16_t test_vtstq_p8(poly8x16_t v1, poly8x16_t v2) {
  return vtstq_p8(v1, v2);
}

// CHECK-LABEL: @test_vtstq_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[TMP2:%.*]] = and <8 x i16> %v1, %v2
// CHECK:   [[TMP3:%.*]] = icmp ne <8 x i16> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP3]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VTST_I]]
uint16x8_t test_vtstq_p16(poly16x8_t v1, poly16x8_t v2) {
  return vtstq_p16(v1, v2);
}

// CHECK-LABEL: @test_vtst_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <1 x i64> %a, %b
// CHECK:   [[TMP3:%.*]] = icmp ne <1 x i64> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <1 x i1> [[TMP3]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VTST_I]]
uint64x1_t test_vtst_s64(int64x1_t a, int64x1_t b) {
  return vtst_s64(a, b);
}

// CHECK-LABEL: @test_vtst_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <1 x i64> %a, %b
// CHECK:   [[TMP3:%.*]] = icmp ne <1 x i64> [[TMP2]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <1 x i1> [[TMP3]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VTST_I]]
uint64x1_t test_vtst_u64(uint64x1_t a, uint64x1_t b) {
  return vtst_u64(a, b);
}

// CHECK-LABEL: @test_vceq_s8(
// CHECK:   [[CMP_I:%.*]] = icmp eq <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vceq_s8(int8x8_t v1, int8x8_t v2) {
  return vceq_s8(v1, v2);
}

// CHECK-LABEL: @test_vceq_s16(
// CHECK:   [[CMP_I:%.*]] = icmp eq <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vceq_s16(int16x4_t v1, int16x4_t v2) {
  return vceq_s16(v1, v2);
}

// CHECK-LABEL: @test_vceq_s32(
// CHECK:   [[CMP_I:%.*]] = icmp eq <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vceq_s32(int32x2_t v1, int32x2_t v2) {
  return vceq_s32(v1, v2);
}

// CHECK-LABEL: @test_vceq_s64(
// CHECK:   [[CMP_I:%.*]] = icmp eq <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vceq_s64(int64x1_t a, int64x1_t b) {
  return vceq_s64(a, b);
}

// CHECK-LABEL: @test_vceq_u64(
// CHECK:   [[CMP_I:%.*]] = icmp eq <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vceq_u64(uint64x1_t a, uint64x1_t b) {
  return vceq_u64(a, b);
}

// CHECK-LABEL: @test_vceq_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp oeq <2 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vceq_f32(float32x2_t v1, float32x2_t v2) {
  return vceq_f32(v1, v2);
}

// CHECK-LABEL: @test_vceq_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp oeq <1 x double> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vceq_f64(float64x1_t a, float64x1_t b) {
  return vceq_f64(a, b);
}

// CHECK-LABEL: @test_vceq_u8(
// CHECK:   [[CMP_I:%.*]] = icmp eq <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vceq_u8(uint8x8_t v1, uint8x8_t v2) {
  return vceq_u8(v1, v2);
}

// CHECK-LABEL: @test_vceq_u16(
// CHECK:   [[CMP_I:%.*]] = icmp eq <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vceq_u16(uint16x4_t v1, uint16x4_t v2) {
  return vceq_u16(v1, v2);
}

// CHECK-LABEL: @test_vceq_u32(
// CHECK:   [[CMP_I:%.*]] = icmp eq <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vceq_u32(uint32x2_t v1, uint32x2_t v2) {
  return vceq_u32(v1, v2);
}

// CHECK-LABEL: @test_vceq_p8(
// CHECK:   [[CMP_I:%.*]] = icmp eq <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vceq_p8(poly8x8_t v1, poly8x8_t v2) {
  return vceq_p8(v1, v2);
}

// CHECK-LABEL: @test_vceqq_s8(
// CHECK:   [[CMP_I:%.*]] = icmp eq <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vceqq_s8(int8x16_t v1, int8x16_t v2) {
  return vceqq_s8(v1, v2);
}

// CHECK-LABEL: @test_vceqq_s16(
// CHECK:   [[CMP_I:%.*]] = icmp eq <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vceqq_s16(int16x8_t v1, int16x8_t v2) {
  return vceqq_s16(v1, v2);
}

// CHECK-LABEL: @test_vceqq_s32(
// CHECK:   [[CMP_I:%.*]] = icmp eq <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vceqq_s32(int32x4_t v1, int32x4_t v2) {
  return vceqq_s32(v1, v2);
}

// CHECK-LABEL: @test_vceqq_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp oeq <4 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vceqq_f32(float32x4_t v1, float32x4_t v2) {
  return vceqq_f32(v1, v2);
}

// CHECK-LABEL: @test_vceqq_u8(
// CHECK:   [[CMP_I:%.*]] = icmp eq <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vceqq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vceqq_u8(v1, v2);
}

// CHECK-LABEL: @test_vceqq_u16(
// CHECK:   [[CMP_I:%.*]] = icmp eq <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vceqq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vceqq_u16(v1, v2);
}

// CHECK-LABEL: @test_vceqq_u32(
// CHECK:   [[CMP_I:%.*]] = icmp eq <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vceqq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vceqq_u32(v1, v2);
}

// CHECK-LABEL: @test_vceqq_p8(
// CHECK:   [[CMP_I:%.*]] = icmp eq <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vceqq_p8(poly8x16_t v1, poly8x16_t v2) {
  return vceqq_p8(v1, v2);
}

// CHECK-LABEL: @test_vceqq_s64(
// CHECK:   [[CMP_I:%.*]] = icmp eq <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vceqq_s64(int64x2_t v1, int64x2_t v2) {
  return vceqq_s64(v1, v2);
}

// CHECK-LABEL: @test_vceqq_u64(
// CHECK:   [[CMP_I:%.*]] = icmp eq <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vceqq_u64(uint64x2_t v1, uint64x2_t v2) {
  return vceqq_u64(v1, v2);
}

// CHECK-LABEL: @test_vceqq_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp oeq <2 x double> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vceqq_f64(float64x2_t v1, float64x2_t v2) {
  return vceqq_f64(v1, v2);
}

// CHECK-LABEL: @test_vcge_s8(
// CHECK:   [[CMP_I:%.*]] = icmp sge <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vcge_s8(int8x8_t v1, int8x8_t v2) {
  return vcge_s8(v1, v2);
}

// CHECK-LABEL: @test_vcge_s16(
// CHECK:   [[CMP_I:%.*]] = icmp sge <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vcge_s16(int16x4_t v1, int16x4_t v2) {
  return vcge_s16(v1, v2);
}

// CHECK-LABEL: @test_vcge_s32(
// CHECK:   [[CMP_I:%.*]] = icmp sge <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcge_s32(int32x2_t v1, int32x2_t v2) {
  return vcge_s32(v1, v2);
}

// CHECK-LABEL: @test_vcge_s64(
// CHECK:   [[CMP_I:%.*]] = icmp sge <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcge_s64(int64x1_t a, int64x1_t b) {
  return vcge_s64(a, b);
}

// CHECK-LABEL: @test_vcge_u64(
// CHECK:   [[CMP_I:%.*]] = icmp uge <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcge_u64(uint64x1_t a, uint64x1_t b) {
  return vcge_u64(a, b);
}

// CHECK-LABEL: @test_vcge_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp oge <2 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcge_f32(float32x2_t v1, float32x2_t v2) {
  return vcge_f32(v1, v2);
}

// CHECK-LABEL: @test_vcge_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp oge <1 x double> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcge_f64(float64x1_t a, float64x1_t b) {
  return vcge_f64(a, b);
}

// CHECK-LABEL: @test_vcge_u8(
// CHECK:   [[CMP_I:%.*]] = icmp uge <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vcge_u8(uint8x8_t v1, uint8x8_t v2) {
  return vcge_u8(v1, v2);
}

// CHECK-LABEL: @test_vcge_u16(
// CHECK:   [[CMP_I:%.*]] = icmp uge <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vcge_u16(uint16x4_t v1, uint16x4_t v2) {
  return vcge_u16(v1, v2);
}

// CHECK-LABEL: @test_vcge_u32(
// CHECK:   [[CMP_I:%.*]] = icmp uge <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcge_u32(uint32x2_t v1, uint32x2_t v2) {
  return vcge_u32(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_s8(
// CHECK:   [[CMP_I:%.*]] = icmp sge <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vcgeq_s8(int8x16_t v1, int8x16_t v2) {
  return vcgeq_s8(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_s16(
// CHECK:   [[CMP_I:%.*]] = icmp sge <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vcgeq_s16(int16x8_t v1, int16x8_t v2) {
  return vcgeq_s16(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_s32(
// CHECK:   [[CMP_I:%.*]] = icmp sge <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcgeq_s32(int32x4_t v1, int32x4_t v2) {
  return vcgeq_s32(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp oge <4 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcgeq_f32(float32x4_t v1, float32x4_t v2) {
  return vcgeq_f32(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_u8(
// CHECK:   [[CMP_I:%.*]] = icmp uge <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vcgeq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vcgeq_u8(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_u16(
// CHECK:   [[CMP_I:%.*]] = icmp uge <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vcgeq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vcgeq_u16(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_u32(
// CHECK:   [[CMP_I:%.*]] = icmp uge <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcgeq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vcgeq_u32(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_s64(
// CHECK:   [[CMP_I:%.*]] = icmp sge <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcgeq_s64(int64x2_t v1, int64x2_t v2) {
  return vcgeq_s64(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_u64(
// CHECK:   [[CMP_I:%.*]] = icmp uge <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcgeq_u64(uint64x2_t v1, uint64x2_t v2) {
  return vcgeq_u64(v1, v2);
}

// CHECK-LABEL: @test_vcgeq_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp oge <2 x double> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcgeq_f64(float64x2_t v1, float64x2_t v2) {
  return vcgeq_f64(v1, v2);
}

// CHECK-LABEL: @test_vcle_s8(
// CHECK:   [[CMP_I:%.*]] = icmp sle <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
// Notes about vcle:
// LE condition predicate implemented as GE, so check reversed operands.
// Using registers other than v0, v1 are possible, but would be odd.
uint8x8_t test_vcle_s8(int8x8_t v1, int8x8_t v2) {
  return vcle_s8(v1, v2);
}

// CHECK-LABEL: @test_vcle_s16(
// CHECK:   [[CMP_I:%.*]] = icmp sle <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vcle_s16(int16x4_t v1, int16x4_t v2) {
  return vcle_s16(v1, v2);
}

// CHECK-LABEL: @test_vcle_s32(
// CHECK:   [[CMP_I:%.*]] = icmp sle <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcle_s32(int32x2_t v1, int32x2_t v2) {
  return vcle_s32(v1, v2);
}

// CHECK-LABEL: @test_vcle_s64(
// CHECK:   [[CMP_I:%.*]] = icmp sle <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcle_s64(int64x1_t a, int64x1_t b) {
  return vcle_s64(a, b);
}

// CHECK-LABEL: @test_vcle_u64(
// CHECK:   [[CMP_I:%.*]] = icmp ule <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcle_u64(uint64x1_t a, uint64x1_t b) {
  return vcle_u64(a, b);
}

// CHECK-LABEL: @test_vcle_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp ole <2 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcle_f32(float32x2_t v1, float32x2_t v2) {
  return vcle_f32(v1, v2);
}

// CHECK-LABEL: @test_vcle_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp ole <1 x double> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcle_f64(float64x1_t a, float64x1_t b) {
  return vcle_f64(a, b);
}

// CHECK-LABEL: @test_vcle_u8(
// CHECK:   [[CMP_I:%.*]] = icmp ule <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vcle_u8(uint8x8_t v1, uint8x8_t v2) {
  return vcle_u8(v1, v2);
}

// CHECK-LABEL: @test_vcle_u16(
// CHECK:   [[CMP_I:%.*]] = icmp ule <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vcle_u16(uint16x4_t v1, uint16x4_t v2) {
  return vcle_u16(v1, v2);
}

// CHECK-LABEL: @test_vcle_u32(
// CHECK:   [[CMP_I:%.*]] = icmp ule <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcle_u32(uint32x2_t v1, uint32x2_t v2) {
  return vcle_u32(v1, v2);
}

// CHECK-LABEL: @test_vcleq_s8(
// CHECK:   [[CMP_I:%.*]] = icmp sle <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vcleq_s8(int8x16_t v1, int8x16_t v2) {
  return vcleq_s8(v1, v2);
}

// CHECK-LABEL: @test_vcleq_s16(
// CHECK:   [[CMP_I:%.*]] = icmp sle <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vcleq_s16(int16x8_t v1, int16x8_t v2) {
  return vcleq_s16(v1, v2);
}

// CHECK-LABEL: @test_vcleq_s32(
// CHECK:   [[CMP_I:%.*]] = icmp sle <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcleq_s32(int32x4_t v1, int32x4_t v2) {
  return vcleq_s32(v1, v2);
}

// CHECK-LABEL: @test_vcleq_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp ole <4 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcleq_f32(float32x4_t v1, float32x4_t v2) {
  return vcleq_f32(v1, v2);
}

// CHECK-LABEL: @test_vcleq_u8(
// CHECK:   [[CMP_I:%.*]] = icmp ule <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vcleq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vcleq_u8(v1, v2);
}

// CHECK-LABEL: @test_vcleq_u16(
// CHECK:   [[CMP_I:%.*]] = icmp ule <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vcleq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vcleq_u16(v1, v2);
}

// CHECK-LABEL: @test_vcleq_u32(
// CHECK:   [[CMP_I:%.*]] = icmp ule <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcleq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vcleq_u32(v1, v2);
}

// CHECK-LABEL: @test_vcleq_s64(
// CHECK:   [[CMP_I:%.*]] = icmp sle <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcleq_s64(int64x2_t v1, int64x2_t v2) {
  return vcleq_s64(v1, v2);
}

// CHECK-LABEL: @test_vcleq_u64(
// CHECK:   [[CMP_I:%.*]] = icmp ule <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcleq_u64(uint64x2_t v1, uint64x2_t v2) {
  return vcleq_u64(v1, v2);
}

// CHECK-LABEL: @test_vcleq_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp ole <2 x double> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcleq_f64(float64x2_t v1, float64x2_t v2) {
  return vcleq_f64(v1, v2);
}

// CHECK-LABEL: @test_vcgt_s8(
// CHECK:   [[CMP_I:%.*]] = icmp sgt <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vcgt_s8(int8x8_t v1, int8x8_t v2) {
  return vcgt_s8(v1, v2);
}

// CHECK-LABEL: @test_vcgt_s16(
// CHECK:   [[CMP_I:%.*]] = icmp sgt <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vcgt_s16(int16x4_t v1, int16x4_t v2) {
  return vcgt_s16(v1, v2);
}

// CHECK-LABEL: @test_vcgt_s32(
// CHECK:   [[CMP_I:%.*]] = icmp sgt <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcgt_s32(int32x2_t v1, int32x2_t v2) {
  return vcgt_s32(v1, v2);
}

// CHECK-LABEL: @test_vcgt_s64(
// CHECK:   [[CMP_I:%.*]] = icmp sgt <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcgt_s64(int64x1_t a, int64x1_t b) {
  return vcgt_s64(a, b);
}

// CHECK-LABEL: @test_vcgt_u64(
// CHECK:   [[CMP_I:%.*]] = icmp ugt <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcgt_u64(uint64x1_t a, uint64x1_t b) {
  return vcgt_u64(a, b);
}

// CHECK-LABEL: @test_vcgt_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp ogt <2 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcgt_f32(float32x2_t v1, float32x2_t v2) {
  return vcgt_f32(v1, v2);
}

// CHECK-LABEL: @test_vcgt_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp ogt <1 x double> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcgt_f64(float64x1_t a, float64x1_t b) {
  return vcgt_f64(a, b);
}

// CHECK-LABEL: @test_vcgt_u8(
// CHECK:   [[CMP_I:%.*]] = icmp ugt <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vcgt_u8(uint8x8_t v1, uint8x8_t v2) {
  return vcgt_u8(v1, v2);
}

// CHECK-LABEL: @test_vcgt_u16(
// CHECK:   [[CMP_I:%.*]] = icmp ugt <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vcgt_u16(uint16x4_t v1, uint16x4_t v2) {
  return vcgt_u16(v1, v2);
}

// CHECK-LABEL: @test_vcgt_u32(
// CHECK:   [[CMP_I:%.*]] = icmp ugt <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcgt_u32(uint32x2_t v1, uint32x2_t v2) {
  return vcgt_u32(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_s8(
// CHECK:   [[CMP_I:%.*]] = icmp sgt <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vcgtq_s8(int8x16_t v1, int8x16_t v2) {
  return vcgtq_s8(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_s16(
// CHECK:   [[CMP_I:%.*]] = icmp sgt <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vcgtq_s16(int16x8_t v1, int16x8_t v2) {
  return vcgtq_s16(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_s32(
// CHECK:   [[CMP_I:%.*]] = icmp sgt <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcgtq_s32(int32x4_t v1, int32x4_t v2) {
  return vcgtq_s32(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp ogt <4 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcgtq_f32(float32x4_t v1, float32x4_t v2) {
  return vcgtq_f32(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_u8(
// CHECK:   [[CMP_I:%.*]] = icmp ugt <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vcgtq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vcgtq_u8(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_u16(
// CHECK:   [[CMP_I:%.*]] = icmp ugt <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vcgtq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vcgtq_u16(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_u32(
// CHECK:   [[CMP_I:%.*]] = icmp ugt <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcgtq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vcgtq_u32(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_s64(
// CHECK:   [[CMP_I:%.*]] = icmp sgt <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcgtq_s64(int64x2_t v1, int64x2_t v2) {
  return vcgtq_s64(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_u64(
// CHECK:   [[CMP_I:%.*]] = icmp ugt <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcgtq_u64(uint64x2_t v1, uint64x2_t v2) {
  return vcgtq_u64(v1, v2);
}

// CHECK-LABEL: @test_vcgtq_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp ogt <2 x double> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcgtq_f64(float64x2_t v1, float64x2_t v2) {
  return vcgtq_f64(v1, v2);
}

// CHECK-LABEL: @test_vclt_s8(
// CHECK:   [[CMP_I:%.*]] = icmp slt <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
// Notes about vclt:
// LT condition predicate implemented as GT, so check reversed operands.
// Using registers other than v0, v1 are possible, but would be odd.
uint8x8_t test_vclt_s8(int8x8_t v1, int8x8_t v2) {
  return vclt_s8(v1, v2);
}

// CHECK-LABEL: @test_vclt_s16(
// CHECK:   [[CMP_I:%.*]] = icmp slt <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vclt_s16(int16x4_t v1, int16x4_t v2) {
  return vclt_s16(v1, v2);
}

// CHECK-LABEL: @test_vclt_s32(
// CHECK:   [[CMP_I:%.*]] = icmp slt <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vclt_s32(int32x2_t v1, int32x2_t v2) {
  return vclt_s32(v1, v2);
}

// CHECK-LABEL: @test_vclt_s64(
// CHECK:   [[CMP_I:%.*]] = icmp slt <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vclt_s64(int64x1_t a, int64x1_t b) {
  return vclt_s64(a, b);
}

// CHECK-LABEL: @test_vclt_u64(
// CHECK:   [[CMP_I:%.*]] = icmp ult <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vclt_u64(uint64x1_t a, uint64x1_t b) {
  return vclt_u64(a, b);
}

// CHECK-LABEL: @test_vclt_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp olt <2 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vclt_f32(float32x2_t v1, float32x2_t v2) {
  return vclt_f32(v1, v2);
}

// CHECK-LABEL: @test_vclt_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp olt <1 x double> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vclt_f64(float64x1_t a, float64x1_t b) {
  return vclt_f64(a, b);
}

// CHECK-LABEL: @test_vclt_u8(
// CHECK:   [[CMP_I:%.*]] = icmp ult <8 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[SEXT_I]]
uint8x8_t test_vclt_u8(uint8x8_t v1, uint8x8_t v2) {
  return vclt_u8(v1, v2);
}

// CHECK-LABEL: @test_vclt_u16(
// CHECK:   [[CMP_I:%.*]] = icmp ult <4 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[SEXT_I]]
uint16x4_t test_vclt_u16(uint16x4_t v1, uint16x4_t v2) {
  return vclt_u16(v1, v2);
}

// CHECK-LABEL: @test_vclt_u32(
// CHECK:   [[CMP_I:%.*]] = icmp ult <2 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vclt_u32(uint32x2_t v1, uint32x2_t v2) {
  return vclt_u32(v1, v2);
}

// CHECK-LABEL: @test_vcltq_s8(
// CHECK:   [[CMP_I:%.*]] = icmp slt <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vcltq_s8(int8x16_t v1, int8x16_t v2) {
  return vcltq_s8(v1, v2);
}

// CHECK-LABEL: @test_vcltq_s16(
// CHECK:   [[CMP_I:%.*]] = icmp slt <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vcltq_s16(int16x8_t v1, int16x8_t v2) {
  return vcltq_s16(v1, v2);
}

// CHECK-LABEL: @test_vcltq_s32(
// CHECK:   [[CMP_I:%.*]] = icmp slt <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcltq_s32(int32x4_t v1, int32x4_t v2) {
  return vcltq_s32(v1, v2);
}

// CHECK-LABEL: @test_vcltq_f32(
// CHECK:   [[CMP_I:%.*]] = fcmp olt <4 x float> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcltq_f32(float32x4_t v1, float32x4_t v2) {
  return vcltq_f32(v1, v2);
}

// CHECK-LABEL: @test_vcltq_u8(
// CHECK:   [[CMP_I:%.*]] = icmp ult <16 x i8> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// CHECK:   ret <16 x i8> [[SEXT_I]]
uint8x16_t test_vcltq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vcltq_u8(v1, v2);
}

// CHECK-LABEL: @test_vcltq_u16(
// CHECK:   [[CMP_I:%.*]] = icmp ult <8 x i16> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[SEXT_I]]
uint16x8_t test_vcltq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vcltq_u16(v1, v2);
}

// CHECK-LABEL: @test_vcltq_u32(
// CHECK:   [[CMP_I:%.*]] = icmp ult <4 x i32> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcltq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vcltq_u32(v1, v2);
}

// CHECK-LABEL: @test_vcltq_s64(
// CHECK:   [[CMP_I:%.*]] = icmp slt <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcltq_s64(int64x2_t v1, int64x2_t v2) {
  return vcltq_s64(v1, v2);
}

// CHECK-LABEL: @test_vcltq_u64(
// CHECK:   [[CMP_I:%.*]] = icmp ult <2 x i64> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcltq_u64(uint64x2_t v1, uint64x2_t v2) {
  return vcltq_u64(v1, v2);
}

// CHECK-LABEL: @test_vcltq_f64(
// CHECK:   [[CMP_I:%.*]] = fcmp olt <2 x double> %v1, %v2
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcltq_f64(float64x2_t v1, float64x2_t v2) {
  return vcltq_f64(v1, v2);
}

// CHECK-LABEL: @test_vhadd_s8(
// CHECK:   [[VHADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.shadd.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VHADD_V_I]]
int8x8_t test_vhadd_s8(int8x8_t v1, int8x8_t v2) {
  return vhadd_s8(v1, v2);
}

// CHECK-LABEL: @test_vhadd_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[VHADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.shadd.v4i16(<4 x i16> %v1, <4 x i16> %v2)
// CHECK:   [[VHADD_V3_I:%.*]] = bitcast <4 x i16> [[VHADD_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VHADD_V2_I]]
int16x4_t test_vhadd_s16(int16x4_t v1, int16x4_t v2) {
  return vhadd_s16(v1, v2);
}

// CHECK-LABEL: @test_vhadd_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[VHADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.shadd.v2i32(<2 x i32> %v1, <2 x i32> %v2)
// CHECK:   [[VHADD_V3_I:%.*]] = bitcast <2 x i32> [[VHADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VHADD_V2_I]]
int32x2_t test_vhadd_s32(int32x2_t v1, int32x2_t v2) {
  return vhadd_s32(v1, v2);
}

// CHECK-LABEL: @test_vhadd_u8(
// CHECK:   [[VHADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uhadd.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VHADD_V_I]]
uint8x8_t test_vhadd_u8(uint8x8_t v1, uint8x8_t v2) {
  return vhadd_u8(v1, v2);
}

// CHECK-LABEL: @test_vhadd_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[VHADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uhadd.v4i16(<4 x i16> %v1, <4 x i16> %v2)
// CHECK:   [[VHADD_V3_I:%.*]] = bitcast <4 x i16> [[VHADD_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VHADD_V2_I]]
uint16x4_t test_vhadd_u16(uint16x4_t v1, uint16x4_t v2) {
  return vhadd_u16(v1, v2);
}

// CHECK-LABEL: @test_vhadd_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[VHADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uhadd.v2i32(<2 x i32> %v1, <2 x i32> %v2)
// CHECK:   [[VHADD_V3_I:%.*]] = bitcast <2 x i32> [[VHADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VHADD_V2_I]]
uint32x2_t test_vhadd_u32(uint32x2_t v1, uint32x2_t v2) {
  return vhadd_u32(v1, v2);
}

// CHECK-LABEL: @test_vhaddq_s8(
// CHECK:   [[VHADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.shadd.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VHADDQ_V_I]]
int8x16_t test_vhaddq_s8(int8x16_t v1, int8x16_t v2) {
  return vhaddq_s8(v1, v2);
}

// CHECK-LABEL: @test_vhaddq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[VHADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.shadd.v8i16(<8 x i16> %v1, <8 x i16> %v2)
// CHECK:   [[VHADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VHADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VHADDQ_V2_I]]
int16x8_t test_vhaddq_s16(int16x8_t v1, int16x8_t v2) {
  return vhaddq_s16(v1, v2);
}

// CHECK-LABEL: @test_vhaddq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[VHADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.shadd.v4i32(<4 x i32> %v1, <4 x i32> %v2)
// CHECK:   [[VHADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VHADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VHADDQ_V2_I]]
int32x4_t test_vhaddq_s32(int32x4_t v1, int32x4_t v2) {
  return vhaddq_s32(v1, v2);
}

// CHECK-LABEL: @test_vhaddq_u8(
// CHECK:   [[VHADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uhadd.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VHADDQ_V_I]]
uint8x16_t test_vhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vhaddq_u8(v1, v2);
}

// CHECK-LABEL: @test_vhaddq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[VHADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uhadd.v8i16(<8 x i16> %v1, <8 x i16> %v2)
// CHECK:   [[VHADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VHADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VHADDQ_V2_I]]
uint16x8_t test_vhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vhaddq_u16(v1, v2);
}

// CHECK-LABEL: @test_vhaddq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[VHADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uhadd.v4i32(<4 x i32> %v1, <4 x i32> %v2)
// CHECK:   [[VHADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VHADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VHADDQ_V2_I]]
uint32x4_t test_vhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vhaddq_u32(v1, v2);
}

// CHECK-LABEL: @test_vhsub_s8(
// CHECK:   [[VHSUB_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.shsub.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VHSUB_V_I]]
int8x8_t test_vhsub_s8(int8x8_t v1, int8x8_t v2) {
  return vhsub_s8(v1, v2);
}

// CHECK-LABEL: @test_vhsub_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[VHSUB_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.shsub.v4i16(<4 x i16> %v1, <4 x i16> %v2)
// CHECK:   [[VHSUB_V3_I:%.*]] = bitcast <4 x i16> [[VHSUB_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VHSUB_V2_I]]
int16x4_t test_vhsub_s16(int16x4_t v1, int16x4_t v2) {
  return vhsub_s16(v1, v2);
}

// CHECK-LABEL: @test_vhsub_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[VHSUB_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.shsub.v2i32(<2 x i32> %v1, <2 x i32> %v2)
// CHECK:   [[VHSUB_V3_I:%.*]] = bitcast <2 x i32> [[VHSUB_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VHSUB_V2_I]]
int32x2_t test_vhsub_s32(int32x2_t v1, int32x2_t v2) {
  return vhsub_s32(v1, v2);
}

// CHECK-LABEL: @test_vhsub_u8(
// CHECK:   [[VHSUB_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uhsub.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VHSUB_V_I]]
uint8x8_t test_vhsub_u8(uint8x8_t v1, uint8x8_t v2) {
  return vhsub_u8(v1, v2);
}

// CHECK-LABEL: @test_vhsub_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[VHSUB_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uhsub.v4i16(<4 x i16> %v1, <4 x i16> %v2)
// CHECK:   [[VHSUB_V3_I:%.*]] = bitcast <4 x i16> [[VHSUB_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VHSUB_V2_I]]
uint16x4_t test_vhsub_u16(uint16x4_t v1, uint16x4_t v2) {
  return vhsub_u16(v1, v2);
}

// CHECK-LABEL: @test_vhsub_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[VHSUB_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uhsub.v2i32(<2 x i32> %v1, <2 x i32> %v2)
// CHECK:   [[VHSUB_V3_I:%.*]] = bitcast <2 x i32> [[VHSUB_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VHSUB_V2_I]]
uint32x2_t test_vhsub_u32(uint32x2_t v1, uint32x2_t v2) {
  return vhsub_u32(v1, v2);
}

// CHECK-LABEL: @test_vhsubq_s8(
// CHECK:   [[VHSUBQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.shsub.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VHSUBQ_V_I]]
int8x16_t test_vhsubq_s8(int8x16_t v1, int8x16_t v2) {
  return vhsubq_s8(v1, v2);
}

// CHECK-LABEL: @test_vhsubq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[VHSUBQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.shsub.v8i16(<8 x i16> %v1, <8 x i16> %v2)
// CHECK:   [[VHSUBQ_V3_I:%.*]] = bitcast <8 x i16> [[VHSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VHSUBQ_V2_I]]
int16x8_t test_vhsubq_s16(int16x8_t v1, int16x8_t v2) {
  return vhsubq_s16(v1, v2);
}

// CHECK-LABEL: @test_vhsubq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[VHSUBQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.shsub.v4i32(<4 x i32> %v1, <4 x i32> %v2)
// CHECK:   [[VHSUBQ_V3_I:%.*]] = bitcast <4 x i32> [[VHSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VHSUBQ_V2_I]]
int32x4_t test_vhsubq_s32(int32x4_t v1, int32x4_t v2) {
  return vhsubq_s32(v1, v2);
}

// CHECK-LABEL: @test_vhsubq_u8(
// CHECK:   [[VHSUBQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uhsub.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VHSUBQ_V_I]]
uint8x16_t test_vhsubq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vhsubq_u8(v1, v2);
}

// CHECK-LABEL: @test_vhsubq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[VHSUBQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uhsub.v8i16(<8 x i16> %v1, <8 x i16> %v2)
// CHECK:   [[VHSUBQ_V3_I:%.*]] = bitcast <8 x i16> [[VHSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VHSUBQ_V2_I]]
uint16x8_t test_vhsubq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vhsubq_u16(v1, v2);
}

// CHECK-LABEL: @test_vhsubq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[VHSUBQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uhsub.v4i32(<4 x i32> %v1, <4 x i32> %v2)
// CHECK:   [[VHSUBQ_V3_I:%.*]] = bitcast <4 x i32> [[VHSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VHSUBQ_V2_I]]
uint32x4_t test_vhsubq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vhsubq_u32(v1, v2);
}

// CHECK-LABEL: @test_vrhadd_s8(
// CHECK:   [[VRHADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.srhadd.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VRHADD_V_I]]
int8x8_t test_vrhadd_s8(int8x8_t v1, int8x8_t v2) {
  return vrhadd_s8(v1, v2);
}

// CHECK-LABEL: @test_vrhadd_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[VRHADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.srhadd.v4i16(<4 x i16> %v1, <4 x i16> %v2)
// CHECK:   [[VRHADD_V3_I:%.*]] = bitcast <4 x i16> [[VRHADD_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VRHADD_V2_I]]
int16x4_t test_vrhadd_s16(int16x4_t v1, int16x4_t v2) {
  return vrhadd_s16(v1, v2);
}

// CHECK-LABEL: @test_vrhadd_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[VRHADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.srhadd.v2i32(<2 x i32> %v1, <2 x i32> %v2)
// CHECK:   [[VRHADD_V3_I:%.*]] = bitcast <2 x i32> [[VRHADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VRHADD_V2_I]]
int32x2_t test_vrhadd_s32(int32x2_t v1, int32x2_t v2) {
  return vrhadd_s32(v1, v2);
}

// CHECK-LABEL: @test_vrhadd_u8(
// CHECK:   [[VRHADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.urhadd.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// CHECK:   ret <8 x i8> [[VRHADD_V_I]]
uint8x8_t test_vrhadd_u8(uint8x8_t v1, uint8x8_t v2) {
  return vrhadd_u8(v1, v2);
}

// CHECK-LABEL: @test_vrhadd_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// CHECK:   [[VRHADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.urhadd.v4i16(<4 x i16> %v1, <4 x i16> %v2)
// CHECK:   [[VRHADD_V3_I:%.*]] = bitcast <4 x i16> [[VRHADD_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VRHADD_V2_I]]
uint16x4_t test_vrhadd_u16(uint16x4_t v1, uint16x4_t v2) {
  return vrhadd_u16(v1, v2);
}

// CHECK-LABEL: @test_vrhadd_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// CHECK:   [[VRHADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.urhadd.v2i32(<2 x i32> %v1, <2 x i32> %v2)
// CHECK:   [[VRHADD_V3_I:%.*]] = bitcast <2 x i32> [[VRHADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VRHADD_V2_I]]
uint32x2_t test_vrhadd_u32(uint32x2_t v1, uint32x2_t v2) {
  return vrhadd_u32(v1, v2);
}

// CHECK-LABEL: @test_vrhaddq_s8(
// CHECK:   [[VRHADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.srhadd.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VRHADDQ_V_I]]
int8x16_t test_vrhaddq_s8(int8x16_t v1, int8x16_t v2) {
  return vrhaddq_s8(v1, v2);
}

// CHECK-LABEL: @test_vrhaddq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[VRHADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.srhadd.v8i16(<8 x i16> %v1, <8 x i16> %v2)
// CHECK:   [[VRHADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VRHADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VRHADDQ_V2_I]]
int16x8_t test_vrhaddq_s16(int16x8_t v1, int16x8_t v2) {
  return vrhaddq_s16(v1, v2);
}

// CHECK-LABEL: @test_vrhaddq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[VRHADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.srhadd.v4i32(<4 x i32> %v1, <4 x i32> %v2)
// CHECK:   [[VRHADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VRHADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VRHADDQ_V2_I]]
int32x4_t test_vrhaddq_s32(int32x4_t v1, int32x4_t v2) {
  return vrhaddq_s32(v1, v2);
}

// CHECK-LABEL: @test_vrhaddq_u8(
// CHECK:   [[VRHADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.urhadd.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// CHECK:   ret <16 x i8> [[VRHADDQ_V_I]]
uint8x16_t test_vrhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vrhaddq_u8(v1, v2);
}

// CHECK-LABEL: @test_vrhaddq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// CHECK:   [[VRHADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.urhadd.v8i16(<8 x i16> %v1, <8 x i16> %v2)
// CHECK:   [[VRHADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VRHADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VRHADDQ_V2_I]]
uint16x8_t test_vrhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vrhaddq_u16(v1, v2);
}

// CHECK-LABEL: @test_vrhaddq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// CHECK:   [[VRHADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.urhadd.v4i32(<4 x i32> %v1, <4 x i32> %v2)
// CHECK:   [[VRHADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VRHADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VRHADDQ_V2_I]]
uint32x4_t test_vrhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vrhaddq_u32(v1, v2);
}

// CHECK-LABEL: @test_vqadd_s8(
// CHECK:   [[VQADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqadd.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VQADD_V_I]]
int8x8_t test_vqadd_s8(int8x8_t a, int8x8_t b) {
  return vqadd_s8(a, b);
}

// CHECK-LABEL: @test_vqadd_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQADD_V3_I:%.*]] = bitcast <4 x i16> [[VQADD_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQADD_V2_I]]
int16x4_t test_vqadd_s16(int16x4_t a, int16x4_t b) {
  return vqadd_s16(a, b);
}

// CHECK-LABEL: @test_vqadd_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQADD_V3_I:%.*]] = bitcast <2 x i32> [[VQADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQADD_V2_I]]
int32x2_t test_vqadd_s32(int32x2_t a, int32x2_t b) {
  return vqadd_s32(a, b);
}

// CHECK-LABEL: @test_vqadd_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VQADD_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqadd.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VQADD_V3_I:%.*]] = bitcast <1 x i64> [[VQADD_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQADD_V2_I]]
int64x1_t test_vqadd_s64(int64x1_t a, int64x1_t b) {
  return vqadd_s64(a, b);
}

// CHECK-LABEL: @test_vqadd_u8(
// CHECK:   [[VQADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqadd.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VQADD_V_I]]
uint8x8_t test_vqadd_u8(uint8x8_t a, uint8x8_t b) {
  return vqadd_u8(a, b);
}

// CHECK-LABEL: @test_vqadd_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqadd.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQADD_V3_I:%.*]] = bitcast <4 x i16> [[VQADD_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQADD_V2_I]]
uint16x4_t test_vqadd_u16(uint16x4_t a, uint16x4_t b) {
  return vqadd_u16(a, b);
}

// CHECK-LABEL: @test_vqadd_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqadd.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQADD_V3_I:%.*]] = bitcast <2 x i32> [[VQADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQADD_V2_I]]
uint32x2_t test_vqadd_u32(uint32x2_t a, uint32x2_t b) {
  return vqadd_u32(a, b);
}

// CHECK-LABEL: @test_vqadd_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VQADD_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqadd.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VQADD_V3_I:%.*]] = bitcast <1 x i64> [[VQADD_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQADD_V2_I]]
uint64x1_t test_vqadd_u64(uint64x1_t a, uint64x1_t b) {
  return vqadd_u64(a, b);
}

// CHECK-LABEL: @test_vqaddq_s8(
// CHECK:   [[VQADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqadd.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VQADDQ_V_I]]
int8x16_t test_vqaddq_s8(int8x16_t a, int8x16_t b) {
  return vqaddq_s8(a, b);
}

// CHECK-LABEL: @test_vqaddq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VQADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQADDQ_V2_I]]
int16x8_t test_vqaddq_s16(int16x8_t a, int16x8_t b) {
  return vqaddq_s16(a, b);
}

// CHECK-LABEL: @test_vqaddq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VQADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQADDQ_V2_I]]
int32x4_t test_vqaddq_s32(int32x4_t a, int32x4_t b) {
  return vqaddq_s32(a, b);
}

// CHECK-LABEL: @test_vqaddq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQADDQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VQADDQ_V3_I:%.*]] = bitcast <2 x i64> [[VQADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQADDQ_V2_I]]
int64x2_t test_vqaddq_s64(int64x2_t a, int64x2_t b) {
  return vqaddq_s64(a, b);
}

// CHECK-LABEL: @test_vqaddq_u8(
// CHECK:   [[VQADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqadd.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VQADDQ_V_I]]
uint8x16_t test_vqaddq_u8(uint8x16_t a, uint8x16_t b) {
  return vqaddq_u8(a, b);
}

// CHECK-LABEL: @test_vqaddq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqadd.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VQADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQADDQ_V2_I]]
uint16x8_t test_vqaddq_u16(uint16x8_t a, uint16x8_t b) {
  return vqaddq_u16(a, b);
}

// CHECK-LABEL: @test_vqaddq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqadd.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VQADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQADDQ_V2_I]]
uint32x4_t test_vqaddq_u32(uint32x4_t a, uint32x4_t b) {
  return vqaddq_u32(a, b);
}

// CHECK-LABEL: @test_vqaddq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQADDQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqadd.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VQADDQ_V3_I:%.*]] = bitcast <2 x i64> [[VQADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQADDQ_V2_I]]
uint64x2_t test_vqaddq_u64(uint64x2_t a, uint64x2_t b) {
  return vqaddq_u64(a, b);
}

// CHECK-LABEL: @test_vqsub_s8(
// CHECK:   [[VQSUB_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqsub.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VQSUB_V_I]]
int8x8_t test_vqsub_s8(int8x8_t a, int8x8_t b) {
  return vqsub_s8(a, b);
}

// CHECK-LABEL: @test_vqsub_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQSUB_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQSUB_V3_I:%.*]] = bitcast <4 x i16> [[VQSUB_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQSUB_V2_I]]
int16x4_t test_vqsub_s16(int16x4_t a, int16x4_t b) {
  return vqsub_s16(a, b);
}

// CHECK-LABEL: @test_vqsub_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQSUB_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQSUB_V3_I:%.*]] = bitcast <2 x i32> [[VQSUB_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQSUB_V2_I]]
int32x2_t test_vqsub_s32(int32x2_t a, int32x2_t b) {
  return vqsub_s32(a, b);
}

// CHECK-LABEL: @test_vqsub_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VQSUB_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqsub.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VQSUB_V3_I:%.*]] = bitcast <1 x i64> [[VQSUB_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQSUB_V2_I]]
int64x1_t test_vqsub_s64(int64x1_t a, int64x1_t b) {
  return vqsub_s64(a, b);
}

// CHECK-LABEL: @test_vqsub_u8(
// CHECK:   [[VQSUB_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqsub.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VQSUB_V_I]]
uint8x8_t test_vqsub_u8(uint8x8_t a, uint8x8_t b) {
  return vqsub_u8(a, b);
}

// CHECK-LABEL: @test_vqsub_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQSUB_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqsub.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQSUB_V3_I:%.*]] = bitcast <4 x i16> [[VQSUB_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQSUB_V2_I]]
uint16x4_t test_vqsub_u16(uint16x4_t a, uint16x4_t b) {
  return vqsub_u16(a, b);
}

// CHECK-LABEL: @test_vqsub_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQSUB_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqsub.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQSUB_V3_I:%.*]] = bitcast <2 x i32> [[VQSUB_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQSUB_V2_I]]
uint32x2_t test_vqsub_u32(uint32x2_t a, uint32x2_t b) {
  return vqsub_u32(a, b);
}

// CHECK-LABEL: @test_vqsub_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VQSUB_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqsub.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VQSUB_V3_I:%.*]] = bitcast <1 x i64> [[VQSUB_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQSUB_V2_I]]
uint64x1_t test_vqsub_u64(uint64x1_t a, uint64x1_t b) {
  return vqsub_u64(a, b);
}

// CHECK-LABEL: @test_vqsubq_s8(
// CHECK:   [[VQSUBQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqsub.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VQSUBQ_V_I]]
int8x16_t test_vqsubq_s8(int8x16_t a, int8x16_t b) {
  return vqsubq_s8(a, b);
}

// CHECK-LABEL: @test_vqsubq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQSUBQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQSUBQ_V3_I:%.*]] = bitcast <8 x i16> [[VQSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQSUBQ_V2_I]]
int16x8_t test_vqsubq_s16(int16x8_t a, int16x8_t b) {
  return vqsubq_s16(a, b);
}

// CHECK-LABEL: @test_vqsubq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQSUBQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQSUBQ_V3_I:%.*]] = bitcast <4 x i32> [[VQSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQSUBQ_V2_I]]
int32x4_t test_vqsubq_s32(int32x4_t a, int32x4_t b) {
  return vqsubq_s32(a, b);
}

// CHECK-LABEL: @test_vqsubq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQSUBQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VQSUBQ_V3_I:%.*]] = bitcast <2 x i64> [[VQSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQSUBQ_V2_I]]
int64x2_t test_vqsubq_s64(int64x2_t a, int64x2_t b) {
  return vqsubq_s64(a, b);
}

// CHECK-LABEL: @test_vqsubq_u8(
// CHECK:   [[VQSUBQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqsub.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VQSUBQ_V_I]]
uint8x16_t test_vqsubq_u8(uint8x16_t a, uint8x16_t b) {
  return vqsubq_u8(a, b);
}

// CHECK-LABEL: @test_vqsubq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQSUBQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqsub.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQSUBQ_V3_I:%.*]] = bitcast <8 x i16> [[VQSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQSUBQ_V2_I]]
uint16x8_t test_vqsubq_u16(uint16x8_t a, uint16x8_t b) {
  return vqsubq_u16(a, b);
}

// CHECK-LABEL: @test_vqsubq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQSUBQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqsub.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQSUBQ_V3_I:%.*]] = bitcast <4 x i32> [[VQSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQSUBQ_V2_I]]
uint32x4_t test_vqsubq_u32(uint32x4_t a, uint32x4_t b) {
  return vqsubq_u32(a, b);
}

// CHECK-LABEL: @test_vqsubq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQSUBQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqsub.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VQSUBQ_V3_I:%.*]] = bitcast <2 x i64> [[VQSUBQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQSUBQ_V2_I]]
uint64x2_t test_vqsubq_u64(uint64x2_t a, uint64x2_t b) {
  return vqsubq_u64(a, b);
}

// CHECK-LABEL: @test_vshl_s8(
// CHECK:   [[VSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VSHL_V_I]]
int8x8_t test_vshl_s8(int8x8_t a, int8x8_t b) {
  return vshl_s8(a, b);
}

// CHECK-LABEL: @test_vshl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VSHL_V3_I:%.*]] = bitcast <4 x i16> [[VSHL_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VSHL_V2_I]]
int16x4_t test_vshl_s16(int16x4_t a, int16x4_t b) {
  return vshl_s16(a, b);
}

// CHECK-LABEL: @test_vshl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VSHL_V3_I:%.*]] = bitcast <2 x i32> [[VSHL_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VSHL_V2_I]]
int32x2_t test_vshl_s32(int32x2_t a, int32x2_t b) {
  return vshl_s32(a, b);
}

// CHECK-LABEL: @test_vshl_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VSHL_V3_I:%.*]] = bitcast <1 x i64> [[VSHL_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VSHL_V2_I]]
int64x1_t test_vshl_s64(int64x1_t a, int64x1_t b) {
  return vshl_s64(a, b);
}

// CHECK-LABEL: @test_vshl_u8(
// CHECK:   [[VSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.ushl.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VSHL_V_I]]
uint8x8_t test_vshl_u8(uint8x8_t a, int8x8_t b) {
  return vshl_u8(a, b);
}

// CHECK-LABEL: @test_vshl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.ushl.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VSHL_V3_I:%.*]] = bitcast <4 x i16> [[VSHL_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VSHL_V2_I]]
uint16x4_t test_vshl_u16(uint16x4_t a, int16x4_t b) {
  return vshl_u16(a, b);
}

// CHECK-LABEL: @test_vshl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.ushl.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VSHL_V3_I:%.*]] = bitcast <2 x i32> [[VSHL_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VSHL_V2_I]]
uint32x2_t test_vshl_u32(uint32x2_t a, int32x2_t b) {
  return vshl_u32(a, b);
}

// CHECK-LABEL: @test_vshl_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.ushl.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VSHL_V3_I:%.*]] = bitcast <1 x i64> [[VSHL_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VSHL_V2_I]]
uint64x1_t test_vshl_u64(uint64x1_t a, int64x1_t b) {
  return vshl_u64(a, b);
}

// CHECK-LABEL: @test_vshlq_s8(
// CHECK:   [[VSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VSHLQ_V_I]]
int8x16_t test_vshlq_s8(int8x16_t a, int8x16_t b) {
  return vshlq_s8(a, b);
}

// CHECK-LABEL: @test_vshlq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VSHLQ_V2_I]]
int16x8_t test_vshlq_s16(int16x8_t a, int16x8_t b) {
  return vshlq_s16(a, b);
}

// CHECK-LABEL: @test_vshlq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VSHLQ_V2_I]]
int32x4_t test_vshlq_s32(int32x4_t a, int32x4_t b) {
  return vshlq_s32(a, b);
}

// CHECK-LABEL: @test_vshlq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VSHLQ_V2_I]]
int64x2_t test_vshlq_s64(int64x2_t a, int64x2_t b) {
  return vshlq_s64(a, b);
}

// CHECK-LABEL: @test_vshlq_u8(
// CHECK:   [[VSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.ushl.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VSHLQ_V_I]]
uint8x16_t test_vshlq_u8(uint8x16_t a, int8x16_t b) {
  return vshlq_u8(a, b);
}

// CHECK-LABEL: @test_vshlq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.ushl.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VSHLQ_V2_I]]
uint16x8_t test_vshlq_u16(uint16x8_t a, int16x8_t b) {
  return vshlq_u16(a, b);
}

// CHECK-LABEL: @test_vshlq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.ushl.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VSHLQ_V2_I]]
uint32x4_t test_vshlq_u32(uint32x4_t a, int32x4_t b) {
  return vshlq_u32(a, b);
}

// CHECK-LABEL: @test_vshlq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.ushl.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VSHLQ_V2_I]]
uint64x2_t test_vshlq_u64(uint64x2_t a, int64x2_t b) {
  return vshlq_u64(a, b);
}

// CHECK-LABEL: @test_vqshl_s8(
// CHECK:   [[VQSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VQSHL_V_I]]
int8x8_t test_vqshl_s8(int8x8_t a, int8x8_t b) {
  return vqshl_s8(a, b);
}

// CHECK-LABEL: @test_vqshl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQSHL_V3_I:%.*]] = bitcast <4 x i16> [[VQSHL_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQSHL_V2_I]]
int16x4_t test_vqshl_s16(int16x4_t a, int16x4_t b) {
  return vqshl_s16(a, b);
}

// CHECK-LABEL: @test_vqshl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQSHL_V3_I:%.*]] = bitcast <2 x i32> [[VQSHL_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQSHL_V2_I]]
int32x2_t test_vqshl_s32(int32x2_t a, int32x2_t b) {
  return vqshl_s32(a, b);
}

// CHECK-LABEL: @test_vqshl_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VQSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VQSHL_V3_I:%.*]] = bitcast <1 x i64> [[VQSHL_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQSHL_V2_I]]
int64x1_t test_vqshl_s64(int64x1_t a, int64x1_t b) {
  return vqshl_s64(a, b);
}

// CHECK-LABEL: @test_vqshl_u8(
// CHECK:   [[VQSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VQSHL_V_I]]
uint8x8_t test_vqshl_u8(uint8x8_t a, int8x8_t b) {
  return vqshl_u8(a, b);
}

// CHECK-LABEL: @test_vqshl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQSHL_V3_I:%.*]] = bitcast <4 x i16> [[VQSHL_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQSHL_V2_I]]
uint16x4_t test_vqshl_u16(uint16x4_t a, int16x4_t b) {
  return vqshl_u16(a, b);
}

// CHECK-LABEL: @test_vqshl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQSHL_V3_I:%.*]] = bitcast <2 x i32> [[VQSHL_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQSHL_V2_I]]
uint32x2_t test_vqshl_u32(uint32x2_t a, int32x2_t b) {
  return vqshl_u32(a, b);
}

// CHECK-LABEL: @test_vqshl_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VQSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VQSHL_V3_I:%.*]] = bitcast <1 x i64> [[VQSHL_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQSHL_V2_I]]
uint64x1_t test_vqshl_u64(uint64x1_t a, int64x1_t b) {
  return vqshl_u64(a, b);
}

// CHECK-LABEL: @test_vqshlq_s8(
// CHECK:   [[VQSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VQSHLQ_V_I]]
int8x16_t test_vqshlq_s8(int8x16_t a, int8x16_t b) {
  return vqshlq_s8(a, b);
}

// CHECK-LABEL: @test_vqshlq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VQSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQSHLQ_V2_I]]
int16x8_t test_vqshlq_s16(int16x8_t a, int16x8_t b) {
  return vqshlq_s16(a, b);
}

// CHECK-LABEL: @test_vqshlq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VQSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQSHLQ_V2_I]]
int32x4_t test_vqshlq_s32(int32x4_t a, int32x4_t b) {
  return vqshlq_s32(a, b);
}

// CHECK-LABEL: @test_vqshlq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VQSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VQSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQSHLQ_V2_I]]
int64x2_t test_vqshlq_s64(int64x2_t a, int64x2_t b) {
  return vqshlq_s64(a, b);
}

// CHECK-LABEL: @test_vqshlq_u8(
// CHECK:   [[VQSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VQSHLQ_V_I]]
uint8x16_t test_vqshlq_u8(uint8x16_t a, int8x16_t b) {
  return vqshlq_u8(a, b);
}

// CHECK-LABEL: @test_vqshlq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VQSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQSHLQ_V2_I]]
uint16x8_t test_vqshlq_u16(uint16x8_t a, int16x8_t b) {
  return vqshlq_u16(a, b);
}

// CHECK-LABEL: @test_vqshlq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VQSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQSHLQ_V2_I]]
uint32x4_t test_vqshlq_u32(uint32x4_t a, int32x4_t b) {
  return vqshlq_u32(a, b);
}

// CHECK-LABEL: @test_vqshlq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VQSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VQSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQSHLQ_V2_I]]
uint64x2_t test_vqshlq_u64(uint64x2_t a, int64x2_t b) {
  return vqshlq_u64(a, b);
}

// CHECK-LABEL: @test_vrshl_s8(
// CHECK:   [[VRSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VRSHL_V_I]]
int8x8_t test_vrshl_s8(int8x8_t a, int8x8_t b) {
  return vrshl_s8(a, b);
}

// CHECK-LABEL: @test_vrshl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VRSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VRSHL_V3_I:%.*]] = bitcast <4 x i16> [[VRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VRSHL_V2_I]]
int16x4_t test_vrshl_s16(int16x4_t a, int16x4_t b) {
  return vrshl_s16(a, b);
}

// CHECK-LABEL: @test_vrshl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VRSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VRSHL_V3_I:%.*]] = bitcast <2 x i32> [[VRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VRSHL_V2_I]]
int32x2_t test_vrshl_s32(int32x2_t a, int32x2_t b) {
  return vrshl_s32(a, b);
}

// CHECK-LABEL: @test_vrshl_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VRSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.srshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VRSHL_V3_I:%.*]] = bitcast <1 x i64> [[VRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VRSHL_V2_I]]
int64x1_t test_vrshl_s64(int64x1_t a, int64x1_t b) {
  return vrshl_s64(a, b);
}

// CHECK-LABEL: @test_vrshl_u8(
// CHECK:   [[VRSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VRSHL_V_I]]
uint8x8_t test_vrshl_u8(uint8x8_t a, int8x8_t b) {
  return vrshl_u8(a, b);
}

// CHECK-LABEL: @test_vrshl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VRSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VRSHL_V3_I:%.*]] = bitcast <4 x i16> [[VRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VRSHL_V2_I]]
uint16x4_t test_vrshl_u16(uint16x4_t a, int16x4_t b) {
  return vrshl_u16(a, b);
}

// CHECK-LABEL: @test_vrshl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VRSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VRSHL_V3_I:%.*]] = bitcast <2 x i32> [[VRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VRSHL_V2_I]]
uint32x2_t test_vrshl_u32(uint32x2_t a, int32x2_t b) {
  return vrshl_u32(a, b);
}

// CHECK-LABEL: @test_vrshl_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VRSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.urshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VRSHL_V3_I:%.*]] = bitcast <1 x i64> [[VRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VRSHL_V2_I]]
uint64x1_t test_vrshl_u64(uint64x1_t a, int64x1_t b) {
  return vrshl_u64(a, b);
}

// CHECK-LABEL: @test_vrshlq_s8(
// CHECK:   [[VRSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VRSHLQ_V_I]]
int8x16_t test_vrshlq_s8(int8x16_t a, int8x16_t b) {
  return vrshlq_s8(a, b);
}

// CHECK-LABEL: @test_vrshlq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VRSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VRSHLQ_V2_I]]
int16x8_t test_vrshlq_s16(int16x8_t a, int16x8_t b) {
  return vrshlq_s16(a, b);
}

// CHECK-LABEL: @test_vrshlq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VRSHLQ_V2_I]]
int32x4_t test_vrshlq_s32(int32x4_t a, int32x4_t b) {
  return vrshlq_s32(a, b);
}

// CHECK-LABEL: @test_vrshlq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VRSHLQ_V2_I]]
int64x2_t test_vrshlq_s64(int64x2_t a, int64x2_t b) {
  return vrshlq_s64(a, b);
}

// CHECK-LABEL: @test_vrshlq_u8(
// CHECK:   [[VRSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VRSHLQ_V_I]]
uint8x16_t test_vrshlq_u8(uint8x16_t a, int8x16_t b) {
  return vrshlq_u8(a, b);
}

// CHECK-LABEL: @test_vrshlq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VRSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VRSHLQ_V2_I]]
uint16x8_t test_vrshlq_u16(uint16x8_t a, int16x8_t b) {
  return vrshlq_u16(a, b);
}

// CHECK-LABEL: @test_vrshlq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VRSHLQ_V2_I]]
uint32x4_t test_vrshlq_u32(uint32x4_t a, int32x4_t b) {
  return vrshlq_u32(a, b);
}

// CHECK-LABEL: @test_vrshlq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VRSHLQ_V2_I]]
uint64x2_t test_vrshlq_u64(uint64x2_t a, int64x2_t b) {
  return vrshlq_u64(a, b);
}

// CHECK-LABEL: @test_vqrshl_s8(
// CHECK:   [[VQRSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VQRSHL_V_I]]
int8x8_t test_vqrshl_s8(int8x8_t a, int8x8_t b) {
  return vqrshl_s8(a, b);
}

// CHECK-LABEL: @test_vqrshl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQRSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQRSHL_V3_I:%.*]] = bitcast <4 x i16> [[VQRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQRSHL_V2_I]]
int16x4_t test_vqrshl_s16(int16x4_t a, int16x4_t b) {
  return vqrshl_s16(a, b);
}

// CHECK-LABEL: @test_vqrshl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQRSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQRSHL_V3_I:%.*]] = bitcast <2 x i32> [[VQRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQRSHL_V2_I]]
int32x2_t test_vqrshl_s32(int32x2_t a, int32x2_t b) {
  return vqrshl_s32(a, b);
}

// CHECK-LABEL: @test_vqrshl_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VQRSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqrshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VQRSHL_V3_I:%.*]] = bitcast <1 x i64> [[VQRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQRSHL_V2_I]]
int64x1_t test_vqrshl_s64(int64x1_t a, int64x1_t b) {
  return vqrshl_s64(a, b);
}

// CHECK-LABEL: @test_vqrshl_u8(
// CHECK:   [[VQRSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VQRSHL_V_I]]
uint8x8_t test_vqrshl_u8(uint8x8_t a, int8x8_t b) {
  return vqrshl_u8(a, b);
}

// CHECK-LABEL: @test_vqrshl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQRSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQRSHL_V3_I:%.*]] = bitcast <4 x i16> [[VQRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQRSHL_V2_I]]
uint16x4_t test_vqrshl_u16(uint16x4_t a, int16x4_t b) {
  return vqrshl_u16(a, b);
}

// CHECK-LABEL: @test_vqrshl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQRSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqrshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQRSHL_V3_I:%.*]] = bitcast <2 x i32> [[VQRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQRSHL_V2_I]]
uint32x2_t test_vqrshl_u32(uint32x2_t a, int32x2_t b) {
  return vqrshl_u32(a, b);
}

// CHECK-LABEL: @test_vqrshl_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VQRSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqrshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   [[VQRSHL_V3_I:%.*]] = bitcast <1 x i64> [[VQRSHL_V2_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQRSHL_V2_I]]
uint64x1_t test_vqrshl_u64(uint64x1_t a, int64x1_t b) {
  return vqrshl_u64(a, b);
}

// CHECK-LABEL: @test_vqrshlq_s8(
// CHECK:   [[VQRSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqrshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VQRSHLQ_V_I]]
int8x16_t test_vqrshlq_s8(int8x16_t a, int8x16_t b) {
  return vqrshlq_s8(a, b);
}

// CHECK-LABEL: @test_vqrshlq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQRSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQRSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQRSHLQ_V2_I]]
int16x8_t test_vqrshlq_s16(int16x8_t a, int16x8_t b) {
  return vqrshlq_s16(a, b);
}

// CHECK-LABEL: @test_vqrshlq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQRSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQRSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQRSHLQ_V2_I]]
int32x4_t test_vqrshlq_s32(int32x4_t a, int32x4_t b) {
  return vqrshlq_s32(a, b);
}

// CHECK-LABEL: @test_vqrshlq_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQRSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqrshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VQRSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VQRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQRSHLQ_V2_I]]
int64x2_t test_vqrshlq_s64(int64x2_t a, int64x2_t b) {
  return vqrshlq_s64(a, b);
}

// CHECK-LABEL: @test_vqrshlq_u8(
// CHECK:   [[VQRSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqrshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VQRSHLQ_V_I]]
uint8x16_t test_vqrshlq_u8(uint8x16_t a, int8x16_t b) {
  return vqrshlq_u8(a, b);
}

// CHECK-LABEL: @test_vqrshlq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQRSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqrshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQRSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQRSHLQ_V2_I]]
uint16x8_t test_vqrshlq_u16(uint16x8_t a, int16x8_t b) {
  return vqrshlq_u16(a, b);
}

// CHECK-LABEL: @test_vqrshlq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQRSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqrshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQRSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQRSHLQ_V2_I]]
uint32x4_t test_vqrshlq_u32(uint32x4_t a, int32x4_t b) {
  return vqrshlq_u32(a, b);
}

// CHECK-LABEL: @test_vqrshlq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQRSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqrshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VQRSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VQRSHLQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQRSHLQ_V2_I]]
uint64x2_t test_vqrshlq_u64(uint64x2_t a, int64x2_t b) {
  return vqrshlq_u64(a, b);
}

// CHECK-LABEL: @test_vsli_n_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VSLI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLI_N]], <1 x i64> [[VSLI_N1]], i32 0)
// CHECK:   ret <1 x i64> [[VSLI_N2]]
poly64x1_t test_vsli_n_p64(poly64x1_t a, poly64x1_t b) {
  return vsli_n_p64(a, b, 0);
}

// CHECK-LABEL: @test_vsliq_n_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VSLI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> [[VSLI_N]], <2 x i64> [[VSLI_N1]], i32 0)
// CHECK:   ret <2 x i64> [[VSLI_N2]]
poly64x2_t test_vsliq_n_p64(poly64x2_t a, poly64x2_t b) {
  return vsliq_n_p64(a, b, 0);
}

// CHECK-LABEL: @test_vmax_s8(
// CHECK:   [[VMAX_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.smax.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VMAX_I]]
int8x8_t test_vmax_s8(int8x8_t a, int8x8_t b) {
  return vmax_s8(a, b);
}

// CHECK-LABEL: @test_vmax_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.smax.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VMAX2_I]]
int16x4_t test_vmax_s16(int16x4_t a, int16x4_t b) {
  return vmax_s16(a, b);
}

// CHECK-LABEL: @test_vmax_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.smax.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VMAX2_I]]
int32x2_t test_vmax_s32(int32x2_t a, int32x2_t b) {
  return vmax_s32(a, b);
}

// CHECK-LABEL: @test_vmax_u8(
// CHECK:   [[VMAX_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.umax.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VMAX_I]]
uint8x8_t test_vmax_u8(uint8x8_t a, uint8x8_t b) {
  return vmax_u8(a, b);
}

// CHECK-LABEL: @test_vmax_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.umax.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VMAX2_I]]
uint16x4_t test_vmax_u16(uint16x4_t a, uint16x4_t b) {
  return vmax_u16(a, b);
}

// CHECK-LABEL: @test_vmax_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.umax.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VMAX2_I]]
uint32x2_t test_vmax_u32(uint32x2_t a, uint32x2_t b) {
  return vmax_u32(a, b);
}

// CHECK-LABEL: @test_vmax_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmax.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VMAX2_I]]
float32x2_t test_vmax_f32(float32x2_t a, float32x2_t b) {
  return vmax_f32(a, b);
}

// CHECK-LABEL: @test_vmaxq_s8(
// CHECK:   [[VMAX_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.smax.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VMAX_I]]
int8x16_t test_vmaxq_s8(int8x16_t a, int8x16_t b) {
  return vmaxq_s8(a, b);
}

// CHECK-LABEL: @test_vmaxq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smax.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VMAX2_I]]
int16x8_t test_vmaxq_s16(int16x8_t a, int16x8_t b) {
  return vmaxq_s16(a, b);
}

// CHECK-LABEL: @test_vmaxq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smax.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VMAX2_I]]
int32x4_t test_vmaxq_s32(int32x4_t a, int32x4_t b) {
  return vmaxq_s32(a, b);
}

// CHECK-LABEL: @test_vmaxq_u8(
// CHECK:   [[VMAX_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.umax.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VMAX_I]]
uint8x16_t test_vmaxq_u8(uint8x16_t a, uint8x16_t b) {
  return vmaxq_u8(a, b);
}

// CHECK-LABEL: @test_vmaxq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umax.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VMAX2_I]]
uint16x8_t test_vmaxq_u16(uint16x8_t a, uint16x8_t b) {
  return vmaxq_u16(a, b);
}

// CHECK-LABEL: @test_vmaxq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umax.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VMAX2_I]]
uint32x4_t test_vmaxq_u32(uint32x4_t a, uint32x4_t b) {
  return vmaxq_u32(a, b);
}

// CHECK-LABEL: @test_vmaxq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmax.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VMAX2_I]]
float32x4_t test_vmaxq_f32(float32x4_t a, float32x4_t b) {
  return vmaxq_f32(a, b);
}

// CHECK-LABEL: @test_vmaxq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmax.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VMAX2_I]]
float64x2_t test_vmaxq_f64(float64x2_t a, float64x2_t b) {
  return vmaxq_f64(a, b);
}

// CHECK-LABEL: @test_vmin_s8(
// CHECK:   [[VMIN_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.smin.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VMIN_I]]
int8x8_t test_vmin_s8(int8x8_t a, int8x8_t b) {
  return vmin_s8(a, b);
}

// CHECK-LABEL: @test_vmin_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.smin.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VMIN2_I]]
int16x4_t test_vmin_s16(int16x4_t a, int16x4_t b) {
  return vmin_s16(a, b);
}

// CHECK-LABEL: @test_vmin_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.smin.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VMIN2_I]]
int32x2_t test_vmin_s32(int32x2_t a, int32x2_t b) {
  return vmin_s32(a, b);
}

// CHECK-LABEL: @test_vmin_u8(
// CHECK:   [[VMIN_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.umin.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VMIN_I]]
uint8x8_t test_vmin_u8(uint8x8_t a, uint8x8_t b) {
  return vmin_u8(a, b);
}

// CHECK-LABEL: @test_vmin_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.umin.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VMIN2_I]]
uint16x4_t test_vmin_u16(uint16x4_t a, uint16x4_t b) {
  return vmin_u16(a, b);
}

// CHECK-LABEL: @test_vmin_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.umin.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VMIN2_I]]
uint32x2_t test_vmin_u32(uint32x2_t a, uint32x2_t b) {
  return vmin_u32(a, b);
}

// CHECK-LABEL: @test_vmin_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmin.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VMIN2_I]]
float32x2_t test_vmin_f32(float32x2_t a, float32x2_t b) {
  return vmin_f32(a, b);
}

// CHECK-LABEL: @test_vminq_s8(
// CHECK:   [[VMIN_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.smin.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VMIN_I]]
int8x16_t test_vminq_s8(int8x16_t a, int8x16_t b) {
  return vminq_s8(a, b);
}

// CHECK-LABEL: @test_vminq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smin.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VMIN2_I]]
int16x8_t test_vminq_s16(int16x8_t a, int16x8_t b) {
  return vminq_s16(a, b);
}

// CHECK-LABEL: @test_vminq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smin.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VMIN2_I]]
int32x4_t test_vminq_s32(int32x4_t a, int32x4_t b) {
  return vminq_s32(a, b);
}

// CHECK-LABEL: @test_vminq_u8(
// CHECK:   [[VMIN_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.umin.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VMIN_I]]
uint8x16_t test_vminq_u8(uint8x16_t a, uint8x16_t b) {
  return vminq_u8(a, b);
}

// CHECK-LABEL: @test_vminq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umin.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VMIN2_I]]
uint16x8_t test_vminq_u16(uint16x8_t a, uint16x8_t b) {
  return vminq_u16(a, b);
}

// CHECK-LABEL: @test_vminq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umin.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VMIN2_I]]
uint32x4_t test_vminq_u32(uint32x4_t a, uint32x4_t b) {
  return vminq_u32(a, b);
}

// CHECK-LABEL: @test_vminq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmin.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VMIN2_I]]
float32x4_t test_vminq_f32(float32x4_t a, float32x4_t b) {
  return vminq_f32(a, b);
}

// CHECK-LABEL: @test_vminq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmin.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VMIN2_I]]
float64x2_t test_vminq_f64(float64x2_t a, float64x2_t b) {
  return vminq_f64(a, b);
}

// CHECK-LABEL: @test_vmaxnm_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VMAXNM2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmaxnm.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VMAXNM2_I]]
float32x2_t test_vmaxnm_f32(float32x2_t a, float32x2_t b) {
  return vmaxnm_f32(a, b);
}

// CHECK-LABEL: @test_vmaxnmq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VMAXNM2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmaxnm.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VMAXNM2_I]]
float32x4_t test_vmaxnmq_f32(float32x4_t a, float32x4_t b) {
  return vmaxnmq_f32(a, b);
}

// CHECK-LABEL: @test_vmaxnmq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VMAXNM2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmaxnm.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VMAXNM2_I]]
float64x2_t test_vmaxnmq_f64(float64x2_t a, float64x2_t b) {
  return vmaxnmq_f64(a, b);
}

// CHECK-LABEL: @test_vminnm_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VMINNM2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fminnm.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VMINNM2_I]]
float32x2_t test_vminnm_f32(float32x2_t a, float32x2_t b) {
  return vminnm_f32(a, b);
}

// CHECK-LABEL: @test_vminnmq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VMINNM2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fminnm.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VMINNM2_I]]
float32x4_t test_vminnmq_f32(float32x4_t a, float32x4_t b) {
  return vminnmq_f32(a, b);
}

// CHECK-LABEL: @test_vminnmq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VMINNM2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fminnm.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VMINNM2_I]]
float64x2_t test_vminnmq_f64(float64x2_t a, float64x2_t b) {
  return vminnmq_f64(a, b);
}

// CHECK-LABEL: @test_vpmax_s8(
// CHECK:   [[VPMAX_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.smaxp.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VPMAX_I]]
int8x8_t test_vpmax_s8(int8x8_t a, int8x8_t b) {
  return vpmax_s8(a, b);
}

// CHECK-LABEL: @test_vpmax_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.smaxp.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VPMAX2_I]]
int16x4_t test_vpmax_s16(int16x4_t a, int16x4_t b) {
  return vpmax_s16(a, b);
}

// CHECK-LABEL: @test_vpmax_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.smaxp.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VPMAX2_I]]
int32x2_t test_vpmax_s32(int32x2_t a, int32x2_t b) {
  return vpmax_s32(a, b);
}

// CHECK-LABEL: @test_vpmax_u8(
// CHECK:   [[VPMAX_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.umaxp.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VPMAX_I]]
uint8x8_t test_vpmax_u8(uint8x8_t a, uint8x8_t b) {
  return vpmax_u8(a, b);
}

// CHECK-LABEL: @test_vpmax_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.umaxp.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VPMAX2_I]]
uint16x4_t test_vpmax_u16(uint16x4_t a, uint16x4_t b) {
  return vpmax_u16(a, b);
}

// CHECK-LABEL: @test_vpmax_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.umaxp.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VPMAX2_I]]
uint32x2_t test_vpmax_u32(uint32x2_t a, uint32x2_t b) {
  return vpmax_u32(a, b);
}

// CHECK-LABEL: @test_vpmax_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmaxp.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VPMAX2_I]]
float32x2_t test_vpmax_f32(float32x2_t a, float32x2_t b) {
  return vpmax_f32(a, b);
}

// CHECK-LABEL: @test_vpmaxq_s8(
// CHECK:   [[VPMAX_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.smaxp.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VPMAX_I]]
int8x16_t test_vpmaxq_s8(int8x16_t a, int8x16_t b) {
  return vpmaxq_s8(a, b);
}

// CHECK-LABEL: @test_vpmaxq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smaxp.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VPMAX2_I]]
int16x8_t test_vpmaxq_s16(int16x8_t a, int16x8_t b) {
  return vpmaxq_s16(a, b);
}

// CHECK-LABEL: @test_vpmaxq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smaxp.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VPMAX2_I]]
int32x4_t test_vpmaxq_s32(int32x4_t a, int32x4_t b) {
  return vpmaxq_s32(a, b);
}

// CHECK-LABEL: @test_vpmaxq_u8(
// CHECK:   [[VPMAX_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.umaxp.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VPMAX_I]]
uint8x16_t test_vpmaxq_u8(uint8x16_t a, uint8x16_t b) {
  return vpmaxq_u8(a, b);
}

// CHECK-LABEL: @test_vpmaxq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umaxp.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VPMAX2_I]]
uint16x8_t test_vpmaxq_u16(uint16x8_t a, uint16x8_t b) {
  return vpmaxq_u16(a, b);
}

// CHECK-LABEL: @test_vpmaxq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umaxp.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VPMAX2_I]]
uint32x4_t test_vpmaxq_u32(uint32x4_t a, uint32x4_t b) {
  return vpmaxq_u32(a, b);
}

// CHECK-LABEL: @test_vpmaxq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmaxp.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VPMAX2_I]]
float32x4_t test_vpmaxq_f32(float32x4_t a, float32x4_t b) {
  return vpmaxq_f32(a, b);
}

// CHECK-LABEL: @test_vpmaxq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VPMAX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmaxp.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VPMAX2_I]]
float64x2_t test_vpmaxq_f64(float64x2_t a, float64x2_t b) {
  return vpmaxq_f64(a, b);
}

// CHECK-LABEL: @test_vpmin_s8(
// CHECK:   [[VPMIN_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sminp.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VPMIN_I]]
int8x8_t test_vpmin_s8(int8x8_t a, int8x8_t b) {
  return vpmin_s8(a, b);
}

// CHECK-LABEL: @test_vpmin_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sminp.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VPMIN2_I]]
int16x4_t test_vpmin_s16(int16x4_t a, int16x4_t b) {
  return vpmin_s16(a, b);
}

// CHECK-LABEL: @test_vpmin_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sminp.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VPMIN2_I]]
int32x2_t test_vpmin_s32(int32x2_t a, int32x2_t b) {
  return vpmin_s32(a, b);
}

// CHECK-LABEL: @test_vpmin_u8(
// CHECK:   [[VPMIN_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uminp.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VPMIN_I]]
uint8x8_t test_vpmin_u8(uint8x8_t a, uint8x8_t b) {
  return vpmin_u8(a, b);
}

// CHECK-LABEL: @test_vpmin_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uminp.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VPMIN2_I]]
uint16x4_t test_vpmin_u16(uint16x4_t a, uint16x4_t b) {
  return vpmin_u16(a, b);
}

// CHECK-LABEL: @test_vpmin_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uminp.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VPMIN2_I]]
uint32x2_t test_vpmin_u32(uint32x2_t a, uint32x2_t b) {
  return vpmin_u32(a, b);
}

// CHECK-LABEL: @test_vpmin_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fminp.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VPMIN2_I]]
float32x2_t test_vpmin_f32(float32x2_t a, float32x2_t b) {
  return vpmin_f32(a, b);
}

// CHECK-LABEL: @test_vpminq_s8(
// CHECK:   [[VPMIN_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sminp.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VPMIN_I]]
int8x16_t test_vpminq_s8(int8x16_t a, int8x16_t b) {
  return vpminq_s8(a, b);
}

// CHECK-LABEL: @test_vpminq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sminp.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VPMIN2_I]]
int16x8_t test_vpminq_s16(int16x8_t a, int16x8_t b) {
  return vpminq_s16(a, b);
}

// CHECK-LABEL: @test_vpminq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sminp.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VPMIN2_I]]
int32x4_t test_vpminq_s32(int32x4_t a, int32x4_t b) {
  return vpminq_s32(a, b);
}

// CHECK-LABEL: @test_vpminq_u8(
// CHECK:   [[VPMIN_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uminp.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VPMIN_I]]
uint8x16_t test_vpminq_u8(uint8x16_t a, uint8x16_t b) {
  return vpminq_u8(a, b);
}

// CHECK-LABEL: @test_vpminq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uminp.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VPMIN2_I]]
uint16x8_t test_vpminq_u16(uint16x8_t a, uint16x8_t b) {
  return vpminq_u16(a, b);
}

// CHECK-LABEL: @test_vpminq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uminp.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VPMIN2_I]]
uint32x4_t test_vpminq_u32(uint32x4_t a, uint32x4_t b) {
  return vpminq_u32(a, b);
}

// CHECK-LABEL: @test_vpminq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fminp.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VPMIN2_I]]
float32x4_t test_vpminq_f32(float32x4_t a, float32x4_t b) {
  return vpminq_f32(a, b);
}

// CHECK-LABEL: @test_vpminq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VPMIN2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fminp.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VPMIN2_I]]
float64x2_t test_vpminq_f64(float64x2_t a, float64x2_t b) {
  return vpminq_f64(a, b);
}

// CHECK-LABEL: @test_vpmaxnm_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VPMAXNM2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmaxnmp.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VPMAXNM2_I]]
float32x2_t test_vpmaxnm_f32(float32x2_t a, float32x2_t b) {
  return vpmaxnm_f32(a, b);
}

// CHECK-LABEL: @test_vpmaxnmq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VPMAXNM2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmaxnmp.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VPMAXNM2_I]]
float32x4_t test_vpmaxnmq_f32(float32x4_t a, float32x4_t b) {
  return vpmaxnmq_f32(a, b);
}

// CHECK-LABEL: @test_vpmaxnmq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VPMAXNM2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmaxnmp.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VPMAXNM2_I]]
float64x2_t test_vpmaxnmq_f64(float64x2_t a, float64x2_t b) {
  return vpmaxnmq_f64(a, b);
}

// CHECK-LABEL: @test_vpminnm_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VPMINNM2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fminnmp.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VPMINNM2_I]]
float32x2_t test_vpminnm_f32(float32x2_t a, float32x2_t b) {
  return vpminnm_f32(a, b);
}

// CHECK-LABEL: @test_vpminnmq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VPMINNM2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fminnmp.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VPMINNM2_I]]
float32x4_t test_vpminnmq_f32(float32x4_t a, float32x4_t b) {
  return vpminnmq_f32(a, b);
}

// CHECK-LABEL: @test_vpminnmq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VPMINNM2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fminnmp.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VPMINNM2_I]]
float64x2_t test_vpminnmq_f64(float64x2_t a, float64x2_t b) {
  return vpminnmq_f64(a, b);
}

// CHECK-LABEL: @test_vpadd_s8(
// CHECK:   [[VPADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VPADD_V_I]]
int8x8_t test_vpadd_s8(int8x8_t a, int8x8_t b) {
  return vpadd_s8(a, b);
}

// CHECK-LABEL: @test_vpadd_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VPADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VPADD_V3_I:%.*]] = bitcast <4 x i16> [[VPADD_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VPADD_V2_I]]
int16x4_t test_vpadd_s16(int16x4_t a, int16x4_t b) {
  return vpadd_s16(a, b);
}

// CHECK-LABEL: @test_vpadd_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VPADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VPADD_V3_I:%.*]] = bitcast <2 x i32> [[VPADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VPADD_V2_I]]
int32x2_t test_vpadd_s32(int32x2_t a, int32x2_t b) {
  return vpadd_s32(a, b);
}

// CHECK-LABEL: @test_vpadd_u8(
// CHECK:   [[VPADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VPADD_V_I]]
uint8x8_t test_vpadd_u8(uint8x8_t a, uint8x8_t b) {
  return vpadd_u8(a, b);
}

// CHECK-LABEL: @test_vpadd_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VPADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VPADD_V3_I:%.*]] = bitcast <4 x i16> [[VPADD_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VPADD_V2_I]]
uint16x4_t test_vpadd_u16(uint16x4_t a, uint16x4_t b) {
  return vpadd_u16(a, b);
}

// CHECK-LABEL: @test_vpadd_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VPADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VPADD_V3_I:%.*]] = bitcast <2 x i32> [[VPADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VPADD_V2_I]]
uint32x2_t test_vpadd_u32(uint32x2_t a, uint32x2_t b) {
  return vpadd_u32(a, b);
}

// CHECK-LABEL: @test_vpadd_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VPADD_V2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   [[VPADD_V3_I:%.*]] = bitcast <2 x float> [[VPADD_V2_I]] to <8 x i8>
// CHECK:   ret <2 x float> [[VPADD_V2_I]]
float32x2_t test_vpadd_f32(float32x2_t a, float32x2_t b) {
  return vpadd_f32(a, b);
}

// CHECK-LABEL: @test_vpaddq_s8(
// CHECK:   [[VPADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VPADDQ_V_I]]
int8x16_t test_vpaddq_s8(int8x16_t a, int8x16_t b) {
  return vpaddq_s8(a, b);
}

// CHECK-LABEL: @test_vpaddq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VPADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VPADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VPADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VPADDQ_V2_I]]
int16x8_t test_vpaddq_s16(int16x8_t a, int16x8_t b) {
  return vpaddq_s16(a, b);
}

// CHECK-LABEL: @test_vpaddq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VPADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VPADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VPADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VPADDQ_V2_I]]
int32x4_t test_vpaddq_s32(int32x4_t a, int32x4_t b) {
  return vpaddq_s32(a, b);
}

// CHECK-LABEL: @test_vpaddq_u8(
// CHECK:   [[VPADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VPADDQ_V_I]]
uint8x16_t test_vpaddq_u8(uint8x16_t a, uint8x16_t b) {
  return vpaddq_u8(a, b);
}

// CHECK-LABEL: @test_vpaddq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VPADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VPADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VPADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VPADDQ_V2_I]]
uint16x8_t test_vpaddq_u16(uint16x8_t a, uint16x8_t b) {
  return vpaddq_u16(a, b);
}

// CHECK-LABEL: @test_vpaddq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VPADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VPADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VPADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VPADDQ_V2_I]]
uint32x4_t test_vpaddq_u32(uint32x4_t a, uint32x4_t b) {
  return vpaddq_u32(a, b);
}

// CHECK-LABEL: @test_vpaddq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VPADDQ_V2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.faddp.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   [[VPADDQ_V3_I:%.*]] = bitcast <4 x float> [[VPADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x float> [[VPADDQ_V2_I]]
float32x4_t test_vpaddq_f32(float32x4_t a, float32x4_t b) {
  return vpaddq_f32(a, b);
}

// CHECK-LABEL: @test_vpaddq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VPADDQ_V2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.faddp.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   [[VPADDQ_V3_I:%.*]] = bitcast <2 x double> [[VPADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x double> [[VPADDQ_V2_I]]
float64x2_t test_vpaddq_f64(float64x2_t a, float64x2_t b) {
  return vpaddq_f64(a, b);
}

// CHECK-LABEL: @test_vqdmulh_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQDMULH_V2_I]]
int16x4_t test_vqdmulh_s16(int16x4_t a, int16x4_t b) {
  return vqdmulh_s16(a, b);
}

// CHECK-LABEL: @test_vqdmulh_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQDMULH_V2_I]]
int32x2_t test_vqdmulh_s32(int32x2_t a, int32x2_t b) {
  return vqdmulh_s32(a, b);
}

// CHECK-LABEL: @test_vqdmulhq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQDMULHQ_V2_I]]
int16x8_t test_vqdmulhq_s16(int16x8_t a, int16x8_t b) {
  return vqdmulhq_s16(a, b);
}

// CHECK-LABEL: @test_vqdmulhq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULHQ_V2_I]]
int32x4_t test_vqdmulhq_s32(int32x4_t a, int32x4_t b) {
  return vqdmulhq_s32(a, b);
}

// CHECK-LABEL: @test_vqrdmulh_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQRDMULH_V2_I]]
int16x4_t test_vqrdmulh_s16(int16x4_t a, int16x4_t b) {
  return vqrdmulh_s16(a, b);
}

// CHECK-LABEL: @test_vqrdmulh_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQRDMULH_V2_I]]
int32x2_t test_vqrdmulh_s32(int32x2_t a, int32x2_t b) {
  return vqrdmulh_s32(a, b);
}

// CHECK-LABEL: @test_vqrdmulhq_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQRDMULHQ_V2_I]]
int16x8_t test_vqrdmulhq_s16(int16x8_t a, int16x8_t b) {
  return vqrdmulhq_s16(a, b);
}

// CHECK-LABEL: @test_vqrdmulhq_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQRDMULHQ_V2_I]]
int32x4_t test_vqrdmulhq_s32(int32x4_t a, int32x4_t b) {
  return vqrdmulhq_s32(a, b);
}

// CHECK-LABEL: @test_vmulx_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmulx.v2f32(<2 x float> %a, <2 x float> %b)
// CHECK:   ret <2 x float> [[VMULX2_I]]
float32x2_t test_vmulx_f32(float32x2_t a, float32x2_t b) {
  return vmulx_f32(a, b);
}

// CHECK-LABEL: @test_vmulxq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmulx.v4f32(<4 x float> %a, <4 x float> %b)
// CHECK:   ret <4 x float> [[VMULX2_I]]
float32x4_t test_vmulxq_f32(float32x4_t a, float32x4_t b) {
  return vmulxq_f32(a, b);
}

// CHECK-LABEL: @test_vmulxq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmulx.v2f64(<2 x double> %a, <2 x double> %b)
// CHECK:   ret <2 x double> [[VMULX2_I]]
float64x2_t test_vmulxq_f64(float64x2_t a, float64x2_t b) {
  return vmulxq_f64(a, b);
}

// CHECK-LABEL: @test_vshl_n_s8(
// CHECK:   [[VSHL_N:%.*]] = shl <8 x i8> %a, splat (i8 3)
// CHECK:   ret <8 x i8> [[VSHL_N]]
int8x8_t test_vshl_n_s8(int8x8_t a) {
  return vshl_n_s8(a, 3);
}

// CHECK-LABEL: @test_vshl_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSHL_N:%.*]] = shl <4 x i16> [[TMP1]], splat (i16 3)
// CHECK:   ret <4 x i16> [[VSHL_N]]
int16x4_t test_vshl_n_s16(int16x4_t a) {
  return vshl_n_s16(a, 3);
}

// CHECK-LABEL: @test_vshl_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VSHL_N:%.*]] = shl <2 x i32> [[TMP1]], splat (i32 3)
// CHECK:   ret <2 x i32> [[VSHL_N]]
int32x2_t test_vshl_n_s32(int32x2_t a) {
  return vshl_n_s32(a, 3);
}

// CHECK-LABEL: @test_vshlq_n_s8(
// CHECK:   [[VSHL_N:%.*]] = shl <16 x i8> %a, splat (i8 3)
// CHECK:   ret <16 x i8> [[VSHL_N]]
int8x16_t test_vshlq_n_s8(int8x16_t a) {
  return vshlq_n_s8(a, 3);
}

// CHECK-LABEL: @test_vshlq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSHL_N:%.*]] = shl <8 x i16> [[TMP1]], splat (i16 3)
// CHECK:   ret <8 x i16> [[VSHL_N]]
int16x8_t test_vshlq_n_s16(int16x8_t a) {
  return vshlq_n_s16(a, 3);
}

// CHECK-LABEL: @test_vshlq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VSHL_N:%.*]] = shl <4 x i32> [[TMP1]], splat (i32 3)
// CHECK:   ret <4 x i32> [[VSHL_N]]
int32x4_t test_vshlq_n_s32(int32x4_t a) {
  return vshlq_n_s32(a, 3);
}

// CHECK-LABEL: @test_vshlq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSHL_N:%.*]] = shl <2 x i64> [[TMP1]], splat (i64 3)
// CHECK:   ret <2 x i64> [[VSHL_N]]
int64x2_t test_vshlq_n_s64(int64x2_t a) {
  return vshlq_n_s64(a, 3);
}

// CHECK-LABEL: @test_vshl_n_u8(
// CHECK:   [[VSHL_N:%.*]] = shl <8 x i8> %a, splat (i8 3)
// CHECK:   ret <8 x i8> [[VSHL_N]]
uint8x8_t test_vshl_n_u8(uint8x8_t a) {
  return vshl_n_u8(a, 3);
}

// CHECK-LABEL: @test_vshl_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSHL_N:%.*]] = shl <4 x i16> [[TMP1]], splat (i16 3)
// CHECK:   ret <4 x i16> [[VSHL_N]]
uint16x4_t test_vshl_n_u16(uint16x4_t a) {
  return vshl_n_u16(a, 3);
}

// CHECK-LABEL: @test_vshl_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VSHL_N:%.*]] = shl <2 x i32> [[TMP1]], splat (i32 3)
// CHECK:   ret <2 x i32> [[VSHL_N]]
uint32x2_t test_vshl_n_u32(uint32x2_t a) {
  return vshl_n_u32(a, 3);
}

// CHECK-LABEL: @test_vshlq_n_u8(
// CHECK:   [[VSHL_N:%.*]] = shl <16 x i8> %a, splat (i8 3)
// CHECK:   ret <16 x i8> [[VSHL_N]]
uint8x16_t test_vshlq_n_u8(uint8x16_t a) {
  return vshlq_n_u8(a, 3);
}

// CHECK-LABEL: @test_vshlq_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSHL_N:%.*]] = shl <8 x i16> [[TMP1]], splat (i16 3)
// CHECK:   ret <8 x i16> [[VSHL_N]]
uint16x8_t test_vshlq_n_u16(uint16x8_t a) {
  return vshlq_n_u16(a, 3);
}

// CHECK-LABEL: @test_vshlq_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VSHL_N:%.*]] = shl <4 x i32> [[TMP1]], splat (i32 3)
// CHECK:   ret <4 x i32> [[VSHL_N]]
uint32x4_t test_vshlq_n_u32(uint32x4_t a) {
  return vshlq_n_u32(a, 3);
}

// CHECK-LABEL: @test_vshlq_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSHL_N:%.*]] = shl <2 x i64> [[TMP1]], splat (i64 3)
// CHECK:   ret <2 x i64> [[VSHL_N]]
uint64x2_t test_vshlq_n_u64(uint64x2_t a) {
  return vshlq_n_u64(a, 3);
}

// CHECK-LABEL: @test_vshr_n_s8(
// CHECK:   [[VSHR_N:%.*]] = ashr <8 x i8> %a, splat (i8 3)
// CHECK:   ret <8 x i8> [[VSHR_N]]
int8x8_t test_vshr_n_s8(int8x8_t a) {
  return vshr_n_s8(a, 3);
}

// CHECK-LABEL: @test_vshr_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSHR_N:%.*]] = ashr <4 x i16> [[TMP1]], splat (i16 3)
// CHECK:   ret <4 x i16> [[VSHR_N]]
int16x4_t test_vshr_n_s16(int16x4_t a) {
  return vshr_n_s16(a, 3);
}

// CHECK-LABEL: @test_vshr_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VSHR_N:%.*]] = ashr <2 x i32> [[TMP1]], splat (i32 3)
// CHECK:   ret <2 x i32> [[VSHR_N]]
int32x2_t test_vshr_n_s32(int32x2_t a) {
  return vshr_n_s32(a, 3);
}

// CHECK-LABEL: @test_vshrq_n_s8(
// CHECK:   [[VSHR_N:%.*]] = ashr <16 x i8> %a, splat (i8 3)
// CHECK:   ret <16 x i8> [[VSHR_N]]
int8x16_t test_vshrq_n_s8(int8x16_t a) {
  return vshrq_n_s8(a, 3);
}

// CHECK-LABEL: @test_vshrq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSHR_N:%.*]] = ashr <8 x i16> [[TMP1]], splat (i16 3)
// CHECK:   ret <8 x i16> [[VSHR_N]]
int16x8_t test_vshrq_n_s16(int16x8_t a) {
  return vshrq_n_s16(a, 3);
}

// CHECK-LABEL: @test_vshrq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VSHR_N:%.*]] = ashr <4 x i32> [[TMP1]], splat (i32 3)
// CHECK:   ret <4 x i32> [[VSHR_N]]
int32x4_t test_vshrq_n_s32(int32x4_t a) {
  return vshrq_n_s32(a, 3);
}

// CHECK-LABEL: @test_vshrq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSHR_N:%.*]] = ashr <2 x i64> [[TMP1]], splat (i64 3)
// CHECK:   ret <2 x i64> [[VSHR_N]]
int64x2_t test_vshrq_n_s64(int64x2_t a) {
  return vshrq_n_s64(a, 3);
}

// CHECK-LABEL: @test_vshr_n_u8(
// CHECK:   [[VSHR_N:%.*]] = lshr <8 x i8> %a, splat (i8 3)
// CHECK:   ret <8 x i8> [[VSHR_N]]
uint8x8_t test_vshr_n_u8(uint8x8_t a) {
  return vshr_n_u8(a, 3);
}

// CHECK-LABEL: @test_vshr_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSHR_N:%.*]] = lshr <4 x i16> [[TMP1]], splat (i16 3)
// CHECK:   ret <4 x i16> [[VSHR_N]]
uint16x4_t test_vshr_n_u16(uint16x4_t a) {
  return vshr_n_u16(a, 3);
}

// CHECK-LABEL: @test_vshr_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VSHR_N:%.*]] = lshr <2 x i32> [[TMP1]], splat (i32 3)
// CHECK:   ret <2 x i32> [[VSHR_N]]
uint32x2_t test_vshr_n_u32(uint32x2_t a) {
  return vshr_n_u32(a, 3);
}

// CHECK-LABEL: @test_vshrq_n_u8(
// CHECK:   [[VSHR_N:%.*]] = lshr <16 x i8> %a, splat (i8 3)
// CHECK:   ret <16 x i8> [[VSHR_N]]
uint8x16_t test_vshrq_n_u8(uint8x16_t a) {
  return vshrq_n_u8(a, 3);
}

// CHECK-LABEL: @test_vshrq_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSHR_N:%.*]] = lshr <8 x i16> [[TMP1]], splat (i16 3)
// CHECK:   ret <8 x i16> [[VSHR_N]]
uint16x8_t test_vshrq_n_u16(uint16x8_t a) {
  return vshrq_n_u16(a, 3);
}

// CHECK-LABEL: @test_vshrq_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VSHR_N:%.*]] = lshr <4 x i32> [[TMP1]], splat (i32 3)
// CHECK:   ret <4 x i32> [[VSHR_N]]
uint32x4_t test_vshrq_n_u32(uint32x4_t a) {
  return vshrq_n_u32(a, 3);
}

// CHECK-LABEL: @test_vshrq_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSHR_N:%.*]] = lshr <2 x i64> [[TMP1]], splat (i64 3)
// CHECK:   ret <2 x i64> [[VSHR_N]]
uint64x2_t test_vshrq_n_u64(uint64x2_t a) {
  return vshrq_n_u64(a, 3);
}

// CHECK-LABEL: @test_vsra_n_s8(
// CHECK:   [[VSRA_N:%.*]] = ashr <8 x i8> %b, splat (i8 3)
// CHECK:   [[TMP0:%.*]] = add <8 x i8> %a, [[VSRA_N]]
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vsra_n_s8(int8x8_t a, int8x8_t b) {
  return vsra_n_s8(a, b, 3);
}

// CHECK-LABEL: @test_vsra_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VSRA_N:%.*]] = ashr <4 x i16> [[TMP3]], splat (i16 3)
// CHECK:   [[TMP4:%.*]] = add <4 x i16> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <4 x i16> [[TMP4]]
int16x4_t test_vsra_n_s16(int16x4_t a, int16x4_t b) {
  return vsra_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vsra_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VSRA_N:%.*]] = ashr <2 x i32> [[TMP3]], splat (i32 3)
// CHECK:   [[TMP4:%.*]] = add <2 x i32> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <2 x i32> [[TMP4]]
int32x2_t test_vsra_n_s32(int32x2_t a, int32x2_t b) {
  return vsra_n_s32(a, b, 3);
}

// CHECK-LABEL: @test_vsraq_n_s8(
// CHECK:   [[VSRA_N:%.*]] = ashr <16 x i8> %b, splat (i8 3)
// CHECK:   [[TMP0:%.*]] = add <16 x i8> %a, [[VSRA_N]]
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vsraq_n_s8(int8x16_t a, int8x16_t b) {
  return vsraq_n_s8(a, b, 3);
}

// CHECK-LABEL: @test_vsraq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VSRA_N:%.*]] = ashr <8 x i16> [[TMP3]], splat (i16 3)
// CHECK:   [[TMP4:%.*]] = add <8 x i16> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <8 x i16> [[TMP4]]
int16x8_t test_vsraq_n_s16(int16x8_t a, int16x8_t b) {
  return vsraq_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vsraq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VSRA_N:%.*]] = ashr <4 x i32> [[TMP3]], splat (i32 3)
// CHECK:   [[TMP4:%.*]] = add <4 x i32> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <4 x i32> [[TMP4]]
int32x4_t test_vsraq_n_s32(int32x4_t a, int32x4_t b) {
  return vsraq_n_s32(a, b, 3);
}

// CHECK-LABEL: @test_vsraq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VSRA_N:%.*]] = ashr <2 x i64> [[TMP3]], splat (i64 3)
// CHECK:   [[TMP4:%.*]] = add <2 x i64> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <2 x i64> [[TMP4]]
int64x2_t test_vsraq_n_s64(int64x2_t a, int64x2_t b) {
  return vsraq_n_s64(a, b, 3);
}

// CHECK-LABEL: @test_vsra_n_u8(
// CHECK:   [[VSRA_N:%.*]] = lshr <8 x i8> %b, splat (i8 3)
// CHECK:   [[TMP0:%.*]] = add <8 x i8> %a, [[VSRA_N]]
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vsra_n_u8(uint8x8_t a, uint8x8_t b) {
  return vsra_n_u8(a, b, 3);
}

// CHECK-LABEL: @test_vsra_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VSRA_N:%.*]] = lshr <4 x i16> [[TMP3]], splat (i16 3)
// CHECK:   [[TMP4:%.*]] = add <4 x i16> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <4 x i16> [[TMP4]]
uint16x4_t test_vsra_n_u16(uint16x4_t a, uint16x4_t b) {
  return vsra_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vsra_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VSRA_N:%.*]] = lshr <2 x i32> [[TMP3]], splat (i32 3)
// CHECK:   [[TMP4:%.*]] = add <2 x i32> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <2 x i32> [[TMP4]]
uint32x2_t test_vsra_n_u32(uint32x2_t a, uint32x2_t b) {
  return vsra_n_u32(a, b, 3);
}

// CHECK-LABEL: @test_vsraq_n_u8(
// CHECK:   [[VSRA_N:%.*]] = lshr <16 x i8> %b, splat (i8 3)
// CHECK:   [[TMP0:%.*]] = add <16 x i8> %a, [[VSRA_N]]
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vsraq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vsraq_n_u8(a, b, 3);
}

// CHECK-LABEL: @test_vsraq_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VSRA_N:%.*]] = lshr <8 x i16> [[TMP3]], splat (i16 3)
// CHECK:   [[TMP4:%.*]] = add <8 x i16> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <8 x i16> [[TMP4]]
uint16x8_t test_vsraq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vsraq_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vsraq_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VSRA_N:%.*]] = lshr <4 x i32> [[TMP3]], splat (i32 3)
// CHECK:   [[TMP4:%.*]] = add <4 x i32> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <4 x i32> [[TMP4]]
uint32x4_t test_vsraq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vsraq_n_u32(a, b, 3);
}

// CHECK-LABEL: @test_vsraq_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VSRA_N:%.*]] = lshr <2 x i64> [[TMP3]], splat (i64 3)
// CHECK:   [[TMP4:%.*]] = add <2 x i64> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <2 x i64> [[TMP4]]
uint64x2_t test_vsraq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vsraq_n_u64(a, b, 3);
}

// CHECK-LABEL: @test_vrshr_n_s8(
// CHECK:   [[VRSHR_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> %a, <8 x i8> splat (i8 -3))
// CHECK:   ret <8 x i8> [[VRSHR_N]]
int8x8_t test_vrshr_n_s8(int8x8_t a) {
  return vrshr_n_s8(a, 3);
}

// CHECK-LABEL: @test_vrshr_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VRSHR_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> [[VRSHR_N]], <4 x i16> splat (i16 -3))
// CHECK:   ret <4 x i16> [[VRSHR_N1]]
int16x4_t test_vrshr_n_s16(int16x4_t a) {
  return vrshr_n_s16(a, 3);
}

// CHECK-LABEL: @test_vrshr_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VRSHR_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> [[VRSHR_N]], <2 x i32> splat (i32 -3))
// CHECK:   ret <2 x i32> [[VRSHR_N1]]
int32x2_t test_vrshr_n_s32(int32x2_t a) {
  return vrshr_n_s32(a, 3);
}

// CHECK-LABEL: @test_vrshrq_n_s8(
// CHECK:   [[VRSHR_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> %a, <16 x i8> splat (i8 -3))
// CHECK:   ret <16 x i8> [[VRSHR_N]]
int8x16_t test_vrshrq_n_s8(int8x16_t a) {
  return vrshrq_n_s8(a, 3);
}

// CHECK-LABEL: @test_vrshrq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VRSHR_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> [[VRSHR_N]], <8 x i16> splat (i16 -3))
// CHECK:   ret <8 x i16> [[VRSHR_N1]]
int16x8_t test_vrshrq_n_s16(int16x8_t a) {
  return vrshrq_n_s16(a, 3);
}

// CHECK-LABEL: @test_vrshrq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VRSHR_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> [[VRSHR_N]], <4 x i32> splat (i32 -3))
// CHECK:   ret <4 x i32> [[VRSHR_N1]]
int32x4_t test_vrshrq_n_s32(int32x4_t a) {
  return vrshrq_n_s32(a, 3);
}

// CHECK-LABEL: @test_vrshrq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VRSHR_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> [[VRSHR_N]], <2 x i64> splat (i64 -3))
// CHECK:   ret <2 x i64> [[VRSHR_N1]]
int64x2_t test_vrshrq_n_s64(int64x2_t a) {
  return vrshrq_n_s64(a, 3);
}

// CHECK-LABEL: @test_vrshr_n_u8(
// CHECK:   [[VRSHR_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> %a, <8 x i8> splat (i8 -3))
// CHECK:   ret <8 x i8> [[VRSHR_N]]
uint8x8_t test_vrshr_n_u8(uint8x8_t a) {
  return vrshr_n_u8(a, 3);
}

// CHECK-LABEL: @test_vrshr_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VRSHR_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> [[VRSHR_N]], <4 x i16> splat (i16 -3))
// CHECK:   ret <4 x i16> [[VRSHR_N1]]
uint16x4_t test_vrshr_n_u16(uint16x4_t a) {
  return vrshr_n_u16(a, 3);
}

// CHECK-LABEL: @test_vrshr_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VRSHR_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> [[VRSHR_N]], <2 x i32> splat (i32 -3))
// CHECK:   ret <2 x i32> [[VRSHR_N1]]
uint32x2_t test_vrshr_n_u32(uint32x2_t a) {
  return vrshr_n_u32(a, 3);
}

// CHECK-LABEL: @test_vrshrq_n_u8(
// CHECK:   [[VRSHR_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> %a, <16 x i8> splat (i8 -3))
// CHECK:   ret <16 x i8> [[VRSHR_N]]
uint8x16_t test_vrshrq_n_u8(uint8x16_t a) {
  return vrshrq_n_u8(a, 3);
}

// CHECK-LABEL: @test_vrshrq_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VRSHR_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> [[VRSHR_N]], <8 x i16> splat (i16 -3))
// CHECK:   ret <8 x i16> [[VRSHR_N1]]
uint16x8_t test_vrshrq_n_u16(uint16x8_t a) {
  return vrshrq_n_u16(a, 3);
}

// CHECK-LABEL: @test_vrshrq_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VRSHR_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> [[VRSHR_N]], <4 x i32> splat (i32 -3))
// CHECK:   ret <4 x i32> [[VRSHR_N1]]
uint32x4_t test_vrshrq_n_u32(uint32x4_t a) {
  return vrshrq_n_u32(a, 3);
}

// CHECK-LABEL: @test_vrshrq_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VRSHR_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> [[VRSHR_N]], <2 x i64> splat (i64 -3))
// CHECK:   ret <2 x i64> [[VRSHR_N1]]
uint64x2_t test_vrshrq_n_u64(uint64x2_t a) {
  return vrshrq_n_u64(a, 3);
}

// CHECK-LABEL: @test_vrsra_n_s8(
// CHECK:   [[VRSHR_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> %b, <8 x i8> splat (i8 -3))
// CHECK:   [[TMP0:%.*]] = add <8 x i8> %a, [[VRSHR_N]]
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vrsra_n_s8(int8x8_t a, int8x8_t b) {
  return vrsra_n_s8(a, b, 3);
}

// CHECK-LABEL: @test_vrsra_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VRSHR_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> [[VRSHR_N]], <4 x i16> splat (i16 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = add <4 x i16> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <4 x i16> [[TMP3]]
int16x4_t test_vrsra_n_s16(int16x4_t a, int16x4_t b) {
  return vrsra_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vrsra_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VRSHR_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> [[VRSHR_N]], <2 x i32> splat (i32 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = add <2 x i32> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <2 x i32> [[TMP3]]
int32x2_t test_vrsra_n_s32(int32x2_t a, int32x2_t b) {
  return vrsra_n_s32(a, b, 3);
}

// CHECK-LABEL: @test_vrsraq_n_s8(
// CHECK:   [[VRSHR_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> %b, <16 x i8> splat (i8 -3))
// CHECK:   [[TMP0:%.*]] = add <16 x i8> %a, [[VRSHR_N]]
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vrsraq_n_s8(int8x16_t a, int8x16_t b) {
  return vrsraq_n_s8(a, b, 3);
}

// CHECK-LABEL: @test_vrsraq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VRSHR_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> [[VRSHR_N]], <8 x i16> splat (i16 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = add <8 x i16> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <8 x i16> [[TMP3]]
int16x8_t test_vrsraq_n_s16(int16x8_t a, int16x8_t b) {
  return vrsraq_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vrsraq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VRSHR_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> [[VRSHR_N]], <4 x i32> splat (i32 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = add <4 x i32> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <4 x i32> [[TMP3]]
int32x4_t test_vrsraq_n_s32(int32x4_t a, int32x4_t b) {
  return vrsraq_n_s32(a, b, 3);
}

// CHECK-LABEL: @test_vrsraq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VRSHR_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> [[VRSHR_N]], <2 x i64> splat (i64 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = add <2 x i64> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <2 x i64> [[TMP3]]
int64x2_t test_vrsraq_n_s64(int64x2_t a, int64x2_t b) {
  return vrsraq_n_s64(a, b, 3);
}

// CHECK-LABEL: @test_vrsra_n_u8(
// CHECK:   [[VRSHR_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> %b, <8 x i8> splat (i8 -3))
// CHECK:   [[TMP0:%.*]] = add <8 x i8> %a, [[VRSHR_N]]
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vrsra_n_u8(uint8x8_t a, uint8x8_t b) {
  return vrsra_n_u8(a, b, 3);
}

// CHECK-LABEL: @test_vrsra_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VRSHR_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> [[VRSHR_N]], <4 x i16> splat (i16 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP3:%.*]] = add <4 x i16> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <4 x i16> [[TMP3]]
uint16x4_t test_vrsra_n_u16(uint16x4_t a, uint16x4_t b) {
  return vrsra_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vrsra_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VRSHR_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> [[VRSHR_N]], <2 x i32> splat (i32 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP3:%.*]] = add <2 x i32> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <2 x i32> [[TMP3]]
uint32x2_t test_vrsra_n_u32(uint32x2_t a, uint32x2_t b) {
  return vrsra_n_u32(a, b, 3);
}

// CHECK-LABEL: @test_vrsraq_n_u8(
// CHECK:   [[VRSHR_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> %b, <16 x i8> splat (i8 -3))
// CHECK:   [[TMP0:%.*]] = add <16 x i8> %a, [[VRSHR_N]]
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vrsraq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vrsraq_n_u8(a, b, 3);
}

// CHECK-LABEL: @test_vrsraq_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VRSHR_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> [[VRSHR_N]], <8 x i16> splat (i16 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP3:%.*]] = add <8 x i16> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <8 x i16> [[TMP3]]
uint16x8_t test_vrsraq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vrsraq_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vrsraq_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VRSHR_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> [[VRSHR_N]], <4 x i32> splat (i32 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP3:%.*]] = add <4 x i32> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <4 x i32> [[TMP3]]
uint32x4_t test_vrsraq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vrsraq_n_u32(a, b, 3);
}

// CHECK-LABEL: @test_vrsraq_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VRSHR_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> [[VRSHR_N]], <2 x i64> splat (i64 -3))
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = add <2 x i64> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <2 x i64> [[TMP3]]
uint64x2_t test_vrsraq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vrsraq_n_u64(a, b, 3);
}

// CHECK-LABEL: @test_vsri_n_s8(
// CHECK:   [[VSRI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// CHECK:   ret <8 x i8> [[VSRI_N]]
int8x8_t test_vsri_n_s8(int8x8_t a, int8x8_t b) {
  return vsri_n_s8(a, b, 3);
}

// CHECK-LABEL: @test_vsri_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VSRI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16> [[VSRI_N]], <4 x i16> [[VSRI_N1]], i32 3)
// CHECK:   ret <4 x i16> [[VSRI_N2]]
int16x4_t test_vsri_n_s16(int16x4_t a, int16x4_t b) {
  return vsri_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vsri_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VSRI_N2:%.*]] = call <2 x i32> @llvm.aarch64.neon.vsri.v2i32(<2 x i32> [[VSRI_N]], <2 x i32> [[VSRI_N1]], i32 3)
// CHECK:   ret <2 x i32> [[VSRI_N2]]
int32x2_t test_vsri_n_s32(int32x2_t a, int32x2_t b) {
  return vsri_n_s32(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_s8(
// CHECK:   [[VSRI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// CHECK:   ret <16 x i8> [[VSRI_N]]
int8x16_t test_vsriq_n_s8(int8x16_t a, int8x16_t b) {
  return vsriq_n_s8(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VSRI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16> [[VSRI_N]], <8 x i16> [[VSRI_N1]], i32 3)
// CHECK:   ret <8 x i16> [[VSRI_N2]]
int16x8_t test_vsriq_n_s16(int16x8_t a, int16x8_t b) {
  return vsriq_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VSRI_N2:%.*]] = call <4 x i32> @llvm.aarch64.neon.vsri.v4i32(<4 x i32> [[VSRI_N]], <4 x i32> [[VSRI_N1]], i32 3)
// CHECK:   ret <4 x i32> [[VSRI_N2]]
int32x4_t test_vsriq_n_s32(int32x4_t a, int32x4_t b) {
  return vsriq_n_s32(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VSRI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsri.v2i64(<2 x i64> [[VSRI_N]], <2 x i64> [[VSRI_N1]], i32 3)
// CHECK:   ret <2 x i64> [[VSRI_N2]]
int64x2_t test_vsriq_n_s64(int64x2_t a, int64x2_t b) {
  return vsriq_n_s64(a, b, 3);
}

// CHECK-LABEL: @test_vsri_n_u8(
// CHECK:   [[VSRI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// CHECK:   ret <8 x i8> [[VSRI_N]]
uint8x8_t test_vsri_n_u8(uint8x8_t a, uint8x8_t b) {
  return vsri_n_u8(a, b, 3);
}

// CHECK-LABEL: @test_vsri_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VSRI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16> [[VSRI_N]], <4 x i16> [[VSRI_N1]], i32 3)
// CHECK:   ret <4 x i16> [[VSRI_N2]]
uint16x4_t test_vsri_n_u16(uint16x4_t a, uint16x4_t b) {
  return vsri_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vsri_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VSRI_N2:%.*]] = call <2 x i32> @llvm.aarch64.neon.vsri.v2i32(<2 x i32> [[VSRI_N]], <2 x i32> [[VSRI_N1]], i32 3)
// CHECK:   ret <2 x i32> [[VSRI_N2]]
uint32x2_t test_vsri_n_u32(uint32x2_t a, uint32x2_t b) {
  return vsri_n_u32(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_u8(
// CHECK:   [[VSRI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// CHECK:   ret <16 x i8> [[VSRI_N]]
uint8x16_t test_vsriq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vsriq_n_u8(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VSRI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16> [[VSRI_N]], <8 x i16> [[VSRI_N1]], i32 3)
// CHECK:   ret <8 x i16> [[VSRI_N2]]
uint16x8_t test_vsriq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vsriq_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VSRI_N2:%.*]] = call <4 x i32> @llvm.aarch64.neon.vsri.v4i32(<4 x i32> [[VSRI_N]], <4 x i32> [[VSRI_N1]], i32 3)
// CHECK:   ret <4 x i32> [[VSRI_N2]]
uint32x4_t test_vsriq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vsriq_n_u32(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VSRI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsri.v2i64(<2 x i64> [[VSRI_N]], <2 x i64> [[VSRI_N1]], i32 3)
// CHECK:   ret <2 x i64> [[VSRI_N2]]
uint64x2_t test_vsriq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vsriq_n_u64(a, b, 3);
}

// CHECK-LABEL: @test_vsri_n_p8(
// CHECK:   [[VSRI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// CHECK:   ret <8 x i8> [[VSRI_N]]
poly8x8_t test_vsri_n_p8(poly8x8_t a, poly8x8_t b) {
  return vsri_n_p8(a, b, 3);
}

// CHECK-LABEL: @test_vsri_n_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VSRI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16> [[VSRI_N]], <4 x i16> [[VSRI_N1]], i32 15)
// CHECK:   ret <4 x i16> [[VSRI_N2]]
poly16x4_t test_vsri_n_p16(poly16x4_t a, poly16x4_t b) {
  return vsri_n_p16(a, b, 15);
}

// CHECK-LABEL: @test_vsriq_n_p8(
// CHECK:   [[VSRI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// CHECK:   ret <16 x i8> [[VSRI_N]]
poly8x16_t test_vsriq_n_p8(poly8x16_t a, poly8x16_t b) {
  return vsriq_n_p8(a, b, 3);
}

// CHECK-LABEL: @test_vsriq_n_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VSRI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16> [[VSRI_N]], <8 x i16> [[VSRI_N1]], i32 15)
// CHECK:   ret <8 x i16> [[VSRI_N2]]
poly16x8_t test_vsriq_n_p16(poly16x8_t a, poly16x8_t b) {
  return vsriq_n_p16(a, b, 15);
}

// CHECK-LABEL: @test_vsli_n_s8(
// CHECK:   [[VSLI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// CHECK:   ret <8 x i8> [[VSLI_N]]
int8x8_t test_vsli_n_s8(int8x8_t a, int8x8_t b) {
  return vsli_n_s8(a, b, 3);
}

// CHECK-LABEL: @test_vsli_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VSLI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> [[VSLI_N]], <4 x i16> [[VSLI_N1]], i32 3)
// CHECK:   ret <4 x i16> [[VSLI_N2]]
int16x4_t test_vsli_n_s16(int16x4_t a, int16x4_t b) {
  return vsli_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vsli_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VSLI_N2:%.*]] = call <2 x i32> @llvm.aarch64.neon.vsli.v2i32(<2 x i32> [[VSLI_N]], <2 x i32> [[VSLI_N1]], i32 3)
// CHECK:   ret <2 x i32> [[VSLI_N2]]
int32x2_t test_vsli_n_s32(int32x2_t a, int32x2_t b) {
  return vsli_n_s32(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_s8(
// CHECK:   [[VSLI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// CHECK:   ret <16 x i8> [[VSLI_N]]
int8x16_t test_vsliq_n_s8(int8x16_t a, int8x16_t b) {
  return vsliq_n_s8(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VSLI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> [[VSLI_N]], <8 x i16> [[VSLI_N1]], i32 3)
// CHECK:   ret <8 x i16> [[VSLI_N2]]
int16x8_t test_vsliq_n_s16(int16x8_t a, int16x8_t b) {
  return vsliq_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VSLI_N2:%.*]] = call <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32> [[VSLI_N]], <4 x i32> [[VSLI_N1]], i32 3)
// CHECK:   ret <4 x i32> [[VSLI_N2]]
int32x4_t test_vsliq_n_s32(int32x4_t a, int32x4_t b) {
  return vsliq_n_s32(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VSLI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> [[VSLI_N]], <2 x i64> [[VSLI_N1]], i32 3)
// CHECK:   ret <2 x i64> [[VSLI_N2]]
int64x2_t test_vsliq_n_s64(int64x2_t a, int64x2_t b) {
  return vsliq_n_s64(a, b, 3);
}

// CHECK-LABEL: @test_vsli_n_u8(
// CHECK:   [[VSLI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// CHECK:   ret <8 x i8> [[VSLI_N]]
uint8x8_t test_vsli_n_u8(uint8x8_t a, uint8x8_t b) {
  return vsli_n_u8(a, b, 3);
}

// CHECK-LABEL: @test_vsli_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VSLI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> [[VSLI_N]], <4 x i16> [[VSLI_N1]], i32 3)
// CHECK:   ret <4 x i16> [[VSLI_N2]]
uint16x4_t test_vsli_n_u16(uint16x4_t a, uint16x4_t b) {
  return vsli_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vsli_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   [[VSLI_N2:%.*]] = call <2 x i32> @llvm.aarch64.neon.vsli.v2i32(<2 x i32> [[VSLI_N]], <2 x i32> [[VSLI_N1]], i32 3)
// CHECK:   ret <2 x i32> [[VSLI_N2]]
uint32x2_t test_vsli_n_u32(uint32x2_t a, uint32x2_t b) {
  return vsli_n_u32(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_u8(
// CHECK:   [[VSLI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// CHECK:   ret <16 x i8> [[VSLI_N]]
uint8x16_t test_vsliq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vsliq_n_u8(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VSLI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> [[VSLI_N]], <8 x i16> [[VSLI_N1]], i32 3)
// CHECK:   ret <8 x i16> [[VSLI_N2]]
uint16x8_t test_vsliq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vsliq_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   [[VSLI_N2:%.*]] = call <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32> [[VSLI_N]], <4 x i32> [[VSLI_N1]], i32 3)
// CHECK:   ret <4 x i32> [[VSLI_N2]]
uint32x4_t test_vsliq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vsliq_n_u32(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VSLI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> [[VSLI_N]], <2 x i64> [[VSLI_N1]], i32 3)
// CHECK:   ret <2 x i64> [[VSLI_N2]]
uint64x2_t test_vsliq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vsliq_n_u64(a, b, 3);
}

// CHECK-LABEL: @test_vsli_n_p8(
// CHECK:   [[VSLI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// CHECK:   ret <8 x i8> [[VSLI_N]]
poly8x8_t test_vsli_n_p8(poly8x8_t a, poly8x8_t b) {
  return vsli_n_p8(a, b, 3);
}

// CHECK-LABEL: @test_vsli_n_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   [[VSLI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> [[VSLI_N]], <4 x i16> [[VSLI_N1]], i32 15)
// CHECK:   ret <4 x i16> [[VSLI_N2]]
poly16x4_t test_vsli_n_p16(poly16x4_t a, poly16x4_t b) {
  return vsli_n_p16(a, b, 15);
}

// CHECK-LABEL: @test_vsliq_n_p8(
// CHECK:   [[VSLI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// CHECK:   ret <16 x i8> [[VSLI_N]]
poly8x16_t test_vsliq_n_p8(poly8x16_t a, poly8x16_t b) {
  return vsliq_n_p8(a, b, 3);
}

// CHECK-LABEL: @test_vsliq_n_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   [[VSLI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> [[VSLI_N]], <8 x i16> [[VSLI_N1]], i32 15)
// CHECK:   ret <8 x i16> [[VSLI_N2]]
poly16x8_t test_vsliq_n_p16(poly16x8_t a, poly16x8_t b) {
  return vsliq_n_p16(a, b, 15);
}

// CHECK-LABEL: @test_vqshlu_n_s8(
// CHECK:   [[VQSHLU_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshlu.v8i8(<8 x i8> %a, <8 x i8> splat (i8 3))
// CHECK:   ret <8 x i8> [[VQSHLU_N]]
uint8x8_t test_vqshlu_n_s8(int8x8_t a) {
  return vqshlu_n_s8(a, 3);
}

// CHECK-LABEL: @test_vqshlu_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VQSHLU_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VQSHLU_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshlu.v4i16(<4 x i16> [[VQSHLU_N]], <4 x i16> splat (i16 3))
// CHECK:   ret <4 x i16> [[VQSHLU_N1]]
uint16x4_t test_vqshlu_n_s16(int16x4_t a) {
  return vqshlu_n_s16(a, 3);
}

// CHECK-LABEL: @test_vqshlu_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VQSHLU_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VQSHLU_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshlu.v2i32(<2 x i32> [[VQSHLU_N]], <2 x i32> splat (i32 3))
// CHECK:   ret <2 x i32> [[VQSHLU_N1]]
uint32x2_t test_vqshlu_n_s32(int32x2_t a) {
  return vqshlu_n_s32(a, 3);
}

// CHECK-LABEL: @test_vqshluq_n_s8(
// CHECK:   [[VQSHLU_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqshlu.v16i8(<16 x i8> %a, <16 x i8> splat (i8 3))
// CHECK:   ret <16 x i8> [[VQSHLU_N]]
uint8x16_t test_vqshluq_n_s8(int8x16_t a) {
  return vqshluq_n_s8(a, 3);
}

// CHECK-LABEL: @test_vqshluq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQSHLU_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHLU_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqshlu.v8i16(<8 x i16> [[VQSHLU_N]], <8 x i16> splat (i16 3))
// CHECK:   ret <8 x i16> [[VQSHLU_N1]]
uint16x8_t test_vqshluq_n_s16(int16x8_t a) {
  return vqshluq_n_s16(a, 3);
}

// CHECK-LABEL: @test_vqshluq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQSHLU_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHLU_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqshlu.v4i32(<4 x i32> [[VQSHLU_N]], <4 x i32> splat (i32 3))
// CHECK:   ret <4 x i32> [[VQSHLU_N1]]
uint32x4_t test_vqshluq_n_s32(int32x4_t a) {
  return vqshluq_n_s32(a, 3);
}

// CHECK-LABEL: @test_vqshluq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQSHLU_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHLU_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqshlu.v2i64(<2 x i64> [[VQSHLU_N]], <2 x i64> splat (i64 3))
// CHECK:   ret <2 x i64> [[VQSHLU_N1]]
uint64x2_t test_vqshluq_n_s64(int64x2_t a) {
  return vqshluq_n_s64(a, 3);
}

// CHECK-LABEL: @test_vshrn_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP2:%.*]] = ashr <8 x i16> [[TMP1]], splat (i16 3)
// CHECK:   [[VSHRN_N:%.*]] = trunc <8 x i16> [[TMP2]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VSHRN_N]]
int8x8_t test_vshrn_n_s16(int16x8_t a) {
  return vshrn_n_s16(a, 3);
}

// CHECK-LABEL: @test_vshrn_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP2:%.*]] = ashr <4 x i32> [[TMP1]], splat (i32 9)
// CHECK:   [[VSHRN_N:%.*]] = trunc <4 x i32> [[TMP2]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VSHRN_N]]
int16x4_t test_vshrn_n_s32(int32x4_t a) {
  return vshrn_n_s32(a, 9);
}

// CHECK-LABEL: @test_vshrn_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP2:%.*]] = ashr <2 x i64> [[TMP1]], splat (i64 19)
// CHECK:   [[VSHRN_N:%.*]] = trunc <2 x i64> [[TMP2]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VSHRN_N]]
int32x2_t test_vshrn_n_s64(int64x2_t a) {
  return vshrn_n_s64(a, 19);
}

// CHECK-LABEL: @test_vshrn_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP2:%.*]] = lshr <8 x i16> [[TMP1]], splat (i16 3)
// CHECK:   [[VSHRN_N:%.*]] = trunc <8 x i16> [[TMP2]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VSHRN_N]]
uint8x8_t test_vshrn_n_u16(uint16x8_t a) {
  return vshrn_n_u16(a, 3);
}

// CHECK-LABEL: @test_vshrn_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP2:%.*]] = lshr <4 x i32> [[TMP1]], splat (i32 9)
// CHECK:   [[VSHRN_N:%.*]] = trunc <4 x i32> [[TMP2]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VSHRN_N]]
uint16x4_t test_vshrn_n_u32(uint32x4_t a) {
  return vshrn_n_u32(a, 9);
}

// CHECK-LABEL: @test_vshrn_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP2:%.*]] = lshr <2 x i64> [[TMP1]], splat (i64 19)
// CHECK:   [[VSHRN_N:%.*]] = trunc <2 x i64> [[TMP2]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VSHRN_N]]
uint32x2_t test_vshrn_n_u64(uint64x2_t a) {
  return vshrn_n_u64(a, 19);
}

// CHECK-LABEL: @test_vshrn_high_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP2:%.*]] = ashr <8 x i16> [[TMP1]], splat (i16 3)
// CHECK:   [[VSHRN_N:%.*]] = trunc <8 x i16> [[TMP2]] to <8 x i8>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VSHRN_N]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vshrn_high_n_s16(int8x8_t a, int16x8_t b) {
  return vshrn_high_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vshrn_high_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP2:%.*]] = ashr <4 x i32> [[TMP1]], splat (i32 9)
// CHECK:   [[VSHRN_N:%.*]] = trunc <4 x i32> [[TMP2]] to <4 x i16>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VSHRN_N]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vshrn_high_n_s32(int16x4_t a, int32x4_t b) {
  return vshrn_high_n_s32(a, b, 9);
}

// CHECK-LABEL: @test_vshrn_high_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP2:%.*]] = ashr <2 x i64> [[TMP1]], splat (i64 19)
// CHECK:   [[VSHRN_N:%.*]] = trunc <2 x i64> [[TMP2]] to <2 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VSHRN_N]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vshrn_high_n_s64(int32x2_t a, int64x2_t b) {
  return vshrn_high_n_s64(a, b, 19);
}

// CHECK-LABEL: @test_vshrn_high_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[TMP2:%.*]] = lshr <8 x i16> [[TMP1]], splat (i16 3)
// CHECK:   [[VSHRN_N:%.*]] = trunc <8 x i16> [[TMP2]] to <8 x i8>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VSHRN_N]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
  return vshrn_high_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vshrn_high_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[TMP2:%.*]] = lshr <4 x i32> [[TMP1]], splat (i32 9)
// CHECK:   [[VSHRN_N:%.*]] = trunc <4 x i32> [[TMP2]] to <4 x i16>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VSHRN_N]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
  return vshrn_high_n_u32(a, b, 9);
}

// CHECK-LABEL: @test_vshrn_high_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP2:%.*]] = lshr <2 x i64> [[TMP1]], splat (i64 19)
// CHECK:   [[VSHRN_N:%.*]] = trunc <2 x i64> [[TMP2]] to <2 x i32>
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VSHRN_N]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
  return vshrn_high_n_u64(a, b, 19);
}

// CHECK-LABEL: @test_vqshrun_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHRUN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> [[VQSHRUN_N]], i32 3)
// CHECK:   ret <8 x i8> [[VQSHRUN_N1]]
uint8x8_t test_vqshrun_n_s16(int16x8_t a) {
  return vqshrun_n_s16(a, 3);
}

// CHECK-LABEL: @test_vqshrun_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHRUN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> [[VQSHRUN_N]], i32 9)
// CHECK:   ret <4 x i16> [[VQSHRUN_N1]]
uint16x4_t test_vqshrun_n_s32(int32x4_t a) {
  return vqshrun_n_s32(a, 9);
}

// CHECK-LABEL: @test_vqshrun_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHRUN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32(<2 x i64> [[VQSHRUN_N]], i32 19)
// CHECK:   ret <2 x i32> [[VQSHRUN_N1]]
uint32x2_t test_vqshrun_n_s64(int64x2_t a) {
  return vqshrun_n_s64(a, 19);
}

// CHECK-LABEL: @test_vqshrun_high_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHRUN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> [[VQSHRUN_N]], i32 3)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQSHRUN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vqshrun_high_n_s16(int8x8_t a, int16x8_t b) {
  return vqshrun_high_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vqshrun_high_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHRUN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> [[VQSHRUN_N]], i32 9)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQSHRUN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vqshrun_high_n_s32(int16x4_t a, int32x4_t b) {
  return vqshrun_high_n_s32(a, b, 9);
}

// CHECK-LABEL: @test_vqshrun_high_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHRUN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32(<2 x i64> [[VQSHRUN_N]], i32 19)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQSHRUN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vqshrun_high_n_s64(int32x2_t a, int64x2_t b) {
  return vqshrun_high_n_s64(a, b, 19);
}

// CHECK-LABEL: @test_vrshrn_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> [[VRSHRN_N]], i32 3)
// CHECK:   ret <8 x i8> [[VRSHRN_N1]]
int8x8_t test_vrshrn_n_s16(int16x8_t a) {
  return vrshrn_n_s16(a, 3);
}

// CHECK-LABEL: @test_vrshrn_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> [[VRSHRN_N]], i32 9)
// CHECK:   ret <4 x i16> [[VRSHRN_N1]]
int16x4_t test_vrshrn_n_s32(int32x4_t a) {
  return vrshrn_n_s32(a, 9);
}

// CHECK-LABEL: @test_vrshrn_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> [[VRSHRN_N]], i32 19)
// CHECK:   ret <2 x i32> [[VRSHRN_N1]]
int32x2_t test_vrshrn_n_s64(int64x2_t a) {
  return vrshrn_n_s64(a, 19);
}

// CHECK-LABEL: @test_vrshrn_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> [[VRSHRN_N]], i32 3)
// CHECK:   ret <8 x i8> [[VRSHRN_N1]]
uint8x8_t test_vrshrn_n_u16(uint16x8_t a) {
  return vrshrn_n_u16(a, 3);
}

// CHECK-LABEL: @test_vrshrn_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> [[VRSHRN_N]], i32 9)
// CHECK:   ret <4 x i16> [[VRSHRN_N1]]
uint16x4_t test_vrshrn_n_u32(uint32x4_t a) {
  return vrshrn_n_u32(a, 9);
}

// CHECK-LABEL: @test_vrshrn_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> [[VRSHRN_N]], i32 19)
// CHECK:   ret <2 x i32> [[VRSHRN_N1]]
uint32x2_t test_vrshrn_n_u64(uint64x2_t a) {
  return vrshrn_n_u64(a, 19);
}

// CHECK-LABEL: @test_vrshrn_high_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> [[VRSHRN_N]], i32 3)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VRSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vrshrn_high_n_s16(int8x8_t a, int16x8_t b) {
  return vrshrn_high_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vrshrn_high_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> [[VRSHRN_N]], i32 9)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VRSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vrshrn_high_n_s32(int16x4_t a, int32x4_t b) {
  return vrshrn_high_n_s32(a, b, 9);
}

// CHECK-LABEL: @test_vrshrn_high_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> [[VRSHRN_N]], i32 19)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VRSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vrshrn_high_n_s64(int32x2_t a, int64x2_t b) {
  return vrshrn_high_n_s64(a, b, 19);
}

// CHECK-LABEL: @test_vrshrn_high_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> [[VRSHRN_N]], i32 3)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VRSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vrshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
  return vrshrn_high_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vrshrn_high_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> [[VRSHRN_N]], i32 9)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VRSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vrshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
  return vrshrn_high_n_u32(a, b, 9);
}

// CHECK-LABEL: @test_vrshrn_high_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> [[VRSHRN_N]], i32 19)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VRSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vrshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
  return vrshrn_high_n_u64(a, b, 19);
}

// CHECK-LABEL: @test_vqrshrun_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQRSHRUN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> [[VQRSHRUN_N]], i32 3)
// CHECK:   ret <8 x i8> [[VQRSHRUN_N1]]
uint8x8_t test_vqrshrun_n_s16(int16x8_t a) {
  return vqrshrun_n_s16(a, 3);
}

// CHECK-LABEL: @test_vqrshrun_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQRSHRUN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> [[VQRSHRUN_N]], i32 9)
// CHECK:   ret <4 x i16> [[VQRSHRUN_N1]]
uint16x4_t test_vqrshrun_n_s32(int32x4_t a) {
  return vqrshrun_n_s32(a, 9);
}

// CHECK-LABEL: @test_vqrshrun_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQRSHRUN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64> [[VQRSHRUN_N]], i32 19)
// CHECK:   ret <2 x i32> [[VQRSHRUN_N1]]
uint32x2_t test_vqrshrun_n_s64(int64x2_t a) {
  return vqrshrun_n_s64(a, 19);
}

// CHECK-LABEL: @test_vqrshrun_high_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQRSHRUN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> [[VQRSHRUN_N]], i32 3)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQRSHRUN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vqrshrun_high_n_s16(int8x8_t a, int16x8_t b) {
  return vqrshrun_high_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vqrshrun_high_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQRSHRUN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> [[VQRSHRUN_N]], i32 9)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQRSHRUN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vqrshrun_high_n_s32(int16x4_t a, int32x4_t b) {
  return vqrshrun_high_n_s32(a, b, 9);
}

// CHECK-LABEL: @test_vqrshrun_high_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQRSHRUN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64> [[VQRSHRUN_N]], i32 19)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQRSHRUN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vqrshrun_high_n_s64(int32x2_t a, int64x2_t b) {
  return vqrshrun_high_n_s64(a, b, 19);
}

// CHECK-LABEL: @test_vqshrn_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16> [[VQSHRN_N]], i32 3)
// CHECK:   ret <8 x i8> [[VQSHRN_N1]]
int8x8_t test_vqshrn_n_s16(int16x8_t a) {
  return vqshrn_n_s16(a, 3);
}

// CHECK-LABEL: @test_vqshrn_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> [[VQSHRN_N]], i32 9)
// CHECK:   ret <4 x i16> [[VQSHRN_N1]]
int16x4_t test_vqshrn_n_s32(int32x4_t a) {
  return vqshrn_n_s32(a, 9);
}

// CHECK-LABEL: @test_vqshrn_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrn.v2i32(<2 x i64> [[VQSHRN_N]], i32 19)
// CHECK:   ret <2 x i32> [[VQSHRN_N1]]
int32x2_t test_vqshrn_n_s64(int64x2_t a) {
  return vqshrn_n_s64(a, 19);
}

// CHECK-LABEL: @test_vqshrn_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> [[VQSHRN_N]], i32 3)
// CHECK:   ret <8 x i8> [[VQSHRN_N1]]
uint8x8_t test_vqshrn_n_u16(uint16x8_t a) {
  return vqshrn_n_u16(a, 3);
}

// CHECK-LABEL: @test_vqshrn_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> [[VQSHRN_N]], i32 9)
// CHECK:   ret <4 x i16> [[VQSHRN_N1]]
uint16x4_t test_vqshrn_n_u32(uint32x4_t a) {
  return vqshrn_n_u32(a, 9);
}

// CHECK-LABEL: @test_vqshrn_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64> [[VQSHRN_N]], i32 19)
// CHECK:   ret <2 x i32> [[VQSHRN_N1]]
uint32x2_t test_vqshrn_n_u64(uint64x2_t a) {
  return vqshrn_n_u64(a, 19);
}

// CHECK-LABEL: @test_vqshrn_high_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16> [[VQSHRN_N]], i32 3)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vqshrn_high_n_s16(int8x8_t a, int16x8_t b) {
  return vqshrn_high_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vqshrn_high_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> [[VQSHRN_N]], i32 9)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vqshrn_high_n_s32(int16x4_t a, int32x4_t b) {
  return vqshrn_high_n_s32(a, b, 9);
}

// CHECK-LABEL: @test_vqshrn_high_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrn.v2i32(<2 x i64> [[VQSHRN_N]], i32 19)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vqshrn_high_n_s64(int32x2_t a, int64x2_t b) {
  return vqshrn_high_n_s64(a, b, 19);
}

// CHECK-LABEL: @test_vqshrn_high_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> [[VQSHRN_N]], i32 3)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vqshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
  return vqshrn_high_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vqshrn_high_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> [[VQSHRN_N]], i32 9)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vqshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
  return vqshrn_high_n_u32(a, b, 9);
}

// CHECK-LABEL: @test_vqshrn_high_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64> [[VQSHRN_N]], i32 19)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vqshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
  return vqshrn_high_n_u64(a, b, 19);
}

// CHECK-LABEL: @test_vqrshrn_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16> [[VQRSHRN_N]], i32 3)
// CHECK:   ret <8 x i8> [[VQRSHRN_N1]]
int8x8_t test_vqrshrn_n_s16(int16x8_t a) {
  return vqrshrn_n_s16(a, 3);
}

// CHECK-LABEL: @test_vqrshrn_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> [[VQRSHRN_N]], i32 9)
// CHECK:   ret <4 x i16> [[VQRSHRN_N1]]
int16x4_t test_vqrshrn_n_s32(int32x4_t a) {
  return vqrshrn_n_s32(a, 9);
}

// CHECK-LABEL: @test_vqrshrn_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshrn.v2i32(<2 x i64> [[VQRSHRN_N]], i32 19)
// CHECK:   ret <2 x i32> [[VQRSHRN_N1]]
int32x2_t test_vqrshrn_n_s64(int64x2_t a) {
  return vqrshrn_n_s64(a, 19);
}

// CHECK-LABEL: @test_vqrshrn_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16> [[VQRSHRN_N]], i32 3)
// CHECK:   ret <8 x i8> [[VQRSHRN_N1]]
uint8x8_t test_vqrshrn_n_u16(uint16x8_t a) {
  return vqrshrn_n_u16(a, 3);
}

// CHECK-LABEL: @test_vqrshrn_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> [[VQRSHRN_N]], i32 9)
// CHECK:   ret <4 x i16> [[VQRSHRN_N1]]
uint16x4_t test_vqrshrn_n_u32(uint32x4_t a) {
  return vqrshrn_n_u32(a, 9);
}

// CHECK-LABEL: @test_vqrshrn_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqrshrn.v2i32(<2 x i64> [[VQRSHRN_N]], i32 19)
// CHECK:   ret <2 x i32> [[VQRSHRN_N1]]
uint32x2_t test_vqrshrn_n_u64(uint64x2_t a) {
  return vqrshrn_n_u64(a, 19);
}

// CHECK-LABEL: @test_vqrshrn_high_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16> [[VQRSHRN_N]], i32 3)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQRSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
int8x16_t test_vqrshrn_high_n_s16(int8x8_t a, int16x8_t b) {
  return vqrshrn_high_n_s16(a, b, 3);
}

// CHECK-LABEL: @test_vqrshrn_high_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> [[VQRSHRN_N]], i32 9)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQRSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
int16x8_t test_vqrshrn_high_n_s32(int16x4_t a, int32x4_t b) {
  return vqrshrn_high_n_s32(a, b, 9);
}

// CHECK-LABEL: @test_vqrshrn_high_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshrn.v2i32(<2 x i64> [[VQRSHRN_N]], i32 19)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQRSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
int32x4_t test_vqrshrn_high_n_s64(int32x2_t a, int64x2_t b) {
  return vqrshrn_high_n_s64(a, b, 19);
}

// CHECK-LABEL: @test_vqrshrn_high_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16> [[VQRSHRN_N]], i32 3)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQRSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I]]
uint8x16_t test_vqrshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
  return vqrshrn_high_n_u16(a, b, 3);
}

// CHECK-LABEL: @test_vqrshrn_high_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> [[VQRSHRN_N]], i32 9)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQRSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I]]
uint16x8_t test_vqrshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
  return vqrshrn_high_n_u32(a, b, 9);
}

// CHECK-LABEL: @test_vqrshrn_high_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqrshrn.v2i32(<2 x i64> [[VQRSHRN_N]], i32 19)
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQRSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I]]
uint32x4_t test_vqrshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
  return vqrshrn_high_n_u64(a, b, 19);
}

// CHECK-LABEL: @test_vshll_n_s8(
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> %a to <8 x i16>
// CHECK:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], splat (i16 3)
// CHECK:   ret <8 x i16> [[VSHLL_N]]
int16x8_t test_vshll_n_s8(int8x8_t a) {
  return vshll_n_s8(a, 3);
}

// CHECK-LABEL: @test_vshll_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP2:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// CHECK:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], splat (i32 9)
// CHECK:   ret <4 x i32> [[VSHLL_N]]
int32x4_t test_vshll_n_s16(int16x4_t a) {
  return vshll_n_s16(a, 9);
}

// CHECK-LABEL: @test_vshll_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// CHECK:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], splat (i64 19)
// CHECK:   ret <2 x i64> [[VSHLL_N]]
int64x2_t test_vshll_n_s32(int32x2_t a) {
  return vshll_n_s32(a, 19);
}

// CHECK-LABEL: @test_vshll_n_u8(
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> %a to <8 x i16>
// CHECK:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], splat (i16 3)
// CHECK:   ret <8 x i16> [[VSHLL_N]]
uint16x8_t test_vshll_n_u8(uint8x8_t a) {
  return vshll_n_u8(a, 3);
}

// CHECK-LABEL: @test_vshll_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP2:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// CHECK:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], splat (i32 9)
// CHECK:   ret <4 x i32> [[VSHLL_N]]
uint32x4_t test_vshll_n_u16(uint16x4_t a) {
  return vshll_n_u16(a, 9);
}

// CHECK-LABEL: @test_vshll_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP2:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// CHECK:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], splat (i64 19)
// CHECK:   ret <2 x i64> [[VSHLL_N]]
uint64x2_t test_vshll_n_u32(uint32x2_t a) {
  return vshll_n_u32(a, 19);
}

// CHECK-LABEL: @test_vshll_high_n_s8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// CHECK:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], splat (i16 3)
// CHECK:   ret <8 x i16> [[VSHLL_N]]
int16x8_t test_vshll_high_n_s8(int8x16_t a) {
  return vshll_high_n_s8(a, 3);
}

// CHECK-LABEL: @test_vshll_high_n_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP2:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// CHECK:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], splat (i32 9)
// CHECK:   ret <4 x i32> [[VSHLL_N]]
int32x4_t test_vshll_high_n_s16(int16x8_t a) {
  return vshll_high_n_s16(a, 9);
}

// CHECK-LABEL: @test_vshll_high_n_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// CHECK:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], splat (i64 19)
// CHECK:   ret <2 x i64> [[VSHLL_N]]
int64x2_t test_vshll_high_n_s32(int32x4_t a) {
  return vshll_high_n_s32(a, 19);
}

// CHECK-LABEL: @test_vshll_high_n_u8(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// CHECK:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], splat (i16 3)
// CHECK:   ret <8 x i16> [[VSHLL_N]]
uint16x8_t test_vshll_high_n_u8(uint8x16_t a) {
  return vshll_high_n_u8(a, 3);
}

// CHECK-LABEL: @test_vshll_high_n_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[TMP2:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// CHECK:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], splat (i32 9)
// CHECK:   ret <4 x i32> [[VSHLL_N]]
uint32x4_t test_vshll_high_n_u16(uint16x8_t a) {
  return vshll_high_n_u16(a, 9);
}

// CHECK-LABEL: @test_vshll_high_n_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[TMP2:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// CHECK:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], splat (i64 19)
// CHECK:   ret <2 x i64> [[VSHLL_N]]
uint64x2_t test_vshll_high_n_u32(uint32x4_t a) {
  return vshll_high_n_u32(a, 19);
}

// CHECK-LABEL: @test_vmovl_s8(
// CHECK:   [[VMOVL_I:%.*]] = sext <8 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[VMOVL_I]]
int16x8_t test_vmovl_s8(int8x8_t a) {
  return vmovl_s8(a);
}

// CHECK-LABEL: @test_vmovl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VMOVL_I:%.*]] = sext <4 x i16> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[VMOVL_I]]
int32x4_t test_vmovl_s16(int16x4_t a) {
  return vmovl_s16(a);
}

// CHECK-LABEL: @test_vmovl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VMOVL_I:%.*]] = sext <2 x i32> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[VMOVL_I]]
int64x2_t test_vmovl_s32(int32x2_t a) {
  return vmovl_s32(a);
}

// CHECK-LABEL: @test_vmovl_u8(
// CHECK:   [[VMOVL_I:%.*]] = zext <8 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[VMOVL_I]]
uint16x8_t test_vmovl_u8(uint8x8_t a) {
  return vmovl_u8(a);
}

// CHECK-LABEL: @test_vmovl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VMOVL_I:%.*]] = zext <4 x i16> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[VMOVL_I]]
uint32x4_t test_vmovl_u16(uint16x4_t a) {
  return vmovl_u16(a);
}

// CHECK-LABEL: @test_vmovl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VMOVL_I:%.*]] = zext <2 x i32> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[VMOVL_I]]
uint64x2_t test_vmovl_u32(uint32x2_t a) {
  return vmovl_u32(a);
}

// CHECK-LABEL: @test_vmovl_high_s8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vmovl_high_s8(int8x16_t a) {
  return vmovl_high_s8(a);
}

// CHECK-LABEL: @test_vmovl_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP1]]
int32x4_t test_vmovl_high_s16(int16x8_t a) {
  return vmovl_high_s16(a);
}

// CHECK-LABEL: @test_vmovl_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP1]]
int64x2_t test_vmovl_high_s32(int32x4_t a) {
  return vmovl_high_s32(a);
}

// CHECK-LABEL: @test_vmovl_high_u8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vmovl_high_u8(uint8x16_t a) {
  return vmovl_high_u8(a);
}

// CHECK-LABEL: @test_vmovl_high_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP1]]
uint32x4_t test_vmovl_high_u16(uint16x8_t a) {
  return vmovl_high_u16(a);
}

// CHECK-LABEL: @test_vmovl_high_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP1]]
uint64x2_t test_vmovl_high_u32(uint32x4_t a) {
  return vmovl_high_u32(a);
}

// CHECK-LABEL: @test_vcvt_n_f32_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VCVT_N1:%.*]] = call <2 x float> @llvm.aarch64.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32> [[VCVT_N]], i32 31)
// CHECK:   ret <2 x float> [[VCVT_N1]]
float32x2_t test_vcvt_n_f32_s32(int32x2_t a) {
  return vcvt_n_f32_s32(a, 31);
}

// CHECK-LABEL: @test_vcvtq_n_f32_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VCVT_N1:%.*]] = call <4 x float> @llvm.aarch64.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32> [[VCVT_N]], i32 31)
// CHECK:   ret <4 x float> [[VCVT_N1]]
float32x4_t test_vcvtq_n_f32_s32(int32x4_t a) {
  return vcvtq_n_f32_s32(a, 31);
}

// CHECK-LABEL: @test_vcvtq_n_f64_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VCVT_N1:%.*]] = call <2 x double> @llvm.aarch64.neon.vcvtfxs2fp.v2f64.v2i64(<2 x i64> [[VCVT_N]], i32 50)
// CHECK:   ret <2 x double> [[VCVT_N1]]
float64x2_t test_vcvtq_n_f64_s64(int64x2_t a) {
  return vcvtq_n_f64_s64(a, 50);
}

// CHECK-LABEL: @test_vcvt_n_f32_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VCVT_N1:%.*]] = call <2 x float> @llvm.aarch64.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32> [[VCVT_N]], i32 31)
// CHECK:   ret <2 x float> [[VCVT_N1]]
float32x2_t test_vcvt_n_f32_u32(uint32x2_t a) {
  return vcvt_n_f32_u32(a, 31);
}

// CHECK-LABEL: @test_vcvtq_n_f32_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VCVT_N1:%.*]] = call <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> [[VCVT_N]], i32 31)
// CHECK:   ret <4 x float> [[VCVT_N1]]
float32x4_t test_vcvtq_n_f32_u32(uint32x4_t a) {
  return vcvtq_n_f32_u32(a, 31);
}

// CHECK-LABEL: @test_vcvtq_n_f64_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VCVT_N1:%.*]] = call <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64> [[VCVT_N]], i32 50)
// CHECK:   ret <2 x double> [[VCVT_N1]]
float64x2_t test_vcvtq_n_f64_u64(uint64x2_t a) {
  return vcvtq_n_f64_u64(a, 50);
}

// CHECK-LABEL: @test_vcvt_n_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VCVT_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float> [[VCVT_N]], i32 31)
// CHECK:   ret <2 x i32> [[VCVT_N1]]
int32x2_t test_vcvt_n_s32_f32(float32x2_t a) {
  return vcvt_n_s32_f32(a, 31);
}

// CHECK-LABEL: @test_vcvtq_n_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VCVT_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float> [[VCVT_N]], i32 31)
// CHECK:   ret <4 x i32> [[VCVT_N1]]
int32x4_t test_vcvtq_n_s32_f32(float32x4_t a) {
  return vcvtq_n_s32_f32(a, 31);
}

// CHECK-LABEL: @test_vcvtq_n_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[VCVT_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.vcvtfp2fxs.v2i64.v2f64(<2 x double> [[VCVT_N]], i32 50)
// CHECK:   ret <2 x i64> [[VCVT_N1]]
int64x2_t test_vcvtq_n_s64_f64(float64x2_t a) {
  return vcvtq_n_s64_f64(a, 50);
}

// CHECK-LABEL: @test_vcvt_n_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[VCVT_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float> [[VCVT_N]], i32 31)
// CHECK:   ret <2 x i32> [[VCVT_N1]]
uint32x2_t test_vcvt_n_u32_f32(float32x2_t a) {
  return vcvt_n_u32_f32(a, 31);
}

// CHECK-LABEL: @test_vcvtq_n_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[VCVT_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float> [[VCVT_N]], i32 31)
// CHECK:   ret <4 x i32> [[VCVT_N1]]
uint32x4_t test_vcvtq_n_u32_f32(float32x4_t a) {
  return vcvtq_n_u32_f32(a, 31);
}

// CHECK-LABEL: @test_vcvtq_n_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[VCVT_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.vcvtfp2fxu.v2i64.v2f64(<2 x double> [[VCVT_N]], i32 50)
// CHECK:   ret <2 x i64> [[VCVT_N1]]
uint64x2_t test_vcvtq_n_u64_f64(float64x2_t a) {
  return vcvtq_n_u64_f64(a, 50);
}

// CHECK-LABEL: @test_vaddl_s8(
// CHECK:   [[VMOVL_I_I:%.*]] = sext <8 x i8> %a to <8 x i16>
// CHECK:   [[VMOVL_I4_I:%.*]] = sext <8 x i8> %b to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vaddl_s8(int8x8_t a, int8x8_t b) {
  return vaddl_s8(a, b);
}

// CHECK-LABEL: @test_vaddl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = sext <4 x i16> %a to <4 x i32>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMOVL_I4_I:%.*]] = sext <4 x i16> %b to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vaddl_s16(int16x4_t a, int16x4_t b) {
  return vaddl_s16(a, b);
}

// CHECK-LABEL: @test_vaddl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = sext <2 x i32> %a to <2 x i64>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMOVL_I4_I:%.*]] = sext <2 x i32> %b to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
int64x2_t test_vaddl_s32(int32x2_t a, int32x2_t b) {
  return vaddl_s32(a, b);
}

// CHECK-LABEL: @test_vaddl_u8(
// CHECK:   [[VMOVL_I_I:%.*]] = zext <8 x i8> %a to <8 x i16>
// CHECK:   [[VMOVL_I4_I:%.*]] = zext <8 x i8> %b to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vaddl_u8(uint8x8_t a, uint8x8_t b) {
  return vaddl_u8(a, b);
}

// CHECK-LABEL: @test_vaddl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <4 x i16> %a to <4 x i32>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMOVL_I4_I:%.*]] = zext <4 x i16> %b to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vaddl_u16(uint16x4_t a, uint16x4_t b) {
  return vaddl_u16(a, b);
}

// CHECK-LABEL: @test_vaddl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <2 x i32> %a to <2 x i64>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMOVL_I4_I:%.*]] = zext <2 x i32> %b to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
uint64x2_t test_vaddl_u32(uint32x2_t a, uint32x2_t b) {
  return vaddl_u32(a, b);
}

// CHECK-LABEL: @test_vaddl_high_s8(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP1:%.*]] = sext <8 x i8> [[SHUFFLE_I_I10_I]] to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> [[TMP0]], [[TMP1]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vaddl_high_s8(int8x16_t a, int8x16_t b) {
  return vaddl_high_s8(a, b);
}

// CHECK-LABEL: @test_vaddl_high_s16(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I10_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = sext <4 x i16> [[SHUFFLE_I_I10_I]] to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> [[TMP1]], [[TMP3]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vaddl_high_s16(int16x8_t a, int16x8_t b) {
  return vaddl_high_s16(a, b);
}

// CHECK-LABEL: @test_vaddl_high_s32(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I10_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = sext <2 x i32> [[SHUFFLE_I_I10_I]] to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> [[TMP1]], [[TMP3]]
// CHECK:   ret <2 x i64> [[ADD_I]]
int64x2_t test_vaddl_high_s32(int32x4_t a, int32x4_t b) {
  return vaddl_high_s32(a, b);
}

// CHECK-LABEL: @test_vaddl_high_u8(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP1:%.*]] = zext <8 x i8> [[SHUFFLE_I_I10_I]] to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> [[TMP0]], [[TMP1]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vaddl_high_u8(uint8x16_t a, uint8x16_t b) {
  return vaddl_high_u8(a, b);
}

// CHECK-LABEL: @test_vaddl_high_u16(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I10_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = zext <4 x i16> [[SHUFFLE_I_I10_I]] to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> [[TMP1]], [[TMP3]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vaddl_high_u16(uint16x8_t a, uint16x8_t b) {
  return vaddl_high_u16(a, b);
}

// CHECK-LABEL: @test_vaddl_high_u32(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I10_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = zext <2 x i32> [[SHUFFLE_I_I10_I]] to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> [[TMP1]], [[TMP3]]
// CHECK:   ret <2 x i64> [[ADD_I]]
uint64x2_t test_vaddl_high_u32(uint32x4_t a, uint32x4_t b) {
  return vaddl_high_u32(a, b);
}

// CHECK-LABEL: @test_vaddw_s8(
// CHECK:   [[VMOVL_I_I:%.*]] = sext <8 x i8> %b to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vaddw_s8(int16x8_t a, int8x8_t b) {
  return vaddw_s8(a, b);
}

// CHECK-LABEL: @test_vaddw_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = sext <4 x i16> %b to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vaddw_s16(int32x4_t a, int16x4_t b) {
  return vaddw_s16(a, b);
}

// CHECK-LABEL: @test_vaddw_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = sext <2 x i32> %b to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
int64x2_t test_vaddw_s32(int64x2_t a, int32x2_t b) {
  return vaddw_s32(a, b);
}

// CHECK-LABEL: @test_vaddw_u8(
// CHECK:   [[VMOVL_I_I:%.*]] = zext <8 x i8> %b to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vaddw_u8(uint16x8_t a, uint8x8_t b) {
  return vaddw_u8(a, b);
}

// CHECK-LABEL: @test_vaddw_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <4 x i16> %b to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vaddw_u16(uint32x4_t a, uint16x4_t b) {
  return vaddw_u16(a, b);
}

// CHECK-LABEL: @test_vaddw_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <2 x i32> %b to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
uint64x2_t test_vaddw_u32(uint64x2_t a, uint32x2_t b) {
  return vaddw_u32(a, b);
}

// CHECK-LABEL: @test_vaddw_high_s8(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[TMP0]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vaddw_high_s8(int16x8_t a, int8x16_t b) {
  return vaddw_high_s8(a, b);
}

// CHECK-LABEL: @test_vaddw_high_s16(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[TMP1]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vaddw_high_s16(int32x4_t a, int16x8_t b) {
  return vaddw_high_s16(a, b);
}

// CHECK-LABEL: @test_vaddw_high_s32(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[TMP1]]
// CHECK:   ret <2 x i64> [[ADD_I]]
int64x2_t test_vaddw_high_s32(int64x2_t a, int32x4_t b) {
  return vaddw_high_s32(a, b);
}

// CHECK-LABEL: @test_vaddw_high_u8(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[TMP0]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vaddw_high_u8(uint16x8_t a, uint8x16_t b) {
  return vaddw_high_u8(a, b);
}

// CHECK-LABEL: @test_vaddw_high_u16(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[TMP1]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vaddw_high_u16(uint32x4_t a, uint16x8_t b) {
  return vaddw_high_u16(a, b);
}

// CHECK-LABEL: @test_vaddw_high_u32(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[TMP1]]
// CHECK:   ret <2 x i64> [[ADD_I]]
uint64x2_t test_vaddw_high_u32(uint64x2_t a, uint32x4_t b) {
  return vaddw_high_u32(a, b);
}

// CHECK-LABEL: @test_vsubl_s8(
// CHECK:   [[VMOVL_I_I:%.*]] = sext <8 x i8> %a to <8 x i16>
// CHECK:   [[VMOVL_I4_I:%.*]] = sext <8 x i8> %b to <8 x i16>
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vsubl_s8(int8x8_t a, int8x8_t b) {
  return vsubl_s8(a, b);
}

// CHECK-LABEL: @test_vsubl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = sext <4 x i16> %a to <4 x i32>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMOVL_I4_I:%.*]] = sext <4 x i16> %b to <4 x i32>
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vsubl_s16(int16x4_t a, int16x4_t b) {
  return vsubl_s16(a, b);
}

// CHECK-LABEL: @test_vsubl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = sext <2 x i32> %a to <2 x i64>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMOVL_I4_I:%.*]] = sext <2 x i32> %b to <2 x i64>
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <2 x i64> [[SUB_I]]
int64x2_t test_vsubl_s32(int32x2_t a, int32x2_t b) {
  return vsubl_s32(a, b);
}

// CHECK-LABEL: @test_vsubl_u8(
// CHECK:   [[VMOVL_I_I:%.*]] = zext <8 x i8> %a to <8 x i16>
// CHECK:   [[VMOVL_I4_I:%.*]] = zext <8 x i8> %b to <8 x i16>
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
uint16x8_t test_vsubl_u8(uint8x8_t a, uint8x8_t b) {
  return vsubl_u8(a, b);
}

// CHECK-LABEL: @test_vsubl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <4 x i16> %a to <4 x i32>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMOVL_I4_I:%.*]] = zext <4 x i16> %b to <4 x i32>
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vsubl_u16(uint16x4_t a, uint16x4_t b) {
  return vsubl_u16(a, b);
}

// CHECK-LABEL: @test_vsubl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <2 x i32> %a to <2 x i64>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMOVL_I4_I:%.*]] = zext <2 x i32> %b to <2 x i64>
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// CHECK:   ret <2 x i64> [[SUB_I]]
uint64x2_t test_vsubl_u32(uint32x2_t a, uint32x2_t b) {
  return vsubl_u32(a, b);
}

// CHECK-LABEL: @test_vsubl_high_s8(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP1:%.*]] = sext <8 x i8> [[SHUFFLE_I_I10_I]] to <8 x i16>
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> [[TMP0]], [[TMP1]]
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vsubl_high_s8(int8x16_t a, int8x16_t b) {
  return vsubl_high_s8(a, b);
}

// CHECK-LABEL: @test_vsubl_high_s16(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I10_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = sext <4 x i16> [[SHUFFLE_I_I10_I]] to <4 x i32>
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> [[TMP1]], [[TMP3]]
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vsubl_high_s16(int16x8_t a, int16x8_t b) {
  return vsubl_high_s16(a, b);
}

// CHECK-LABEL: @test_vsubl_high_s32(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I10_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = sext <2 x i32> [[SHUFFLE_I_I10_I]] to <2 x i64>
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> [[TMP1]], [[TMP3]]
// CHECK:   ret <2 x i64> [[SUB_I]]
int64x2_t test_vsubl_high_s32(int32x4_t a, int32x4_t b) {
  return vsubl_high_s32(a, b);
}

// CHECK-LABEL: @test_vsubl_high_u8(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP1:%.*]] = zext <8 x i8> [[SHUFFLE_I_I10_I]] to <8 x i16>
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> [[TMP0]], [[TMP1]]
// CHECK:   ret <8 x i16> [[SUB_I]]
uint16x8_t test_vsubl_high_u8(uint8x16_t a, uint8x16_t b) {
  return vsubl_high_u8(a, b);
}

// CHECK-LABEL: @test_vsubl_high_u16(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I10_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = zext <4 x i16> [[SHUFFLE_I_I10_I]] to <4 x i32>
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> [[TMP1]], [[TMP3]]
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vsubl_high_u16(uint16x8_t a, uint16x8_t b) {
  return vsubl_high_u16(a, b);
}

// CHECK-LABEL: @test_vsubl_high_u32(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// CHECK:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I10_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = zext <2 x i32> [[SHUFFLE_I_I10_I]] to <2 x i64>
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> [[TMP1]], [[TMP3]]
// CHECK:   ret <2 x i64> [[SUB_I]]
uint64x2_t test_vsubl_high_u32(uint32x4_t a, uint32x4_t b) {
  return vsubl_high_u32(a, b);
}

// CHECK-LABEL: @test_vsubw_s8(
// CHECK:   [[VMOVL_I_I:%.*]] = sext <8 x i8> %b to <8 x i16>
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[VMOVL_I_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vsubw_s8(int16x8_t a, int8x8_t b) {
  return vsubw_s8(a, b);
}

// CHECK-LABEL: @test_vsubw_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = sext <4 x i16> %b to <4 x i32>
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMOVL_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vsubw_s16(int32x4_t a, int16x4_t b) {
  return vsubw_s16(a, b);
}

// CHECK-LABEL: @test_vsubw_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = sext <2 x i32> %b to <2 x i64>
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMOVL_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I]]
int64x2_t test_vsubw_s32(int64x2_t a, int32x2_t b) {
  return vsubw_s32(a, b);
}

// CHECK-LABEL: @test_vsubw_u8(
// CHECK:   [[VMOVL_I_I:%.*]] = zext <8 x i8> %b to <8 x i16>
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[VMOVL_I_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
uint16x8_t test_vsubw_u8(uint16x8_t a, uint8x8_t b) {
  return vsubw_u8(a, b);
}

// CHECK-LABEL: @test_vsubw_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <4 x i16> %b to <4 x i32>
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMOVL_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vsubw_u16(uint32x4_t a, uint16x4_t b) {
  return vsubw_u16(a, b);
}

// CHECK-LABEL: @test_vsubw_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <2 x i32> %b to <2 x i64>
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMOVL_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I]]
uint64x2_t test_vsubw_u32(uint64x2_t a, uint32x2_t b) {
  return vsubw_u32(a, b);
}

// CHECK-LABEL: @test_vsubw_high_s8(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[TMP0]]
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vsubw_high_s8(int16x8_t a, int8x16_t b) {
  return vsubw_high_s8(a, b);
}

// CHECK-LABEL: @test_vsubw_high_s16(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[TMP1]]
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vsubw_high_s16(int32x4_t a, int16x8_t b) {
  return vsubw_high_s16(a, b);
}

// CHECK-LABEL: @test_vsubw_high_s32(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[TMP1]]
// CHECK:   ret <2 x i64> [[SUB_I]]
int64x2_t test_vsubw_high_s32(int64x2_t a, int32x4_t b) {
  return vsubw_high_s32(a, b);
}

// CHECK-LABEL: @test_vsubw_high_u8(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[TMP0]]
// CHECK:   ret <8 x i16> [[SUB_I]]
uint16x8_t test_vsubw_high_u8(uint16x8_t a, uint8x16_t b) {
  return vsubw_high_u8(a, b);
}

// CHECK-LABEL: @test_vsubw_high_u16(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[TMP1]]
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vsubw_high_u16(uint32x4_t a, uint16x8_t b) {
  return vsubw_high_u16(a, b);
}

// CHECK-LABEL: @test_vsubw_high_u32(
// CHECK:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[TMP1]]
// CHECK:   ret <2 x i64> [[SUB_I]]
uint64x2_t test_vsubw_high_u32(uint64x2_t a, uint32x4_t b) {
  return vsubw_high_u32(a, b);
}

// CHECK-LABEL: @test_vaddhn_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VADDHN_I:%.*]] = add <8 x i16> %a, %b
// CHECK:   [[VADDHN1_I:%.*]] = lshr <8 x i16> [[VADDHN_I]], splat (i16 8)
// CHECK:   [[VADDHN2_I:%.*]] = trunc <8 x i16> [[VADDHN1_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VADDHN2_I]]
int8x8_t test_vaddhn_s16(int16x8_t a, int16x8_t b) {
  return vaddhn_s16(a, b);
}

// CHECK-LABEL: @test_vaddhn_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VADDHN_I:%.*]] = add <4 x i32> %a, %b
// CHECK:   [[VADDHN1_I:%.*]] = lshr <4 x i32> [[VADDHN_I]], splat (i32 16)
// CHECK:   [[VADDHN2_I:%.*]] = trunc <4 x i32> [[VADDHN1_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VADDHN2_I]]
int16x4_t test_vaddhn_s32(int32x4_t a, int32x4_t b) {
  return vaddhn_s32(a, b);
}

// CHECK-LABEL: @test_vaddhn_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VADDHN_I:%.*]] = add <2 x i64> %a, %b
// CHECK:   [[VADDHN1_I:%.*]] = lshr <2 x i64> [[VADDHN_I]], splat (i64 32)
// CHECK:   [[VADDHN2_I:%.*]] = trunc <2 x i64> [[VADDHN1_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VADDHN2_I]]
int32x2_t test_vaddhn_s64(int64x2_t a, int64x2_t b) {
  return vaddhn_s64(a, b);
}

// CHECK-LABEL: @test_vaddhn_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VADDHN_I:%.*]] = add <8 x i16> %a, %b
// CHECK:   [[VADDHN1_I:%.*]] = lshr <8 x i16> [[VADDHN_I]], splat (i16 8)
// CHECK:   [[VADDHN2_I:%.*]] = trunc <8 x i16> [[VADDHN1_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VADDHN2_I]]
uint8x8_t test_vaddhn_u16(uint16x8_t a, uint16x8_t b) {
  return vaddhn_u16(a, b);
}

// CHECK-LABEL: @test_vaddhn_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VADDHN_I:%.*]] = add <4 x i32> %a, %b
// CHECK:   [[VADDHN1_I:%.*]] = lshr <4 x i32> [[VADDHN_I]], splat (i32 16)
// CHECK:   [[VADDHN2_I:%.*]] = trunc <4 x i32> [[VADDHN1_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VADDHN2_I]]
uint16x4_t test_vaddhn_u32(uint32x4_t a, uint32x4_t b) {
  return vaddhn_u32(a, b);
}

// CHECK-LABEL: @test_vaddhn_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VADDHN_I:%.*]] = add <2 x i64> %a, %b
// CHECK:   [[VADDHN1_I:%.*]] = lshr <2 x i64> [[VADDHN_I]], splat (i64 32)
// CHECK:   [[VADDHN2_I:%.*]] = trunc <2 x i64> [[VADDHN1_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VADDHN2_I]]
uint32x2_t test_vaddhn_u64(uint64x2_t a, uint64x2_t b) {
  return vaddhn_u64(a, b);
}

// CHECK-LABEL: @test_vaddhn_high_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VADDHN_I_I:%.*]] = add <8 x i16> %a, %b
// CHECK:   [[VADDHN1_I_I:%.*]] = lshr <8 x i16> [[VADDHN_I_I]], splat (i16 8)
// CHECK:   [[VADDHN2_I_I:%.*]] = trunc <8 x i16> [[VADDHN1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VADDHN2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
int8x16_t test_vaddhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  return vaddhn_high_s16(r, a, b);
}

// CHECK-LABEL: @test_vaddhn_high_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VADDHN_I_I:%.*]] = add <4 x i32> %a, %b
// CHECK:   [[VADDHN1_I_I:%.*]] = lshr <4 x i32> [[VADDHN_I_I]], splat (i32 16)
// CHECK:   [[VADDHN2_I_I:%.*]] = trunc <4 x i32> [[VADDHN1_I_I]] to <4 x i16>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VADDHN2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
int16x8_t test_vaddhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  return vaddhn_high_s32(r, a, b);
}

// CHECK-LABEL: @test_vaddhn_high_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VADDHN_I_I:%.*]] = add <2 x i64> %a, %b
// CHECK:   [[VADDHN1_I_I:%.*]] = lshr <2 x i64> [[VADDHN_I_I]], splat (i64 32)
// CHECK:   [[VADDHN2_I_I:%.*]] = trunc <2 x i64> [[VADDHN1_I_I]] to <2 x i32>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VADDHN2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
int32x4_t test_vaddhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  return vaddhn_high_s64(r, a, b);
}

// CHECK-LABEL: @test_vaddhn_high_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VADDHN_I_I:%.*]] = add <8 x i16> %a, %b
// CHECK:   [[VADDHN1_I_I:%.*]] = lshr <8 x i16> [[VADDHN_I_I]], splat (i16 8)
// CHECK:   [[VADDHN2_I_I:%.*]] = trunc <8 x i16> [[VADDHN1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VADDHN2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
uint8x16_t test_vaddhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  return vaddhn_high_u16(r, a, b);
}

// CHECK-LABEL: @test_vaddhn_high_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VADDHN_I_I:%.*]] = add <4 x i32> %a, %b
// CHECK:   [[VADDHN1_I_I:%.*]] = lshr <4 x i32> [[VADDHN_I_I]], splat (i32 16)
// CHECK:   [[VADDHN2_I_I:%.*]] = trunc <4 x i32> [[VADDHN1_I_I]] to <4 x i16>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VADDHN2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
uint16x8_t test_vaddhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  return vaddhn_high_u32(r, a, b);
}

// CHECK-LABEL: @test_vaddhn_high_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VADDHN_I_I:%.*]] = add <2 x i64> %a, %b
// CHECK:   [[VADDHN1_I_I:%.*]] = lshr <2 x i64> [[VADDHN_I_I]], splat (i64 32)
// CHECK:   [[VADDHN2_I_I:%.*]] = trunc <2 x i64> [[VADDHN1_I_I]] to <2 x i32>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VADDHN2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
uint32x4_t test_vaddhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  return vaddhn_high_u64(r, a, b);
}

// CHECK-LABEL: @test_vraddhn_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i8> [[VRADDHN_V2_I]]
int8x8_t test_vraddhn_s16(int16x8_t a, int16x8_t b) {
  return vraddhn_s16(a, b);
}

// CHECK-LABEL: @test_vraddhn_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRADDHN_V3_I:%.*]] = bitcast <4 x i16> [[VRADDHN_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VRADDHN_V2_I]]
int16x4_t test_vraddhn_s32(int32x4_t a, int32x4_t b) {
  return vraddhn_s32(a, b);
}

// CHECK-LABEL: @test_vraddhn_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRADDHN_V3_I:%.*]] = bitcast <2 x i32> [[VRADDHN_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VRADDHN_V2_I]]
int32x2_t test_vraddhn_s64(int64x2_t a, int64x2_t b) {
  return vraddhn_s64(a, b);
}

// CHECK-LABEL: @test_vraddhn_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i8> [[VRADDHN_V2_I]]
uint8x8_t test_vraddhn_u16(uint16x8_t a, uint16x8_t b) {
  return vraddhn_u16(a, b);
}

// CHECK-LABEL: @test_vraddhn_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRADDHN_V3_I:%.*]] = bitcast <4 x i16> [[VRADDHN_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VRADDHN_V2_I]]
uint16x4_t test_vraddhn_u32(uint32x4_t a, uint32x4_t b) {
  return vraddhn_u32(a, b);
}

// CHECK-LABEL: @test_vraddhn_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRADDHN_V3_I:%.*]] = bitcast <2 x i32> [[VRADDHN_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VRADDHN_V2_I]]
uint32x2_t test_vraddhn_u64(uint64x2_t a, uint64x2_t b) {
  return vraddhn_u64(a, b);
}

// CHECK-LABEL: @test_vraddhn_high_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VRADDHN_V2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
int8x16_t test_vraddhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  return vraddhn_high_s16(r, a, b);
}

// CHECK-LABEL: @test_vraddhn_high_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRADDHN_V3_I_I:%.*]] = bitcast <4 x i16> [[VRADDHN_V2_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VRADDHN_V2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
int16x8_t test_vraddhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  return vraddhn_high_s32(r, a, b);
}

// CHECK-LABEL: @test_vraddhn_high_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRADDHN_V3_I_I:%.*]] = bitcast <2 x i32> [[VRADDHN_V2_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VRADDHN_V2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
int32x4_t test_vraddhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  return vraddhn_high_s64(r, a, b);
}

// CHECK-LABEL: @test_vraddhn_high_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VRADDHN_V2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
uint8x16_t test_vraddhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  return vraddhn_high_u16(r, a, b);
}

// CHECK-LABEL: @test_vraddhn_high_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRADDHN_V3_I_I:%.*]] = bitcast <4 x i16> [[VRADDHN_V2_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VRADDHN_V2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
uint16x8_t test_vraddhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  return vraddhn_high_u32(r, a, b);
}

// CHECK-LABEL: @test_vraddhn_high_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRADDHN_V2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRADDHN_V3_I_I:%.*]] = bitcast <2 x i32> [[VRADDHN_V2_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VRADDHN_V2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
uint32x4_t test_vraddhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  return vraddhn_high_u64(r, a, b);
}

// CHECK-LABEL: @test_vsubhn_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSUBHN_I:%.*]] = sub <8 x i16> %a, %b
// CHECK:   [[VSUBHN1_I:%.*]] = lshr <8 x i16> [[VSUBHN_I]], splat (i16 8)
// CHECK:   [[VSUBHN2_I:%.*]] = trunc <8 x i16> [[VSUBHN1_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VSUBHN2_I]]
int8x8_t test_vsubhn_s16(int16x8_t a, int16x8_t b) {
  return vsubhn_s16(a, b);
}

// CHECK-LABEL: @test_vsubhn_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSUBHN_I:%.*]] = sub <4 x i32> %a, %b
// CHECK:   [[VSUBHN1_I:%.*]] = lshr <4 x i32> [[VSUBHN_I]], splat (i32 16)
// CHECK:   [[VSUBHN2_I:%.*]] = trunc <4 x i32> [[VSUBHN1_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VSUBHN2_I]]
int16x4_t test_vsubhn_s32(int32x4_t a, int32x4_t b) {
  return vsubhn_s32(a, b);
}

// CHECK-LABEL: @test_vsubhn_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSUBHN_I:%.*]] = sub <2 x i64> %a, %b
// CHECK:   [[VSUBHN1_I:%.*]] = lshr <2 x i64> [[VSUBHN_I]], splat (i64 32)
// CHECK:   [[VSUBHN2_I:%.*]] = trunc <2 x i64> [[VSUBHN1_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VSUBHN2_I]]
int32x2_t test_vsubhn_s64(int64x2_t a, int64x2_t b) {
  return vsubhn_s64(a, b);
}

// CHECK-LABEL: @test_vsubhn_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSUBHN_I:%.*]] = sub <8 x i16> %a, %b
// CHECK:   [[VSUBHN1_I:%.*]] = lshr <8 x i16> [[VSUBHN_I]], splat (i16 8)
// CHECK:   [[VSUBHN2_I:%.*]] = trunc <8 x i16> [[VSUBHN1_I]] to <8 x i8>
// CHECK:   ret <8 x i8> [[VSUBHN2_I]]
uint8x8_t test_vsubhn_u16(uint16x8_t a, uint16x8_t b) {
  return vsubhn_u16(a, b);
}

// CHECK-LABEL: @test_vsubhn_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSUBHN_I:%.*]] = sub <4 x i32> %a, %b
// CHECK:   [[VSUBHN1_I:%.*]] = lshr <4 x i32> [[VSUBHN_I]], splat (i32 16)
// CHECK:   [[VSUBHN2_I:%.*]] = trunc <4 x i32> [[VSUBHN1_I]] to <4 x i16>
// CHECK:   ret <4 x i16> [[VSUBHN2_I]]
uint16x4_t test_vsubhn_u32(uint32x4_t a, uint32x4_t b) {
  return vsubhn_u32(a, b);
}

// CHECK-LABEL: @test_vsubhn_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSUBHN_I:%.*]] = sub <2 x i64> %a, %b
// CHECK:   [[VSUBHN1_I:%.*]] = lshr <2 x i64> [[VSUBHN_I]], splat (i64 32)
// CHECK:   [[VSUBHN2_I:%.*]] = trunc <2 x i64> [[VSUBHN1_I]] to <2 x i32>
// CHECK:   ret <2 x i32> [[VSUBHN2_I]]
uint32x2_t test_vsubhn_u64(uint64x2_t a, uint64x2_t b) {
  return vsubhn_u64(a, b);
}

// CHECK-LABEL: @test_vsubhn_high_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSUBHN_I_I:%.*]] = sub <8 x i16> %a, %b
// CHECK:   [[VSUBHN1_I_I:%.*]] = lshr <8 x i16> [[VSUBHN_I_I]], splat (i16 8)
// CHECK:   [[VSUBHN2_I_I:%.*]] = trunc <8 x i16> [[VSUBHN1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VSUBHN2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
int8x16_t test_vsubhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  return vsubhn_high_s16(r, a, b);
}

// CHECK-LABEL: @test_vsubhn_high_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSUBHN_I_I:%.*]] = sub <4 x i32> %a, %b
// CHECK:   [[VSUBHN1_I_I:%.*]] = lshr <4 x i32> [[VSUBHN_I_I]], splat (i32 16)
// CHECK:   [[VSUBHN2_I_I:%.*]] = trunc <4 x i32> [[VSUBHN1_I_I]] to <4 x i16>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VSUBHN2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
int16x8_t test_vsubhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  return vsubhn_high_s32(r, a, b);
}

// CHECK-LABEL: @test_vsubhn_high_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSUBHN_I_I:%.*]] = sub <2 x i64> %a, %b
// CHECK:   [[VSUBHN1_I_I:%.*]] = lshr <2 x i64> [[VSUBHN_I_I]], splat (i64 32)
// CHECK:   [[VSUBHN2_I_I:%.*]] = trunc <2 x i64> [[VSUBHN1_I_I]] to <2 x i32>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VSUBHN2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
int32x4_t test_vsubhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  return vsubhn_high_s64(r, a, b);
}

// CHECK-LABEL: @test_vsubhn_high_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSUBHN_I_I:%.*]] = sub <8 x i16> %a, %b
// CHECK:   [[VSUBHN1_I_I:%.*]] = lshr <8 x i16> [[VSUBHN_I_I]], splat (i16 8)
// CHECK:   [[VSUBHN2_I_I:%.*]] = trunc <8 x i16> [[VSUBHN1_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VSUBHN2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
uint8x16_t test_vsubhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  return vsubhn_high_u16(r, a, b);
}

// CHECK-LABEL: @test_vsubhn_high_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSUBHN_I_I:%.*]] = sub <4 x i32> %a, %b
// CHECK:   [[VSUBHN1_I_I:%.*]] = lshr <4 x i32> [[VSUBHN_I_I]], splat (i32 16)
// CHECK:   [[VSUBHN2_I_I:%.*]] = trunc <4 x i32> [[VSUBHN1_I_I]] to <4 x i16>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VSUBHN2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
uint16x8_t test_vsubhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  return vsubhn_high_u32(r, a, b);
}

// CHECK-LABEL: @test_vsubhn_high_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSUBHN_I_I:%.*]] = sub <2 x i64> %a, %b
// CHECK:   [[VSUBHN1_I_I:%.*]] = lshr <2 x i64> [[VSUBHN_I_I]], splat (i64 32)
// CHECK:   [[VSUBHN2_I_I:%.*]] = trunc <2 x i64> [[VSUBHN1_I_I]] to <2 x i32>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VSUBHN2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
uint32x4_t test_vsubhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  return vsubhn_high_u64(r, a, b);
}

// CHECK-LABEL: @test_vrsubhn_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i8> [[VRSUBHN_V2_I]]
int8x8_t test_vrsubhn_s16(int16x8_t a, int16x8_t b) {
  return vrsubhn_s16(a, b);
}

// CHECK-LABEL: @test_vrsubhn_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRSUBHN_V3_I:%.*]] = bitcast <4 x i16> [[VRSUBHN_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VRSUBHN_V2_I]]
int16x4_t test_vrsubhn_s32(int32x4_t a, int32x4_t b) {
  return vrsubhn_s32(a, b);
}

// CHECK-LABEL: @test_vrsubhn_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRSUBHN_V3_I:%.*]] = bitcast <2 x i32> [[VRSUBHN_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VRSUBHN_V2_I]]
int32x2_t test_vrsubhn_s64(int64x2_t a, int64x2_t b) {
  return vrsubhn_s64(a, b);
}

// CHECK-LABEL: @test_vrsubhn_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i8> [[VRSUBHN_V2_I]]
uint8x8_t test_vrsubhn_u16(uint16x8_t a, uint16x8_t b) {
  return vrsubhn_u16(a, b);
}

// CHECK-LABEL: @test_vrsubhn_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRSUBHN_V3_I:%.*]] = bitcast <4 x i16> [[VRSUBHN_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VRSUBHN_V2_I]]
uint16x4_t test_vrsubhn_u32(uint32x4_t a, uint32x4_t b) {
  return vrsubhn_u32(a, b);
}

// CHECK-LABEL: @test_vrsubhn_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRSUBHN_V3_I:%.*]] = bitcast <2 x i32> [[VRSUBHN_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VRSUBHN_V2_I]]
uint32x2_t test_vrsubhn_u64(uint64x2_t a, uint64x2_t b) {
  return vrsubhn_u64(a, b);
}

// CHECK-LABEL: @test_vrsubhn_high_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VRSUBHN_V2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
int8x16_t test_vrsubhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
  return vrsubhn_high_s16(r, a, b);
}

// CHECK-LABEL: @test_vrsubhn_high_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRSUBHN_V3_I_I:%.*]] = bitcast <4 x i16> [[VRSUBHN_V2_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VRSUBHN_V2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
int16x8_t test_vrsubhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
  return vrsubhn_high_s32(r, a, b);
}

// CHECK-LABEL: @test_vrsubhn_high_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRSUBHN_V3_I_I:%.*]] = bitcast <2 x i32> [[VRSUBHN_V2_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VRSUBHN_V2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
int32x4_t test_vrsubhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
  return vrsubhn_high_s64(r, a, b);
}

// CHECK-LABEL: @test_vrsubhn_high_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VRSUBHN_V2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   ret <16 x i8> [[SHUFFLE_I_I]]
uint8x16_t test_vrsubhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
  return vrsubhn_high_u16(r, a, b);
}

// CHECK-LABEL: @test_vrsubhn_high_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// CHECK:   [[VRSUBHN_V3_I_I:%.*]] = bitcast <4 x i16> [[VRSUBHN_V2_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VRSUBHN_V2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK:   ret <8 x i16> [[SHUFFLE_I_I]]
uint16x8_t test_vrsubhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
  return vrsubhn_high_u32(r, a, b);
}

// CHECK-LABEL: @test_vrsubhn_high_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VRSUBHN_V2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VRSUBHN_V3_I_I:%.*]] = bitcast <2 x i32> [[VRSUBHN_V2_I_I]] to <8 x i8>
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VRSUBHN_V2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:   ret <4 x i32> [[SHUFFLE_I_I]]
uint32x4_t test_vrsubhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
  return vrsubhn_high_u64(r, a, b);
}

// CHECK-LABEL: @test_vabdl_s8(
// CHECK:   [[VABD_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   [[VMOVL_I_I:%.*]] = zext <8 x i8> [[VABD_I_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VMOVL_I_I]]
int16x8_t test_vabdl_s8(int8x8_t a, int8x8_t b) {
  return vabdl_s8(a, b);
}

// CHECK-LABEL: @test_vabdl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VMOVL_I_I]]
int32x4_t test_vabdl_s16(int16x4_t a, int16x4_t b) {
  return vabdl_s16(a, b);
}

// CHECK-LABEL: @test_vabdl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VMOVL_I_I]]
int64x2_t test_vabdl_s32(int32x2_t a, int32x2_t b) {
  return vabdl_s32(a, b);
}

// CHECK-LABEL: @test_vabdl_u8(
// CHECK:   [[VABD_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   [[VMOVL_I_I:%.*]] = zext <8 x i8> [[VABD_I_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VMOVL_I_I]]
uint16x8_t test_vabdl_u8(uint8x8_t a, uint8x8_t b) {
  return vabdl_u8(a, b);
}

// CHECK-LABEL: @test_vabdl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VMOVL_I_I]]
uint32x4_t test_vabdl_u16(uint16x4_t a, uint16x4_t b) {
  return vabdl_u16(a, b);
}

// CHECK-LABEL: @test_vabdl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VABD2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VMOVL_I_I]]
uint64x2_t test_vabdl_u32(uint32x2_t a, uint32x2_t b) {
  return vabdl_u32(a, b);
}

// CHECK-LABEL: @test_vabal_s8(
// CHECK:   [[VABD_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %b, <8 x i8> %c)
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I]] to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vabal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  return vabal_s8(a, b, c);
}

// CHECK-LABEL: @test_vabal_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// CHECK:   [[VABD2_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %b, <4 x i16> %c)
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I]] to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vabal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vabal_s16(a, b, c);
}

// CHECK-LABEL: @test_vabal_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// CHECK:   [[VABD2_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %b, <2 x i32> %c)
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I]] to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
int64x2_t test_vabal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vabal_s32(a, b, c);
}

// CHECK-LABEL: @test_vabal_u8(
// CHECK:   [[VABD_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %b, <8 x i8> %c)
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I]] to <8 x i16>
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vabal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  return vabal_u8(a, b, c);
}

// CHECK-LABEL: @test_vabal_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// CHECK:   [[VABD2_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %b, <4 x i16> %c)
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I]] to <4 x i32>
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vabal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  return vabal_u16(a, b, c);
}

// CHECK-LABEL: @test_vabal_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// CHECK:   [[VABD2_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %b, <2 x i32> %c)
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I]] to <2 x i64>
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
uint64x2_t test_vabal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  return vabal_u32(a, b, c);
}

// CHECK-LABEL: @test_vabdl_high_s8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VABD_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VMOVL_I_I_I]]
int16x8_t test_vabdl_high_s8(int8x16_t a, int8x16_t b) {
  return vabdl_high_s8(a, b);
}

// CHECK-LABEL: @test_vabdl_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VABD2_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VMOVL_I_I_I]]
int32x4_t test_vabdl_high_s16(int16x8_t a, int16x8_t b) {
  return vabdl_high_s16(a, b);
}

// CHECK-LABEL: @test_vabdl_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VABD2_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VMOVL_I_I_I]]
int64x2_t test_vabdl_high_s32(int32x4_t a, int32x4_t b) {
  return vabdl_high_s32(a, b);
}

// CHECK-LABEL: @test_vabdl_high_u8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VABD_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I]] to <8 x i16>
// CHECK:   ret <8 x i16> [[VMOVL_I_I_I]]
uint16x8_t test_vabdl_high_u8(uint8x16_t a, uint8x16_t b) {
  return vabdl_high_u8(a, b);
}

// CHECK-LABEL: @test_vabdl_high_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VABD2_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I]] to <4 x i32>
// CHECK:   ret <4 x i32> [[VMOVL_I_I_I]]
uint32x4_t test_vabdl_high_u16(uint16x8_t a, uint16x8_t b) {
  return vabdl_high_u16(a, b);
}

// CHECK-LABEL: @test_vabdl_high_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VABD2_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VMOVL_I_I_I]]
uint64x2_t test_vabdl_high_u32(uint32x4_t a, uint32x4_t b) {
  return vabdl_high_u32(a, b);
}

// CHECK-LABEL: @test_vabal_high_s8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VABD_I_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   [[VMOVL_I_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I_I]] to <8 x i16>
// CHECK:   [[ADD_I_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I_I]]
int16x8_t test_vabal_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
  return vabal_high_s8(a, b, c);
}

// CHECK-LABEL: @test_vabal_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VABD2_I_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I_I]] to <4 x i32>
// CHECK:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I_I]]
int32x4_t test_vabal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  return vabal_high_s16(a, b, c);
}

// CHECK-LABEL: @test_vabal_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VABD2_I_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I_I]] to <2 x i64>
// CHECK:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I_I]]
int64x2_t test_vabal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  return vabal_high_s32(a, b, c);
}

// CHECK-LABEL: @test_vabal_high_u8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VABD_I_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   [[VMOVL_I_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I_I]] to <8 x i16>
// CHECK:   [[ADD_I_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I_I]]
uint16x8_t test_vabal_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
  return vabal_high_u8(a, b, c);
}

// CHECK-LABEL: @test_vabal_high_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VABD2_I_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I_I]] to <4 x i32>
// CHECK:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I_I]]
uint32x4_t test_vabal_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
  return vabal_high_u16(a, b, c);
}

// CHECK-LABEL: @test_vabal_high_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VABD2_I_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I_I]] to <8 x i8>
// CHECK:   [[VMOVL_I_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I_I]] to <2 x i64>
// CHECK:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I_I]]
uint64x2_t test_vabal_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
  return vabal_high_u32(a, b, c);
}

// CHECK-LABEL: @test_vmull_s8(
// CHECK:   [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i16> [[VMULL_I]]
int16x8_t test_vmull_s8(int8x8_t a, int8x8_t b) {
  return vmull_s8(a, b);
}

// CHECK-LABEL: @test_vmull_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_s16(int16x4_t a, int16x4_t b) {
  return vmull_s16(a, b);
}

// CHECK-LABEL: @test_vmull_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_s32(int32x2_t a, int32x2_t b) {
  return vmull_s32(a, b);
}

// CHECK-LABEL: @test_vmull_u8(
// CHECK:   [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i16> [[VMULL_I]]
uint16x8_t test_vmull_u8(uint8x8_t a, uint8x8_t b) {
  return vmull_u8(a, b);
}

// CHECK-LABEL: @test_vmull_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_u16(uint16x4_t a, uint16x4_t b) {
  return vmull_u16(a, b);
}

// CHECK-LABEL: @test_vmull_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_u32(uint32x2_t a, uint32x2_t b) {
  return vmull_u32(a, b);
}

// CHECK-LABEL: @test_vmull_high_s8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   ret <8 x i16> [[VMULL_I_I]]
int16x8_t test_vmull_high_s8(int8x16_t a, int8x16_t b) {
  return vmull_high_s8(a, b);
}

// CHECK-LABEL: @test_vmull_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   ret <4 x i32> [[VMULL2_I_I]]
int32x4_t test_vmull_high_s16(int16x8_t a, int16x8_t b) {
  return vmull_high_s16(a, b);
}

// CHECK-LABEL: @test_vmull_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   ret <2 x i64> [[VMULL2_I_I]]
int64x2_t test_vmull_high_s32(int32x4_t a, int32x4_t b) {
  return vmull_high_s32(a, b);
}

// CHECK-LABEL: @test_vmull_high_u8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   ret <8 x i16> [[VMULL_I_I]]
uint16x8_t test_vmull_high_u8(uint8x16_t a, uint8x16_t b) {
  return vmull_high_u8(a, b);
}

// CHECK-LABEL: @test_vmull_high_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   ret <4 x i32> [[VMULL2_I_I]]
uint32x4_t test_vmull_high_u16(uint16x8_t a, uint16x8_t b) {
  return vmull_high_u16(a, b);
}

// CHECK-LABEL: @test_vmull_high_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   ret <2 x i64> [[VMULL2_I_I]]
uint64x2_t test_vmull_high_u32(uint32x4_t a, uint32x4_t b) {
  return vmull_high_u32(a, b);
}

// CHECK-LABEL: @test_vmlal_s8(
// CHECK:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %b, <8 x i8> %c)
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMULL_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vmlal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  return vmlal_s8(a, b, c);
}

// CHECK-LABEL: @test_vmlal_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> %c)
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vmlal_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlal_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> %c)
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
int64x2_t test_vmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vmlal_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlal_u8(
// CHECK:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %b, <8 x i8> %c)
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMULL_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vmlal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  return vmlal_u8(a, b, c);
}

// CHECK-LABEL: @test_vmlal_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> %c)
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vmlal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmlal_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlal_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> %c)
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
uint64x2_t test_vmlal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmlal_u32(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_s8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VMULL_I_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <8 x i16> %a, [[VMULL_I_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I_I]]
int16x8_t test_vmlal_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
  return vmlal_high_s8(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I_I]]
int32x4_t test_vmlal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  return vmlal_high_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I_I]]
int64x2_t test_vmlal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  return vmlal_high_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_u8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VMULL_I_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <8 x i16> %a, [[VMULL_I_I_I]]
// CHECK:   ret <8 x i16> [[ADD_I_I]]
uint16x8_t test_vmlal_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
  return vmlal_high_u8(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I_I]]
uint32x4_t test_vmlal_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
  return vmlal_high_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I_I]]
uint64x2_t test_vmlal_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
  return vmlal_high_u32(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_s8(
// CHECK:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %b, <8 x i8> %c)
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[VMULL_I_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vmlsl_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
  return vmlsl_s8(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> %c)
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vmlsl_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> %c)
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I]]
int64x2_t test_vmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vmlsl_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_u8(
// CHECK:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %b, <8 x i8> %c)
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[VMULL_I_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
uint16x8_t test_vmlsl_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
  return vmlsl_u8(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> %c)
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vmlsl_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
  return vmlsl_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> %c)
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I]]
uint64x2_t test_vmlsl_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
  return vmlsl_u32(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_s8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VMULL_I_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <8 x i16> %a, [[VMULL_I_I_I]]
// CHECK:   ret <8 x i16> [[SUB_I_I]]
int16x8_t test_vmlsl_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
  return vmlsl_high_s8(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I_I]]
int32x4_t test_vmlsl_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  return vmlsl_high_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I_I]]
int64x2_t test_vmlsl_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  return vmlsl_high_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_u8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VMULL_I_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <8 x i16> %a, [[VMULL_I_I_I]]
// CHECK:   ret <8 x i16> [[SUB_I_I]]
uint16x8_t test_vmlsl_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
  return vmlsl_high_u8(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I_I]]
uint32x4_t test_vmlsl_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
  return vmlsl_high_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I_I]]
uint64x2_t test_vmlsl_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
  return vmlsl_high_u32(a, b, c);
}

// CHECK-LABEL: @test_vqdmull_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %a, <4 x i16> %b)
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_s16(int16x4_t a, int16x4_t b) {
  return vqdmull_s16(a, b);
}

// CHECK-LABEL: @test_vqdmull_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %a, <2 x i32> %b)
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_s32(int32x2_t a, int32x2_t b) {
  return vqdmull_s32(a, b);
}

// CHECK-LABEL: @test_vqdmlal_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> %c)
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vqdmlal_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlal_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> %c)
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vqdmlal_s32(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsl_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> %c)
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vqdmlsl_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsl_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> %c)
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vqdmlsl_s32(a, b, c);
}

// CHECK-LABEL: @test_vqdmull_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[VQDMULL_V3_I_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I_I]]
int32x4_t test_vqdmull_high_s16(int16x8_t a, int16x8_t b) {
  return vqdmull_high_s16(a, b);
}

// CHECK-LABEL: @test_vqdmull_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[VQDMULL_V3_I_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I_I]]
int64x2_t test_vqdmull_high_s32(int32x4_t a, int32x4_t b) {
  return vqdmull_high_s32(a, b);
}

// CHECK-LABEL: @test_vqdmlal_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[VQDMLAL_V3_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I_I]]
int32x4_t test_vqdmlal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  return vqdmlal_high_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlal_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[VQDMLAL_V3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I_I]]
int64x2_t test_vqdmlal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  return vqdmlal_high_s32(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsl_high_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// CHECK:   [[VQDMLSL_V3_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I_I]]
int32x4_t test_vqdmlsl_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
  return vqdmlsl_high_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsl_high_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// CHECK:   [[VQDMLSL_V3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I_I]]
int64x2_t test_vqdmlsl_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
  return vqdmlsl_high_s32(a, b, c);
}

// CHECK-LABEL: @test_vmull_p8(
// CHECK:   [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i16> [[VMULL_I]]
poly16x8_t test_vmull_p8(poly8x8_t a, poly8x8_t b) {
  return vmull_p8(a, b);
}

// CHECK-LABEL: @test_vmull_high_p8(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// CHECK:   ret <8 x i16> [[VMULL_I_I]]
poly16x8_t test_vmull_high_p8(poly8x16_t a, poly8x16_t b) {
  return vmull_high_p8(a, b);
}

// CHECK-LABEL: @test_vaddd_s64(
// CHECK:   [[VADDD_I:%.*]] = add i64 %a, %b
// CHECK:   ret i64 [[VADDD_I]]
int64_t test_vaddd_s64(int64_t a, int64_t b) {
  return vaddd_s64(a, b);
}

// CHECK-LABEL: @test_vaddd_u64(
// CHECK:   [[VADDD_I:%.*]] = add i64 %a, %b
// CHECK:   ret i64 [[VADDD_I]]
uint64_t test_vaddd_u64(uint64_t a, uint64_t b) {
  return vaddd_u64(a, b);
}

// CHECK-LABEL: @test_vsubd_s64(
// CHECK:   [[VSUBD_I:%.*]] = sub i64 %a, %b
// CHECK:   ret i64 [[VSUBD_I]]
int64_t test_vsubd_s64(int64_t a, int64_t b) {
  return vsubd_s64(a, b);
}

// CHECK-LABEL: @test_vsubd_u64(
// CHECK:   [[VSUBD_I:%.*]] = sub i64 %a, %b
// CHECK:   ret i64 [[VSUBD_I]]
uint64_t test_vsubd_u64(uint64_t a, uint64_t b) {
  return vsubd_u64(a, b);
}

// CHECK-LABEL: @test_vqaddb_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VQADDB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqadd.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQADDB_S8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
int8_t test_vqaddb_s8(int8_t a, int8_t b) {
  return vqaddb_s8(a, b);
}

// CHECK-LABEL: @test_vqaddh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQADDH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQADDH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
int16_t test_vqaddh_s16(int16_t a, int16_t b) {
  return vqaddh_s16(a, b);
}

// CHECK-LABEL: @test_vqadds_s32(
// CHECK:   [[VQADDS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqadd.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQADDS_S32_I]]
int32_t test_vqadds_s32(int32_t a, int32_t b) {
  return vqadds_s32(a, b);
}

// CHECK-LABEL: @test_vqaddd_s64(
// CHECK:   [[VQADDD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqadd.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VQADDD_S64_I]]
int64_t test_vqaddd_s64(int64_t a, int64_t b) {
  return vqaddd_s64(a, b);
}

// CHECK-LABEL: @test_vqaddb_u8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VQADDB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqadd.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQADDB_U8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
uint8_t test_vqaddb_u8(uint8_t a, uint8_t b) {
  return vqaddb_u8(a, b);
}

// CHECK-LABEL: @test_vqaddh_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQADDH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqadd.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQADDH_U16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vqaddh_u16(uint16_t a, uint16_t b) {
  return vqaddh_u16(a, b);
}

// CHECK-LABEL: @test_vqadds_u32(
// CHECK:   [[VQADDS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uqadd.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQADDS_U32_I]]
uint32_t test_vqadds_u32(uint32_t a, uint32_t b) {
  return vqadds_u32(a, b);
}

// CHECK-LABEL: @test_vqaddd_u64(
// CHECK:   [[VQADDD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uqadd.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VQADDD_U64_I]]
uint64_t test_vqaddd_u64(uint64_t a, uint64_t b) {
  return vqaddd_u64(a, b);
}

// CHECK-LABEL: @test_vqsubb_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VQSUBB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqsub.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQSUBB_S8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
int8_t test_vqsubb_s8(int8_t a, int8_t b) {
  return vqsubb_s8(a, b);
}

// CHECK-LABEL: @test_vqsubh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQSUBH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQSUBH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
int16_t test_vqsubh_s16(int16_t a, int16_t b) {
  return vqsubh_s16(a, b);
}

// CHECK-LABEL: @test_vqsubs_s32(
// CHECK:   [[VQSUBS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqsub.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQSUBS_S32_I]]
int32_t test_vqsubs_s32(int32_t a, int32_t b) {
  return vqsubs_s32(a, b);
}

// CHECK-LABEL: @test_vqsubd_s64(
// CHECK:   [[VQSUBD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqsub.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VQSUBD_S64_I]]
int64_t test_vqsubd_s64(int64_t a, int64_t b) {
  return vqsubd_s64(a, b);
}

// CHECK-LABEL: @test_vqsubb_u8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VQSUBB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqsub.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQSUBB_U8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
uint8_t test_vqsubb_u8(uint8_t a, uint8_t b) {
  return vqsubb_u8(a, b);
}

// CHECK-LABEL: @test_vqsubh_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQSUBH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqsub.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQSUBH_U16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vqsubh_u16(uint16_t a, uint16_t b) {
  return vqsubh_u16(a, b);
}

// CHECK-LABEL: @test_vqsubs_u32(
// CHECK:   [[VQSUBS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uqsub.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQSUBS_U32_I]]
uint32_t test_vqsubs_u32(uint32_t a, uint32_t b) {
  return vqsubs_u32(a, b);
}

// CHECK-LABEL: @test_vqsubd_u64(
// CHECK:   [[VQSUBD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uqsub.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VQSUBD_U64_I]]
uint64_t test_vqsubd_u64(uint64_t a, uint64_t b) {
  return vqsubd_u64(a, b);
}

// CHECK-LABEL: @test_vshld_s64(
// CHECK:   [[VSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sshl.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VSHLD_S64_I]]
int64_t test_vshld_s64(int64_t a, int64_t b) {
  return vshld_s64(a, b);
}

// CHECK-LABEL: @test_vshld_u64(
// CHECK:   [[VSHLD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.ushl.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VSHLD_U64_I]]
uint64_t test_vshld_u64(uint64_t a, int64_t b) {
  return vshld_u64(a, b);
}

// CHECK-LABEL: @test_vqshlb_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VQSHLB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQSHLB_S8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
int8_t test_vqshlb_s8(int8_t a, int8_t b) {
  return vqshlb_s8(a, b);
}

// CHECK-LABEL: @test_vqshlh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQSHLH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQSHLH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
int16_t test_vqshlh_s16(int16_t a, int16_t b) {
  return vqshlh_s16(a, b);
}

// CHECK-LABEL: @test_vqshls_s32(
// CHECK:   [[VQSHLS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqshl.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQSHLS_S32_I]]
int32_t test_vqshls_s32(int32_t a, int32_t b) {
  return vqshls_s32(a, b);
}

// CHECK-LABEL: @test_vqshld_s64(
// CHECK:   [[VQSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqshl.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VQSHLD_S64_I]]
int64_t test_vqshld_s64(int64_t a, int64_t b) {
  return vqshld_s64(a, b);
}

// CHECK-LABEL: @test_vqshlb_u8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VQSHLB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQSHLB_U8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
uint8_t test_vqshlb_u8(uint8_t a, int8_t b) {
  return vqshlb_u8(a, b);
}

// CHECK-LABEL: @test_vqshlh_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQSHLH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQSHLH_U16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vqshlh_u16(uint16_t a, int16_t b) {
  return vqshlh_u16(a, b);
}

// CHECK-LABEL: @test_vqshls_u32(
// CHECK:   [[VQSHLS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uqshl.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQSHLS_U32_I]]
uint32_t test_vqshls_u32(uint32_t a, int32_t b) {
  return vqshls_u32(a, b);
}

// CHECK-LABEL: @test_vqshld_u64(
// CHECK:   [[VQSHLD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uqshl.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VQSHLD_U64_I]]
uint64_t test_vqshld_u64(uint64_t a, int64_t b) {
  return vqshld_u64(a, b);
}

// CHECK-LABEL: @test_vrshld_s64(
// CHECK:   [[VRSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.srshl.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VRSHLD_S64_I]]
int64_t test_vrshld_s64(int64_t a, int64_t b) {
  return vrshld_s64(a, b);
}

// CHECK-LABEL: @test_vrshld_u64(
// CHECK:   [[VRSHLD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.urshl.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VRSHLD_U64_I]]
uint64_t test_vrshld_u64(uint64_t a, int64_t b) {
  return vrshld_u64(a, b);
}

// CHECK-LABEL: @test_vqrshlb_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VQRSHLB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQRSHLB_S8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
int8_t test_vqrshlb_s8(int8_t a, int8_t b) {
  return vqrshlb_s8(a, b);
}

// CHECK-LABEL: @test_vqrshlh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQRSHLH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQRSHLH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
int16_t test_vqrshlh_s16(int16_t a, int16_t b) {
  return vqrshlh_s16(a, b);
}

// CHECK-LABEL: @test_vqrshls_s32(
// CHECK:   [[VQRSHLS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqrshl.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQRSHLS_S32_I]]
int32_t test_vqrshls_s32(int32_t a, int32_t b) {
  return vqrshls_s32(a, b);
}

// CHECK-LABEL: @test_vqrshld_s64(
// CHECK:   [[VQRSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqrshl.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VQRSHLD_S64_I]]
int64_t test_vqrshld_s64(int64_t a, int64_t b) {
  return vqrshld_s64(a, b);
}

// CHECK-LABEL: @test_vqrshlb_u8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VQRSHLB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQRSHLB_U8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
uint8_t test_vqrshlb_u8(uint8_t a, int8_t b) {
  return vqrshlb_u8(a, b);
}

// CHECK-LABEL: @test_vqrshlh_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQRSHLH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQRSHLH_U16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vqrshlh_u16(uint16_t a, int16_t b) {
  return vqrshlh_u16(a, b);
}

// CHECK-LABEL: @test_vqrshls_u32(
// CHECK:   [[VQRSHLS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uqrshl.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQRSHLS_U32_I]]
uint32_t test_vqrshls_u32(uint32_t a, int32_t b) {
  return vqrshls_u32(a, b);
}

// CHECK-LABEL: @test_vqrshld_u64(
// CHECK:   [[VQRSHLD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uqrshl.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VQRSHLD_U64_I]]
uint64_t test_vqrshld_u64(uint64_t a, int64_t b) {
  return vqrshld_u64(a, b);
}

// CHECK-LABEL: @test_vpaddd_s64(
// CHECK:   [[VPADDD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64> %a)
// CHECK:   ret i64 [[VPADDD_S64_I]]
int64_t test_vpaddd_s64(int64x2_t a) {
  return vpaddd_s64(a);
}

// CHECK-LABEL: @test_vpadds_f32(
// CHECK:   [[LANE0_I:%.*]] = extractelement <2 x float> %a, i64 0
// CHECK:   [[LANE1_I:%.*]] = extractelement <2 x float> %a, i64 1
// CHECK:   [[VPADDD_I:%.*]] = fadd float [[LANE0_I]], [[LANE1_I]]
// CHECK:   ret float [[VPADDD_I]]
float32_t test_vpadds_f32(float32x2_t a) {
  return vpadds_f32(a);
}

// CHECK-LABEL: @test_vpaddd_f64(
// CHECK:   [[LANE0_I:%.*]] = extractelement <2 x double> %a, i64 0
// CHECK:   [[LANE1_I:%.*]] = extractelement <2 x double> %a, i64 1
// CHECK:   [[VPADDD_I:%.*]] = fadd double [[LANE0_I]], [[LANE1_I]]
// CHECK:   ret double [[VPADDD_I]]
float64_t test_vpaddd_f64(float64x2_t a) {
  return vpaddd_f64(a);
}

// CHECK-LABEL: @test_vpmaxnms_f32(
// CHECK:   [[VPMAXNMS_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxnmv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VPMAXNMS_F32_I]]
float32_t test_vpmaxnms_f32(float32x2_t a) {
  return vpmaxnms_f32(a);
}

// CHECK-LABEL: @test_vpmaxnmqd_f64(
// CHECK:   [[VPMAXNMQD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmaxnmv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VPMAXNMQD_F64_I]]
float64_t test_vpmaxnmqd_f64(float64x2_t a) {
  return vpmaxnmqd_f64(a);
}

// CHECK-LABEL: @test_vpmaxs_f32(
// CHECK:   [[VPMAXS_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VPMAXS_F32_I]]
float32_t test_vpmaxs_f32(float32x2_t a) {
  return vpmaxs_f32(a);
}

// CHECK-LABEL: @test_vpmaxqd_f64(
// CHECK:   [[VPMAXQD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmaxv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VPMAXQD_F64_I]]
float64_t test_vpmaxqd_f64(float64x2_t a) {
  return vpmaxqd_f64(a);
}

// CHECK-LABEL: @test_vpminnms_f32(
// CHECK:   [[VPMINNMS_F32_I:%.*]] = call float @llvm.aarch64.neon.fminnmv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VPMINNMS_F32_I]]
float32_t test_vpminnms_f32(float32x2_t a) {
  return vpminnms_f32(a);
}

// CHECK-LABEL: @test_vpminnmqd_f64(
// CHECK:   [[VPMINNMQD_F64_I:%.*]] = call double @llvm.aarch64.neon.fminnmv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VPMINNMQD_F64_I]]
float64_t test_vpminnmqd_f64(float64x2_t a) {
  return vpminnmqd_f64(a);
}

// CHECK-LABEL: @test_vpmins_f32(
// CHECK:   [[VPMINS_F32_I:%.*]] = call float @llvm.aarch64.neon.fminv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VPMINS_F32_I]]
float32_t test_vpmins_f32(float32x2_t a) {
  return vpmins_f32(a);
}

// CHECK-LABEL: @test_vpminqd_f64(
// CHECK:   [[VPMINQD_F64_I:%.*]] = call double @llvm.aarch64.neon.fminv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VPMINQD_F64_I]]
float64_t test_vpminqd_f64(float64x2_t a) {
  return vpminqd_f64(a);
}

// CHECK-LABEL: @test_vqdmulhh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQDMULHH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQDMULHH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
int16_t test_vqdmulhh_s16(int16_t a, int16_t b) {
  return vqdmulhh_s16(a, b);
}

// CHECK-LABEL: @test_vqdmulhs_s32(
// CHECK:   [[VQDMULHS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqdmulh.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQDMULHS_S32_I]]
int32_t test_vqdmulhs_s32(int32_t a, int32_t b) {
  return vqdmulhs_s32(a, b);
}

// CHECK-LABEL: @test_vqrdmulhh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQRDMULHH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQRDMULHH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
int16_t test_vqrdmulhh_s16(int16_t a, int16_t b) {
  return vqrdmulhh_s16(a, b);
}

// CHECK-LABEL: @test_vqrdmulhs_s32(
// CHECK:   [[VQRDMULHS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VQRDMULHS_S32_I]]
int32_t test_vqrdmulhs_s32(int32_t a, int32_t b) {
  return vqrdmulhs_s32(a, b);
}

// CHECK-LABEL: @test_vmulxs_f32(
// CHECK:   [[VMULXS_F32_I:%.*]] = call float @llvm.aarch64.neon.fmulx.f32(float %a, float %b)
// CHECK:   ret float [[VMULXS_F32_I]]
float32_t test_vmulxs_f32(float32_t a, float32_t b) {
  return vmulxs_f32(a, b);
}

// CHECK-LABEL: @test_vmulxd_f64(
// CHECK:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double %a, double %b)
// CHECK:   ret double [[VMULXD_F64_I]]
float64_t test_vmulxd_f64(float64_t a, float64_t b) {
  return vmulxd_f64(a, b);
}

// CHECK-LABEL: @test_vmulx_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmulx.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x double> [[VMULX2_I]]
float64x1_t test_vmulx_f64(float64x1_t a, float64x1_t b) {
  return vmulx_f64(a, b);
}

// CHECK-LABEL: @test_vrecpss_f32(
// CHECK:   [[VRECPS_I:%.*]] = call float @llvm.aarch64.neon.frecps.f32(float %a, float %b)
// CHECK:   ret float [[VRECPS_I]]
float32_t test_vrecpss_f32(float32_t a, float32_t b) {
  return vrecpss_f32(a, b);
}

// CHECK-LABEL: @test_vrecpsd_f64(
// CHECK:   [[VRECPS_I:%.*]] = call double @llvm.aarch64.neon.frecps.f64(double %a, double %b)
// CHECK:   ret double [[VRECPS_I]]
float64_t test_vrecpsd_f64(float64_t a, float64_t b) {
  return vrecpsd_f64(a, b);
}

// CHECK-LABEL: @test_vrsqrtss_f32(
// CHECK:   [[VRSQRTSS_F32_I:%.*]] = call float @llvm.aarch64.neon.frsqrts.f32(float %a, float %b)
// CHECK:   ret float [[VRSQRTSS_F32_I]]
float32_t test_vrsqrtss_f32(float32_t a, float32_t b) {
  return vrsqrtss_f32(a, b);
}

// CHECK-LABEL: @test_vrsqrtsd_f64(
// CHECK:   [[VRSQRTSD_F64_I:%.*]] = call double @llvm.aarch64.neon.frsqrts.f64(double %a, double %b)
// CHECK:   ret double [[VRSQRTSD_F64_I]]
float64_t test_vrsqrtsd_f64(float64_t a, float64_t b) {
  return vrsqrtsd_f64(a, b);
}

// CHECK-LABEL: @test_vcvts_f32_s32(
// CHECK:   [[TMP0:%.*]] = sitofp i32 %a to float
// CHECK:   ret float [[TMP0]]
float32_t test_vcvts_f32_s32(int32_t a) {
  return vcvts_f32_s32(a);
}

// CHECK-LABEL: @test_vcvtd_f64_s64(
// CHECK:   [[TMP0:%.*]] = sitofp i64 %a to double
// CHECK:   ret double [[TMP0]]
float64_t test_vcvtd_f64_s64(int64_t a) {
  return vcvtd_f64_s64(a);
}

// CHECK-LABEL: @test_vcvts_f32_u32(
// CHECK:   [[TMP0:%.*]] = uitofp i32 %a to float
// CHECK:   ret float [[TMP0]]
float32_t test_vcvts_f32_u32(uint32_t a) {
  return vcvts_f32_u32(a);
}

// CHECK-LABEL: @test_vcvtd_f64_u64(
// CHECK:   [[TMP0:%.*]] = uitofp i64 %a to double
// CHECK:   ret double [[TMP0]]
float64_t test_vcvtd_f64_u64(uint64_t a) {
  return vcvtd_f64_u64(a);
}

// CHECK-LABEL: @test_vrecpes_f32(
// CHECK:   [[VRECPES_F32_I:%.*]] = call float @llvm.aarch64.neon.frecpe.f32(float %a)
// CHECK:   ret float [[VRECPES_F32_I]]
float32_t test_vrecpes_f32(float32_t a) {
  return vrecpes_f32(a);
}

// CHECK-LABEL: @test_vrecped_f64(
// CHECK:   [[VRECPED_F64_I:%.*]] = call double @llvm.aarch64.neon.frecpe.f64(double %a)
// CHECK:   ret double [[VRECPED_F64_I]]
float64_t test_vrecped_f64(float64_t a) {
  return vrecped_f64(a);
}

// CHECK-LABEL: @test_vrecpxs_f32(
// CHECK:   [[VRECPXS_F32_I:%.*]] = call float @llvm.aarch64.neon.frecpx.f32(float %a)
// CHECK:   ret float [[VRECPXS_F32_I]]
float32_t test_vrecpxs_f32(float32_t a) {
  return vrecpxs_f32(a);
}

// CHECK-LABEL: @test_vrecpxd_f64(
// CHECK:   [[VRECPXD_F64_I:%.*]] = call double @llvm.aarch64.neon.frecpx.f64(double %a)
// CHECK:   ret double [[VRECPXD_F64_I]]
float64_t test_vrecpxd_f64(float64_t a) {
  return vrecpxd_f64(a);
}

// CHECK-LABEL: @test_vrsqrte_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VRSQRTE_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.ursqrte.v2i32(<2 x i32> %a)
// CHECK:   ret <2 x i32> [[VRSQRTE_V1_I]]
uint32x2_t test_vrsqrte_u32(uint32x2_t a) {
  return vrsqrte_u32(a);
}

// CHECK-LABEL: @test_vrsqrteq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VRSQRTEQ_V1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.ursqrte.v4i32(<4 x i32> %a)
// CHECK:   ret <4 x i32> [[VRSQRTEQ_V1_I]]
uint32x4_t test_vrsqrteq_u32(uint32x4_t a) {
  return vrsqrteq_u32(a);
}

// CHECK-LABEL: @test_vrsqrtes_f32(
// CHECK:   [[VRSQRTES_F32_I:%.*]] = call float @llvm.aarch64.neon.frsqrte.f32(float %a)
// CHECK:   ret float [[VRSQRTES_F32_I]]
float32_t test_vrsqrtes_f32(float32_t a) {
  return vrsqrtes_f32(a);
}

// CHECK-LABEL: @test_vrsqrted_f64(
// CHECK:   [[VRSQRTED_F64_I:%.*]] = call double @llvm.aarch64.neon.frsqrte.f64(double %a)
// CHECK:   ret double [[VRSQRTED_F64_I]]
float64_t test_vrsqrted_f64(float64_t a) {
  return vrsqrted_f64(a);
}

// CHECK-LABEL: @test_vld1q_u8(
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, ptr %a, align 1
// CHECK:   ret <16 x i8> [[TMP1]]
uint8x16_t test_vld1q_u8(uint8_t const *a) {
  return vld1q_u8(a);
}

// CHECK-LABEL: @test_vld1q_u16(
// CHECK:   [[TMP2:%.*]] = load <8 x i16>, ptr %a, align 2
// CHECK:   ret <8 x i16> [[TMP2]]
uint16x8_t test_vld1q_u16(uint16_t const *a) {
  return vld1q_u16(a);
}

// CHECK-LABEL: @test_vld1q_u32(
// CHECK:   [[TMP2:%.*]] = load <4 x i32>, ptr %a, align 4
// CHECK:   ret <4 x i32> [[TMP2]]
uint32x4_t test_vld1q_u32(uint32_t const *a) {
  return vld1q_u32(a);
}

// CHECK-LABEL: @test_vld1q_u64(
// CHECK:   [[TMP2:%.*]] = load <2 x i64>, ptr %a, align 8
// CHECK:   ret <2 x i64> [[TMP2]]
uint64x2_t test_vld1q_u64(uint64_t const *a) {
  return vld1q_u64(a);
}

// CHECK-LABEL: @test_vld1q_s8(
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, ptr %a, align 1
// CHECK:   ret <16 x i8> [[TMP1]]
int8x16_t test_vld1q_s8(int8_t const *a) {
  return vld1q_s8(a);
}

// CHECK-LABEL: @test_vld1q_s16(
// CHECK:   [[TMP2:%.*]] = load <8 x i16>, ptr %a, align 2
// CHECK:   ret <8 x i16> [[TMP2]]
int16x8_t test_vld1q_s16(int16_t const *a) {
  return vld1q_s16(a);
}

// CHECK-LABEL: @test_vld1q_s32(
// CHECK:   [[TMP2:%.*]] = load <4 x i32>, ptr %a, align 4
// CHECK:   ret <4 x i32> [[TMP2]]
int32x4_t test_vld1q_s32(int32_t const *a) {
  return vld1q_s32(a);
}

// CHECK-LABEL: @test_vld1q_s64(
// CHECK:   [[TMP2:%.*]] = load <2 x i64>, ptr %a, align 8
// CHECK:   ret <2 x i64> [[TMP2]]
int64x2_t test_vld1q_s64(int64_t const *a) {
  return vld1q_s64(a);
}

// CHECK-LABEL: @test_vld1q_f16(
// CHECK:   [[TMP2:%.*]] = load <8 x half>, ptr %a, align 2
// CHECK:   ret <8 x half> [[TMP2]]
float16x8_t test_vld1q_f16(float16_t const *a) {
  return vld1q_f16(a);
}

// CHECK-LABEL: @test_vld1q_f32(
// CHECK:   [[TMP2:%.*]] = load <4 x float>, ptr %a, align 4
// CHECK:   ret <4 x float> [[TMP2]]
float32x4_t test_vld1q_f32(float32_t const *a) {
  return vld1q_f32(a);
}

// CHECK-LABEL: @test_vld1q_f64(
// CHECK:   [[TMP2:%.*]] = load <2 x double>, ptr %a, align 8
// CHECK:   ret <2 x double> [[TMP2]]
float64x2_t test_vld1q_f64(float64_t const *a) {
  return vld1q_f64(a);
}

// CHECK-LABEL: @test_vld1q_p8(
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, ptr %a, align 1
// CHECK:   ret <16 x i8> [[TMP1]]
poly8x16_t test_vld1q_p8(poly8_t const *a) {
  return vld1q_p8(a);
}

// CHECK-LABEL: @test_vld1q_p16(
// CHECK:   [[TMP2:%.*]] = load <8 x i16>, ptr %a, align 2
// CHECK:   ret <8 x i16> [[TMP2]]
poly16x8_t test_vld1q_p16(poly16_t const *a) {
  return vld1q_p16(a);
}

// CHECK-LABEL: @test_vld1_u8(
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// CHECK:   ret <8 x i8> [[TMP1]]
uint8x8_t test_vld1_u8(uint8_t const *a) {
  return vld1_u8(a);
}

// CHECK-LABEL: @test_vld1_u16(
// CHECK:   [[TMP2:%.*]] = load <4 x i16>, ptr %a, align 2
// CHECK:   ret <4 x i16> [[TMP2]]
uint16x4_t test_vld1_u16(uint16_t const *a) {
  return vld1_u16(a);
}

// CHECK-LABEL: @test_vld1_u32(
// CHECK:   [[TMP2:%.*]] = load <2 x i32>, ptr %a, align 4
// CHECK:   ret <2 x i32> [[TMP2]]
uint32x2_t test_vld1_u32(uint32_t const *a) {
  return vld1_u32(a);
}

// CHECK-LABEL: @test_vld1_u64(
// CHECK:   [[TMP2:%.*]] = load <1 x i64>, ptr %a, align 8
// CHECK:   ret <1 x i64> [[TMP2]]
uint64x1_t test_vld1_u64(uint64_t const *a) {
  return vld1_u64(a);
}

// CHECK-LABEL: @test_vld1_s8(
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// CHECK:   ret <8 x i8> [[TMP1]]
int8x8_t test_vld1_s8(int8_t const *a) {
  return vld1_s8(a);
}

// CHECK-LABEL: @test_vld1_s16(
// CHECK:   [[TMP2:%.*]] = load <4 x i16>, ptr %a, align 2
// CHECK:   ret <4 x i16> [[TMP2]]
int16x4_t test_vld1_s16(int16_t const *a) {
  return vld1_s16(a);
}

// CHECK-LABEL: @test_vld1_s32(
// CHECK:   [[TMP2:%.*]] = load <2 x i32>, ptr %a, align 4
// CHECK:   ret <2 x i32> [[TMP2]]
int32x2_t test_vld1_s32(int32_t const *a) {
  return vld1_s32(a);
}

// CHECK-LABEL: @test_vld1_s64(
// CHECK:   [[TMP2:%.*]] = load <1 x i64>, ptr %a, align 8
// CHECK:   ret <1 x i64> [[TMP2]]
int64x1_t test_vld1_s64(int64_t const *a) {
  return vld1_s64(a);
}

// CHECK-LABEL: @test_vld1_f16(
// CHECK:   [[TMP2:%.*]] = load <4 x half>, ptr %a, align 2
// CHECK:   ret <4 x half> [[TMP2]]
float16x4_t test_vld1_f16(float16_t const *a) {
  return vld1_f16(a);
}

// CHECK-LABEL: @test_vld1_f32(
// CHECK:   [[TMP2:%.*]] = load <2 x float>, ptr %a, align 4
// CHECK:   ret <2 x float> [[TMP2]]
float32x2_t test_vld1_f32(float32_t const *a) {
  return vld1_f32(a);
}

// CHECK-LABEL: @test_vld1_f64(
// CHECK:   [[TMP2:%.*]] = load <1 x double>, ptr %a, align 8
// CHECK:   ret <1 x double> [[TMP2]]
float64x1_t test_vld1_f64(float64_t const *a) {
  return vld1_f64(a);
}

// CHECK-LABEL: @test_vld1_p8(
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// CHECK:   ret <8 x i8> [[TMP1]]
poly8x8_t test_vld1_p8(poly8_t const *a) {
  return vld1_p8(a);
}

// CHECK-LABEL: @test_vld1_p16(
// CHECK:   [[TMP2:%.*]] = load <4 x i16>, ptr %a, align 2
// CHECK:   ret <4 x i16> [[TMP2]]
poly16x4_t test_vld1_p16(poly16_t const *a) {
  return vld1_p16(a);
}

// CHECK-LABEL: @test_vld1_u8_void(
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// CHECK:   ret <8 x i8> [[TMP1]]
uint8x8_t test_vld1_u8_void(void *a) {
  return vld1_u8(a);
}

// CHECK-LABEL: @test_vld1_u16_void(
// CHECK:   [[TMP1:%.*]] = load <4 x i16>, ptr %a, align 1
// CHECK:   ret <4 x i16> [[TMP1]]
uint16x4_t test_vld1_u16_void(void *a) {
  return vld1_u16(a);
}

// CHECK-LABEL: @test_vld1_u32_void(
// CHECK:   [[TMP1:%.*]] = load <2 x i32>, ptr %a, align 1
// CHECK:   ret <2 x i32> [[TMP1]]
uint32x2_t test_vld1_u32_void(void *a) {
  return vld1_u32(a);
}

// CHECK-LABEL: @test_vld1_u64_void(
// CHECK:   [[TMP1:%.*]] = load <1 x i64>, ptr %a, align 1
// CHECK:   ret <1 x i64> [[TMP1]]
uint64x1_t test_vld1_u64_void(void *a) {
  return vld1_u64(a);
}

// CHECK-LABEL: @test_vld1_s8_void(
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// CHECK:   ret <8 x i8> [[TMP1]]
int8x8_t test_vld1_s8_void(void *a) {
  return vld1_s8(a);
}

// CHECK-LABEL: @test_vld1_s16_void(
// CHECK:   [[TMP1:%.*]] = load <4 x i16>, ptr %a, align 1
// CHECK:   ret <4 x i16> [[TMP1]]
int16x4_t test_vld1_s16_void(void *a) {
  return vld1_s16(a);
}

// CHECK-LABEL: @test_vld1_s32_void(
// CHECK:   [[TMP1:%.*]] = load <2 x i32>, ptr %a, align 1
// CHECK:   ret <2 x i32> [[TMP1]]
int32x2_t test_vld1_s32_void(void *a) {
  return vld1_s32(a);
}

// CHECK-LABEL: @test_vld1_s64_void(
// CHECK:   [[TMP1:%.*]] = load <1 x i64>, ptr %a, align 1
// CHECK:   ret <1 x i64> [[TMP1]]
int64x1_t test_vld1_s64_void(void *a) {
  return vld1_s64(a);
}

// CHECK-LABEL: @test_vld1_f16_void(
// CHECK:   [[TMP1:%.*]] = load <4 x half>, ptr %a, align 1
// CHECK:   ret <4 x half> [[TMP1]]
float16x4_t test_vld1_f16_void(void *a) {
  return vld1_f16(a);
}

// CHECK-LABEL: @test_vld1_f32_void(
// CHECK:   [[TMP1:%.*]] = load <2 x float>, ptr %a, align 1
// CHECK:   ret <2 x float> [[TMP1]]
float32x2_t test_vld1_f32_void(void *a) {
  return vld1_f32(a);
}

// CHECK-LABEL: @test_vld1_f64_void(
// CHECK:   [[TMP1:%.*]] = load <1 x double>, ptr %a, align 1
// CHECK:   ret <1 x double> [[TMP1]]
float64x1_t test_vld1_f64_void(void *a) {
  return vld1_f64(a);
}

// CHECK-LABEL: @test_vld1_p8_void(
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// CHECK:   ret <8 x i8> [[TMP1]]
poly8x8_t test_vld1_p8_void(void *a) {
  return vld1_p8(a);
}

// CHECK-LABEL: @test_vld1_p16_void(
// CHECK:   [[TMP1:%.*]] = load <4 x i16>, ptr %a, align 1
// CHECK:   ret <4 x i16> [[TMP1]]
poly16x4_t test_vld1_p16_void(void *a) {
  return vld1_p16(a);
}

// CHECK-LABEL: @test_vld2q_u8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x2_t [[TMP5]]
uint8x16x2_t test_vld2q_u8(uint8_t const *a) {
  return vld2q_u8(a);
}

// CHECK-LABEL: @test_vld2q_u16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x2_t [[TMP6]]
uint16x8x2_t test_vld2q_u16(uint16_t const *a) {
  return vld2q_u16(a);
}

// CHECK-LABEL: @test_vld2q_u32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
// CHECK:   store { <4 x i32>, <4 x i32> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x2_t [[TMP6]]
uint32x4x2_t test_vld2q_u32(uint32_t const *a) {
  return vld2q_u32(a);
}

// CHECK-LABEL: @test_vld2q_u64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x2x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x2_t [[TMP6]]
uint64x2x2_t test_vld2q_u64(uint64_t const *a) {
  return vld2q_u64(a);
}

// CHECK-LABEL: @test_vld2q_s8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x2_t [[TMP5]]
int8x16x2_t test_vld2q_s8(int8_t const *a) {
  return vld2q_s8(a);
}

// CHECK-LABEL: @test_vld2q_s16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x2_t [[TMP6]]
int16x8x2_t test_vld2q_s16(int16_t const *a) {
  return vld2q_s16(a);
}

// CHECK-LABEL: @test_vld2q_s32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
// CHECK:   store { <4 x i32>, <4 x i32> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x2_t [[TMP6]]
int32x4x2_t test_vld2q_s32(int32_t const *a) {
  return vld2q_s32(a);
}

// CHECK-LABEL: @test_vld2q_s64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x2x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x2_t [[TMP6]]
int64x2x2_t test_vld2q_s64(int64_t const *a) {
  return vld2q_s64(a);
}

// CHECK-LABEL: @test_vld2q_f16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2.v8f16.p0(ptr %a)
// CHECK:   store { <8 x half>, <8 x half> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float16x8x2_t [[TMP6]]
float16x8x2_t test_vld2q_f16(float16_t const *a) {
  return vld2q_f16(a);
}

// CHECK-LABEL: @test_vld2q_f32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <4 x float>, <4 x float> } @llvm.aarch64.neon.ld2.v4f32.p0(ptr %a)
// CHECK:   store { <4 x float>, <4 x float> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float32x4x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x2_t [[TMP6]]
float32x4x2_t test_vld2q_f32(float32_t const *a) {
  return vld2q_f32(a);
}

// CHECK-LABEL: @test_vld2q_f64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <2 x double>, <2 x double> } @llvm.aarch64.neon.ld2.v2f64.p0(ptr %a)
// CHECK:   store { <2 x double>, <2 x double> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x2_t [[TMP6]]
float64x2x2_t test_vld2q_f64(float64_t const *a) {
  return vld2q_f64(a);
}

// CHECK-LABEL: @test_vld2q_p8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x2_t [[TMP5]]
poly8x16x2_t test_vld2q_p8(poly8_t const *a) {
  return vld2q_p8(a);
}

// CHECK-LABEL: @test_vld2q_p16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly16x8x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x2_t [[TMP6]]
poly16x8x2_t test_vld2q_p16(poly16_t const *a) {
  return vld2q_p16(a);
}

// CHECK-LABEL: @test_vld2_u8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x2_t [[TMP5]]
uint8x8x2_t test_vld2_u8(uint8_t const *a) {
  return vld2_u8(a);
}

// CHECK-LABEL: @test_vld2_u16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x2_t [[TMP6]]
uint16x4x2_t test_vld2_u16(uint16_t const *a) {
  return vld2_u16(a);
}

// CHECK-LABEL: @test_vld2_u32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld2.v2i32.p0(ptr %a)
// CHECK:   store { <2 x i32>, <2 x i32> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x2_t [[TMP6]]
uint32x2x2_t test_vld2_u32(uint32_t const *a) {
  return vld2_u32(a);
}

// CHECK-LABEL: @test_vld2_u64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x1x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint64x1x2_t [[TMP6]]
uint64x1x2_t test_vld2_u64(uint64_t const *a) {
  return vld2_u64(a);
}

// CHECK-LABEL: @test_vld2_s8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x2_t [[TMP5]]
int8x8x2_t test_vld2_s8(int8_t const *a) {
  return vld2_s8(a);
}

// CHECK-LABEL: @test_vld2_s16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x2_t [[TMP6]]
int16x4x2_t test_vld2_s16(int16_t const *a) {
  return vld2_s16(a);
}

// CHECK-LABEL: @test_vld2_s32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld2.v2i32.p0(ptr %a)
// CHECK:   store { <2 x i32>, <2 x i32> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x2_t [[TMP6]]
int32x2x2_t test_vld2_s32(int32_t const *a) {
  return vld2_s32(a);
}

// CHECK-LABEL: @test_vld2_s64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x1x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int64x1x2_t [[TMP6]]
int64x1x2_t test_vld2_s64(int64_t const *a) {
  return vld2_s64(a);
}

// CHECK-LABEL: @test_vld2_f16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2.v4f16.p0(ptr %a)
// CHECK:   store { <4 x half>, <4 x half> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float16x4x2_t [[TMP6]]
float16x4x2_t test_vld2_f16(float16_t const *a) {
  return vld2_f16(a);
}

// CHECK-LABEL: @test_vld2_f32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <2 x float>, <2 x float> } @llvm.aarch64.neon.ld2.v2f32.p0(ptr %a)
// CHECK:   store { <2 x float>, <2 x float> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float32x2x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x2_t [[TMP6]]
float32x2x2_t test_vld2_f32(float32_t const *a) {
  return vld2_f32(a);
}

// CHECK-LABEL: @test_vld2_f64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <1 x double>, <1 x double> } @llvm.aarch64.neon.ld2.v1f64.p0(ptr %a)
// CHECK:   store { <1 x double>, <1 x double> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x2_t [[TMP6]]
float64x1x2_t test_vld2_f64(float64_t const *a) {
  return vld2_f64(a);
}

// CHECK-LABEL: @test_vld2_p8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x2_t [[TMP5]]
poly8x8x2_t test_vld2_p8(poly8_t const *a) {
  return vld2_p8(a);
}

// CHECK-LABEL: @test_vld2_p16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly16x4x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x2_t [[TMP6]]
poly16x4x2_t test_vld2_p16(poly16_t const *a) {
  return vld2_p16(a);
}

// CHECK-LABEL: @test_vld3q_u8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x3_t [[TMP5]]
uint8x16x3_t test_vld3q_u8(uint8_t const *a) {
  return vld3q_u8(a);
}

// CHECK-LABEL: @test_vld3q_u16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint16x8x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x3_t [[TMP6]]
uint16x8x3_t test_vld3q_u16(uint16_t const *a) {
  return vld3q_u16(a);
}

// CHECK-LABEL: @test_vld3q_u32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0(ptr %a)
// CHECK:   store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint32x4x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x3_t [[TMP6]]
uint32x4x3_t test_vld3q_u32(uint32_t const *a) {
  return vld3q_u32(a);
}

// CHECK-LABEL: @test_vld3q_u64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x2x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x3_t [[TMP6]]
uint64x2x3_t test_vld3q_u64(uint64_t const *a) {
  return vld3q_u64(a);
}

// CHECK-LABEL: @test_vld3q_s8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x3_t [[TMP5]]
int8x16x3_t test_vld3q_s8(int8_t const *a) {
  return vld3q_s8(a);
}

// CHECK-LABEL: @test_vld3q_s16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int16x8x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x3_t [[TMP6]]
int16x8x3_t test_vld3q_s16(int16_t const *a) {
  return vld3q_s16(a);
}

// CHECK-LABEL: @test_vld3q_s32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0(ptr %a)
// CHECK:   store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int32x4x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x3_t [[TMP6]]
int32x4x3_t test_vld3q_s32(int32_t const *a) {
  return vld3q_s32(a);
}

// CHECK-LABEL: @test_vld3q_s64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x2x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x3_t [[TMP6]]
int64x2x3_t test_vld3q_s64(int64_t const *a) {
  return vld3q_s64(a);
}

// CHECK-LABEL: @test_vld3q_f16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3.v8f16.p0(ptr %a)
// CHECK:   store { <8 x half>, <8 x half>, <8 x half> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float16x8x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float16x8x3_t [[TMP6]]
float16x8x3_t test_vld3q_f16(float16_t const *a) {
  return vld3q_f16(a);
}

// CHECK-LABEL: @test_vld3q_f32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld3.v4f32.p0(ptr %a)
// CHECK:   store { <4 x float>, <4 x float>, <4 x float> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float32x4x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x3_t [[TMP6]]
float32x4x3_t test_vld3q_f32(float32_t const *a) {
  return vld3q_f32(a);
}

// CHECK-LABEL: @test_vld3q_f64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld3.v2f64.p0(ptr %a)
// CHECK:   store { <2 x double>, <2 x double>, <2 x double> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x3_t [[TMP6]]
float64x2x3_t test_vld3q_f64(float64_t const *a) {
  return vld3q_f64(a);
}

// CHECK-LABEL: @test_vld3q_p8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x3_t [[TMP5]]
poly8x16x3_t test_vld3q_p8(poly8_t const *a) {
  return vld3q_p8(a);
}

// CHECK-LABEL: @test_vld3q_p16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly16x8x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x3_t [[TMP6]]
poly16x8x3_t test_vld3q_p16(poly16_t const *a) {
  return vld3q_p16(a);
}

// CHECK-LABEL: @test_vld3_u8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x3_t [[TMP5]]
uint8x8x3_t test_vld3_u8(uint8_t const *a) {
  return vld3_u8(a);
}

// CHECK-LABEL: @test_vld3_u16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint16x4x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x3_t [[TMP6]]
uint16x4x3_t test_vld3_u16(uint16_t const *a) {
  return vld3_u16(a);
}

// CHECK-LABEL: @test_vld3_u32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld3.v2i32.p0(ptr %a)
// CHECK:   store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint32x2x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x3_t [[TMP6]]
uint32x2x3_t test_vld3_u32(uint32_t const *a) {
  return vld3_u32(a);
}

// CHECK-LABEL: @test_vld3_u64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x1x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint64x1x3_t [[TMP6]]
uint64x1x3_t test_vld3_u64(uint64_t const *a) {
  return vld3_u64(a);
}

// CHECK-LABEL: @test_vld3_s8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x3_t [[TMP5]]
int8x8x3_t test_vld3_s8(int8_t const *a) {
  return vld3_s8(a);
}

// CHECK-LABEL: @test_vld3_s16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int16x4x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x3_t [[TMP6]]
int16x4x3_t test_vld3_s16(int16_t const *a) {
  return vld3_s16(a);
}

// CHECK-LABEL: @test_vld3_s32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld3.v2i32.p0(ptr %a)
// CHECK:   store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int32x2x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x3_t [[TMP6]]
int32x2x3_t test_vld3_s32(int32_t const *a) {
  return vld3_s32(a);
}

// CHECK-LABEL: @test_vld3_s64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x1x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int64x1x3_t [[TMP6]]
int64x1x3_t test_vld3_s64(int64_t const *a) {
  return vld3_s64(a);
}

// CHECK-LABEL: @test_vld3_f16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3.v4f16.p0(ptr %a)
// CHECK:   store { <4 x half>, <4 x half>, <4 x half> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float16x4x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float16x4x3_t [[TMP6]]
float16x4x3_t test_vld3_f16(float16_t const *a) {
  return vld3_f16(a);
}

// CHECK-LABEL: @test_vld3_f32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld3.v2f32.p0(ptr %a)
// CHECK:   store { <2 x float>, <2 x float>, <2 x float> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float32x2x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x3_t [[TMP6]]
float32x2x3_t test_vld3_f32(float32_t const *a) {
  return vld3_f32(a);
}

// CHECK-LABEL: @test_vld3_f64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld3.v1f64.p0(ptr %a)
// CHECK:   store { <1 x double>, <1 x double>, <1 x double> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x3_t [[TMP6]]
float64x1x3_t test_vld3_f64(float64_t const *a) {
  return vld3_f64(a);
}

// CHECK-LABEL: @test_vld3_p8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x3_t [[TMP5]]
poly8x8x3_t test_vld3_p8(poly8_t const *a) {
  return vld3_p8(a);
}

// CHECK-LABEL: @test_vld3_p16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly16x4x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x3_t [[TMP6]]
poly16x4x3_t test_vld3_p16(poly16_t const *a) {
  return vld3_p16(a);
}

// CHECK-LABEL: @test_vld4q_u8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x16x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint8x16x4_t [[TMP5]]
uint8x16x4_t test_vld4q_u8(uint8_t const *a) {
  return vld4q_u8(a);
}

// CHECK-LABEL: @test_vld4q_u16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint16x8x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint16x8x4_t [[TMP6]]
uint16x8x4_t test_vld4q_u16(uint16_t const *a) {
  return vld4q_u16(a);
}

// CHECK-LABEL: @test_vld4q_u32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4.v4i32.p0(ptr %a)
// CHECK:   store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint32x4x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint32x4x4_t [[TMP6]]
uint32x4x4_t test_vld4q_u32(uint32_t const *a) {
  return vld4q_u32(a);
}

// CHECK-LABEL: @test_vld4q_u64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x2x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.uint64x2x4_t [[TMP6]]
uint64x2x4_t test_vld4q_u64(uint64_t const *a) {
  return vld4q_u64(a);
}

// CHECK-LABEL: @test_vld4q_s8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.int8x16x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int8x16x4_t [[TMP5]]
int8x16x4_t test_vld4q_s8(int8_t const *a) {
  return vld4q_s8(a);
}

// CHECK-LABEL: @test_vld4q_s16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int16x8x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int16x8x4_t [[TMP6]]
int16x8x4_t test_vld4q_s16(int16_t const *a) {
  return vld4q_s16(a);
}

// CHECK-LABEL: @test_vld4q_s32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4.v4i32.p0(ptr %a)
// CHECK:   store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int32x4x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int32x4x4_t [[TMP6]]
int32x4x4_t test_vld4q_s32(int32_t const *a) {
  return vld4q_s32(a);
}

// CHECK-LABEL: @test_vld4q_s64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x2x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.int64x2x4_t [[TMP6]]
int64x2x4_t test_vld4q_s64(int64_t const *a) {
  return vld4q_s64(a);
}

// CHECK-LABEL: @test_vld4q_f16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4.v8f16.p0(ptr %a)
// CHECK:   store { <8 x half>, <8 x half>, <8 x half>, <8 x half> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float16x8x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float16x8x4_t [[TMP6]]
float16x8x4_t test_vld4q_f16(float16_t const *a) {
  return vld4q_f16(a);
}

// CHECK-LABEL: @test_vld4q_f32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld4.v4f32.p0(ptr %a)
// CHECK:   store { <4 x float>, <4 x float>, <4 x float>, <4 x float> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float32x4x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float32x4x4_t [[TMP6]]
float32x4x4_t test_vld4q_f32(float32_t const *a) {
  return vld4q_f32(a);
}

// CHECK-LABEL: @test_vld4q_f64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld4.v2f64.p0(ptr %a)
// CHECK:   store { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x4_t [[TMP6]]
float64x2x4_t test_vld4q_f64(float64_t const *a) {
  return vld4q_f64(a);
}

// CHECK-LABEL: @test_vld4q_p8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4.v16i8.p0(ptr %a)
// CHECK:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x16x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly8x16x4_t [[TMP5]]
poly8x16x4_t test_vld4q_p8(poly8_t const *a) {
  return vld4q_p8(a);
}

// CHECK-LABEL: @test_vld4q_p16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4.v8i16.p0(ptr %a)
// CHECK:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly16x8x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly16x8x4_t [[TMP6]]
poly16x8x4_t test_vld4q_p16(poly16_t const *a) {
  return vld4q_p16(a);
}

// CHECK-LABEL: @test_vld4_u8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.uint8x8x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint8x8x4_t [[TMP5]]
uint8x8x4_t test_vld4_u8(uint8_t const *a) {
  return vld4_u8(a);
}

// CHECK-LABEL: @test_vld4_u16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint16x4x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint16x4x4_t [[TMP6]]
uint16x4x4_t test_vld4_u16(uint16_t const *a) {
  return vld4_u16(a);
}

// CHECK-LABEL: @test_vld4_u32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld4.v2i32.p0(ptr %a)
// CHECK:   store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint32x2x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint32x2x4_t [[TMP6]]
uint32x2x4_t test_vld4_u32(uint32_t const *a) {
  return vld4_u32(a);
}

// CHECK-LABEL: @test_vld4_u64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.uint64x1x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.uint64x1x4_t [[TMP6]]
uint64x1x4_t test_vld4_u64(uint64_t const *a) {
  return vld4_u64(a);
}

// CHECK-LABEL: @test_vld4_s8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.int8x8x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int8x8x4_t [[TMP5]]
int8x8x4_t test_vld4_s8(int8_t const *a) {
  return vld4_s8(a);
}

// CHECK-LABEL: @test_vld4_s16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int16x4x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int16x4x4_t [[TMP6]]
int16x4x4_t test_vld4_s16(int16_t const *a) {
  return vld4_s16(a);
}

// CHECK-LABEL: @test_vld4_s32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld4.v2i32.p0(ptr %a)
// CHECK:   store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int32x2x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int32x2x4_t [[TMP6]]
int32x2x4_t test_vld4_s32(int32_t const *a) {
  return vld4_s32(a);
}

// CHECK-LABEL: @test_vld4_s64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.int64x1x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.int64x1x4_t [[TMP6]]
int64x1x4_t test_vld4_s64(int64_t const *a) {
  return vld4_s64(a);
}

// CHECK-LABEL: @test_vld4_f16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4.v4f16.p0(ptr %a)
// CHECK:   store { <4 x half>, <4 x half>, <4 x half>, <4 x half> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float16x4x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float16x4x4_t [[TMP6]]
float16x4x4_t test_vld4_f16(float16_t const *a) {
  return vld4_f16(a);
}

// CHECK-LABEL: @test_vld4_f32(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld4.v2f32.p0(ptr %a)
// CHECK:   store { <2 x float>, <2 x float>, <2 x float>, <2 x float> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float32x2x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float32x2x4_t [[TMP6]]
float32x2x4_t test_vld4_f32(float32_t const *a) {
  return vld4_f32(a);
}

// CHECK-LABEL: @test_vld4_f64(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld4.v1f64.p0(ptr %a)
// CHECK:   store { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x4_t [[TMP6]]
float64x1x4_t test_vld4_f64(float64_t const *a) {
  return vld4_f64(a);
}

// CHECK-LABEL: @test_vld4_p8(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4.v8i8.p0(ptr %a)
// CHECK:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP5:%.*]] = load %struct.poly8x8x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly8x8x4_t [[TMP5]]
poly8x8x4_t test_vld4_p8(poly8_t const *a) {
  return vld4_p8(a);
}

// CHECK-LABEL: @test_vld4_p16(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4.v4i16.p0(ptr %a)
// CHECK:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly16x4x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly16x4x4_t [[TMP6]]
poly16x4x4_t test_vld4_p16(poly16_t const *a) {
  return vld4_p16(a);
}

// CHECK-LABEL: @test_vst1q_u8(
// CHECK:   store <16 x i8> %b, ptr %a
// CHECK:   ret void
void test_vst1q_u8(uint8_t *a, uint8x16_t b) {
  vst1q_u8(a, b);
}

// CHECK-LABEL: @test_vst1q_u16(
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   store <8 x i16> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_u16(uint16_t *a, uint16x8_t b) {
  vst1q_u16(a, b);
}

// CHECK-LABEL: @test_vst1q_u32(
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   store <4 x i32> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_u32(uint32_t *a, uint32x4_t b) {
  vst1q_u32(a, b);
}

// CHECK-LABEL: @test_vst1q_u64(
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   store <2 x i64> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_u64(uint64_t *a, uint64x2_t b) {
  vst1q_u64(a, b);
}

// CHECK-LABEL: @test_vst1q_s8(
// CHECK:   store <16 x i8> %b, ptr %a
// CHECK:   ret void
void test_vst1q_s8(int8_t *a, int8x16_t b) {
  vst1q_s8(a, b);
}

// CHECK-LABEL: @test_vst1q_s16(
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   store <8 x i16> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_s16(int16_t *a, int16x8_t b) {
  vst1q_s16(a, b);
}

// CHECK-LABEL: @test_vst1q_s32(
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// CHECK:   store <4 x i32> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_s32(int32_t *a, int32x4_t b) {
  vst1q_s32(a, b);
}

// CHECK-LABEL: @test_vst1q_s64(
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   store <2 x i64> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_s64(int64_t *a, int64x2_t b) {
  vst1q_s64(a, b);
}

// CHECK-LABEL: @test_vst1q_f16(
// CHECK:   [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// CHECK:   store <8 x half> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_f16(float16_t *a, float16x8_t b) {
  vst1q_f16(a, b);
}

// CHECK-LABEL: @test_vst1q_f32(
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   store <4 x float> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_f32(float32_t *a, float32x4_t b) {
  vst1q_f32(a, b);
}

// CHECK-LABEL: @test_vst1q_f64(
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   store <2 x double> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_f64(float64_t *a, float64x2_t b) {
  vst1q_f64(a, b);
}

// CHECK-LABEL: @test_vst1q_p8(
// CHECK:   store <16 x i8> %b, ptr %a
// CHECK:   ret void
void test_vst1q_p8(poly8_t *a, poly8x16_t b) {
  vst1q_p8(a, b);
}

// CHECK-LABEL: @test_vst1q_p16(
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:   store <8 x i16> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1q_p16(poly16_t *a, poly16x8_t b) {
  vst1q_p16(a, b);
}

// CHECK-LABEL: @test_vst1_u8(
// CHECK:   store <8 x i8> %b, ptr %a
// CHECK:   ret void
void test_vst1_u8(uint8_t *a, uint8x8_t b) {
  vst1_u8(a, b);
}

// CHECK-LABEL: @test_vst1_u16(
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   store <4 x i16> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_u16(uint16_t *a, uint16x4_t b) {
  vst1_u16(a, b);
}

// CHECK-LABEL: @test_vst1_u32(
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   store <2 x i32> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_u32(uint32_t *a, uint32x2_t b) {
  vst1_u32(a, b);
}

// CHECK-LABEL: @test_vst1_u64(
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   store <1 x i64> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_u64(uint64_t *a, uint64x1_t b) {
  vst1_u64(a, b);
}

// CHECK-LABEL: @test_vst1_s8(
// CHECK:   store <8 x i8> %b, ptr %a
// CHECK:   ret void
void test_vst1_s8(int8_t *a, int8x8_t b) {
  vst1_s8(a, b);
}

// CHECK-LABEL: @test_vst1_s16(
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   store <4 x i16> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_s16(int16_t *a, int16x4_t b) {
  vst1_s16(a, b);
}

// CHECK-LABEL: @test_vst1_s32(
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK:   store <2 x i32> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_s32(int32_t *a, int32x2_t b) {
  vst1_s32(a, b);
}

// CHECK-LABEL: @test_vst1_s64(
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   store <1 x i64> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_s64(int64_t *a, int64x1_t b) {
  vst1_s64(a, b);
}

// CHECK-LABEL: @test_vst1_f16(
// CHECK:   [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// CHECK:   store <4 x half> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_f16(float16_t *a, float16x4_t b) {
  vst1_f16(a, b);
}

// CHECK-LABEL: @test_vst1_f32(
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   store <2 x float> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_f32(float32_t *a, float32x2_t b) {
  vst1_f32(a, b);
}

// CHECK-LABEL: @test_vst1_f64(
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// CHECK:   store <1 x double> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_f64(float64_t *a, float64x1_t b) {
  vst1_f64(a, b);
}

// CHECK-LABEL: @test_vst1_p8(
// CHECK:   store <8 x i8> %b, ptr %a
// CHECK:   ret void
void test_vst1_p8(poly8_t *a, poly8x8_t b) {
  vst1_p8(a, b);
}

// CHECK-LABEL: @test_vst1_p16(
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:   store <4 x i16> [[TMP3]], ptr %a
// CHECK:   ret void
void test_vst1_p16(poly16_t *a, poly16x4_t b) {
  vst1_p16(a, b);
}

// CHECK-LABEL: @test_vst2q_u8(
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x16x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// CHECK:   ret void
void test_vst2q_u8(uint8_t *a, uint8x16x2_t b) {
  vst2q_u8(a, b);
}

// CHECK-LABEL: @test_vst2q_u16(
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_u16(uint16_t *a, uint16x8x2_t b) {
  vst2q_u16(a, b);
}

// CHECK-LABEL: @test_vst2q_u32(
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> [[TMP7]], <4 x i32> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_u32(uint32_t *a, uint32x4x2_t b) {
  vst2q_u32(a, b);
}

// CHECK-LABEL: @test_vst2q_u64(
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_u64(uint64_t *a, uint64x2x2_t b) {
  vst2q_u64(a, b);
}

// CHECK-LABEL: @test_vst2q_s8(
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x16x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// CHECK:   ret void
void test_vst2q_s8(int8_t *a, int8x16x2_t b) {
  vst2q_s8(a, b);
}

// CHECK-LABEL: @test_vst2q_s16(
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_s16(int16_t *a, int16x8x2_t b) {
  vst2q_s16(a, b);
}

// CHECK-LABEL: @test_vst2q_s32(
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> [[TMP7]], <4 x i32> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_s32(int32_t *a, int32x4x2_t b) {
  vst2q_s32(a, b);
}

// CHECK-LABEL: @test_vst2q_s64(
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_s64(int64_t *a, int64x2x2_t b) {
  vst2q_s64(a, b);
}

// CHECK-LABEL: @test_vst2q_f16(
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x half>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x half>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// CHECK:   call void @llvm.aarch64.neon.st2.v8f16.p0(<8 x half> [[TMP7]], <8 x half> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_f16(float16_t *a, float16x8x2_t b) {
  vst2q_f16(a, b);
}

// CHECK-LABEL: @test_vst2q_f32(
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x float>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x float>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK:   call void @llvm.aarch64.neon.st2.v4f32.p0(<4 x float> [[TMP7]], <4 x float> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_f32(float32_t *a, float32x4x2_t b) {
  vst2q_f32(a, b);
}

// CHECK-LABEL: @test_vst2q_f64(
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st2.v2f64.p0(<2 x double> [[TMP7]], <2 x double> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_f64(float64_t *a, float64x2x2_t b) {
  vst2q_f64(a, b);
}

// CHECK-LABEL: @test_vst2q_p8(
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x16x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x16x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// CHECK:   ret void
void test_vst2q_p8(poly8_t *a, poly8x16x2_t b) {
  vst2q_p8(a, b);
}

// CHECK-LABEL: @test_vst2q_p16(
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2q_p16(poly16_t *a, poly16x8x2_t b) {
  vst2q_p16(a, b);
}

// CHECK-LABEL: @test_vst2_u8(
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// CHECK:   ret void
void test_vst2_u8(uint8_t *a, uint8x8x2_t b) {
  vst2_u8(a, b);
}

// CHECK-LABEL: @test_vst2_u16(
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_u16(uint16_t *a, uint16x4x2_t b) {
  vst2_u16(a, b);
}

// CHECK-LABEL: @test_vst2_u32(
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32> [[TMP7]], <2 x i32> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_u32(uint32_t *a, uint32x2x2_t b) {
  vst2_u32(a, b);
}

// CHECK-LABEL: @test_vst2_u64(
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x1x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_u64(uint64_t *a, uint64x1x2_t b) {
  vst2_u64(a, b);
}

// CHECK-LABEL: @test_vst2_s8(
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// CHECK:   ret void
void test_vst2_s8(int8_t *a, int8x8x2_t b) {
  vst2_s8(a, b);
}

// CHECK-LABEL: @test_vst2_s16(
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_s16(int16_t *a, int16x4x2_t b) {
  vst2_s16(a, b);
}

// CHECK-LABEL: @test_vst2_s32(
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32> [[TMP7]], <2 x i32> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_s32(int32_t *a, int32x2x2_t b) {
  vst2_s32(a, b);
}

// CHECK-LABEL: @test_vst2_s64(
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x1x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_s64(int64_t *a, int64x1x2_t b) {
  vst2_s64(a, b);
}

// CHECK-LABEL: @test_vst2_f16(
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x half>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x half>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// CHECK:   call void @llvm.aarch64.neon.st2.v4f16.p0(<4 x half> [[TMP7]], <4 x half> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_f16(float16_t *a, float16x4x2_t b) {
  vst2_f16(a, b);
}

// CHECK-LABEL: @test_vst2_f32(
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x float>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x float>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK:   call void @llvm.aarch64.neon.st2.v2f32.p0(<2 x float> [[TMP7]], <2 x float> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_f32(float32_t *a, float32x2x2_t b) {
  vst2_f32(a, b);
}

// CHECK-LABEL: @test_vst2_f64(
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x1x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st2.v1f64.p0(<1 x double> [[TMP7]], <1 x double> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_f64(float64_t *a, float64x1x2_t b) {
  vst2_f64(a, b);
}

// CHECK-LABEL: @test_vst2_p8(
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x8x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x8x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// CHECK:   ret void
void test_vst2_p8(poly8_t *a, poly8x8x2_t b) {
  vst2_p8(a, b);
}

// CHECK-LABEL: @test_vst2_p16(
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x4x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x4x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst2_p16(poly16_t *a, poly16x4x2_t b) {
  vst2_p16(a, b);
}

// CHECK-LABEL: @test_vst3q_u8(
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x16x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// CHECK:   call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// CHECK:   ret void
void test_vst3q_u8(uint8_t *a, uint8x16x3_t b) {
  vst3q_u8(a, b);
}

// CHECK-LABEL: @test_vst3q_u16(
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_u16(uint16_t *a, uint16x8x3_t b) {
  vst3q_u16(a, b);
}

// CHECK-LABEL: @test_vst3q_u32(
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_u32(uint32_t *a, uint32x4x3_t b) {
  vst3q_u32(a, b);
}

// CHECK-LABEL: @test_vst3q_u64(
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_u64(uint64_t *a, uint64x2x3_t b) {
  vst3q_u64(a, b);
}

// CHECK-LABEL: @test_vst3q_s8(
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x16x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// CHECK:   call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// CHECK:   ret void
void test_vst3q_s8(int8_t *a, int8x16x3_t b) {
  vst3q_s8(a, b);
}

// CHECK-LABEL: @test_vst3q_s16(
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_s16(int16_t *a, int16x8x3_t b) {
  vst3q_s16(a, b);
}

// CHECK-LABEL: @test_vst3q_s32(
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_s32(int32_t *a, int32x4x3_t b) {
  vst3q_s32(a, b);
}

// CHECK-LABEL: @test_vst3q_s64(
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_s64(int64_t *a, int64x2x3_t b) {
  vst3q_s64(a, b);
}

// CHECK-LABEL: @test_vst3q_f16(
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x half>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x half> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x half>
// CHECK:   call void @llvm.aarch64.neon.st3.v8f16.p0(<8 x half> [[TMP9]], <8 x half> [[TMP10]], <8 x half> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_f16(float16_t *a, float16x8x3_t b) {
  vst3q_f16(a, b);
}

// CHECK-LABEL: @test_vst3q_f32(
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x float>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x float> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x float>
// CHECK:   call void @llvm.aarch64.neon.st3.v4f32.p0(<4 x float> [[TMP9]], <4 x float> [[TMP10]], <4 x float> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_f32(float32_t *a, float32x4x3_t b) {
  vst3q_f32(a, b);
}

// CHECK-LABEL: @test_vst3q_f64(
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x double>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st3.v2f64.p0(<2 x double> [[TMP9]], <2 x double> [[TMP10]], <2 x double> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_f64(float64_t *a, float64x2x3_t b) {
  vst3q_f64(a, b);
}

// CHECK-LABEL: @test_vst3q_p8(
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x16x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// CHECK:   call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// CHECK:   ret void
void test_vst3q_p8(poly8_t *a, poly8x16x3_t b) {
  vst3q_p8(a, b);
}

// CHECK-LABEL: @test_vst3q_p16(
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3q_p16(poly16_t *a, poly16x8x3_t b) {
  vst3q_p16(a, b);
}

// CHECK-LABEL: @test_vst3_u8(
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK:   call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// CHECK:   ret void
void test_vst3_u8(uint8_t *a, uint8x8x3_t b) {
  vst3_u8(a, b);
}

// CHECK-LABEL: @test_vst3_u16(
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_u16(uint16_t *a, uint16x4x3_t b) {
  vst3_u16(a, b);
}

// CHECK-LABEL: @test_vst3_u32(
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_u32(uint32_t *a, uint32x2x3_t b) {
  vst3_u32(a, b);
}

// CHECK-LABEL: @test_vst3_u64(
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x1x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_u64(uint64_t *a, uint64x1x3_t b) {
  vst3_u64(a, b);
}

// CHECK-LABEL: @test_vst3_s8(
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK:   call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// CHECK:   ret void
void test_vst3_s8(int8_t *a, int8x8x3_t b) {
  vst3_s8(a, b);
}

// CHECK-LABEL: @test_vst3_s16(
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_s16(int16_t *a, int16x4x3_t b) {
  vst3_s16(a, b);
}

// CHECK-LABEL: @test_vst3_s32(
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_s32(int32_t *a, int32x2x3_t b) {
  vst3_s32(a, b);
}

// CHECK-LABEL: @test_vst3_s64(
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x1x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_s64(int64_t *a, int64x1x3_t b) {
  vst3_s64(a, b);
}

// CHECK-LABEL: @test_vst3_f16(
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x half>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x half> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x half>
// CHECK:   call void @llvm.aarch64.neon.st3.v4f16.p0(<4 x half> [[TMP9]], <4 x half> [[TMP10]], <4 x half> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_f16(float16_t *a, float16x4x3_t b) {
  vst3_f16(a, b);
}

// CHECK-LABEL: @test_vst3_f32(
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x float>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x float> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x float>
// CHECK:   call void @llvm.aarch64.neon.st3.v2f32.p0(<2 x float> [[TMP9]], <2 x float> [[TMP10]], <2 x float> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_f32(float32_t *a, float32x2x3_t b) {
  vst3_f32(a, b);
}

// CHECK-LABEL: @test_vst3_f64(
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x1x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x double>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st3.v1f64.p0(<1 x double> [[TMP9]], <1 x double> [[TMP10]], <1 x double> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_f64(float64_t *a, float64x1x3_t b) {
  vst3_f64(a, b);
}

// CHECK-LABEL: @test_vst3_p8(
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x8x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK:   call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// CHECK:   ret void
void test_vst3_p8(poly8_t *a, poly8x8x3_t b) {
  vst3_p8(a, b);
}

// CHECK-LABEL: @test_vst3_p16(
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x4x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst3_p16(poly16_t *a, poly16x4x3_t b) {
  vst3_p16(a, b);
}

// CHECK-LABEL: @test_vst4q_u8(
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align 16
// CHECK:   call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// CHECK:   ret void
void test_vst4q_u8(uint8_t *a, uint8x16x4_t b) {
  vst4q_u8(a, b);
}

// CHECK-LABEL: @test_vst4q_u16(
// CHECK:   [[B:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_u16(uint16_t *a, uint16x8x4_t b) {
  vst4q_u16(a, b);
}

// CHECK-LABEL: @test_vst4q_u32(
// CHECK:   [[B:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i32>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i32> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_u32(uint32_t *a, uint32x4x4_t b) {
  vst4q_u32(a, b);
}

// CHECK-LABEL: @test_vst4q_u64(
// CHECK:   [[B:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_u64(uint64_t *a, uint64x2x4_t b) {
  vst4q_u64(a, b);
}

// CHECK-LABEL: @test_vst4q_s8(
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align 16
// CHECK:   call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// CHECK:   ret void
void test_vst4q_s8(int8_t *a, int8x16x4_t b) {
  vst4q_s8(a, b);
}

// CHECK-LABEL: @test_vst4q_s16(
// CHECK:   [[B:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_s16(int16_t *a, int16x8x4_t b) {
  vst4q_s16(a, b);
}

// CHECK-LABEL: @test_vst4q_s32(
// CHECK:   [[B:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i32>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i32> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x i32>
// CHECK:   call void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_s32(int32_t *a, int32x4x4_t b) {
  vst4q_s32(a, b);
}

// CHECK-LABEL: @test_vst4q_s64(
// CHECK:   [[B:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_s64(int64_t *a, int64x2x4_t b) {
  vst4q_s64(a, b);
}

// CHECK-LABEL: @test_vst4q_f16(
// CHECK:   [[B:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x half>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x half> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <8 x half>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <8 x half> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x half>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x half>
// CHECK:   call void @llvm.aarch64.neon.st4.v8f16.p0(<8 x half> [[TMP11]], <8 x half> [[TMP12]], <8 x half> [[TMP13]], <8 x half> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_f16(float16_t *a, float16x8x4_t b) {
  vst4q_f16(a, b);
}

// CHECK-LABEL: @test_vst4q_f32(
// CHECK:   [[B:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x4x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x float>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <4 x float> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x float>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <4 x float> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x float>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x float>
// CHECK:   call void @llvm.aarch64.neon.st4.v4f32.p0(<4 x float> [[TMP11]], <4 x float> [[TMP12]], <4 x float> [[TMP13]], <4 x float> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_f32(float32_t *a, float32x4x4_t b) {
  vst4q_f32(a, b);
}

// CHECK-LABEL: @test_vst4q_f64(
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x double>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x double>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x double> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st4.v2f64.p0(<2 x double> [[TMP11]], <2 x double> [[TMP12]], <2 x double> [[TMP13]], <2 x double> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_f64(float64_t *a, float64x2x4_t b) {
  vst4q_f64(a, b);
}

// CHECK-LABEL: @test_vst4q_p8(
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align 16
// CHECK:   call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// CHECK:   ret void
void test_vst4q_p8(poly8_t *a, poly8x16x4_t b) {
  vst4q_p8(a, b);
}

// CHECK-LABEL: @test_vst4q_p16(
// CHECK:   [[B:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x8x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4q_p16(poly16_t *a, poly16x8x4_t b) {
  vst4q_p16(a, b);
}

// CHECK-LABEL: @test_vst4_u8(
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// CHECK:   call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// CHECK:   ret void
void test_vst4_u8(uint8_t *a, uint8x8x4_t b) {
  vst4_u8(a, b);
}

// CHECK-LABEL: @test_vst4_u16(
// CHECK:   [[B:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_u16(uint16_t *a, uint16x4x4_t b) {
  vst4_u16(a, b);
}

// CHECK-LABEL: @test_vst4_u32(
// CHECK:   [[B:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i32>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i32> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_u32(uint32_t *a, uint32x2x4_t b) {
  vst4_u32(a, b);
}

// CHECK-LABEL: @test_vst4_u64(
// CHECK:   [[B:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.uint64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_u64(uint64_t *a, uint64x1x4_t b) {
  vst4_u64(a, b);
}

// CHECK-LABEL: @test_vst4_s8(
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// CHECK:   call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// CHECK:   ret void
void test_vst4_s8(int8_t *a, int8x8x4_t b) {
  vst4_s8(a, b);
}

// CHECK-LABEL: @test_vst4_s16(
// CHECK:   [[B:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_s16(int16_t *a, int16x4x4_t b) {
  vst4_s16(a, b);
}

// CHECK-LABEL: @test_vst4_s32(
// CHECK:   [[B:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i32>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i32> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x i32>
// CHECK:   call void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_s32(int32_t *a, int32x2x4_t b) {
  vst4_s32(a, b);
}

// CHECK-LABEL: @test_vst4_s64(
// CHECK:   [[B:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.int64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_s64(int64_t *a, int64x1x4_t b) {
  vst4_s64(a, b);
}

// CHECK-LABEL: @test_vst4_f16(
// CHECK:   [[B:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x half>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x half> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x half>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <4 x half> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x half>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x half>
// CHECK:   call void @llvm.aarch64.neon.st4.v4f16.p0(<4 x half> [[TMP11]], <4 x half> [[TMP12]], <4 x half> [[TMP13]], <4 x half> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_f16(float16_t *a, float16x4x4_t b) {
  vst4_f16(a, b);
}

// CHECK-LABEL: @test_vst4_f32(
// CHECK:   [[B:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float32x2x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x float>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <2 x float> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x float>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <2 x float> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x float>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x float>
// CHECK:   call void @llvm.aarch64.neon.st4.v2f32.p0(<2 x float> [[TMP11]], <2 x float> [[TMP12]], <2 x float> [[TMP13]], <2 x float> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_f32(float32_t *a, float32x2x4_t b) {
  vst4_f32(a, b);
}

// CHECK-LABEL: @test_vst4_f64(
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x double>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x double>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x double> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st4.v1f64.p0(<1 x double> [[TMP11]], <1 x double> [[TMP12]], <1 x double> [[TMP13]], <1 x double> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_f64(float64_t *a, float64x1x4_t b) {
  vst4_f64(a, b);
}

// CHECK-LABEL: @test_vst4_p8(
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// CHECK:   call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// CHECK:   ret void
void test_vst4_p8(poly8_t *a, poly8x8x4_t b) {
  vst4_p8(a, b);
}

// CHECK-LABEL: @test_vst4_p16(
// CHECK:   [[B:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly16x4x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// CHECK:   call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst4_p16(poly16_t *a, poly16x4x4_t b) {
  vst4_p16(a, b);
}

// CHECK-LABEL: @test_vld1q_f64_x2(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[VLD1XN:%.*]] = call { <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x2.v2f64.p0(ptr %a)
// CHECK:   store { <2 x double>, <2 x double> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x2_t [[TMP6]]
float64x2x2_t test_vld1q_f64_x2(float64_t const *a) {
  return vld1q_f64_x2(a);
}

// CHECK-LABEL: @test_vld1q_p64_x2(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x2.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x2_t [[TMP6]]
poly64x2x2_t test_vld1q_p64_x2(poly64_t const *a) {
  return vld1q_p64_x2(a);
}

// CHECK-LABEL: @test_vld1_f64_x2(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[VLD1XN:%.*]] = call { <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x2.v1f64.p0(ptr %a)
// CHECK:   store { <1 x double>, <1 x double> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x2_t [[TMP6]]
float64x1x2_t test_vld1_f64_x2(float64_t const *a) {
  return vld1_f64_x2(a);
}

// CHECK-LABEL: @test_vld1_p64_x2(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x2.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x2_t [[TMP6]]
poly64x1x2_t test_vld1_p64_x2(poly64_t const *a) {
  return vld1_p64_x2(a);
}

// CHECK-LABEL: @test_vld1q_f64_x3(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[VLD1XN:%.*]] = call { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x3.v2f64.p0(ptr %a)
// CHECK:   store { <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x3_t [[TMP6]]
float64x2x3_t test_vld1q_f64_x3(float64_t const *a) {
  return vld1q_f64_x3(a);
}

// CHECK-LABEL: @test_vld1q_p64_x3(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x3.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x3_t [[TMP6]]
poly64x2x3_t test_vld1q_p64_x3(poly64_t const *a) {
  return vld1q_p64_x3(a);
}

// CHECK-LABEL: @test_vld1_f64_x3(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[VLD1XN:%.*]] = call { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x3.v1f64.p0(ptr %a)
// CHECK:   store { <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x3_t [[TMP6]]
float64x1x3_t test_vld1_f64_x3(float64_t const *a) {
  return vld1_f64_x3(a);
}

// CHECK-LABEL: @test_vld1_p64_x3(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x3.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x3_t [[TMP6]]
poly64x1x3_t test_vld1_p64_x3(poly64_t const *a) {
  return vld1_p64_x3(a);
}

// CHECK-LABEL: @test_vld1q_f64_x4(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[VLD1XN:%.*]] = call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x4.v2f64.p0(ptr %a)
// CHECK:   store { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x2x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.float64x2x4_t [[TMP6]]
float64x2x4_t test_vld1q_f64_x4(float64_t const *a) {
  return vld1q_f64_x4(a);
}

// CHECK-LABEL: @test_vld1q_p64_x4(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x4.v2i64.p0(ptr %a)
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x4_t [[TMP6]]
poly64x2x4_t test_vld1q_p64_x4(poly64_t const *a) {
  return vld1q_p64_x4(a);
}

// CHECK-LABEL: @test_vld1_f64_x4(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[VLD1XN:%.*]] = call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x4.v1f64.p0(ptr %a)
// CHECK:   store { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.float64x1x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.float64x1x4_t [[TMP6]]
float64x1x4_t test_vld1_f64_x4(float64_t const *a) {
  return vld1_f64_x4(a);
}

// CHECK-LABEL: @test_vld1_p64_x4(
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x4.v1i64.p0(ptr %a)
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x4_t [[TMP6]]
poly64x1x4_t test_vld1_p64_x4(poly64_t const *a) {
  return vld1_p64_x4(a);
}

// CHECK-LABEL: @test_vst1q_f64_x2(
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st1x2.v2f64.p0(<2 x double> [[TMP7]], <2 x double> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst1q_f64_x2(float64_t *a, float64x2x2_t b) {
  vst1q_f64_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_p64_x2(
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly64x2x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst1q_p64_x2(poly64_t *a, poly64x2x2_t b) {
  vst1q_p64_x2(a, b);
}

// CHECK-LABEL: @test_vst1_f64_x2(
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x1x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st1x2.v1f64.p0(<1 x double> [[TMP7]], <1 x double> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst1_f64_x2(float64_t *a, float64x1x2_t b) {
  vst1_f64_x2(a, b);
}

// CHECK-LABEL: @test_vst1_p64_x2(
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly64x1x2_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %a)
// CHECK:   ret void
void test_vst1_p64_x2(poly64_t *a, poly64x1x2_t b) {
  vst1_p64_x2(a, b);
}

// CHECK-LABEL: @test_vst1q_f64_x3(
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x double>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st1x3.v2f64.p0(<2 x double> [[TMP9]], <2 x double> [[TMP10]], <2 x double> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst1q_f64_x3(float64_t *a, float64x2x3_t b) {
  vst1q_f64_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_p64_x3(
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly64x2x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst1q_p64_x3(poly64_t *a, poly64x2x3_t b) {
  vst1q_p64_x3(a, b);
}

// CHECK-LABEL: @test_vst1_f64_x3(
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x1x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x double>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st1x3.v1f64.p0(<1 x double> [[TMP9]], <1 x double> [[TMP10]], <1 x double> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst1_f64_x3(float64_t *a, float64x1x3_t b) {
  vst1_f64_x3(a, b);
}

// CHECK-LABEL: @test_vst1_p64_x3(
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly64x1x3_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %a)
// CHECK:   ret void
void test_vst1_p64_x3(poly64_t *a, poly64x1x3_t b) {
  vst1_p64_x3(a, b);
}

// CHECK-LABEL: @test_vst1q_f64_x4(
// CHECK:   [[B:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x double>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x double>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x double> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x double>
// CHECK:   call void @llvm.aarch64.neon.st1x4.v2f64.p0(<2 x double> [[TMP11]], <2 x double> [[TMP12]], <2 x double> [[TMP13]], <2 x double> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst1q_f64_x4(float64_t *a, float64x2x4_t b) {
  vst1q_f64_x4(a, b);
}

// CHECK-LABEL: @test_vst1q_p64_x4(
// CHECK:   [[B:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly64x2x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX6]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst1q_p64_x4(poly64_t *a, poly64x2x4_t b) {
  vst1q_p64_x4(a, b);
}

// CHECK-LABEL: @test_vst1_f64_x4(
// CHECK:   [[B:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.float64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x double>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x double>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x double> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x double>
// CHECK:   call void @llvm.aarch64.neon.st1x4.v1f64.p0(<1 x double> [[TMP11]], <1 x double> [[TMP12]], <1 x double> [[TMP13]], <1 x double> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst1_f64_x4(float64_t *a, float64x1x4_t b) {
  vst1_f64_x4(a, b);
}

// CHECK-LABEL: @test_vst1_p64_x4(
// CHECK:   [[B:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw %struct.poly64x1x4_t, ptr [[B]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// CHECK:   [[VAL:%.*]] = getelementptr inbounds nuw %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds nuw %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL3:%.*]] = getelementptr inbounds nuw %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL5:%.*]] = getelementptr inbounds nuw %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL5]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX6]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %a)
// CHECK:   ret void
void test_vst1_p64_x4(poly64_t *a, poly64x1x4_t b) {
  vst1_p64_x4(a, b);
}

// CHECK-LABEL: @test_vceqd_s64(
// CHECK:   [[TMP0:%.*]] = icmp eq i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vceqd_s64(int64_t a, int64_t b) {
  return (uint64_t)vceqd_s64(a, b);
}

// CHECK-LABEL: @test_vceqd_u64(
// CHECK:   [[TMP0:%.*]] = icmp eq i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vceqd_u64(uint64_t a, uint64_t b) {
  return (int64_t)vceqd_u64(a, b);
}

// CHECK-LABEL: @test_vceqzd_s64(
// CHECK:   [[TMP0:%.*]] = icmp eq i64 %a, 0
// CHECK:   [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQZ_I]]
uint64_t test_vceqzd_s64(int64_t a) {
  return (uint64_t)vceqzd_s64(a);
}

// CHECK-LABEL: @test_vceqzd_u64(
// CHECK:   [[TMP0:%.*]] = icmp eq i64 %a, 0
// CHECK:   [[VCEQZD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQZD_I]]
int64_t test_vceqzd_u64(int64_t a) {
  return (int64_t)vceqzd_u64(a);
}

// CHECK-LABEL: @test_vcged_s64(
// CHECK:   [[TMP0:%.*]] = icmp sge i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vcged_s64(int64_t a, int64_t b) {
  return (uint64_t)vcged_s64(a, b);
}

// CHECK-LABEL: @test_vcged_u64(
// CHECK:   [[TMP0:%.*]] = icmp uge i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vcged_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vcged_u64(a, b);
}

// CHECK-LABEL: @test_vcgezd_s64(
// CHECK:   [[TMP0:%.*]] = icmp sge i64 %a, 0
// CHECK:   [[VCGEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCGEZ_I]]
uint64_t test_vcgezd_s64(int64_t a) {
  return (uint64_t)vcgezd_s64(a);
}

// CHECK-LABEL: @test_vcgtd_s64(
// CHECK:   [[TMP0:%.*]] = icmp sgt i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vcgtd_s64(int64_t a, int64_t b) {
  return (uint64_t)vcgtd_s64(a, b);
}

// CHECK-LABEL: @test_vcgtd_u64(
// CHECK:   [[TMP0:%.*]] = icmp ugt i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vcgtd_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vcgtd_u64(a, b);
}

// CHECK-LABEL: @test_vcgtzd_s64(
// CHECK:   [[TMP0:%.*]] = icmp sgt i64 %a, 0
// CHECK:   [[VCGTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCGTZ_I]]
uint64_t test_vcgtzd_s64(int64_t a) {
  return (uint64_t)vcgtzd_s64(a);
}

// CHECK-LABEL: @test_vcled_s64(
// CHECK:   [[TMP0:%.*]] = icmp sle i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vcled_s64(int64_t a, int64_t b) {
  return (uint64_t)vcled_s64(a, b);
}

// CHECK-LABEL: @test_vcled_u64(
// CHECK:   [[TMP0:%.*]] = icmp ule i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vcled_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vcled_u64(a, b);
}

// CHECK-LABEL: @test_vclezd_s64(
// CHECK:   [[TMP0:%.*]] = icmp sle i64 %a, 0
// CHECK:   [[VCLEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCLEZ_I]]
uint64_t test_vclezd_s64(int64_t a) {
  return (uint64_t)vclezd_s64(a);
}

// CHECK-LABEL: @test_vcltd_s64(
// CHECK:   [[TMP0:%.*]] = icmp slt i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vcltd_s64(int64_t a, int64_t b) {
  return (uint64_t)vcltd_s64(a, b);
}

// CHECK-LABEL: @test_vcltd_u64(
// CHECK:   [[TMP0:%.*]] = icmp ult i64 %a, %b
// CHECK:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQD_I]]
uint64_t test_vcltd_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vcltd_u64(a, b);
}

// CHECK-LABEL: @test_vcltzd_s64(
// CHECK:   [[TMP0:%.*]] = icmp slt i64 %a, 0
// CHECK:   [[VCLTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCLTZ_I]]
uint64_t test_vcltzd_s64(int64_t a) {
  return (uint64_t)vcltzd_s64(a);
}

// CHECK-LABEL: @test_vtstd_s64(
// CHECK:   [[TMP0:%.*]] = and i64 %a, %b
// CHECK:   [[TMP1:%.*]] = icmp ne i64 [[TMP0]], 0
// CHECK:   [[VTSTD_I:%.*]] = sext i1 [[TMP1]] to i64
// CHECK:   ret i64 [[VTSTD_I]]
uint64_t test_vtstd_s64(int64_t a, int64_t b) {
  return (uint64_t)vtstd_s64(a, b);
}

// CHECK-LABEL: @test_vtstd_u64(
// CHECK:   [[TMP0:%.*]] = and i64 %a, %b
// CHECK:   [[TMP1:%.*]] = icmp ne i64 [[TMP0]], 0
// CHECK:   [[VTSTD_I:%.*]] = sext i1 [[TMP1]] to i64
// CHECK:   ret i64 [[VTSTD_I]]
uint64_t test_vtstd_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vtstd_u64(a, b);
}

// CHECK-LABEL: @test_vabsd_s64(
// CHECK:   [[VABSD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.abs.i64(i64 %a)
// CHECK:   ret i64 [[VABSD_S64_I]]
int64_t test_vabsd_s64(int64_t a) {
  return (int64_t)vabsd_s64(a);
}

// CHECK-LABEL: @test_vqabsb_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[VQABSB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqabs.v8i8(<8 x i8> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQABSB_S8_I]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqabsb_s8(int8_t a) {
  return (int8_t)vqabsb_s8(a);
}

// CHECK-LABEL: @test_vqabsh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQABSH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqabs.v4i16(<4 x i16> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQABSH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqabsh_s16(int16_t a) {
  return (int16_t)vqabsh_s16(a);
}

// CHECK-LABEL: @test_vqabss_s32(
// CHECK:   [[VQABSS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqabs.i32(i32 %a)
// CHECK:   ret i32 [[VQABSS_S32_I]]
int32_t test_vqabss_s32(int32_t a) {
  return (int32_t)vqabss_s32(a);
}

// CHECK-LABEL: @test_vqabsd_s64(
// CHECK:   [[VQABSD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqabs.i64(i64 %a)
// CHECK:   ret i64 [[VQABSD_S64_I]]
int64_t test_vqabsd_s64(int64_t a) {
  return (int64_t)vqabsd_s64(a);
}

// CHECK-LABEL: @test_vnegd_s64(
// CHECK:   [[VNEGD_I:%.*]] = sub i64 0, %a
// CHECK:   ret i64 [[VNEGD_I]]
int64_t test_vnegd_s64(int64_t a) {
  return (int64_t)vnegd_s64(a);
}

// CHECK-LABEL: @test_vqnegb_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[VQNEGB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqneg.v8i8(<8 x i8> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQNEGB_S8_I]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqnegb_s8(int8_t a) {
  return (int8_t)vqnegb_s8(a);
}

// CHECK-LABEL: @test_vqnegh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQNEGH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqneg.v4i16(<4 x i16> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQNEGH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqnegh_s16(int16_t a) {
  return (int16_t)vqnegh_s16(a);
}

// CHECK-LABEL: @test_vqnegs_s32(
// CHECK:   [[VQNEGS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqneg.i32(i32 %a)
// CHECK:   ret i32 [[VQNEGS_S32_I]]
int32_t test_vqnegs_s32(int32_t a) {
  return (int32_t)vqnegs_s32(a);
}

// CHECK-LABEL: @test_vqnegd_s64(
// CHECK:   [[VQNEGD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqneg.i64(i64 %a)
// CHECK:   ret i64 [[VQNEGD_S64_I]]
int64_t test_vqnegd_s64(int64_t a) {
  return (int64_t)vqnegd_s64(a);
}

// CHECK-LABEL: @test_vuqaddb_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VUQADDB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.suqadd.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VUQADDB_S8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
int8_t test_vuqaddb_s8(int8_t a, uint8_t b) {
  return (int8_t)vuqaddb_s8(a, b);
}

// CHECK-LABEL: @test_vuqaddh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VUQADDH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VUQADDH_S16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
int16_t test_vuqaddh_s16(int16_t a, uint16_t b) {
  return (int16_t)vuqaddh_s16(a, b);
}

// CHECK-LABEL: @test_vuqadds_s32(
// CHECK:   [[VUQADDS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.suqadd.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VUQADDS_S32_I]]
int32_t test_vuqadds_s32(int32_t a, uint32_t b) {
  return (int32_t)vuqadds_s32(a, b);
}

// CHECK-LABEL: @test_vuqaddd_s64(
// CHECK:   [[VUQADDD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.suqadd.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VUQADDD_S64_I]]
int64_t test_vuqaddd_s64(int64_t a, uint64_t b) {
  return (int64_t)vuqaddd_s64(a, b);
}

// CHECK-LABEL: @test_vsqaddb_u8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// CHECK:   [[VSQADDB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.usqadd.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <8 x i8> [[VSQADDB_U8_I]], i64 0
// CHECK:   ret i8 [[TMP2]]
uint8_t test_vsqaddb_u8(uint8_t a, int8_t b) {
  return (uint8_t)vsqaddb_u8(a, b);
}

// CHECK-LABEL: @test_vsqaddh_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VSQADDH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.usqadd.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i16> [[VSQADDH_U16_I]], i64 0
// CHECK:   ret i16 [[TMP2]]
uint16_t test_vsqaddh_u16(uint16_t a, int16_t b) {
  return (uint16_t)vsqaddh_u16(a, b);
}

// CHECK-LABEL: @test_vsqadds_u32(
// CHECK:   [[VSQADDS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.usqadd.i32(i32 %a, i32 %b)
// CHECK:   ret i32 [[VSQADDS_U32_I]]
uint32_t test_vsqadds_u32(uint32_t a, int32_t b) {
  return (uint32_t)vsqadds_u32(a, b);
}

// CHECK-LABEL: @test_vsqaddd_u64(
// CHECK:   [[VSQADDD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.usqadd.i64(i64 %a, i64 %b)
// CHECK:   ret i64 [[VSQADDD_U64_I]]
uint64_t test_vsqaddd_u64(uint64_t a, int64_t b) {
  return (uint64_t)vsqaddd_u64(a, b);
}

// CHECK-LABEL: @test_vqdmlalh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %c, i64 0
// CHECK:   [[VQDMLXL_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[LANE0_I:%.*]] = extractelement <4 x i32> [[VQDMLXL_I]], i64 0
// CHECK:   [[VQDMLXL1_I:%.*]] = call i32 @llvm.aarch64.neon.sqadd.i32(i32 %a, i32 [[LANE0_I]])
// CHECK:   ret i32 [[VQDMLXL1_I]]
int32_t test_vqdmlalh_s16(int32_t a, int16_t b, int16_t c) {
  return (int32_t)vqdmlalh_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlals_s32(
// CHECK:   [[VQDMLXL_I:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %b, i32 %c)
// CHECK:   [[VQDMLXL1_I:%.*]] = call i64 @llvm.aarch64.neon.sqadd.i64(i64 %a, i64 [[VQDMLXL_I]])
// CHECK:   ret i64 [[VQDMLXL1_I]]
int64_t test_vqdmlals_s32(int64_t a, int32_t b, int32_t c) {
  return (int64_t)vqdmlals_s32(a, b, c);
}

// CHECK-LABEL: @test_vqdmlslh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %c, i64 0
// CHECK:   [[VQDMLXL_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[LANE0_I:%.*]] = extractelement <4 x i32> [[VQDMLXL_I]], i64 0
// CHECK:   [[VQDMLXL1_I:%.*]] = call i32 @llvm.aarch64.neon.sqsub.i32(i32 %a, i32 [[LANE0_I]])
// CHECK:   ret i32 [[VQDMLXL1_I]]
int32_t test_vqdmlslh_s16(int32_t a, int16_t b, int16_t c) {
  return (int32_t)vqdmlslh_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsls_s32(
// CHECK:   [[VQDMLXL_I:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %b, i32 %c)
// CHECK:   [[VQDMLXL1_I:%.*]] = call i64 @llvm.aarch64.neon.sqsub.i64(i64 %a, i64 [[VQDMLXL_I]])
// CHECK:   ret i64 [[VQDMLXL1_I]]
int64_t test_vqdmlsls_s32(int64_t a, int32_t b, int32_t c) {
  return (int64_t)vqdmlsls_s32(a, b, c);
}

// CHECK-LABEL: @test_vqdmullh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// CHECK:   [[VQDMULLH_S16_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// CHECK:   [[TMP2:%.*]] = extractelement <4 x i32> [[VQDMULLH_S16_I]], i64 0
// CHECK:   ret i32 [[TMP2]]
int32_t test_vqdmullh_s16(int16_t a, int16_t b) {
  return (int32_t)vqdmullh_s16(a, b);
}

// CHECK-LABEL: @test_vqdmulls_s32(
// CHECK:   [[VQDMULLS_S32_I:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %a, i32 %b)
// CHECK:   ret i64 [[VQDMULLS_S32_I]]
int64_t test_vqdmulls_s32(int32_t a, int32_t b) {
  return (int64_t)vqdmulls_s32(a, b);
}

// CHECK-LABEL: @test_vqmovunh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQMOVUNH_S16_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtun.v8i8(<8 x i16> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQMOVUNH_S16_I]], i64 0
// CHECK:   ret i8 [[TMP1]]
uint8_t test_vqmovunh_s16(int16_t a) {
  return (uint8_t)vqmovunh_s16(a);
}

// CHECK-LABEL: @test_vqmovuns_s32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQMOVUNS_S32_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQMOVUNS_S32_I]], i64 0
// CHECK:   ret i16 [[TMP1]]
uint16_t test_vqmovuns_s32(int32_t a) {
  return (uint16_t)vqmovuns_s32(a);
}

// CHECK-LABEL: @test_vqmovund_s64(
// CHECK:   [[VQMOVUND_S64_I:%.*]] = call i32 @llvm.aarch64.neon.scalar.sqxtun.i32.i64(i64 %a)
// CHECK:   ret i32 [[VQMOVUND_S64_I]]
uint32_t test_vqmovund_s64(int64_t a) {
  return (uint32_t)vqmovund_s64(a);
}

// CHECK-LABEL: @test_vqmovnh_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQMOVNH_S16_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtn.v8i8(<8 x i16> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQMOVNH_S16_I]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqmovnh_s16(int16_t a) {
  return (int8_t)vqmovnh_s16(a);
}

// CHECK-LABEL: @test_vqmovns_s32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQMOVNS_S32_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtn.v4i16(<4 x i32> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQMOVNS_S32_I]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqmovns_s32(int32_t a) {
  return (int16_t)vqmovns_s32(a);
}

// CHECK-LABEL: @test_vqmovnd_s64(
// CHECK:   [[VQMOVND_S64_I:%.*]] = call i32 @llvm.aarch64.neon.scalar.sqxtn.i32.i64(i64 %a)
// CHECK:   ret i32 [[VQMOVND_S64_I]]
int32_t test_vqmovnd_s64(int64_t a) {
  return (int32_t)vqmovnd_s64(a);
}

// CHECK-LABEL: @test_vqmovnh_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQMOVNH_U16_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqxtn.v8i8(<8 x i16> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQMOVNH_U16_I]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqmovnh_u16(int16_t a) {
  return (int8_t)vqmovnh_u16(a);
}

// CHECK-LABEL: @test_vqmovns_u32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQMOVNS_U32_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqxtn.v4i16(<4 x i32> [[TMP0]])
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQMOVNS_U32_I]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqmovns_u32(int32_t a) {
  return (int16_t)vqmovns_u32(a);
}

// CHECK-LABEL: @test_vqmovnd_u64(
// CHECK:   [[VQMOVND_U64_I:%.*]] = call i32 @llvm.aarch64.neon.scalar.uqxtn.i32.i64(i64 %a)
// CHECK:   ret i32 [[VQMOVND_U64_I]]
int32_t test_vqmovnd_u64(int64_t a) {
  return (int32_t)vqmovnd_u64(a);
}

// CHECK-LABEL: @test_vceqs_f32(
// CHECK:   [[TMP0:%.*]] = fcmp oeq float %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCMPD_I]]
uint32_t test_vceqs_f32(float32_t a, float32_t b) {
  return (uint32_t)vceqs_f32(a, b);
}

// CHECK-LABEL: @test_vceqd_f64(
// CHECK:   [[TMP0:%.*]] = fcmp oeq double %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCMPD_I]]
uint64_t test_vceqd_f64(float64_t a, float64_t b) {
  return (uint64_t)vceqd_f64(a, b);
}

// CHECK-LABEL: @test_vceqzs_f32(
// CHECK:   [[TMP0:%.*]] = fcmp oeq float %a, 0.000000e+00
// CHECK:   [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCEQZ_I]]
uint32_t test_vceqzs_f32(float32_t a) {
  return (uint32_t)vceqzs_f32(a);
}

// CHECK-LABEL: @test_vceqzd_f64(
// CHECK:   [[TMP0:%.*]] = fcmp oeq double %a, 0.000000e+00
// CHECK:   [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCEQZ_I]]
uint64_t test_vceqzd_f64(float64_t a) {
  return (uint64_t)vceqzd_f64(a);
}

// CHECK-LABEL: @test_vcges_f32(
// CHECK:   [[TMP0:%.*]] = fcmp oge float %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCMPD_I]]
uint32_t test_vcges_f32(float32_t a, float32_t b) {
  return (uint32_t)vcges_f32(a, b);
}

// CHECK-LABEL: @test_vcged_f64(
// CHECK:   [[TMP0:%.*]] = fcmp oge double %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCMPD_I]]
uint64_t test_vcged_f64(float64_t a, float64_t b) {
  return (uint64_t)vcged_f64(a, b);
}

// CHECK-LABEL: @test_vcgezs_f32(
// CHECK:   [[TMP0:%.*]] = fcmp oge float %a, 0.000000e+00
// CHECK:   [[VCGEZ_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCGEZ_I]]
uint32_t test_vcgezs_f32(float32_t a) {
  return (uint32_t)vcgezs_f32(a);
}

// CHECK-LABEL: @test_vcgezd_f64(
// CHECK:   [[TMP0:%.*]] = fcmp oge double %a, 0.000000e+00
// CHECK:   [[VCGEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCGEZ_I]]
uint64_t test_vcgezd_f64(float64_t a) {
  return (uint64_t)vcgezd_f64(a);
}

// CHECK-LABEL: @test_vcgts_f32(
// CHECK:   [[TMP0:%.*]] = fcmp ogt float %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCMPD_I]]
uint32_t test_vcgts_f32(float32_t a, float32_t b) {
  return (uint32_t)vcgts_f32(a, b);
}

// CHECK-LABEL: @test_vcgtd_f64(
// CHECK:   [[TMP0:%.*]] = fcmp ogt double %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCMPD_I]]
uint64_t test_vcgtd_f64(float64_t a, float64_t b) {
  return (uint64_t)vcgtd_f64(a, b);
}

// CHECK-LABEL: @test_vcgtzs_f32(
// CHECK:   [[TMP0:%.*]] = fcmp ogt float %a, 0.000000e+00
// CHECK:   [[VCGTZ_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCGTZ_I]]
uint32_t test_vcgtzs_f32(float32_t a) {
  return (uint32_t)vcgtzs_f32(a);
}

// CHECK-LABEL: @test_vcgtzd_f64(
// CHECK:   [[TMP0:%.*]] = fcmp ogt double %a, 0.000000e+00
// CHECK:   [[VCGTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCGTZ_I]]
uint64_t test_vcgtzd_f64(float64_t a) {
  return (uint64_t)vcgtzd_f64(a);
}

// CHECK-LABEL: @test_vcles_f32(
// CHECK:   [[TMP0:%.*]] = fcmp ole float %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCMPD_I]]
uint32_t test_vcles_f32(float32_t a, float32_t b) {
  return (uint32_t)vcles_f32(a, b);
}

// CHECK-LABEL: @test_vcled_f64(
// CHECK:   [[TMP0:%.*]] = fcmp ole double %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCMPD_I]]
uint64_t test_vcled_f64(float64_t a, float64_t b) {
  return (uint64_t)vcled_f64(a, b);
}

// CHECK-LABEL: @test_vclezs_f32(
// CHECK:   [[TMP0:%.*]] = fcmp ole float %a, 0.000000e+00
// CHECK:   [[VCLEZ_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCLEZ_I]]
uint32_t test_vclezs_f32(float32_t a) {
  return (uint32_t)vclezs_f32(a);
}

// CHECK-LABEL: @test_vclezd_f64(
// CHECK:   [[TMP0:%.*]] = fcmp ole double %a, 0.000000e+00
// CHECK:   [[VCLEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCLEZ_I]]
uint64_t test_vclezd_f64(float64_t a) {
  return (uint64_t)vclezd_f64(a);
}

// CHECK-LABEL: @test_vclts_f32(
// CHECK:   [[TMP0:%.*]] = fcmp olt float %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCMPD_I]]
uint32_t test_vclts_f32(float32_t a, float32_t b) {
  return (uint32_t)vclts_f32(a, b);
}

// CHECK-LABEL: @test_vcltd_f64(
// CHECK:   [[TMP0:%.*]] = fcmp olt double %a, %b
// CHECK:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCMPD_I]]
uint64_t test_vcltd_f64(float64_t a, float64_t b) {
  return (uint64_t)vcltd_f64(a, b);
}

// CHECK-LABEL: @test_vcltzs_f32(
// CHECK:   [[TMP0:%.*]] = fcmp olt float %a, 0.000000e+00
// CHECK:   [[VCLTZ_I:%.*]] = sext i1 [[TMP0]] to i32
// CHECK:   ret i32 [[VCLTZ_I]]
uint32_t test_vcltzs_f32(float32_t a) {
  return (uint32_t)vcltzs_f32(a);
}

// CHECK-LABEL: @test_vcltzd_f64(
// CHECK:   [[TMP0:%.*]] = fcmp olt double %a, 0.000000e+00
// CHECK:   [[VCLTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// CHECK:   ret i64 [[VCLTZ_I]]
uint64_t test_vcltzd_f64(float64_t a) {
  return (uint64_t)vcltzd_f64(a);
}

// CHECK-LABEL: @test_vcages_f32(
// CHECK:   [[VCAGES_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facge.i32.f32(float %a, float %b)
// CHECK:   ret i32 [[VCAGES_F32_I]]
uint32_t test_vcages_f32(float32_t a, float32_t b) {
  return (uint32_t)vcages_f32(a, b);
}

// CHECK-LABEL: @test_vcaged_f64(
// CHECK:   [[VCAGED_F64_I:%.*]] = call i64 @llvm.aarch64.neon.facge.i64.f64(double %a, double %b)
// CHECK:   ret i64 [[VCAGED_F64_I]]
uint64_t test_vcaged_f64(float64_t a, float64_t b) {
  return (uint64_t)vcaged_f64(a, b);
}

// CHECK-LABEL: @test_vcagts_f32(
// CHECK:   [[VCAGTS_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facgt.i32.f32(float %a, float %b)
// CHECK:   ret i32 [[VCAGTS_F32_I]]
uint32_t test_vcagts_f32(float32_t a, float32_t b) {
  return (uint32_t)vcagts_f32(a, b);
}

// CHECK-LABEL: @test_vcagtd_f64(
// CHECK:   [[VCAGTD_F64_I:%.*]] = call i64 @llvm.aarch64.neon.facgt.i64.f64(double %a, double %b)
// CHECK:   ret i64 [[VCAGTD_F64_I]]
uint64_t test_vcagtd_f64(float64_t a, float64_t b) {
  return (uint64_t)vcagtd_f64(a, b);
}

// CHECK-LABEL: @test_vcales_f32(
// CHECK:   [[VCALES_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facge.i32.f32(float %b, float %a)
// CHECK:   ret i32 [[VCALES_F32_I]]
uint32_t test_vcales_f32(float32_t a, float32_t b) {
  return (uint32_t)vcales_f32(a, b);
}

// CHECK-LABEL: @test_vcaled_f64(
// CHECK:   [[VCALED_F64_I:%.*]] = call i64 @llvm.aarch64.neon.facge.i64.f64(double %b, double %a)
// CHECK:   ret i64 [[VCALED_F64_I]]
uint64_t test_vcaled_f64(float64_t a, float64_t b) {
  return (uint64_t)vcaled_f64(a, b);
}

// CHECK-LABEL: @test_vcalts_f32(
// CHECK:   [[VCALTS_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facgt.i32.f32(float %b, float %a)
// CHECK:   ret i32 [[VCALTS_F32_I]]
uint32_t test_vcalts_f32(float32_t a, float32_t b) {
  return (uint32_t)vcalts_f32(a, b);
}

// CHECK-LABEL: @test_vcaltd_f64(
// CHECK:   [[VCALTD_F64_I:%.*]] = call i64 @llvm.aarch64.neon.facgt.i64.f64(double %b, double %a)
// CHECK:   ret i64 [[VCALTD_F64_I]]
uint64_t test_vcaltd_f64(float64_t a, float64_t b) {
  return (uint64_t)vcaltd_f64(a, b);
}

// CHECK-LABEL: @test_vshrd_n_s64(
// CHECK:   [[SHRD_N:%.*]] = ashr i64 %a, 1
// CHECK:   ret i64 [[SHRD_N]]
int64_t test_vshrd_n_s64(int64_t a) {
  return (int64_t)vshrd_n_s64(a, 1);
}

// CHECK-LABEL: @test_vshr_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSHR_N:%.*]] = ashr <1 x i64> [[TMP1]], splat (i64 1)
// CHECK:   ret <1 x i64> [[VSHR_N]]
int64x1_t test_vshr_n_s64(int64x1_t a) {
  return vshr_n_s64(a, 1);
}

// CHECK-LABEL: @test_vshrd_n_u64(
// CHECK:   ret i64 0
uint64_t test_vshrd_n_u64(uint64_t a) {
  return (uint64_t)vshrd_n_u64(a, 64);
}

// CHECK-LABEL: @test_vshrd_n_u64_2(
// CHECK:   ret i64 0
uint64_t test_vshrd_n_u64_2() {
  uint64_t a = UINT64_C(0xf000000000000000);
  return vshrd_n_u64(a, 64);
}

// CHECK-LABEL: @test_vshr_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSHR_N:%.*]] = lshr <1 x i64> [[TMP1]], splat (i64 1)
// CHECK:   ret <1 x i64> [[VSHR_N]]
uint64x1_t test_vshr_n_u64(uint64x1_t a) {
  return vshr_n_u64(a, 1);
}

// CHECK-LABEL: @test_vrshrd_n_s64(
// CHECK:   [[VRSHR_N:%.*]] = call i64 @llvm.aarch64.neon.srshl.i64(i64 %a, i64 -63)
// CHECK:   ret i64 [[VRSHR_N]]
int64_t test_vrshrd_n_s64(int64_t a) {
  return (int64_t)vrshrd_n_s64(a, 63);
}

// CHECK-LABEL: @test_vrshr_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.srshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> splat (i64 -1))
// CHECK:   ret <1 x i64> [[VRSHR_N1]]
int64x1_t test_vrshr_n_s64(int64x1_t a) {
  return vrshr_n_s64(a, 1);
}

// CHECK-LABEL: @test_vrshrd_n_u64(
// CHECK:   [[VRSHR_N:%.*]] = call i64 @llvm.aarch64.neon.urshl.i64(i64 %a, i64 -63)
// CHECK:   ret i64 [[VRSHR_N]]
uint64_t test_vrshrd_n_u64(uint64_t a) {
  return (uint64_t)vrshrd_n_u64(a, 63);
}

// CHECK-LABEL: @test_vrshr_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.urshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> splat (i64 -1))
// CHECK:   ret <1 x i64> [[VRSHR_N1]]
uint64x1_t test_vrshr_n_u64(uint64x1_t a) {
  return vrshr_n_u64(a, 1);
}

// CHECK-LABEL: @test_vsrad_n_s64(
// CHECK:   [[SHRD_N:%.*]] = ashr i64 %b, 63
// CHECK:   [[TMP0:%.*]] = add i64 %a, [[SHRD_N]]
// CHECK:   ret i64 [[TMP0]]
int64_t test_vsrad_n_s64(int64_t a, int64_t b) {
  return (int64_t)vsrad_n_s64(a, b, 63);
}

// CHECK-LABEL: @test_vsra_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VSRA_N:%.*]] = ashr <1 x i64> [[TMP3]], splat (i64 1)
// CHECK:   [[TMP4:%.*]] = add <1 x i64> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <1 x i64> [[TMP4]]
int64x1_t test_vsra_n_s64(int64x1_t a, int64x1_t b) {
  return vsra_n_s64(a, b, 1);
}

// CHECK-LABEL: @test_vsrad_n_u64(
// CHECK:   [[SHRD_N:%.*]] = lshr i64 %b, 63
// CHECK:   [[TMP0:%.*]] = add i64 %a, [[SHRD_N]]
// CHECK:   ret i64 [[TMP0]]
uint64_t test_vsrad_n_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vsrad_n_u64(a, b, 63);
}

// CHECK-LABEL: @test_vsrad_n_u64_2(
// CHECK:   ret i64 %a
uint64_t test_vsrad_n_u64_2(uint64_t a, uint64_t b) {
  return (uint64_t)vsrad_n_u64(a, b, 64);
}

// CHECK-LABEL: @test_vsra_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VSRA_N:%.*]] = lshr <1 x i64> [[TMP3]], splat (i64 1)
// CHECK:   [[TMP4:%.*]] = add <1 x i64> [[TMP2]], [[VSRA_N]]
// CHECK:   ret <1 x i64> [[TMP4]]
uint64x1_t test_vsra_n_u64(uint64x1_t a, uint64x1_t b) {
  return vsra_n_u64(a, b, 1);
}

// CHECK-LABEL: @test_vrsrad_n_s64(
// CHECK:   [[TMP0:%.*]] = call i64 @llvm.aarch64.neon.srshl.i64(i64 %b, i64 -63)
// CHECK:   [[TMP1:%.*]] = add i64 %a, [[TMP0]]
// CHECK:   ret i64 [[TMP1]]
int64_t test_vrsrad_n_s64(int64_t a, int64_t b) {
  return (int64_t)vrsrad_n_s64(a, b, 63);
}

// CHECK-LABEL: @test_vrsra_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.srshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> splat (i64 -1))
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = add <1 x i64> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <1 x i64> [[TMP3]]
int64x1_t test_vrsra_n_s64(int64x1_t a, int64x1_t b) {
  return vrsra_n_s64(a, b, 1);
}

// CHECK-LABEL: @test_vrsrad_n_u64(
// CHECK:   [[TMP0:%.*]] = call i64 @llvm.aarch64.neon.urshl.i64(i64 %b, i64 -63)
// CHECK:   [[TMP1:%.*]] = add i64 %a, [[TMP0]]
// CHECK:   ret i64 [[TMP1]]
uint64_t test_vrsrad_n_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vrsrad_n_u64(a, b, 63);
}

// CHECK-LABEL: @test_vrsra_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.urshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> splat (i64 -1))
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = add <1 x i64> [[TMP2]], [[VRSHR_N1]]
// CHECK:   ret <1 x i64> [[TMP3]]
uint64x1_t test_vrsra_n_u64(uint64x1_t a, uint64x1_t b) {
  return vrsra_n_u64(a, b, 1);
}

// CHECK-LABEL: @test_vshld_n_s64(
// CHECK:   [[SHLD_N:%.*]] = shl i64 %a, 1
// CHECK:   ret i64 [[SHLD_N]]
int64_t test_vshld_n_s64(int64_t a) {
  return (int64_t)vshld_n_s64(a, 1);
}

// CHECK-LABEL: @test_vshl_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSHL_N:%.*]] = shl <1 x i64> [[TMP1]], splat (i64 1)
// CHECK:   ret <1 x i64> [[VSHL_N]]
int64x1_t test_vshl_n_s64(int64x1_t a) {
  return vshl_n_s64(a, 1);
}

// CHECK-LABEL: @test_vshld_n_u64(
// CHECK:   [[SHLD_N:%.*]] = shl i64 %a, 63
// CHECK:   ret i64 [[SHLD_N]]
uint64_t test_vshld_n_u64(uint64_t a) {
  return (uint64_t)vshld_n_u64(a, 63);
}

// CHECK-LABEL: @test_vshl_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSHL_N:%.*]] = shl <1 x i64> [[TMP1]], splat (i64 1)
// CHECK:   ret <1 x i64> [[VSHL_N]]
uint64x1_t test_vshl_n_u64(uint64x1_t a) {
  return vshl_n_u64(a, 1);
}

// CHECK-LABEL: @test_vqshlb_n_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[VQSHLB_N_S8:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> <i8 7, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHLB_N_S8]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqshlb_n_s8(int8_t a) {
  return (int8_t)vqshlb_n_s8(a, 7);
}

// CHECK-LABEL: @test_vqshlh_n_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQSHLH_N_S16:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> <i16 15, i16 poison, i16 poison, i16 poison>)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHLH_N_S16]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqshlh_n_s16(int16_t a) {
  return (int16_t)vqshlh_n_s16(a, 15);
}

// CHECK-LABEL: @test_vqshls_n_s32(
// CHECK:   [[VQSHLS_N_S32:%.*]] = call i32 @llvm.aarch64.neon.sqshl.i32(i32 %a, i32 31)
// CHECK:   ret i32 [[VQSHLS_N_S32]]
int32_t test_vqshls_n_s32(int32_t a) {
  return (int32_t)vqshls_n_s32(a, 31);
}

// CHECK-LABEL: @test_vqshld_n_s64(
// CHECK:   [[VQSHL_N:%.*]] = call i64 @llvm.aarch64.neon.sqshl.i64(i64 %a, i64 63)
// CHECK:   ret i64 [[VQSHL_N]]
int64_t test_vqshld_n_s64(int64_t a) {
  return (int64_t)vqshld_n_s64(a, 63);
}

// CHECK-LABEL: @test_vqshl_n_s8(
// CHECK:   [[VQSHL_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> %a, <8 x i8> zeroinitializer)
// CHECK:   ret <8 x i8> [[VQSHL_N]]
int8x8_t test_vqshl_n_s8(int8x8_t a) {
  return vqshl_n_s8(a, 0);
}

// CHECK-LABEL: @test_vqshlq_n_s8(
// CHECK:   [[VQSHL_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqshl.v16i8(<16 x i8> %a, <16 x i8> zeroinitializer)
// CHECK:   ret <16 x i8> [[VQSHL_N]]
int8x16_t test_vqshlq_n_s8(int8x16_t a) {
  return vqshlq_n_s8(a, 0);
}

// CHECK-LABEL: @test_vqshl_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VQSHL_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> [[VQSHL_N]], <4 x i16> zeroinitializer)
// CHECK:   ret <4 x i16> [[VQSHL_N1]]
int16x4_t test_vqshl_n_s16(int16x4_t a) {
  return vqshl_n_s16(a, 0);
}

// CHECK-LABEL: @test_vqshlq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHL_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqshl.v8i16(<8 x i16> [[VQSHL_N]], <8 x i16> zeroinitializer)
// CHECK:   ret <8 x i16> [[VQSHL_N1]]
int16x8_t test_vqshlq_n_s16(int16x8_t a) {
  return vqshlq_n_s16(a, 0);
}

// CHECK-LABEL: @test_vqshl_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VQSHL_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshl.v2i32(<2 x i32> [[VQSHL_N]], <2 x i32> zeroinitializer)
// CHECK:   ret <2 x i32> [[VQSHL_N1]]
int32x2_t test_vqshl_n_s32(int32x2_t a) {
  return vqshl_n_s32(a, 0);
}

// CHECK-LABEL: @test_vqshlq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHL_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqshl.v4i32(<4 x i32> [[VQSHL_N]], <4 x i32> zeroinitializer)
// CHECK:   ret <4 x i32> [[VQSHL_N1]]
int32x4_t test_vqshlq_n_s32(int32x4_t a) {
  return vqshlq_n_s32(a, 0);
}

// CHECK-LABEL: @test_vqshlq_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHL_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqshl.v2i64(<2 x i64> [[VQSHL_N]], <2 x i64> zeroinitializer)
// CHECK:   ret <2 x i64> [[VQSHL_N1]]
int64x2_t test_vqshlq_n_s64(int64x2_t a) {
  return vqshlq_n_s64(a, 0);
}

// CHECK-LABEL: @test_vqshl_n_u8(
// CHECK:   [[VQSHL_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> %a, <8 x i8> zeroinitializer)
// CHECK:   ret <8 x i8> [[VQSHL_N]]
uint8x8_t test_vqshl_n_u8(uint8x8_t a) {
  return vqshl_n_u8(a, 0);
}

// CHECK-LABEL: @test_vqshlq_n_u8(
// CHECK:   [[VQSHL_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqshl.v16i8(<16 x i8> %a, <16 x i8> zeroinitializer)
// CHECK:   ret <16 x i8> [[VQSHL_N]]
uint8x16_t test_vqshlq_n_u8(uint8x16_t a) {
  return vqshlq_n_u8(a, 0);
}

// CHECK-LABEL: @test_vqshl_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:   [[VQSHL_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> [[VQSHL_N]], <4 x i16> zeroinitializer)
// CHECK:   ret <4 x i16> [[VQSHL_N1]]
uint16x4_t test_vqshl_n_u16(uint16x4_t a) {
  return vqshl_n_u16(a, 0);
}

// CHECK-LABEL: @test_vqshlq_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:   [[VQSHL_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqshl.v8i16(<8 x i16> [[VQSHL_N]], <8 x i16> zeroinitializer)
// CHECK:   ret <8 x i16> [[VQSHL_N1]]
uint16x8_t test_vqshlq_n_u16(uint16x8_t a) {
  return vqshlq_n_u16(a, 0);
}

// CHECK-LABEL: @test_vqshl_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// CHECK:   [[VQSHL_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshl.v2i32(<2 x i32> [[VQSHL_N]], <2 x i32> zeroinitializer)
// CHECK:   ret <2 x i32> [[VQSHL_N1]]
uint32x2_t test_vqshl_n_u32(uint32x2_t a) {
  return vqshl_n_u32(a, 0);
}

// CHECK-LABEL: @test_vqshlq_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// CHECK:   [[VQSHL_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqshl.v4i32(<4 x i32> [[VQSHL_N]], <4 x i32> zeroinitializer)
// CHECK:   ret <4 x i32> [[VQSHL_N1]]
uint32x4_t test_vqshlq_n_u32(uint32x4_t a) {
  return vqshlq_n_u32(a, 0);
}

// CHECK-LABEL: @test_vqshlq_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VQSHL_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqshl.v2i64(<2 x i64> [[VQSHL_N]], <2 x i64> zeroinitializer)
// CHECK:   ret <2 x i64> [[VQSHL_N1]]
uint64x2_t test_vqshlq_n_u64(uint64x2_t a) {
  return vqshlq_n_u64(a, 0);
}

// CHECK-LABEL: @test_vqshl_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VQSHL_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqshl.v1i64(<1 x i64> [[VQSHL_N]], <1 x i64> splat (i64 1))
// CHECK:   ret <1 x i64> [[VQSHL_N1]]
int64x1_t test_vqshl_n_s64(int64x1_t a) {
  return vqshl_n_s64(a, 1);
}

// CHECK-LABEL: @test_vqshlb_n_u8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[VQSHLB_N_U8:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> <i8 7, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHLB_N_U8]], i64 0
// CHECK:   ret i8 [[TMP1]]
uint8_t test_vqshlb_n_u8(uint8_t a) {
  return (uint8_t)vqshlb_n_u8(a, 7);
}

// CHECK-LABEL: @test_vqshlh_n_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQSHLH_N_U16:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> <i16 15, i16 poison, i16 poison, i16 poison>)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHLH_N_U16]], i64 0
// CHECK:   ret i16 [[TMP1]]
uint16_t test_vqshlh_n_u16(uint16_t a) {
  return (uint16_t)vqshlh_n_u16(a, 15);
}

// CHECK-LABEL: @test_vqshls_n_u32(
// CHECK:   [[VQSHLS_N_U32:%.*]] = call i32 @llvm.aarch64.neon.uqshl.i32(i32 %a, i32 31)
// CHECK:   ret i32 [[VQSHLS_N_U32]]
uint32_t test_vqshls_n_u32(uint32_t a) {
  return (uint32_t)vqshls_n_u32(a, 31);
}

// CHECK-LABEL: @test_vqshld_n_u64(
// CHECK:   [[VQSHL_N:%.*]] = call i64 @llvm.aarch64.neon.uqshl.i64(i64 %a, i64 63)
// CHECK:   ret i64 [[VQSHL_N]]
uint64_t test_vqshld_n_u64(uint64_t a) {
  return (uint64_t)vqshld_n_u64(a, 63);
}

// CHECK-LABEL: @test_vqshl_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VQSHL_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqshl.v1i64(<1 x i64> [[VQSHL_N]], <1 x i64> splat (i64 1))
// CHECK:   ret <1 x i64> [[VQSHL_N1]]
uint64x1_t test_vqshl_n_u64(uint64x1_t a) {
  return vqshl_n_u64(a, 1);
}

// CHECK-LABEL: @test_vqshlub_n_s8(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// CHECK:   [[VQSHLUB_N_S8:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshlu.v8i8(<8 x i8> [[TMP0]], <8 x i8> <i8 7, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHLUB_N_S8]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqshlub_n_s8(int8_t a) {
  return (int8_t)vqshlub_n_s8(a, 7);
}

// CHECK-LABEL: @test_vqshluh_n_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQSHLUH_N_S16:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshlu.v4i16(<4 x i16> [[TMP0]], <4 x i16> <i16 15, i16 poison, i16 poison, i16 poison>)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHLUH_N_S16]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqshluh_n_s16(int16_t a) {
  return (int16_t)vqshluh_n_s16(a, 15);
}

// CHECK-LABEL: @test_vqshlus_n_s32(
// CHECK:   [[VQSHLUS_N_S32:%.*]] = call i32 @llvm.aarch64.neon.sqshlu.i32(i32 %a, i32 31)
// CHECK:   ret i32 [[VQSHLUS_N_S32]]
int32_t test_vqshlus_n_s32(int32_t a) {
  return (int32_t)vqshlus_n_s32(a, 31);
}

// CHECK-LABEL: @test_vqshlud_n_s64(
// CHECK:   [[VQSHLU_N:%.*]] = call i64 @llvm.aarch64.neon.sqshlu.i64(i64 %a, i64 63)
// CHECK:   ret i64 [[VQSHLU_N]]
int64_t test_vqshlud_n_s64(int64_t a) {
  return (int64_t)vqshlud_n_s64(a, 63);
}

// CHECK-LABEL: @test_vqshlu_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VQSHLU_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VQSHLU_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqshlu.v1i64(<1 x i64> [[VQSHLU_N]], <1 x i64> splat (i64 1))
// CHECK:   ret <1 x i64> [[VQSHLU_N1]]
uint64x1_t test_vqshlu_n_s64(int64x1_t a) {
  return vqshlu_n_s64(a, 1);
}

// CHECK-LABEL: @test_vsrid_n_s64(
// CHECK:   [[VSRID_N_S64:%.*]] = bitcast i64 %a to <1 x i64>
// CHECK:   [[VSRID_N_S641:%.*]] = bitcast i64 %b to <1 x i64>
// CHECK:   [[VSRID_N_S642:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRID_N_S64]], <1 x i64> [[VSRID_N_S641]], i32 63)
// CHECK:   [[VSRID_N_S643:%.*]] = bitcast <1 x i64> [[VSRID_N_S642]] to i64
// CHECK:   ret i64 [[VSRID_N_S643]]
int64_t test_vsrid_n_s64(int64_t a, int64_t b) {
  return (int64_t)vsrid_n_s64(a, b, 63);
}

// CHECK-LABEL: @test_vsri_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VSRI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRI_N]], <1 x i64> [[VSRI_N1]], i32 1)
// CHECK:   ret <1 x i64> [[VSRI_N2]]
int64x1_t test_vsri_n_s64(int64x1_t a, int64x1_t b) {
  return vsri_n_s64(a, b, 1);
}

// CHECK-LABEL: @test_vsrid_n_u64(
// CHECK:   [[VSRID_N_U64:%.*]] = bitcast i64 %a to <1 x i64>
// CHECK:   [[VSRID_N_U641:%.*]] = bitcast i64 %b to <1 x i64>
// CHECK:   [[VSRID_N_U642:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRID_N_U64]], <1 x i64> [[VSRID_N_U641]], i32 63)
// CHECK:   [[VSRID_N_U643:%.*]] = bitcast <1 x i64> [[VSRID_N_U642]] to i64
// CHECK:   ret i64 [[VSRID_N_U643]]
uint64_t test_vsrid_n_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vsrid_n_u64(a, b, 63);
}

// CHECK-LABEL: @test_vsri_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VSRI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRI_N]], <1 x i64> [[VSRI_N1]], i32 1)
// CHECK:   ret <1 x i64> [[VSRI_N2]]
uint64x1_t test_vsri_n_u64(uint64x1_t a, uint64x1_t b) {
  return vsri_n_u64(a, b, 1);
}

// CHECK-LABEL: @test_vslid_n_s64(
// CHECK:   [[VSLID_N_S64:%.*]] = bitcast i64 %a to <1 x i64>
// CHECK:   [[VSLID_N_S641:%.*]] = bitcast i64 %b to <1 x i64>
// CHECK:   [[VSLID_N_S642:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLID_N_S64]], <1 x i64> [[VSLID_N_S641]], i32 63)
// CHECK:   [[VSLID_N_S643:%.*]] = bitcast <1 x i64> [[VSLID_N_S642]] to i64
// CHECK:   ret i64 [[VSLID_N_S643]]
int64_t test_vslid_n_s64(int64_t a, int64_t b) {
  return (int64_t)vslid_n_s64(a, b, 63);
}

// CHECK-LABEL: @test_vsli_n_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VSLI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLI_N]], <1 x i64> [[VSLI_N1]], i32 1)
// CHECK:   ret <1 x i64> [[VSLI_N2]]
int64x1_t test_vsli_n_s64(int64x1_t a, int64x1_t b) {
  return vsli_n_s64(a, b, 1);
}

// CHECK-LABEL: @test_vslid_n_u64(
// CHECK:   [[VSLID_N_U64:%.*]] = bitcast i64 %a to <1 x i64>
// CHECK:   [[VSLID_N_U641:%.*]] = bitcast i64 %b to <1 x i64>
// CHECK:   [[VSLID_N_U642:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLID_N_U64]], <1 x i64> [[VSLID_N_U641]], i32 63)
// CHECK:   [[VSLID_N_U643:%.*]] = bitcast <1 x i64> [[VSLID_N_U642]] to i64
// CHECK:   ret i64 [[VSLID_N_U643]]
uint64_t test_vslid_n_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vslid_n_u64(a, b, 63);
}

// CHECK-LABEL: @test_vsli_n_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VSLI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLI_N]], <1 x i64> [[VSLI_N1]], i32 1)
// CHECK:   ret <1 x i64> [[VSLI_N2]]
uint64x1_t test_vsli_n_u64(uint64x1_t a, uint64x1_t b) {
  return vsli_n_u64(a, b, 1);
}

// CHECK-LABEL: @test_vqshrnh_n_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQSHRNH_N_S16:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16> [[TMP0]], i32 8)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHRNH_N_S16]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqshrnh_n_s16(int16_t a) {
  return (int8_t)vqshrnh_n_s16(a, 8);
}

// CHECK-LABEL: @test_vqshrns_n_s32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQSHRNS_N_S32:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> [[TMP0]], i32 16)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHRNS_N_S32]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqshrns_n_s32(int32_t a) {
  return (int16_t)vqshrns_n_s32(a, 16);
}

// CHECK-LABEL: @test_vqshrnd_n_s64(
// CHECK:   [[VQSHRND_N_S64:%.*]] = call i32 @llvm.aarch64.neon.sqshrn.i32(i64 %a, i32 32)
// CHECK:   ret i32 [[VQSHRND_N_S64]]
int32_t test_vqshrnd_n_s64(int64_t a) {
  return (int32_t)vqshrnd_n_s64(a, 32);
}

// CHECK-LABEL: @test_vqshrnh_n_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQSHRNH_N_U16:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> [[TMP0]], i32 8)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHRNH_N_U16]], i64 0
// CHECK:   ret i8 [[TMP1]]
uint8_t test_vqshrnh_n_u16(uint16_t a) {
  return (uint8_t)vqshrnh_n_u16(a, 8);
}

// CHECK-LABEL: @test_vqshrns_n_u32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQSHRNS_N_U32:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> [[TMP0]], i32 16)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHRNS_N_U32]], i64 0
// CHECK:   ret i16 [[TMP1]]
uint16_t test_vqshrns_n_u32(uint32_t a) {
  return (uint16_t)vqshrns_n_u32(a, 16);
}

// CHECK-LABEL: @test_vqshrnd_n_u64(
// CHECK:   [[VQSHRND_N_U64:%.*]] = call i32 @llvm.aarch64.neon.uqshrn.i32(i64 %a, i32 32)
// CHECK:   ret i32 [[VQSHRND_N_U64]]
uint32_t test_vqshrnd_n_u64(uint64_t a) {
  return (uint32_t)vqshrnd_n_u64(a, 32);
}

// CHECK-LABEL: @test_vqrshrnh_n_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQRSHRNH_N_S16:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16> [[TMP0]], i32 8)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQRSHRNH_N_S16]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqrshrnh_n_s16(int16_t a) {
  return (int8_t)vqrshrnh_n_s16(a, 8);
}

// CHECK-LABEL: @test_vqrshrns_n_s32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQRSHRNS_N_S32:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> [[TMP0]], i32 16)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQRSHRNS_N_S32]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqrshrns_n_s32(int32_t a) {
  return (int16_t)vqrshrns_n_s32(a, 16);
}

// CHECK-LABEL: @test_vqrshrnd_n_s64(
// CHECK:   [[VQRSHRND_N_S64:%.*]] = call i32 @llvm.aarch64.neon.sqrshrn.i32(i64 %a, i32 32)
// CHECK:   ret i32 [[VQRSHRND_N_S64]]
int32_t test_vqrshrnd_n_s64(int64_t a) {
  return (int32_t)vqrshrnd_n_s64(a, 32);
}

// CHECK-LABEL: @test_vqrshrnh_n_u16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQRSHRNH_N_U16:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16> [[TMP0]], i32 8)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQRSHRNH_N_U16]], i64 0
// CHECK:   ret i8 [[TMP1]]
uint8_t test_vqrshrnh_n_u16(uint16_t a) {
  return (uint8_t)vqrshrnh_n_u16(a, 8);
}

// CHECK-LABEL: @test_vqrshrns_n_u32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQRSHRNS_N_U32:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> [[TMP0]], i32 16)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQRSHRNS_N_U32]], i64 0
// CHECK:   ret i16 [[TMP1]]
uint16_t test_vqrshrns_n_u32(uint32_t a) {
  return (uint16_t)vqrshrns_n_u32(a, 16);
}

// CHECK-LABEL: @test_vqrshrnd_n_u64(
// CHECK:   [[VQRSHRND_N_U64:%.*]] = call i32 @llvm.aarch64.neon.uqrshrn.i32(i64 %a, i32 32)
// CHECK:   ret i32 [[VQRSHRND_N_U64]]
uint32_t test_vqrshrnd_n_u64(uint64_t a) {
  return (uint32_t)vqrshrnd_n_u64(a, 32);
}

// CHECK-LABEL: @test_vqshrunh_n_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQSHRUNH_N_S16:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> [[TMP0]], i32 8)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHRUNH_N_S16]], i64 0
// CHECK:   ret i8 [[TMP1]]
int8_t test_vqshrunh_n_s16(int16_t a) {
  return (int8_t)vqshrunh_n_s16(a, 8);
}

// CHECK-LABEL: @test_vqshruns_n_s32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQSHRUNS_N_S32:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> [[TMP0]], i32 16)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHRUNS_N_S32]], i64 0
// CHECK:   ret i16 [[TMP1]]
int16_t test_vqshruns_n_s32(int32_t a) {
  return (int16_t)vqshruns_n_s32(a, 16);
}

// CHECK-LABEL: @test_vqshrund_n_s64(
// CHECK:   [[VQSHRUND_N_S64:%.*]] = call i32 @llvm.aarch64.neon.sqshrun.i32(i64 %a, i32 32)
// CHECK:   ret i32 [[VQSHRUND_N_S64]]
int32_t test_vqshrund_n_s64(int64_t a) {
  return (int32_t)vqshrund_n_s64(a, 32);
}

// CHECK-LABEL: @test_vqrshrunh_n_s16(
// CHECK:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// CHECK:   [[VQRSHRUNH_N_S16:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> [[TMP0]], i32 8)
// CHECK:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQRSHRUNH_N_S16]], i64 0
// CHECK:   ret i8 [[TMP1]]
uint8_t test_vqrshrunh_n_s16(int16_t a) {
  return (uint8_t)vqrshrunh_n_s16(a, 8);
}

// CHECK-LABEL: @test_vqrshruns_n_s32(
// CHECK:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// CHECK:   [[VQRSHRUNS_N_S32:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> [[TMP0]], i32 16)
// CHECK:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQRSHRUNS_N_S32]], i64 0
// CHECK:   ret i16 [[TMP1]]
uint16_t test_vqrshruns_n_s32(int32_t a) {
  return (uint16_t)vqrshruns_n_s32(a, 16);
}

// CHECK-LABEL: @test_vqrshrund_n_s64(
// CHECK:   [[VQRSHRUND_N_S64:%.*]] = call i32 @llvm.aarch64.neon.sqrshrun.i32(i64 %a, i32 32)
// CHECK:   ret i32 [[VQRSHRUND_N_S64]]
uint32_t test_vqrshrund_n_s64(int64_t a) {
  return (uint32_t)vqrshrund_n_s64(a, 32);
}

// CHECK-LABEL: @test_vcvts_n_f32_s32(
// CHECK:   [[VCVTS_N_F32_S32:%.*]] = call float @llvm.aarch64.neon.vcvtfxs2fp.f32.i32(i32 %a, i32 1)
// CHECK:   ret float [[VCVTS_N_F32_S32]]
float32_t test_vcvts_n_f32_s32(int32_t a) {
  return vcvts_n_f32_s32(a, 1);
}

// CHECK-LABEL: @test_vcvtd_n_f64_s64(
// CHECK:   [[VCVTD_N_F64_S64:%.*]] = call double @llvm.aarch64.neon.vcvtfxs2fp.f64.i64(i64 %a, i32 1)
// CHECK:   ret double [[VCVTD_N_F64_S64]]
float64_t test_vcvtd_n_f64_s64(int64_t a) {
  return vcvtd_n_f64_s64(a, 1);
}

// CHECK-LABEL: @test_vcvts_n_f32_u32(
// CHECK:   [[VCVTS_N_F32_U32:%.*]] = call float @llvm.aarch64.neon.vcvtfxu2fp.f32.i32(i32 %a, i32 32)
// CHECK:   ret float [[VCVTS_N_F32_U32]]
float32_t test_vcvts_n_f32_u32(uint32_t a) {
  return vcvts_n_f32_u32(a, 32);
}

// CHECK-LABEL: @test_vcvtd_n_f64_u64(
// CHECK:   [[VCVTD_N_F64_U64:%.*]] = call double @llvm.aarch64.neon.vcvtfxu2fp.f64.i64(i64 %a, i32 64)
// CHECK:   ret double [[VCVTD_N_F64_U64]]
float64_t test_vcvtd_n_f64_u64(uint64_t a) {
  return vcvtd_n_f64_u64(a, 64);
}

// CHECK-LABEL: @test_vcvts_n_s32_f32(
// CHECK:   [[VCVTS_N_S32_F32:%.*]] = call i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f32(float %a, i32 1)
// CHECK:   ret i32 [[VCVTS_N_S32_F32]]
int32_t test_vcvts_n_s32_f32(float32_t a) {
  return (int32_t)vcvts_n_s32_f32(a, 1);
}

// CHECK-LABEL: @test_vcvtd_n_s64_f64(
// CHECK:   [[VCVTD_N_S64_F64:%.*]] = call i64 @llvm.aarch64.neon.vcvtfp2fxs.i64.f64(double %a, i32 1)
// CHECK:   ret i64 [[VCVTD_N_S64_F64]]
int64_t test_vcvtd_n_s64_f64(float64_t a) {
  return (int64_t)vcvtd_n_s64_f64(a, 1);
}

// CHECK-LABEL: @test_vcvts_n_u32_f32(
// CHECK:   [[VCVTS_N_U32_F32:%.*]] = call i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f32(float %a, i32 32)
// CHECK:   ret i32 [[VCVTS_N_U32_F32]]
uint32_t test_vcvts_n_u32_f32(float32_t a) {
  return (uint32_t)vcvts_n_u32_f32(a, 32);
}

// CHECK-LABEL: @test_vcvtd_n_u64_f64(
// CHECK:   [[VCVTD_N_U64_F64:%.*]] = call i64 @llvm.aarch64.neon.vcvtfp2fxu.i64.f64(double %a, i32 64)
// CHECK:   ret i64 [[VCVTD_N_U64_F64]]
uint64_t test_vcvtd_n_u64_f64(float64_t a) {
  return (uint64_t)vcvtd_n_u64_f64(a, 64);
}

// CHECK-LABEL: @test_vreinterpret_s8_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_s16(int16x4_t a) {
  return vreinterpret_s8_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_s32(int32x2_t a) {
  return vreinterpret_s8_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_s64(int64x1_t a) {
  return vreinterpret_s8_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_u8(
// CHECK:   ret <8 x i8> %a
int8x8_t test_vreinterpret_s8_u8(uint8x8_t a) {
  return vreinterpret_s8_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_u16(uint16x4_t a) {
  return vreinterpret_s8_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_u32(uint32x2_t a) {
  return vreinterpret_s8_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_u64(uint64x1_t a) {
  return vreinterpret_s8_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_f16(float16x4_t a) {
  return vreinterpret_s8_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_f32(float32x2_t a) {
  return vreinterpret_s8_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_f64(float64x1_t a) {
  return vreinterpret_s8_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_p8(
// CHECK:   ret <8 x i8> %a
int8x8_t test_vreinterpret_s8_p8(poly8x8_t a) {
  return vreinterpret_s8_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_p16(poly16x4_t a) {
  return vreinterpret_s8_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_s8_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
int8x8_t test_vreinterpret_s8_p64(poly64x1_t a) {
  return vreinterpret_s8_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_s8(int8x8_t a) {
  return vreinterpret_s16_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_s32(int32x2_t a) {
  return vreinterpret_s16_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_s64(int64x1_t a) {
  return vreinterpret_s16_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_u8(uint8x8_t a) {
  return vreinterpret_s16_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_u16(
// CHECK:   ret <4 x i16> %a
int16x4_t test_vreinterpret_s16_u16(uint16x4_t a) {
  return vreinterpret_s16_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_u32(uint32x2_t a) {
  return vreinterpret_s16_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_u64(uint64x1_t a) {
  return vreinterpret_s16_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_f16(float16x4_t a) {
  return vreinterpret_s16_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_f32(float32x2_t a) {
  return vreinterpret_s16_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_f64(float64x1_t a) {
  return vreinterpret_s16_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_p8(poly8x8_t a) {
  return vreinterpret_s16_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_p16(
// CHECK:   ret <4 x i16> %a
int16x4_t test_vreinterpret_s16_p16(poly16x4_t a) {
  return vreinterpret_s16_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_s16_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
int16x4_t test_vreinterpret_s16_p64(poly64x1_t a) {
  return vreinterpret_s16_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_s8(int8x8_t a) {
  return vreinterpret_s32_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_s16(int16x4_t a) {
  return vreinterpret_s32_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_s64(int64x1_t a) {
  return vreinterpret_s32_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_u8(uint8x8_t a) {
  return vreinterpret_s32_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_u16(uint16x4_t a) {
  return vreinterpret_s32_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_u32(
// CHECK:   ret <2 x i32> %a
int32x2_t test_vreinterpret_s32_u32(uint32x2_t a) {
  return vreinterpret_s32_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_u64(uint64x1_t a) {
  return vreinterpret_s32_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_f16(float16x4_t a) {
  return vreinterpret_s32_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_f32(float32x2_t a) {
  return vreinterpret_s32_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_f64(float64x1_t a) {
  return vreinterpret_s32_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_p8(poly8x8_t a) {
  return vreinterpret_s32_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_p16(poly16x4_t a) {
  return vreinterpret_s32_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_s32_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
int32x2_t test_vreinterpret_s32_p64(poly64x1_t a) {
  return vreinterpret_s32_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_s8(int8x8_t a) {
  return vreinterpret_s64_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_s16(int16x4_t a) {
  return vreinterpret_s64_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_s32(int32x2_t a) {
  return vreinterpret_s64_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_u8(uint8x8_t a) {
  return vreinterpret_s64_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_u16(uint16x4_t a) {
  return vreinterpret_s64_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_u32(uint32x2_t a) {
  return vreinterpret_s64_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_u64(
// CHECK:   ret <1 x i64> %a
int64x1_t test_vreinterpret_s64_u64(uint64x1_t a) {
  return vreinterpret_s64_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_f16(float16x4_t a) {
  return vreinterpret_s64_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_f32(float32x2_t a) {
  return vreinterpret_s64_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_f64(float64x1_t a) {
  return vreinterpret_s64_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_p8(poly8x8_t a) {
  return vreinterpret_s64_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
int64x1_t test_vreinterpret_s64_p16(poly16x4_t a) {
  return vreinterpret_s64_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_s64_p64(
// CHECK:   ret <1 x i64> %a
int64x1_t test_vreinterpret_s64_p64(poly64x1_t a) {
  return vreinterpret_s64_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_s8(
// CHECK:   ret <8 x i8> %a
uint8x8_t test_vreinterpret_u8_s8(int8x8_t a) {
  return vreinterpret_u8_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_s16(int16x4_t a) {
  return vreinterpret_u8_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_s32(int32x2_t a) {
  return vreinterpret_u8_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_s64(int64x1_t a) {
  return vreinterpret_u8_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_u16(uint16x4_t a) {
  return vreinterpret_u8_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_u32(uint32x2_t a) {
  return vreinterpret_u8_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_u64(uint64x1_t a) {
  return vreinterpret_u8_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_f16(float16x4_t a) {
  return vreinterpret_u8_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_f32(float32x2_t a) {
  return vreinterpret_u8_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_f64(float64x1_t a) {
  return vreinterpret_u8_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_p8(
// CHECK:   ret <8 x i8> %a
uint8x8_t test_vreinterpret_u8_p8(poly8x8_t a) {
  return vreinterpret_u8_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_p16(poly16x4_t a) {
  return vreinterpret_u8_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_u8_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
uint8x8_t test_vreinterpret_u8_p64(poly64x1_t a) {
  return vreinterpret_u8_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_s8(int8x8_t a) {
  return vreinterpret_u16_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_s16(
// CHECK:   ret <4 x i16> %a
uint16x4_t test_vreinterpret_u16_s16(int16x4_t a) {
  return vreinterpret_u16_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_s32(int32x2_t a) {
  return vreinterpret_u16_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_s64(int64x1_t a) {
  return vreinterpret_u16_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_u8(uint8x8_t a) {
  return vreinterpret_u16_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_u32(uint32x2_t a) {
  return vreinterpret_u16_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_u64(uint64x1_t a) {
  return vreinterpret_u16_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_f16(float16x4_t a) {
  return vreinterpret_u16_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_f32(float32x2_t a) {
  return vreinterpret_u16_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_f64(float64x1_t a) {
  return vreinterpret_u16_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_p8(poly8x8_t a) {
  return vreinterpret_u16_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_p16(
// CHECK:   ret <4 x i16> %a
uint16x4_t test_vreinterpret_u16_p16(poly16x4_t a) {
  return vreinterpret_u16_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_u16_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
uint16x4_t test_vreinterpret_u16_p64(poly64x1_t a) {
  return vreinterpret_u16_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_s8(int8x8_t a) {
  return vreinterpret_u32_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_s16(int16x4_t a) {
  return vreinterpret_u32_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_s32(
// CHECK:   ret <2 x i32> %a
uint32x2_t test_vreinterpret_u32_s32(int32x2_t a) {
  return vreinterpret_u32_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_s64(int64x1_t a) {
  return vreinterpret_u32_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_u8(uint8x8_t a) {
  return vreinterpret_u32_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_u16(uint16x4_t a) {
  return vreinterpret_u32_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_u64(uint64x1_t a) {
  return vreinterpret_u32_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_f16(float16x4_t a) {
  return vreinterpret_u32_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_f32(float32x2_t a) {
  return vreinterpret_u32_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_f64(float64x1_t a) {
  return vreinterpret_u32_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_p8(poly8x8_t a) {
  return vreinterpret_u32_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_p16(poly16x4_t a) {
  return vreinterpret_u32_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_u32_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// CHECK:   ret <2 x i32> [[TMP0]]
uint32x2_t test_vreinterpret_u32_p64(poly64x1_t a) {
  return vreinterpret_u32_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_s8(int8x8_t a) {
  return vreinterpret_u64_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_s16(int16x4_t a) {
  return vreinterpret_u64_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_s32(int32x2_t a) {
  return vreinterpret_u64_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_s64(
// CHECK:   ret <1 x i64> %a
uint64x1_t test_vreinterpret_u64_s64(int64x1_t a) {
  return vreinterpret_u64_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_u8(uint8x8_t a) {
  return vreinterpret_u64_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_u16(uint16x4_t a) {
  return vreinterpret_u64_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_u32(uint32x2_t a) {
  return vreinterpret_u64_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_f16(float16x4_t a) {
  return vreinterpret_u64_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_f32(float32x2_t a) {
  return vreinterpret_u64_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_f64(float64x1_t a) {
  return vreinterpret_u64_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_p8(poly8x8_t a) {
  return vreinterpret_u64_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
uint64x1_t test_vreinterpret_u64_p16(poly16x4_t a) {
  return vreinterpret_u64_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_u64_p64(
// CHECK:   ret <1 x i64> %a
uint64x1_t test_vreinterpret_u64_p64(poly64x1_t a) {
  return vreinterpret_u64_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_s8(int8x8_t a) {
  return vreinterpret_f16_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_s16(int16x4_t a) {
  return vreinterpret_f16_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_s32(int32x2_t a) {
  return vreinterpret_f16_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_s64(int64x1_t a) {
  return vreinterpret_f16_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_u8(uint8x8_t a) {
  return vreinterpret_f16_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_u16(uint16x4_t a) {
  return vreinterpret_f16_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_u32(uint32x2_t a) {
  return vreinterpret_f16_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_u64(uint64x1_t a) {
  return vreinterpret_f16_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_f32(float32x2_t a) {
  return vreinterpret_f16_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_f64(float64x1_t a) {
  return vreinterpret_f16_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_p8(poly8x8_t a) {
  return vreinterpret_f16_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_p16(poly16x4_t a) {
  return vreinterpret_f16_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_f16_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x half>
// CHECK:   ret <4 x half> [[TMP0]]
float16x4_t test_vreinterpret_f16_p64(poly64x1_t a) {
  return vreinterpret_f16_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_s8(int8x8_t a) {
  return vreinterpret_f32_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_s16(int16x4_t a) {
  return vreinterpret_f32_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_s32(int32x2_t a) {
  return vreinterpret_f32_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_s64(int64x1_t a) {
  return vreinterpret_f32_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_u8(uint8x8_t a) {
  return vreinterpret_f32_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_u16(uint16x4_t a) {
  return vreinterpret_f32_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_u32(uint32x2_t a) {
  return vreinterpret_f32_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_u64(uint64x1_t a) {
  return vreinterpret_f32_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_f16(float16x4_t a) {
  return vreinterpret_f32_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_f64(float64x1_t a) {
  return vreinterpret_f32_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_p8(poly8x8_t a) {
  return vreinterpret_f32_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_p16(poly16x4_t a) {
  return vreinterpret_f32_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_f32_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x float>
// CHECK:   ret <2 x float> [[TMP0]]
float32x2_t test_vreinterpret_f32_p64(poly64x1_t a) {
  return vreinterpret_f32_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_s8(int8x8_t a) {
  return vreinterpret_f64_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_s16(int16x4_t a) {
  return vreinterpret_f64_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_s32(int32x2_t a) {
  return vreinterpret_f64_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_s64(int64x1_t a) {
  return vreinterpret_f64_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_u8(uint8x8_t a) {
  return vreinterpret_f64_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_u16(uint16x4_t a) {
  return vreinterpret_f64_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_u32(uint32x2_t a) {
  return vreinterpret_f64_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_u64(uint64x1_t a) {
  return vreinterpret_f64_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_f16(float16x4_t a) {
  return vreinterpret_f64_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_f32(float32x2_t a) {
  return vreinterpret_f64_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_p8(poly8x8_t a) {
  return vreinterpret_f64_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_p16(poly16x4_t a) {
  return vreinterpret_f64_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_f64_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <1 x double>
// CHECK:   ret <1 x double> [[TMP0]]
float64x1_t test_vreinterpret_f64_p64(poly64x1_t a) {
  return vreinterpret_f64_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_s8(
// CHECK:   ret <8 x i8> %a
poly8x8_t test_vreinterpret_p8_s8(int8x8_t a) {
  return vreinterpret_p8_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_s16(int16x4_t a) {
  return vreinterpret_p8_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_s32(int32x2_t a) {
  return vreinterpret_p8_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_s64(int64x1_t a) {
  return vreinterpret_p8_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_u8(
// CHECK:   ret <8 x i8> %a
poly8x8_t test_vreinterpret_p8_u8(uint8x8_t a) {
  return vreinterpret_p8_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_u16(uint16x4_t a) {
  return vreinterpret_p8_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_u32(uint32x2_t a) {
  return vreinterpret_p8_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_u64(uint64x1_t a) {
  return vreinterpret_p8_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_f16(float16x4_t a) {
  return vreinterpret_p8_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_f32(float32x2_t a) {
  return vreinterpret_p8_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_f64(float64x1_t a) {
  return vreinterpret_p8_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_p16(poly16x4_t a) {
  return vreinterpret_p8_p16(a);
}

// CHECK-LABEL: @test_vreinterpret_p8_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   ret <8 x i8> [[TMP0]]
poly8x8_t test_vreinterpret_p8_p64(poly64x1_t a) {
  return vreinterpret_p8_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_s8(int8x8_t a) {
  return vreinterpret_p16_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_s16(
// CHECK:   ret <4 x i16> %a
poly16x4_t test_vreinterpret_p16_s16(int16x4_t a) {
  return vreinterpret_p16_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_s32(int32x2_t a) {
  return vreinterpret_p16_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_s64(int64x1_t a) {
  return vreinterpret_p16_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_u8(uint8x8_t a) {
  return vreinterpret_p16_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_u16(
// CHECK:   ret <4 x i16> %a
poly16x4_t test_vreinterpret_p16_u16(uint16x4_t a) {
  return vreinterpret_p16_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_u32(uint32x2_t a) {
  return vreinterpret_p16_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_u64(uint64x1_t a) {
  return vreinterpret_p16_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_f16(float16x4_t a) {
  return vreinterpret_p16_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_f32(float32x2_t a) {
  return vreinterpret_p16_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_f64(float64x1_t a) {
  return vreinterpret_p16_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_p8(poly8x8_t a) {
  return vreinterpret_p16_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_p16_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// CHECK:   ret <4 x i16> [[TMP0]]
poly16x4_t test_vreinterpret_p16_p64(poly64x1_t a) {
  return vreinterpret_p16_p64(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_s8(int8x8_t a) {
  return vreinterpret_p64_s8(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_s16(int16x4_t a) {
  return vreinterpret_p64_s16(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_s32(int32x2_t a) {
  return vreinterpret_p64_s32(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_s64(
// CHECK:   ret <1 x i64> %a
poly64x1_t test_vreinterpret_p64_s64(int64x1_t a) {
  return vreinterpret_p64_s64(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_u8(uint8x8_t a) {
  return vreinterpret_p64_u8(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_u16(uint16x4_t a) {
  return vreinterpret_p64_u16(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_u32(uint32x2_t a) {
  return vreinterpret_p64_u32(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_u64(
// CHECK:   ret <1 x i64> %a
poly64x1_t test_vreinterpret_p64_u64(uint64x1_t a) {
  return vreinterpret_p64_u64(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_f16(float16x4_t a) {
  return vreinterpret_p64_f16(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_f32(float32x2_t a) {
  return vreinterpret_p64_f32(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_f64(float64x1_t a) {
  return vreinterpret_p64_f64(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_p8(poly8x8_t a) {
  return vreinterpret_p64_p8(a);
}

// CHECK-LABEL: @test_vreinterpret_p64_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vreinterpret_p64_p16(poly16x4_t a) {
  return vreinterpret_p64_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_s16(int16x8_t a) {
  return vreinterpretq_s8_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_s32(int32x4_t a) {
  return vreinterpretq_s8_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_s64(int64x2_t a) {
  return vreinterpretq_s8_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_u8(
// CHECK:   ret <16 x i8> %a
int8x16_t test_vreinterpretq_s8_u8(uint8x16_t a) {
  return vreinterpretq_s8_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_u16(uint16x8_t a) {
  return vreinterpretq_s8_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_u32(uint32x4_t a) {
  return vreinterpretq_s8_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_u64(uint64x2_t a) {
  return vreinterpretq_s8_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_f16(float16x8_t a) {
  return vreinterpretq_s8_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_f32(float32x4_t a) {
  return vreinterpretq_s8_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_f64(float64x2_t a) {
  return vreinterpretq_s8_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_p8(
// CHECK:   ret <16 x i8> %a
int8x16_t test_vreinterpretq_s8_p8(poly8x16_t a) {
  return vreinterpretq_s8_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_p16(poly16x8_t a) {
  return vreinterpretq_s8_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s8_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_p64(poly64x2_t a) {
  return vreinterpretq_s8_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_s8(int8x16_t a) {
  return vreinterpretq_s16_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_s32(int32x4_t a) {
  return vreinterpretq_s16_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_s64(int64x2_t a) {
  return vreinterpretq_s16_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_u8(uint8x16_t a) {
  return vreinterpretq_s16_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_u16(
// CHECK:   ret <8 x i16> %a
int16x8_t test_vreinterpretq_s16_u16(uint16x8_t a) {
  return vreinterpretq_s16_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_u32(uint32x4_t a) {
  return vreinterpretq_s16_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_u64(uint64x2_t a) {
  return vreinterpretq_s16_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_f16(float16x8_t a) {
  return vreinterpretq_s16_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_f32(float32x4_t a) {
  return vreinterpretq_s16_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_f64(float64x2_t a) {
  return vreinterpretq_s16_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_p8(poly8x16_t a) {
  return vreinterpretq_s16_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_p16(
// CHECK:   ret <8 x i16> %a
int16x8_t test_vreinterpretq_s16_p16(poly16x8_t a) {
  return vreinterpretq_s16_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s16_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_p64(poly64x2_t a) {
  return vreinterpretq_s16_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_s8(int8x16_t a) {
  return vreinterpretq_s32_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_s16(int16x8_t a) {
  return vreinterpretq_s32_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_s64(int64x2_t a) {
  return vreinterpretq_s32_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_u8(uint8x16_t a) {
  return vreinterpretq_s32_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_u16(uint16x8_t a) {
  return vreinterpretq_s32_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_u32(
// CHECK:   ret <4 x i32> %a
int32x4_t test_vreinterpretq_s32_u32(uint32x4_t a) {
  return vreinterpretq_s32_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_u64(uint64x2_t a) {
  return vreinterpretq_s32_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_f16(float16x8_t a) {
  return vreinterpretq_s32_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_f32(float32x4_t a) {
  return vreinterpretq_s32_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_f64(float64x2_t a) {
  return vreinterpretq_s32_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_p8(poly8x16_t a) {
  return vreinterpretq_s32_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_p16(poly16x8_t a) {
  return vreinterpretq_s32_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s32_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_p64(poly64x2_t a) {
  return vreinterpretq_s32_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_s8(int8x16_t a) {
  return vreinterpretq_s64_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_s16(int16x8_t a) {
  return vreinterpretq_s64_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_s32(int32x4_t a) {
  return vreinterpretq_s64_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_u8(uint8x16_t a) {
  return vreinterpretq_s64_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_u16(uint16x8_t a) {
  return vreinterpretq_s64_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_u32(uint32x4_t a) {
  return vreinterpretq_s64_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_u64(
// CHECK:   ret <2 x i64> %a
int64x2_t test_vreinterpretq_s64_u64(uint64x2_t a) {
  return vreinterpretq_s64_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_f16(float16x8_t a) {
  return vreinterpretq_s64_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_f32(float32x4_t a) {
  return vreinterpretq_s64_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_f64(float64x2_t a) {
  return vreinterpretq_s64_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_p8(poly8x16_t a) {
  return vreinterpretq_s64_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_p16(poly16x8_t a) {
  return vreinterpretq_s64_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_s64_p64(
// CHECK:   ret <2 x i64> %a
int64x2_t test_vreinterpretq_s64_p64(poly64x2_t a) {
  return vreinterpretq_s64_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_s8(
// CHECK:   ret <16 x i8> %a
uint8x16_t test_vreinterpretq_u8_s8(int8x16_t a) {
  return vreinterpretq_u8_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_s16(int16x8_t a) {
  return vreinterpretq_u8_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_s32(int32x4_t a) {
  return vreinterpretq_u8_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_s64(int64x2_t a) {
  return vreinterpretq_u8_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_u16(uint16x8_t a) {
  return vreinterpretq_u8_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_u32(uint32x4_t a) {
  return vreinterpretq_u8_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_u64(uint64x2_t a) {
  return vreinterpretq_u8_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_f16(float16x8_t a) {
  return vreinterpretq_u8_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_f32(float32x4_t a) {
  return vreinterpretq_u8_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_f64(float64x2_t a) {
  return vreinterpretq_u8_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_p8(
// CHECK:   ret <16 x i8> %a
uint8x16_t test_vreinterpretq_u8_p8(poly8x16_t a) {
  return vreinterpretq_u8_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_p16(poly16x8_t a) {
  return vreinterpretq_u8_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u8_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_p64(poly64x2_t a) {
  return vreinterpretq_u8_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_s8(int8x16_t a) {
  return vreinterpretq_u16_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_s16(
// CHECK:   ret <8 x i16> %a
uint16x8_t test_vreinterpretq_u16_s16(int16x8_t a) {
  return vreinterpretq_u16_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_s32(int32x4_t a) {
  return vreinterpretq_u16_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_s64(int64x2_t a) {
  return vreinterpretq_u16_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_u8(uint8x16_t a) {
  return vreinterpretq_u16_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_u32(uint32x4_t a) {
  return vreinterpretq_u16_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_u64(uint64x2_t a) {
  return vreinterpretq_u16_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_f16(float16x8_t a) {
  return vreinterpretq_u16_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_f32(float32x4_t a) {
  return vreinterpretq_u16_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_f64(float64x2_t a) {
  return vreinterpretq_u16_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_p8(poly8x16_t a) {
  return vreinterpretq_u16_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_p16(
// CHECK:   ret <8 x i16> %a
uint16x8_t test_vreinterpretq_u16_p16(poly16x8_t a) {
  return vreinterpretq_u16_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u16_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_p64(poly64x2_t a) {
  return vreinterpretq_u16_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_s8(int8x16_t a) {
  return vreinterpretq_u32_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_s16(int16x8_t a) {
  return vreinterpretq_u32_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_s32(
// CHECK:   ret <4 x i32> %a
uint32x4_t test_vreinterpretq_u32_s32(int32x4_t a) {
  return vreinterpretq_u32_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_s64(int64x2_t a) {
  return vreinterpretq_u32_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_u8(uint8x16_t a) {
  return vreinterpretq_u32_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_u16(uint16x8_t a) {
  return vreinterpretq_u32_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_u64(uint64x2_t a) {
  return vreinterpretq_u32_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_f16(float16x8_t a) {
  return vreinterpretq_u32_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_f32(float32x4_t a) {
  return vreinterpretq_u32_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_f64(float64x2_t a) {
  return vreinterpretq_u32_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_p8(poly8x16_t a) {
  return vreinterpretq_u32_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_p16(poly16x8_t a) {
  return vreinterpretq_u32_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u32_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_p64(poly64x2_t a) {
  return vreinterpretq_u32_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_s8(int8x16_t a) {
  return vreinterpretq_u64_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_s16(int16x8_t a) {
  return vreinterpretq_u64_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_s32(int32x4_t a) {
  return vreinterpretq_u64_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_s64(
// CHECK:   ret <2 x i64> %a
uint64x2_t test_vreinterpretq_u64_s64(int64x2_t a) {
  return vreinterpretq_u64_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_u8(uint8x16_t a) {
  return vreinterpretq_u64_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_u16(uint16x8_t a) {
  return vreinterpretq_u64_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_u32(uint32x4_t a) {
  return vreinterpretq_u64_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_f16(float16x8_t a) {
  return vreinterpretq_u64_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_f32(float32x4_t a) {
  return vreinterpretq_u64_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_f64(float64x2_t a) {
  return vreinterpretq_u64_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_p8(poly8x16_t a) {
  return vreinterpretq_u64_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_p16(poly16x8_t a) {
  return vreinterpretq_u64_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_u64_p64(
// CHECK:   ret <2 x i64> %a
uint64x2_t test_vreinterpretq_u64_p64(poly64x2_t a) {
  return vreinterpretq_u64_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_s8(int8x16_t a) {
  return vreinterpretq_f16_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_s16(int16x8_t a) {
  return vreinterpretq_f16_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_s32(int32x4_t a) {
  return vreinterpretq_f16_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_s64(int64x2_t a) {
  return vreinterpretq_f16_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_u8(uint8x16_t a) {
  return vreinterpretq_f16_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_u16(uint16x8_t a) {
  return vreinterpretq_f16_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_u32(uint32x4_t a) {
  return vreinterpretq_f16_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_u64(uint64x2_t a) {
  return vreinterpretq_f16_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_f32(float32x4_t a) {
  return vreinterpretq_f16_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_f64(float64x2_t a) {
  return vreinterpretq_f16_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_p8(poly8x16_t a) {
  return vreinterpretq_f16_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_p16(poly16x8_t a) {
  return vreinterpretq_f16_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f16_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x half>
// CHECK:   ret <8 x half> [[TMP0]]
float16x8_t test_vreinterpretq_f16_p64(poly64x2_t a) {
  return vreinterpretq_f16_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_s8(int8x16_t a) {
  return vreinterpretq_f32_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_s16(int16x8_t a) {
  return vreinterpretq_f32_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_s32(int32x4_t a) {
  return vreinterpretq_f32_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_s64(int64x2_t a) {
  return vreinterpretq_f32_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_u8(uint8x16_t a) {
  return vreinterpretq_f32_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_u16(uint16x8_t a) {
  return vreinterpretq_f32_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_u32(uint32x4_t a) {
  return vreinterpretq_f32_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_u64(uint64x2_t a) {
  return vreinterpretq_f32_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_f16(float16x8_t a) {
  return vreinterpretq_f32_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_f64(float64x2_t a) {
  return vreinterpretq_f32_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_p8(poly8x16_t a) {
  return vreinterpretq_f32_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_p16(poly16x8_t a) {
  return vreinterpretq_f32_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f32_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_p64(poly64x2_t a) {
  return vreinterpretq_f32_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_s8(int8x16_t a) {
  return vreinterpretq_f64_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_s16(int16x8_t a) {
  return vreinterpretq_f64_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_s32(int32x4_t a) {
  return vreinterpretq_f64_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_s64(int64x2_t a) {
  return vreinterpretq_f64_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_u8(uint8x16_t a) {
  return vreinterpretq_f64_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_u16(uint16x8_t a) {
  return vreinterpretq_f64_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_u32(uint32x4_t a) {
  return vreinterpretq_f64_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_u64(uint64x2_t a) {
  return vreinterpretq_f64_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_f16(float16x8_t a) {
  return vreinterpretq_f64_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_f32(float32x4_t a) {
  return vreinterpretq_f64_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_p8(poly8x16_t a) {
  return vreinterpretq_f64_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_p16(poly16x8_t a) {
  return vreinterpretq_f64_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_f64_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_p64(poly64x2_t a) {
  return vreinterpretq_f64_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_s8(
// CHECK:   ret <16 x i8> %a
poly8x16_t test_vreinterpretq_p8_s8(int8x16_t a) {
  return vreinterpretq_p8_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_s16(int16x8_t a) {
  return vreinterpretq_p8_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_s32(int32x4_t a) {
  return vreinterpretq_p8_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_s64(int64x2_t a) {
  return vreinterpretq_p8_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_u8(
// CHECK:   ret <16 x i8> %a
poly8x16_t test_vreinterpretq_p8_u8(uint8x16_t a) {
  return vreinterpretq_p8_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_u16(uint16x8_t a) {
  return vreinterpretq_p8_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_u32(uint32x4_t a) {
  return vreinterpretq_p8_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_u64(uint64x2_t a) {
  return vreinterpretq_p8_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_f16(float16x8_t a) {
  return vreinterpretq_p8_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_f32(float32x4_t a) {
  return vreinterpretq_p8_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_f64(float64x2_t a) {
  return vreinterpretq_p8_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_p16(poly16x8_t a) {
  return vreinterpretq_p8_p16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p8_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_p64(poly64x2_t a) {
  return vreinterpretq_p8_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_s8(int8x16_t a) {
  return vreinterpretq_p16_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_s16(
// CHECK:   ret <8 x i16> %a
poly16x8_t test_vreinterpretq_p16_s16(int16x8_t a) {
  return vreinterpretq_p16_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_s32(int32x4_t a) {
  return vreinterpretq_p16_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_s64(int64x2_t a) {
  return vreinterpretq_p16_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_u8(uint8x16_t a) {
  return vreinterpretq_p16_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_u16(
// CHECK:   ret <8 x i16> %a
poly16x8_t test_vreinterpretq_p16_u16(uint16x8_t a) {
  return vreinterpretq_p16_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_u32(uint32x4_t a) {
  return vreinterpretq_p16_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_u64(uint64x2_t a) {
  return vreinterpretq_p16_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_f16(float16x8_t a) {
  return vreinterpretq_p16_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_f32(float32x4_t a) {
  return vreinterpretq_p16_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_f64(float64x2_t a) {
  return vreinterpretq_p16_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_p8(poly8x16_t a) {
  return vreinterpretq_p16_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_p16_p64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_p64(poly64x2_t a) {
  return vreinterpretq_p16_p64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_s8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_s8(int8x16_t a) {
  return vreinterpretq_p64_s8(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_s16(int16x8_t a) {
  return vreinterpretq_p64_s16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_s32(int32x4_t a) {
  return vreinterpretq_p64_s32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_s64(
// CHECK:   ret <2 x i64> %a
poly64x2_t test_vreinterpretq_p64_s64(int64x2_t a) {
  return vreinterpretq_p64_s64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_u8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_u8(uint8x16_t a) {
  return vreinterpretq_p64_u8(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_u16(uint16x8_t a) {
  return vreinterpretq_p64_u16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_u32(uint32x4_t a) {
  return vreinterpretq_p64_u32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_u64(
// CHECK:   ret <2 x i64> %a
poly64x2_t test_vreinterpretq_p64_u64(uint64x2_t a) {
  return vreinterpretq_p64_u64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_f16(float16x8_t a) {
  return vreinterpretq_p64_f16(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_f32(float32x4_t a) {
  return vreinterpretq_p64_f32(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_f64(float64x2_t a) {
  return vreinterpretq_p64_f64(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_p8(
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_p8(poly8x16_t a) {
  return vreinterpretq_p64_p8(a);
}

// CHECK-LABEL: @test_vreinterpretq_p64_p16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_p16(poly16x8_t a) {
  return vreinterpretq_p64_p16(a);
}

// CHECK-LABEL: @test_vabds_f32(
// CHECK:   [[VABDS_F32_I:%.*]] = call float @llvm.aarch64.sisd.fabd.f32(float %a, float %b)
// CHECK:   ret float [[VABDS_F32_I]]
float32_t test_vabds_f32(float32_t a, float32_t b) {
  return vabds_f32(a, b);
}

// CHECK-LABEL: @test_vabdd_f64(
// CHECK:   [[VABDD_F64_I:%.*]] = call double @llvm.aarch64.sisd.fabd.f64(double %a, double %b)
// CHECK:   ret double [[VABDD_F64_I]]
float64_t test_vabdd_f64(float64_t a, float64_t b) {
  return vabdd_f64(a, b);
}

// CHECK-LABEL: @test_vuqaddq_s8(
// CHECK: entry:
// CHECK-NEXT:  [[V:%.*]] = call <16 x i8> @llvm.aarch64.neon.suqadd.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK-NEXT:  ret <16 x i8> [[V]]
int8x16_t test_vuqaddq_s8(int8x16_t a, uint8x16_t b) {
  return vuqaddq_s8(a, b);
}

// CHECK-LABEL: @test_vuqaddq_s32(
// CHECK: [[V:%.*]] = call <4 x i32> @llvm.aarch64.neon.suqadd.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK-NEXT:  ret <4 x i32> [[V]]
int32x4_t test_vuqaddq_s32(int32x4_t a, uint32x4_t b) {
  return vuqaddq_s32(a, b);
}

// CHECK-LABEL: @test_vuqaddq_s64(
// CHECK: [[V:%.*]] = call <2 x i64> @llvm.aarch64.neon.suqadd.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK-NEXT:  ret <2 x i64> [[V]]
int64x2_t test_vuqaddq_s64(int64x2_t a, uint64x2_t b) {
  return vuqaddq_s64(a, b);
}

// CHECK-LABEL: @test_vuqaddq_s16(
// CHECK: [[V:%.*]] = call <8 x i16> @llvm.aarch64.neon.suqadd.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK-NEXT:  ret <8 x i16> [[V]]
int16x8_t test_vuqaddq_s16(int16x8_t a, uint16x8_t b) {
  return vuqaddq_s16(a, b);
}

// CHECK-LABEL: @test_vuqadd_s8(
// CHECK: entry:
// CHECK-NEXT: [[V:%.*]] = call <8 x i8> @llvm.aarch64.neon.suqadd.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK-NEXT: ret <8 x i8> [[V]]
int8x8_t test_vuqadd_s8(int8x8_t a, uint8x8_t b) {
  return vuqadd_s8(a, b);
}

// CHECK-LABEL: @test_vuqadd_s32(
// CHECK: [[V:%.*]] = call <2 x i32> @llvm.aarch64.neon.suqadd.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK-NEXT:  ret <2 x i32> [[V]]
int32x2_t test_vuqadd_s32(int32x2_t a, uint32x2_t b) {
  return vuqadd_s32(a, b);
}

// CHECK-LABEL: @test_vuqadd_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VUQADD2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.suqadd.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   ret <1 x i64> [[VUQADD2_I]]
int64x1_t test_vuqadd_s64(int64x1_t a, uint64x1_t b) {
  return vuqadd_s64(a, b);
}

// CHECK-LABEL: @test_vuqadd_s16(
// CHECK: [[V:%.*]] = call <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK-NEXT:  ret <4 x i16> [[V]]
int16x4_t test_vuqadd_s16(int16x4_t a, uint16x4_t b) {
  return vuqadd_s16(a, b);
}

// CHECK-LABEL: @test_vsqadd_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSQADD2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64> %a, <1 x i64> %b)
// CHECK:   ret <1 x i64> [[VSQADD2_I]]
uint64x1_t test_vsqadd_u64(uint64x1_t a, int64x1_t b) {
  return vsqadd_u64(a, b);
}

// CHECK-LABEL: @test_vsqadd_u8(
// CHECK:   [[VSQADD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.usqadd.v8i8(<8 x i8> %a, <8 x i8> %b)
// CHECK:   ret <8 x i8> [[VSQADD_I]]
uint8x8_t test_vsqadd_u8(uint8x8_t a, int8x8_t b) {
  return vsqadd_u8(a, b);
}

// CHECK-LABEL: @test_vsqaddq_u8(
// CHECK:   [[VSQADD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.usqadd.v16i8(<16 x i8> %a, <16 x i8> %b)
// CHECK:   ret <16 x i8> [[VSQADD_I]]
uint8x16_t test_vsqaddq_u8(uint8x16_t a, int8x16_t b) {
  return vsqaddq_u8(a, b);
}

// CHECK-LABEL: @test_vsqadd_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VSQADD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.usqadd.v4i16(<4 x i16> %a, <4 x i16> %b)
// CHECK:   ret <4 x i16> [[VSQADD2_I]]
uint16x4_t test_vsqadd_u16(uint16x4_t a, int16x4_t b) {
  return vsqadd_u16(a, b);
}

// CHECK-LABEL: @test_vsqaddq_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// CHECK:   [[VSQADD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.usqadd.v8i16(<8 x i16> %a, <8 x i16> %b)
// CHECK:   ret <8 x i16> [[VSQADD2_I]]
uint16x8_t test_vsqaddq_u16(uint16x8_t a, int16x8_t b) {
  return vsqaddq_u16(a, b);
}

// CHECK-LABEL: @test_vsqadd_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VSQADD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.usqadd.v2i32(<2 x i32> %a, <2 x i32> %b)
// CHECK:   ret <2 x i32> [[VSQADD2_I]]
uint32x2_t test_vsqadd_u32(uint32x2_t a, int32x2_t b) {
  return vsqadd_u32(a, b);
}

// CHECK-LABEL: @test_vsqaddq_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// CHECK:   [[VSQADD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.usqadd.v4i32(<4 x i32> %a, <4 x i32> %b)
// CHECK:   ret <4 x i32> [[VSQADD2_I]]
uint32x4_t test_vsqaddq_u32(uint32x4_t a, int32x4_t b) {
  return vsqaddq_u32(a, b);
}

// CHECK-LABEL: @test_vsqaddq_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSQADD2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.usqadd.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   ret <2 x i64> [[VSQADD2_I]]
uint64x2_t test_vsqaddq_u64(uint64x2_t a, int64x2_t b) {
  return vsqaddq_u64(a, b);
}

// CHECK-LABEL: @test_vabs_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.abs.v1i64(<1 x i64> %a)
// CHECK:   ret <1 x i64> [[VABS1_I]]
int64x1_t test_vabs_s64(int64x1_t a) {
  return vabs_s64(a);
}

// CHECK-LABEL: @test_vqabs_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VQABS_V1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqabs.v1i64(<1 x i64> %a)
// CHECK:   [[VQABS_V2_I:%.*]] = bitcast <1 x i64> [[VQABS_V1_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQABS_V1_I]]
int64x1_t test_vqabs_s64(int64x1_t a) {
  return vqabs_s64(a);
}

// CHECK-LABEL: @test_vqneg_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VQNEG_V1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqneg.v1i64(<1 x i64> %a)
// CHECK:   [[VQNEG_V2_I:%.*]] = bitcast <1 x i64> [[VQNEG_V1_I]] to <8 x i8>
// CHECK:   ret <1 x i64> [[VQNEG_V1_I]]
int64x1_t test_vqneg_s64(int64x1_t a) {
  return vqneg_s64(a);
}

// CHECK-LABEL: @test_vneg_s64(
// CHECK:   [[SUB_I:%.*]] = sub <1 x i64> zeroinitializer, %a
// CHECK:   ret <1 x i64> [[SUB_I]]
int64x1_t test_vneg_s64(int64x1_t a) {
  return vneg_s64(a);
}

// CHECK-LABEL: @test_vaddv_f32(
// CHECK:   [[VADDV_F32_I:%.*]] = call float @llvm.aarch64.neon.faddv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VADDV_F32_I]]
float32_t test_vaddv_f32(float32x2_t a) {
  return vaddv_f32(a);
}

// CHECK-LABEL: @test_vaddvq_f32(
// CHECK:   [[VADDVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.faddv.f32.v4f32(<4 x float> %a)
// CHECK:   ret float [[VADDVQ_F32_I]]
float32_t test_vaddvq_f32(float32x4_t a) {
  return vaddvq_f32(a);
}

// CHECK-LABEL: @test_vaddvq_f64(
// CHECK:   [[VADDVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.faddv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VADDVQ_F64_I]]
float64_t test_vaddvq_f64(float64x2_t a) {
  return vaddvq_f64(a);
}

// CHECK-LABEL: @test_vmaxv_f32(
// CHECK:   [[VMAXV_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VMAXV_F32_I]]
float32_t test_vmaxv_f32(float32x2_t a) {
  return vmaxv_f32(a);
}

// CHECK-LABEL: @test_vmaxvq_f64(
// CHECK:   [[VMAXVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.fmaxv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VMAXVQ_F64_I]]
float64_t test_vmaxvq_f64(float64x2_t a) {
  return vmaxvq_f64(a);
}

// CHECK-LABEL: @test_vminv_f32(
// CHECK:   [[VMINV_F32_I:%.*]] = call float @llvm.aarch64.neon.fminv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VMINV_F32_I]]
float32_t test_vminv_f32(float32x2_t a) {
  return vminv_f32(a);
}

// CHECK-LABEL: @test_vminvq_f64(
// CHECK:   [[VMINVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.fminv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VMINVQ_F64_I]]
float64_t test_vminvq_f64(float64x2_t a) {
  return vminvq_f64(a);
}

// CHECK-LABEL: @test_vmaxnmvq_f64(
// CHECK:   [[VMAXNMVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.fmaxnmv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VMAXNMVQ_F64_I]]
float64_t test_vmaxnmvq_f64(float64x2_t a) {
  return vmaxnmvq_f64(a);
}

// CHECK-LABEL: @test_vmaxnmv_f32(
// CHECK:   [[VMAXNMV_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxnmv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VMAXNMV_F32_I]]
float32_t test_vmaxnmv_f32(float32x2_t a) {
  return vmaxnmv_f32(a);
}

// CHECK-LABEL: @test_vminnmvq_f64(
// CHECK:   [[VMINNMVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.fminnmv.f64.v2f64(<2 x double> %a)
// CHECK:   ret double [[VMINNMVQ_F64_I]]
float64_t test_vminnmvq_f64(float64x2_t a) {
  return vminnmvq_f64(a);
}

// CHECK-LABEL: @test_vminnmv_f32(
// CHECK:   [[VMINNMV_F32_I:%.*]] = call float @llvm.aarch64.neon.fminnmv.f32.v2f32(<2 x float> %a)
// CHECK:   ret float [[VMINNMV_F32_I]]
float32_t test_vminnmv_f32(float32x2_t a) {
  return vminnmv_f32(a);
}

// CHECK-LABEL: @test_vpaddq_s64(
// CHECK:   [[VPADDQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VPADDQ_V3_I:%.*]] = bitcast <2 x i64> [[VPADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VPADDQ_V2_I]]
int64x2_t test_vpaddq_s64(int64x2_t a, int64x2_t b) {
  return vpaddq_s64(a, b);
}

// CHECK-LABEL: @test_vpaddq_u64(
// CHECK:   [[VPADDQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64> %a, <2 x i64> %b)
// CHECK:   [[VPADDQ_V3_I:%.*]] = bitcast <2 x i64> [[VPADDQ_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VPADDQ_V2_I]]
uint64x2_t test_vpaddq_u64(uint64x2_t a, uint64x2_t b) {
  return vpaddq_u64(a, b);
}

// CHECK-LABEL: @test_vpaddd_u64(
// CHECK:   [[VPADDD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64> %a)
// CHECK:   ret i64 [[VPADDD_U64_I]]
uint64_t test_vpaddd_u64(uint64x2_t a) {
  return vpaddd_u64(a);
}

// CHECK-LABEL: @test_vaddvq_s64(
// CHECK:   [[VADDVQ_S64_I:%.*]] = call i64 @llvm.aarch64.neon.saddv.i64.v2i64(<2 x i64> %a)
// CHECK:   ret i64 [[VADDVQ_S64_I]]
int64_t test_vaddvq_s64(int64x2_t a) {
  return vaddvq_s64(a);
}

// CHECK-LABEL: @test_vaddvq_u64(
// CHECK:   [[VADDVQ_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64> %a)
// CHECK:   ret i64 [[VADDVQ_U64_I]]
uint64_t test_vaddvq_u64(uint64x2_t a) {
  return vaddvq_u64(a);
}

// CHECK-LABEL: @test_vadd_f64(
// CHECK:   [[ADD_I:%.*]] = fadd <1 x double> %a, %b
// CHECK:   ret <1 x double> [[ADD_I]]
float64x1_t test_vadd_f64(float64x1_t a, float64x1_t b) {
  return vadd_f64(a, b);
}

// CHECK-LABEL: @test_vmul_f64(
// CHECK:   [[MUL_I:%.*]] = fmul <1 x double> %a, %b
// CHECK:   ret <1 x double> [[MUL_I]]
float64x1_t test_vmul_f64(float64x1_t a, float64x1_t b) {
  return vmul_f64(a, b);
}

// CHECK-LABEL: @test_vdiv_f64(
// CHECK:   [[DIV_I:%.*]] = fdiv <1 x double> %a, %b
// CHECK:   ret <1 x double> [[DIV_I]]
float64x1_t test_vdiv_f64(float64x1_t a, float64x1_t b) {
  return vdiv_f64(a, b);
}

// CHECK-LABEL: @test_vmla_f64(
// CHECK:   [[MUL_I:%.*]] = fmul <1 x double> %b, %c
// CHECK:   [[ADD_I:%.*]] = fadd <1 x double> %a, [[MUL_I]]
// CHECK:   ret <1 x double> [[ADD_I]]
float64x1_t test_vmla_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  return vmla_f64(a, b, c);
}

// CHECK-LABEL: @test_vmls_f64(
// CHECK:   [[MUL_I:%.*]] = fmul <1 x double> %b, %c
// CHECK:   [[SUB_I:%.*]] = fsub <1 x double> %a, [[MUL_I]]
// CHECK:   ret <1 x double> [[SUB_I]]
float64x1_t test_vmls_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  return vmls_f64(a, b, c);
}

// CHECK-LABEL: @test_vfma_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> %c to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> %b, <1 x double> %c, <1 x double> %a)
// CHECK:   ret <1 x double> [[TMP3]]
float64x1_t test_vfma_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  return vfma_f64(a, b, c);
}

// CHECK-LABEL: @test_vfms_f64(
// CHECK:   [[SUB_I:%.*]] = fneg <1 x double> %b
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> [[SUB_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> %c to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[SUB_I]], <1 x double> %c, <1 x double> %a)
// CHECK:   ret <1 x double> [[TMP3]]
float64x1_t test_vfms_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  return vfms_f64(a, b, c);
}

// CHECK-LABEL: @test_vsub_f64(
// CHECK:   [[SUB_I:%.*]] = fsub <1 x double> %a, %b
// CHECK:   ret <1 x double> [[SUB_I]]
float64x1_t test_vsub_f64(float64x1_t a, float64x1_t b) {
  return vsub_f64(a, b);
}

// CHECK-LABEL: @test_vabd_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VABD2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fabd.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x double> [[VABD2_I]]
float64x1_t test_vabd_f64(float64x1_t a, float64x1_t b) {
  return vabd_f64(a, b);
}

// CHECK-LABEL: @test_vmax_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VMAX2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmax.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x double> [[VMAX2_I]]
float64x1_t test_vmax_f64(float64x1_t a, float64x1_t b) {
  return vmax_f64(a, b);
}

// CHECK-LABEL: @test_vmin_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VMIN2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmin.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x double> [[VMIN2_I]]
float64x1_t test_vmin_f64(float64x1_t a, float64x1_t b) {
  return vmin_f64(a, b);
}

// CHECK-LABEL: @test_vmaxnm_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VMAXNM2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmaxnm.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x double> [[VMAXNM2_I]]
float64x1_t test_vmaxnm_f64(float64x1_t a, float64x1_t b) {
  return vmaxnm_f64(a, b);
}

// CHECK-LABEL: @test_vminnm_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VMINNM2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fminnm.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x double> [[VMINNM2_I]]
float64x1_t test_vminnm_f64(float64x1_t a, float64x1_t b) {
  return vminnm_f64(a, b);
}

// CHECK-LABEL: @test_vabs_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VABS1_I:%.*]] = call <1 x double> @llvm.fabs.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VABS1_I]]
float64x1_t test_vabs_f64(float64x1_t a) {
  return vabs_f64(a);
}

// CHECK-LABEL: @test_vneg_f64(
// CHECK:   [[SUB_I:%.*]] = fneg <1 x double> %a
// CHECK:   ret <1 x double> [[SUB_I]]
float64x1_t test_vneg_f64(float64x1_t a) {
  return vneg_f64(a);
}

// CHECK-LABEL: @test_vcvt_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtzs.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[TMP1]]
int64x1_t test_vcvt_s64_f64(float64x1_t a) {
  return vcvt_s64_f64(a);
}

// CHECK-LABEL: @test_vcvt_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtzu.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[TMP1]]
uint64x1_t test_vcvt_u64_f64(float64x1_t a) {
  return vcvt_u64_f64(a);
}

// CHECK-LABEL: @test_vcvtn_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVTN1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtns.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[VCVTN1_I]]
int64x1_t test_vcvtn_s64_f64(float64x1_t a) {
  return vcvtn_s64_f64(a);
}

// CHECK-LABEL: @test_vcvtn_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVTN1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtnu.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[VCVTN1_I]]
uint64x1_t test_vcvtn_u64_f64(float64x1_t a) {
  return vcvtn_u64_f64(a);
}

// CHECK-LABEL: @test_vcvtp_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVTP1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtps.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[VCVTP1_I]]
int64x1_t test_vcvtp_s64_f64(float64x1_t a) {
  return vcvtp_s64_f64(a);
}

// CHECK-LABEL: @test_vcvtp_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVTP1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtpu.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[VCVTP1_I]]
uint64x1_t test_vcvtp_u64_f64(float64x1_t a) {
  return vcvtp_u64_f64(a);
}

// CHECK-LABEL: @test_vcvtm_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVTM1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtms.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[VCVTM1_I]]
int64x1_t test_vcvtm_s64_f64(float64x1_t a) {
  return vcvtm_s64_f64(a);
}

// CHECK-LABEL: @test_vcvtm_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVTM1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtmu.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[VCVTM1_I]]
uint64x1_t test_vcvtm_u64_f64(float64x1_t a) {
  return vcvtm_u64_f64(a);
}

// CHECK-LABEL: @test_vcvta_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVTA1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtas.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[VCVTA1_I]]
int64x1_t test_vcvta_s64_f64(float64x1_t a) {
  return vcvta_s64_f64(a);
}

// CHECK-LABEL: @test_vcvta_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVTA1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtau.v1i64.v1f64(<1 x double> %a)
// CHECK:   ret <1 x i64> [[VCVTA1_I]]
uint64x1_t test_vcvta_u64_f64(float64x1_t a) {
  return vcvta_u64_f64(a);
}

// CHECK-LABEL: @test_vcvt_f64_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VCVT_I:%.*]] = sitofp <1 x i64> %a to <1 x double>
// CHECK:   ret <1 x double> [[VCVT_I]]
float64x1_t test_vcvt_f64_s64(int64x1_t a) {
  return vcvt_f64_s64(a);
}

// CHECK-LABEL: @test_vcvt_f64_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VCVT_I:%.*]] = uitofp <1 x i64> %a to <1 x double>
// CHECK:   ret <1 x double> [[VCVT_I]]
float64x1_t test_vcvt_f64_u64(uint64x1_t a) {
  return vcvt_f64_u64(a);
}

// CHECK-LABEL: @test_vcvt_n_s64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// CHECK:   [[VCVT_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.vcvtfp2fxs.v1i64.v1f64(<1 x double> [[VCVT_N]], i32 64)
// CHECK:   ret <1 x i64> [[VCVT_N1]]
int64x1_t test_vcvt_n_s64_f64(float64x1_t a) {
  return vcvt_n_s64_f64(a, 64);
}

// CHECK-LABEL: @test_vcvt_n_u64_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// CHECK:   [[VCVT_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.vcvtfp2fxu.v1i64.v1f64(<1 x double> [[VCVT_N]], i32 64)
// CHECK:   ret <1 x i64> [[VCVT_N1]]
uint64x1_t test_vcvt_n_u64_f64(float64x1_t a) {
  return vcvt_n_u64_f64(a, 64);
}

// CHECK-LABEL: @test_vcvt_n_f64_s64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VCVT_N1:%.*]] = call <1 x double> @llvm.aarch64.neon.vcvtfxs2fp.v1f64.v1i64(<1 x i64> [[VCVT_N]], i32 64)
// CHECK:   ret <1 x double> [[VCVT_N1]]
float64x1_t test_vcvt_n_f64_s64(int64x1_t a) {
  return vcvt_n_f64_s64(a, 64);
}

// CHECK-LABEL: @test_vcvt_n_f64_u64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VCVT_N1:%.*]] = call <1 x double> @llvm.aarch64.neon.vcvtfxu2fp.v1f64.v1i64(<1 x i64> [[VCVT_N]], i32 64)
// CHECK:   ret <1 x double> [[VCVT_N1]]
float64x1_t test_vcvt_n_f64_u64(uint64x1_t a) {
  return vcvt_n_f64_u64(a, 64);
}

// CHECK-LABEL: @test_vrndn_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRNDN1_I:%.*]] = call <1 x double> @llvm.roundeven.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRNDN1_I]]
float64x1_t test_vrndn_f64(float64x1_t a) {
  return vrndn_f64(a);
}

// CHECK-LABEL: @test_vrnda_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRNDA1_I:%.*]] = call <1 x double> @llvm.round.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRNDA1_I]]
float64x1_t test_vrnda_f64(float64x1_t a) {
  return vrnda_f64(a);
}

// CHECK-LABEL: @test_vrndp_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRNDP1_I:%.*]] = call <1 x double> @llvm.ceil.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRNDP1_I]]
float64x1_t test_vrndp_f64(float64x1_t a) {
  return vrndp_f64(a);
}

// CHECK-LABEL: @test_vrndm_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRNDM1_I:%.*]] = call <1 x double> @llvm.floor.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRNDM1_I]]
float64x1_t test_vrndm_f64(float64x1_t a) {
  return vrndm_f64(a);
}

// CHECK-LABEL: @test_vrndx_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRNDX1_I:%.*]] = call <1 x double> @llvm.rint.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRNDX1_I]]
float64x1_t test_vrndx_f64(float64x1_t a) {
  return vrndx_f64(a);
}

// CHECK-LABEL: @test_vrnd_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRNDZ1_I:%.*]] = call <1 x double> @llvm.trunc.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRNDZ1_I]]
float64x1_t test_vrnd_f64(float64x1_t a) {
  return vrnd_f64(a);
}

// CHECK-LABEL: @test_vrndi_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRNDI1_I:%.*]] = call <1 x double> @llvm.nearbyint.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRNDI1_I]]
float64x1_t test_vrndi_f64(float64x1_t a) {
  return vrndi_f64(a);
}

// CHECK-LABEL: @test_vrsqrte_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRSQRTE_V1_I:%.*]] = call <1 x double> @llvm.aarch64.neon.frsqrte.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRSQRTE_V1_I]]
float64x1_t test_vrsqrte_f64(float64x1_t a) {
  return vrsqrte_f64(a);
}

// CHECK-LABEL: @test_vrecpe_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VRECPE_V1_I:%.*]] = call <1 x double> @llvm.aarch64.neon.frecpe.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VRECPE_V1_I]]
float64x1_t test_vrecpe_f64(float64x1_t a) {
  return vrecpe_f64(a);
}

// CHECK-LABEL: @test_vsqrt_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[VSQRT_I:%.*]] = call <1 x double> @llvm.sqrt.v1f64(<1 x double> %a)
// CHECK:   ret <1 x double> [[VSQRT_I]]
float64x1_t test_vsqrt_f64(float64x1_t a) {
  return vsqrt_f64(a);
}

// CHECK-LABEL: @test_vrecps_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VRECPS_V2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.frecps.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   ret <1 x double> [[VRECPS_V2_I]]
float64x1_t test_vrecps_f64(float64x1_t a, float64x1_t b) {
  return vrecps_f64(a, b);
}

// CHECK-LABEL: @test_vrsqrts_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[VRSQRTS_V2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.frsqrts.v1f64(<1 x double> %a, <1 x double> %b)
// CHECK:   [[VRSQRTS_V3_I:%.*]] = bitcast <1 x double> [[VRSQRTS_V2_I]] to <8 x i8>
// CHECK:   ret <1 x double> [[VRSQRTS_V2_I]]
float64x1_t test_vrsqrts_f64(float64x1_t a, float64x1_t b) {
  return vrsqrts_f64(a, b);
}

// CHECK-LABEL: @test_vminv_s32(
// CHECK:   [[VMINV_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sminv.i32.v2i32(<2 x i32> %a)
// CHECK:   ret i32 [[VMINV_S32_I]]
int32_t test_vminv_s32(int32x2_t a) {
  return vminv_s32(a);
}

// CHECK-LABEL: @test_vminv_u32(
// CHECK:   [[VMINV_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uminv.i32.v2i32(<2 x i32> %a)
// CHECK:   ret i32 [[VMINV_U32_I]]
uint32_t test_vminv_u32(uint32x2_t a) {
  return vminv_u32(a);
}

// CHECK-LABEL: @test_vmaxv_s32(
// CHECK:   [[VMAXV_S32_I:%.*]] = call i32 @llvm.aarch64.neon.smaxv.i32.v2i32(<2 x i32> %a)
// CHECK:   ret i32 [[VMAXV_S32_I]]
int32_t test_vmaxv_s32(int32x2_t a) {
  return vmaxv_s32(a);
}

// CHECK-LABEL: @test_vmaxv_u32(
// CHECK:   [[VMAXV_U32_I:%.*]] = call i32 @llvm.aarch64.neon.umaxv.i32.v2i32(<2 x i32> %a)
// CHECK:   ret i32 [[VMAXV_U32_I]]
uint32_t test_vmaxv_u32(uint32x2_t a) {
  return vmaxv_u32(a);
}

// CHECK-LABEL: @test_vaddv_s32(
// CHECK:   [[VADDV_S32_I:%.*]] = call i32 @llvm.aarch64.neon.saddv.i32.v2i32(<2 x i32> %a)
// CHECK:   ret i32 [[VADDV_S32_I]]
int32_t test_vaddv_s32(int32x2_t a) {
  return vaddv_s32(a);
}

// CHECK-LABEL: @test_vaddv_u32(
// CHECK:   [[VADDV_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uaddv.i32.v2i32(<2 x i32> %a)
// CHECK:   ret i32 [[VADDV_U32_I]]
uint32_t test_vaddv_u32(uint32x2_t a) {
  return vaddv_u32(a);
}

// CHECK-LABEL: @test_vaddlv_s32(
// CHECK:   [[VADDLV_S32_I:%.*]] = call i64 @llvm.aarch64.neon.saddlv.i64.v2i32(<2 x i32> %a)
// CHECK:   ret i64 [[VADDLV_S32_I]]
int64_t test_vaddlv_s32(int32x2_t a) {
  return vaddlv_s32(a);
}

// CHECK-LABEL: @test_vaddlv_u32(
// CHECK:   [[VADDLV_U32_I:%.*]] = call i64 @llvm.aarch64.neon.uaddlv.i64.v2i32(<2 x i32> %a)
// CHECK:   ret i64 [[VADDLV_U32_I]]
uint64_t test_vaddlv_u32(uint32x2_t a) {
  return vaddlv_u32(a);
}
