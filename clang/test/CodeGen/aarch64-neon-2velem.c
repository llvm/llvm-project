// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

// CHECK-LABEL: @test_vmla_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[ADD]]
int16x4_t test_vmla_lane_s16(int16x4_t a, int16x4_t b, int16x4_t v) {
  return vmla_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlaq_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[ADD]]
int16x8_t test_vmlaq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t v) {
  return vmlaq_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmla_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[ADD]]
int32x2_t test_vmla_lane_s32(int32x2_t a, int32x2_t b, int32x2_t v) {
  return vmla_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlaq_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlaq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t v) {
  return vmlaq_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmla_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[ADD]]
int16x4_t test_vmla_laneq_s16(int16x4_t a, int16x4_t b, int16x8_t v) {
  return vmla_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlaq_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[ADD]]
int16x8_t test_vmlaq_laneq_s16(int16x8_t a, int16x8_t b, int16x8_t v) {
  return vmlaq_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmla_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[ADD]]
int32x2_t test_vmla_laneq_s32(int32x2_t a, int32x2_t b, int32x4_t v) {
  return vmla_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlaq_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlaq_laneq_s32(int32x4_t a, int32x4_t b, int32x4_t v) {
  return vmlaq_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmls_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[SUB]]
int16x4_t test_vmls_lane_s16(int16x4_t a, int16x4_t b, int16x4_t v) {
  return vmls_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsq_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[SUB]]
int16x8_t test_vmlsq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t v) {
  return vmlsq_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmls_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[SUB]]
int32x2_t test_vmls_lane_s32(int32x2_t a, int32x2_t b, int32x2_t v) {
  return vmls_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlsq_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t v) {
  return vmlsq_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmls_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[SUB]]
int16x4_t test_vmls_laneq_s16(int16x4_t a, int16x4_t b, int16x8_t v) {
  return vmls_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlsq_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[SUB]]
int16x8_t test_vmlsq_laneq_s16(int16x8_t a, int16x8_t b, int16x8_t v) {
  return vmlsq_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmls_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[SUB]]
int32x2_t test_vmls_laneq_s32(int32x2_t a, int32x2_t b, int32x4_t v) {
  return vmls_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsq_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsq_laneq_s32(int32x4_t a, int32x4_t b, int32x4_t v) {
  return vmlsq_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmul_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i16> [[MUL]]
int16x4_t test_vmul_lane_s16(int16x4_t a, int16x4_t v) {
  return vmul_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vmulq_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <8 x i16> [[MUL]]
int16x8_t test_vmulq_lane_s16(int16x8_t a, int16x4_t v) {
  return vmulq_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vmul_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <2 x i32> [[MUL]]
int32x2_t test_vmul_lane_s32(int32x2_t a, int32x2_t v) {
  return vmul_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vmulq_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i32> [[MUL]]
int32x4_t test_vmulq_lane_s32(int32x4_t a, int32x2_t v) {
  return vmulq_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vmul_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i16> [[MUL]]
uint16x4_t test_vmul_lane_u16(uint16x4_t a, uint16x4_t v) {
  return vmul_lane_u16(a, v, 3);
}

// CHECK-LABEL: @test_vmulq_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <8 x i16> [[MUL]]
uint16x8_t test_vmulq_lane_u16(uint16x8_t a, uint16x4_t v) {
  return vmulq_lane_u16(a, v, 3);
}

// CHECK-LABEL: @test_vmul_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <2 x i32> [[MUL]]
uint32x2_t test_vmul_lane_u32(uint32x2_t a, uint32x2_t v) {
  return vmul_lane_u32(a, v, 1);
}

// CHECK-LABEL: @test_vmulq_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i32> [[MUL]]
uint32x4_t test_vmulq_lane_u32(uint32x4_t a, uint32x2_t v) {
  return vmulq_lane_u32(a, v, 1);
}

// CHECK-LABEL: @test_vmul_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i16> [[MUL]]
int16x4_t test_vmul_laneq_s16(int16x4_t a, int16x8_t v) {
  return vmul_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vmulq_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <8 x i16> [[MUL]]
int16x8_t test_vmulq_laneq_s16(int16x8_t a, int16x8_t v) {
  return vmulq_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vmul_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <2 x i32> [[MUL]]
int32x2_t test_vmul_laneq_s32(int32x2_t a, int32x4_t v) {
  return vmul_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vmulq_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i32> [[MUL]]
int32x4_t test_vmulq_laneq_s32(int32x4_t a, int32x4_t v) {
  return vmulq_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vmul_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i16> [[MUL]]
uint16x4_t test_vmul_laneq_u16(uint16x4_t a, uint16x8_t v) {
  return vmul_laneq_u16(a, v, 7);
}

// CHECK-LABEL: @test_vmulq_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <8 x i16> [[MUL]]
uint16x8_t test_vmulq_laneq_u16(uint16x8_t a, uint16x8_t v) {
  return vmulq_laneq_u16(a, v, 7);
}

// CHECK-LABEL: @test_vmul_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <2 x i32> [[MUL]]
uint32x2_t test_vmul_laneq_u32(uint32x2_t a, uint32x4_t v) {
  return vmul_laneq_u32(a, v, 3);
}

// CHECK-LABEL: @test_vmulq_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i32> [[MUL]]
uint32x4_t test_vmulq_laneq_u32(uint32x4_t a, uint32x4_t v) {
  return vmulq_laneq_u32(a, v, 3);
}

// CHECK-LABEL: @test_vfma_lane_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <2 x i32> <i32 1, i32 1>
// CHECK:   [[FMLA:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[FMLA2:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[FMLA]], <2 x float> [[LANE]], <2 x float> [[FMLA1]])
// CHECK:   ret <2 x float> [[FMLA2]]
float32x2_t test_vfma_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vfma_lane_f32(a, b, v, 1);
}

// CHECK-LABEL: @test_vfmaq_lane_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[FMLA:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[FMLA2:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[FMLA]], <4 x float> [[LANE]], <4 x float> [[FMLA1]])
// CHECK:   ret <4 x float> [[FMLA2]]
float32x4_t test_vfmaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vfmaq_lane_f32(a, b, v, 1);
}

// CHECK-LABEL: @test_vfma_laneq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP5]], <4 x float> [[TMP5]], <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP6:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[TMP4]], <2 x float> [[TMP3]])
// CHECK:   ret <2 x float> [[TMP6]]
float32x2_t test_vfma_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vfma_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: @test_vfmaq_laneq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP5]], <4 x float> [[TMP5]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP6:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[TMP4]], <4 x float> [[TMP3]])
// CHECK:   ret <4 x float> [[TMP6]]
float32x4_t test_vfmaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vfmaq_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: @test_vfms_lane_f32(
// CHECK:   [[SUB:%.*]] = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SUB]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <2 x i32> <i32 1, i32 1>
// CHECK:   [[FMLA:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[FMLA2:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[FMLA]], <2 x float> [[LANE]], <2 x float> [[FMLA1]])
// CHECK:   ret <2 x float> [[FMLA2]]
float32x2_t test_vfms_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vfms_lane_f32(a, b, v, 1);
}

// CHECK-LABEL: @test_vfmsq_lane_f32(
// CHECK:   [[SUB:%.*]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SUB]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[FMLA:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[FMLA2:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[FMLA]], <4 x float> [[LANE]], <4 x float> [[FMLA1]])
// CHECK:   ret <4 x float> [[FMLA2]]
float32x4_t test_vfmsq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vfmsq_lane_f32(a, b, v, 1);
}

// CHECK-LABEL: @test_vfms_laneq_f32(
// CHECK:   [[SUB:%.*]] = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SUB]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP5]], <4 x float> [[TMP5]], <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP6:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[TMP4]], <2 x float> [[TMP3]])
// CHECK:   ret <2 x float> [[TMP6]]
float32x2_t test_vfms_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vfms_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: @test_vfmsq_laneq_f32(
// CHECK:   [[SUB:%.*]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SUB]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP5]], <4 x float> [[TMP5]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP6:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[TMP4]], <4 x float> [[TMP3]])
// CHECK:   ret <4 x float> [[TMP6]]
float32x4_t test_vfmsq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vfmsq_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: @test_vfmaq_lane_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// CHECK:   [[LANE:%.*]] = shufflevector <1 x double> [[TMP3]], <1 x double> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   [[FMLA:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[FMLA2:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[FMLA]], <2 x double> [[LANE]], <2 x double> [[FMLA1]])
// CHECK:   ret <2 x double> [[FMLA2]]
float64x2_t test_vfmaq_lane_f64(float64x2_t a, float64x2_t b, float64x1_t v) {
  return vfmaq_lane_f64(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmaq_laneq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x double> [[TMP5]], <2 x double> [[TMP5]], <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP6:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[TMP4]], <2 x double> [[TMP3]])
// CHECK:   ret <2 x double> [[TMP6]]
float64x2_t test_vfmaq_laneq_f64(float64x2_t a, float64x2_t b, float64x2_t v) {
  return vfmaq_laneq_f64(a, b, v, 1);
}

// CHECK-LABEL: @test_vfmsq_lane_f64(
// CHECK:   [[SUB:%.*]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> [[SUB]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// CHECK:   [[LANE:%.*]] = shufflevector <1 x double> [[TMP3]], <1 x double> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   [[FMLA:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[FMLA2:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[FMLA]], <2 x double> [[LANE]], <2 x double> [[FMLA1]])
// CHECK:   ret <2 x double> [[FMLA2]]
float64x2_t test_vfmsq_lane_f64(float64x2_t a, float64x2_t b, float64x1_t v) {
  return vfmsq_lane_f64(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmsq_laneq_f64(
// CHECK:   [[SUB:%.*]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> [[SUB]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x double> [[TMP5]], <2 x double> [[TMP5]], <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP6:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[TMP4]], <2 x double> [[TMP3]])
// CHECK:   ret <2 x double> [[TMP6]]
float64x2_t test_vfmsq_laneq_f64(float64x2_t a, float64x2_t b, float64x2_t v) {
  return vfmsq_laneq_f64(a, b, v, 1);
}

// CHECK-LABEL: @test_vfmas_laneq_f32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[EXTRACT:%.*]] = extractelement <4 x float> [[TMP1]], i32 3
// CHECK:   [[TMP2:%.*]] = call float @llvm.fma.f32(float %b, float [[EXTRACT]], float %a)
// CHECK:   ret float [[TMP2]]
float32_t test_vfmas_laneq_f32(float32_t a, float32_t b, float32x4_t v) {
  return vfmas_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: @test_vfmsd_lane_f64(
// CHECK:   [[SUB:%.*]] = fsub double -0.000000e+00, %b
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %v to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// CHECK:   [[EXTRACT:%.*]] = extractelement <1 x double> [[TMP1]], i32 0
// CHECK:   [[TMP2:%.*]] = call double @llvm.fma.f64(double [[SUB]], double [[EXTRACT]], double %a)
// CHECK:   ret double [[TMP2]]
float64_t test_vfmsd_lane_f64(float64_t a, float64_t b, float64x1_t v) {
  return vfmsd_lane_f64(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmss_laneq_f32(
// CHECK:   [[SUB:%.*]] = fsub float -0.000000e+00, %b
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[EXTRACT:%.*]] = extractelement <4 x float> [[TMP1]], i32 3
// CHECK:   [[TMP2:%.*]] = call float @llvm.fma.f32(float [[SUB]], float [[EXTRACT]], float %a)
// CHECK:   ret float [[TMP2]]
float32_t test_vfmss_laneq_f32(float32_t a, float32_t b, float32x4_t v) {
  return vfmss_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: @test_vfmsd_laneq_f64(
// CHECK:   [[SUB:%.*]] = fsub double -0.000000e+00, %b
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[EXTRACT:%.*]] = extractelement <2 x double> [[TMP1]], i32 1
// CHECK:   [[TMP2:%.*]] = call double @llvm.fma.f64(double [[SUB]], double [[EXTRACT]], double %a)
// CHECK:   ret double [[TMP2]]
float64_t test_vfmsd_laneq_f64(float64_t a, float64_t b, float64x2_t v) {
  return vfmsd_laneq_f64(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlal_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_lane_s16(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vmlal_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlal_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_lane_s32(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vmlal_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlal_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_laneq_s16(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vmlal_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlal_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_laneq_s32(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vmlal_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlal_high_lane_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_high_lane_s16(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vmlal_high_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlal_high_lane_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_high_lane_s32(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vmlal_high_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlal_high_laneq_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_high_laneq_s16(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vmlal_high_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlal_high_laneq_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_high_laneq_s32(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vmlal_high_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsl_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_lane_s16(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vmlsl_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsl_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_lane_s32(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vmlsl_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlsl_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_laneq_s16(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vmlsl_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlsl_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_laneq_s32(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vmlsl_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsl_high_lane_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_high_lane_s16(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vmlsl_high_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsl_high_lane_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_high_lane_s32(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vmlsl_high_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlsl_high_laneq_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_high_laneq_s16(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vmlsl_high_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlsl_high_laneq_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_high_laneq_s32(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vmlsl_high_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlal_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_lane_u16(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vmlal_lane_u16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlal_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_lane_u32(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vmlal_lane_u32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlal_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_laneq_u16(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vmlal_laneq_u16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlal_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_laneq_u32(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vmlal_laneq_u32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlal_high_lane_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_high_lane_u16(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vmlal_high_lane_u16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlal_high_lane_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_high_lane_u32(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vmlal_high_lane_u32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlal_high_laneq_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_high_laneq_u16(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vmlal_high_laneq_u16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlal_high_laneq_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_high_laneq_u32(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vmlal_high_laneq_u32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsl_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_lane_u16(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vmlsl_lane_u16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsl_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_lane_u32(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vmlsl_lane_u32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlsl_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_laneq_u16(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vmlsl_laneq_u16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlsl_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_laneq_u32(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vmlsl_laneq_u32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsl_high_lane_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_high_lane_u16(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vmlsl_high_lane_u16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsl_high_lane_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_high_lane_u32(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vmlsl_high_lane_u32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlsl_high_laneq_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_high_laneq_u16(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vmlsl_high_laneq_u16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlsl_high_laneq_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_high_laneq_u32(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vmlsl_high_laneq_u32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmull_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_lane_s16(int16x4_t a, int16x4_t v) {
  return vmull_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vmull_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_lane_s32(int32x2_t a, int32x2_t v) {
  return vmull_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vmull_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_lane_u16(uint16x4_t a, uint16x4_t v) {
  return vmull_lane_u16(a, v, 3);
}

// CHECK-LABEL: @test_vmull_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_lane_u32(uint32x2_t a, uint32x2_t v) {
  return vmull_lane_u32(a, v, 1);
}

// CHECK-LABEL: @test_vmull_high_lane_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_high_lane_s16(int16x8_t a, int16x4_t v) {
  return vmull_high_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vmull_high_lane_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_high_lane_s32(int32x4_t a, int32x2_t v) {
  return vmull_high_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vmull_high_lane_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_high_lane_u16(uint16x8_t a, uint16x4_t v) {
  return vmull_high_lane_u16(a, v, 3);
}

// CHECK-LABEL: @test_vmull_high_lane_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_high_lane_u32(uint32x4_t a, uint32x2_t v) {
  return vmull_high_lane_u32(a, v, 1);
}

// CHECK-LABEL: @test_vmull_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_laneq_s16(int16x4_t a, int16x8_t v) {
  return vmull_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vmull_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_laneq_s32(int32x2_t a, int32x4_t v) {
  return vmull_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vmull_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_laneq_u16(uint16x4_t a, uint16x8_t v) {
  return vmull_laneq_u16(a, v, 7);
}

// CHECK-LABEL: @test_vmull_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_laneq_u32(uint32x2_t a, uint32x4_t v) {
  return vmull_laneq_u32(a, v, 3);
}

// CHECK-LABEL: @test_vmull_high_laneq_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_high_laneq_s16(int16x8_t a, int16x8_t v) {
  return vmull_high_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vmull_high_laneq_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_high_laneq_s32(int32x4_t a, int32x4_t v) {
  return vmull_high_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vmull_high_laneq_u16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_high_laneq_u16(uint16x8_t a, uint16x8_t v) {
  return vmull_high_laneq_u16(a, v, 7);
}

// CHECK-LABEL: @test_vmull_high_laneq_u32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_high_laneq_u32(uint32x4_t a, uint32x4_t v) {
  return vmull_high_laneq_u32(a, v, 3);
}

// CHECK-LABEL: @test_vqdmlal_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_lane_s16(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vqdmlal_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmlal_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_lane_s32(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vqdmlal_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vqdmlal_high_lane_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_high_lane_s16(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vqdmlal_high_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmlal_high_lane_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_high_lane_s32(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vqdmlal_high_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vqdmlsl_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_lane_s16(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vqdmlsl_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmlsl_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_lane_s32(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vqdmlsl_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vqdmlsl_high_lane_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_high_lane_s16(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vqdmlsl_high_lane_s16(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmlsl_high_lane_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_high_lane_s32(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vqdmlsl_high_lane_s32(a, b, v, 1);
}

// CHECK-LABEL: @test_vqdmull_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_lane_s16(int16x4_t a, int16x4_t v) {
  return vqdmull_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vqdmull_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_lane_s32(int32x2_t a, int32x2_t v) {
  return vqdmull_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vqdmull_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_laneq_s16(int16x4_t a, int16x8_t v) {
  return vqdmull_laneq_s16(a, v, 3);
}

// CHECK-LABEL: @test_vqdmull_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_laneq_s32(int32x2_t a, int32x4_t v) {
  return vqdmull_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vqdmull_high_lane_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_high_lane_s16(int16x8_t a, int16x4_t v) {
  return vqdmull_high_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vqdmull_high_lane_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_high_lane_s32(int32x4_t a, int32x2_t v) {
  return vqdmull_high_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vqdmull_high_laneq_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_high_laneq_s16(int16x8_t a, int16x8_t v) {
  return vqdmull_high_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vqdmull_high_laneq_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_high_laneq_s32(int32x4_t a, int32x4_t v) {
  return vqdmull_high_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vqdmulh_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQDMULH_V2_I]]
int16x4_t test_vqdmulh_lane_s16(int16x4_t a, int16x4_t v) {
  return vqdmulh_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vqdmulhq_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16> %a, <8 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQDMULHQ_V2_I]]
int16x8_t test_vqdmulhq_lane_s16(int16x8_t a, int16x4_t v) {
  return vqdmulhq_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vqdmulh_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQDMULH_V2_I]]
int32x2_t test_vqdmulh_lane_s32(int32x2_t a, int32x2_t v) {
  return vqdmulh_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vqdmulhq_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> %a, <4 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULHQ_V2_I]]
int32x4_t test_vqdmulhq_lane_s32(int32x4_t a, int32x2_t v) {
  return vqdmulhq_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vqrdmulh_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQRDMULH_V2_I]]
int16x4_t test_vqrdmulh_lane_s16(int16x4_t a, int16x4_t v) {
  return vqrdmulh_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vqrdmulhq_lane_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %a, <8 x i16> [[SHUFFLE]])
// CHECK:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQRDMULHQ_V2_I]]
int16x8_t test_vqrdmulhq_lane_s16(int16x8_t a, int16x4_t v) {
  return vqrdmulhq_lane_s16(a, v, 3);
}

// CHECK-LABEL: @test_vqrdmulh_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQRDMULH_V2_I]]
int32x2_t test_vqrdmulh_lane_s32(int32x2_t a, int32x2_t v) {
  return vqrdmulh_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vqrdmulhq_lane_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %a, <4 x i32> [[SHUFFLE]])
// CHECK:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQRDMULHQ_V2_I]]
int32x4_t test_vqrdmulhq_lane_s32(int32x4_t a, int32x2_t v) {
  return vqrdmulhq_lane_s32(a, v, 1);
}

// CHECK-LABEL: @test_vmul_lane_f32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %a, [[SHUFFLE]]
// CHECK:   ret <2 x float> [[MUL]]
float32x2_t test_vmul_lane_f32(float32x2_t a, float32x2_t v) {
  return vmul_lane_f32(a, v, 1);
}

// CHECK-LABEL: @test_vmul_lane_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %v to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to double
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// CHECK:   [[EXTRACT:%.*]] = extractelement <1 x double> [[TMP3]], i32 0
// CHECK:   [[TMP4:%.*]] = fmul double [[TMP2]], [[EXTRACT]]
// CHECK:   [[TMP5:%.*]] = bitcast double [[TMP4]] to <1 x double>
// CHECK:   ret <1 x double> [[TMP5]]

float64x1_t test_vmul_lane_f64(float64x1_t a, float64x1_t v) {
  return vmul_lane_f64(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_lane_f32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %a, [[SHUFFLE]]
// CHECK:   ret <4 x float> [[MUL]]

float32x4_t test_vmulq_lane_f32(float32x4_t a, float32x2_t v) {
  return vmulq_lane_f32(a, v, 1);
}

// CHECK-LABEL: @test_vmulq_lane_f64(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <1 x double> %v, <1 x double> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <2 x double> %a, [[SHUFFLE]]
// CHECK:   ret <2 x double> [[MUL]]
float64x2_t test_vmulq_lane_f64(float64x2_t a, float64x1_t v) {
  return vmulq_lane_f64(a, v, 0);
}

// CHECK-LABEL: @test_vmul_laneq_f32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %a, [[SHUFFLE]]
// CHECK:   ret <2 x float> [[MUL]]
float32x2_t test_vmul_laneq_f32(float32x2_t a, float32x4_t v) {
  return vmul_laneq_f32(a, v, 3);
}

// CHECK-LABEL: @test_vmul_laneq_f64(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to double
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[EXTRACT:%.*]] = extractelement <2 x double> [[TMP3]], i32 1
// CHECK:   [[TMP4:%.*]] = fmul double [[TMP2]], [[EXTRACT]]
// CHECK:   [[TMP5:%.*]] = bitcast double [[TMP4]] to <1 x double>
// CHECK:   ret <1 x double> [[TMP5]]
float64x1_t test_vmul_laneq_f64(float64x1_t a, float64x2_t v) {
  return vmul_laneq_f64(a, v, 1);
}

// CHECK-LABEL: @test_vmulq_laneq_f32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %a, [[SHUFFLE]]
// CHECK:   ret <4 x float> [[MUL]]

float32x4_t test_vmulq_laneq_f32(float32x4_t a, float32x4_t v) {
  return vmulq_laneq_f32(a, v, 3);
}

// CHECK-LABEL: @test_vmulq_laneq_f64(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x double> %v, <2 x double> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = fmul <2 x double> %a, [[SHUFFLE]]
// CHECK:   ret <2 x double> [[MUL]]
float64x2_t test_vmulq_laneq_f64(float64x2_t a, float64x2_t v) {
  return vmulq_laneq_f64(a, v, 1);
}

// CHECK-LABEL: @test_vmulx_lane_f32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmulx.v2f32(<2 x float> %a, <2 x float> [[SHUFFLE]])
// CHECK:   ret <2 x float> [[VMULX2_I]]
float32x2_t test_vmulx_lane_f32(float32x2_t a, float32x2_t v) {
  return vmulx_lane_f32(a, v, 1);
}

// CHECK-LABEL: @test_vmulxq_lane_f32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmulx.v4f32(<4 x float> %a, <4 x float> [[SHUFFLE]])
// CHECK:   ret <4 x float> [[VMULX2_I]]
float32x4_t test_vmulxq_lane_f32(float32x4_t a, float32x2_t v) {
  return vmulxq_lane_f32(a, v, 1);
}

// CHECK-LABEL: @test_vmulxq_lane_f64(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <1 x double> %v, <1 x double> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmulx.v2f64(<2 x double> %a, <2 x double> [[SHUFFLE]])
// CHECK:   ret <2 x double> [[VMULX2_I]]
float64x2_t test_vmulxq_lane_f64(float64x2_t a, float64x1_t v) {
  return vmulxq_lane_f64(a, v, 0);
}

// CHECK-LABEL: @test_vmulx_laneq_f32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmulx.v2f32(<2 x float> %a, <2 x float> [[SHUFFLE]])
// CHECK:   ret <2 x float> [[VMULX2_I]]
float32x2_t test_vmulx_laneq_f32(float32x2_t a, float32x4_t v) {
  return vmulx_laneq_f32(a, v, 3);
}

// CHECK-LABEL: @test_vmulxq_laneq_f32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmulx.v4f32(<4 x float> %a, <4 x float> [[SHUFFLE]])
// CHECK:   ret <4 x float> [[VMULX2_I]]
float32x4_t test_vmulxq_laneq_f32(float32x4_t a, float32x4_t v) {
  return vmulxq_laneq_f32(a, v, 3);
}

// CHECK-LABEL: @test_vmulxq_laneq_f64(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x double> %v, <2 x double> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmulx.v2f64(<2 x double> %a, <2 x double> [[SHUFFLE]])
// CHECK:   ret <2 x double> [[VMULX2_I]]
float64x2_t test_vmulxq_laneq_f64(float64x2_t a, float64x2_t v) {
  return vmulxq_laneq_f64(a, v, 1);
}

// CHECK-LABEL: @test_vmla_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[ADD]]
int16x4_t test_vmla_lane_s16_0(int16x4_t a, int16x4_t b, int16x4_t v) {
  return vmla_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlaq_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[ADD]]
int16x8_t test_vmlaq_lane_s16_0(int16x8_t a, int16x8_t b, int16x4_t v) {
  return vmlaq_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmla_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[ADD]]
int32x2_t test_vmla_lane_s32_0(int32x2_t a, int32x2_t b, int32x2_t v) {
  return vmla_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlaq_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlaq_lane_s32_0(int32x4_t a, int32x4_t b, int32x2_t v) {
  return vmlaq_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmla_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[ADD]]
int16x4_t test_vmla_laneq_s16_0(int16x4_t a, int16x4_t b, int16x8_t v) {
  return vmla_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlaq_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[ADD]]
int16x8_t test_vmlaq_laneq_s16_0(int16x8_t a, int16x8_t b, int16x8_t v) {
  return vmlaq_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmla_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[ADD]]
int32x2_t test_vmla_laneq_s32_0(int32x2_t a, int32x2_t b, int32x4_t v) {
  return vmla_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlaq_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlaq_laneq_s32_0(int32x4_t a, int32x4_t b, int32x4_t v) {
  return vmlaq_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmls_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[SUB]]
int16x4_t test_vmls_lane_s16_0(int16x4_t a, int16x4_t b, int16x4_t v) {
  return vmls_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsq_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[SUB]]
int16x8_t test_vmlsq_lane_s16_0(int16x8_t a, int16x8_t b, int16x4_t v) {
  return vmlsq_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmls_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[SUB]]
int32x2_t test_vmls_lane_s32_0(int32x2_t a, int32x2_t b, int32x2_t v) {
  return vmls_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsq_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsq_lane_s32_0(int32x4_t a, int32x4_t b, int32x2_t v) {
  return vmlsq_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmls_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[SUB]]
int16x4_t test_vmls_laneq_s16_0(int16x4_t a, int16x4_t b, int16x8_t v) {
  return vmls_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsq_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[SUB]]
int16x8_t test_vmlsq_laneq_s16_0(int16x8_t a, int16x8_t b, int16x8_t v) {
  return vmlsq_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmls_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[SUB]]
int32x2_t test_vmls_laneq_s32_0(int32x2_t a, int32x2_t b, int32x4_t v) {
  return vmls_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsq_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsq_laneq_s32_0(int32x4_t a, int32x4_t b, int32x4_t v) {
  return vmlsq_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmul_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i16> [[MUL]]
int16x4_t test_vmul_lane_s16_0(int16x4_t a, int16x4_t v) {
  return vmul_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <8 x i16> [[MUL]]
int16x8_t test_vmulq_lane_s16_0(int16x8_t a, int16x4_t v) {
  return vmulq_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vmul_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <2 x i32> [[MUL]]
int32x2_t test_vmul_lane_s32_0(int32x2_t a, int32x2_t v) {
  return vmul_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i32> [[MUL]]
int32x4_t test_vmulq_lane_s32_0(int32x4_t a, int32x2_t v) {
  return vmulq_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmul_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i16> [[MUL]]
uint16x4_t test_vmul_lane_u16_0(uint16x4_t a, uint16x4_t v) {
  return vmul_lane_u16(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <8 x i16> [[MUL]]
uint16x8_t test_vmulq_lane_u16_0(uint16x8_t a, uint16x4_t v) {
  return vmulq_lane_u16(a, v, 0);
}

// CHECK-LABEL: @test_vmul_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <2 x i32> [[MUL]]
uint32x2_t test_vmul_lane_u32_0(uint32x2_t a, uint32x2_t v) {
  return vmul_lane_u32(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i32> [[MUL]]
uint32x4_t test_vmulq_lane_u32_0(uint32x4_t a, uint32x2_t v) {
  return vmulq_lane_u32(a, v, 0);
}

// CHECK-LABEL: @test_vmul_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i16> [[MUL]]
int16x4_t test_vmul_laneq_s16_0(int16x4_t a, int16x8_t v) {
  return vmul_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <8 x i16> [[MUL]]
int16x8_t test_vmulq_laneq_s16_0(int16x8_t a, int16x8_t v) {
  return vmulq_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vmul_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <2 x i32> [[MUL]]
int32x2_t test_vmul_laneq_s32_0(int32x2_t a, int32x4_t v) {
  return vmul_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i32> [[MUL]]
int32x4_t test_vmulq_laneq_s32_0(int32x4_t a, int32x4_t v) {
  return vmulq_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmul_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i16> [[MUL]]
uint16x4_t test_vmul_laneq_u16_0(uint16x4_t a, uint16x8_t v) {
  return vmul_laneq_u16(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %a, [[SHUFFLE]]
// CHECK:   ret <8 x i16> [[MUL]]
uint16x8_t test_vmulq_laneq_u16_0(uint16x8_t a, uint16x8_t v) {
  return vmulq_laneq_u16(a, v, 0);
}

// CHECK-LABEL: @test_vmul_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <2 x i32> [[MUL]]
uint32x2_t test_vmul_laneq_u32_0(uint32x2_t a, uint32x4_t v) {
  return vmul_laneq_u32(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %a, [[SHUFFLE]]
// CHECK:   ret <4 x i32> [[MUL]]
uint32x4_t test_vmulq_laneq_u32_0(uint32x4_t a, uint32x4_t v) {
  return vmulq_laneq_u32(a, v, 0);
}

// CHECK-LABEL: @test_vfma_lane_f32_0(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   [[FMLA:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[FMLA2:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[FMLA]], <2 x float> [[LANE]], <2 x float> [[FMLA1]])
// CHECK:   ret <2 x float> [[FMLA2]]
float32x2_t test_vfma_lane_f32_0(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vfma_lane_f32(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmaq_lane_f32_0(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   [[FMLA:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[FMLA2:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[FMLA]], <4 x float> [[LANE]], <4 x float> [[FMLA1]])
// CHECK:   ret <4 x float> [[FMLA2]]
float32x4_t test_vfmaq_lane_f32_0(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vfmaq_lane_f32(a, b, v, 0);
}

// CHECK-LABEL: @test_vfma_laneq_f32_0(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP5]], <4 x float> [[TMP5]], <2 x i32> zeroinitializer
// CHECK:   [[TMP6:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[TMP4]], <2 x float> [[TMP3]])
// CHECK:   ret <2 x float> [[TMP6]]
float32x2_t test_vfma_laneq_f32_0(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vfma_laneq_f32(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmaq_laneq_f32_0(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP5]], <4 x float> [[TMP5]], <4 x i32> zeroinitializer
// CHECK:   [[TMP6:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[TMP4]], <4 x float> [[TMP3]])
// CHECK:   ret <4 x float> [[TMP6]]
float32x4_t test_vfmaq_laneq_f32_0(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vfmaq_laneq_f32(a, b, v, 0);
}

// CHECK-LABEL: @test_vfms_lane_f32_0(
// CHECK:   [[SUB:%.*]] = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SUB]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <2 x i32> zeroinitializer
// CHECK:   [[FMLA:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[FMLA2:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[FMLA]], <2 x float> [[LANE]], <2 x float> [[FMLA1]])
// CHECK:   ret <2 x float> [[FMLA2]]
float32x2_t test_vfms_lane_f32_0(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vfms_lane_f32(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmsq_lane_f32_0(
// CHECK:   [[SUB:%.*]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SUB]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> %v to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x float> [[TMP3]], <2 x float> [[TMP3]], <4 x i32> zeroinitializer
// CHECK:   [[FMLA:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[FMLA1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[FMLA2:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[FMLA]], <4 x float> [[LANE]], <4 x float> [[FMLA1]])
// CHECK:   ret <4 x float> [[FMLA2]]
float32x4_t test_vfmsq_lane_f32_0(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vfmsq_lane_f32(a, b, v, 0);
}

// CHECK-LABEL: @test_vfms_laneq_f32_0(
// CHECK:   [[SUB:%.*]] = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SUB]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// CHECK:   [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP5]], <4 x float> [[TMP5]], <2 x i32> zeroinitializer
// CHECK:   [[TMP6:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[LANE]], <2 x float> [[TMP4]], <2 x float> [[TMP3]])
// CHECK:   ret <2 x float> [[TMP6]]
float32x2_t test_vfms_laneq_f32_0(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vfms_laneq_f32(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmsq_laneq_f32_0(
// CHECK:   [[SUB:%.*]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SUB]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x float>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x float> [[TMP5]], <4 x float> [[TMP5]], <4 x i32> zeroinitializer
// CHECK:   [[TMP6:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[LANE]], <4 x float> [[TMP4]], <4 x float> [[TMP3]])
// CHECK:   ret <4 x float> [[TMP6]]
float32x4_t test_vfmsq_laneq_f32_0(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vfmsq_laneq_f32(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmaq_laneq_f64_0(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x double> [[TMP5]], <2 x double> [[TMP5]], <2 x i32> zeroinitializer
// CHECK:   [[TMP6:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[TMP4]], <2 x double> [[TMP3]])
// CHECK:   ret <2 x double> [[TMP6]]
float64x2_t test_vfmaq_laneq_f64_0(float64x2_t a, float64x2_t b, float64x2_t v) {
  return vfmaq_laneq_f64(a, b, v, 0);
}

// CHECK-LABEL: @test_vfmsq_laneq_f64_0(
// CHECK:   [[SUB:%.*]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %b
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> [[SUB]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// CHECK:   [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// CHECK:   [[LANE:%.*]] = shufflevector <2 x double> [[TMP5]], <2 x double> [[TMP5]], <2 x i32> zeroinitializer
// CHECK:   [[TMP6:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[TMP4]], <2 x double> [[TMP3]])
// CHECK:   ret <2 x double> [[TMP6]]
float64x2_t test_vfmsq_laneq_f64_0(float64x2_t a, float64x2_t b, float64x2_t v) {
  return vfmsq_laneq_f64(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_lane_s16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vmlal_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_lane_s32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vmlal_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_laneq_s16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vmlal_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_laneq_s32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vmlal_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_high_lane_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_high_lane_s16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vmlal_high_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_high_lane_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_high_lane_s32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vmlal_high_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_high_laneq_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_high_laneq_s16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vmlal_high_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_high_laneq_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_high_laneq_s32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vmlal_high_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_lane_s16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vmlsl_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_lane_s32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vmlsl_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_laneq_s16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vmlsl_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_laneq_s32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vmlsl_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_high_lane_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_high_lane_s16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vmlsl_high_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_high_lane_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_high_lane_s32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vmlsl_high_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_high_laneq_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_high_laneq_s16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vmlsl_high_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_high_laneq_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_high_laneq_s32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vmlsl_high_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_lane_u16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vmlal_lane_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_lane_u32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vmlal_lane_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_laneq_u16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vmlal_laneq_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_laneq_u32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vmlal_laneq_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_high_lane_u16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_high_lane_u16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vmlal_high_lane_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_high_lane_u32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_high_lane_u32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vmlal_high_lane_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_high_laneq_u16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[ADD]]
int32x4_t test_vmlal_high_laneq_u16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vmlal_high_laneq_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlal_high_laneq_u32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[ADD:%.*]] = add <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[ADD]]
int64x2_t test_vmlal_high_laneq_u32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vmlal_high_laneq_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_lane_u16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vmlsl_lane_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_lane_u32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vmlsl_lane_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_laneq_u16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vmlsl_laneq_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_laneq_u32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vmlsl_laneq_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_high_lane_u16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_high_lane_u16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vmlsl_high_lane_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_high_lane_u32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_high_lane_u32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vmlsl_high_lane_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_high_laneq_u16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[VMULL2_I]]
// CHECK:   ret <4 x i32> [[SUB]]
int32x4_t test_vmlsl_high_laneq_u16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vmlsl_high_laneq_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsl_high_laneq_u32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[SUB:%.*]] = sub <2 x i64> %a, [[VMULL2_I]]
// CHECK:   ret <2 x i64> [[SUB]]
int64x2_t test_vmlsl_high_laneq_u32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vmlsl_high_laneq_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmull_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_lane_s16_0(int16x4_t a, int16x4_t v) {
  return vmull_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vmull_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_lane_s32_0(int32x2_t a, int32x2_t v) {
  return vmull_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmull_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_lane_u16_0(uint16x4_t a, uint16x4_t v) {
  return vmull_lane_u16(a, v, 0);
}

// CHECK-LABEL: @test_vmull_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_lane_u32_0(uint32x2_t a, uint32x2_t v) {
  return vmull_lane_u32(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_lane_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_high_lane_s16_0(int16x8_t a, int16x4_t v) {
  return vmull_high_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_lane_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_high_lane_s32_0(int32x4_t a, int32x2_t v) {
  return vmull_high_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_lane_u16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_high_lane_u16_0(uint16x8_t a, uint16x4_t v) {
  return vmull_high_lane_u16(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_lane_u32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_high_lane_u32_0(uint32x4_t a, uint32x2_t v) {
  return vmull_high_lane_u32(a, v, 0);
}

// CHECK-LABEL: @test_vmull_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_laneq_s16_0(int16x4_t a, int16x8_t v) {
  return vmull_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vmull_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_laneq_s32_0(int32x2_t a, int32x4_t v) {
  return vmull_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmull_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_laneq_u16_0(uint16x4_t a, uint16x8_t v) {
  return vmull_laneq_u16(a, v, 0);
}

// CHECK-LABEL: @test_vmull_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_laneq_u32_0(uint32x2_t a, uint32x4_t v) {
  return vmull_laneq_u32(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_laneq_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
int32x4_t test_vmull_high_laneq_s16_0(int16x8_t a, int16x8_t v) {
  return vmull_high_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_laneq_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
int64x2_t test_vmull_high_laneq_s32_0(int32x4_t a, int32x4_t v) {
  return vmull_high_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_laneq_u16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VMULL2_I]]
uint32x4_t test_vmull_high_laneq_u16_0(uint16x8_t a, uint16x8_t v) {
  return vmull_high_laneq_u16(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_laneq_u32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   ret <2 x i64> [[VMULL2_I]]
uint64x2_t test_vmull_high_laneq_u32_0(uint32x4_t a, uint32x4_t v) {
  return vmull_high_laneq_u32(a, v, 0);
}

// CHECK-LABEL: @test_vqdmlal_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_lane_s16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vqdmlal_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlal_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_lane_s32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vqdmlal_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlal_high_lane_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_high_lane_s16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vqdmlal_high_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlal_high_lane_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_high_lane_s32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vqdmlal_high_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlsl_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_lane_s16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  return vqdmlsl_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlsl_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_lane_s32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  return vqdmlsl_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlsl_high_lane_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_high_lane_s16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  return vqdmlsl_high_lane_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlsl_high_lane_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_high_lane_s32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  return vqdmlsl_high_lane_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmull_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_lane_s16_0(int16x4_t a, int16x4_t v) {
  return vqdmull_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqdmull_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_lane_s32_0(int32x2_t a, int32x2_t v) {
  return vqdmull_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqdmull_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_laneq_s16_0(int16x4_t a, int16x8_t v) {
  return vqdmull_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqdmull_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_laneq_s32_0(int32x2_t a, int32x4_t v) {
  return vqdmull_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqdmull_high_lane_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_high_lane_s16_0(int16x8_t a, int16x4_t v) {
  return vqdmull_high_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqdmull_high_lane_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_high_lane_s32_0(int32x4_t a, int32x2_t v) {
  return vqdmull_high_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqdmull_high_laneq_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V2_I]]
int32x4_t test_vqdmull_high_laneq_s16_0(int16x8_t a, int16x8_t v) {
  return vqdmull_high_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqdmull_high_laneq_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V2_I]]
int64x2_t test_vqdmull_high_laneq_s32_0(int32x4_t a, int32x4_t v) {
  return vqdmull_high_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqdmulh_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQDMULH_V2_I]]
int16x4_t test_vqdmulh_lane_s16_0(int16x4_t a, int16x4_t v) {
  return vqdmulh_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqdmulhq_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16> %a, <8 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQDMULHQ_V2_I]]
int16x8_t test_vqdmulhq_lane_s16_0(int16x8_t a, int16x4_t v) {
  return vqdmulhq_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqdmulh_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQDMULH_V2_I]]
int32x2_t test_vqdmulh_lane_s32_0(int32x2_t a, int32x2_t v) {
  return vqdmulh_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqdmulhq_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> %a, <4 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULHQ_V2_I]]
int32x4_t test_vqdmulhq_lane_s32_0(int32x4_t a, int32x2_t v) {
  return vqdmulhq_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqrdmulh_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQRDMULH_V2_I]]
int16x4_t test_vqrdmulh_lane_s16_0(int16x4_t a, int16x4_t v) {
  return vqrdmulh_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqrdmulhq_lane_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %a, <8 x i16> [[SHUFFLE]])
// CHECK:   ret <8 x i16> [[VQRDMULHQ_V2_I]]
int16x8_t test_vqrdmulhq_lane_s16_0(int16x8_t a, int16x4_t v) {
  return vqrdmulhq_lane_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqrdmulh_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQRDMULH_V2_I]]
int32x2_t test_vqrdmulh_lane_s32_0(int32x2_t a, int32x2_t v) {
  return vqrdmulh_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqrdmulhq_lane_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %a, <4 x i32> [[SHUFFLE]])
// CHECK:   ret <4 x i32> [[VQRDMULHQ_V2_I]]
int32x4_t test_vqrdmulhq_lane_s32_0(int32x4_t a, int32x2_t v) {
  return vqrdmulhq_lane_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmul_lane_f32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %a, [[SHUFFLE]]
// CHECK:   ret <2 x float> [[MUL]]
float32x2_t test_vmul_lane_f32_0(float32x2_t a, float32x2_t v) {
  return vmul_lane_f32(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_lane_f32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %a, [[SHUFFLE]]
// CHECK:   ret <4 x float> [[MUL]]
float32x4_t test_vmulq_lane_f32_0(float32x4_t a, float32x2_t v) {
  return vmulq_lane_f32(a, v, 0);
}

// CHECK-LABEL: @test_vmul_laneq_f32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %a, [[SHUFFLE]]
// CHECK:   ret <2 x float> [[MUL]]
float32x2_t test_vmul_laneq_f32_0(float32x2_t a, float32x4_t v) {
  return vmul_laneq_f32(a, v, 0);
}

// CHECK-LABEL: @test_vmul_laneq_f64_0(
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> %v to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to double
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// CHECK:   [[EXTRACT:%.*]] = extractelement <2 x double> [[TMP3]], i32 0
// CHECK:   [[TMP4:%.*]] = fmul double [[TMP2]], [[EXTRACT]]
// CHECK:   [[TMP5:%.*]] = bitcast double [[TMP4]] to <1 x double>
// CHECK:   ret <1 x double> [[TMP5]]
float64x1_t test_vmul_laneq_f64_0(float64x1_t a, float64x2_t v) {
  return vmul_laneq_f64(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_laneq_f32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %a, [[SHUFFLE]]
// CHECK:   ret <4 x float> [[MUL]]
float32x4_t test_vmulq_laneq_f32_0(float32x4_t a, float32x4_t v) {
  return vmulq_laneq_f32(a, v, 0);
}

// CHECK-LABEL: @test_vmulq_laneq_f64_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x double> %v, <2 x double> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <2 x double> %a, [[SHUFFLE]]
// CHECK:   ret <2 x double> [[MUL]]
float64x2_t test_vmulq_laneq_f64_0(float64x2_t a, float64x2_t v) {
  return vmulq_laneq_f64(a, v, 0);
}

// CHECK-LABEL: @test_vmulx_lane_f32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmulx.v2f32(<2 x float> %a, <2 x float> [[SHUFFLE]])
// CHECK:   ret <2 x float> [[VMULX2_I]]
float32x2_t test_vmulx_lane_f32_0(float32x2_t a, float32x2_t v) {
  return vmulx_lane_f32(a, v, 0);
}

// CHECK-LABEL: @test_vmulxq_lane_f32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmulx.v4f32(<4 x float> %a, <4 x float> [[SHUFFLE]])
// CHECK:   ret <4 x float> [[VMULX2_I]]
float32x4_t test_vmulxq_lane_f32_0(float32x4_t a, float32x2_t v) {
  return vmulxq_lane_f32(a, v, 0);
}

// CHECK-LABEL: @test_vmulxq_lane_f64_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <1 x double> %v, <1 x double> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmulx.v2f64(<2 x double> %a, <2 x double> [[SHUFFLE]])
// CHECK:   ret <2 x double> [[VMULX2_I]]
float64x2_t test_vmulxq_lane_f64_0(float64x2_t a, float64x1_t v) {
  return vmulxq_lane_f64(a, v, 0);
}

// CHECK-LABEL: @test_vmulx_laneq_f32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmulx.v2f32(<2 x float> %a, <2 x float> [[SHUFFLE]])
// CHECK:   ret <2 x float> [[VMULX2_I]]
float32x2_t test_vmulx_laneq_f32_0(float32x2_t a, float32x4_t v) {
  return vmulx_laneq_f32(a, v, 0);
}

// CHECK-LABEL: @test_vmulxq_laneq_f32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmulx.v4f32(<4 x float> %a, <4 x float> [[SHUFFLE]])
// CHECK:   ret <4 x float> [[VMULX2_I]]
float32x4_t test_vmulxq_laneq_f32_0(float32x4_t a, float32x4_t v) {
  return vmulxq_laneq_f32(a, v, 0);
}

// CHECK-LABEL: @test_vmulxq_laneq_f64_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x double> %v, <2 x double> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x double> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VMULX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmulx.v2f64(<2 x double> %a, <2 x double> [[SHUFFLE]])
// CHECK:   ret <2 x double> [[VMULX2_I]]
float64x2_t test_vmulxq_laneq_f64_0(float64x2_t a, float64x2_t v) {
  return vmulxq_laneq_f64(a, v, 0);
}

// CHECK-LABEL: @test_vmull_high_n_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VMULL5_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   ret <4 x i32> [[VMULL5_I_I]]
int32x4_t test_vmull_high_n_s16(int16x8_t a, int16_t b) {
  return vmull_high_n_s16(a, b);
}

// CHECK-LABEL: @test_vmull_high_n_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %b, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VMULL3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   ret <2 x i64> [[VMULL3_I_I]]
int64x2_t test_vmull_high_n_s32(int32x4_t a, int32_t b) {
  return vmull_high_n_s32(a, b);
}

// CHECK-LABEL: @test_vmull_high_n_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VMULL5_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   ret <4 x i32> [[VMULL5_I_I]]
uint32x4_t test_vmull_high_n_u16(uint16x8_t a, uint16_t b) {
  return vmull_high_n_u16(a, b);
}

// CHECK-LABEL: @test_vmull_high_n_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %b, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VMULL3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   ret <2 x i64> [[VMULL3_I_I]]
uint64x2_t test_vmull_high_n_u32(uint32x4_t a, uint32_t b) {
  return vmull_high_n_u32(a, b);
}

// CHECK-LABEL: @test_vqdmull_high_n_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VQDMULL_V5_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   [[VQDMULL_V6_I_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V5_I_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V5_I_I]]
int32x4_t test_vqdmull_high_n_s16(int16x8_t a, int16_t b) {
  return vqdmull_high_n_s16(a, b);
}

// CHECK-LABEL: @test_vqdmull_high_n_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %b, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VQDMULL_V3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   [[VQDMULL_V4_I_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V3_I_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V3_I_I]]
int64x2_t test_vqdmull_high_n_s32(int32x4_t a, int32_t b) {
  return vqdmull_high_n_s32(a, b);
}

// CHECK-LABEL: @test_vmlal_high_n_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %c, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I_I]]
int32x4_t test_vmlal_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) {
  return vmlal_high_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_n_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %c, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I_I]]
int64x2_t test_vmlal_high_n_s32(int64x2_t a, int32x4_t b, int32_t c) {
  return vmlal_high_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_n_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %c, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I_I]]
uint32x4_t test_vmlal_high_n_u16(uint32x4_t a, uint16x8_t b, uint16_t c) {
  return vmlal_high_n_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlal_high_n_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %c, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I_I]]
uint64x2_t test_vmlal_high_n_u32(uint64x2_t a, uint32x4_t b, uint32_t c) {
  return vmlal_high_n_u32(a, b, c);
}

// CHECK-LABEL: @test_vqdmlal_high_n_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %c, i32 3
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VQDMLAL5_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   [[VQDMLAL_V6_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL5_I_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V6_I_I]]
int32x4_t test_vqdmlal_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) {
  return vqdmlal_high_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlal_high_n_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %c, i32 1
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VQDMLAL3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   [[VQDMLAL_V4_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL3_I_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V4_I_I]]
int64x2_t test_vqdmlal_high_n_s32(int64x2_t a, int32x4_t b, int32_t c) {
  return vqdmlal_high_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_n_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %c, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I_I]]
int32x4_t test_vmlsl_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) {
  return vmlsl_high_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_n_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %c, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I_I]]
int64x2_t test_vmlsl_high_n_s32(int64x2_t a, int32x4_t b, int32_t c) {
  return vmlsl_high_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_n_u16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %c, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I_I]]
uint32x4_t test_vmlsl_high_n_u16(uint32x4_t a, uint16x8_t b, uint16_t c) {
  return vmlsl_high_n_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_high_n_u32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %c, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   [[SUB_I_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I_I]]
uint64x2_t test_vmlsl_high_n_u32(uint64x2_t a, uint32x4_t b, uint32_t c) {
  return vmlsl_high_n_u32(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsl_high_n_s16(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <4 x i16> [[VECINIT_I_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I_I]], i16 %c, i32 3
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VECINIT3_I_I]] to <8 x i8>
// CHECK:   [[VQDMLAL5_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[VECINIT3_I_I]])
// CHECK:   [[VQDMLSL_V6_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL5_I_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V6_I_I]]
int32x4_t test_vqdmlsl_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) {
  return vqdmlsl_high_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsl_high_n_s32(
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// CHECK:   [[VECINIT_I_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I_I:%.*]] = insertelement <2 x i32> [[VECINIT_I_I]], i32 %c, i32 1
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VECINIT1_I_I]] to <8 x i8>
// CHECK:   [[VQDMLAL3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[VECINIT1_I_I]])
// CHECK:   [[VQDMLSL_V4_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL3_I_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V4_I_I]]
int64x2_t test_vqdmlsl_high_n_s32(int64x2_t a, int32x4_t b, int32_t c) {
  return vqdmlsl_high_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmul_n_f32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x float> undef, float %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x float> [[VECINIT_I]], float %b, i32 1
// CHECK:   [[MUL_I:%.*]] = fmul <2 x float> %a, [[VECINIT1_I]]
// CHECK:   ret <2 x float> [[MUL_I]]
float32x2_t test_vmul_n_f32(float32x2_t a, float32_t b) {
  return vmul_n_f32(a, b);
}

// CHECK-LABEL: @test_vmulq_n_f32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x float> undef, float %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x float> [[VECINIT_I]], float %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x float> [[VECINIT1_I]], float %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x float> [[VECINIT2_I]], float %b, i32 3
// CHECK:   [[MUL_I:%.*]] = fmul <4 x float> %a, [[VECINIT3_I]]
// CHECK:   ret <4 x float> [[MUL_I]]
float32x4_t test_vmulq_n_f32(float32x4_t a, float32_t b) {
  return vmulq_n_f32(a, b);
}

// CHECK-LABEL: @test_vmulq_n_f64(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x double> undef, double %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x double> [[VECINIT_I]], double %b, i32 1
// CHECK:   [[MUL_I:%.*]] = fmul <2 x double> %a, [[VECINIT1_I]]
// CHECK:   ret <2 x double> [[MUL_I]]
float64x2_t test_vmulq_n_f64(float64x2_t a, float64_t b) {
  return vmulq_n_f64(a, b);
}

// CHECK-LABEL: @test_vfma_n_f32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x float> undef, float %n, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x float> [[VECINIT_I]], float %n, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> %b, <2 x float> [[VECINIT1_I]], <2 x float> %a)
// CHECK:   ret <2 x float> [[TMP3]]
float32x2_t test_vfma_n_f32(float32x2_t a, float32x2_t b, float32_t n) {
  return vfma_n_f32(a, b, n);
}

// CHECK-LABEL: @test_vfma_n_f64(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <1 x double> undef, double %n, i32 0
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> [[VECINIT_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> %b, <1 x double> [[VECINIT_I]], <1 x double> %a)
// CHECK:   ret <1 x double> [[TMP3]]
float64x1_t test_vfma_n_f64(float64x1_t a, float64x1_t b, float64_t n) {
  return vfma_n_f64(a, b, n);
}

// CHECK-LABEL: @test_vfmaq_n_f32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x float> undef, float %n, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x float> [[VECINIT_I]], float %n, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x float> [[VECINIT1_I]], float %n, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x float> [[VECINIT2_I]], float %n, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> [[VECINIT3_I]] to <16 x i8>
// CHECK:   [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %b, <4 x float> [[VECINIT3_I]], <4 x float> %a)
// CHECK:   ret <4 x float> [[TMP3]]
float32x4_t test_vfmaq_n_f32(float32x4_t a, float32x4_t b, float32_t n) {
  return vfmaq_n_f32(a, b, n);
}

// CHECK-LABEL: @test_vfms_n_f32(
// CHECK:   [[SUB_I:%.*]] = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x float> undef, float %n, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x float> [[VECINIT_I]], float %n, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> [[SUB_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[SUB_I]], <2 x float> [[VECINIT1_I]], <2 x float> %a)
// CHECK:   ret <2 x float> [[TMP3]]
float32x2_t test_vfms_n_f32(float32x2_t a, float32x2_t b, float32_t n) {
  return vfms_n_f32(a, b, n);
}

// CHECK-LABEL: @test_vfms_n_f64(
// CHECK:   [[SUB_I:%.*]] = fsub <1 x double> <double -0.000000e+00>, %b
// CHECK:   [[VECINIT_I:%.*]] = insertelement <1 x double> undef, double %n, i32 0
// CHECK:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x double> [[SUB_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <1 x double> [[VECINIT_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[SUB_I]], <1 x double> [[VECINIT_I]], <1 x double> %a)
// CHECK:   ret <1 x double> [[TMP3]]
float64x1_t test_vfms_n_f64(float64x1_t a, float64x1_t b, float64_t n) {
  return vfms_n_f64(a, b, n);
}

// CHECK-LABEL: @test_vfmsq_n_f32(
// CHECK:   [[SUB_I:%.*]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %b
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x float> undef, float %n, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x float> [[VECINIT_I]], float %n, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x float> [[VECINIT1_I]], float %n, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x float> [[VECINIT2_I]], float %n, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> [[SUB_I]] to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> [[VECINIT3_I]] to <16 x i8>
// CHECK:   [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[SUB_I]], <4 x float> [[VECINIT3_I]], <4 x float> %a)
// CHECK:   ret <4 x float> [[TMP3]]
float32x4_t test_vfmsq_n_f32(float32x4_t a, float32x4_t b, float32_t n) {
  return vfmsq_n_f32(a, b, n);
}

// CHECK-LABEL: @test_vmul_n_s16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %a, [[VECINIT3_I]]
// CHECK:   ret <4 x i16> [[MUL_I]]
int16x4_t test_vmul_n_s16(int16x4_t a, int16_t b) {
  return vmul_n_s16(a, b);
}

// CHECK-LABEL: @test_vmulq_n_s16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <8 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <8 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <8 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <8 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[VECINIT4_I:%.*]] = insertelement <8 x i16> [[VECINIT3_I]], i16 %b, i32 4
// CHECK:   [[VECINIT5_I:%.*]] = insertelement <8 x i16> [[VECINIT4_I]], i16 %b, i32 5
// CHECK:   [[VECINIT6_I:%.*]] = insertelement <8 x i16> [[VECINIT5_I]], i16 %b, i32 6
// CHECK:   [[VECINIT7_I:%.*]] = insertelement <8 x i16> [[VECINIT6_I]], i16 %b, i32 7
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %a, [[VECINIT7_I]]
// CHECK:   ret <8 x i16> [[MUL_I]]
int16x8_t test_vmulq_n_s16(int16x8_t a, int16_t b) {
  return vmulq_n_s16(a, b);
}

// CHECK-LABEL: @test_vmul_n_s32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %a, [[VECINIT1_I]]
// CHECK:   ret <2 x i32> [[MUL_I]]
int32x2_t test_vmul_n_s32(int32x2_t a, int32_t b) {
  return vmul_n_s32(a, b);
}

// CHECK-LABEL: @test_vmulq_n_s32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i32> [[VECINIT1_I]], i32 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i32> [[VECINIT2_I]], i32 %b, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %a, [[VECINIT3_I]]
// CHECK:   ret <4 x i32> [[MUL_I]]
int32x4_t test_vmulq_n_s32(int32x4_t a, int32_t b) {
  return vmulq_n_s32(a, b);
}

// CHECK-LABEL: @test_vmul_n_u16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %a, [[VECINIT3_I]]
// CHECK:   ret <4 x i16> [[MUL_I]]
uint16x4_t test_vmul_n_u16(uint16x4_t a, uint16_t b) {
  return vmul_n_u16(a, b);
}

// CHECK-LABEL: @test_vmulq_n_u16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <8 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <8 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <8 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <8 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[VECINIT4_I:%.*]] = insertelement <8 x i16> [[VECINIT3_I]], i16 %b, i32 4
// CHECK:   [[VECINIT5_I:%.*]] = insertelement <8 x i16> [[VECINIT4_I]], i16 %b, i32 5
// CHECK:   [[VECINIT6_I:%.*]] = insertelement <8 x i16> [[VECINIT5_I]], i16 %b, i32 6
// CHECK:   [[VECINIT7_I:%.*]] = insertelement <8 x i16> [[VECINIT6_I]], i16 %b, i32 7
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %a, [[VECINIT7_I]]
// CHECK:   ret <8 x i16> [[MUL_I]]
uint16x8_t test_vmulq_n_u16(uint16x8_t a, uint16_t b) {
  return vmulq_n_u16(a, b);
}

// CHECK-LABEL: @test_vmul_n_u32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %a, [[VECINIT1_I]]
// CHECK:   ret <2 x i32> [[MUL_I]]
uint32x2_t test_vmul_n_u32(uint32x2_t a, uint32_t b) {
  return vmul_n_u32(a, b);
}

// CHECK-LABEL: @test_vmulq_n_u32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i32> [[VECINIT1_I]], i32 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i32> [[VECINIT2_I]], i32 %b, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %a, [[VECINIT3_I]]
// CHECK:   ret <4 x i32> [[MUL_I]]
uint32x4_t test_vmulq_n_u32(uint32x4_t a, uint32_t b) {
  return vmulq_n_u32(a, b);
}

// CHECK-LABEL: @test_vmull_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VMULL5_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %a, <4 x i16> [[VECINIT3_I]])
// CHECK:   ret <4 x i32> [[VMULL5_I]]
int32x4_t test_vmull_n_s16(int16x4_t a, int16_t b) {
  return vmull_n_s16(a, b);
}

// CHECK-LABEL: @test_vmull_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VMULL3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %a, <2 x i32> [[VECINIT1_I]])
// CHECK:   ret <2 x i64> [[VMULL3_I]]
int64x2_t test_vmull_n_s32(int32x2_t a, int32_t b) {
  return vmull_n_s32(a, b);
}

// CHECK-LABEL: @test_vmull_n_u16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VMULL5_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %a, <4 x i16> [[VECINIT3_I]])
// CHECK:   ret <4 x i32> [[VMULL5_I]]
uint32x4_t test_vmull_n_u16(uint16x4_t a, uint16_t b) {
  return vmull_n_u16(a, b);
}

// CHECK-LABEL: @test_vmull_n_u32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VMULL3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %a, <2 x i32> [[VECINIT1_I]])
// CHECK:   ret <2 x i64> [[VMULL3_I]]
uint64x2_t test_vmull_n_u32(uint32x2_t a, uint32_t b) {
  return vmull_n_u32(a, b);
}

// CHECK-LABEL: @test_vqdmull_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VQDMULL_V5_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %a, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[VQDMULL_V6_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V5_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULL_V5_I]]
int32x4_t test_vqdmull_n_s16(int16x4_t a, int16_t b) {
  return vqdmull_n_s16(a, b);
}

// CHECK-LABEL: @test_vqdmull_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VQDMULL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %a, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[VQDMULL_V4_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V3_I]] to <16 x i8>
// CHECK:   ret <2 x i64> [[VQDMULL_V3_I]]
int64x2_t test_vqdmull_n_s32(int32x2_t a, int32_t b) {
  return vqdmull_n_s32(a, b);
}

// CHECK-LABEL: @test_vqdmulh_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VQDMULH_V5_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> %a, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[VQDMULH_V6_I:%.*]] = bitcast <4 x i16> [[VQDMULH_V5_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQDMULH_V5_I]]
int16x4_t test_vqdmulh_n_s16(int16x4_t a, int16_t b) {
  return vqdmulh_n_s16(a, b);
}

// CHECK-LABEL: @test_vqdmulhq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <8 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <8 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <8 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <8 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[VECINIT4_I:%.*]] = insertelement <8 x i16> [[VECINIT3_I]], i16 %b, i32 4
// CHECK:   [[VECINIT5_I:%.*]] = insertelement <8 x i16> [[VECINIT4_I]], i16 %b, i32 5
// CHECK:   [[VECINIT6_I:%.*]] = insertelement <8 x i16> [[VECINIT5_I]], i16 %b, i32 6
// CHECK:   [[VECINIT7_I:%.*]] = insertelement <8 x i16> [[VECINIT6_I]], i16 %b, i32 7
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[VECINIT7_I]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V9_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16> %a, <8 x i16> [[VECINIT7_I]])
// CHECK:   [[VQDMULHQ_V10_I:%.*]] = bitcast <8 x i16> [[VQDMULHQ_V9_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQDMULHQ_V9_I]]
int16x8_t test_vqdmulhq_n_s16(int16x8_t a, int16_t b) {
  return vqdmulhq_n_s16(a, b);
}

// CHECK-LABEL: @test_vqdmulh_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VQDMULH_V3_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> %a, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[VQDMULH_V4_I:%.*]] = bitcast <2 x i32> [[VQDMULH_V3_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQDMULH_V3_I]]
int32x2_t test_vqdmulh_n_s32(int32x2_t a, int32_t b) {
  return vqdmulh_n_s32(a, b);
}

// CHECK-LABEL: @test_vqdmulhq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i32> [[VECINIT1_I]], i32 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i32> [[VECINIT2_I]], i32 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[VECINIT3_I]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V5_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> %a, <4 x i32> [[VECINIT3_I]])
// CHECK:   [[VQDMULHQ_V6_I:%.*]] = bitcast <4 x i32> [[VQDMULHQ_V5_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULHQ_V5_I]]
int32x4_t test_vqdmulhq_n_s32(int32x4_t a, int32_t b) {
  return vqdmulhq_n_s32(a, b);
}

// CHECK-LABEL: @test_vqrdmulh_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VQRDMULH_V5_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %a, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[VQRDMULH_V6_I:%.*]] = bitcast <4 x i16> [[VQRDMULH_V5_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQRDMULH_V5_I]]
int16x4_t test_vqrdmulh_n_s16(int16x4_t a, int16_t b) {
  return vqrdmulh_n_s16(a, b);
}

// CHECK-LABEL: @test_vqrdmulhq_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <8 x i16> undef, i16 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <8 x i16> [[VECINIT_I]], i16 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <8 x i16> [[VECINIT1_I]], i16 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <8 x i16> [[VECINIT2_I]], i16 %b, i32 3
// CHECK:   [[VECINIT4_I:%.*]] = insertelement <8 x i16> [[VECINIT3_I]], i16 %b, i32 4
// CHECK:   [[VECINIT5_I:%.*]] = insertelement <8 x i16> [[VECINIT4_I]], i16 %b, i32 5
// CHECK:   [[VECINIT6_I:%.*]] = insertelement <8 x i16> [[VECINIT5_I]], i16 %b, i32 6
// CHECK:   [[VECINIT7_I:%.*]] = insertelement <8 x i16> [[VECINIT6_I]], i16 %b, i32 7
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[VECINIT7_I]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V9_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %a, <8 x i16> [[VECINIT7_I]])
// CHECK:   [[VQRDMULHQ_V10_I:%.*]] = bitcast <8 x i16> [[VQRDMULHQ_V9_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQRDMULHQ_V9_I]]
int16x8_t test_vqrdmulhq_n_s16(int16x8_t a, int16_t b) {
  return vqrdmulhq_n_s16(a, b);
}

// CHECK-LABEL: @test_vqrdmulh_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VQRDMULH_V3_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %a, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[VQRDMULH_V4_I:%.*]] = bitcast <2 x i32> [[VQRDMULH_V3_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQRDMULH_V3_I]]
int32x2_t test_vqrdmulh_n_s32(int32x2_t a, int32_t b) {
  return vqrdmulh_n_s32(a, b);
}

// CHECK-LABEL: @test_vqrdmulhq_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i32> undef, i32 %b, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i32> [[VECINIT_I]], i32 %b, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i32> [[VECINIT1_I]], i32 %b, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i32> [[VECINIT2_I]], i32 %b, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[VECINIT3_I]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V5_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %a, <4 x i32> [[VECINIT3_I]])
// CHECK:   [[VQRDMULHQ_V6_I:%.*]] = bitcast <4 x i32> [[VQRDMULHQ_V5_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQRDMULHQ_V5_I]]
int32x4_t test_vqrdmulhq_n_s32(int32x4_t a, int32_t b) {
  return vqrdmulhq_n_s32(a, b);
}

// CHECK-LABEL: @test_vmla_n_s16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %b, [[VECINIT3_I]]
// CHECK:   [[ADD_I:%.*]] = add <4 x i16> %a, [[MUL_I]]
// CHECK:   ret <4 x i16> [[ADD_I]]
int16x4_t test_vmla_n_s16(int16x4_t a, int16x4_t b, int16_t c) {
  return vmla_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlaq_n_s16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <8 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <8 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <8 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <8 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[VECINIT4_I:%.*]] = insertelement <8 x i16> [[VECINIT3_I]], i16 %c, i32 4
// CHECK:   [[VECINIT5_I:%.*]] = insertelement <8 x i16> [[VECINIT4_I]], i16 %c, i32 5
// CHECK:   [[VECINIT6_I:%.*]] = insertelement <8 x i16> [[VECINIT5_I]], i16 %c, i32 6
// CHECK:   [[VECINIT7_I:%.*]] = insertelement <8 x i16> [[VECINIT6_I]], i16 %c, i32 7
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %b, [[VECINIT7_I]]
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[MUL_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
int16x8_t test_vmlaq_n_s16(int16x8_t a, int16x8_t b, int16_t c) {
  return vmlaq_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vmla_n_s32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %b, [[VECINIT1_I]]
// CHECK:   [[ADD_I:%.*]] = add <2 x i32> %a, [[MUL_I]]
// CHECK:   ret <2 x i32> [[ADD_I]]
int32x2_t test_vmla_n_s32(int32x2_t a, int32x2_t b, int32_t c) {
  return vmla_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlaq_n_s32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i32> [[VECINIT1_I]], i32 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i32> [[VECINIT2_I]], i32 %c, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %b, [[VECINIT3_I]]
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[MUL_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vmlaq_n_s32(int32x4_t a, int32x4_t b, int32_t c) {
  return vmlaq_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmla_n_u16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %b, [[VECINIT3_I]]
// CHECK:   [[ADD_I:%.*]] = add <4 x i16> %a, [[MUL_I]]
// CHECK:   ret <4 x i16> [[ADD_I]]
uint16x4_t test_vmla_n_u16(uint16x4_t a, uint16x4_t b, uint16_t c) {
  return vmla_n_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlaq_n_u16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <8 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <8 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <8 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <8 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[VECINIT4_I:%.*]] = insertelement <8 x i16> [[VECINIT3_I]], i16 %c, i32 4
// CHECK:   [[VECINIT5_I:%.*]] = insertelement <8 x i16> [[VECINIT4_I]], i16 %c, i32 5
// CHECK:   [[VECINIT6_I:%.*]] = insertelement <8 x i16> [[VECINIT5_I]], i16 %c, i32 6
// CHECK:   [[VECINIT7_I:%.*]] = insertelement <8 x i16> [[VECINIT6_I]], i16 %c, i32 7
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %b, [[VECINIT7_I]]
// CHECK:   [[ADD_I:%.*]] = add <8 x i16> %a, [[MUL_I]]
// CHECK:   ret <8 x i16> [[ADD_I]]
uint16x8_t test_vmlaq_n_u16(uint16x8_t a, uint16x8_t b, uint16_t c) {
  return vmlaq_n_u16(a, b, c);
}

// CHECK-LABEL: @test_vmla_n_u32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %b, [[VECINIT1_I]]
// CHECK:   [[ADD_I:%.*]] = add <2 x i32> %a, [[MUL_I]]
// CHECK:   ret <2 x i32> [[ADD_I]]
uint32x2_t test_vmla_n_u32(uint32x2_t a, uint32x2_t b, uint32_t c) {
  return vmla_n_u32(a, b, c);
}

// CHECK-LABEL: @test_vmlaq_n_u32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i32> [[VECINIT1_I]], i32 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i32> [[VECINIT2_I]], i32 %c, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %b, [[VECINIT3_I]]
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[MUL_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vmlaq_n_u32(uint32x4_t a, uint32x4_t b, uint32_t c) {
  return vmlaq_n_u32(a, b, c);
}

// CHECK-LABEL: @test_vmlal_n_s16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
int32x4_t test_vmlal_n_s16(int32x4_t a, int16x4_t b, int16_t c) {
  return vmlal_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlal_n_s32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
int64x2_t test_vmlal_n_s32(int64x2_t a, int32x2_t b, int32_t c) {
  return vmlal_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlal_n_u16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I]]
// CHECK:   ret <4 x i32> [[ADD_I]]
uint32x4_t test_vmlal_n_u16(uint32x4_t a, uint16x4_t b, uint16_t c) {
  return vmlal_n_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlal_n_u32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I]]
// CHECK:   ret <2 x i64> [[ADD_I]]
uint64x2_t test_vmlal_n_u32(uint64x2_t a, uint32x2_t b, uint32_t c) {
  return vmlal_n_u32(a, b, c);
}

// CHECK-LABEL: @test_vqdmlal_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VQDMLAL5_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[VQDMLAL_V6_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL5_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V6_I]]
int32x4_t test_vqdmlal_n_s16(int32x4_t a, int16x4_t b, int16_t c) {
  return vqdmlal_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlal_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VQDMLAL3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[VQDMLAL_V4_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL3_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V4_I]]
int64x2_t test_vqdmlal_n_s32(int64x2_t a, int32x2_t b, int32_t c) {
  return vqdmlal_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmls_n_s16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %b, [[VECINIT3_I]]
// CHECK:   [[SUB_I:%.*]] = sub <4 x i16> %a, [[MUL_I]]
// CHECK:   ret <4 x i16> [[SUB_I]]
int16x4_t test_vmls_n_s16(int16x4_t a, int16x4_t b, int16_t c) {
  return vmls_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlsq_n_s16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <8 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <8 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <8 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <8 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[VECINIT4_I:%.*]] = insertelement <8 x i16> [[VECINIT3_I]], i16 %c, i32 4
// CHECK:   [[VECINIT5_I:%.*]] = insertelement <8 x i16> [[VECINIT4_I]], i16 %c, i32 5
// CHECK:   [[VECINIT6_I:%.*]] = insertelement <8 x i16> [[VECINIT5_I]], i16 %c, i32 6
// CHECK:   [[VECINIT7_I:%.*]] = insertelement <8 x i16> [[VECINIT6_I]], i16 %c, i32 7
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %b, [[VECINIT7_I]]
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[MUL_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
int16x8_t test_vmlsq_n_s16(int16x8_t a, int16x8_t b, int16_t c) {
  return vmlsq_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vmls_n_s32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %b, [[VECINIT1_I]]
// CHECK:   [[SUB_I:%.*]] = sub <2 x i32> %a, [[MUL_I]]
// CHECK:   ret <2 x i32> [[SUB_I]]
int32x2_t test_vmls_n_s32(int32x2_t a, int32x2_t b, int32_t c) {
  return vmls_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlsq_n_s32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i32> [[VECINIT1_I]], i32 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i32> [[VECINIT2_I]], i32 %c, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %b, [[VECINIT3_I]]
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[MUL_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vmlsq_n_s32(int32x4_t a, int32x4_t b, int32_t c) {
  return vmlsq_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmls_n_u16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i16> %b, [[VECINIT3_I]]
// CHECK:   [[SUB_I:%.*]] = sub <4 x i16> %a, [[MUL_I]]
// CHECK:   ret <4 x i16> [[SUB_I]]
uint16x4_t test_vmls_n_u16(uint16x4_t a, uint16x4_t b, uint16_t c) {
  return vmls_n_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlsq_n_u16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <8 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <8 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <8 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <8 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[VECINIT4_I:%.*]] = insertelement <8 x i16> [[VECINIT3_I]], i16 %c, i32 4
// CHECK:   [[VECINIT5_I:%.*]] = insertelement <8 x i16> [[VECINIT4_I]], i16 %c, i32 5
// CHECK:   [[VECINIT6_I:%.*]] = insertelement <8 x i16> [[VECINIT5_I]], i16 %c, i32 6
// CHECK:   [[VECINIT7_I:%.*]] = insertelement <8 x i16> [[VECINIT6_I]], i16 %c, i32 7
// CHECK:   [[MUL_I:%.*]] = mul <8 x i16> %b, [[VECINIT7_I]]
// CHECK:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[MUL_I]]
// CHECK:   ret <8 x i16> [[SUB_I]]
uint16x8_t test_vmlsq_n_u16(uint16x8_t a, uint16x8_t b, uint16_t c) {
  return vmlsq_n_u16(a, b, c);
}

// CHECK-LABEL: @test_vmls_n_u32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[MUL_I:%.*]] = mul <2 x i32> %b, [[VECINIT1_I]]
// CHECK:   [[SUB_I:%.*]] = sub <2 x i32> %a, [[MUL_I]]
// CHECK:   ret <2 x i32> [[SUB_I]]
uint32x2_t test_vmls_n_u32(uint32x2_t a, uint32x2_t b, uint32_t c) {
  return vmls_n_u32(a, b, c);
}

// CHECK-LABEL: @test_vmlsq_n_u32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i32> [[VECINIT1_I]], i32 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i32> [[VECINIT2_I]], i32 %c, i32 3
// CHECK:   [[MUL_I:%.*]] = mul <4 x i32> %b, [[VECINIT3_I]]
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[MUL_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vmlsq_n_u32(uint32x4_t a, uint32x4_t b, uint32_t c) {
  return vmlsq_n_u32(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_n_s16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
int32x4_t test_vmlsl_n_s16(int32x4_t a, int16x4_t b, int16_t c) {
  return vmlsl_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_n_s32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I]]
int64x2_t test_vmlsl_n_s32(int64x2_t a, int32x2_t b, int32_t c) {
  return vmlsl_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_n_u16(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I]]
// CHECK:   ret <4 x i32> [[SUB_I]]
uint32x4_t test_vmlsl_n_u16(uint32x4_t a, uint16x4_t b, uint16_t c) {
  return vmlsl_n_u16(a, b, c);
}

// CHECK-LABEL: @test_vmlsl_n_u32(
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I]]
// CHECK:   ret <2 x i64> [[SUB_I]]
uint64x2_t test_vmlsl_n_u32(uint64x2_t a, uint32x2_t b, uint32_t c) {
  return vmlsl_n_u32(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsl_n_s16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x i16> undef, i16 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x i16> [[VECINIT_I]], i16 %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x i16> [[VECINIT1_I]], i16 %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x i16> [[VECINIT2_I]], i16 %c, i32 3
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[VECINIT3_I]] to <8 x i8>
// CHECK:   [[VQDMLAL5_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[VECINIT3_I]])
// CHECK:   [[VQDMLSL_V6_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL5_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V6_I]]
int32x4_t test_vqdmlsl_n_s16(int32x4_t a, int16x4_t b, int16_t c) {
  return vqdmlsl_n_s16(a, b, c);
}

// CHECK-LABEL: @test_vqdmlsl_n_s32(
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i32> undef, i32 %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i32> [[VECINIT_I]], i32 %c, i32 1
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[VQDMLAL3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[VECINIT1_I]])
// CHECK:   [[VQDMLSL_V4_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL3_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V4_I]]
int64x2_t test_vqdmlsl_n_s32(int64x2_t a, int32x2_t b, int32_t c) {
  return vqdmlsl_n_s32(a, b, c);
}

// CHECK-LABEL: @test_vmla_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[ADD]]
uint16x4_t test_vmla_lane_u16_0(uint16x4_t a, uint16x4_t b, uint16x4_t v) {
  return vmla_lane_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlaq_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[ADD]]
uint16x8_t test_vmlaq_lane_u16_0(uint16x8_t a, uint16x8_t b, uint16x4_t v) {
  return vmlaq_lane_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmla_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[ADD]]
uint32x2_t test_vmla_lane_u32_0(uint32x2_t a, uint32x2_t b, uint32x2_t v) {
  return vmla_lane_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlaq_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[ADD]]
uint32x4_t test_vmlaq_lane_u32_0(uint32x4_t a, uint32x4_t b, uint32x2_t v) {
  return vmlaq_lane_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmla_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[ADD]]
uint16x4_t test_vmla_laneq_u16_0(uint16x4_t a, uint16x4_t b, uint16x8_t v) {
  return vmla_laneq_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlaq_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[ADD]]
uint16x8_t test_vmlaq_laneq_u16_0(uint16x8_t a, uint16x8_t b, uint16x8_t v) {
  return vmlaq_laneq_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmla_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[ADD]]
uint32x2_t test_vmla_laneq_u32_0(uint32x2_t a, uint32x2_t b, uint32x4_t v) {
  return vmla_laneq_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlaq_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[ADD]]
uint32x4_t test_vmlaq_laneq_u32_0(uint32x4_t a, uint32x4_t b, uint32x4_t v) {
  return vmlaq_laneq_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlal_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_laneq_s16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vqdmlal_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlal_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_laneq_s32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vqdmlal_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlal_high_laneq_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_high_laneq_s16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vqdmlal_high_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlal_high_laneq_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_high_laneq_s32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vqdmlal_high_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmls_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[SUB]]
uint16x4_t test_vmls_lane_u16_0(uint16x4_t a, uint16x4_t b, uint16x4_t v) {
  return vmls_lane_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsq_lane_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[SUB]]
uint16x8_t test_vmlsq_lane_u16_0(uint16x8_t a, uint16x8_t b, uint16x4_t v) {
  return vmlsq_lane_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmls_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[SUB]]
uint32x2_t test_vmls_lane_u32_0(uint32x2_t a, uint32x2_t b, uint32x2_t v) {
  return vmls_lane_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsq_lane_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[SUB]]
uint32x4_t test_vmlsq_lane_u32_0(uint32x4_t a, uint32x4_t b, uint32x2_t v) {
  return vmlsq_lane_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmls_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[SUB]]
uint16x4_t test_vmls_laneq_u16_0(uint16x4_t a, uint16x4_t b, uint16x8_t v) {
  return vmls_laneq_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsq_laneq_u16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[SUB]]
uint16x8_t test_vmlsq_laneq_u16_0(uint16x8_t a, uint16x8_t b, uint16x8_t v) {
  return vmlsq_laneq_u16(a, b, v, 0);
}

// CHECK-LABEL: @test_vmls_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[SUB]]
uint32x2_t test_vmls_laneq_u32_0(uint32x2_t a, uint32x2_t b, uint32x4_t v) {
  return vmls_laneq_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vmlsq_laneq_u32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[SUB]]
uint32x4_t test_vmlsq_laneq_u32_0(uint32x4_t a, uint32x4_t b, uint32x4_t v) {
  return vmlsq_laneq_u32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlsl_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_laneq_s16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vqdmlsl_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlsl_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_laneq_s32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vqdmlsl_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlsl_high_laneq_s16_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_high_laneq_s16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vqdmlsl_high_laneq_s16(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmlsl_high_laneq_s32_0(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_high_laneq_s32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vqdmlsl_high_laneq_s32(a, b, v, 0);
}

// CHECK-LABEL: @test_vqdmulh_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQDMULH_V2_I]]
int16x4_t test_vqdmulh_laneq_s16_0(int16x4_t a, int16x8_t v) {
  return vqdmulh_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqdmulhq_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16> %a, <8 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQDMULHQ_V2_I]]
int16x8_t test_vqdmulhq_laneq_s16_0(int16x8_t a, int16x8_t v) {
  return vqdmulhq_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqdmulh_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQDMULH_V2_I]]
int32x2_t test_vqdmulh_laneq_s32_0(int32x2_t a, int32x4_t v) {
  return vqdmulh_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqdmulhq_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> %a, <4 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULHQ_V2_I]]
int32x4_t test_vqdmulhq_laneq_s32_0(int32x4_t a, int32x4_t v) {
  return vqdmulhq_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqrdmulh_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQRDMULH_V2_I]]
int16x4_t test_vqrdmulh_laneq_s16_0(int16x4_t a, int16x8_t v) {
  return vqrdmulh_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqrdmulhq_laneq_s16_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %a, <8 x i16> [[SHUFFLE]])
// CHECK:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQRDMULHQ_V2_I]]
int16x8_t test_vqrdmulhq_laneq_s16_0(int16x8_t a, int16x8_t v) {
  return vqrdmulhq_laneq_s16(a, v, 0);
}

// CHECK-LABEL: @test_vqrdmulh_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQRDMULH_V2_I]]
int32x2_t test_vqrdmulh_laneq_s32_0(int32x2_t a, int32x4_t v) {
  return vqrdmulh_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vqrdmulhq_laneq_s32_0(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> zeroinitializer
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %a, <4 x i32> [[SHUFFLE]])
// CHECK:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQRDMULHQ_V2_I]]
int32x4_t test_vqrdmulhq_laneq_s32_0(int32x4_t a, int32x4_t v) {
  return vqrdmulhq_laneq_s32(a, v, 0);
}

// CHECK-LABEL: @test_vmla_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[ADD]]
uint16x4_t test_vmla_lane_u16(uint16x4_t a, uint16x4_t b, uint16x4_t v) {
  return vmla_lane_u16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlaq_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[ADD]]
uint16x8_t test_vmlaq_lane_u16(uint16x8_t a, uint16x8_t b, uint16x4_t v) {
  return vmlaq_lane_u16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmla_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[ADD]]
uint32x2_t test_vmla_lane_u32(uint32x2_t a, uint32x2_t b, uint32x2_t v) {
  return vmla_lane_u32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlaq_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[ADD]]
uint32x4_t test_vmlaq_lane_u32(uint32x4_t a, uint32x4_t b, uint32x2_t v) {
  return vmlaq_lane_u32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmla_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[ADD]]
uint16x4_t test_vmla_laneq_u16(uint16x4_t a, uint16x4_t b, uint16x8_t v) {
  return vmla_laneq_u16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlaq_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[ADD]]
uint16x8_t test_vmlaq_laneq_u16(uint16x8_t a, uint16x8_t b, uint16x8_t v) {
  return vmlaq_laneq_u16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmla_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[ADD]]
uint32x2_t test_vmla_laneq_u32(uint32x2_t a, uint32x2_t b, uint32x4_t v) {
  return vmla_laneq_u32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlaq_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = add <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[ADD]]
uint32x4_t test_vmlaq_laneq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t v) {
  return vmlaq_laneq_u32(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmlal_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_laneq_s16(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vqdmlal_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vqdmlal_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_laneq_s32(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vqdmlal_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmlal_high_laneq_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLAL_V3_I]]
int32x4_t test_vqdmlal_high_laneq_s16(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vqdmlal_high_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vqdmlal_high_laneq_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLAL_V3_I]]
int64x2_t test_vqdmlal_high_laneq_s32(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vqdmlal_high_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmls_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[SUB]]
uint16x4_t test_vmls_lane_u16(uint16x4_t a, uint16x4_t b, uint16x4_t v) {
  return vmls_lane_u16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsq_lane_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i16> %v, <4 x i16> %v, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[SUB]]
uint16x8_t test_vmlsq_lane_u16(uint16x8_t a, uint16x8_t b, uint16x4_t v) {
  return vmlsq_lane_u16(a, b, v, 3);
}

// CHECK-LABEL: @test_vmls_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[SUB]]
uint32x2_t test_vmls_lane_u32(uint32x2_t a, uint32x2_t b, uint32x2_t v) {
  return vmls_lane_u32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmlsq_lane_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x i32> %v, <2 x i32> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[SUB]]
uint32x4_t test_vmlsq_lane_u32(uint32x4_t a, uint32x4_t b, uint32x2_t v) {
  return vmlsq_lane_u32(a, b, v, 1);
}

// CHECK-LABEL: @test_vmls_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <4 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i16> %a, [[MUL]]
// CHECK:   ret <4 x i16> [[SUB]]
uint16x4_t test_vmls_laneq_u16(uint16x4_t a, uint16x4_t b, uint16x8_t v) {
  return vmls_laneq_u16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmlsq_laneq_u16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[MUL:%.*]] = mul <8 x i16> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <8 x i16> %a, [[MUL]]
// CHECK:   ret <8 x i16> [[SUB]]
uint16x8_t test_vmlsq_laneq_u16(uint16x8_t a, uint16x8_t b, uint16x8_t v) {
  return vmlsq_laneq_u16(a, b, v, 7);
}

// CHECK-LABEL: @test_vmls_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <2 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <2 x i32> %a, [[MUL]]
// CHECK:   ret <2 x i32> [[SUB]]
uint32x2_t test_vmls_laneq_u32(uint32x2_t a, uint32x2_t b, uint32x4_t v) {
  return vmls_laneq_u32(a, b, v, 3);
}

// CHECK-LABEL: @test_vmlsq_laneq_u32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = mul <4 x i32> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = sub <4 x i32> %a, [[MUL]]
// CHECK:   ret <4 x i32> [[SUB]]
uint32x4_t test_vmlsq_laneq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t v) {
  return vmlsq_laneq_u32(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmlsl_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_laneq_s16(int32x4_t a, int16x4_t b, int16x8_t v) {
  return vqdmlsl_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vqdmlsl_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_laneq_s32(int64x2_t a, int32x2_t b, int32x4_t v) {
  return vqdmlsl_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmlsl_high_laneq_s16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I]], <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I]])
// CHECK:   ret <4 x i32> [[VQDMLSL_V3_I]]
int32x4_t test_vqdmlsl_high_laneq_s16(int32x4_t a, int16x8_t b, int16x8_t v) {
  return vqdmlsl_high_laneq_s16(a, b, v, 7);
}

// CHECK-LABEL: @test_vqdmlsl_high_laneq_s32(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I]], <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I]])
// CHECK:   ret <2 x i64> [[VQDMLSL_V3_I]]
int64x2_t test_vqdmlsl_high_laneq_s32(int64x2_t a, int32x4_t b, int32x4_t v) {
  return vqdmlsl_high_laneq_s32(a, b, v, 3);
}

// CHECK-LABEL: @test_vqdmulh_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQDMULH_V2_I]]
int16x4_t test_vqdmulh_laneq_s16(int16x4_t a, int16x8_t v) {
  return vqdmulh_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vqdmulhq_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16> %a, <8 x i16> [[SHUFFLE]])
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQDMULHQ_V2_I]]
int16x8_t test_vqdmulhq_laneq_s16(int16x8_t a, int16x8_t v) {
  return vqdmulhq_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vqdmulh_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQDMULH_V2_I]]
int32x2_t test_vqdmulh_laneq_s32(int32x2_t a, int32x4_t v) {
  return vqdmulh_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vqdmulhq_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> %a, <4 x i32> [[SHUFFLE]])
// CHECK:   [[VQDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQDMULHQ_V2_I]]
int32x4_t test_vqdmulhq_laneq_s32(int32x4_t a, int32x4_t v) {
  return vqdmulhq_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vqrdmulh_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> %a, <4 x i16> [[SHUFFLE]])
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <4 x i16> [[VQRDMULH_V2_I]]
int16x4_t test_vqrdmulh_laneq_s16(int16x4_t a, int16x8_t v) {
  return vqrdmulh_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vqrdmulhq_laneq_s16(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <8 x i16> %v, <8 x i16> %v, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i16> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> %a, <8 x i16> [[SHUFFLE]])
// CHECK:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <8 x i16> [[VQRDMULHQ_V2_I]]
int16x8_t test_vqrdmulhq_laneq_s16(int16x8_t a, int16x8_t v) {
  return vqrdmulhq_laneq_s16(a, v, 7);
}

// CHECK-LABEL: @test_vqrdmulh_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK:   [[VQRDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> %a, <2 x i32> [[SHUFFLE]])
// CHECK:   [[VQRDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQRDMULH_V2_I]] to <8 x i8>
// CHECK:   ret <2 x i32> [[VQRDMULH_V2_I]]
int32x2_t test_vqrdmulh_laneq_s32(int32x2_t a, int32x4_t v) {
  return vqrdmulh_laneq_s32(a, v, 3);
}

// CHECK-LABEL: @test_vqrdmulhq_laneq_s32(
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x i32> %v, <4 x i32> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK:   [[VQRDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %a, <4 x i32> [[SHUFFLE]])
// CHECK:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRDMULHQ_V2_I]] to <16 x i8>
// CHECK:   ret <4 x i32> [[VQRDMULHQ_V2_I]]
int32x4_t test_vqrdmulhq_laneq_s32(int32x4_t a, int32x4_t v) {
  return vqrdmulhq_laneq_s32(a, v, 3);
}
