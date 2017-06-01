// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -target-feature +neon -ffreestanding -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s --check-prefix CHECK-BE

#include <arm_neon.h>

int8_t test_vdupb_lane_s8(int8x8_t src) {
  return vdupb_lane_s8(src, 2);
  // CHECK-LABEL: @test_vdupb_lane_s8
  // CHECK: extractelement <8 x i8> %src, i32 2

  // CHECK-BE-LABEL: @test_vdupb_lane_s8
  // CHECK-BE: [[REV:%.*]] = shufflevector <8 x i8> {{.*}}, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  // CHECK-BE: extractelement <8 x i8> [[REV]], i32 2
}

uint8_t test_vdupb_lane_u8(uint8x8_t src) {
  return vdupb_lane_u8(src, 2);
  // CHECK-LABEL: @test_vdupb_lane_u8
  // CHECK: extractelement <8 x i8> %src, i32 2

  // CHECK-BE-LABEL: @test_vdupb_lane_u8
  // CHECK-BE: [[REV:%.*]] = shufflevector <8 x i8> {{.*}}, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  // CHECK-BE: extractelement <8 x i8> [[REV]], i32 2
}

int16_t test_vduph_lane_s16(int16x4_t src) {
  return vduph_lane_s16(src, 2);
  // CHECK-LABEL: @test_vduph_lane_s16
  // CHECK: [[TMP1:%.*]] = bitcast <4 x i16> %src to [[TYPE:.*]]
  // CHECK: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <4 x i16>
  // CHECK: extractelement <4 x i16> [[TMP2]], i32 2

  // CHECK-BE-LABEL: @test_vduph_lane_s16
  // CHECK-BE: [[REV:%.*]] = shufflevector <4 x i16> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK-BE: [[TMP1:%.*]] = bitcast <4 x i16> [[REV]] to [[TYPE:.*]]
  // CHECK-BE: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <4 x i16>
  // CHECK-BE: extractelement <4 x i16> [[TMP2]], i32 2
}

uint16_t test_vduph_lane_u16(uint16x4_t src) {
  return vduph_lane_u16(src, 2);
  // CHECK-LABEL: @test_vduph_lane_u16
  // CHECK: [[TMP1:%.*]] = bitcast <4 x i16> %src to [[TYPE:.*]]
  // CHECK: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <4 x i16>
  // CHECK: extractelement <4 x i16> [[TMP2]], i32 2

  // CHECK-BE-LABEL: @test_vduph_lane_u16
  // CHECK-BE: [[REV:%.*]] = shufflevector <4 x i16> {{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK-BE: [[TMP1:%.*]] = bitcast <4 x i16> [[REV]] to [[TYPE:.*]]
  // CHECK-BE: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <4 x i16>
  // CHECK-BE: extractelement <4 x i16> [[TMP2]], i32 2
}

int32_t test_vdups_lane_s32(int32x2_t src) {
  return vdups_lane_s32(src, 0);
  // CHECK-LABEL: @test_vdups_lane_s32
  // CHECK: [[TMP1:%.*]] = bitcast <2 x i32> %src to [[TYPE:.*]]
  // CHECK: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <2 x i32>
  // CHECK: extractelement <2 x i32> [[TMP2]], i32 0

  // CHECK-BE-LABEL: @test_vdups_lane_s32
  // CHECK-BE: [[REV:%.*]] = shufflevector <2 x i32> {{.*}}, <2 x i32> <i32 1, i32 0>
  // CHECK-BE: [[TMP1:%.*]] = bitcast <2 x i32> [[REV]] to [[TYPE:.*]]
  // CHECK-BE: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <2 x i32>
  // CHECK-BE: extractelement <2 x i32> [[TMP2]], i32 0
}

uint32_t test_vdups_lane_u32(uint32x2_t src) {
  return vdups_lane_u32(src, 0);
  // CHECK-LABEL: @test_vdups_lane_u32
  // CHECK: [[TMP1:%.*]] = bitcast <2 x i32> %src to [[TYPE:.*]]
  // CHECK: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <2 x i32>
  // CHECK: extractelement <2 x i32> [[TMP2]], i32 0

  // CHECK-BE-LABEL: @test_vdups_lane_u32
  // CHECK-BE: [[REV:%.*]] = shufflevector <2 x i32> {{.*}}, <2 x i32> <i32 1, i32 0>
  // CHECK-BE: [[TMP1:%.*]] = bitcast <2 x i32> [[REV]] to [[TYPE:.*]]
  // CHECK-BE: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <2 x i32>
  // CHECK-BE: extractelement <2 x i32> [[TMP2]], i32 0
}

float32_t test_vdups_lane_f32(float32x2_t src) {
  return vdups_lane_f32(src, 0);
  // CHECK-LABEL: @test_vdups_lane_f32
  // CHECK: [[TMP1:%.*]] = bitcast <2 x float> %src to [[TYPE:.*]]
  // CHECK: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <2 x float>
  // CHECK: extractelement <2 x float> [[TMP2]], i32 0

  // CHECK-BE-LABEL: @test_vdups_lane_f32
  // CHECK-BE: [[REV:%.*]] = shufflevector <2 x float> {{.*}}, <2 x i32> <i32 1, i32 0>
  // CHECK-BE: [[TMP1:%.*]] = bitcast <2 x float> [[REV]] to [[TYPE:.*]]
  // CHECK-BE: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <2 x float>
  // CHECK-BE: extractelement <2 x float> [[TMP2]], i32 0
}

int64_t test_vdupd_lane_s64(int64x1_t src) {
  return vdupd_lane_s64(src, 0);
  // CHECK-LABEL: @test_vdupd_lane_s64
  // CHECK: [[TMP1:%.*]] = bitcast <1 x i64> %src to [[TYPE:.*]]
  // CHECK: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <1 x i64>
  // CHECK: extractelement <1 x i64> [[TMP2]], i32 0

  // CHECK-BE-LABEL: @test_vdupd_lane_s64
  // CHECK-BE: extractelement <1 x i64> {{.*}}, i32 0
}

uint64_t test_vdupd_lane_u64(uint64x1_t src) {
  return vdupd_lane_u64(src, 0);
  // CHECK-LABEL: @test_vdupd_lane_u64
  // CHECK: [[TMP1:%.*]] = bitcast <1 x i64> %src to [[TYPE:.*]]
  // CHECK: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <1 x i64>
  // CHECK: extractelement <1 x i64> [[TMP2]], i32 0

  // CHECK-BE-LABEL: @test_vdupd_lane_u64
  // CHECK-BE: extractelement <1 x i64> {{.*}}, i32 0
}

float64_t test_vdupd_lane_f64(float64x1_t src) {
  return vdupd_lane_f64(src, 0);
  // CHECK-LABEL: @test_vdupd_lane_f64
  // CHECK: [[TMP1:%.*]] = bitcast <1 x double> %src to [[TYPE:.*]]
  // CHECK: [[TMP2:%.*]] = bitcast [[TYPE]] [[TMP1]] to <1 x double>
  // CHECK: extractelement <1 x double> [[TMP2]], i32 0

  // CHECK-BE-LABEL: @test_vdupd_lane_f64
  // CHECK-BE: extractelement <1 x double> {{.*}}, i32 0
}
