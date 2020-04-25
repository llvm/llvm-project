// RUN: %clang_cc1 -triple armv8.6a-arm-none-eabi -target-feature +neon -target-feature +fullfp16 -target-feature +i8mm \
// RUN: -fallow-half-arguments-and-returns -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg -sroa \
// RUN: | FileCheck %s

// REQUIRES: arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: test_vmmlaq_s32
// CHECK: [[VAL:%.*]] = call <4 x i32> @llvm.arm.neon.smmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b)
// CHECK: ret <4 x i32> [[VAL]]
int32x4_t test_vmmlaq_s32(int32x4_t r, int8x16_t a, int8x16_t b) {
  return vmmlaq_s32(r, a, b);
}

// CHECK-LABEL: test_vmmlaq_u32
// CHECK: [[VAL:%.*]] = call <4 x i32> @llvm.arm.neon.ummla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b)
// CHECK: ret <4 x i32> [[VAL]]
uint32x4_t test_vmmlaq_u32(uint32x4_t r, uint8x16_t a, uint8x16_t b) {
  return vmmlaq_u32(r, a, b);
}

// CHECK-LABEL: test_vusmmlaq_s32
// CHECK: [[VAL:%.*]] = call <4 x i32> @llvm.arm.neon.usmmla.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> %b)
// CHECK: ret <4 x i32> [[VAL]]
int32x4_t test_vusmmlaq_s32(int32x4_t r, uint8x16_t a, int8x16_t b) {
  return vusmmlaq_s32(r, a, b);
}

// CHECK-LABEL: test_vusdot_s32
// CHECK: [[VAL:%.*]] = call <2 x i32> @llvm.arm.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> %b)
// CHECK: ret <2 x i32> [[VAL]]
int32x2_t test_vusdot_s32(int32x2_t r, uint8x8_t a, int8x8_t b) {
  return vusdot_s32(r, a, b);
}

// CHECK-LABEL: test_vusdot_lane_s32
// CHECK: [[TMP0:%.*]] = bitcast <8 x i8> %b to <2 x i32>
// CHECK: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK: [[LANE:%.*]] = shufflevector <2 x i32> [[TMP2]], <2 x i32> [[TMP2]], <2 x i32> zeroinitializer
// CHECK: [[TMP3:%.*]] = bitcast <2 x i32> [[LANE]] to <8 x i8>
// CHECK: [[TMP4:%.*]] = bitcast <2 x i32> %r to <8 x i8>
// CHECK: [[OP:%.*]] = call <2 x i32> @llvm.arm.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> %a, <8 x i8> [[TMP3]])
// CHECK: ret <2 x i32> [[OP]]
int32x2_t test_vusdot_lane_s32(int32x2_t r, uint8x8_t a, int8x8_t b) {
  return vusdot_lane_s32(r, a, b, 0);
}

// CHECK-LABEL: test_vsudot_lane_s32
// CHECK: [[TMP0:%.*]] = bitcast <8 x i8> %b to <2 x i32>
// CHECK: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK: [[LANE:%.*]] = shufflevector <2 x i32> [[TMP2]], <2 x i32> [[TMP2]], <2 x i32> zeroinitializer
// CHECK: [[TMP3:%.*]] = bitcast <2 x i32> [[LANE]] to <8 x i8>
// CHECK: [[TMP4:%.*]] = bitcast <2 x i32> %r to <8 x i8>
// CHECK: [[OP:%.*]] = call <2 x i32> @llvm.arm.neon.usdot.v2i32.v8i8(<2 x i32> %r, <8 x i8> [[TMP3]], <8 x i8> %a)
// CHECK: ret <2 x i32> [[OP]]
int32x2_t test_vsudot_lane_s32(int32x2_t r, int8x8_t a, uint8x8_t b) {
  return vsudot_lane_s32(r, a, b, 0);
}

// CHECK-LABEL: test_vusdotq_lane_s32
// CHECK: [[TMP0:%.*]] = bitcast <8 x i8> %b to <2 x i32>
// CHECK: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK: [[LANE:%.*]] = shufflevector <2 x i32> [[TMP2]], <2 x i32> [[TMP2]], <4 x i32> zeroinitializer
// CHECK: [[TMP4:%.*]] = bitcast <4 x i32> [[LANE]] to <16 x i8>
// CHECK: [[TMP5:%.*]] = bitcast <4 x i32> %r to <16 x i8>
// CHECK: [[OP:%.*]] = call <4 x i32> @llvm.arm.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %a, <16 x i8> [[TMP4]])
// CHECK: ret <4 x i32> [[OP]]
int32x4_t test_vusdotq_lane_s32(int32x4_t r, uint8x16_t a, int8x8_t b) {
  return vusdotq_lane_s32(r, a, b, 0);
}

// CHECK-LABEL: test_vsudotq_lane_s32
// CHECK: [[TMP0:%.*]] = bitcast <8 x i8> %b to <2 x i32>
// CHECK: [[TMP1:%.*]] = bitcast <2 x i32> [[TMP0]] to <8 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// CHECK: [[LANE:%.*]] = shufflevector <2 x i32> [[TMP2]], <2 x i32> [[TMP2]], <4 x i32> zeroinitializer
// CHECK: [[TMP4:%.*]] = bitcast <4 x i32> %r to <16 x i8>
// CHECK: [[OP:%.*]] = call <4 x i32> @llvm.arm.neon.usdot.v4i32.v16i8(<4 x i32> %r, <16 x i8> %3, <16 x i8> %a)
// CHECK: ret <4 x i32> [[OP]]
int32x4_t test_vsudotq_lane_s32(int32x4_t r, int8x16_t a, uint8x8_t b) {
  return vsudotq_lane_s32(r, a, b, 0);
}
