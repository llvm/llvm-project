// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +dotprod \
// RUN: -disable-O0-optnone  -emit-llvm -o - %s | opt -S -passes=instcombine | FileCheck %s

// REQUIRES: aarch64-registered-target

// Test AArch64 Armv8.2-A dot product intrinsics

#include <arm_neon.h>

uint32x2_t test_vdot_u32(uint32x2_t a, uint8x8_t b, uint8x8_t c) {
// CHECK-LABEL: define{{.*}} <2 x i32> @test_vdot_u32(<2 x i32> noundef %a, <8 x i8> noundef %b, <8 x i8> noundef %c)
// CHECK: [[RESULT:%.*]] = call <2 x i32> @llvm.aarch64.neon.udot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c)
// CHECK: ret <2 x i32> [[RESULT]]
  return vdot_u32(a, b, c);
}

uint32x4_t test_vdotq_u32(uint32x4_t a, uint8x16_t b, uint8x16_t c) {
// CHECK-LABEL: define{{.*}} <4 x i32> @test_vdotq_u32(<4 x i32> noundef %a, <16 x i8> noundef %b, <16 x i8> noundef %c)
// CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.aarch64.neon.udot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c)
// CHECK: ret <4 x i32> [[RESULT]]
  return vdotq_u32(a, b, c);
}

int32x2_t test_vdot_s32(int32x2_t a, int8x8_t b, int8x8_t c) {
// CHECK-LABEL: define{{.*}} <2 x i32> @test_vdot_s32(<2 x i32> noundef %a, <8 x i8> noundef %b, <8 x i8> noundef %c)
// CHECK: [[RESULT:%.*]] = call <2 x i32> @llvm.aarch64.neon.sdot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> %c)
// CHECK: ret <2 x i32> [[RESULT]]
  return vdot_s32(a, b, c);
}

int32x4_t test_vdotq_s32(int32x4_t a, int8x16_t b, int8x16_t c) {
// CHECK-LABEL: define{{.*}} <4 x i32> @test_vdotq_s32(<4 x i32> noundef %a, <16 x i8> noundef %b, <16 x i8> noundef %c)
// CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> %c)
// CHECK: ret <4 x i32> [[RESULT]]
  return vdotq_s32(a, b, c);
}

uint32x2_t test_vdot_lane_u32(uint32x2_t a, uint8x8_t b, uint8x8_t c) {
// CHECK-LABEL: define{{.*}} <2 x i32> @test_vdot_lane_u32(<2 x i32> noundef %a, <8 x i8> noundef %b, <8 x i8> noundef %c)
// CHECK: [[CAST1:%.*]] = bitcast <8 x i8> %c to <2 x i32>
// CHECK: [[SHUFFLE:%.*]] = shufflevector <2 x i32> [[CAST1]], <2 x i32> poison, <2 x i32> <i32 1, i32 1>
// CHECK: [[CAST2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK: [[RESULT:%.*]] = call <2 x i32> @llvm.aarch64.neon.udot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> [[CAST2]])
// CHECK: ret <2 x i32> [[RESULT]]
  return vdot_lane_u32(a, b, c, 1);
}

uint32x4_t test_vdotq_lane_u32(uint32x4_t a, uint8x16_t b, uint8x8_t c) {
// CHECK-LABEL: define{{.*}} <4 x i32> @test_vdotq_lane_u32(<4 x i32> noundef %a, <16 x i8> noundef %b, <8 x i8> noundef %c)
// CHECK: [[CAST1:%.*]] = bitcast <8 x i8> %c to <2 x i32>
// CHECK: [[SHUFFLE:%.*]] = shufflevector <2 x i32> [[CAST1]], <2 x i32> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK: [[CAST2:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.aarch64.neon.udot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> [[CAST2]])
// CHECK: ret <4 x i32> [[RESULT]]
  return vdotq_lane_u32(a, b, c, 1);
}

uint32x2_t test_vdot_laneq_u32(uint32x2_t a, uint8x8_t b, uint8x16_t c) {
// CHECK-LABEL: define{{.*}} <2 x i32> @test_vdot_laneq_u32(<2 x i32> noundef %a, <8 x i8> noundef %b, <16 x i8> noundef %c)
// CHECK: [[CAST1:%.*]] = bitcast <16 x i8> %c to <4 x i32>
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x i32> [[CAST1]], <4 x i32> poison, <2 x i32> <i32 1, i32 1>
// CHECK: [[CAST2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK: [[RESULT:%.*]] = call <2 x i32> @llvm.aarch64.neon.udot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> [[CAST2]])
// CHECK: ret <2 x i32> [[RESULT]]
  return vdot_laneq_u32(a, b, c, 1);
}

uint32x4_t test_vdotq_laneq_u32(uint32x4_t a, uint8x16_t b, uint8x16_t c) {
// CHECK-LABEL: define{{.*}} <4 x i32> @test_vdotq_laneq_u32(<4 x i32> noundef %a, <16 x i8> noundef %b, <16 x i8> noundef %c)
// CHECK: [[CAST1:%.*]] = bitcast <16 x i8> %c to <4 x i32>
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x i32> [[CAST1]], <4 x i32> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK: [[CAST2:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.aarch64.neon.udot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> [[CAST2]])
// CHECK: ret <4 x i32> [[RESULT]]
  return vdotq_laneq_u32(a, b, c, 1);
}

int32x2_t test_vdot_lane_s32(int32x2_t a, int8x8_t b, int8x8_t c) {
// CHECK-LABEL: define{{.*}} <2 x i32> @test_vdot_lane_s32(<2 x i32> noundef %a, <8 x i8> noundef %b, <8 x i8> noundef %c)
// CHECK: [[CAST1:%.*]] = bitcast <8 x i8> %c to <2 x i32>
// CHECK: [[SHUFFLE:%.*]] = shufflevector <2 x i32> [[CAST1]], <2 x i32> poison, <2 x i32> <i32 1, i32 1>
// CHECK: [[CAST2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK: [[RESULT:%.*]] = call <2 x i32> @llvm.aarch64.neon.sdot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> [[CAST2]])
// CHECK: ret <2 x i32> [[RESULT]]
  return vdot_lane_s32(a, b, c, 1);
}

int32x4_t test_vdotq_lane_s32(int32x4_t a, int8x16_t b, int8x8_t c) {
// CHECK-LABEL: define{{.*}} <4 x i32> @test_vdotq_lane_s32(<4 x i32> noundef %a, <16 x i8> noundef %b, <8 x i8> noundef %c)
// CHECK: [[CAST1:%.*]] = bitcast <8 x i8> %c to <2 x i32>
// CHECK: [[SHUFFLE:%.*]] = shufflevector <2 x i32> [[CAST1]], <2 x i32> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK: [[CAST2:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> [[CAST2]])
// CHECK: ret <4 x i32> [[RESULT]]
  return vdotq_lane_s32(a, b, c, 1);
}

int32x2_t test_vdot_laneq_s32(int32x2_t a, int8x8_t b, int8x16_t c) {
// CHECK-LABEL: define{{.*}} <2 x i32> @test_vdot_laneq_s32(<2 x i32> noundef %a, <8 x i8> noundef %b, <16 x i8> noundef %c)
// CHECK: [[CAST1:%.*]] = bitcast <16 x i8> %c to <4 x i32>
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x i32> [[CAST1]], <4 x i32> poison, <2 x i32> <i32 1, i32 1>
// CHECK: [[CAST2:%.*]] = bitcast <2 x i32> [[SHUFFLE]] to <8 x i8>
// CHECK: [[RESULT:%.*]] = call <2 x i32> @llvm.aarch64.neon.sdot.v2i32.v8i8(<2 x i32> %a, <8 x i8> %b, <8 x i8> [[CAST2]])
// CHECK: ret <2 x i32> [[RESULT]]
  return vdot_laneq_s32(a, b, c, 1);
}

int32x4_t test_vdotq_laneq_s32(int32x4_t a, int8x16_t b, int8x16_t c) {
// CHECK-LABEL: define{{.*}} <4 x i32> @test_vdotq_laneq_s32(<4 x i32> noundef %a, <16 x i8> noundef %b, <16 x i8> noundef %c)
// CHECK: [[CAST1:%.*]] = bitcast <16 x i8> %c to <4 x i32>
// CHECK: [[SHUFFLE:%.*]] = shufflevector <4 x i32> [[CAST1]], <4 x i32> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK: [[CAST2:%.*]] = bitcast <4 x i32> [[SHUFFLE]] to <16 x i8>
// CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32> %a, <16 x i8> %b, <16 x i8> [[CAST2]])
// CHECK: ret <4 x i32> [[RESULT]]
  return vdotq_laneq_s32(a, b, c, 1);
}

