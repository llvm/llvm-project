// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=ALL,CIR %}

//=============================================================================
// NOTES
//
// This file contains tests that were originally located in
//  * clang/test/CodeGen/AArch64/neon-vget.c
//  * clang/test/CodeGen/AArch64/poly64.c
// The main difference is the use of RUN lines that enable ClangIR lowering;
// therefore only builtins currently supported by ClangIR are tested here.
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// Extract one element from vector
//===------------------------------------------------------===//

// ALL-LABEL: @test_vget_lane_u8(
uint8_t test_vget_lane_u8(uint8x8_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<8 x !u8i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <8 x i8> %{{.*}}, i32 7
// LLVM: ret i8 [[VGET_LANE]]
  return vget_lane_u8(a, 7);
}

// ALL-LABEL: @test_vget_lane_u16(
uint16_t test_vget_lane_u16(uint16x4_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !u16i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<4 x !u16i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <4 x i16> %{{.*}}, i32 3
// LLVM: ret i16 [[VGET_LANE]]
  return vget_lane_u16(a, 3);
}

// ALL-LABEL: @test_vget_lane_u32(
uint32_t test_vget_lane_u32(uint32x2_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !u32i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<2 x !u32i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <2 x i32> %{{.*}}, i32 1
// LLVM: ret i32 [[VGET_LANE]]
  return vget_lane_u32(a, 1);
}

// ALL-LABEL: @test_vget_lane_s8(
int8_t test_vget_lane_s8(int8x8_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<8 x !u8i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <8 x i8> %{{.*}}, i32 7
// LLVM: ret i8 [[VGET_LANE]]
  return vget_lane_s8(a, 7);
}

// ALL-LABEL: @test_vget_lane_s16(
int16_t test_vget_lane_s16(int16x4_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !u16i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<4 x !u16i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <4 x i16> %{{.*}}, i32 3
// LLVM: ret i16 [[VGET_LANE]]
  return vget_lane_s16(a, 3);
}

// ALL-LABEL: @test_vget_lane_s32(
int32_t test_vget_lane_s32(int32x2_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !u32i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<2 x !u32i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <2 x i32> %{{.*}}, i32 1
// LLVM: ret i32 [[VGET_LANE]]
  return vget_lane_s32(a, 1);
}

// ALL-LABEL: @test_vget_lane_p8(
poly8_t test_vget_lane_p8(poly8x8_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<8 x !u8i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <8 x i8> %{{.*}}, i32 7
// LLVM: ret i8 [[VGET_LANE]]
  return vget_lane_p8(a, 7);
}

// ALL-LABEL: @test_vget_lane_p16(
poly16_t test_vget_lane_p16(poly16x4_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !u16i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<4 x !u16i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <4 x i16> %{{.*}}, i32 3
// LLVM: ret i16 [[VGET_LANE]]
  return vget_lane_p16(a, 3);
}

// ALL-LABEL: @test_vget_lane_f16(
float32_t test_vget_lane_f16(float16x4_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !u16i>
// CIR: [[ELEM:%.*]] = cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<4 x !u16i>
// CIR: cir.store align(2) [[ELEM]], %{{.*}} : !u16i, !cir.ptr<!u16i>
// CIR: [[S16PTR:%.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!u16i> -> !cir.ptr<!s16i>
// CIR: %{{.*}} = cir.load align(2) [[S16PTR]] : !cir.ptr<!s16i>, !s16i
// CIR: [[F16PTR:%.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!s16i> -> !cir.ptr<!cir.f16>
// CIR: [[HALF:%.*]] = cir.load align(2) [[F16PTR]] : !cir.ptr<!cir.f16>, !cir.f16
// CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.f16 -> !cir.float
// CIR: cir.return %{{.*}} : !cir.float

// LLVM: [[TMP:%.*]] = bitcast <4 x half> %{{.*}} to <4 x i16>
// LLVM: [[VGET_LANE:%.*]] = extractelement <4 x i16> [[TMP]], i32 1
// LLVM: [[HALF:%.*]] = bitcast i16 [[VGET_LANE]] to half
// LLVM: [[RES:%.*]] = fpext half [[HALF]] to float
// LLVM: ret float [[RES]]
  return vget_lane_f16(a, 1);
}

// ALL-LABEL: @test_vget_lane_f32(
float32_t test_vget_lane_f32(float32x2_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<2 x !cir.float>

// LLVM: [[VGET_LANE:%.*]] = extractelement <2 x float> %{{.*}}, i32 1
// LLVM: ret float [[VGET_LANE]]
  return vget_lane_f32(a, 1);
}

// ALL-LABEL: @test_vget_lane_f64(
float64_t test_vget_lane_f64(float64x1_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<1 x !cir.double>

// LLVM: [[VGET_LANE:%.*]] = extractelement <1 x double> %{{.*}}, i32 0
// LLVM: ret double [[VGET_LANE]]
  return vget_lane_f64(a, 0);
}

// ALL-LABEL: @test_vgetq_lane_u8(
uint8_t test_vgetq_lane_u8(uint8x16_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<16 x !u8i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %{{.*}}, i32 15
// LLVM: ret i8 [[VGETQ_LANE]]
  return vgetq_lane_u8(a, 15);
}

// ALL-LABEL: @test_vgetq_lane_u16(
uint16_t test_vgetq_lane_u16(uint16x8_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<8 x !u16i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %{{.*}}, i32 7
// LLVM: ret i16 [[VGETQ_LANE]]
  return vgetq_lane_u16(a, 7);
}

// ALL-LABEL: @test_vgetq_lane_u32(
uint32_t test_vgetq_lane_u32(uint32x4_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !u32i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<4 x !u32i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %{{.*}}, i32 3
// LLVM: ret i32 [[VGETQ_LANE]]
  return vgetq_lane_u32(a, 3);
}

// ALL-LABEL: @test_vgetq_lane_s8(
int8_t test_vgetq_lane_s8(int8x16_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<16 x !u8i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %{{.*}}, i32 15
// LLVM: ret i8 [[VGETQ_LANE]]
  return vgetq_lane_s8(a, 15);
}

// ALL-LABEL: @test_vgetq_lane_s16(
int16_t test_vgetq_lane_s16(int16x8_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<8 x !u16i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %{{.*}}, i32 7
// LLVM: ret i16 [[VGETQ_LANE]]
  return vgetq_lane_s16(a, 7);
}

// ALL-LABEL: @test_vgetq_lane_s32(
int32_t test_vgetq_lane_s32(int32x4_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<4 x !u32i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<4 x !u32i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <4 x i32> %{{.*}}, i32 3
// LLVM: ret i32 [[VGETQ_LANE]]
  return vgetq_lane_s32(a, 3);
}

// ALL-LABEL: @test_vgetq_lane_p8(
poly8_t test_vgetq_lane_p8(poly8x16_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<16 x !u8i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <16 x i8> %{{.*}}, i32 15
// LLVM: ret i8 [[VGETQ_LANE]]
  return vgetq_lane_p8(a, 15);
}

// ALL-LABEL: @test_vgetq_lane_p16(
poly16_t test_vgetq_lane_p16(poly16x8_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<8 x !u16i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <8 x i16> %{{.*}}, i32 7
// LLVM: ret i16 [[VGETQ_LANE]]
  return vgetq_lane_p16(a, 7);
}

// ALL-LABEL: @test_vgetq_lane_f16(
float32_t test_vgetq_lane_f16(float16x8_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<8 x !u16i>
// CIR: [[ELEM:%.*]] = cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<8 x !u16i>
// CIR: cir.store align(2) [[ELEM]], %{{.*}} : !u16i, !cir.ptr<!u16i>
// CIR: [[S16PTR:%.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!u16i> -> !cir.ptr<!s16i>
// CIR: %{{.*}} = cir.load align(2) [[S16PTR]] : !cir.ptr<!s16i>, !s16i
// CIR: [[F16PTR:%.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!s16i> -> !cir.ptr<!cir.f16>
// CIR: [[HALF:%.*]] = cir.load align(2) [[F16PTR]] : !cir.ptr<!cir.f16>, !cir.f16
// CIR: %{{.*}} = cir.cast floating %{{.*}} : !cir.f16 -> !cir.float
// CIR: cir.return %{{.*}} : !cir.float

// LLVM: [[TMP:%.*]] = bitcast <8 x half> %{{.*}} to <8 x i16>
// LLVM: [[VGETQ_LANE:%.*]] = extractelement <8 x i16> [[TMP]], i32 3
// LLVM: [[HALF:%.*]] = bitcast i16 [[VGETQ_LANE]] to half
// LLVM: [[RES:%.*]] = fpext half [[HALF]] to float
// LLVM: ret float [[RES]]
  return vgetq_lane_f16(a, 3);
}

// ALL-LABEL: @test_vgetq_lane_f32(
float32_t test_vgetq_lane_f32(float32x4_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<4 x !cir.float>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <4 x float> %{{.*}}, i32 3
// LLVM: ret float [[VGETQ_LANE]]
  return vgetq_lane_f32(a, 3);
}

// ALL-LABEL: @test_vgetq_lane_f64(
float64_t test_vgetq_lane_f64(float64x2_t a) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<2 x !cir.double>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <2 x double> %{{.*}}, i32 1
// LLVM: ret double [[VGETQ_LANE]]
  return vgetq_lane_f64(a, 1);
}

// ALL-LABEL: @test_vget_lane_s64(
int64_t test_vget_lane_s64(int64x1_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<1 x !u64i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<1 x !u64i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <1 x i64> %{{.*}}, i32 0
// LLVM: ret i64 [[VGET_LANE]]
  return vget_lane_s64(a, 0);
}

// ALL-LABEL: @test_vget_lane_u64(
uint64_t test_vget_lane_u64(uint64x1_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<1 x !u64i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<1 x !u64i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <1 x i64> %{{.*}}, i32 0
// LLVM: ret i64 [[VGET_LANE]]
  return vget_lane_u64(a, 0);
}

// ALL-LABEL: @test_vget_lane_p64(
poly64_t test_vget_lane_p64(poly64x1_t v) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<1 x !u64i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<1 x !u64i>

// LLVM: [[VGET_LANE:%.*]] = extractelement <1 x i64> %{{.*}}, i32 0
// LLVM: ret i64 [[VGET_LANE]]
  return vget_lane_p64(v, 0);
}

// ALL-LABEL: @test_vgetq_lane_s64(
int64_t test_vgetq_lane_s64(int64x2_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<2 x !u64i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %{{.*}}, i32 1
// LLVM: ret i64 [[VGETQ_LANE]]
  return vgetq_lane_s64(a, 1);
}

// ALL-LABEL: @test_vgetq_lane_u64(
uint64_t test_vgetq_lane_u64(uint64x2_t a) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<2 x !u64i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %{{.*}}, i32 1
// LLVM: ret i64 [[VGETQ_LANE]]
  return vgetq_lane_u64(a, 1);
}

// ALL-LABEL: @test_vgetq_lane_p64(
poly64_t test_vgetq_lane_p64(poly64x2_t v) {
// CIR: [[V:%.*]] = cir.cast bitcast %{{.*}} : {{.*}} -> !cir.vector<2 x !u64i>
// CIR: cir.vec.extract [[V]][%{{.*}} : {{.*}}] : !cir.vector<2 x !u64i>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %{{.*}}, i32 1
// LLVM: ret i64 [[VGETQ_LANE]]
  return vgetq_lane_p64(v, 1);
}
