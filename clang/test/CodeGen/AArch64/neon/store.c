// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa,instcombine | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefixes=ALL,CIR %}

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.1.11. Store
// https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#store
//===------------------------------------------------------===//

// ALL-LABEL: @test_vst1_f16(
void test_vst1_f16(float16_t *a, float16x4_t b) {
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.vector<4 x !cir.f16>, !cir.ptr<!cir.vector<4 x !cir.f16>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]])
// LLVM: store <4 x half> [[B]], ptr [[A]], align 2
// LLVM: ret void
  vst1_f16(a, b);
}

// ALL-LABEL: @test_vst1_f32(
void test_vst1_f32(float32_t *a, float32x2_t b) {
// CIR: cir.store align(4) {{.*}}, {{.*}} : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]])
// LLVM: store <2 x float> [[B]], ptr [[A]], align 4
// LLVM: ret void
  vst1_f32(a, b);
}

// ALL-LABEL: @test_vst1_f64(
void test_vst1_f64(float64_t *a, float64x1_t b) {
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.vector<1 x !cir.double>, !cir.ptr<!cir.vector<1 x !cir.double>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]])
// LLVM: store <1 x double> [[B]], ptr [[A]], align 8
// LLVM: ret void
  vst1_f64(a, b);
}

// ALL-LABEL: @test_vst1_mf8(
void test_vst1_mf8(mfloat8_t *a, mfloat8x8_t val) {
// CIR: cir.store align(1) {{.*}}, {{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}}[[B:%.*]])
// LLVM: store <8 x i8> [[B]], ptr [[A]], align 1
// LLVM: ret void
  vst1_mf8(a, val);
}

// ALL-LABEL: @test_vst1_p16(
void test_vst1_p16(poly16_t *a, poly16x4_t b) {
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: store <4 x i16> [[B]], ptr [[A]], align 2
// LLVM: ret void
  vst1_p16(a, b);
}

// ALL-LABEL: @test_vst1_p64(
void test_vst1_p64(poly64_t * ptr, poly64x1_t val) {
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.vector<1 x !s64i>, !cir.ptr<!cir.vector<1 x !s64i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: store <1 x i64> [[B]], ptr [[A]], align 8
// LLVM: ret void
  vst1_p64(ptr, val);
}

// ALL-LABEL: @test_vst1_p8(
void test_vst1_p8(poly8_t *a, poly8x8_t b) {
// CIR: cir.store align(1) {{.*}}, {{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: store <8 x i8> [[B]], ptr [[A]], align 1
// LLVM: ret void
  vst1_p8(a, b);
}

// ALL-LABEL: @test_vst1q_f16(
void test_vst1q_f16(float16_t *a, float16x8_t b) {
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.vector<8 x !cir.f16>, !cir.ptr<!cir.vector<8 x !cir.f16>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]])
// LLVM: store <8 x half> [[B]], ptr [[A]], align 2
// LLVM: ret void
  vst1q_f16(a, b);
}

// ALL-LABEL: @test_vst1q_f32(
void test_vst1q_f32(float32_t *a, float32x4_t b) {
// CIR: cir.store align(4) {{.*}}, {{.*}} : !cir.vector<4 x !cir.float>, !cir.ptr<!cir.vector<4 x !cir.float>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]])
// LLVM: store <4 x float> [[B]], ptr [[A]], align 4
// LLVM: ret void
  vst1q_f32(a, b);
}

// ALL-LABEL: @test_vst1q_f64(
void test_vst1q_f64(float64_t *a, float64x2_t b) {
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]])
// LLVM: store <2 x double> [[B]], ptr [[A]], align 8
// LLVM: ret void
  vst1q_f64(a, b);
}

// ALL-LABEL: @test_vst1q_mf8(
void test_vst1q_mf8(mfloat8_t *a, mfloat8x16_t val) {
// CIR: cir.store align(1) {{.*}}, {{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}}[[B:%.*]])
// LLVM: store <16 x i8> [[B]], ptr [[A]], align 1
// LLVM: ret void
  vst1q_mf8(a, val);
}

// ALL-LABEL: @test_vst1q_p16(
void test_vst1q_p16(poly16_t *a, poly16x8_t b) {
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: store <8 x i16> [[B]], ptr [[A]], align 2
// LLVM: ret void
  vst1q_p16(a, b);
}

// ALL-LABEL: @test_vst1q_p64(
void test_vst1q_p64(poly64_t * ptr, poly64x2_t val) {
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: store <2 x i64> [[B]], ptr [[A]], align 8
// LLVM: ret void
  vst1q_p64(ptr, val);
}

// ALL-LABEL: @test_vst1q_p8(
void test_vst1q_p8(poly8_t *a, poly8x16_t b) {
// CIR: cir.store align(1) {{.*}}, {{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: store <16 x i8> [[B]], ptr [[A]], align 1
// LLVM: ret void
  vst1q_p8(a, b);
}

// ALL-LABEL: @test_vst1q_s16(
void test_vst1q_s16(int16_t *a, int16x8_t b) {
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.vector<8 x !s16i>, !cir.ptr<!cir.vector<8 x !s16i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: store <8 x i16> [[B]], ptr [[A]], align 2
// LLVM: ret void
  vst1q_s16(a, b);
}

// ALL-LABEL: @test_vst1q_s32(
void test_vst1q_s32(int32_t *a, int32x4_t b) {
// CIR: cir.store align(4) {{.*}}, {{.*}} : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: store <4 x i32> [[B]], ptr [[A]], align 4
// LLVM: ret void
  vst1q_s32(a, b);
}

// ALL-LABEL: @test_vst1q_s64(
void test_vst1q_s64(int64_t *a, int64x2_t b) {
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: store <2 x i64> [[B]], ptr [[A]], align 8
// LLVM: ret void
  vst1q_s64(a, b);
}

// ALL-LABEL: @test_vst1q_s8(
void test_vst1q_s8(int8_t *a, int8x16_t b) {
// CIR: cir.store align(1) {{.*}}, {{.*}} : !cir.vector<16 x !s8i>, !cir.ptr<!cir.vector<16 x !s8i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: store <16 x i8> [[B]], ptr [[A]], align 1
// LLVM: ret void
  vst1q_s8(a, b);
}

// ALL-LABEL: @test_vst1q_u16(
void test_vst1q_u16(uint16_t *a, uint16x8_t b) {
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.vector<8 x !u16i>, !cir.ptr<!cir.vector<8 x !u16i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: store <8 x i16> [[B]], ptr [[A]], align 2
// LLVM: ret void
  vst1q_u16(a, b);
}

// ALL-LABEL: @test_vst1q_u32(
void test_vst1q_u32(uint32_t *a, uint32x4_t b) {
// CIR: cir.store align(4) {{.*}}, {{.*}} : !cir.vector<4 x !u32i>, !cir.ptr<!cir.vector<4 x !u32i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: store <4 x i32> [[B]], ptr [[A]], align 4
// LLVM: ret void
  vst1q_u32(a, b);
}

// ALL-LABEL: @test_vst1q_u64(
void test_vst1q_u64(uint64_t *a, uint64x2_t b) {
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.vector<2 x !u64i>, !cir.ptr<!cir.vector<2 x !u64i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: store <2 x i64> [[B]], ptr [[A]], align 8
// LLVM: ret void
  vst1q_u64(a, b);
}

// ALL-LABEL: @test_vst1q_u8(
void test_vst1q_u8(uint8_t *a, uint8x16_t b) {
// CIR: cir.store align(1) {{.*}}, {{.*}} : !cir.vector<16 x !u8i>, !cir.ptr<!cir.vector<16 x !u8i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: store <16 x i8> [[B]], ptr [[A]], align 1
// LLVM: ret void
  vst1q_u8(a, b);
}

// ALL-LABEL: @test_vst1_s16(
void test_vst1_s16(int16_t *a, int16x4_t b) {
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: store <4 x i16> [[B]], ptr [[A]], align 2
// LLVM: ret void
  vst1_s16(a, b);
}

// ALL-LABEL: @test_vst1_s32(
void test_vst1_s32(int32_t *a, int32x2_t b) {
// CIR: cir.store align(4) {{.*}}, {{.*}} : !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: store <2 x i32> [[B]], ptr [[A]], align 4
// LLVM: ret void
  vst1_s32(a, b);
}

// ALL-LABEL: @test_vst1_s64(
void test_vst1_s64(int64_t *a, int64x1_t b) {
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.vector<1 x !s64i>, !cir.ptr<!cir.vector<1 x !s64i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: store <1 x i64> [[B]], ptr [[A]], align 8
// LLVM: ret void
  vst1_s64(a, b);
}

// ALL-LABEL: @test_vst1_s8(
void test_vst1_s8(int8_t *a, int8x8_t b) {
// CIR: cir.store align(1) {{.*}}, {{.*}} : !cir.vector<8 x !s8i>, !cir.ptr<!cir.vector<8 x !s8i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: store <8 x i8> [[B]], ptr [[A]], align 1
// LLVM: ret void
  vst1_s8(a, b);
}

// ALL-LABEL: @test_vst1_u16(
void test_vst1_u16(uint16_t *a, uint16x4_t b) {
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.vector<4 x !u16i>, !cir.ptr<!cir.vector<4 x !u16i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: store <4 x i16> [[B]], ptr [[A]], align 2
// LLVM: ret void
  vst1_u16(a, b);
}

// ALL-LABEL: @test_vst1_u32(
void test_vst1_u32(uint32_t *a, uint32x2_t b) {
// CIR: cir.store align(4) {{.*}}, {{.*}} : !cir.vector<2 x !u32i>, !cir.ptr<!cir.vector<2 x !u32i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: store <2 x i32> [[B]], ptr [[A]], align 4
// LLVM: ret void
  vst1_u32(a, b);
}

// ALL-LABEL: @test_vst1_u64(
void test_vst1_u64(uint64_t *a, uint64x1_t b) {
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.vector<1 x !u64i>, !cir.ptr<!cir.vector<1 x !u64i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: store <1 x i64> [[B]], ptr [[A]], align 8
// LLVM: ret void
  vst1_u64(a, b);
}

// ALL-LABEL: @test_vst1_u8(
void test_vst1_u8(uint8_t *a, uint8x8_t b) {
// CIR: cir.store align(1) {{.*}}, {{.*}} : !cir.vector<8 x !u8i>, !cir.ptr<!cir.vector<8 x !u8i>>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: store <8 x i8> [[B]], ptr [[A]], align 1
// LLVM: ret void
  vst1_u8(a, b);
}

// ALL-LABEL: @test_vst1_lane_f16(
void test_vst1_lane_f16(float16_t  *a, float16x4_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<4 x !cir.f16>
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.f16, !cir.ptr<!cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <4 x half> [[B]], i64 3
// LLVM: store half [[TMP1]], ptr [[A]], align 2
// LLVM: ret void
  vst1_lane_f16(a, b, 3);
}

// ALL-LABEL: @test_vst1_lane_f32(
void test_vst1_lane_f32(float32_t  *a, float32x2_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<2 x !cir.float>
// CIR: cir.store align(4) {{.*}}, {{.*}} : !cir.float, !cir.ptr<!cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x float> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <2 x float> [[B]], i64 1
// LLVM: store float [[TMP1]], ptr [[A]], align 4
// LLVM: ret void
  vst1_lane_f32(a, b, 1);
}

// ALL-LABEL: @test_vst1_lane_f64(
void test_vst1_lane_f64(float64_t  *a, float64x1_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<1 x !cir.double>
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.double, !cir.ptr<!cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <1 x double> [[B]], i64 0
// LLVM: store double [[TMP1]], ptr [[A]], align 8
// LLVM: ret void
  vst1_lane_f64(a, b, 0);
}

// ALL-LABEL: @test_vst1_lane_mf8(
void test_vst1_lane_mf8(mfloat8_t *a, mfloat8x8_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<8 x !u8i>
// CIR: cir.store align(1) {{.*}}, {{.*}} : !u8i, !cir.ptr<!u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}}[[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <8 x i8> [[B]], i64 7
// LLVM: store i8 [[TMP1]], ptr [[A]], align 1
// LLVM: ret void
  vst1_lane_mf8(a, b, 7);
}

// ALL-LABEL: @test_vst1_lane_p16(
void test_vst1_lane_p16(poly16_t  *a, poly16x4_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<4 x !s16i>
// CIR: cir.store align(2) {{.*}}, {{.*}} : !s16i, !cir.ptr<!s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <4 x i16> [[B]], i64 3
// LLVM: store i16 [[TMP1]], ptr [[A]], align 2
// LLVM: ret void
  vst1_lane_p16(a, b, 3);
}

// ALL-LABEL: @test_vst1_lane_p64(
void test_vst1_lane_p64(poly64_t  *a, poly64x1_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<1 x !s64i>
// CIR: cir.store align(8) {{.*}}, {{.*}} : !s64i, !cir.ptr<!s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <1 x i64> [[B]], i64 0
// LLVM: store i64 [[TMP1]], ptr [[A]], align 8
// LLVM: ret void
  vst1_lane_p64(a, b, 0);
}

// ALL-LABEL: @test_vst1_lane_p8(
void test_vst1_lane_p8(poly8_t  *a, poly8x8_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<8 x !s8i>
// CIR: cir.store align(1) {{.*}}, {{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <8 x i8> [[B]], i64 7
// LLVM: store i8 [[TMP1]], ptr [[A]], align 1
// LLVM: ret void
  vst1_lane_p8(a, b, 7);
}

// ALL-LABEL: @test_vst1_lane_s16(
void test_vst1_lane_s16(int16_t  *a, int16x4_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<4 x !s16i>
// CIR: cir.store align(2) {{.*}}, {{.*}} : !s16i, !cir.ptr<!s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <4 x i16> [[B]], i64 3
// LLVM: store i16 [[TMP1]], ptr [[A]], align 2
// LLVM: ret void
  vst1_lane_s16(a, b, 3);
}

// ALL-LABEL: @test_vst1_lane_s32(
void test_vst1_lane_s32(int32_t  *a, int32x2_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<2 x !s32i>
// CIR: cir.store align(4) {{.*}}, {{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <2 x i32> [[B]], i64 1
// LLVM: store i32 [[TMP1]], ptr [[A]], align 4
// LLVM: ret void
  vst1_lane_s32(a, b, 1);
}

// ALL-LABEL: @test_vst1_lane_s64(
void test_vst1_lane_s64(int64_t  *a, int64x1_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<1 x !s64i>
// CIR: cir.store align(8) {{.*}}, {{.*}} : !s64i, !cir.ptr<!s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <1 x i64> [[B]], i64 0
// LLVM: store i64 [[TMP1]], ptr [[A]], align 8
// LLVM: ret void
  vst1_lane_s64(a, b, 0);
}

// ALL-LABEL: @test_vst1_lane_s8(
void test_vst1_lane_s8(int8_t  *a, int8x8_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<8 x !s8i>
// CIR: cir.store align(1) {{.*}}, {{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <8 x i8> [[B]], i64 7
// LLVM: store i8 [[TMP1]], ptr [[A]], align 1
// LLVM: ret void
  vst1_lane_s8(a, b, 7);
}

// ALL-LABEL: @test_vst1_lane_u16(
void test_vst1_lane_u16(uint16_t  *a, uint16x4_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<4 x !u16i>
// CIR: cir.store align(2) {{.*}}, {{.*}} : !u16i, !cir.ptr<!u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <4 x i16> [[B]], i64 3
// LLVM: store i16 [[TMP1]], ptr [[A]], align 2
// LLVM: ret void
  vst1_lane_u16(a, b, 3);
}

// ALL-LABEL: @test_vst1_lane_u32(
void test_vst1_lane_u32(uint32_t  *a, uint32x2_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<2 x !u32i>
// CIR: cir.store align(4) {{.*}}, {{.*}} : !u32i, !cir.ptr<!u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <2 x i32> [[B]], i64 1
// LLVM: store i32 [[TMP1]], ptr [[A]], align 4
// LLVM: ret void
  vst1_lane_u32(a, b, 1);
}

// ALL-LABEL: @test_vst1_lane_u64(
void test_vst1_lane_u64(uint64_t  *a, uint64x1_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<1 x !u64i>
// CIR: cir.store align(8) {{.*}}, {{.*}} : !u64i, !cir.ptr<!u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <1 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <1 x i64> [[B]], i64 0
// LLVM: store i64 [[TMP1]], ptr [[A]], align 8
// LLVM: ret void
  vst1_lane_u64(a, b, 0);
}

// ALL-LABEL: @test_vst1_lane_u8(
void test_vst1_lane_u8(uint8_t  *a, uint8x8_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<8 x !u8i>
// CIR: cir.store align(1) {{.*}}, {{.*}} : !u8i, !cir.ptr<!u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <8 x i8> [[B]], i64 7
// LLVM: store i8 [[TMP1]], ptr [[A]], align 1
// LLVM: ret void
  vst1_lane_u8(a, b, 7);
}

// ALL-LABEL: @test_vst1q_lane_f16(
void test_vst1q_lane_f16(float16_t  *a, float16x8_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<8 x !cir.f16>
// CIR: cir.store align(2) {{.*}}, {{.*}} : !cir.f16, !cir.ptr<!cir.f16>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <8 x half> [[B]], i64 7
// LLVM: store half [[TMP1]], ptr [[A]], align 2
// LLVM: ret void
  vst1q_lane_f16(a, b, 7);
}

// ALL-LABEL: @test_vst1q_lane_f32(
void test_vst1q_lane_f32(float32_t  *a, float32x4_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<4 x !cir.float>
// CIR: cir.store align(4) {{.*}}, {{.*}} : !cir.float, !cir.ptr<!cir.float>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x float> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <4 x float> [[B]], i64 3
// LLVM: store float [[TMP1]], ptr [[A]], align 4
// LLVM: ret void
  vst1q_lane_f32(a, b, 3);
}

// ALL-LABEL: @test_vst1q_lane_f64(
void test_vst1q_lane_f64(float64_t  *a, float64x2_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<2 x !cir.double>
// CIR: cir.store align(8) {{.*}}, {{.*}} : !cir.double, !cir.ptr<!cir.double>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <2 x double> [[B]], i64 1
// LLVM: store double [[TMP1]], ptr [[A]], align 8
// LLVM: ret void
  vst1q_lane_f64(a, b, 1);
}

// ALL-LABEL: @test_vst1q_lane_mf8(
void test_vst1q_lane_mf8(mfloat8_t *a, mfloat8x16_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<16 x !u8i>
// CIR: cir.store align(1) {{.*}}, {{.*}} : !u8i, !cir.ptr<!u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}}[[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <16 x i8> [[B]], i64 15
// LLVM: store i8 [[TMP1]], ptr [[A]], align 1
// LLVM: ret void
  vst1q_lane_mf8(a, b, 15);
}

// ALL-LABEL: @test_vst1q_lane_p16(
void test_vst1q_lane_p16(poly16_t  *a, poly16x8_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<8 x !s16i>
// CIR: cir.store align(2) {{.*}}, {{.*}} : !s16i, !cir.ptr<!s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <8 x i16> [[B]], i64 7
// LLVM: store i16 [[TMP1]], ptr [[A]], align 2
// LLVM: ret void
  vst1q_lane_p16(a, b, 7);
}

// ALL-LABEL: @test_vst1q_lane_p64(
void test_vst1q_lane_p64(poly64_t  *a, poly64x2_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<2 x !s64i>
// CIR: cir.store align(8) {{.*}}, {{.*}} : !s64i, !cir.ptr<!s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <2 x i64> [[B]], i64 1
// LLVM: store i64 [[TMP1]], ptr [[A]], align 8
// LLVM: ret void
  vst1q_lane_p64(a, b, 1);
}

// ALL-LABEL: @test_vst1q_lane_p8(
void test_vst1q_lane_p8(poly8_t  *a, poly8x16_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<16 x !s8i>
// CIR: cir.store align(1) {{.*}}, {{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <16 x i8> [[B]], i64 15
// LLVM: store i8 [[TMP1]], ptr [[A]], align 1
// LLVM: ret void
  vst1q_lane_p8(a, b, 15);
}

// ALL-LABEL: @test_vst1q_lane_s16(
void test_vst1q_lane_s16(int16_t  *a, int16x8_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<8 x !s16i>
// CIR: cir.store align(2) {{.*}}, {{.*}} : !s16i, !cir.ptr<!s16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <8 x i16> [[B]], i64 7
// LLVM: store i16 [[TMP1]], ptr [[A]], align 2
// LLVM: ret void
  vst1q_lane_s16(a, b, 7);
}

// ALL-LABEL: @test_vst1q_lane_s32(
void test_vst1q_lane_s32(int32_t  *a, int32x4_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<4 x !s32i>
// CIR: cir.store align(4) {{.*}}, {{.*}} : !s32i, !cir.ptr<!s32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <4 x i32> [[B]], i64 3
// LLVM: store i32 [[TMP1]], ptr [[A]], align 4
// LLVM: ret void
  vst1q_lane_s32(a, b, 3);
}

// ALL-LABEL: @test_vst1q_lane_s64(
void test_vst1q_lane_s64(int64_t  *a, int64x2_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<2 x !s64i>
// CIR: cir.store align(8) {{.*}}, {{.*}} : !s64i, !cir.ptr<!s64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <2 x i64> [[B]], i64 1
// LLVM: store i64 [[TMP1]], ptr [[A]], align 8
// LLVM: ret void
  vst1q_lane_s64(a, b, 1);
}

// ALL-LABEL: @test_vst1q_lane_s8(
void test_vst1q_lane_s8(int8_t  *a, int8x16_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<16 x !s8i>
// CIR: cir.store align(1) {{.*}}, {{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <16 x i8> [[B]], i64 15
// LLVM: store i8 [[TMP1]], ptr [[A]], align 1
// LLVM: ret void
  vst1q_lane_s8(a, b, 15);
}

// ALL-LABEL: @test_vst1q_lane_u16(
void test_vst1q_lane_u16(uint16_t  *a, uint16x8_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<8 x !u16i>
// CIR: cir.store align(2) {{.*}}, {{.*}} : !u16i, !cir.ptr<!u16i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <8 x i16> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <8 x i16> [[B]], i64 7
// LLVM: store i16 [[TMP1]], ptr [[A]], align 2
// LLVM: ret void
  vst1q_lane_u16(a, b, 7);
}

// ALL-LABEL: @test_vst1q_lane_u32(
void test_vst1q_lane_u32(uint32_t  *a, uint32x4_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<4 x !u32i>
// CIR: cir.store align(4) {{.*}}, {{.*}} : !u32i, !cir.ptr<!u32i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <4 x i32> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <4 x i32> [[B]], i64 3
// LLVM: store i32 [[TMP1]], ptr [[A]], align 4
// LLVM: ret void
  vst1q_lane_u32(a, b, 3);
}

// ALL-LABEL: @test_vst1q_lane_u64(
void test_vst1q_lane_u64(uint64_t  *a, uint64x2_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<2 x !u64i>
// CIR: cir.store align(8) {{.*}}, {{.*}} : !u64i, !cir.ptr<!u64i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <2 x i64> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <2 x i64> [[B]], i64 1
// LLVM: store i64 [[TMP1]], ptr [[A]], align 8
// LLVM: ret void
  vst1q_lane_u64(a, b, 1);
}

// ALL-LABEL: @test_vst1q_lane_u8(
void test_vst1q_lane_u8(uint8_t  *a, uint8x16_t b) {
// CIR: {{.*}} = cir.vec.extract {{.*}}[{{.*}} : !u64i] : !cir.vector<16 x !u8i>
// CIR: cir.store align(1) {{.*}}, {{.*}} : !u8i, !cir.ptr<!u8i>

// LLVM-SAME: ptr {{.*}} [[A:%.*]], <16 x i8> {{.*}} [[B:%.*]])
// LLVM: [[TMP1:%.*]] = extractelement <16 x i8> [[B]], i64 15
// LLVM: store i8 [[TMP1]], ptr [[A]], align 1
// LLVM: ret void
  vst1q_lane_u8(a, b, 15);
}

// ALL-LABEL: @test_vst1_f64_x2(
void test_vst1_f64_x2(float64_t *a, float64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v1f64.p0(<1 x double> {{.*}}, <1 x double> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f64_x2(a, b);
}

// ALL-LABEL: @test_vst1_f64_x3(
void test_vst1_f64_x3(float64_t *a, float64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v1f64.p0(<1 x double> {{.*}}, <1 x double> {{.*}}, <1 x double> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f64_x3(a, b);
}

// ALL-LABEL: @test_vst1_f64_x4(
void test_vst1_f64_x4(float64_t *a, float64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v1f64.p0(<1 x double> {{.*}}, <1 x double> {{.*}}, <1 x double> {{.*}}, <1 x double> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f64_x4(a, b);
}

// ALL-LABEL: @test_vst1_mf8_x2(
void test_vst1_mf8_x2(mfloat8_t *a, mfloat8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_mf8_x2(a, b);
}

// ALL-LABEL: @test_vst1_mf8_x3(
void test_vst1_mf8_x3(mfloat8_t *a, mfloat8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_mf8_x3(a, b);
}

// ALL-LABEL: @test_vst1_mf8_x4(
void test_vst1_mf8_x4(mfloat8_t *a, mfloat8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_mf8_x4(a, b);
}

// ALL-LABEL: @test_vst1_p64_x2(
void test_vst1_p64_x2(poly64_t *a, poly64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p64_x2(a, b);
}

// ALL-LABEL: @test_vst1_p64_x3(
void test_vst1_p64_x3(poly64_t *a, poly64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p64_x3(a, b);
}

// ALL-LABEL: @test_vst1_p64_x4(
void test_vst1_p64_x4(poly64_t *a, poly64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p64_x4(a, b);
}

// ALL-LABEL: @test_vst1q_f64_x2(
void test_vst1q_f64_x2(float64_t *a, float64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v2f64.p0(<2 x double> {{.*}}, <2 x double> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f64_x2(a, b);
}

// ALL-LABEL: @test_vst1q_f64_x3(
void test_vst1q_f64_x3(float64_t *a, float64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v2f64.p0(<2 x double> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f64_x3(a, b);
}

// ALL-LABEL: @test_vst1q_f64_x4(
void test_vst1q_f64_x4(float64_t *a, float64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v2f64.p0(<2 x double> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}, <2 x double> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f64_x4(a, b);
}

// ALL-LABEL: @test_vst1q_mf8_x2(
void test_vst1q_mf8_x2(mfloat8_t *a, mfloat8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_mf8_x2(a, b);
}

// ALL-LABEL: @test_vst1q_mf8_x3(
void test_vst1q_mf8_x3(mfloat8_t *a, mfloat8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_mf8_x3(a, b);
}

// ALL-LABEL: @test_vst1q_mf8_x4(
void test_vst1q_mf8_x4(mfloat8_t *a, mfloat8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_mf8_x4(a, b);
}

// ALL-LABEL: @test_vst1q_p64_x2(
void test_vst1q_p64_x2(poly64_t *a, poly64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p64_x2(a, b);
}

// ALL-LABEL: @test_vst1q_p64_x3(
void test_vst1q_p64_x3(poly64_t *a, poly64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p64_x3(a, b);
}

// ALL-LABEL: @test_vst1q_p64_x4(
void test_vst1q_p64_x4(poly64_t *a, poly64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p64_x4(a, b);
}

// NOTE: the vst1_x2/x3/x4 and vst1q_x2/x3/x4 tests below are duplicated
// from clang/test/CodeGen/arm-neon-vst.c; this file only covers AArch64
// codegen for these intrinsics.

// ALL-LABEL: @test_vst1_f16_x2(
void test_vst1_f16_x2(float16_t *a, float16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v4f16.p0(<4 x half> {{.*}}, <4 x half> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f16_x2(a, b);
}

// ALL-LABEL: @test_vst1_f32_x2(
void test_vst1_f32_x2(float32_t *a, float32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v2f32.p0(<2 x float> {{.*}}, <2 x float> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f32_x2(a, b);
}

// ALL-LABEL: @test_vst1_p16_x2(
void test_vst1_p16_x2(poly16_t *a, poly16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p16_x2(a, b);
}

// ALL-LABEL: @test_vst1_p8_x2(
void test_vst1_p8_x2(poly8_t *a, poly8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p8_x2(a, b);
}

// ALL-LABEL: @test_vst1_s16_x2(
void test_vst1_s16_x2(int16_t *a, int16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s16_x2(a, b);
}

// ALL-LABEL: @test_vst1_s32_x2(
void test_vst1_s32_x2(int32_t *a, int32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v2i32.p0(<2 x i32> {{.*}}, <2 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s32_x2(a, b);
}

// ALL-LABEL: @test_vst1_s64_x2(
void test_vst1_s64_x2(int64_t *a, int64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s64_x2(a, b);
}

// ALL-LABEL: @test_vst1_s8_x2(
void test_vst1_s8_x2(int8_t *a, int8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s8_x2(a, b);
}

// ALL-LABEL: @test_vst1_u16_x2(
void test_vst1_u16_x2(uint16_t *a, uint16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u16_x2(a, b);
}

// ALL-LABEL: @test_vst1_u32_x2(
void test_vst1_u32_x2(uint32_t *a, uint32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v2i32.p0(<2 x i32> {{.*}}, <2 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u32_x2(a, b);
}

// ALL-LABEL: @test_vst1_u64_x2(
void test_vst1_u64_x2(uint64_t *a, uint64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u64_x2(a, b);
}

// ALL-LABEL: @test_vst1_u8_x2(
void test_vst1_u8_x2(uint8_t *a, uint8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u8_x2(a, b);
}

// ALL-LABEL: @test_vst1_f16_x3(
void test_vst1_f16_x3(float16_t *a, float16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v4f16.p0(<4 x half> {{.*}}, <4 x half> {{.*}}, <4 x half> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f16_x3(a, b);
}

// ALL-LABEL: @test_vst1_f32_x3(
void test_vst1_f32_x3(float32_t *a, float32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v2f32.p0(<2 x float> {{.*}}, <2 x float> {{.*}}, <2 x float> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f32_x3(a, b);
}

// ALL-LABEL: @test_vst1_p16_x3(
void test_vst1_p16_x3(poly16_t *a, poly16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p16_x3(a, b);
}

// ALL-LABEL: @test_vst1_p8_x3(
void test_vst1_p8_x3(poly8_t *a, poly8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p8_x3(a, b);
}

// ALL-LABEL: @test_vst1_s16_x3(
void test_vst1_s16_x3(int16_t *a, int16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s16_x3(a, b);
}

// ALL-LABEL: @test_vst1_s32_x3(
void test_vst1_s32_x3(int32_t *a, int32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v2i32.p0(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s32_x3(a, b);
}

// ALL-LABEL: @test_vst1_s64_x3(
void test_vst1_s64_x3(int64_t *a, int64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s64_x3(a, b);
}

// ALL-LABEL: @test_vst1_s8_x3(
void test_vst1_s8_x3(int8_t *a, int8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s8_x3(a, b);
}

// ALL-LABEL: @test_vst1_u16_x3(
void test_vst1_u16_x3(uint16_t *a, uint16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u16_x3(a, b);
}

// ALL-LABEL: @test_vst1_u32_x3(
void test_vst1_u32_x3(uint32_t *a, uint32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v2i32.p0(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u32_x3(a, b);
}

// ALL-LABEL: @test_vst1_u64_x3(
void test_vst1_u64_x3(uint64_t *a, uint64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u64_x3(a, b);
}

// ALL-LABEL: @test_vst1_u8_x3(
void test_vst1_u8_x3(uint8_t *a, uint8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u8_x3(a, b);
}

// ALL-LABEL: @test_vst1_f16_x4(
void test_vst1_f16_x4(float16_t *a, float16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v4f16.p0(<4 x half> {{.*}}, <4 x half> {{.*}}, <4 x half> {{.*}}, <4 x half> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f16_x4(a, b);
}

// ALL-LABEL: @test_vst1_f32_x4(
void test_vst1_f32_x4(float32_t *a, float32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v2f32.p0(<2 x float> {{.*}}, <2 x float> {{.*}}, <2 x float> {{.*}}, <2 x float> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_f32_x4(a, b);
}

// ALL-LABEL: @test_vst1_p16_x4(
void test_vst1_p16_x4(poly16_t *a, poly16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p16_x4(a, b);
}

// ALL-LABEL: @test_vst1_p8_x4(
void test_vst1_p8_x4(poly8_t *a, poly8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_p8_x4(a, b);
}

// ALL-LABEL: @test_vst1_s16_x4(
void test_vst1_s16_x4(int16_t *a, int16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s16_x4(a, b);
}

// ALL-LABEL: @test_vst1_s32_x4(
void test_vst1_s32_x4(int32_t *a, int32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v2i32.p0(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s32_x4(a, b);
}

// ALL-LABEL: @test_vst1_s64_x4(
void test_vst1_s64_x4(int64_t *a, int64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s64_x4(a, b);
}

// ALL-LABEL: @test_vst1_s8_x4(
void test_vst1_s8_x4(int8_t *a, int8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_s8_x4(a, b);
}

// ALL-LABEL: @test_vst1_u16_x4(
void test_vst1_u16_x4(uint16_t *a, uint16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v4i16.p0(<4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, <4 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u16_x4(a, b);
}

// ALL-LABEL: @test_vst1_u32_x4(
void test_vst1_u32_x4(uint32_t *a, uint32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v2i32.p0(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u32_x4(a, b);
}

// ALL-LABEL: @test_vst1_u64_x4(
void test_vst1_u64_x4(uint64_t *a, uint64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u64_x4(a, b);
}

// ALL-LABEL: @test_vst1_u8_x4(
void test_vst1_u8_x4(uint8_t *a, uint8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v8i8.p0(<8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, <8 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1_u8_x4(a, b);
}

// ALL-LABEL: @test_vst1q_f16_x2(
void test_vst1q_f16_x2(float16_t *a, float16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v8f16.p0(<8 x half> {{.*}}, <8 x half> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f16_x2(a, b);
}

// ALL-LABEL: @test_vst1q_f32_x2(
void test_vst1q_f32_x2(float32_t *a, float32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v4f32.p0(<4 x float> {{.*}}, <4 x float> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f32_x2(a, b);
}

// ALL-LABEL: @test_vst1q_p16_x2(
void test_vst1q_p16_x2(poly16_t *a, poly16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p16_x2(a, b);
}

// ALL-LABEL: @test_vst1q_p8_x2(
void test_vst1q_p8_x2(poly8_t *a, poly8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p8_x2(a, b);
}

// ALL-LABEL: @test_vst1q_s16_x2(
void test_vst1q_s16_x2(int16_t *a, int16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s16_x2(a, b);
}

// ALL-LABEL: @test_vst1q_s32_x2(
void test_vst1q_s32_x2(int32_t *a, int32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v4i32.p0(<4 x i32> {{.*}}, <4 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s32_x2(a, b);
}

// ALL-LABEL: @test_vst1q_s64_x2(
void test_vst1q_s64_x2(int64_t *a, int64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s64_x2(a, b);
}

// ALL-LABEL: @test_vst1q_s8_x2(
void test_vst1q_s8_x2(int8_t *a, int8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s8_x2(a, b);
}

// ALL-LABEL: @test_vst1q_u16_x2(
void test_vst1q_u16_x2(uint16_t *a, uint16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u16_x2(a, b);
}

// ALL-LABEL: @test_vst1q_u32_x2(
void test_vst1q_u32_x2(uint32_t *a, uint32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v4i32.p0(<4 x i32> {{.*}}, <4 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u32_x2(a, b);
}

// ALL-LABEL: @test_vst1q_u64_x2(
void test_vst1q_u64_x2(uint64_t *a, uint64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u64_x2(a, b);
}

// ALL-LABEL: @test_vst1q_u8_x2(
void test_vst1q_u8_x2(uint8_t *a, uint8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x2" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x2.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u8_x2(a, b);
}

// ALL-LABEL: @test_vst1q_f16_x3(
void test_vst1q_f16_x3(float16_t *a, float16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v8f16.p0(<8 x half> {{.*}}, <8 x half> {{.*}}, <8 x half> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f16_x3(a, b);
}

// ALL-LABEL: @test_vst1q_f32_x3(
void test_vst1q_f32_x3(float32_t *a, float32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v4f32.p0(<4 x float> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f32_x3(a, b);
}

// ALL-LABEL: @test_vst1q_p16_x3(
void test_vst1q_p16_x3(poly16_t *a, poly16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p16_x3(a, b);
}

// ALL-LABEL: @test_vst1q_p8_x3(
void test_vst1q_p8_x3(poly8_t *a, poly8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p8_x3(a, b);
}

// ALL-LABEL: @test_vst1q_s16_x3(
void test_vst1q_s16_x3(int16_t *a, int16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s16_x3(a, b);
}

// ALL-LABEL: @test_vst1q_s32_x3(
void test_vst1q_s32_x3(int32_t *a, int32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v4i32.p0(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s32_x3(a, b);
}

// ALL-LABEL: @test_vst1q_s64_x3(
void test_vst1q_s64_x3(int64_t *a, int64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s64_x3(a, b);
}

// ALL-LABEL: @test_vst1q_s8_x3(
void test_vst1q_s8_x3(int8_t *a, int8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s8_x3(a, b);
}

// ALL-LABEL: @test_vst1q_u16_x3(
void test_vst1q_u16_x3(uint16_t *a, uint16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u16_x3(a, b);
}

// ALL-LABEL: @test_vst1q_u32_x3(
void test_vst1q_u32_x3(uint32_t *a, uint32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v4i32.p0(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u32_x3(a, b);
}

// ALL-LABEL: @test_vst1q_u64_x3(
void test_vst1q_u64_x3(uint64_t *a, uint64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u64_x3(a, b);
}

// ALL-LABEL: @test_vst1q_u8_x3(
void test_vst1q_u8_x3(uint8_t *a, uint8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x3" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x3.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u8_x3(a, b);
}

// ALL-LABEL: @test_vst1q_f16_x4(
void test_vst1q_f16_x4(float16_t *a, float16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v8f16.p0(<8 x half> {{.*}}, <8 x half> {{.*}}, <8 x half> {{.*}}, <8 x half> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f16_x4(a, b);
}

// ALL-LABEL: @test_vst1q_f32_x4(
void test_vst1q_f32_x4(float32_t *a, float32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v4f32.p0(<4 x float> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_f32_x4(a, b);
}

// ALL-LABEL: @test_vst1q_p16_x4(
void test_vst1q_p16_x4(poly16_t *a, poly16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p16_x4(a, b);
}

// ALL-LABEL: @test_vst1q_p8_x4(
void test_vst1q_p8_x4(poly8_t *a, poly8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_p8_x4(a, b);
}

// ALL-LABEL: @test_vst1q_s16_x4(
void test_vst1q_s16_x4(int16_t *a, int16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s16_x4(a, b);
}

// ALL-LABEL: @test_vst1q_s32_x4(
void test_vst1q_s32_x4(int32_t *a, int32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v4i32.p0(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s32_x4(a, b);
}

// ALL-LABEL: @test_vst1q_s64_x4(
void test_vst1q_s64_x4(int64_t *a, int64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s64_x4(a, b);
}

// ALL-LABEL: @test_vst1q_s8_x4(
void test_vst1q_s8_x4(int8_t *a, int8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_s8_x4(a, b);
}

// ALL-LABEL: @test_vst1q_u16_x4(
void test_vst1q_u16_x4(uint16_t *a, uint16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v8i16.p0(<8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, <8 x i16> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u16_x4(a, b);
}

// ALL-LABEL: @test_vst1q_u32_x4(
void test_vst1q_u32_x4(uint32_t *a, uint32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v4i32.p0(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u32_x4(a, b);
}

// ALL-LABEL: @test_vst1q_u64_x4(
void test_vst1q_u64_x4(uint64_t *a, uint64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, <2 x i64> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u64_x4(a, b);
}

// ALL-LABEL: @test_vst1q_u8_x4(
void test_vst1q_u8_x4(uint8_t *a, uint8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st1x4" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM: call void @llvm.aarch64.neon.st1x4.v16i8.p0(<16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i8> {{.*}}, ptr {{.*}})
// LLVM: ret void
  vst1q_u8_x4(a, b);
}

// ALL-LABEL: @test_vstrq_p128(
void test_vstrq_p128(poly128_t *ptr, poly128_t val) {
// CIR: cir.cast bitcast {{.*}} : !cir.ptr<!void> -> !cir.ptr<!u128i>
// CIR: cir.store align(16) {{.*}}, {{.*}} : !u128i, !cir.ptr<!u128i>

// LLVM-SAME: ptr {{.*}} [[PTR:%.*]], i128 {{.*}} [[VAL:%.*]])
// LLVM: store i128 [[VAL]], ptr [[PTR]], align 16
// LLVM: ret void
  vstrq_p128(ptr, val);
}

// ALL-LABEL: @test_vst2_f16(
void test_vst2_f16(float16_t *a, float16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v4f16.p0(<4 x half> [[V0]], <4 x half> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_f16(a, b);
}

// ALL-LABEL: @test_vst2_f32(
void test_vst2_f32(float32_t *a, float32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v2f32.p0(<2 x float> [[V0]], <2 x float> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_f32(a, b);
}

// ALL-LABEL: @test_vst2_f64(
void test_vst2_f64(float64_t *a, float64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v1f64.p0(<1 x double> [[V0]], <1 x double> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_f64(a, b);
}

// ALL-LABEL: @test_vst2_mf8(
void test_vst2_mf8(mfloat8_t *a, mfloat8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_mf8(a, b);
}

// ALL-LABEL: @test_vst2_p16(
void test_vst2_p16(poly16_t *a, poly16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_p16(a, b);
}

// ALL-LABEL: @test_vst2_p64(
void test_vst2_p64(poly64_t * ptr, poly64x1x2_t val) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_p64(ptr, val);
}

// ALL-LABEL: @test_vst2_p8(
void test_vst2_p8(poly8_t *a, poly8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_p8(a, b);
}

// ALL-LABEL: @test_vst2_s16(
void test_vst2_s16(int16_t *a, int16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_s16(a, b);
}

// ALL-LABEL: @test_vst2_s32(
void test_vst2_s32(int32_t *a, int32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_s32(a, b);
}

// ALL-LABEL: @test_vst2_s64(
void test_vst2_s64(int64_t *a, int64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_s64(a, b);
}

// ALL-LABEL: @test_vst2_s8(
void test_vst2_s8(int8_t *a, int8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_s8(a, b);
}

// ALL-LABEL: @test_vst2_u16(
void test_vst2_u16(uint16_t *a, uint16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_u16(a, b);
}

// ALL-LABEL: @test_vst2_u32(
void test_vst2_u32(uint32_t *a, uint32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_u32(a, b);
}

// ALL-LABEL: @test_vst2_u64(
void test_vst2_u64(uint64_t *a, uint64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_u64(a, b);
}

// ALL-LABEL: @test_vst2_u8(
void test_vst2_u8(uint8_t *a, uint8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2_u8(a, b);
}

// ALL-LABEL: @test_vst2q_f16(
void test_vst2q_f16(float16_t *a, float16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v8f16.p0(<8 x half> [[V0]], <8 x half> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_f16(a, b);
}

// ALL-LABEL: @test_vst2q_f32(
void test_vst2q_f32(float32_t *a, float32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v4f32.p0(<4 x float> [[V0]], <4 x float> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_f32(a, b);
}

// ALL-LABEL: @test_vst2q_f64(
void test_vst2q_f64(float64_t *a, float64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v2f64.p0(<2 x double> [[V0]], <2 x double> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_f64(a, b);
}

// ALL-LABEL: @test_vst2q_mf8(
void test_vst2q_mf8(mfloat8_t *a, mfloat8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_mf8(a, b);
}

// ALL-LABEL: @test_vst2q_p16(
void test_vst2q_p16(poly16_t *a, poly16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_p16(a, b);
}

// ALL-LABEL: @test_vst2q_p64(
void test_vst2q_p64(poly64_t * ptr, poly64x2x2_t val) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_p64(ptr, val);
}

// ALL-LABEL: @test_vst2q_p8(
void test_vst2q_p8(poly8_t *a, poly8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_p8(a, b);
}

// ALL-LABEL: @test_vst2q_s16(
void test_vst2q_s16(int16_t *a, int16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_s16(a, b);
}

// ALL-LABEL: @test_vst2q_s32(
void test_vst2q_s32(int32_t *a, int32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_s32(a, b);
}

// ALL-LABEL: @test_vst2q_s64(
void test_vst2q_s64(int64_t *a, int64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_s64(a, b);
}

// ALL-LABEL: @test_vst2q_s8(
void test_vst2q_s8(int8_t *a, int8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_s8(a, b);
}

// ALL-LABEL: @test_vst2q_u16(
void test_vst2q_u16(uint16_t *a, uint16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_u16(a, b);
}

// ALL-LABEL: @test_vst2q_u32(
void test_vst2q_u32(uint32_t *a, uint32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_u32(a, b);
}

// ALL-LABEL: @test_vst2q_u64(
void test_vst2q_u64(uint64_t *a, uint64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_u64(a, b);
}

// ALL-LABEL: @test_vst2q_u8(
void test_vst2q_u8(uint8_t *a, uint8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], ptr [[A]])
// LLVM: ret void
  vst2q_u8(a, b);
}

// ALL-LABEL: @test_vst2_lane_f16(
void test_vst2_lane_f16(float16_t  *a, float16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v4f16.p0(<4 x half> [[V0]], <4 x half> [[V1]], i64 3, ptr [[A]])
// LLVM: ret void
  vst2_lane_f16(a, b, 3);
}

// ALL-LABEL: @test_vst2_lane_f32(
void test_vst2_lane_f32(float32_t  *a, float32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v2f32.p0(<2 x float> [[V0]], <2 x float> [[V1]], i64 1, ptr [[A]])
// LLVM: ret void
  vst2_lane_f32(a, b, 1);
}

// ALL-LABEL: @test_vst2_lane_f64(
void test_vst2_lane_f64(float64_t  *a, float64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v1f64.p0(<1 x double> [[V0]], <1 x double> [[V1]], i64 0, ptr [[A]])
// LLVM: ret void
  vst2_lane_f64(a, b, 0);
}

// ALL-LABEL: @test_vst2_lane_mf8(
void test_vst2_lane_mf8(mfloat8_t *a, mfloat8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], i64 7, ptr [[A]])
// LLVM: ret void
  vst2_lane_mf8(a, b, 7);
}

// ALL-LABEL: @test_vst2_lane_p16(
void test_vst2_lane_p16(poly16_t  *a, poly16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], i64 3, ptr [[A]])
// LLVM: ret void
  vst2_lane_p16(a, b, 3);
}

// ALL-LABEL: @test_vst2_lane_p64(
void test_vst2_lane_p64(poly64_t  *a, poly64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], i64 0, ptr [[A]])
// LLVM: ret void
  vst2_lane_p64(a, b, 0);
}

// ALL-LABEL: @test_vst2_lane_p8(
void test_vst2_lane_p8(poly8_t  *a, poly8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], i64 7, ptr [[A]])
// LLVM: ret void
  vst2_lane_p8(a, b, 7);
}

// ALL-LABEL: @test_vst2_lane_s16(
void test_vst2_lane_s16(int16_t  *a, int16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], i64 3, ptr [[A]])
// LLVM: ret void
  vst2_lane_s16(a, b, 3);
}

// ALL-LABEL: @test_vst2_lane_s32(
void test_vst2_lane_s32(int32_t  *a, int32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], i64 1, ptr [[A]])
// LLVM: ret void
  vst2_lane_s32(a, b, 1);
}

// ALL-LABEL: @test_vst2_lane_s64(
void test_vst2_lane_s64(int64_t  *a, int64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], i64 0, ptr [[A]])
// LLVM: ret void
  vst2_lane_s64(a, b, 0);
}

// ALL-LABEL: @test_vst2_lane_s8(
void test_vst2_lane_s8(int8_t  *a, int8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], i64 7, ptr [[A]])
// LLVM: ret void
  vst2_lane_s8(a, b, 7);
}

// ALL-LABEL: @test_vst2_lane_u16(
void test_vst2_lane_u16(uint16_t  *a, uint16x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], i64 3, ptr [[A]])
// LLVM: ret void
  vst2_lane_u16(a, b, 3);
}

// ALL-LABEL: @test_vst2_lane_u32(
void test_vst2_lane_u32(uint32_t  *a, uint32x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], i64 1, ptr [[A]])
// LLVM: ret void
  vst2_lane_u32(a, b, 1);
}

// ALL-LABEL: @test_vst2_lane_u64(
void test_vst2_lane_u64(uint64_t  *a, uint64x1x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], i64 0, ptr [[A]])
// LLVM: ret void
  vst2_lane_u64(a, b, 0);
}

// ALL-LABEL: @test_vst2_lane_u8(
void test_vst2_lane_u8(uint8_t  *a, uint8x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], i64 7, ptr [[A]])
// LLVM: ret void
  vst2_lane_u8(a, b, 7);
}

// ALL-LABEL: @test_vst2q_lane_f16(
void test_vst2q_lane_f16(float16_t  *a, float16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v8f16.p0(<8 x half> [[V0]], <8 x half> [[V1]], i64 7, ptr [[A]])
// LLVM: ret void
  vst2q_lane_f16(a, b, 7);
}

// ALL-LABEL: @test_vst2q_lane_f32(
void test_vst2q_lane_f32(float32_t  *a, float32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v4f32.p0(<4 x float> [[V0]], <4 x float> [[V1]], i64 3, ptr [[A]])
// LLVM: ret void
  vst2q_lane_f32(a, b, 3);
}

// ALL-LABEL: @test_vst2q_lane_f64(
void test_vst2q_lane_f64(float64_t  *a, float64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v2f64.p0(<2 x double> [[V0]], <2 x double> [[V1]], i64 1, ptr [[A]])
// LLVM: ret void
  vst2q_lane_f64(a, b, 1);
}

// ALL-LABEL: @test_vst2q_lane_mf8(
void test_vst2q_lane_mf8(mfloat8_t *a, mfloat8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], i64 15, ptr [[A]])
// LLVM: ret void
  vst2q_lane_mf8(a, b, 15);
}

// ALL-LABEL: @test_vst2q_lane_p16(
void test_vst2q_lane_p16(poly16_t  *a, poly16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], i64 7, ptr [[A]])
// LLVM: ret void
  vst2q_lane_p16(a, b, 7);
}

// ALL-LABEL: @test_vst2q_lane_p64(
void test_vst2q_lane_p64(poly64_t  *a, poly64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], i64 1, ptr [[A]])
// LLVM: ret void
  vst2q_lane_p64(a, b, 1);
}

// ALL-LABEL: @test_vst2q_lane_p8(
void test_vst2q_lane_p8(poly8_t  *a, poly8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], i64 15, ptr [[A]])
// LLVM: ret void
  vst2q_lane_p8(a, b, 15);
}

// ALL-LABEL: @test_vst2q_lane_s16(
void test_vst2q_lane_s16(int16_t  *a, int16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], i64 7, ptr [[A]])
// LLVM: ret void
  vst2q_lane_s16(a, b, 7);
}

// ALL-LABEL: @test_vst2q_lane_s32(
void test_vst2q_lane_s32(int32_t  *a, int32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], i64 3, ptr [[A]])
// LLVM: ret void
  vst2q_lane_s32(a, b, 3);
}

// ALL-LABEL: @test_vst2q_lane_s64(
void test_vst2q_lane_s64(int64_t  *a, int64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], i64 1, ptr [[A]])
// LLVM: ret void
  vst2q_lane_s64(a, b, 1);
}

// ALL-LABEL: @test_vst2q_lane_s8(
void test_vst2q_lane_s8(int8_t  *a, int8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], i64 15, ptr [[A]])
// LLVM: ret void
  vst2q_lane_s8(a, b, 15);
}

// ALL-LABEL: @test_vst2q_lane_u16(
void test_vst2q_lane_u16(uint16_t  *a, uint16x8x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], i64 7, ptr [[A]])
// LLVM: ret void
  vst2q_lane_u16(a, b, 7);
}

// ALL-LABEL: @test_vst2q_lane_u32(
void test_vst2q_lane_u32(uint32_t  *a, uint32x4x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], i64 3, ptr [[A]])
// LLVM: ret void
  vst2q_lane_u32(a, b, 3);
}

// ALL-LABEL: @test_vst2q_lane_u64(
void test_vst2q_lane_u64(uint64_t  *a, uint64x2x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], i64 1, ptr [[A]])
// LLVM: ret void
  vst2q_lane_u64(a, b, 1);
}

// ALL-LABEL: @test_vst2q_lane_u8(
void test_vst2q_lane_u8(uint8_t  *a, uint8x16x2_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st2lane" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM: call void @llvm.aarch64.neon.st2lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], i64 15, ptr [[A]])
// LLVM: ret void
  vst2q_lane_u8(a, b, 15);
}

// ALL-LABEL: @test_vst3_f16(
void test_vst3_f16(float16_t *a, float16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v4f16.p0(<4 x half> [[V0]], <4 x half> [[V1]], <4 x half> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_f16(a, b);
}

// ALL-LABEL: @test_vst3_f32(
void test_vst3_f32(float32_t *a, float32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v2f32.p0(<2 x float> [[V0]], <2 x float> [[V1]], <2 x float> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_f32(a, b);
}

// ALL-LABEL: @test_vst3_f64(
void test_vst3_f64(float64_t *a, float64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v1f64.p0(<1 x double> [[V0]], <1 x double> [[V1]], <1 x double> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_f64(a, b);
}

// ALL-LABEL: @test_vst3_mf8(
void test_vst3_mf8(mfloat8_t *a, mfloat8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_mf8(a, b);
}

// ALL-LABEL: @test_vst3_p16(
void test_vst3_p16(poly16_t *a, poly16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_p16(a, b);
}

// ALL-LABEL: @test_vst3_p64(
void test_vst3_p64(poly64_t * ptr, poly64x1x3_t val) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_p64(ptr, val);
}

// ALL-LABEL: @test_vst3_p8(
void test_vst3_p8(poly8_t *a, poly8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_p8(a, b);
}

// ALL-LABEL: @test_vst3_s16(
void test_vst3_s16(int16_t *a, int16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_s16(a, b);
}

// ALL-LABEL: @test_vst3_s32(
void test_vst3_s32(int32_t *a, int32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], <2 x i32> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_s32(a, b);
}

// ALL-LABEL: @test_vst3_s64(
void test_vst3_s64(int64_t *a, int64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_s64(a, b);
}

// ALL-LABEL: @test_vst3_s8(
void test_vst3_s8(int8_t *a, int8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_s8(a, b);
}

// ALL-LABEL: @test_vst3_u16(
void test_vst3_u16(uint16_t *a, uint16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_u16(a, b);
}

// ALL-LABEL: @test_vst3_u32(
void test_vst3_u32(uint32_t *a, uint32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], <2 x i32> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_u32(a, b);
}

// ALL-LABEL: @test_vst3_u64(
void test_vst3_u64(uint64_t *a, uint64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_u64(a, b);
}

// ALL-LABEL: @test_vst3_u8(
void test_vst3_u8(uint8_t *a, uint8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3_u8(a, b);
}

// ALL-LABEL: @test_vst3q_f16(
void test_vst3q_f16(float16_t *a, float16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v8f16.p0(<8 x half> [[V0]], <8 x half> [[V1]], <8 x half> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_f16(a, b);
}

// ALL-LABEL: @test_vst3q_f32(
void test_vst3q_f32(float32_t *a, float32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v4f32.p0(<4 x float> [[V0]], <4 x float> [[V1]], <4 x float> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_f32(a, b);
}

// ALL-LABEL: @test_vst3q_f64(
void test_vst3q_f64(float64_t *a, float64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v2f64.p0(<2 x double> [[V0]], <2 x double> [[V1]], <2 x double> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_f64(a, b);
}

// ALL-LABEL: @test_vst3q_mf8(
void test_vst3q_mf8(mfloat8_t *a, mfloat8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_mf8(a, b);
}

// ALL-LABEL: @test_vst3q_p16(
void test_vst3q_p16(poly16_t *a, poly16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_p16(a, b);
}

// ALL-LABEL: @test_vst3q_p64(
void test_vst3q_p64(poly64_t * ptr, poly64x2x3_t val) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_p64(ptr, val);
}

// ALL-LABEL: @test_vst3q_p8(
void test_vst3q_p8(poly8_t *a, poly8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_p8(a, b);
}

// ALL-LABEL: @test_vst3q_s16(
void test_vst3q_s16(int16_t *a, int16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_s16(a, b);
}

// ALL-LABEL: @test_vst3q_s32(
void test_vst3q_s32(int32_t *a, int32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], <4 x i32> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_s32(a, b);
}

// ALL-LABEL: @test_vst3q_s64(
void test_vst3q_s64(int64_t *a, int64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_s64(a, b);
}

// ALL-LABEL: @test_vst3q_s8(
void test_vst3q_s8(int8_t *a, int8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_s8(a, b);
}

// ALL-LABEL: @test_vst3q_u16(
void test_vst3q_u16(uint16_t *a, uint16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_u16(a, b);
}

// ALL-LABEL: @test_vst3q_u32(
void test_vst3q_u32(uint32_t *a, uint32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], <4 x i32> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_u32(a, b);
}

// ALL-LABEL: @test_vst3q_u64(
void test_vst3q_u64(uint64_t *a, uint64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_u64(a, b);
}

// ALL-LABEL: @test_vst3q_u8(
void test_vst3q_u8(uint8_t *a, uint8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], ptr [[A]])
// LLVM: ret void
  vst3q_u8(a, b);
}

// ALL-LABEL: @test_vst3_lane_f16(
void test_vst3_lane_f16(float16_t  *a, float16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v4f16.p0(<4 x half> [[V0]], <4 x half> [[V1]], <4 x half> [[V2]], i64 3, ptr [[A]])
// LLVM: ret void
  vst3_lane_f16(a, b, 3);
}

// ALL-LABEL: @test_vst3_lane_f32(
void test_vst3_lane_f32(float32_t  *a, float32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v2f32.p0(<2 x float> [[V0]], <2 x float> [[V1]], <2 x float> [[V2]], i64 1, ptr [[A]])
// LLVM: ret void
  vst3_lane_f32(a, b, 1);
}

// ALL-LABEL: @test_vst3_lane_f64(
void test_vst3_lane_f64(float64_t  *a, float64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v1f64.p0(<1 x double> [[V0]], <1 x double> [[V1]], <1 x double> [[V2]], i64 0, ptr [[A]])
// LLVM: ret void
  vst3_lane_f64(a, b, 0);
}

// ALL-LABEL: @test_vst3_lane_mf8(
void test_vst3_lane_mf8(mfloat8_t *a, mfloat8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], i64 7, ptr [[A]])
// LLVM: ret void
  vst3_lane_mf8(a, b, 7);
}

// ALL-LABEL: @test_vst3_lane_p16(
void test_vst3_lane_p16(poly16_t  *a, poly16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], i64 3, ptr [[A]])
// LLVM: ret void
  vst3_lane_p16(a, b, 3);
}

// ALL-LABEL: @test_vst3_lane_p64(
void test_vst3_lane_p64(poly64_t  *a, poly64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], i64 0, ptr [[A]])
// LLVM: ret void
  vst3_lane_p64(a, b, 0);
}

// ALL-LABEL: @test_vst3_lane_p8(
void test_vst3_lane_p8(poly8_t  *a, poly8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], i64 7, ptr [[A]])
// LLVM: ret void
  vst3_lane_p8(a, b, 7);
}

// ALL-LABEL: @test_vst3_lane_s16(
void test_vst3_lane_s16(int16_t  *a, int16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], i64 3, ptr [[A]])
// LLVM: ret void
  vst3_lane_s16(a, b, 3);
}

// ALL-LABEL: @test_vst3_lane_s32(
void test_vst3_lane_s32(int32_t  *a, int32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], <2 x i32> [[V2]], i64 1, ptr [[A]])
// LLVM: ret void
  vst3_lane_s32(a, b, 1);
}

// ALL-LABEL: @test_vst3_lane_s64(
void test_vst3_lane_s64(int64_t  *a, int64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], i64 0, ptr [[A]])
// LLVM: ret void
  vst3_lane_s64(a, b, 0);
}

// ALL-LABEL: @test_vst3_lane_s8(
void test_vst3_lane_s8(int8_t  *a, int8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], i64 7, ptr [[A]])
// LLVM: ret void
  vst3_lane_s8(a, b, 7);
}

// ALL-LABEL: @test_vst3_lane_u16(
void test_vst3_lane_u16(uint16_t  *a, uint16x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], i64 3, ptr [[A]])
// LLVM: ret void
  vst3_lane_u16(a, b, 3);
}

// ALL-LABEL: @test_vst3_lane_u32(
void test_vst3_lane_u32(uint32_t  *a, uint32x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], <2 x i32> [[V2]], i64 1, ptr [[A]])
// LLVM: ret void
  vst3_lane_u32(a, b, 1);
}

// ALL-LABEL: @test_vst3_lane_u64(
void test_vst3_lane_u64(uint64_t  *a, uint64x1x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], i64 0, ptr [[A]])
// LLVM: ret void
  vst3_lane_u64(a, b, 0);
}

// ALL-LABEL: @test_vst3_lane_u8(
void test_vst3_lane_u8(uint8_t  *a, uint8x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], i64 7, ptr [[A]])
// LLVM: ret void
  vst3_lane_u8(a, b, 7);
}

// ALL-LABEL: @test_vst3q_lane_f16(
void test_vst3q_lane_f16(float16_t  *a, float16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v8f16.p0(<8 x half> [[V0]], <8 x half> [[V1]], <8 x half> [[V2]], i64 7, ptr [[A]])
// LLVM: ret void
  vst3q_lane_f16(a, b, 7);
}

// ALL-LABEL: @test_vst3q_lane_f32(
void test_vst3q_lane_f32(float32_t  *a, float32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v4f32.p0(<4 x float> [[V0]], <4 x float> [[V1]], <4 x float> [[V2]], i64 3, ptr [[A]])
// LLVM: ret void
  vst3q_lane_f32(a, b, 3);
}

// ALL-LABEL: @test_vst3q_lane_f64(
void test_vst3q_lane_f64(float64_t  *a, float64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v2f64.p0(<2 x double> [[V0]], <2 x double> [[V1]], <2 x double> [[V2]], i64 1, ptr [[A]])
// LLVM: ret void
  vst3q_lane_f64(a, b, 1);
}

// ALL-LABEL: @test_vst3q_lane_mf8(
void test_vst3q_lane_mf8(mfloat8_t *a, mfloat8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], i64 15, ptr [[A]])
// LLVM: ret void
  vst3q_lane_mf8(a, b, 15);
}

// ALL-LABEL: @test_vst3q_lane_p16(
void test_vst3q_lane_p16(poly16_t  *a, poly16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], i64 7, ptr [[A]])
// LLVM: ret void
  vst3q_lane_p16(a, b, 7);
}

// ALL-LABEL: @test_vst3q_lane_p64(
void test_vst3q_lane_p64(poly64_t  *a, poly64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], i64 1, ptr [[A]])
// LLVM: ret void
  vst3q_lane_p64(a, b, 1);
}

// ALL-LABEL: @test_vst3q_lane_p8(
void test_vst3q_lane_p8(poly8_t  *a, poly8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], i64 15, ptr [[A]])
// LLVM: ret void
  vst3q_lane_p8(a, b, 15);
}

// ALL-LABEL: @test_vst3q_lane_s16(
void test_vst3q_lane_s16(int16_t  *a, int16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], i64 7, ptr [[A]])
// LLVM: ret void
  vst3q_lane_s16(a, b, 7);
}

// ALL-LABEL: @test_vst3q_lane_s32(
void test_vst3q_lane_s32(int32_t  *a, int32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], <4 x i32> [[V2]], i64 3, ptr [[A]])
// LLVM: ret void
  vst3q_lane_s32(a, b, 3);
}

// ALL-LABEL: @test_vst3q_lane_s64(
void test_vst3q_lane_s64(int64_t  *a, int64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], i64 1, ptr [[A]])
// LLVM: ret void
  vst3q_lane_s64(a, b, 1);
}

// ALL-LABEL: @test_vst3q_lane_s8(
void test_vst3q_lane_s8(int8_t  *a, int8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], i64 15, ptr [[A]])
// LLVM: ret void
  vst3q_lane_s8(a, b, 15);
}

// ALL-LABEL: @test_vst3q_lane_u16(
void test_vst3q_lane_u16(uint16_t  *a, uint16x8x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], i64 7, ptr [[A]])
// LLVM: ret void
  vst3q_lane_u16(a, b, 7);
}

// ALL-LABEL: @test_vst3q_lane_u32(
void test_vst3q_lane_u32(uint32_t  *a, uint32x4x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], <4 x i32> [[V2]], i64 3, ptr [[A]])
// LLVM: ret void
  vst3q_lane_u32(a, b, 3);
}

// ALL-LABEL: @test_vst3q_lane_u64(
void test_vst3q_lane_u64(uint64_t  *a, uint64x2x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], i64 1, ptr [[A]])
// LLVM: ret void
  vst3q_lane_u64(a, b, 1);
}

// ALL-LABEL: @test_vst3q_lane_u8(
void test_vst3q_lane_u8(uint8_t  *a, uint8x16x3_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st3lane" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM: call void @llvm.aarch64.neon.st3lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], i64 15, ptr [[A]])
// LLVM: ret void
  vst3q_lane_u8(a, b, 15);
}

// ALL-LABEL: @test_vst4_f16(
void test_vst4_f16(float16_t *a, float16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v4f16.p0(<4 x half> [[V0]], <4 x half> [[V1]], <4 x half> [[V2]], <4 x half> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_f16(a, b);
}

// ALL-LABEL: @test_vst4_f32(
void test_vst4_f32(float32_t *a, float32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v2f32.p0(<2 x float> [[V0]], <2 x float> [[V1]], <2 x float> [[V2]], <2 x float> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_f32(a, b);
}

// ALL-LABEL: @test_vst4_f64(
void test_vst4_f64(float64_t *a, float64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v1f64.p0(<1 x double> [[V0]], <1 x double> [[V1]], <1 x double> [[V2]], <1 x double> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_f64(a, b);
}

// ALL-LABEL: @test_vst4_mf8(
void test_vst4_mf8(mfloat8_t *a, mfloat8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], <8 x i8> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_mf8(a, b);
}

// ALL-LABEL: @test_vst4_p16(
void test_vst4_p16(poly16_t *a, poly16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], <4 x i16> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_p16(a, b);
}

// ALL-LABEL: @test_vst4_p64(
void test_vst4_p64(poly64_t * ptr, poly64x1x4_t val) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], <1 x i64> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_p64(ptr, val);
}

// ALL-LABEL: @test_vst4_p8(
void test_vst4_p8(poly8_t *a, poly8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], <8 x i8> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_p8(a, b);
}

// ALL-LABEL: @test_vst4_s16(
void test_vst4_s16(int16_t *a, int16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], <4 x i16> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_s16(a, b);
}

// ALL-LABEL: @test_vst4_s32(
void test_vst4_s32(int32_t *a, int32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], <2 x i32> [[V2]], <2 x i32> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_s32(a, b);
}

// ALL-LABEL: @test_vst4_s64(
void test_vst4_s64(int64_t *a, int64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], <1 x i64> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_s64(a, b);
}

// ALL-LABEL: @test_vst4_s8(
void test_vst4_s8(int8_t *a, int8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], <8 x i8> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_s8(a, b);
}

// ALL-LABEL: @test_vst4_u16(
void test_vst4_u16(uint16_t *a, uint16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], <4 x i16> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_u16(a, b);
}

// ALL-LABEL: @test_vst4_u32(
void test_vst4_u32(uint32_t *a, uint32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], <2 x i32> [[V2]], <2 x i32> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_u32(a, b);
}

// ALL-LABEL: @test_vst4_u64(
void test_vst4_u64(uint64_t *a, uint64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], <1 x i64> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_u64(a, b);
}

// ALL-LABEL: @test_vst4_u8(
void test_vst4_u8(uint8_t *a, uint8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], <8 x i8> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4_u8(a, b);
}

// ALL-LABEL: @test_vst4q_f16(
void test_vst4q_f16(float16_t *a, float16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v8f16.p0(<8 x half> [[V0]], <8 x half> [[V1]], <8 x half> [[V2]], <8 x half> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_f16(a, b);
}

// ALL-LABEL: @test_vst4q_f32(
void test_vst4q_f32(float32_t *a, float32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v4f32.p0(<4 x float> [[V0]], <4 x float> [[V1]], <4 x float> [[V2]], <4 x float> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_f32(a, b);
}

// ALL-LABEL: @test_vst4q_f64(
void test_vst4q_f64(float64_t *a, float64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v2f64.p0(<2 x double> [[V0]], <2 x double> [[V1]], <2 x double> [[V2]], <2 x double> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_f64(a, b);
}

// ALL-LABEL: @test_vst4q_mf8(
void test_vst4q_mf8(mfloat8_t *a, mfloat8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], <16 x i8> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_mf8(a, b);
}

// ALL-LABEL: @test_vst4q_p16(
void test_vst4q_p16(poly16_t *a, poly16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], <8 x i16> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_p16(a, b);
}

// ALL-LABEL: @test_vst4q_p64(
void test_vst4q_p64(poly64_t * ptr, poly64x2x4_t val) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], <2 x i64> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_p64(ptr, val);
}

// ALL-LABEL: @test_vst4q_p8(
void test_vst4q_p8(poly8_t *a, poly8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], <16 x i8> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_p8(a, b);
}

// ALL-LABEL: @test_vst4q_s16(
void test_vst4q_s16(int16_t *a, int16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], <8 x i16> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_s16(a, b);
}

// ALL-LABEL: @test_vst4q_s32(
void test_vst4q_s32(int32_t *a, int32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], <4 x i32> [[V2]], <4 x i32> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_s32(a, b);
}

// ALL-LABEL: @test_vst4q_s64(
void test_vst4q_s64(int64_t *a, int64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], <2 x i64> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_s64(a, b);
}

// ALL-LABEL: @test_vst4q_s8(
void test_vst4q_s8(int8_t *a, int8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], <16 x i8> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_s8(a, b);
}

// ALL-LABEL: @test_vst4q_u16(
void test_vst4q_u16(uint16_t *a, uint16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], <8 x i16> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_u16(a, b);
}

// ALL-LABEL: @test_vst4q_u32(
void test_vst4q_u32(uint32_t *a, uint32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], <4 x i32> [[V2]], <4 x i32> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_u32(a, b);
}

// ALL-LABEL: @test_vst4q_u64(
void test_vst4q_u64(uint64_t *a, uint64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], <2 x i64> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_u64(a, b);
}

// ALL-LABEL: @test_vst4q_u8(
void test_vst4q_u8(uint8_t *a, uint8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], <16 x i8> [[V3]], ptr [[A]])
// LLVM: ret void
  vst4q_u8(a, b);
}

// ALL-LABEL: @test_vst4_lane_f16(
void test_vst4_lane_f16(float16_t  *a, float16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !cir.vector<4 x !cir.f16>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v4f16.p0(<4 x half> [[V0]], <4 x half> [[V1]], <4 x half> [[V2]], <4 x half> [[V3]], i64 3, ptr [[A]])
// LLVM: ret void
  vst4_lane_f16(a, b, 3);
}

// ALL-LABEL: @test_vst4_lane_f32(
void test_vst4_lane_f32(float32_t  *a, float32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !cir.vector<2 x !cir.float>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v2f32.p0(<2 x float> [[V0]], <2 x float> [[V1]], <2 x float> [[V2]], <2 x float> [[V3]], i64 1, ptr [[A]])
// LLVM: ret void
  vst4_lane_f32(a, b, 1);
}

// ALL-LABEL: @test_vst4_lane_f64(
void test_vst4_lane_f64(float64_t  *a, float64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !cir.vector<1 x !cir.double>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v1f64.p0(<1 x double> [[V0]], <1 x double> [[V1]], <1 x double> [[V2]], <1 x double> [[V3]], i64 0, ptr [[A]])
// LLVM: ret void
  vst4_lane_f64(a, b, 0);
}

// ALL-LABEL: @test_vst4_lane_mf8(
void test_vst4_lane_mf8(mfloat8_t *a, mfloat8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], <8 x i8> [[V3]], i64 7, ptr [[A]])
// LLVM: ret void
  vst4_lane_mf8(a, b, 7);
}

// ALL-LABEL: @test_vst4_lane_p16(
void test_vst4_lane_p16(poly16_t  *a, poly16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], <4 x i16> [[V3]], i64 3, ptr [[A]])
// LLVM: ret void
  vst4_lane_p16(a, b, 3);
}

// ALL-LABEL: @test_vst4_lane_p64(
void test_vst4_lane_p64(poly64_t  *a, poly64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], <1 x i64> [[V3]], i64 0, ptr [[A]])
// LLVM: ret void
  vst4_lane_p64(a, b, 0);
}

// ALL-LABEL: @test_vst4_lane_p8(
void test_vst4_lane_p8(poly8_t  *a, poly8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], <8 x i8> [[V3]], i64 7, ptr [[A]])
// LLVM: ret void
  vst4_lane_p8(a, b, 7);
}

// ALL-LABEL: @test_vst4_lane_s16(
void test_vst4_lane_s16(int16_t  *a, int16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !cir.vector<4 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], <4 x i16> [[V3]], i64 3, ptr [[A]])
// LLVM: ret void
  vst4_lane_s16(a, b, 3);
}

// ALL-LABEL: @test_vst4_lane_s32(
void test_vst4_lane_s32(int32_t  *a, int32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !cir.vector<2 x !s32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], <2 x i32> [[V2]], <2 x i32> [[V3]], i64 1, ptr [[A]])
// LLVM: ret void
  vst4_lane_s32(a, b, 1);
}

// ALL-LABEL: @test_vst4_lane_s64(
void test_vst4_lane_s64(int64_t  *a, int64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !cir.vector<1 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], <1 x i64> [[V3]], i64 0, ptr [[A]])
// LLVM: ret void
  vst4_lane_s64(a, b, 0);
}

// ALL-LABEL: @test_vst4_lane_s8(
void test_vst4_lane_s8(int8_t  *a, int8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !cir.vector<8 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], <8 x i8> [[V3]], i64 7, ptr [[A]])
// LLVM: ret void
  vst4_lane_s8(a, b, 7);
}

// ALL-LABEL: @test_vst4_lane_u16(
void test_vst4_lane_u16(uint16_t  *a, uint16x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !cir.vector<4 x !u16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v4i16.p0(<4 x i16> [[V0]], <4 x i16> [[V1]], <4 x i16> [[V2]], <4 x i16> [[V3]], i64 3, ptr [[A]])
// LLVM: ret void
  vst4_lane_u16(a, b, 3);
}

// ALL-LABEL: @test_vst4_lane_u32(
void test_vst4_lane_u32(uint32_t  *a, uint32x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !cir.vector<2 x !u32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v2i32.p0(<2 x i32> [[V0]], <2 x i32> [[V1]], <2 x i32> [[V2]], <2 x i32> [[V3]], i64 1, ptr [[A]])
// LLVM: ret void
  vst4_lane_u32(a, b, 1);
}

// ALL-LABEL: @test_vst4_lane_u64(
void test_vst4_lane_u64(uint64_t  *a, uint64x1x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !cir.vector<1 x !u64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v1i64.p0(<1 x i64> [[V0]], <1 x i64> [[V1]], <1 x i64> [[V2]], <1 x i64> [[V3]], i64 0, ptr [[A]])
// LLVM: ret void
  vst4_lane_u64(a, b, 0);
}

// ALL-LABEL: @test_vst4_lane_u8(
void test_vst4_lane_u8(uint8_t  *a, uint8x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !cir.vector<8 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v8i8.p0(<8 x i8> [[V0]], <8 x i8> [[V1]], <8 x i8> [[V2]], <8 x i8> [[V3]], i64 7, ptr [[A]])
// LLVM: ret void
  vst4_lane_u8(a, b, 7);
}

// ALL-LABEL: @test_vst4q_lane_f16(
void test_vst4q_lane_f16(float16_t  *a, float16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.f16>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v8f16.p0(<8 x half> [[V0]], <8 x half> [[V1]], <8 x half> [[V2]], <8 x half> [[V3]], i64 7, ptr [[A]])
// LLVM: ret void
  vst4q_lane_f16(a, b, 7);
}

// ALL-LABEL: @test_vst4q_lane_f32(
void test_vst4q_lane_f32(float32_t  *a, float32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v4f32.p0(<4 x float> [[V0]], <4 x float> [[V1]], <4 x float> [[V2]], <4 x float> [[V3]], i64 3, ptr [[A]])
// LLVM: ret void
  vst4q_lane_f32(a, b, 3);
}

// ALL-LABEL: @test_vst4q_lane_f64(
void test_vst4q_lane_f64(float64_t  *a, float64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v2f64.p0(<2 x double> [[V0]], <2 x double> [[V1]], <2 x double> [[V2]], <2 x double> [[V3]], i64 1, ptr [[A]])
// LLVM: ret void
  vst4q_lane_f64(a, b, 1);
}

// ALL-LABEL: @test_vst4q_lane_mf8(
void test_vst4q_lane_mf8(mfloat8_t *a, mfloat8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], <16 x i8> [[V3]], i64 15, ptr [[A]])
// LLVM: ret void
  vst4q_lane_mf8(a, b, 15);
}

// ALL-LABEL: @test_vst4q_lane_p16(
void test_vst4q_lane_p16(poly16_t  *a, poly16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], <8 x i16> [[V3]], i64 7, ptr [[A]])
// LLVM: ret void
  vst4q_lane_p16(a, b, 7);
}

// ALL-LABEL: @test_vst4q_lane_p64(
void test_vst4q_lane_p64(poly64_t  *a, poly64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], <2 x i64> [[V3]], i64 1, ptr [[A]])
// LLVM: ret void
  vst4q_lane_p64(a, b, 1);
}

// ALL-LABEL: @test_vst4q_lane_p8(
void test_vst4q_lane_p8(poly8_t  *a, poly8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], <16 x i8> [[V3]], i64 15, ptr [[A]])
// LLVM: ret void
  vst4q_lane_p8(a, b, 15);
}

// ALL-LABEL: @test_vst4q_lane_s16(
void test_vst4q_lane_s16(int16_t  *a, int16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !cir.vector<8 x !s16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], <8 x i16> [[V3]], i64 7, ptr [[A]])
// LLVM: ret void
  vst4q_lane_s16(a, b, 7);
}

// ALL-LABEL: @test_vst4q_lane_s32(
void test_vst4q_lane_s32(int32_t  *a, int32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], <4 x i32> [[V2]], <4 x i32> [[V3]], i64 3, ptr [[A]])
// LLVM: ret void
  vst4q_lane_s32(a, b, 3);
}

// ALL-LABEL: @test_vst4q_lane_s64(
void test_vst4q_lane_s64(int64_t  *a, int64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], <2 x i64> [[V3]], i64 1, ptr [[A]])
// LLVM: ret void
  vst4q_lane_s64(a, b, 1);
}

// ALL-LABEL: @test_vst4q_lane_s8(
void test_vst4q_lane_s8(int8_t  *a, int8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !cir.vector<16 x !s8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], <16 x i8> [[V3]], i64 15, ptr [[A]])
// LLVM: ret void
  vst4q_lane_s8(a, b, 15);
}

// ALL-LABEL: @test_vst4q_lane_u16(
void test_vst4q_lane_u16(uint16_t  *a, uint16x8x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !cir.vector<8 x !u16i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v8i16.p0(<8 x i16> [[V0]], <8 x i16> [[V1]], <8 x i16> [[V2]], <8 x i16> [[V3]], i64 7, ptr [[A]])
// LLVM: ret void
  vst4q_lane_u16(a, b, 7);
}

// ALL-LABEL: @test_vst4q_lane_u32(
void test_vst4q_lane_u32(uint32_t  *a, uint32x4x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !cir.vector<4 x !u32i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v4i32.p0(<4 x i32> [[V0]], <4 x i32> [[V1]], <4 x i32> [[V2]], <4 x i32> [[V3]], i64 3, ptr [[A]])
// LLVM: ret void
  vst4q_lane_u32(a, b, 3);
}

// ALL-LABEL: @test_vst4q_lane_u64(
void test_vst4q_lane_u64(uint64_t  *a, uint64x2x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v2i64.p0(<2 x i64> [[V0]], <2 x i64> [[V1]], <2 x i64> [[V2]], <2 x i64> [[V3]], i64 1, ptr [[A]])
// LLVM: ret void
  vst4q_lane_u64(a, b, 1);
}

// ALL-LABEL: @test_vst4q_lane_u8(
void test_vst4q_lane_u8(uint8_t  *a, uint8x16x4_t b) {
// CIR: cir.call_llvm_intrinsic "aarch64.neon.st4lane" {{.*}} : (!cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !cir.vector<16 x !u8i>, !s64i, !cir.ptr<!void>) -> !void

// LLVM-SAME: ptr {{[a-z ]*}}[[A:%[a-z0-9_.]+]],
// LLVM-DAG: [[V0:%.*]] = extractvalue {{.*}} [[B:%[a-z0-9_.]+]],{{( 0,)?}} 0{{$}}
// LLVM-DAG: [[V1:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 1{{$}}
// LLVM-DAG: [[V2:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 2{{$}}
// LLVM-DAG: [[V3:%.*]] = extractvalue {{.*}} [[B]],{{( 0,)?}} 3{{$}}
// LLVM: call void @llvm.aarch64.neon.st4lane.v16i8.p0(<16 x i8> [[V0]], <16 x i8> [[V1]], <16 x i8> [[V2]], <16 x i8> [[V3]], i64 15, ptr [[A]])
// LLVM: ret void
  vst4q_lane_u8(a, b, 15);
}
