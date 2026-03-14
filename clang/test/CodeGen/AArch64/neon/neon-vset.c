// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=ALL,CIR %}

//=============================================================================
// NOTES
//
// This file contains tests originally located in:
//  * clang/test/CodeGen/AArch64/neon-vget.c  (vset_lane_* / vsetq_lane_* section)
//  * clang/test/CodeGen/AArch64/bf16-getset-intrinsics.c  (bf16 vset variants)
//
// The main difference is the use of RUN lines that enable ClangIR lowering;
// therefore only builtins currently supported by ClangIR are tested here.
//
// The f16 variants (vset_lane_f16 / vsetq_lane_f16) are intentionally omitted:
// they are implemented in arm_neon.h via pointer-based bit_cast rather than
// a direct builtin call, producing complex alloca/store/load sequences that
// differ between the two code paths.
//
// The p8/p16 poly variants (vset_lane_p8, vset_lane_p16, vsetq_lane_p8,
// vsetq_lane_p16) are intentionally omitted: the arm_neon.h macros for these
// perform an implicit poly-vector → int-vector conversion which is rejected
// when -flax-vector-conversions=none is in effect.
//
// ACLE reference:
//  https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#set-vector-lane
//=============================================================================

#include <arm_neon.h>

//===------------------------------------------------------===//
// 64-bit vector (vset_lane_*)
//===------------------------------------------------------===//

// ALL-LABEL: @test_vset_lane_u8(
uint8x8_t test_vset_lane_u8(uint8_t a, uint8x8_t b) {
// CIR: cir.vec.insert

// LLVM: [[VSET_LANE:%.*]] = insertelement <8 x i8> %{{.*}}, i8 %{{.*}}, i32 7
// LLVM: ret <8 x i8> [[VSET_LANE]]
  return vset_lane_u8(a, b, 7);
}

// ALL-LABEL: @test_vset_lane_u16(
uint16x4_t test_vset_lane_u16(uint16_t a, uint16x4_t b) {
// CIR: cir.vec.insert

// LLVM: [[VSET_LANE:%.*]] = insertelement <4 x i16> %{{.*}}, i16 %{{.*}}, i32 3
// LLVM: ret <4 x i16> [[VSET_LANE]]
  return vset_lane_u16(a, b, 3);
}

// ALL-LABEL: @test_vset_lane_u32(
uint32x2_t test_vset_lane_u32(uint32_t a, uint32x2_t b) {
// CIR: cir.vec.insert

// LLVM: [[VSET_LANE:%.*]] = insertelement <2 x i32> %{{.*}}, i32 %{{.*}}, i32 1
// LLVM: ret <2 x i32> [[VSET_LANE]]
  return vset_lane_u32(a, b, 1);
}

// ALL-LABEL: @test_vset_lane_s8(
int8x8_t test_vset_lane_s8(int8_t a, int8x8_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<8 x !s8i>

// LLVM: [[VSET_LANE:%.*]] = insertelement <8 x i8> %{{.*}}, i8 %{{.*}}, i32 7
// LLVM: ret <8 x i8> [[VSET_LANE]]
  return vset_lane_s8(a, b, 7);
}

// ALL-LABEL: @test_vset_lane_s16(
int16x4_t test_vset_lane_s16(int16_t a, int16x4_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<4 x !s16i>

// LLVM: [[VSET_LANE:%.*]] = insertelement <4 x i16> %{{.*}}, i16 %{{.*}}, i32 3
// LLVM: ret <4 x i16> [[VSET_LANE]]
  return vset_lane_s16(a, b, 3);
}

// ALL-LABEL: @test_vset_lane_s32(
int32x2_t test_vset_lane_s32(int32_t a, int32x2_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<2 x !s32i>

// LLVM: [[VSET_LANE:%.*]] = insertelement <2 x i32> %{{.*}}, i32 %{{.*}}, i32 1
// LLVM: ret <2 x i32> [[VSET_LANE]]
  return vset_lane_s32(a, b, 1);
}

// ALL-LABEL: @test_vset_lane_f32(
float32x2_t test_vset_lane_f32(float32_t a, float32x2_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<2 x !cir.float>

// LLVM: [[VSET_LANE:%.*]] = insertelement <2 x float> %{{.*}}, float %{{.*}}, i32 1
// LLVM: ret <2 x float> [[VSET_LANE]]
  return vset_lane_f32(a, b, 1);
}

// ALL-LABEL: @test_vset_lane_s64(
int64x1_t test_vset_lane_s64(int64_t a, int64x1_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<1 x !s64i>

// LLVM: [[VSET_LANE:%.*]] = insertelement <1 x i64> %{{.*}}, i64 %{{.*}}, i32 0
// LLVM: ret <1 x i64> [[VSET_LANE]]
  return vset_lane_s64(a, b, 0);
}

// ALL-LABEL: @test_vset_lane_u64(
uint64x1_t test_vset_lane_u64(uint64_t a, uint64x1_t b) {
// CIR: cir.vec.insert

// LLVM: [[VSET_LANE:%.*]] = insertelement <1 x i64> %{{.*}}, i64 %{{.*}}, i32 0
// LLVM: ret <1 x i64> [[VSET_LANE]]
  return vset_lane_u64(a, b, 0);
}

// ALL-LABEL: @test_vset_lane_f64(
float64x1_t test_vset_lane_f64(float64_t a, float64x1_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<1 x !cir.double>

// LLVM: [[VSET_LANE:%.*]] = insertelement <1 x double> %{{.*}}, double %{{.*}}, i32 0
// LLVM: ret <1 x double> [[VSET_LANE]]
  return vset_lane_f64(a, b, 0);
}

// ALL-LABEL: @test_vset_lane_bf16(
bfloat16x4_t test_vset_lane_bf16(bfloat16_t a, bfloat16x4_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<4 x !cir.bf16>

// LLVM: [[VSET_LANE:%.*]] = insertelement <4 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 1
// LLVM: ret <4 x bfloat> [[VSET_LANE]]
  return vset_lane_bf16(a, b, 1);
}

//===------------------------------------------------------===//
// 128-bit vector (vsetq_lane_*)
//===------------------------------------------------------===//

// ALL-LABEL: @test_vsetq_lane_u8(
uint8x16_t test_vsetq_lane_u8(uint8_t a, uint8x16_t b) {
// CIR: cir.vec.insert

// LLVM: [[VSET_LANE:%.*]] = insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 15
// LLVM: ret <16 x i8> [[VSET_LANE]]
  return vsetq_lane_u8(a, b, 15);
}

// ALL-LABEL: @test_vsetq_lane_u16(
uint16x8_t test_vsetq_lane_u16(uint16_t a, uint16x8_t b) {
// CIR: cir.vec.insert

// LLVM: [[VSET_LANE:%.*]] = insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 7
// LLVM: ret <8 x i16> [[VSET_LANE]]
  return vsetq_lane_u16(a, b, 7);
}

// ALL-LABEL: @test_vsetq_lane_u32(
uint32x4_t test_vsetq_lane_u32(uint32_t a, uint32x4_t b) {
// CIR: cir.vec.insert

// LLVM: [[VSET_LANE:%.*]] = insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 3
// LLVM: ret <4 x i32> [[VSET_LANE]]
  return vsetq_lane_u32(a, b, 3);
}

// ALL-LABEL: @test_vsetq_lane_s8(
int8x16_t test_vsetq_lane_s8(int8_t a, int8x16_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<16 x !s8i>

// LLVM: [[VSET_LANE:%.*]] = insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 15
// LLVM: ret <16 x i8> [[VSET_LANE]]
  return vsetq_lane_s8(a, b, 15);
}

// ALL-LABEL: @test_vsetq_lane_s16(
int16x8_t test_vsetq_lane_s16(int16_t a, int16x8_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<8 x !s16i>

// LLVM: [[VSET_LANE:%.*]] = insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 7
// LLVM: ret <8 x i16> [[VSET_LANE]]
  return vsetq_lane_s16(a, b, 7);
}

// ALL-LABEL: @test_vsetq_lane_s32(
int32x4_t test_vsetq_lane_s32(int32_t a, int32x4_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<4 x !s32i>

// LLVM: [[VSET_LANE:%.*]] = insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 3
// LLVM: ret <4 x i32> [[VSET_LANE]]
  return vsetq_lane_s32(a, b, 3);
}

// ALL-LABEL: @test_vsetq_lane_f32(
float32x4_t test_vsetq_lane_f32(float32_t a, float32x4_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<4 x !cir.float>

// LLVM: [[VSET_LANE:%.*]] = insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 3
// LLVM: ret <4 x float> [[VSET_LANE]]
  return vsetq_lane_f32(a, b, 3);
}

// ALL-LABEL: @test_vsetq_lane_s64(
int64x2_t test_vsetq_lane_s64(int64_t a, int64x2_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<2 x !s64i>

// LLVM: [[VSET_LANE:%.*]] = insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
// LLVM: ret <2 x i64> [[VSET_LANE]]
  return vsetq_lane_s64(a, b, 1);
}

// ALL-LABEL: @test_vsetq_lane_u64(
uint64x2_t test_vsetq_lane_u64(uint64_t a, uint64x2_t b) {
// CIR: cir.vec.insert

// LLVM: [[VSET_LANE:%.*]] = insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
// LLVM: ret <2 x i64> [[VSET_LANE]]
  return vsetq_lane_u64(a, b, 1);
}

// ALL-LABEL: @test_vsetq_lane_f64(
float64x2_t test_vsetq_lane_f64(float64_t a, float64x2_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<2 x !cir.double>

// LLVM: [[VSET_LANE:%.*]] = insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
// LLVM: ret <2 x double> [[VSET_LANE]]
  return vsetq_lane_f64(a, b, 1);
}

// ALL-LABEL: @test_vsetq_lane_bf16(
bfloat16x8_t test_vsetq_lane_bf16(bfloat16_t a, bfloat16x8_t b) {
// CIR: cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : {{.*}}] : !cir.vector<8 x !cir.bf16>

// LLVM: [[VSET_LANE:%.*]] = insertelement <8 x bfloat> %{{.*}}, bfloat %{{.*}}, i32 7
// LLVM: ret <8 x bfloat> [[VSET_LANE]]
  return vsetq_lane_bf16(a, b, 7);
}
