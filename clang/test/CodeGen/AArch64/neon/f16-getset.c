// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=ALL,CIR %}

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.7.2.4 Set all lanes to the same value
//===------------------------------------------------------===//

// ALL-LABEL: @test_vdup_n_f16(
float16x4_t test_vdup_n_f16(float16_t a) {
  // CIR: cir.vec.create(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !cir.f16, !cir.f16, !cir.f16, !cir.f16) : !cir.vector<4 x !cir.f16>

  // LLVM-SAME: half noundef [[A:%.*]])
  // LLVM:      [[VECINIT:%.*]] = insertelement <4 x half> poison, half [[A]], i{{32|64}} 0
  // LLVM-NEXT: [[VECINIT1:%.*]] = insertelement <4 x half> [[VECINIT]], half [[A]], i{{32|64}} 1
  // LLVM-NEXT: [[VECINIT2:%.*]] = insertelement <4 x half> [[VECINIT1]], half [[A]], i{{32|64}} 2
  // LLVM-NEXT: [[VECINIT3:%.*]] = insertelement <4 x half> [[VECINIT2]], half [[A]], i{{32|64}} 3
  // LLVM:      ret <4 x half> [[VECINIT3]]
  return vdup_n_f16(a);
}

// ALL-LABEL: @test_vdupq_n_f16(
float16x8_t test_vdupq_n_f16(float16_t a) {
  // CIR: cir.vec.create(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !cir.f16, !cir.f16, !cir.f16, !cir.f16, !cir.f16, !cir.f16, !cir.f16, !cir.f16) : !cir.vector<8 x !cir.f16>

  // LLVM-SAME: half noundef [[A:%.*]])
  // LLVM:      [[VECINIT:%.*]] = insertelement <8 x half> poison, half [[A]], i{{32|64}} 0
  // LLVM-NEXT: [[VECINIT1:%.*]] = insertelement <8 x half> [[VECINIT]], half [[A]], i{{32|64}} 1
  // LLVM-NEXT: [[VECINIT2:%.*]] = insertelement <8 x half> [[VECINIT1]], half [[A]], i{{32|64}} 2
  // LLVM-NEXT: [[VECINIT3:%.*]] = insertelement <8 x half> [[VECINIT2]], half [[A]], i{{32|64}} 3
  // LLVM-NEXT: [[VECINIT4:%.*]] = insertelement <8 x half> [[VECINIT3]], half [[A]], i{{32|64}} 4
  // LLVM-NEXT: [[VECINIT5:%.*]] = insertelement <8 x half> [[VECINIT4]], half [[A]], i{{32|64}} 5
  // LLVM-NEXT: [[VECINIT6:%.*]] = insertelement <8 x half> [[VECINIT5]], half [[A]], i{{32|64}} 6
  // LLVM-NEXT: [[VECINIT7:%.*]] = insertelement <8 x half> [[VECINIT6]], half [[A]], i{{32|64}} 7
  // LLVM:      ret <8 x half> [[VECINIT7]]
  return vdupq_n_f16(a);
}

// ALL-LABEL: @test_vdup_lane_f16(
float16x4_t test_vdup_lane_f16(float16x4_t a) {
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.f16>

  // LLVM-SAME: <4 x half> noundef [[A:%.*]])
  // LLVM:      [[LANE:%.*]] = shufflevector <4 x half> {{.*}}, <4 x half> {{.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  // LLVM:      ret <4 x half> [[LANE]]
  return vdup_lane_f16(a, 3);
}

// ALL-LABEL: @test_vdupq_lane_f16(
float16x8_t test_vdupq_lane_f16(float16x4_t a) {
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<8 x !cir.f16>

  // LLVM-SAME: <4 x half> noundef [[A:%.*]])
  // LLVM:      [[LANE:%.*]] = shufflevector <4 x half> {{.*}}, <4 x half> {{.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  // LLVM:      ret <8 x half> [[LANE]]
  return vdupq_lane_f16(a, 3);
}

// ALL-LABEL: @test_vdup_laneq_f16(
float16x4_t test_vdup_laneq_f16(float16x8_t a) {
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<4 x !cir.f16>

  // LLVM-SAME: <8 x half> noundef [[A:%.*]])
  // LLVM:      [[LANE:%.*]] = shufflevector <8 x half> {{.*}}, <8 x half> {{.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  // LLVM:      ret <4 x half> [[LANE]]
  return vdup_laneq_f16(a, 1);
}

// ALL-LABEL: @test_vdupq_laneq_f16(
float16x8_t test_vdupq_laneq_f16(float16x8_t a) {
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.f16>

  // LLVM-SAME: <8 x half> noundef [[A:%.*]])
  // LLVM:      [[LANE:%.*]] = shufflevector <8 x half> {{.*}}, <8 x half> {{.*}}, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  // LLVM:      ret <8 x half> [[LANE]]
  return vdupq_laneq_f16(a, 7);
}

// ALL-LABEL: @test_vduph_lane_f16(
float16_t test_vduph_lane_f16(float16x4_t vec) {
  // CIR: cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<4 x !cir.f16>

  // LLVM-SAME: <4 x half> {{.*}}[[VEC:%.*]])
  // LLVM:      [[VGET_LANE:%.*]] = extractelement <4 x half> [[VEC]], i32 3
  // LLVM:      ret half [[VGET_LANE]]
  return vduph_lane_f16(vec, 3);
}

// ALL-LABEL: @test_vduph_laneq_f16(
float16_t test_vduph_laneq_f16(float16x8_t vec) {
  // CIR: cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<8 x !cir.f16>

  // LLVM-SAME: <8 x half> {{.*}}[[VEC:%.*]])
  // LLVM:      [[VGETQ_LANE:%.*]] = extractelement <8 x half> [[VEC]], i32 7
  // LLVM:      ret half [[VGETQ_LANE]]
  return vduph_laneq_f16(vec, 7);
}
