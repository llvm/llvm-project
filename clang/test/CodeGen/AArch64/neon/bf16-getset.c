// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon -target-feature +bf16           -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +bf16 -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +bf16 -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefixes=ALL,CIR %}

#include <arm_neon.h>

//=============================================================================
// NOTES
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#bitwise-equal-to-zero
//=============================================================================

//===------------------------------------------------------===//
// 2.4.1.2. Set all lanes to the same value
//===------------------------------------------------------===//

// ALL-LABEL: @test_vduph_lane_bf16(
// LLVM-SAME: <4 x bfloat> {{.*}}[[V:%.*]]) #[[ATTR0:[0-9]+]] {
bfloat16_t test_vduph_lane_bf16(bfloat16x4_t v) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<4 x !cir.bf16>

// LLVM: [[VGET_LANE:%.*]] = extractelement <4 x bfloat> %{{.*}}, i32 1
// LLVM: ret bfloat [[VGET_LANE]]
return vduph_lane_bf16(v, 1);
}

// ALL-LABEL: @test_vduph_laneq_bf16(
// LLVM-SAME: <8 x bfloat> {{.*}}[[V:%.*]]) #[[ATTR0:[0-9]+]] {
bfloat16_t test_vduph_laneq_bf16(bfloat16x8_t v) {
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<8 x !cir.bf16>

// LLVM: [[VGETQ_LANE:%.*]] = extractelement <8 x bfloat> %{{.*}}, i32 7
// LLVM: ret bfloat [[VGETQ_LANE]]
return vduph_laneq_bf16(v, 7);
}

// ALL-LABEL: @test_vdup_lane_bf16(
// LLVM-SAME: <4 x bfloat> {{.*}}[[V:%.*]]) #[[ATTR0:[0-9]+]] {
bfloat16x4_t test_vdup_lane_bf16(bfloat16x4_t v) {
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.bf16>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<4 x !cir.bf16>

  // LLVM: [[TMP0:%.*]] = bitcast <4 x bfloat> [[V]] to <4 x i16>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[TMP0]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x bfloat>
  // LLVM: [[SHUF:%.*]] = shufflevector <4 x bfloat> [[TMP2]], {{.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  // LLVM: ret <4 x bfloat> {{%.*}}
  return vdup_lane_bf16(v, 1);
}

// ALL-LABEL: @test_vdupq_lane_bf16(
// LLVM-SAME: <4 x bfloat> {{.*}}[[V:%.*]]) #[[ATTR0:[0-9]+]] {
bfloat16x8_t test_vdupq_lane_bf16(bfloat16x4_t v) {
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !cir.bf16>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<8 x !cir.bf16>

  // LLVM: [[TMP0:%.*]] = bitcast <4 x bfloat> [[V]] to <4 x i16>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[TMP0]] to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x bfloat>
  // LLVM: [[SHUF:%.*]] = shufflevector <4 x bfloat> [[TMP2]], {{.*}}, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // LLVM: ret <8 x bfloat> {{%.*}}
  return vdupq_lane_bf16(v, 1);
}

// ALL-LABEL: @test_vdup_laneq_bf16(
// LLVM-SAME: <8 x bfloat> {{.*}}[[V:%.*]]) #[[ATTR0:[0-9]+]] {
bfloat16x4_t test_vdup_laneq_bf16(bfloat16x8_t v) {
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.bf16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.bf16>

  // LLVM: [[TMP0:%.*]] = bitcast <8 x bfloat> [[V]] to <8 x i16>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[TMP0]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x bfloat>
  // LLVM: [[SHUF:%.*]] = shufflevector <8 x bfloat> [[TMP2]], {{.*}}, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  // LLVM: ret <4 x bfloat> {{%.*}}
  return vdup_laneq_bf16(v, 7);
}

// ALL-LABEL: @test_vdupq_laneq_bf16(
// LLVM-SAME: <8 x bfloat> {{.*}}[[V:%.*]]) #[[ATTR0:[0-9]+]] {
bfloat16x8_t test_vdupq_laneq_bf16(bfloat16x8_t v) {
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !cir.bf16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.bf16>

  // LLVM: [[TMP0:%.*]] = bitcast <8 x bfloat> [[V]] to <8 x i16>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[TMP0]] to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x bfloat>
  // LLVM: [[SHUF:%.*]] = shufflevector <8 x bfloat> [[TMP2]], {{.*}}, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  // LLVM: ret <8 x bfloat> {{%.*}}
  return vdupq_laneq_bf16(v, 7);
}

// LLVM-LABEL: @test_vdup_n_bf16(
// LLVM-SAME: bfloat {{.*}}[[V:%.*]]) #[[ATTR0:[0-9]+]] {
// CIR-LABEL: @vdup_n_bf16(
  bfloat16x4_t test_vdup_n_bf16(bfloat16_t v) {
    // CIR: cir.vec.create(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !cir.bf16, !cir.bf16, !cir.bf16, !cir.bf16) : !cir.vector<4 x !cir.bf16>
  
    // LLVM:      [[VECINIT_I:%.*]] = insertelement <4 x bfloat> poison, bfloat [[V]], i{{32|64}} 0
    // LLVM-NEXT: [[VECINIT1_I:%.*]] = insertelement <4 x bfloat> [[VECINIT_I]], bfloat [[V]], i{{32|64}} 1
    // LLVM-NEXT: [[VECINIT2_I:%.*]] = insertelement <4 x bfloat> [[VECINIT1_I]], bfloat [[V]], i{{32|64}} 2
    // LLVM-NEXT: [[VECINIT3_I:%.*]] = insertelement <4 x bfloat> [[VECINIT2_I]], bfloat [[V]], i{{32|64}} 3
    // LLVM-NEXT: ret <4 x bfloat> [[VECINIT3_I]]
    return vdup_n_bf16(v);
  }
  
  // LLVM-LABEL: @test_vdupq_n_bf16(
  // LLVM-SAME: bfloat {{.*}}[[V:%.*]]) #[[ATTR0:[0-9]+]] {
  // CIR-LABEL: @vdupq_n_bf16(
  bfloat16x8_t test_vdupq_n_bf16(bfloat16_t v) {
    // CIR: cir.vec.create(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !cir.bf16, !cir.bf16, !cir.bf16, !cir.bf16, !cir.bf16, !cir.bf16, !cir.bf16, !cir.bf16) : !cir.vector<8 x !cir.bf16>
  
    // LLVM:      [[VECINIT_I:%.*]] = insertelement <8 x bfloat> poison, bfloat [[V]], i{{32|64}} 0
    // LLVM-NEXT: [[VECINIT1_I:%.*]] = insertelement <8 x bfloat> [[VECINIT_I]], bfloat [[V]], i{{32|64}} 1
    // LLVM-NEXT: [[VECINIT2_I:%.*]] = insertelement <8 x bfloat> [[VECINIT1_I]], bfloat [[V]], i{{32|64}} 2
    // LLVM-NEXT: [[VECINIT3_I:%.*]] = insertelement <8 x bfloat> [[VECINIT2_I]], bfloat [[V]], i{{32|64}} 3
    // LLVM-NEXT: [[VECINIT4_I:%.*]] = insertelement <8 x bfloat> [[VECINIT3_I]], bfloat [[V]], i{{32|64}} 4
    // LLVM-NEXT: [[VECINIT5_I:%.*]] = insertelement <8 x bfloat> [[VECINIT4_I]], bfloat [[V]], i{{32|64}} 5
    // LLVM-NEXT: [[VECINIT6_I:%.*]] = insertelement <8 x bfloat> [[VECINIT5_I]], bfloat [[V]], i{{32|64}} 6
    // LLVM-NEXT: [[VECINIT7_I:%.*]] = insertelement <8 x bfloat> [[VECINIT6_I]], bfloat [[V]], i{{32|64}} 7
    // LLVM-NEXT: ret <8 x bfloat> [[VECINIT7_I]]
    return vdupq_n_bf16(v);
  }

//===------------------------------------------------------===//
// 2.14.1.4 Split vectors
//===------------------------------------------------------===//

// ALL-LABEL: @test_vget_high_bf16(
bfloat16x4_t test_vget_high_bf16(bfloat16x8_t a) {
  // CIR: cir.call @vget_high_bf16({{%.*}}) : (!cir.vector<8 x !cir.bf16>

  // LLVM-SAME: <8 x bfloat> {{.*}} [[A:%.*]])
  // LLVM: [[SHUFFLE_I:%.*]] = shufflevector <8 x bfloat> [[A:%.*]], <8 x bfloat> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  // LLVM: ret <4 x bfloat> [[SHUFFLE_I]]
  return vget_high_bf16(a);
}

// ALL-LABEL: @test_vget_low_bf16(
bfloat16x4_t test_vget_low_bf16(bfloat16x8_t a) {
  // CIR: cir.call @vget_low_bf16({{%.*}}) : (!cir.vector<8 x !cir.bf16>

  // LLVM-SAME: <8 x bfloat> {{.*}} [[A:%.*]])
  // LLVM: [[SHUFFLE_I:%.*]] = shufflevector <8 x bfloat> [[A:%.*]], <8 x bfloat> [[A]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: ret <4 x bfloat> [[SHUFFLE_I]]
  return vget_low_bf16(a);
}

// ALL-LABEL: @test_vget_lane_bf16(
bfloat16_t test_vget_lane_bf16(bfloat16x4_t v) {
  // CIR: cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<4 x !cir.bf16>

  // LLVM-SAME: <4 x bfloat> {{.*}} [[V:%.*]])
  // LLVM: [[VGET_LANE:%.*]] = extractelement <4 x bfloat> [[V]], i32 1
  // LLVM: ret bfloat [[VGET_LANE]]
  return vget_lane_bf16(v, 1);
}

// ALL-LABEL: @test_vgetq_lane_bf16(
bfloat16_t test_vgetq_lane_bf16(bfloat16x8_t v) {
  // CIR: cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<8 x !cir.bf16>

  // LLVM-SAME: <8 x bfloat> {{.*}} [[V:%.*]])
  // LLVM: [[VGETQ_LANE:%.*]] = extractelement <8 x bfloat> [[V]], i32 7
  // LLVM: ret bfloat [[VGETQ_LANE]]
  return vgetq_lane_bf16(v, 7);
}
