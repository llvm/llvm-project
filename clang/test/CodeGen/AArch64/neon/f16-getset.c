// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=ALL,CIR %}

#include <arm_neon.h>

//===------------------------------------------------------===//
// 2.7.2.4 Set all lanes to the same value
//===------------------------------------------------------===//
// TODO: Add the remaining intrinsics from this group.

// ALL-LABEL: @test_vduph_lane_f16(
float16_t test_vduph_lane_f16(float16x4_t vec) {
// LLVM-SAME: <4 x half> {{.*}}[[VEC:%.*]])
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<4 x !cir.f16>
// LLVM: [[VGET_LANE:%.*]] = extractelement <4 x half> [[VEC]], i32 3
// LLVM-NEXT: ret half [[VGET_LANE]]
  return vduph_lane_f16(vec, 3);
}

// ALL-LABEL: @test_vduph_laneq_f16(
float16_t test_vduph_laneq_f16(float16x8_t vec) {
// LLVM-SAME: <8 x half> {{.*}}[[VEC:%.*]])
// CIR: cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<8 x !cir.f16>
// LLVM: [[VGETQ_LANE:%.*]] = extractelement <8 x half> [[VEC]], i32 7
// LLVM-NEXT: ret half [[VGETQ_LANE]]
  return vduph_laneq_f16(vec, 7);
}

