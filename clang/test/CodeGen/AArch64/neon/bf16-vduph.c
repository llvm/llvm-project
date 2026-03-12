// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

#include <arm_neon.h>

// LLVM-LABEL: @test_vduph_lane_bf16(
// CIR-LABEL: @test_vduph_lane_bf16(
  bfloat16_t test_vduph_lane_bf16(bfloat16x4_t v) {
    // CIR: cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !cir.bf16>
    // LLVM: %{{.*}} = extractelement <4 x bfloat> %{{.*}}, i{{32|64}} 1
    // LLVM: ret bfloat %{{.*}}
    return vduph_lane_bf16(v, 1);
  }
  
  // LLVM-LABEL: @test_vduph_laneq_bf16(
  // CIR-LABEL: @test_vduph_laneq_bf16(
  bfloat16_t test_vduph_laneq_bf16(bfloat16x8_t v) {
    // CIR: cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<8 x !cir.bf16>
    // LLVM: %{{.*}} = extractelement <8 x bfloat> %{{.*}}, i{{32|64}} 7
    // LLVM: ret bfloat %{{.*}}
    return vduph_laneq_bf16(v, 7);
  }
  