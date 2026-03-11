// REQUIRES: aarch64-registered-target || arm-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none           -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefixes=LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +bf16 -disable-O0-optnone -flax-vector-conversions=none -fclangir -emit-cir  -o - %s |                               FileCheck %s --check-prefixes=CIR %}

typedef __bf16 bfloat16_t;
typedef __attribute__((neon_vector_type(4))) bfloat16_t bfloat16x4_t;
typedef __attribute__((neon_vector_type(8))) bfloat16_t bfloat16x8_t;

// LLVM-LABEL: @test_vduph_lane_bf16(
// LLVM-SAME: <4 x bfloat> {{.*}} [[V:%.*]])
// LLVM:      [[VGET_LANE:%.*]] = extractelement <4 x bfloat> [[V]], i{{32|64}} 1
// LLVM:      ret bfloat [[VGET_LANE]]
// CIR-LABEL: @test_vduph_lane_bf16(
// CIR:       cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !cir.bf16>
bfloat16_t test_vduph_lane_bf16(bfloat16x4_t v) {
  return __builtin_bit_cast(bfloat16_t, __builtin_neon_vduph_lane_bf16(v, 1));
}

// LLVM-LABEL: @test_vduph_laneq_bf16(
// LLVM-SAME: <8 x bfloat> {{.*}} [[V:%.*]])
// LLVM:      [[VGETQ_LANE:%.*]] = extractelement <8 x bfloat> [[V]], i{{32|64}} 7
// LLVM:      ret bfloat [[VGETQ_LANE]]
// CIR-LABEL: @test_vduph_laneq_bf16(
// CIR:       cir.vec.extract %{{.*}}[%{{.*}} : !u64i] : !cir.vector<8 x !cir.bf16>
bfloat16_t test_vduph_laneq_bf16(bfloat16x8_t v) {
  return __builtin_bit_cast(bfloat16_t,
                            __builtin_neon_vduph_laneq_bf16(v, 7));
}
