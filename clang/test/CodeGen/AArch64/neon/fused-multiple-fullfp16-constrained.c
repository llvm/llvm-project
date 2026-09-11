// REQUIRES: aarch64-registered-target

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=maytrap -DEXCEPT=1 -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM --implicit-check-not=fpexcept.maytrap --implicit-check-not=' @llvm.fma.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=maytrap -DEXCEPT=1 -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefix=CIR --implicit-check-not='except_mode = maytrap' --implicit-check-not='cir.call_llvm_intrinsic "fma"' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=strict             -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM --implicit-check-not=' @llvm.fma.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=strict             -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefix=CIR --implicit-check-not='cir.call_llvm_intrinsic "fma"' %}

#if EXCEPT
#pragma float_control(except, on)
#endif

#include <arm_neon.h>

// LLVM-LABEL: @test_vfma_f16(
// CIR-LABEL: @vfma_f16(
float16x4_t test_vfma_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
// CIR: cir.fma %{{.*}}, %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM: [[B_I:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM: [[FMA:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[B_CAST]], <4 x half> [[C_CAST]], <4 x half> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <4 x half> [[FMA]]
  return vfma_f16(a, b, c);
}

// LLVM-LABEL: @test_vfmaq_f16(
// CIR-LABEL: @vfmaq_f16(
float16x8_t test_vfmaq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
// CIR: cir.fma %{{.*}}, %{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM: [[B_I:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM: [[FMA:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[B_CAST]], <8 x half> [[C_CAST]], <8 x half> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <8 x half> [[FMA]]
  return vfmaq_f16(a, b, c);
}

// LLVM-LABEL: @test_vfma_lane_f16(
// CIR-LABEL: @test_vfma_lane_f16(
float16x4_t test_vfma_lane_f16(float16x4_t a, float16x4_t b,
                                float16x4_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.vector<4 x !cir.f16> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM: [[B_I:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM: [[LANE:%.*]] = shufflevector <4 x half> [[C_CAST]], <4 x half> {{.*}}, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// LLVM: [[FMA:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[B_CAST]], <4 x half> [[LANE]], <4 x half> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <4 x half> [[FMA]]
  return vfma_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: @test_vfmaq_lane_f16(
// CIR-LABEL: @test_vfmaq_lane_f16(
float16x8_t test_vfmaq_lane_f16(float16x8_t a, float16x8_t b,
                                 float16x4_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16>) [#cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i] : !cir.vector<8 x !cir.f16>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.vector<8 x !cir.f16> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM: [[B_I:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM: [[C_I:%.*]] = bitcast <4 x half> [[C]] to <4 x i16>
// LLVM: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM: [[C_BYTES:%.*]] = bitcast <4 x i16> [[C_I]] to <8 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <4 x half>
// LLVM: [[LANE:%.*]] = shufflevector <4 x half> [[C_CAST]], <4 x half> {{.*}}, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// LLVM: [[FMA:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[B_CAST]], <8 x half> [[LANE]], <8 x half> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <8 x half> [[FMA]]
  return vfmaq_lane_f16(a, b, c, 3);
}

// LLVM-LABEL: @test_vfma_laneq_f16(
// CIR-LABEL: @test_vfma_laneq_f16(
float16x4_t test_vfma_laneq_f16(float16x4_t a, float16x4_t b,
                                 float16x8_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma [[LANE]], %{{.*}}, %{{.*}} : !cir.vector<4 x !cir.f16> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]], <4 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM: [[B_I:%.*]] = bitcast <4 x half> [[B]] to <4 x i16>
// LLVM: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <4 x i16> [[B_I]] to <8 x i8>
// LLVM: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <4 x half>
// LLVM: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM: [[LANE:%.*]] = shufflevector <8 x half> [[C_CAST]], <8 x half> {{.*}}, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// LLVM: [[FMA:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[B_CAST]], <4 x half> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <4 x half> [[FMA]]
  return vfma_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: @test_vfmaq_laneq_f16(
// CIR-LABEL: @test_vfmaq_laneq_f16(
float16x8_t test_vfmaq_laneq_f16(float16x8_t a, float16x8_t b,
                                  float16x8_t c) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16>) [#cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.f16>
// CIR: cir.fma [[LANE]], %{{.*}}, %{{.*}} : !cir.vector<8 x !cir.f16> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]], <8 x half> {{.*}} [[B:%.*]], <8 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM: [[B_I:%.*]] = bitcast <8 x half> [[B]] to <8 x i16>
// LLVM: [[C_I:%.*]] = bitcast <8 x half> [[C]] to <8 x i16>
// LLVM: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <8 x i16> [[B_I]] to <16 x i8>
// LLVM: [[C_BYTES:%.*]] = bitcast <8 x i16> [[C_I]] to <16 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <8 x half>
// LLVM: [[C_CAST:%.*]] = bitcast <16 x i8> [[C_BYTES]] to <8 x half>
// LLVM: [[LANE:%.*]] = shufflevector <8 x half> [[C_CAST]], <8 x half> {{.*}}, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// LLVM: [[FMA:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[B_CAST]], <8 x half> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <8 x half> [[FMA]]
  return vfmaq_laneq_f16(a, b, c, 7);
}

// LLVM-LABEL: @test_vfmah_lane_f16(
// CIR-LABEL: @test_vfmah_lane_f16(
float16_t test_vfmah_lane_f16(float16_t a, float16_t b, float16x4_t c) {
// CIR: [[INDEX:%.*]] = cir.const #cir.int<3> : !u64i
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}{{\[}}[[INDEX]] : !u64i] : !cir.vector<4 x !cir.f16>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.f16 {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], <4 x half> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[LANE:%.*]] = extractelement <4 x half> [[C]], i{{32|64}} 3
// LLVM: [[FMA:%.*]] = call half @llvm.experimental.constrained.fma.f16(half [[B]], half [[LANE]], half [[A]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret half [[FMA]]
  return vfmah_lane_f16(a, b, c, 3);
}
