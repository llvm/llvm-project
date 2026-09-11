// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1_cg_arm64_neon -ffp-exception-behavior=strict -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM --implicit-check-not=' @llvm.fma.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefix=CIR --implicit-check-not='cir.call_llvm_intrinsic "fma"' %}

#include <arm_neon.h>

// LLVM-LABEL: @test_vfma_f64(
// CIR-LABEL: @vfma_f64(
float64x1_t test_vfma_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
// CIR: cir.fma %{{.*}}, %{{.*}}, %{{.*}} : !cir.vector<1 x !cir.double> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <1 x double> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[A_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[A_I]], i64 0
// LLVM: [[B_I:%.*]] = bitcast <1 x double> [[B]] to i64
// LLVM: [[B_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[B_I]], i64 0
// LLVM: [[C_I:%.*]] = bitcast <1 x double> [[C]] to i64
// LLVM: [[C_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[C_I]], i64 0
// LLVM: [[A_BYTES:%.*]] = bitcast <1 x i64> [[A_INSERT]] to <8 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <1 x i64> [[B_INSERT]] to <8 x i8>
// LLVM: [[C_BYTES:%.*]] = bitcast <1 x i64> [[C_INSERT]] to <8 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <1 x double>
// LLVM: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to <1 x double>
// LLVM: [[C_CAST:%.*]] = bitcast <8 x i8> [[C_BYTES]] to <1 x double>
// LLVM: [[FMA:%.*]] = call <1 x double> @llvm.experimental.constrained.fma.v1f64(<1 x double> [[B_CAST]], <1 x double> [[C_CAST]], <1 x double> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <1 x double> [[FMA]]
  return vfma_f64(a, b, c);
}

// LLVM-LABEL: @test_vfma_laneq_f64(
// CIR-LABEL: @test_vfma_laneq_f64(
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b,
                                 float64x2_t v) {
// CIR: [[INDEX:%.*]] = cir.const #cir.int<0> : !u64i
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}{{\[}}[[INDEX]] : !u64i] : !cir.vector<2 x !cir.double>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <1 x double> {{.*}} [[A:%.*]], <1 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <1 x double> [[A]] to i64
// LLVM: [[A_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[A_I]], i64 0
// LLVM: [[B_I:%.*]] = bitcast <1 x double> [[B]] to i64
// LLVM: [[B_INSERT:%.*]] = insertelement <1 x i64> undef, i64 [[B_I]], i64 0
// LLVM: [[V_I:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM: [[A_BYTES:%.*]] = bitcast <1 x i64> [[A_INSERT]] to <8 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <1 x i64> [[B_INSERT]] to <8 x i8>
// LLVM: [[V_BYTES:%.*]] = bitcast <2 x i64> [[V_I]] to <16 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to double
// LLVM: [[B_CAST:%.*]] = bitcast <8 x i8> [[B_BYTES]] to double
// LLVM: [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <2 x double>
// LLVM: [[LANE:%.*]] = extractelement <2 x double> [[V_CAST]], i{{32|64}} 0
// LLVM: [[FMA:%.*]] = call double @llvm.experimental.constrained.fma.f64(double [[B_CAST]], double [[LANE]], double [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: [[RESULT:%.*]] = bitcast double [[FMA]] to <1 x double>
// LLVM: ret <1 x double> [[RESULT]]
  return vfma_laneq_f64(a, b, v, 0);
}

// LLVM-LABEL: @test_vfmaq_laneq_f64(
// CIR-LABEL: @test_vfmaq_laneq_f64(
float64x2_t test_vfmaq_laneq_f64(float64x2_t a, float64x2_t b,
                                  float64x2_t v) {
// CIR: [[LANE:%.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !cir.double>
// CIR: cir.fma [[LANE]], %{{.*}}, %{{.*}} : !cir.vector<2 x !cir.double> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]], <2 x double> {{.*}} [[B:%.*]], <2 x double> {{.*}} [[V:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[B_I:%.*]] = bitcast <2 x double> [[B]] to <2 x i64>
// LLVM: [[V_I:%.*]] = bitcast <2 x double> [[V]] to <2 x i64>
// LLVM: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM: [[B_BYTES:%.*]] = bitcast <2 x i64> [[B_I]] to <16 x i8>
// LLVM: [[V_BYTES:%.*]] = bitcast <2 x i64> [[V_I]] to <16 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM: [[B_CAST:%.*]] = bitcast <16 x i8> [[B_BYTES]] to <2 x double>
// LLVM: [[V_CAST:%.*]] = bitcast <16 x i8> [[V_BYTES]] to <2 x double>
// LLVM: [[LANE:%.*]] = shufflevector <2 x double> [[V_CAST]], <2 x double> {{.*}}, <2 x i32> <i32 1, i32 1>
// LLVM: [[FMA:%.*]] = call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> [[LANE]], <2 x double> [[B_CAST]], <2 x double> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <2 x double> [[FMA]]
  return vfmaq_laneq_f64(a, b, v, 1);
}

// LLVM-LABEL: @test_vfmas_lane_f32(
// CIR-LABEL: @test_vfmas_lane_f32(
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
// CIR: [[INDEX:%.*]] = cir.const #cir.int<1> : !u64i
// CIR: [[LANE:%.*]] = cir.vec.extract %{{.*}}{{\[}}[[INDEX]] : !u64i] : !cir.vector<2 x !cir.float>
// CIR: cir.fma %{{.*}}, [[LANE]], %{{.*}} : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: float {{.*}} [[A:%.*]], float {{.*}} [[B:%.*]], <2 x float> {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[LANE:%.*]] = extractelement <2 x float> [[C]], i{{32|64}} 1
// LLVM: [[FMA:%.*]] = call float @llvm.experimental.constrained.fma.f32(float [[B]], float [[LANE]], float [[A]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret float [[FMA]]
  return vfmas_lane_f32(a, b, c, 1);
}
