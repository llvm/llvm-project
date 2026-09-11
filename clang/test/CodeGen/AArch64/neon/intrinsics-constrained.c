// REQUIRES: aarch64-registered-target

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=maytrap -DEXCEPT=1 -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM --implicit-check-not=fpexcept.maytrap --implicit-check-not=' @llvm.sqrt.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=maytrap -DEXCEPT=1 -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefix=CIR --implicit-check-not='except_mode = maytrap' --implicit-check-not='cir.call_llvm_intrinsic "sqrt"' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=strict             -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM --implicit-check-not=' @llvm.sqrt.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=strict             -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefix=CIR --implicit-check-not='cir.call_llvm_intrinsic "sqrt"' %}

#if EXCEPT
#pragma float_control(except, on)
#endif

#include <arm_neon.h>

// LLVM-LABEL: @test_vsqrt_f16(
// CIR-LABEL: @vsqrt_f16(
float16x4_t test_vsqrt_f16(float16x4_t a) {
// CIR: cir.sqrt %{{.*}} : !cir.vector<4 x !cir.f16> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <4 x half> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <4 x half> [[A]] to <4 x i16>
// LLVM: [[A_BYTES:%.*]] = bitcast <4 x i16> [[A_I]] to <8 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <8 x i8> [[A_BYTES]] to <4 x half>
// LLVM: [[SQRT:%.*]] = call <4 x half> @llvm.experimental.constrained.sqrt.v4f16(<4 x half> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <4 x half> [[SQRT]]
  return vsqrt_f16(a);
}

// LLVM-LABEL: @test_vsqrtq_f16(
// CIR-LABEL: @vsqrtq_f16(
float16x8_t test_vsqrtq_f16(float16x8_t a) {
// CIR: cir.sqrt %{{.*}} : !cir.vector<8 x !cir.f16> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <8 x half> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <8 x half> [[A]] to <8 x i16>
// LLVM: [[A_BYTES:%.*]] = bitcast <8 x i16> [[A_I]] to <16 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <8 x half>
// LLVM: [[SQRT:%.*]] = call <8 x half> @llvm.experimental.constrained.sqrt.v8f16(<8 x half> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <8 x half> [[SQRT]]
  return vsqrtq_f16(a);
}

// LLVM-LABEL: @test_vsqrtq_f64(
// CIR-LABEL: @vsqrtq_f64(
float64x2_t test_vsqrtq_f64(float64x2_t a) {
// CIR: cir.sqrt %{{.*}} : !cir.vector<2 x !cir.double> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM: [[SQRT:%.*]] = call <2 x double> @llvm.experimental.constrained.sqrt.v2f64(<2 x double> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <2 x double> [[SQRT]]
  return vsqrtq_f64(a);
}
