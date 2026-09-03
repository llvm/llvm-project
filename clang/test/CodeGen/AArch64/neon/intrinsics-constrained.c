// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM --implicit-check-not=' @llvm.sqrt.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -fexperimental-strict-floating-point -ffp-exception-behavior=strict -fclangir -emit-cir  %s -disable-O0-optnone |                               FileCheck %s --check-prefix=CIR --implicit-check-not='cir.call_llvm_intrinsic "sqrt"' %}

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

// LLVM-LABEL: @test_vsqrtq_f64(
// LLVM: call <2 x double> @llvm.experimental.constrained.sqrt.v2f64({{.*}}, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CIR-LABEL: cir.func {{.*}}@test_vsqrtq_f64(
// CIR: cir.sqrt %{{.*}} : !cir.vector<2 x !cir.double> {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}
float64x2_t test_vsqrtq_f64(float64x2_t a) {
  return vsqrtq_f64(a);
}
