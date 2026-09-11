// REQUIRES: aarch64-registered-target

// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=maytrap -DEXCEPT=1 -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefix=LLVM --implicit-check-not=fpexcept.maytrap --implicit-check-not=' @llvm.fma.' --implicit-check-not=' @llvm.sqrt.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=maytrap -DEXCEPT=1 -fclangir -emit-cir  %s -disable-O0-optnone |                                      FileCheck %s --check-prefix=CIR --implicit-check-not='except_mode = maytrap' --implicit-check-not='cir.call_llvm_intrinsic "fma"' --implicit-check-not='cir.call_llvm_intrinsic "sqrt"' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=strict             -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefix=LLVM --implicit-check-not=' @llvm.fma.' --implicit-check-not=' @llvm.sqrt.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=strict             -fclangir -emit-cir  %s -disable-O0-optnone |                                      FileCheck %s --check-prefix=CIR --implicit-check-not='cir.call_llvm_intrinsic "fma"' --implicit-check-not='cir.call_llvm_intrinsic "sqrt"' %}

#if EXCEPT
#pragma float_control(except, on)
#endif

#include <arm_fp16.h>

// LLVM-LABEL: @test_vsqrth_f16(
// CIR-LABEL: @test_vsqrth_f16(
float16_t test_vsqrth_f16(float16_t a) {
// CIR: cir.sqrt %{{.*}} : !cir.f16 {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: half {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[SQRT:%.*]] = call half @llvm.experimental.constrained.sqrt.f16(half [[A]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret half [[SQRT]]
  return vsqrth_f16(a);
}

// LLVM-LABEL: @test_vfmah_f16(
// CIR-LABEL: @test_vfmah_f16(
float16_t test_vfmah_f16(float16_t a, float16_t b, float16_t c) {
// CIR: cir.fma %{{.*}}, %{{.*}}, %{{.*}} : !cir.f16 {fenv = #cir.fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>}

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], half {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[FMA:%.*]] = call half @llvm.experimental.constrained.fma.f16(half [[B]], half [[C]], half [[A]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret half [[FMA]]
  return vfmah_f16(a, b, c);
}
