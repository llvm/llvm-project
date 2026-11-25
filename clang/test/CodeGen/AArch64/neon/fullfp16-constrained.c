// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1_cg_arm64_neon -target-feature +fullfp16 -ffp-exception-behavior=strict -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefixes=ALL,LLVM --implicit-check-not=' @llvm.fma.' --implicit-check-not=' @llvm.sqrt.'
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=maytrap -DEXCEPT=1 -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefixes=ALL,LLVM --implicit-check-not=fpexcept.maytrap --implicit-check-not=' @llvm.fma.' --implicit-check-not=' @llvm.sqrt.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=maytrap -DEXCEPT=1 -fclangir -emit-cir  %s -disable-O0-optnone |                                      FileCheck %s --check-prefixes=ALL,CIR --implicit-check-not='except_mode = maytrap' --implicit-check-not='cir.call_llvm_intrinsic "fma"' --implicit-check-not='cir.call_llvm_intrinsic "sqrt"' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=strict             -fclangir -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefixes=ALL,LLVM --implicit-check-not=' @llvm.fma.' --implicit-check-not=' @llvm.sqrt.' %}
// RUN: %if cir-enabled %{%clang_cc1_cg_arm64_neon -target-feature +fullfp16 -fexperimental-strict-floating-point -ffp-exception-behavior=strict             -fclangir -emit-cir  %s -disable-O0-optnone |                                      FileCheck %s --check-prefixes=ALL,CIR --implicit-check-not='cir.call_llvm_intrinsic "fma"' --implicit-check-not='cir.call_llvm_intrinsic "sqrt"' %}

#if EXCEPT
#pragma float_control(except, on)
#endif

#include <arm_fp16.h>

// ALL-LABEL: @test_vsqrth_f16(
float16_t test_vsqrth_f16(float16_t a) {
// CIR: cir.sqrt %{{.*}} : !cir.f16 fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: half {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[SQRT:%.*]] = call half @llvm.experimental.constrained.sqrt.f16(half [[A]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret half [[SQRT]]
  return vsqrth_f16(a);
}

// ALL-LABEL: @test_vfmah_f16(
float16_t test_vfmah_f16(float16_t a, float16_t b, float16_t c) {
// CIR: cir.fma %{{.*}}, %{{.*}}, %{{.*}} : !cir.f16 fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]], half {{.*}} [[C:%.*]]) {{.*}} {
// LLVM: [[FMA:%.*]] = call half @llvm.experimental.constrained.fma.f16(half [[B]], half [[C]], half [[A]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret half [[FMA]]
  return vfmah_f16(a, b, c);
}

// ALL-LABEL: @test_vceqzh_f16(
uint16_t test_vceqzh_f16(float16_t a) {
// CIR: cir.fcmp oeq %{{.*}}, %{{.*}} : !cir.f16 fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: half {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f16(half [[A]], half 0.000000e+00, metadata !"oeq", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i16
// LLVM: ret i16 [[RES]]
  return vceqzh_f16(a);
}

// ALL-LABEL: @test_vcgezh_f16(
uint16_t test_vcgezh_f16(float16_t a) {
// CIR: cir.fcmp oge %{{.*}}, %{{.*}} : !cir.f16 signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: half {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half [[A]], half 0.000000e+00, metadata !"oge", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i16
// LLVM: ret i16 [[RES]]
  return vcgezh_f16(a);
}

// ALL-LABEL: @test_vclezh_f16(
uint16_t test_vclezh_f16(float16_t a) {
// CIR: cir.fcmp ole %{{.*}}, %{{.*}} : !cir.f16 signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: half {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half [[A]], half 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i16
// LLVM: ret i16 [[RES]]
  return vclezh_f16(a);
}

// ALL-LABEL: @test_vcgtzh_f16(
uint16_t test_vcgtzh_f16(float16_t a) {
// CIR: cir.fcmp ogt %{{.*}}, %{{.*}} : !cir.f16 signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: half {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half [[A]], half 0.000000e+00, metadata !"ogt", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i16
// LLVM: ret i16 [[RES]]
  return vcgtzh_f16(a);
}

// ALL-LABEL: @test_vcltzh_f16(
uint16_t test_vcltzh_f16(float16_t a) {
// CIR: cir.fcmp olt %{{.*}}, %{{.*}} : !cir.f16 signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: half {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half [[A]], half 0.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i16
// LLVM: ret i16 [[RES]]
  return vcltzh_f16(a);
}
