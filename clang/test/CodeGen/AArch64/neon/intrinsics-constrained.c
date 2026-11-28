// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1_cg_arm64_neon -target-feature +fullfp16 -ffp-exception-behavior=maytrap -DEXCEPT=1 -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM --implicit-check-not=fpexcept.maytrap --implicit-check-not=' @llvm.sqrt.'
// RUN: %clang_cc1_cg_arm64_neon -target-feature +fullfp16 -ffp-exception-behavior=strict -emit-llvm %s -disable-O0-optnone | opt -S -passes=mem2reg,sroa | FileCheck %s --check-prefix=LLVM --implicit-check-not=' @llvm.sqrt.'
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
// CIR: cir.sqrt %{{.*}} : !cir.vector<4 x !cir.f16> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

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
// CIR: cir.sqrt %{{.*}} : !cir.vector<8 x !cir.f16> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

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
// CIR: cir.sqrt %{{.*}} : !cir.vector<2 x !cir.double> fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: <2 x double> {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[A_I:%.*]] = bitcast <2 x double> [[A]] to <2 x i64>
// LLVM: [[A_BYTES:%.*]] = bitcast <2 x i64> [[A_I]] to <16 x i8>
// LLVM: [[A_CAST:%.*]] = bitcast <16 x i8> [[A_BYTES]] to <2 x double>
// LLVM: [[SQRT:%.*]] = call <2 x double> @llvm.experimental.constrained.sqrt.v2f64(<2 x double> [[A_CAST]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// LLVM: ret <2 x double> [[SQRT]]
  return vsqrtq_f64(a);
}

// LLVM-LABEL: @test_vceqzs_f32(
// CIR-LABEL: @vceqzs_f32(
uint32_t test_vceqzs_f32(float32_t a) {
// CIR: cir.fcmp oeq %{{.*}}, %{{.*}} : !cir.float fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: float {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float [[A]], float 0.000000e+00, metadata !"oeq", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i32
// LLVM: ret i32 [[RES]]
  return vceqzs_f32(a);
}

// LLVM-LABEL: @test_vcgezs_f32(
// CIR-LABEL: @vcgezs_f32(
uint32_t test_vcgezs_f32(float32_t a) {
// CIR: cir.fcmp oge %{{.*}}, %{{.*}} : !cir.float signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: float {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f32(float [[A]], float 0.000000e+00, metadata !"oge", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i32
// LLVM: ret i32 [[RES]]
  return vcgezs_f32(a);
}

// LLVM-LABEL: @test_vclezs_f32(
// CIR-LABEL: @vclezs_f32(
uint32_t test_vclezs_f32(float32_t a) {
// CIR: cir.fcmp ole %{{.*}}, %{{.*}} : !cir.float signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: float {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f32(float [[A]], float 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i32
// LLVM: ret i32 [[RES]]
  return vclezs_f32(a);
}

// LLVM-LABEL: @test_vcgtzs_f32(
// CIR-LABEL: @vcgtzs_f32(
uint32_t test_vcgtzs_f32(float32_t a) {
// CIR: cir.fcmp ogt %{{.*}}, %{{.*}} : !cir.float signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: float {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f32(float [[A]], float 0.000000e+00, metadata !"ogt", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i32
// LLVM: ret i32 [[RES]]
  return vcgtzs_f32(a);
}

// LLVM-LABEL: @test_vcltzs_f32(
// CIR-LABEL: @vcltzs_f32(
uint32_t test_vcltzs_f32(float32_t a) {
// CIR: cir.fcmp olt %{{.*}}, %{{.*}} : !cir.float signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: float {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f32(float [[A]], float 0.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i32
// LLVM: ret i32 [[RES]]
  return vcltzs_f32(a);
}

// LLVM-LABEL: @test_vceqzd_f64(
// CIR-LABEL: @vceqzd_f64(
uint64_t test_vceqzd_f64(float64_t a) {
// CIR: cir.fcmp oeq %{{.*}}, %{{.*}} : !cir.double fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: double {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double [[A]], double 0.000000e+00, metadata !"oeq", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i64
// LLVM: ret i64 [[RES]]
  return vceqzd_f64(a);
}

// LLVM-LABEL: @test_vcgezd_f64(
// CIR-LABEL: @vcgezd_f64(
uint64_t test_vcgezd_f64(float64_t a) {
// CIR: cir.fcmp oge %{{.*}}, %{{.*}} : !cir.double signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: double {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f64(double [[A]], double 0.000000e+00, metadata !"oge", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i64
// LLVM: ret i64 [[RES]]
  return vcgezd_f64(a);
}

// LLVM-LABEL: @test_vclezd_f64(
// CIR-LABEL: @vclezd_f64(
uint64_t test_vclezd_f64(float64_t a) {
// CIR: cir.fcmp ole %{{.*}}, %{{.*}} : !cir.double signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: double {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f64(double [[A]], double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i64
// LLVM: ret i64 [[RES]]
  return vclezd_f64(a);
}

// LLVM-LABEL: @test_vcgtzd_f64(
// CIR-LABEL: @vcgtzd_f64(
uint64_t test_vcgtzd_f64(float64_t a) {
// CIR: cir.fcmp ogt %{{.*}}, %{{.*}} : !cir.double signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: double {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f64(double [[A]], double 0.000000e+00, metadata !"ogt", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i64
// LLVM: ret i64 [[RES]]
  return vcgtzd_f64(a);
}

// LLVM-LABEL: @test_vcltzd_f64(
// CIR-LABEL: @vcltzd_f64(
uint64_t test_vcltzd_f64(float64_t a) {
// CIR: cir.fcmp olt %{{.*}}, %{{.*}} : !cir.double signaling fenv<dynamic_rounding_mode = tonearest, except_mode = unknown, strict_except = true>

// LLVM-SAME: double {{.*}} [[A:%.*]]) {{.*}} {
// LLVM: [[CMP:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f64(double [[A]], double 0.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
// LLVM: [[RES:%.*]] = sext i1 [[CMP]] to i64
// LLVM: ret i64 [[RES]]
  return vcltzd_f64(a);
}
