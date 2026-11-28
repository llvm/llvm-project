// RUN: %clang_cc1 -triple x86_64-unknown-unknown -complex-range=promoted \
// RUN:   -ffp-exception-behavior=maytrap -emit-llvm -o - %s \
// RUN:   | FileCheck --implicit-check-not=fpexcept.maytrap \
// RUN:               --implicit-check-not=round.tonearest %s

// An operation that is evaluated in a wider type converts its operands to that
// type and its result back. Test that these conversions pick up the exception
// and rounding metadata of the operation from the AST instead of the global
// default from the command line.
// Any cases of "fpexcept.maytrap" in this test are clang bugs.
// Any cases of "round.tonearest" in this test are clang bugs.

#pragma float_control(except, on)
#pragma STDC FENV_ROUND FE_UPWARD

_Complex float a, b, c;
_Complex _Float16 ha, hb, hc;
_Float16 x, y, z;
__bf16 bx, by, bz;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
half4 vx, vy, vz;

// The promoted complex range divides in double.

// CHECK-LABEL: @div(
// CHECK-COUNT-4: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK-COUNT-11: call double @llvm.experimental.constrained.f{{mul|add|sub|div}}.f64(double %{{.*}}, double %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fptrunc.f32.f64(double %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void div(void) {
  c = a / b;
}

// Without AVX512-FP16, _Float16 arithmetic is done in float. __bf16 arithmetic
// is done in float too.

// CHECK-LABEL: @complex_mul(
// CHECK-COUNT-4: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK-COUNT-2: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void complex_mul(void) {
  hc = ha * hb;
}

// CHECK-LABEL: @complex_neg(
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK-COUNT-2: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void complex_neg(void) {
  hc = -ha;
}

// CHECK-LABEL: @complex_plus(
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK-COUNT-2: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void complex_plus(void) {
  hc = +ha;
}

// CHECK-LABEL: @mul(
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
// CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void mul(void) {
  z = x * y;
}

// CHECK-LABEL: @neg(
// CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void neg(void) {
  z = -x;
}

// CHECK-LABEL: @plus(
// CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void plus(void) {
  z = +x;
}

// CHECK-LABEL: @mul_assign(
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void mul_assign(void) {
  z *= y;
}

// CHECK-LABEL: @bf16_mul(
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fpext.f32.bf16(bfloat %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call bfloat @llvm.experimental.constrained.fptrunc.bf16.f32(float %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void bf16_mul(void) {
  bz = bx * by;
}

// CHECK-LABEL: @vector_mul(
// CHECK-COUNT-2: call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call <4 x half> @llvm.experimental.constrained.fptrunc.v4f16.v4f32(<4 x float> %{{.*}}, metadata !"round.upward", metadata !"fpexcept.strict")
void vector_mul(void) {
  vz = vx * vy;
}
