// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --implicit-check-not=fpexcept.ignore
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --implicit-check-not=fpexcept.ignore

// An operation that is evaluated in a wider type converts its operands to that
// type and its result back. The conversions are part of the operation and take
// its environment, not the default environment of the function. ClangIR and
// classic codegen share the check lines.
//
// Without AVX512-FP16, _Float16 arithmetic is done in float.
//
// FIXME: ClangIR does not constrain the conversions of a complex operand and of
// a complex result.

#pragma STDC FENV_ACCESS ON

_Complex _Float16 a, c;
_Float16 x, y, z;
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
half4 vx, vy, vz;
float f;
double d;

void complex_mul(void) { c = a * x; }

// CHECK-LABEL: define{{.*}} void @complex_mul()
// CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

void mul(void) { z = x * y; }

// CHECK-LABEL: define{{.*}} void @mul()
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

void neg(void) { z = -x; }

// CHECK-LABEL: define{{.*}} void @neg()
// CHECK: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

void mul_assign(void) { z *= y; }

// CHECK-LABEL: define{{.*}} void @mul_assign()
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fpext.f32.f16(half %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.fmul.f32(float %{{.*}}, float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK: call half @llvm.experimental.constrained.fptrunc.f16.f32(float %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

void vector_mul(void) { vz = vx * vy; }

// CHECK-LABEL: define{{.*}} void @vector_mul()
// CHECK-COUNT-2: call <4 x float> @llvm.experimental.constrained.fpext.v4f32.v4f16(<4 x half> %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call <4 x half> @llvm.experimental.constrained.fptrunc.v4f16.v4f32(<4 x float> %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

// The conversions of a compound assignment are part of it too.

void float_add_assign(void) { f += d; }

// CHECK-LABEL: define{{.*}} void @float_add_assign()
// CHECK: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK: call double @llvm.experimental.constrained.fadd.f64(double %{{.*}}, double %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK: call float @llvm.experimental.constrained.fptrunc.f32.f64(double %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
