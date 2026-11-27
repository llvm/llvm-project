// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s \
// RUN:   --implicit-check-not="{{= (fpext|fptrunc|fptosi|sitofp|fcmp|fadd) }}"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// Complex casts and conversions to bool under FENV_ACCESS. ClangIR and
// classic codegen share every CHECK line, so both emit the same intrinsics
// with the same metadata. The lines end before the call-site attributes. The
// implicit check rejects a plain instruction in the ClangIR output that no
// CHECK line matches.
//
// FIXME: ClangIR does not mark the calls in a strictfp function `strictfp`.

#pragma STDC FENV_ACCESS ON

_Complex float cf;
_Complex double cd;
_Complex int ci;
_Bool b;
float f;
double d;
int i;

void float_complex(void) { cd = cf; }

// CHECK-LABEL: define{{.*}} void @float_complex()
// CHECK-COUNT-2: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")

void float_complex_to_int_complex(void) { ci = cf; }

// CHECK-LABEL: define{{.*}} void @float_complex_to_int_complex()
// CHECK-COUNT-2: call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %{{.*}}, metadata !"fpexcept.strict")

void int_complex_to_float_complex(void) { cf = ci; }

// CHECK-LABEL: define{{.*}} void @int_complex_to_float_complex()
// CHECK-COUNT-2: call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

// A narrowing cast rounds, so it takes the rounding mode as well.

_Complex float float_complex_narrowing(_Complex double x) {
  return (_Complex float)x;
}

// CHECK-LABEL: define{{.*}} @float_complex_narrowing(
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fptrunc.f32.f64(double %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

// A compound assignment converts the left operand to the type of the
// computation and the result back.

void compound_assign(_Complex double x) { cf += x; }

// CHECK-LABEL: define{{.*}} void @compound_assign(
// CHECK-COUNT-2: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")
// CHECK-COUNT-2: call double @llvm.experimental.constrained.fadd.f64(double %{{.*}}, double %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK-COUNT-2: call float @llvm.experimental.constrained.fptrunc.f32.f64(double %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")

// A conversion between a real and a complex value of another width converts
// the real value with a scalar cast.

void real_to_complex(void) { cd = f; }

// CHECK-LABEL: define{{.*}} void @real_to_complex()
// CHECK: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")

void complex_to_real(void) { d = cf; }

// CHECK-LABEL: define{{.*}} void @complex_to_real()
// CHECK: call double @llvm.experimental.constrained.fpext.f64.f32(float %{{.*}}, metadata !"fpexcept.strict")

// A conversion to bool compares both parts with zero. The comparison is
// quiet.

void float_complex_to_bool(void) { b = cf; }

// CHECK-LABEL: define{{.*}} void @float_complex_to_bool()
// CHECK: %[[REAL:.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")
// CHECK: %[[IMAG:.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")
// CHECK: or i1 %[[REAL]], %[[IMAG]]

void float_to_bool(void) { b = f; }

// CHECK-LABEL: define{{.*}} void @float_to_bool()
// CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")

// A condition is converted to bool without a cast expression.

void complex_condition(void) {
  if (cf)
    i = 1;
}

// CHECK-LABEL: define{{.*}} void @complex_condition()
// CHECK-COUNT-2: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")

void float_condition(void) {
  if (f)
    i = 1;
}

// CHECK-LABEL: define{{.*}} void @float_condition()
// CHECK: call i1 @llvm.experimental.constrained.fcmp.f32(float %{{.*}}, float 0.000000e+00, metadata !"une", metadata !"fpexcept.strict")

// Without FENV_ACCESS the casts stay plain instructions.

#pragma STDC FENV_ACCESS OFF

void unconstrained(void) {
  cd = cf;
  b = cf;
}

// CHECK-LABEL: define{{.*}} void @unconstrained()
// CHECK-COUNT-2: fpext float %{{.*}} to double
// CHECK-COUNT-2: fcmp une float %{{.*}}, 0.000000e+00
