// RUN: %clang_cc1 -triple sparc-unknown-linux-gnu -frounding-math \
// RUN:   -ffp-exception-behavior=strict -O2 -verify -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,STRICT
// RUN: %clang_cc1 -triple sparcv9-unknown-linux-gnu -frounding-math \
// RUN:   -ffp-exception-behavior=strict -O2 -verify -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,STRICT
// RUN: %clang_cc1 -triple sparcv9-unknown-openbsd -frounding-math -O2 -verify \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ROUNDING
// expected-no-diagnostics

// SPARC supports constrained floating-point, so -frounding-math and
// -ffp-exception-behavior=strict are honoured rather than overridden, and
// floating-point operations are not constant folded.

// CHECK-LABEL: define{{.*}} double @div(
// STRICT: call double @llvm.experimental.constrained.fdiv.f64(double 1.000000e+00, double 3.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict")
// ROUNDING: call double @llvm.experimental.constrained.fdiv.f64(double 1.000000e+00, double 3.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.ignore")
double div(void) { return 1.0 / 3.0; }

// CHECK-LABEL: define{{.*}} float @fmuladd(
// STRICT: call float @llvm.experimental.constrained.fmul.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.strict")
// STRICT: call float @llvm.experimental.constrained.fadd.f32(float %{{.*}}, float %c, metadata !"round.dynamic", metadata !"fpexcept.strict")
float fmuladd(float a, float b, float c) { return a * b + c; }

// CHECK-LABEL: define{{.*}} i32 @less(
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f64(double %a, double %b, metadata !"olt", metadata !"fpexcept.strict")
int less(double a, double b) { return a < b; }

// CHECK-LABEL: define{{.*}} i32 @equal(
// STRICT: call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"oeq", metadata !"fpexcept.strict")
int equal(double a, double b) { return a == b; }
