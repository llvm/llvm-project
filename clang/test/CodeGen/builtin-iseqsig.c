// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,IGNORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,IGNORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -ffp-exception-behavior=strict | FileCheck %s --check-prefixes=CHECK,STRICT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -ffp-exception-behavior=strict -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,STRICT

// __builtin_iseqsig is a signaling equality compare: it is emitted as a
// constrained fcmps when FP exceptions are modeled, and as fcmp oeq otherwise.

// CHECK-LABEL: define {{.*}}i32 @f_float(
// IGNORE: fcmp oeq float %{{.*}}, %{{.*}}
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
int f_float(float x, float y) { return __builtin_iseqsig(x, y); }

// CHECK-LABEL: define {{.*}}i32 @f_double(
// IGNORE: fcmp oeq double %{{.*}}, %{{.*}}
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
int f_double(double x, double y) { return __builtin_iseqsig(x, y); }

// CHECK-LABEL: define {{.*}}i32 @f_ldouble(
// IGNORE: fcmp oeq x86_fp80 %{{.*}}, %{{.*}}
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f80(x86_fp80 %{{.*}}, x86_fp80 %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
int f_ldouble(long double x, long double y) { return __builtin_iseqsig(x, y); }

// CHECK-LABEL: define {{.*}}i32 @f_float16(
// IGNORE: fcmp oeq half %{{.*}}, %{{.*}}
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f16(half %{{.*}}, half %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
int f_float16(_Float16 x, _Float16 y) { return __builtin_iseqsig(x, y); }

// Mixed types use the usual arithmetic conversions, like the other comparison
// builtins.
// CHECK-LABEL: define {{.*}}i32 @f_mixed(
// IGNORE: fpext float %{{.*}} to double
// STRICT: call double @llvm.experimental.constrained.fpext.f64.f32(
// IGNORE: fcmp oeq double %{{.*}}, %{{.*}}
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
int f_mixed(float x, double y) { return __builtin_iseqsig(x, y); }

// CHECK-LABEL: define {{.*}}i32 @f_int(
// IGNORE: sitofp i32 %{{.*}} to double
// STRICT: call double @llvm.experimental.constrained.sitofp.f64.i32(
// IGNORE: fcmp oeq double %{{.*}}, %{{.*}}
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
int f_int(double x, int y) { return __builtin_iseqsig(x, y); }

// Constant operands. A NaN raises "invalid", so it is only folded when FP
// exceptions are not modeled.
// CHECK-LABEL: define {{.*}}i32 @c_qnan(
// IGNORE-NEXT: entry:
// IGNORE-NEXT: ret i32 0
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f64(double +qnan, double 1.000000e+00, metadata !"oeq", metadata !"fpexcept.strict")
int c_qnan(void) { return __builtin_iseqsig(__builtin_nan(""), 1.0); }

// CHECK-LABEL: define {{.*}}i32 @c_snan(
// IGNORE-NEXT: entry:
// IGNORE-NEXT: ret i32 0
// STRICT: call i1 @llvm.experimental.constrained.fcmps.f64(double 1.000000e+00, double +snan(0x4000000000000), metadata !"oeq", metadata !"fpexcept.strict")
int c_snan(void) { return __builtin_iseqsig(1.0, __builtin_nans("")); }

// CHECK-LABEL: define {{.*}}i32 @c_equal(
// CHECK-NEXT: entry:
// CHECK-NEXT: ret i32 1
int c_equal(void) { return __builtin_iseqsig(0.0, -0.0); }

// CHECK-LABEL: define {{.*}}i32 @c_unequal(
// CHECK-NEXT: entry:
// CHECK-NEXT: ret i32 0
int c_unequal(void) { return __builtin_iseqsig(1.0, __builtin_inf()); }

// FENV_ACCESS also makes the compare signaling and keeps it unfolded.
#pragma STDC FENV_ACCESS ON
// CHECK-LABEL: define {{.*}}i32 @fenv_qnan(
// CHECK: call i1 @llvm.experimental.constrained.fcmps.f64(double +qnan, double 1.000000e+00, metadata !"oeq", metadata !"fpexcept.strict")
int fenv_qnan(void) { return __builtin_iseqsig(__builtin_nan(""), 1.0); }
