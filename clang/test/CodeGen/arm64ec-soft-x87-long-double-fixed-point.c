// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -ffixed-point -emit-llvm -o - %s | FileCheck %s

// A conversion to `bool` tests the `x86_fp80` value directly against zero,
// so a magnitude below the smallest `double` is still nonzero.
// CHECK-LABEL: define dso_local i32 @to_bool(
// CHECK-NOT:     x87ld.to.double
// CHECK:         fcmp une x86_fp80 {{%.*}}, 0.000000e+00
int to_bool(long double x) { return x ? 1 : 0; }

// CHECK-LABEL: define dso_local i32 @not_ld(
// CHECK-NOT:     x87ld.to.double
// CHECK:         fcmp une x86_fp80 {{%.*}}, 0.000000e+00
int not_ld(long double x) { return !x; }

// A conversion between `long double` and a fixed-point type is an ordinary
// floating-point conversion, which the backend softens.
// CHECK-LABEL: define dso_local i128 @from_accum(
// CHECK:         sitofp i16 {{%.*}} to x86_fp80
long double from_accum(short _Accum a) { return a; }

// CHECK-LABEL: define dso_local i16 @to_accum(
// CHECK:         fptosi x86_fp80 {{%.*}} to i16
short _Accum to_accum(long double x) { return x; }

// CHECK-LABEL: define dso_local i128 @from_fract(
// CHECK:         sitofp i8 {{%.*}} to x86_fp80
long double from_fract(short _Fract a) { return a; }

// CHECK-LABEL: define dso_local i8 @to_fract(
// CHECK:         fptosi x86_fp80 {{%.*}} to i8
short _Fract to_fract(long double x) { return x; }

// A fixed-point to `bool` conversion does not arrive here at all, and it must
// not be routed through `double` either.
// CHECK-LABEL: define dso_local i32 @accum_to_bool(
// CHECK-NOT:     x87ld
// CHECK:         icmp ne i16 {{%.*}}, 0
int accum_to_bool(short _Accum a) { return a ? 1 : 0; }
