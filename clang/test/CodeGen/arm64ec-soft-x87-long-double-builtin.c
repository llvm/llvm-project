// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -emit-llvm -o - %s | FileCheck %s

// The value itself is a real `x86_fp80`, so every builtin that takes or
// returns a floating-point value is emitted as the ordinary floating-point
// intrinsic and softened by the backend; the frontend does not lower the
// operation on the integer image.

// `__builtin_elementwise_abs` is `fabsl`.
// CHECK-LABEL: define dso_local i128 @elementwise_abs(
// CHECK:         call x86_fp80 @llvm.fabs.f80
long double elementwise_abs(long double x) { return __builtin_elementwise_abs(x); }

// `__builtin_elementwise_copysign` is `copysignl`.
// CHECK-LABEL: define dso_local i128 @elementwise_copysign(
// CHECK:         call x86_fp80 @llvm.copysign.f80
long double elementwise_copysign(long double x, long double y) {
  return __builtin_elementwise_copysign(x, y);
}

// `__builtin_canonicalizel` is a real floating-point canonicalization.
// CHECK-LABEL: define dso_local i128 @canonical(
// CHECK:         call x86_fp80 @llvm.canonicalize.f80
long double canonical(long double x) { return __builtin_canonicalizel(x); }

// The constants stay `x86_fp80` constants and are widened to the 16-byte
// image by the backend, where the sign and the payload are preserved.
// CHECK-LABEL: define dso_local i128 @canonical_signalling(
// CHECK:         call x86_fp80 @llvm.canonicalize.f80(x86_fp80 +snan(
long double canonical_signalling(void) {
  return __builtin_canonicalizel(__builtin_nansl(""));
}

// CHECK-LABEL: define dso_local i128 @canonical_negative_signalling(
// CHECK:         call x86_fp80 @llvm.canonicalize.f80(x86_fp80 -snan(
long double canonical_negative_signalling(void) {
  return __builtin_canonicalizel(-__builtin_nansl(""));
}

// CHECK-LABEL: define dso_local i128 @canonical_quiet(
// CHECK:         call x86_fp80 @llvm.canonicalize.f80(x86_fp80 +qnan)
long double canonical_quiet(void) {
  return __builtin_canonicalizel(__builtin_nanl(""));
}

// CHECK-LABEL: define dso_local i128 @canonical_infinity(
// CHECK:         call x86_fp80 @llvm.canonicalize.f80(x86_fp80 +inf)
long double canonical_infinity(void) {
  return __builtin_canonicalizel(__builtin_infl());
}

// `__builtin_iszero`, `__builtin_issubnormal` and `__builtin_issignaling`
// classify the floating-point value with the generic FP-class intrinsic; the
// masks are the ones defined by the LangRef.
// CHECK-LABEL: define dso_local i32 @is_zero(
// CHECK:         call i1 @llvm.is.fpclass.f80({{.*}}i32 96)
int is_zero(long double x) { return __builtin_iszero(x); }

// CHECK-LABEL: define dso_local i32 @is_subnormal(
// CHECK:         call i1 @llvm.is.fpclass.f80({{.*}}i32 144)
int is_subnormal(long double x) { return __builtin_issubnormal(x); }

// CHECK-LABEL: define dso_local i32 @is_signaling(
// CHECK:         call i1 @llvm.is.fpclass.f80({{.*}}i32 1)
int is_signaling(long double x) { return __builtin_issignaling(x); }
