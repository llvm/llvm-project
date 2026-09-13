// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -w -o - -fmath-errno %s | FileCheck %s --check-prefix=CHECK-ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -w -o - -fno-math-errno %s | FileCheck %s --check-prefix=CHECK-NO-ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -w -o - -frounding-math %s | FileCheck %s --check-prefix=CHECK-ROUNDING
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -w -o - -ffp-exception-behavior=strict %s | FileCheck %s --check-prefix=CHECK-STRICT

// An exact or inexact exp that doesn't set errno folds even with -fmath-errno:
float test_expf_1() {
  return __builtin_expf(1.0f);
}
// CHECK-ERRNO-LABEL: @test_expf_1
// CHECK-ERRNO: ret float 0x3FF5BF0A80000000
// CHECK-NO-ERRNO-LABEL: @test_expf_1
// CHECK-NO-ERRNO: ret float 0x3FF5BF0A80000000
// CHECK-ROUNDING-LABEL: @test_expf_1
// CHECK-ROUNDING: call float @llvm.experimental.constrained.exp.f32(float 1.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.ignore")

// Exact exp(0.0) folds even with -frounding-math and strict exception mode:
float test_expf_zero() {
  return __builtin_expf(0.0f);
}
// CHECK-ERRNO-LABEL: @test_expf_zero
// CHECK-ERRNO: ret float 1.000000e+00
// CHECK-NO-ERRNO-LABEL: @test_expf_zero
// CHECK-NO-ERRNO: ret float 1.000000e+00
// CHECK-ROUNDING-LABEL: @test_expf_zero
// CHECK-ROUNDING: ret float 1.000000e+00
// CHECK-STRICT-LABEL: @test_expf_zero
// CHECK-STRICT: ret float 1.000000e+00

// Exact infinities do not set errno, so they fold even under -fmath-errno:
float test_expf_pos_inf() {
  return __builtin_expf(__builtin_inff());
}
// CHECK-ERRNO-LABEL: @test_expf_pos_inf
// CHECK-ERRNO: ret float +inf
// CHECK-NO-ERRNO-LABEL: @test_expf_pos_inf
// CHECK-NO-ERRNO: ret float +inf

float test_expf_neg_inf() {
  return __builtin_expf(-__builtin_inff());
}
// CHECK-ERRNO-LABEL: @test_expf_neg_inf
// CHECK-ERRNO: ret float 0.000000e+00
// CHECK-NO-ERRNO-LABEL: @test_expf_neg_inf
// CHECK-NO-ERRNO: ret float 0.000000e+00

// Quiet NaN does not set errno or raise exceptions, so it folds to NaN:
float test_expf_nan() {
  return __builtin_expf(__builtin_nanf(""));
}
// CHECK-ERRNO-LABEL: @test_expf_nan
// CHECK-ERRNO: ret float 0x7FF8000000000000
// CHECK-NO-ERRNO-LABEL: @test_expf_nan
// CHECK-NO-ERRNO: ret float 0x7FF8000000000000


// Inexact exp raises FE_INEXACT, so it cannot fold under strict exception mode:
float test_expf_strict_inexact() {
  return __builtin_expf(1.0f);
}
// CHECK-STRICT-LABEL: @test_expf_strict_inexact
// CHECK-STRICT: call float @llvm.experimental.constrained.exp.f32(float 1.000000e+00, metadata !"round.tonearest", metadata !"fpexcept.strict")

// Overflow sets errno under -fmath-errno, so it cannot fold:
float test_expf_overflow() {
  return __builtin_expf(100.0f);
}
// CHECK-ERRNO-LABEL: @test_expf_overflow
// CHECK-ERRNO: call float @expf(float noundef 1.000000e+02)
// CHECK-NO-ERRNO-LABEL: @test_expf_overflow
// CHECK-NO-ERRNO: ret float +inf
// CHECK-STRICT-LABEL: @test_expf_overflow
// CHECK-STRICT: call float @llvm.experimental.constrained.exp.f32(float 1.000000e+02, metadata !"round.tonearest", metadata !"fpexcept.strict")

// Underflow to zero sets errno under -fmath-errno, so it cannot fold:
float test_expf_underflow() {
  return __builtin_expf(-100.0f);
}
// CHECK-ERRNO-LABEL: @test_expf_underflow
// CHECK-ERRNO: call float @expf(float noundef -1.000000e+02)
// CHECK-NO-ERRNO-LABEL: @test_expf_underflow
// CHECK-NO-ERRNO: ret float 0.000000e+00
// CHECK-STRICT-LABEL: @test_expf_underflow
// CHECK-STRICT: call float @llvm.experimental.constrained.exp.f32(float -1.000000e+02, metadata !"round.tonearest", metadata !"fpexcept.strict")

// Underflow to denormal sets errno under -fmath-errno, so it cannot fold:
float test_expf_denormal() {
  return __builtin_expf(-88.0f);
}
// CHECK-ERRNO-LABEL: @test_expf_denormal
// CHECK-ERRNO: call float @expf(float noundef -8.800000e+01)
// CHECK-NO-ERRNO-LABEL: @test_expf_denormal
// CHECK-NO-ERRNO: ret float 0x37F6DC7000000000
// CHECK-STRICT-LABEL: @test_expf_denormal
// CHECK-STRICT: call float @llvm.experimental.constrained.exp.f32(float -8.800000e+01, metadata !"round.tonearest", metadata !"fpexcept.strict")

double test_exp_overflow() {
  return __builtin_exp(1000.0);
}
// CHECK-ERRNO-LABEL: @test_exp_overflow
// CHECK-ERRNO: call double @exp(double noundef 1.000000e+03)
// CHECK-NO-ERRNO-LABEL: @test_exp_overflow
// CHECK-NO-ERRNO: ret double +inf
// CHECK-STRICT-LABEL: @test_exp_overflow
// CHECK-STRICT: call double @llvm.experimental.constrained.exp.f64(double 1.000000e+03, metadata !"round.tonearest", metadata !"fpexcept.strict")

double test_exp_underflow() {
  return __builtin_exp(-1000.0);
}
// CHECK-ERRNO-LABEL: @test_exp_underflow
// CHECK-ERRNO: call double @exp(double noundef -1.000000e+03)
// CHECK-NO-ERRNO-LABEL: @test_exp_underflow
// CHECK-NO-ERRNO: ret double 0.000000e+00
// CHECK-STRICT-LABEL: @test_exp_underflow
// CHECK-STRICT: call double @llvm.experimental.constrained.exp.f64(double -1.000000e+03, metadata !"round.tonearest", metadata !"fpexcept.strict")

// Constant rounding mode is ignored for runtime calls; with -frounding-math, dynamic rounding mode prevents folding.
// FIXME: Once FE_DOWNWARD is supported in APFloat::exp, this should constant-fold to:
// ret double 0x4005BF0A8B145769
double test_exp_rounding_math() {
  #pragma STDC FENV_ROUND FE_DOWNWARD
  return __builtin_exp(1.0);
}
// CHECK-ROUNDING-LABEL: @test_exp_rounding_math
// CHECK-ROUNDING: call double @llvm.experimental.constrained.exp.f64(double 1.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.ignore")

// Overriding -frounding-math with static tonearest rounding pragma allows folding:
float test_rounding_math_override() {
  #pragma STDC FENV_ROUND FE_TONEAREST
  return __builtin_expf(1.0f);
}
// CHECK-ROUNDING-LABEL: @test_rounding_math_override
// CHECK-ROUNDING: ret float 0x3FF5BF0A80000000

// Local dynamic rounding mode pragma prevents inexact folding without -frounding-math:
float test_pragma_fenv_round_dynamic() {
  #pragma STDC FENV_ROUND FE_DYNAMIC
  return __builtin_expf(1.0f);
}
// CHECK-ERRNO-LABEL: @test_pragma_fenv_round_dynamic
// CHECK-ERRNO: call float @llvm.experimental.constrained.exp.f32(float 1.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.ignore")

// Local dynamic rounding mode pragma still allows exact folding:
float test_pragma_fenv_round_dynamic_exact() {
  #pragma STDC FENV_ROUND FE_DYNAMIC
  return __builtin_expf(0.0f);
}
// CHECK-ERRNO-LABEL: @test_pragma_fenv_round_dynamic_exact
// CHECK-ERRNO: ret float 1.000000e+00

// Pragma STDC FENV_ACCESS ON prevents folding of inexact calls:
float test_fenv_access_inexact() {
  #pragma STDC FENV_ACCESS ON
  return __builtin_expf(1.0f);
}
// CHECK-ERRNO-LABEL: @test_fenv_access_inexact
// CHECK-ERRNO: call float @llvm.experimental.constrained.exp.f32(float 1.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict")
// CHECK-NO-ERRNO-LABEL: @test_fenv_access_inexact
// CHECK-NO-ERRNO: call float @llvm.experimental.constrained.exp.f32(float 1.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict")

// Pragma STDC FENV_ACCESS ON still permits folding of exact calls:
float test_fenv_access_exact() {
  #pragma STDC FENV_ACCESS ON
  return __builtin_expf(0.0f);
}
// CHECK-ERRNO-LABEL: @test_fenv_access_exact
// CHECK-ERRNO: ret float 1.000000e+00
// CHECK-NO-ERRNO-LABEL: @test_fenv_access_exact
// CHECK-NO-ERRNO: ret float 1.000000e+00

// Global variable initializers are evaluated at compile time using default FP settings (C17 7.6.1p2):
float global_exp_inexact = __builtin_expf(1.0f);
// CHECK-STRICT: @global_exp_inexact = {{.*}}global float 0x3FF5BF0A80000000
