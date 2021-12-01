// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svfloat16_t test_svtmad_f16(svfloat16_t op1, svfloat16_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  return SVE_ACLE_FUNC(svtmad,_f16,,)(op1, op2, -1);
}

svfloat16_t test_svtmad_f16_1(svfloat16_t op1, svfloat16_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  return SVE_ACLE_FUNC(svtmad,_f16,,)(op1, op2, 8);
}

svfloat32_t test_svtmad_f32(svfloat32_t op1, svfloat32_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  return SVE_ACLE_FUNC(svtmad,_f32,,)(op1, op2, -1);
}

svfloat32_t test_svtmad_f32_1(svfloat32_t op1, svfloat32_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  return SVE_ACLE_FUNC(svtmad,_f32,,)(op1, op2, 8);
}

svfloat64_t test_svtmad_f64(svfloat64_t op1, svfloat64_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  return SVE_ACLE_FUNC(svtmad,_f64,,)(op1, op2, -1);
}

svfloat64_t test_svtmad_f64_1(svfloat64_t op1, svfloat64_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  return SVE_ACLE_FUNC(svtmad,_f64,,)(op1, op2, 8);
}
