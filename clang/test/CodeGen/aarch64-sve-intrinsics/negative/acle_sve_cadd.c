// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svfloat16_t test_svcadd_f16_m(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f16,_m,)(pg, op1, op2, 0);
}

svfloat32_t test_svcadd_f32_m(svbool_t pg, svfloat32_t op1, svfloat32_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f32,_m,)(pg, op1, op2, 0);
}

svfloat64_t test_svcadd_f64_m(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f64,_m,)(pg, op1, op2, 0);
}
