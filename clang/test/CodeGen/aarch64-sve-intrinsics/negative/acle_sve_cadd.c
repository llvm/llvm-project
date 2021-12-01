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

svfloat16_t test_svcadd_f16_z(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f16,_z,)(pg, op1, op2, 0);
}

svfloat16_t test_svcadd_f16_z_1(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f16,_z,)(pg, op1, op2, 272);
}

svfloat16_t test_svcadd_f16_z_2(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f16,_z,)(pg, op1, op2, 91);
}

svfloat16_t test_svcadd_f16_z_3(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f16,_z,)(pg, op1, op2, 180);
}

svfloat16_t test_svcadd_f16_z_4(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f16,_z,)(pg, op1, op2, 271);
}

svfloat32_t test_svcadd_f32_z(svbool_t pg, svfloat32_t op1, svfloat32_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f32,_z,)(pg, op1, op2, 0);
}

svfloat64_t test_svcadd_f64_z(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f64,_z,)(pg, op1, op2, 0);
}

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

svfloat16_t test_svcadd_f16_x(svbool_t pg, svfloat16_t op1, svfloat16_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f16,_x,)(pg, op1, op2, 0);
}

svfloat32_t test_svcadd_f32_x(svbool_t pg, svfloat32_t op1, svfloat32_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f32,_x,)(pg, op1, op2, 0);
}

svfloat64_t test_svcadd_f64_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svcadd,_f64,_x,)(pg, op1, op2, 0);
}
