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

svfloat16_t test_svcmla_f16_z(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 19);
}

svfloat16_t test_svcmla_f16_z_1(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 1);
}

svfloat16_t test_svcmla_f16_z_2(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 18);
}

svfloat16_t test_svcmla_f16_z_3(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 91);
}

svfloat16_t test_svcmla_f16_z_4(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, op1, op2, op3, 181);
}

svfloat32_t test_svcmla_f32_z(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f32,_z,)(pg, op1, op2, op3, 19);
}

svfloat64_t test_svcmla_f64_z(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f64,_z,)(pg, op1, op2, op3, 19);
}

svfloat16_t test_svcmla_f16_m(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f16,_m,)(pg, op1, op2, op3, 19);
}

svfloat32_t test_svcmla_f32_m(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f32,_m,)(pg, op1, op2, op3, 19);
}

svfloat64_t test_svcmla_f64_m(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f64,_m,)(pg, op1, op2, op3, 19);
}

svfloat16_t test_svcmla_f16_x(svbool_t pg, svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f16,_x,)(pg, op1, op2, op3, 19);
}

svfloat32_t test_svcmla_f32_x(svbool_t pg, svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f32,_x,)(pg, op1, op2, op3, 19);
}

svfloat64_t test_svcmla_f64_x(svbool_t pg, svfloat64_t op1, svfloat64_t op2, svfloat64_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla,_f64,_x,)(pg, op1, op2, op3, 19);
}

svfloat16_t test_svcmla_lane_f16(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, -1, 0);
}

svfloat16_t test_svcmla_lane_f16_1(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, -1, 90);
}

svfloat16_t test_svcmla_lane_f16_2(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, -1, 180);
}

svfloat16_t test_svcmla_lane_f16_3(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, -1, 270);
}

svfloat16_t test_svcmla_lane_f16_4(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, 0, 19);
}

svfloat16_t test_svcmla_lane_f16_5(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, 1, 19);
}

svfloat16_t test_svcmla_lane_f16_6(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, 2, 19);
}

svfloat16_t test_svcmla_lane_f16_7(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, 3, 19);
}

svfloat16_t test_svcmla_lane_f16_8(svfloat16_t op1, svfloat16_t op2, svfloat16_t op3)
{
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svcmla_lane,_f16,,)(op1, op2, op3, 4, 0);
}

svfloat32_t test_svcmla_lane_f32(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  return SVE_ACLE_FUNC(svcmla_lane,_f32,,)(op1, op2, op3, -1, 0);
}

svfloat32_t test_svcmla_lane_f32_1(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla_lane,_f32,,)(op1, op2, op3, 0, 19);
}

svfloat32_t test_svcmla_lane_f32_2(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcmla_lane,_f32,,)(op1, op2, op3, 1, 19);
}

svfloat32_t test_svcmla_lane_f32_3(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return SVE_ACLE_FUNC(svcmla_lane,_f32,,)(op1, op2, op3, 2, 0);
}

svfloat32_t test_svcmla_lane_f32_4(svfloat32_t op1, svfloat32_t op2, svfloat32_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  return SVE_ACLE_FUNC(svcmla_lane,_f32,,)(op1, op2, op3, 3, 180);
}
