// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

void test_90_270(svbool_t pg)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f16,_x,)(pg, svundef_f16(), svundef_f16(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f16,_z,)(pg, svundef_f16(), svundef_f16(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f16,_m,)(pg, svundef_f16(), svundef_f16(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f32,_x,)(pg, svundef_f32(), svundef_f32(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f32,_z,)(pg, svundef_f32(), svundef_f32(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f32,_m,)(pg, svundef_f32(), svundef_f32(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f64,_x,)(pg, svundef_f64(), svundef_f64(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f64,_z,)(pg, svundef_f64(), svundef_f64(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_f64,_m,)(pg, svundef_f64(), svundef_f64(), 0);
}

void test_0_90_180_270(svbool_t pg)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f16,_x,)(pg, svundef_f16(), svundef_f16(), svundef_f16(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f16,_z,)(pg, svundef_f16(), svundef_f16(), svundef_f16(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f16,_m,)(pg, svundef_f16(), svundef_f16(), svundef_f16(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla_lane,_f16,,)(svundef_f16(), svundef_f16(), svundef_f16(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f32,_x,)(pg, svundef_f32(), svundef_f32(), svundef_f32(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f32,_z,)(pg, svundef_f32(), svundef_f32(), svundef_f32(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f32,_m,)(pg, svundef_f32(), svundef_f32(), svundef_f32(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla_lane,_f32,,)(svundef_f32(), svundef_f32(), svundef_f32(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f64,_x,)(pg, svundef_f64(), svundef_f64(), svundef_f64(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f64,_z,)(pg, svundef_f64(), svundef_f64(), svundef_f64(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_f64,_m,)(pg, svundef_f64(), svundef_f64(), svundef_f64(), 19);
}
