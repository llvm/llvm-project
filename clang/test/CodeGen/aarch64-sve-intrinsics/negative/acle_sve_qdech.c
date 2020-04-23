// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svint16_t test_svqdech_pat_s16(svint16_t op)
{
  // expected-error@+1 {{argument value 0 is outside the valid range [1, 16]}}
  return svqdech_pat_s16(op, SV_VL8, 0);
}

svint16_t test_svqdech_pat_s16_2(svint16_t op)
{
  // expected-error@+1 {{argument value 17 is outside the valid range [1, 16]}}
  return svqdech_pat_s16(op, SV_VL16, 17);
}

svuint16_t test_svqdech_pat_u16(svuint16_t op)
{
  // expected-error@+1 {{argument value 0 is outside the valid range [1, 16]}}
  return svqdech_pat_u16(op, SV_VL32, 0);
}

svuint16_t test_svqdech_pat_u16_2(svuint16_t op)
{
  // expected-error@+1 {{argument value 17 is outside the valid range [1, 16]}}
  return svqdech_pat_u16(op, SV_VL64, 17);
}
