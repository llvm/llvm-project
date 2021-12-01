// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svint8_t test_svqcadd_s8(svint8_t op1, svint8_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svqcadd,_s8,,)(op1, op2, 0);
}

svint16_t test_svqcadd_s16(svint16_t op1, svint16_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svqcadd,_s16,,)(op1, op2, 91);
}

svint32_t test_svqcadd_s32(svint32_t op1, svint32_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svqcadd,_s32,,)(op1, op2, 269);
}

svint64_t test_svqcadd_s64(svint64_t op1, svint64_t op2)
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  return SVE_ACLE_FUNC(svqcadd,_s64,,)(op1, op2, 30);
}
