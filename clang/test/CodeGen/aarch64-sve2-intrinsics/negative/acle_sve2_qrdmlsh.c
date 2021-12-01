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

svint16_t test_svqrdmlsh_lane_s16(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  return SVE_ACLE_FUNC(svqrdmlsh_lane,_s16,,)(op1, op2, op3, -1);
}

svint32_t test_svqrdmlsh_lane_s32(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svqrdmlsh_lane,_s32,,)(op1, op2, op3, 4);
}

svint64_t test_svqrdmlsh_lane_s64(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  return SVE_ACLE_FUNC(svqrdmlsh_lane,_s64,,)(op1, op2, op3, 2);
}
