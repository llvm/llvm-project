// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=warning %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=warning %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svint32_t test_svcdot_s32(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcdot,_s32,,)(op1, op2, op3, 19);
}

svint64_t test_svcdot_s64(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcdot,_s64,,)(op1, op2, op3, 19);
}

svint32_t test_svcdot_lane_s32(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svcdot_lane,_s32,,)(op1, op2, op3, -1, 0);
}

svint32_t test_svcdot_lane_s32_1(svint32_t op1, svint8_t op2, svint8_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcdot_lane,_s32,,)(op1, op2, op3, 0, 19);
}

svint64_t test_svcdot_lane_s64(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  return SVE_ACLE_FUNC(svcdot_lane,_s64,,)(op1, op2, op3, -1, 0);
}

svint64_t test_svcdot_lane_s64_1(svint64_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svcdot_lane,_s64,,)(op1, op2, op3, 0, 19);
}
