// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svint8_t test_svqrdcmlah_s8(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 19);
}

svint8_t test_svqrdcmlah_s8_1(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 1);
}

svint8_t test_svqrdcmlah_s8_2(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 18);
}

svint8_t test_svqrdcmlah_s8_3(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 91);
}

svint8_t test_svqrdcmlah_s8_4(svint8_t op1, svint8_t op2, svint8_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(op1, op2, op3, 181);
}

svint16_t test_svqrdcmlah_s16(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s16,,)(op1, op2, op3, 19);
}

svint32_t test_svqrdcmlah_s32(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s32,,)(op1, op2, op3, 19);
}

svint64_t test_svqrdcmlah_s64(svint64_t op1, svint64_t op2, svint64_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah,_s64,,)(op1, op2, op3, 19);
}

svint16_t test_svqrdcmlah_lane_s16(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, -1, 0);
}

svint16_t test_svqrdcmlah_lane_s16_1(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, 4, 0);
}

svint16_t test_svqrdcmlah_lane_s16_2(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, -1, 90);
}

svint16_t test_svqrdcmlah_lane_s16_3(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, -1, 180);
}

svint16_t test_svqrdcmlah_lane_s16_4(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, -1, 270);
}

svint16_t test_svqrdcmlah_lane_s16_5(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, 0, 19);
}

svint16_t test_svqrdcmlah_lane_s16_6(svint16_t op1, svint16_t op2, svint16_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(op1, op2, op3, 1, 19);
}

svint32_t test_svqrdcmlah_lane_s32(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s32,,)(op1, op2, op3, 2, 0);
}

svint32_t test_svqrdcmlah_lane_s32_1(svint32_t op1, svint32_t op2, svint32_t op3)
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  return SVE_ACLE_FUNC(svqrdcmlah_lane,_s32,,)(op1, op2, op3, 0, 19);
}
