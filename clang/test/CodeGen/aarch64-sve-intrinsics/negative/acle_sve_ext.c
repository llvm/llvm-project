// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svint8_t test_svext_s8(svint8_t op1, svint8_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 255]}}
  return SVE_ACLE_FUNC(svext,_s8,,)(op1, op2, -1);
}

svint8_t test_svext_s8_1(svint8_t op1, svint8_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 255]}}
  return SVE_ACLE_FUNC(svext,_s8,,)(op1, op2, 256);
}

svint16_t test_svext_s16(svint16_t op1, svint16_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 127]}}
  return SVE_ACLE_FUNC(svext,_s16,,)(op1, op2, -1);
}

svint16_t test_svext_s16_1(svint16_t op1, svint16_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 127]}}
  return SVE_ACLE_FUNC(svext,_s16,,)(op1, op2, 128);
}

svint32_t test_svext_s32(svint32_t op1, svint32_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  return SVE_ACLE_FUNC(svext,_s32,,)(op1, op2, -1);
}

svint32_t test_svext_s32_1(svint32_t op1, svint32_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  return SVE_ACLE_FUNC(svext,_s32,,)(op1, op2, 64);
}

svint64_t test_svext_s64(svint64_t op1, svint64_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  return SVE_ACLE_FUNC(svext,_s64,,)(op1, op2, -1);
}

svint64_t test_svext_s64_1(svint64_t op1, svint64_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  return SVE_ACLE_FUNC(svext,_s64,,)(op1, op2, 32);
}

svuint8_t test_svext_u8(svuint8_t op1, svuint8_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 255]}}
  return SVE_ACLE_FUNC(svext,_u8,,)(op1, op2, -1);
}

svuint16_t test_svext_u16(svuint16_t op1, svuint16_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 127]}}
  return SVE_ACLE_FUNC(svext,_u16,,)(op1, op2, 128);
}

svuint32_t test_svext_u32(svuint32_t op1, svuint32_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  return SVE_ACLE_FUNC(svext,_u32,,)(op1, op2, -1);
}

svuint64_t test_svext_u64(svuint64_t op1, svuint64_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  return SVE_ACLE_FUNC(svext,_u64,,)(op1, op2, 32);
}

svfloat16_t test_svext_f16(svfloat16_t op1, svfloat16_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 127]}}
  return SVE_ACLE_FUNC(svext,_f16,,)(op1, op2, -1);
}

svfloat32_t test_svext_f32(svfloat32_t op1, svfloat32_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  return SVE_ACLE_FUNC(svext,_f32,,)(op1, op2, 64);
}

svfloat64_t test_svext_f64(svfloat64_t op1, svfloat64_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  return SVE_ACLE_FUNC(svext,_f64,,)(op1, op2, -1);
}
