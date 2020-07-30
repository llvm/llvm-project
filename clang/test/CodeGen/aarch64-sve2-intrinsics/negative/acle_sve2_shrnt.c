// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svint8_t test_svshrnt_n_s16(svint8_t op1, svint16_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  return SVE_ACLE_FUNC(svshrnt,_n_s16,,)(op1, op2, 0);
}

svint16_t test_svshrnt_n_s32(svint16_t op1, svint32_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svshrnt,_n_s32,,)(op1, op2, 0);
}

svint32_t test_svshrnt_n_s64(svint32_t op1, svint64_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  return SVE_ACLE_FUNC(svshrnt,_n_s64,,)(op1, op2, 0);
}

svuint8_t test_svshrnt_n_u16(svuint8_t op1, svuint16_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  return SVE_ACLE_FUNC(svshrnt,_n_u16,,)(op1, op2, 0);
}

svuint16_t test_svshrnt_n_u32(svuint16_t op1, svuint32_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svshrnt,_n_u32,,)(op1, op2, 0);
}

svuint32_t test_svshrnt_n_u64(svuint32_t op1, svuint64_t op2)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  return SVE_ACLE_FUNC(svshrnt,_n_u64,,)(op1, op2, 0);
}
