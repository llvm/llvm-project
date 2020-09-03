// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=note %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8x3_t test_svset3_s8(svint8x3_t tuple, svint8_t x)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_s8,,)(tuple, -1, x);
}

svint16x3_t test_svset3_s16(svint16x3_t tuple, svint16_t x)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_s16,,)(tuple, 3, x);
}

svint32x3_t test_svset3_s32(svint32x3_t tuple, svint32_t x)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_s32,,)(tuple, -1, x);
}

svint64x3_t test_svset3_s64(svint64x3_t tuple, svint64_t x)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_s64,,)(tuple, 3, x);
}

svuint8x3_t test_svset3_u8(svuint8x3_t tuple, svuint8_t x)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_u8,,)(tuple, -1, x);
}

svuint16x3_t test_svset3_u16(svuint16x3_t tuple, svuint16_t x)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_u16,,)(tuple, 3, x);
}

svuint32x3_t test_svset3_u32(svuint32x3_t tuple, svuint32_t x)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_u32,,)(tuple, -1, x);
}

svuint64x3_t test_svset3_u64(svuint64x3_t tuple, svuint64_t x)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_u64,,)(tuple, 3, x);
}

svfloat16x3_t test_svset3_f16(svfloat16x3_t tuple, svfloat16_t x)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_f16,,)(tuple, -1, x);
}

svfloat32x3_t test_svset3_f32(svfloat32x3_t tuple, svfloat32_t x)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_f32,,)(tuple, 3, x);
}

svfloat64x3_t test_svset3_f64(svfloat64x3_t tuple, svfloat64_t x)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svset3,_f64,,)(tuple, -1, x);
}

svint8x3_t test_svset3_s8_var(svint8x3_t tuple, uint64_t imm_index, svint8_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_s8,,)(tuple, imm_index, x);
}

svint16x3_t test_svset3_s16_var(svint16x3_t tuple, uint64_t imm_index, svint16_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_s16,,)(tuple, imm_index, x);
}

svint32x3_t test_svset3_s32_var(svint32x3_t tuple, uint64_t imm_index, svint32_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_s32,,)(tuple, imm_index, x);
}

svint64x3_t test_svset3_s64_var(svint64x3_t tuple, uint64_t imm_index, svint64_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_s64,,)(tuple, imm_index, x);
}

svuint8x3_t test_svset3_u8_var(svuint8x3_t tuple, uint64_t imm_index, svuint8_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_u8,,)(tuple, imm_index, x);
}

svuint16x3_t test_svset3_u16_var(svuint16x3_t tuple, uint64_t imm_index, svuint16_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_u16,,)(tuple, imm_index, x);
}

svuint32x3_t test_svset3_u32_var(svuint32x3_t tuple, uint64_t imm_index, svuint32_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_u32,,)(tuple, imm_index, x);
}

svuint64x3_t test_svset3_u64_var(svuint64x3_t tuple, uint64_t imm_index, svuint64_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_u64,,)(tuple, imm_index, x);
}

svfloat16x3_t test_svset3_f16_var(svfloat16x3_t tuple, uint64_t imm_index, svfloat16_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_f16,,)(tuple, imm_index, x);
}

svfloat32x3_t test_svset3_f32_var(svfloat32x3_t tuple, uint64_t imm_index, svfloat32_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_f32,,)(tuple, imm_index, x);
}

svfloat64x3_t test_svset3_f64_var(svfloat64x3_t tuple, uint64_t imm_index, svfloat64_t x)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svset3,_f64,,)(tuple, imm_index, x);
}
