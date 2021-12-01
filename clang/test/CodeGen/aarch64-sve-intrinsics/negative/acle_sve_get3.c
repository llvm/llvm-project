// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify -verify-ignore-unexpected=note %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svint8_t test_svget3_s8(svint8x3_t tuple)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_s8,,)(tuple, 3);
}

svint16_t test_svget3_s16(svint16x3_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_s16,,)(tuple, -1);
}

svint32_t test_svget3_s32(svint32x3_t tuple)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_s32,,)(tuple, 3);
}

svint64_t test_svget3_s64(svint64x3_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_s64,,)(tuple, -1);
}

svuint8_t test_svget3_u8(svuint8x3_t tuple)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_u8,,)(tuple, 3);
}

svuint16_t test_svget3_u16(svuint16x3_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_u16,,)(tuple, -1);
}

svuint32_t test_svget3_u32(svuint32x3_t tuple)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_u32,,)(tuple, 3);
}

svuint64_t test_svget3_u64(svuint64x3_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_u64,,)(tuple, -1);
}

svfloat16_t test_svget3_f16(svfloat16x3_t tuple)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_f16,,)(tuple, 3);
}

svfloat32_t test_svget3_f32(svfloat32x3_t tuple)
{
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_f32,,)(tuple, -1);
}

svfloat64_t test_svget3_f64(svfloat64x3_t tuple)
{
  // expected-error@+1 {{argument value 3 is outside the valid range [0, 2]}}
  return SVE_ACLE_FUNC(svget3,_f64,,)(tuple, 3);
}

svint8_t test_svget3_s8_var(svint8x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_s8,,)(tuple, imm_index);
}

svint16_t test_svget3_s16_var(svint16x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_s16,,)(tuple, imm_index);
}

svint32_t test_svget3_s32_var(svint32x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_s32,,)(tuple, imm_index);
}

svint64_t test_svget3_s64_var(svint64x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_s64,,)(tuple, imm_index);
}

svuint8_t test_svget3_u8_var(svuint8x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_u8,,)(tuple, imm_index);
}

svuint16_t test_svget3_u16_var(svuint16x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_u16,,)(tuple, imm_index);
}

svuint32_t test_svget3_u32_var(svuint32x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_u32,,)(tuple, imm_index);
}

svuint64_t test_svget3_u64_var(svuint64x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_u64,,)(tuple, imm_index);
}

svfloat16_t test_svget3_f16_var(svfloat16x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_f16,,)(tuple, imm_index);
}

svfloat32_t test_svget3_f32_var(svfloat32x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_f32,,)(tuple, imm_index);
}

svfloat64_t test_svget3_f64_var(svfloat64x3_t tuple, uint64_t imm_index)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  return SVE_ACLE_FUNC(svget3,_f64,,)(tuple, imm_index);
}
