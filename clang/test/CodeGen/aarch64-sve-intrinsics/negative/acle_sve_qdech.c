// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

int32_t test_svqdech_n_s32(int32_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_n_s32,,)(op, 0);
}

int32_t test_svqdech_n_s32_1(int32_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_n_s32,,)(op, 17);
}

int64_t test_svqdech_n_s64(int64_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_n_s64,,)(op, 0);
}

int64_t test_svqdech_n_s64_1(int64_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_n_s64,,)(op, 17);
}

uint32_t test_svqdech_n_u32(uint32_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_n_u32,,)(op, 0);
}

uint32_t test_svqdech_n_u32_1(uint32_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_n_u32,,)(op, 17);
}

uint64_t test_svqdech_n_u64(uint64_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_n_u64,,)(op, 0);
}

uint64_t test_svqdech_n_u64_1(uint64_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_n_u64,,)(op, 17);
}

int32_t test_svqdech_pat_n_s32(int32_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_n_s32,,)(op, SV_POW2, 0);
}

int32_t test_svqdech_pat_n_s32_1(int32_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_n_s32,,)(op, SV_VL1, 17);
}

int64_t test_svqdech_pat_n_s64(int64_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_n_s64,,)(op, SV_VL2, 0);
}

int64_t test_svqdech_pat_n_s64_1(int64_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_n_s64,,)(op, SV_VL3, 17);
}

uint32_t test_svqdech_pat_n_u32(uint32_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_n_u32,,)(op, SV_VL4, 0);
}

uint32_t test_svqdech_pat_n_u32_1(uint32_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_n_u32,,)(op, SV_VL5, 17);
}

uint64_t test_svqdech_pat_n_u64(uint64_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_n_u64,,)(op, SV_VL6, 0);
}

uint64_t test_svqdech_pat_n_u64_1(uint64_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_n_u64,,)(op, SV_VL7, 17);
}

svint16_t test_svqdech_s16(svint16_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_s16,,)(op, 0);
}

svint16_t test_svqdech_s16_1(svint16_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_s16,,)(op, 17);
}

svuint16_t test_svqdech_u16(svuint16_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_u16,,)(op, 0);
}

svuint16_t test_svqdech_u16_1(svuint16_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech,_u16,,)(op, 17);
}

svint16_t test_svqdech_pat_s16(svint16_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_s16,,)(op, SV_VL8, 0);
}

svint16_t test_svqdech_pat_s16_1(svint16_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_s16,,)(op, SV_VL16, 17);
}

svuint16_t test_svqdech_pat_u16(svuint16_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL32, 0);
}

svuint16_t test_svqdech_pat_u16_1(svuint16_t op)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  return SVE_ACLE_FUNC(svqdech_pat,_u16,,)(op, SV_VL64, 17);
}
