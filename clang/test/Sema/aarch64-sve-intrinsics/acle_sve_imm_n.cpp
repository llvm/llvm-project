// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

void test_range_1_8(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svasrd,_n_s8,_x,)(pg, svundef_s8(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svasrd,_n_s8,_z,)(pg, svundef_s8(), 9);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 8]}}
  SVE_ACLE_FUNC(svasrd,_n_s8,_m,)(pg, svundef_s8(), 0);
}

void test_range_1_16(svbool_t pg, int32_t i32, uint32_t u32, int64_t i64, uint64_t u64)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svasrd,_n_s16,_x,)(pg, svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svasrd,_n_s16,_z,)(pg, svundef_s16(), 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svasrd,_n_s16,_m,)(pg, svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecb,_n_s32,,)(i32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecb_pat,_n_s32,,)(i32, SV_POW2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecb,_n_u32,,)(u32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecb_pat,_n_u32,,)(u32, SV_VL4, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecb,_n_s64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecb_pat,_n_s64,,)(i64, SV_VL2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecb,_n_u64,,)(u64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecb_pat,_n_u64,,)(u64, SV_VL6, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd,_n_s32,,)(i32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd_pat,_n_s32,,)(i32, SV_POW2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd,_n_u32,,)(u32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd_pat,_n_u32,,)(u32, SV_VL4, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd,_n_s64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd_pat,_n_s64,,)(i64, SV_VL2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd,_n_u64,,)(u64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd_pat,_n_u64,,)(u64, SV_VL6, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech,_n_s32,,)(i32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech_pat,_n_s32,,)(i32, SV_POW2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech,_n_u32,,)(u32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech_pat,_n_u32,,)(u32, SV_VL4, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech,_n_s64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech_pat,_n_s64,,)(i64, SV_VL2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech,_n_u64,,)(u64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech_pat,_n_u64,,)(u64, SV_VL6, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw,_n_s32,,)(i32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw_pat,_n_s32,,)(i32, SV_POW2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw,_n_u32,,)(u32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw_pat,_n_u32,,)(u32, SV_VL4, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw,_n_s64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw_pat,_n_s64,,)(i64, SV_VL2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw,_n_u64,,)(u64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw_pat,_n_u64,,)(u64, SV_VL6, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincb,_n_s32,,)(i32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincb_pat,_n_s32,,)(i32, SV_POW2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincb,_n_u32,,)(u32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincb_pat,_n_u32,,)(u32, SV_VL4, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincb,_n_s64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincb_pat,_n_s64,,)(i64, SV_VL2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincb,_n_u64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincb_pat,_n_u64,,)(u64, SV_VL6, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd,_n_s32,,)(i32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd_pat,_n_s32,,)(i32, SV_POW2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd,_n_u32,,)(u32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd_pat,_n_u32,,)(u32, SV_VL4, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd,_n_s64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd_pat,_n_s64,,)(i64, SV_VL2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd,_n_u64,,)(u64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd_pat,_n_u64,,)(u64, SV_VL6, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch,_n_s32,,)(i32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch_pat,_n_s32,,)(i32, SV_POW2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch,_n_u32,,)(u32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch_pat,_n_u32,,)(u32, SV_VL4, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch,_n_s64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch_pat,_n_s64,,)(i64, SV_VL2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch,_n_u64,,)(u64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch_pat,_n_u64,,)(u64, SV_VL6, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw,_n_s32,,)(i32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw_pat,_n_s32,,)(i32, SV_POW2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw,_n_u32,,)(u32, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw_pat,_n_u32,,)(u32, SV_VL4, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw,_n_s64,,)(i64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw_pat,_n_s64,,)(i64, SV_VL2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw,_n_u64,,)(u64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw_pat,_n_u64,,)(u64, SV_VL6, 0);
}

void test_range_1_32(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svasrd,_n_s32,_x,)(pg, svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svasrd,_n_s32,_z,)(pg, svundef_s32(), 33);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 32]}}
  SVE_ACLE_FUNC(svasrd,_n_s32,_m,)(pg, svundef_s32(), 0);
}

void test_range_1_64(svbool_t pg)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svasrd,_n_s64,_x,)(pg, svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svasrd,_n_s64,_z,)(pg, svundef_s64(), 65);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 64]}}
  SVE_ACLE_FUNC(svasrd,_n_s64,_m,)(pg, svundef_s64(), 0);
}
