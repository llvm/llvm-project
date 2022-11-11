// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

void test_range_0_7()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmla_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmla_lane,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlalb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlalb_lane,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlalb_lane,_f32,,)(svundef_f32(), svundef_f16(), svundef_f16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlalt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlalt_lane,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlalt_lane,_f32,,)(svundef_f32(), svundef_f16(), svundef_f16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmls_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmls_lane,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlslb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(),  -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlslb_lane,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlslb_lane,_f32,,)(svundef_f32(), svundef_f16(), svundef_f16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlslt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlslt_lane,_u32,,)(svundef_u32(), svundef_u16(), svundef_u16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmlslt_lane,_f32,,)(svundef_f32(), svundef_f16(), svundef_f16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmul_lane,_s16,,)(svundef_s16(), svundef_s16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmul_lane,_u16,,)(svundef_u16(), svundef_u16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmullb_lane,_s32,,)(svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmullb_lane,_u32,,)(svundef_u16(), svundef_u16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmullt_lane,_s32,,)(svundef_s16(), svundef_s16(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmullt_lane,_u32,,)(svundef_u16(), svundef_u16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqdmlalb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqdmlalt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqdmulh_lane,_s16,,)(svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqdmlslb_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqdmlslt_lane,_s32,,)(svundef_s32(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqdmullb_lane,_s32,,)(svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqdmullt_lane,_s32,,)(svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqrdmlah_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqrdmlsh_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svqrdmulh_lane,_s16,,)(svundef_s16(), svundef_s16(), -1);
}

void test_range_0_3()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svcdot_lane,_s32,,)(svundef_s32(), svundef_s8(), svundef_s8(), -1, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svcmla_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), -1, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svcmla_lane,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16(), -1, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmla_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmla_lane,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmlalb_lane,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmlalb_lane,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmlalt_lane,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmlalt_lane,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmls_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmls_lane,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmlslb_lane,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmlslb_lane,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmlslt_lane,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmlslt_lane,_u64,,)(svundef_u64(), svundef_u32(), svundef_u32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmul_lane,_s32,,)(svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmul_lane,_u32,,)(svundef_u32(), svundef_u32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmullb_lane,_s64,,)(svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmullb_lane,_u64,,)(svundef_u32(), svundef_u32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmullt_lane,_s64,,)(svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmullt_lane,_u64,,)(svundef_u32(), svundef_u32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqdmulh_lane,_s32,,)(svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), -1, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqrdmlah_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqdmlalb_lane,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqdmlalt_lane,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqrdmlsh_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqdmlslb_lane,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqdmlslt_lane,_s64,,)(svundef_s64(), svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqrdmulh_lane,_s32,,)(svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqdmullb_lane,_s64,,)(svundef_s32(), svundef_s32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svqdmullt_lane,_s64,,)(svundef_s32(), svundef_s32(), -1);
}

void test_range_0_1()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svcdot_lane,_s64,,)(svundef_s64(), svundef_s16(), svundef_s16(), -1, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svcmla_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), -1, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svcmla_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svcmla_lane,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32(), -1, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svmla_lane,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svmla_lane,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svmls_lane,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svmls_lane,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svmul_lane,_s64,,)(svundef_s64(), svundef_s64(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svmul_lane,_u64,,)(svundef_u64(), svundef_u64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svqdmulh_lane,_s64,,)(svundef_s64(), svundef_s64(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svqrdcmlah_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 2, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svqrdmlah_lane,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svqrdmlsh_lane,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svqrdmulh_lane,_s64,,)(svundef_s64(), svundef_s64(), 2);
}
