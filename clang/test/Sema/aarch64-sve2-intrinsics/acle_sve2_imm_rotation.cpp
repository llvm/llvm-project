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

void test_90_270()
{
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_s8,,)(svundef_s8(), svundef_s8(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_u8,,)(svundef_u8(), svundef_u8(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_s16,,)(svundef_s16(), svundef_s16(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_u16,,)(svundef_u16(), svundef_u16(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_s32,,)(svundef_s32(), svundef_s32(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_u32,,)(svundef_u32(), svundef_u32(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_s64,,)(svundef_s64(), svundef_s64(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svcadd,_u64,,)(svundef_u64(), svundef_u64(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svqcadd,_s8,,)(svundef_s8(), svundef_s8(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svqcadd,_s16,,)(svundef_s16(), svundef_s16(), 180);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svqcadd,_s32,,)(svundef_s32(), svundef_s32(), 0);
  // expected-error@+1 {{argument should be the value 90 or 270}}
  SVE_ACLE_FUNC(svqcadd,_s64,,)(svundef_s64(), svundef_s64(), 180);
}

void test_0_90_180_270()
{
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcdot,_s32,,)(svundef_s32(), svundef_s8(), svundef_s8(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcdot_lane,_s32,,)(svundef_s32(), svundef_s8(), svundef_s8(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcdot,_s64,,)(svundef_s64(), svundef_s16(), svundef_s16(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcdot_lane,_s64,,)(svundef_s64(), svundef_s16(), svundef_s16(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_u8,,)(svundef_u8(), svundef_u8(), svundef_u8(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla_lane,_u16,,)(svundef_u16(), svundef_u16(), svundef_u16(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 1, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla_lane,_u32,,)(svundef_u32(), svundef_u32(), svundef_u32(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svcmla,_u64,,)(svundef_u64(), svundef_u64(), svundef_u64(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svqrdcmlah,_s8,,)(svundef_s8(), svundef_s8(), svundef_s8(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svqrdcmlah,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svqrdcmlah_lane,_s16,,)(svundef_s16(), svundef_s16(), svundef_s16(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svqrdcmlah,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svqrdcmlah_lane,_s32,,)(svundef_s32(), svundef_s32(), svundef_s32(), 0, 19);
  // expected-error@+1 {{argument should be the value 0, 90, 180 or 270}}
  SVE_ACLE_FUNC(svqrdcmlah,_s64,,)(svundef_s64(), svundef_s64(), svundef_s64(), 19);
}
