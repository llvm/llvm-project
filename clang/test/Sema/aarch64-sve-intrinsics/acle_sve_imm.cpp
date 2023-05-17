// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify -verify-ignore-unexpected=error %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

void test_range_0_255()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 255]}}
  SVE_ACLE_FUNC(svext,_s8,,)(svundef_s8(), svundef_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 255]}}
  SVE_ACLE_FUNC(svext,_u8,,)(svundef_u8(), svundef_u8(), -1);
}

void test_range_0_127()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 127]}}
  SVE_ACLE_FUNC(svext,_s16,,)(svundef_s16(), svundef_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 127]}}
  SVE_ACLE_FUNC(svext,_u16,,)(svundef_u16(), svundef_u16(), 128);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 127]}}
  SVE_ACLE_FUNC(svext,_f16,,)(svundef_f16(), svundef_f16(), -1);
}

void test_range_0_63()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  SVE_ACLE_FUNC(svext,_s32,,)(svundef_s32(), svundef_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  SVE_ACLE_FUNC(svext,_u32,,)(svundef_u32(), svundef_u32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 63]}}
  SVE_ACLE_FUNC(svext,_f32,,)(svundef_f32(), svundef_f32(), 64);
}

void test_range_0_31()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svext,_s64,,)(svundef_s64(), svundef_s64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svext,_u64,,)(svundef_u64(), svundef_u64(), 32);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 31]}}
  SVE_ACLE_FUNC(svext,_f64,,)(svundef_f64(), svundef_f64(), -1);
}

void test_range_0_1()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_s8,,)(svundef2_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_s16,,)(svundef2_s16(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_s32,,)(svundef2_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_s64,,)(svundef2_s64(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_u8,,)(svundef2_u8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_u16,,)(svundef2_u16(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_u32,,)(svundef2_u32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_u64,,)(svundef2_u64(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_f16,,)(svundef2_f16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_f32,,)(svundef2_f32(), 2);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svget2,_f64,,)(svundef2_f64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_s8,,)(svundef2_s8(), 2, svundef_s8());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_s16,,)(svundef2_s16(), -1, svundef_s16());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_s32,,)(svundef2_s32(), 2, svundef_s32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_s64,,)(svundef2_s64(), -1, svundef_s64());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_u8,,)(svundef2_u8(), 2, svundef_u8());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_u16,,)(svundef2_u16(), -1, svundef_u16());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_u32,,)(svundef2_u32(), 2, svundef_u32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_u64,,)(svundef2_u64(), -1, svundef_u64());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_f16,,)(svundef2_f16(), 2, svundef_f16());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_f32,,)(svundef2_f32(), -1, svundef_f32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svset2,_f64,,)(svundef2_f64(), 2, svundef_f64());
}

void test_range_0_2()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_s8,,)(svundef3_s8(), 3);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_s16,,)(svundef3_s16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_s32,,)(svundef3_s32(), 3);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_s64,,)(svundef3_s64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_u8,,)(svundef3_u8(), 3);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_u16,,)(svundef3_u16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_u32,,)(svundef3_u32(), 3);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_u64,,)(svundef3_u64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_f16,,)(svundef3_f16(), 3);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_f32,,)(svundef3_f32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svget3,_f64,,)(svundef3_f64(), 3);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_s8,,)(svundef3_s8(), -1, svundef_s8());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_s16,,)(svundef3_s16(), -1, svundef_s16());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_s32,,)(svundef3_s32(), 3, svundef_s32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_s64,,)(svundef3_s64(), -1, svundef_s64());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_u8,,)(svundef3_u8(), 3, svundef_u8());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_u16,,)(svundef3_u16(), -1, svundef_u16());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_u32,,)(svundef3_u32(), 3, svundef_u32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_u64,,)(svundef3_u64(), -1, svundef_u64());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_f16,,)(svundef3_f16(), 3, svundef_f16());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_f32,,)(svundef3_f32(), -1, svundef_f32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 2]}}
  SVE_ACLE_FUNC(svset3,_f64,,)(svundef3_f64(), 3, svundef_f64());
}

void test_range_0_3()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_s8,,)(svundef4_s8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_s16,,)(svundef4_s16(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_s32,,)(svundef4_s32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_s64,,)(svundef4_s64(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_u8,,)(svundef4_u8(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_u16,,)(svundef4_u16(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_u32,,)(svundef4_u32(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_u64,,)(svundef4_u64(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_f16,,)(svundef4_f16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_f32,,)(svundef4_f32(), 4);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svget4,_f64,,)(svundef4_f64(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_s8,,)(svundef4_s8(), -1, svundef_s8());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_s32,,)(svundef4_s32(), 4, svundef_s32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_s64,,)(svundef4_s64(), -1, svundef_s64());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_u8,,)(svundef4_u8(), 4, svundef_u8());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_u16,,)(svundef4_u16(), -1, svundef_u16());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_u32,,)(svundef4_u32(), 4, svundef_u32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_u64,,)(svundef4_u64(), -1, svundef_u64());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_f16,,)(svundef4_f16(), 4, svundef_f16());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_f32,,)(svundef4_f32(), -1, svundef_f32());
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svset4,_f64,,)(svundef4_f64(), 4, svundef_f64());
}

void test_range_0_7()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svtmad,_f16,,)(svundef_f16(), svundef_f16(), -1);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svtmad,_f32,,)(svundef_f32(), svundef_f32(), 8);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svtmad,_f64,,)(svundef_f64(), svundef_f64(), -1);
}

void test_range_0_13(svbool_t pg, const void *const_void_ptr)
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 13]}}
  svprfb(pg, const_void_ptr, svprfop(14));
  // expected-error-re@+1 {{must be a constant integer}}
  svprfb_vnum(pg, const_void_ptr, 0, svprfop(-1));
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 13]}}
  svprfd(pg, const_void_ptr, svprfop(14));
  // expected-error-re@+1 {{must be a constant integer}}
  svprfd_vnum(pg, const_void_ptr, 0, svprfop(-1));
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 13]}}
  svprfh(pg, const_void_ptr, svprfop(14));
  // expected-error-re@+1 {{must be a constant integer}}
  svprfh_vnum(pg, const_void_ptr, 0, svprfop(-1));
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [0, 13]}}
  svprfw(pg, const_void_ptr, svprfop(14));
  // expected-error-re@+1 {{must be a constant integer}}
  svprfw_vnum(pg, const_void_ptr, 0, svprfop(-1));
}

void test_range_1_16()
{
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd,_s64,,)(svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd_pat,_s64,,)(svundef_s64(), SV_VL8, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd,_u64,,)(svundef_u64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecd_pat,_u64,,)(svundef_u64(), SV_VL64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech,_s16,,)(svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech,_u16,,)(svundef_u16(), 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech_pat,_s16,,)(svundef_s16(), SV_VL8, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdech_pat,_u16,,)(svundef_u16(), SV_VL64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw,_s32,,)(svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw,_u32,,)(svundef_u32(), 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw_pat,_s32,,)(svundef_s32(), SV_VL8, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqdecw_pat,_u32,,)(svundef_u32(), SV_VL64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd,_s64,,)(svundef_s64(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd,_u64,,)(svundef_u64(), 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd_pat,_s64,,)(svundef_s64(), SV_VL8, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincd_pat,_u64,,)(svundef_u64(), SV_VL64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch,_s16,,)(svundef_s16(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch,_u16,,)(svundef_u16(), 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch_pat,_s16,,)(svundef_s16(), SV_VL8, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqinch_pat,_u16,,)(svundef_u16(), SV_VL64, 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw,_s32,,)(svundef_s32(), 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw,_u32,,)(svundef_u32(), 17);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw_pat,_s32,,)(svundef_s32(), SV_VL8, 0);
  // expected-error-re@+1 {{argument value {{[0-9]+}} is outside the valid range [1, 16]}}
  SVE_ACLE_FUNC(svqincw_pat,_u32,,)(svundef_u32(), SV_VL64, 17);
}

void test_constant(uint64_t u64)
{
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_s8,,)(svundef2_s8(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_s16,,)(svundef2_s16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_s32,,)(svundef2_s32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_s64,,)(svundef2_s64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_u8,,)(svundef2_u8(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_u16,,)(svundef2_u16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_u32,,)(svundef2_u32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_u64,,)(svundef2_u64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_f16,,)(svundef2_f16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_f32,,)(svundef2_f32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget2,_f64,,)(svundef2_f64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_s8,,)(svundef3_s8(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_s16,,)(svundef3_s16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_s32,,)(svundef3_s32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_s64,,)(svundef3_s64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_u8,,)(svundef3_u8(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_u16,,)(svundef3_u16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_u32,,)(svundef3_u32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_u64,,)(svundef3_u64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_f16,,)(svundef3_f16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_f32,,)(svundef3_f32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget3,_f64,,)(svundef3_f64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_s8,,)(svundef4_s8(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_s16,,)(svundef4_s16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_s32,,)(svundef4_s32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_s64,,)(svundef4_s64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_u8,,)(svundef4_u8(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_u16,,)(svundef4_u16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_u32,,)(svundef4_u32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_u64,,)(svundef4_u64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_f16,,)(svundef4_f16(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_f32,,)(svundef4_f32(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svget4,_f64,,)(svundef4_f64(), u64);
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_s8,,)(svundef2_s8(), u64, svundef_s8());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_s16,,)(svundef2_s16(), u64, svundef_s16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_s32,,)(svundef2_s32(), u64, svundef_s32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_s64,,)(svundef2_s64(), u64, svundef_s64());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_u8,,)(svundef2_u8(), u64, svundef_u8());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_u16,,)(svundef2_u16(), u64, svundef_u16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_u32,,)(svundef2_u32(), u64, svundef_u32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_u64,,)(svundef2_u64(), u64, svundef_u64());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_f16,,)(svundef2_f16(), u64, svundef_f16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_f32,,)(svundef2_f32(), u64, svundef_f32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset2,_f64,,)(svundef2_f64(), u64, svundef_f64());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_s8,,)(svundef3_s8(), u64, svundef_s8());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_s16,,)(svundef3_s16(), u64, svundef_s16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_s32,,)(svundef3_s32(), u64, svundef_s32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_s64,,)(svundef3_s64(), u64, svundef_s64());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_u8,,)(svundef3_u8(), u64, svundef_u8());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_u16,,)(svundef3_u16(), u64, svundef_u16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_u32,,)(svundef3_u32(), u64, svundef_u32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_u64,,)(svundef3_u64(), u64, svundef_u64());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_f16,,)(svundef3_f16(), u64, svundef_f16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_f32,,)(svundef3_f32(), u64, svundef_f32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset3,_f64,,)(svundef3_f64(), u64, svundef_f64());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_s8,,)(svundef4_s8(), u64, svundef_s8());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_s16,,)(svundef4_s16(), u64, svundef_s16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_s32,,)(svundef4_s32(), u64, svundef_s32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_s64,,)(svundef4_s64(), u64, svundef_s64());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_u8,,)(svundef4_u8(), u64, svundef_u8());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_u16,,)(svundef4_u16(), u64, svundef_u16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_u32,,)(svundef4_u32(), u64, svundef_u32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_u64,,)(svundef4_u64(), u64, svundef_u64());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_f16,,)(svundef4_f16(), u64, svundef_f16());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_f32,,)(svundef4_f32(), u64, svundef_f32());
  // expected-error-re@+1 {{argument to '{{.*}}' must be a constant integer}}
  SVE_ACLE_FUNC(svset4,_f64,,)(svundef4_f64(), u64, svundef_f64());
}

void test_num_args()
{
  // expected-note@* {{requires 0 arguments, but 1 was provided}}
  SVE_ACLE_FUNC(svcnt,, b,)(1);
  // expected-note@* {{requires 0 arguments, but 1 was provided}}
  SVE_ACLE_FUNC(svcnt,, h,)(2);
  // expected-note@* {{requires 0 arguments, but 1 was provided}}
  SVE_ACLE_FUNC(svcnt,, w,)(3);
  // expected-note@* {{requires 0 arguments, but 1 was provided}}
  SVE_ACLE_FUNC(svcnt,, d,)(4);
  // expected-note@* {{requires 0 arguments, but 1 was provided}}
  SVE_ACLE_FUNC(svptrue,, _b8,)(1);
  // expected-note@* {{requires 0 arguments, but 1 was provided}}
  SVE_ACLE_FUNC(svptrue,, _b16,)(2);
  // expected-note@* {{requires 0 arguments, but 1 was provided}}
  SVE_ACLE_FUNC(svptrue,, _b32,)(3);
  // expected-note@* {{requires 0 arguments, but 1 was provided}}
  SVE_ACLE_FUNC(svptrue,, _b64,)(4); 
}

void test_enum(svbool_t pg, const void *const_void_ptr, uint64_t u64)
{
  // Test type checks on svpattern and svprfop enums.

  // expected-note@* {{no known conversion from 'svpattern' to 'enum svprfop'}}
  SVE_ACLE_FUNC(svprfb,,,)(pg, const_void_ptr, SV_VL1);
  // expected-note@* + {{no known conversion from 'svprfop' to 'enum svpattern'}}
  SVE_ACLE_FUNC(svqdecb_pat,_n_u64,,)(u64, SV_PLDL1KEEP, 1);
}
