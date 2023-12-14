// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -fsyntax-only -verify -verify-ignore-unexpected=error %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sme_draft_spec_subject_to_change.h>

void test_range_0_0(uint32_t slice, svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svld1_hor_za8,,,)(-1, slice, pg, ptr);
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svst1_ver_za8,,,)(1, slice, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za8,,,)(-1, slice, pg, ptr, 1);
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za8,,,)(1, slice, pg, ptr, 1);

  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svread_hor_za8, _s8, _m,)(svundef_s8(), pg, -1, slice);
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svread_ver_za8, _s8, _m,)(svundef_s8(), pg, 1, slice);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svwrite_hor_za8, _s8, _m,)(-1, slice, pg, svundef_s8());
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svwrite_ver_za8, _s8, _m,)(1, slice, pg, svundef_s8());
}

void test_range_0_1(uint32_t slice, svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svld1_hor_za16,,,)(-1, slice, pg, ptr);
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svst1_ver_za16,,,)(2, slice, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za16,,,)(-1, slice, pg, ptr, 1);
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za16,,,)(2, slice, pg, ptr, 1);

  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svread_hor_za16, _s16, _m,)(svundef_s16(), pg, -1, slice);
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svread_ver_za16, _s16, _m,)(svundef_s16(), pg, 2, slice);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svwrite_hor_za16, _s16, _m,)(-1, slice, pg, svundef_s16());
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svwrite_ver_za16, _s16, _m,)(2, slice, pg, svundef_s16());
}

void test_range_0_3(uint32_t slice, svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svld1_hor_za32,,,)(-1, slice, pg, ptr);
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svst1_ver_za32,,,)(4, slice, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za32,,,)(-1, slice, pg, ptr, 1);
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za32,,,)(4, slice, pg, ptr, 1);

  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svread_hor_za32, _s32, _m,)(svundef_s32(), pg, -1, slice);
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svread_ver_za32, _s32, _m,)(svundef_s32(), pg, 4, slice);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svwrite_hor_za32, _s32, _m,)(-1, slice, pg, svundef_s32());
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svwrite_ver_za32, _s32, _m,)(4, slice, pg, svundef_s32());

  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svaddha_za32, _s32, _m,)(4, pg, pg, svundef_s32());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svaddva_za32, _s32, _m,)(-1, pg, pg, svundef_s32());

  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmopa_za32, _s8, _m,)(4, pg, pg, svundef_s8(), svundef_s8());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svmops_za32, _s8, _m,)(-1, pg, pg, svundef_s8(), svundef_s8());
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svsumopa_za32, _s8, _m,)(4, pg, pg, svundef_s8(), svundef_u8());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svsumops_za32, _s8, _m,)(-1, pg, pg, svundef_s8(), svundef_u8());
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svusmopa_za32, _u8, _m,)(4, pg, pg, svundef_u8(), svundef_s8());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svusmops_za32, _u8, _m,)(-1, pg, pg, svundef_u8(), svundef_s8());
}

void test_range_0_7(uint32_t slice, svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svld1_hor_za64,,,)(-1, slice, pg, ptr);
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svst1_ver_za64,,,)(8, slice, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za64,,,)(-1, slice, pg, ptr, 1);
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za64,,,)(8, slice, pg, ptr, 1);

  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svread_hor_za64, _s64, _m,)(svundef_s64(), pg, -1, slice);
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svread_ver_za64, _s64, _m,)(svundef_s64(), pg, 8, slice);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svwrite_hor_za64, _s64, _m,)(-1, slice, pg, svundef_s64());
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svwrite_ver_za64, _s64, _m,)(8, slice, pg, svundef_s64());

  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svaddha_za64, _s64, _m,)(8, pg, pg, svundef_s64());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svaddva_za64, _s64, _m,)(-1, pg, pg, svundef_s64());

  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmopa_za64, _s16, _m,)(8, pg, pg, svundef_s16(), svundef_s16());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmops_za64, _s16, _m,)(-1, pg, pg, svundef_s16(), svundef_s16());
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svsumopa_za64, _s16, _m,)(8, pg, pg, svundef_s16(), svundef_u16());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svsumops_za64, _s16, _m,)(-1, pg, pg, svundef_s16(), svundef_u16());
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svusmopa_za64, _u16, _m,)(8, pg, pg, svundef_u16(), svundef_s16());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svusmops_za64, _u16, _m,)(-1, pg, pg, svundef_u16(), svundef_s16());

  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmopa_za64, _f64, _m,)(8, pg, pg, svundef_f64(), svundef_f64());
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svmops_za64, _f64, _m,)(-1, pg, pg, svundef_f64(), svundef_f64());
}

void test_range_0_15(uint32_t slice, svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svld1_hor_za128,,,)(-1, slice, pg, ptr);
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svst1_ver_za128,,,)(16, slice, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za128,,,)(-1, slice, pg, ptr, 1);
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za128,,,)(16, slice, pg, ptr, 1);

  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svread_hor_za128, _s8, _m,)(svundef_s8(), pg, -1, slice);
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svread_ver_za128, _s8, _m,)(svundef_s8(), pg, 16, slice);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svwrite_hor_za128, _s8, _m,)(-1, slice, pg, svundef_s8());
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svwrite_ver_za128, _s8, _m,)(16, slice, pg, svundef_s8());
}

void test_range_0_255(svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 256 is outside the valid range [0, 255]}}
  SVE_ACLE_FUNC(svzero_mask_za,,,)(256);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 255]}}
  SVE_ACLE_FUNC(svzero_mask_za,,,)(-1);
}

void test_constant(uint64_t u64, svbool_t pg, void *ptr) {
  SVE_ACLE_FUNC(svld1_hor_za8,,,)(u64, u64, pg, ptr);  // expected-error {{argument to 'svld1_hor_za8' must be a constant integer}}
  SVE_ACLE_FUNC(svst1_hor_za32,,,)(u64, 0, pg, ptr); // expected-error {{argument to 'svst1_hor_za32' must be a constant integer}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za8,,,)(u64, 0, pg, ptr, u64);  // expected-error {{argument to 'svld1_hor_vnum_za8' must be a constant integer}}
  SVE_ACLE_FUNC(svst1_hor_vnum_za32,,,)(u64, 0, pg, ptr, u64); // expected-error {{argument to 'svst1_hor_vnum_za32' must be a constant integer}}

  SVE_ACLE_FUNC(svread_ver_za16, _s16, _m,)(svundef_s16(), pg, u64, 0);  // expected-error-re {{argument to 'svread_ver_za16{{.*}}_m' must be a constant integer}}
  SVE_ACLE_FUNC(svwrite_ver_za64, _s64, _m,)(u64, 0, pg, svundef_s64()); // expected-error-re {{argument to 'svwrite_ver_za64{{.*}}_m' must be a constant integer}}
}
