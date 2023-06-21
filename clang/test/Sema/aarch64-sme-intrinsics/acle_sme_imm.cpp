// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -DDISABLE_SME_ATTRIBUTES -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -fsyntax-only -verify -verify-ignore-unexpected=error %s
// RUN: %clang_cc1 -DDISABLE_SME_ATTRIBUTES -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -fsyntax-only -verify -verify-ignore-unexpected=error %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sme_draft_spec_subject_to_change.h>

#ifdef DISABLE_SME_ATTRIBUTES
#define ARM_STREAMING_ATTR
#else
#define ARM_STREAMING_ATTR __attribute__((arm_streaming))
#endif

ARM_STREAMING_ATTR
void test_range_0_0(svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svld1_hor_za8,,,)(-1, -1, 0, pg, ptr);
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svst1_ver_za8,,,)(1, -1, 15, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svld1_hor_za128,,,)(0, -1, -1, pg, ptr);
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svst1_ver_za128,,,)(15, -1, 1, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za8,,,)(-1, -1, 0, pg, ptr, 1);
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za8,,,)(1, -1, 15, pg, ptr, 1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za128,,,)(0, -1, -1, pg, ptr, 1);
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za128,,,)(15, -1, 1, pg, ptr, 1);
}

ARM_STREAMING_ATTR
void test_range_0_1(svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svld1_hor_za16,,,)(-1, -1, 0, pg, ptr);
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svst1_ver_za16,,,)(2, -1, 7, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svld1_hor_za64,,,)(0, -1, -1, pg, ptr);
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svst1_ver_za64,,,)(7, -1, 2, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za16,,,)(-1, -1, 0, pg, ptr, 1);
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za16,,,)(2, -1, 7, pg, ptr, 1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za64,,,)(0, -1, -1, pg, ptr, 1);
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za64,,,)(7, -1, 2, pg, ptr, 1);
}

ARM_STREAMING_ATTR
void test_range_0_3(svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svld1_hor_za32,,,)(-1, -1, 0, pg, ptr);
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svst1_ver_za32,,,)(4, -1, 3, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svld1_hor_za32,,,)(0, -1, -1, pg, ptr);
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svst1_ver_za32,,,)(3, -1, 4, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za32,,,)(-1, -1, 0, pg, ptr, 1);
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za32,,,)(4, -1, 3, pg, ptr, 1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za32,,,)(0, -1, -1, pg, ptr, 1);
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za32,,,)(3, -1, 4, pg, ptr, 1);
}

ARM_STREAMING_ATTR
void test_range_0_7(svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svld1_hor_za64,,,)(-1, -1, 0, pg, ptr);
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svst1_ver_za64,,,)(8, -1, 1, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svld1_hor_za16,,,)(0, -1, -1, pg, ptr);
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svst1_ver_za16,,,)(1, -1, 8, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za64,,,)(-1, -1, 0, pg, ptr, 1);
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za64,,,)(8, -1, 1, pg, ptr, 1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za16,,,)(0, -1, -1, pg, ptr, 1);
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za16,,,)(1, -1, 8, pg, ptr, 1);
}

ARM_STREAMING_ATTR
void test_range_0_15(svbool_t pg, void *ptr) {
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svld1_hor_za128,,,)(-1, -1, 0, pg, ptr);
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svst1_ver_za128,,,)(16, -1, 0, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svld1_hor_za8,,,)(0, -1, -1, pg, ptr);
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svst1_ver_za8,,,)(0, -1, 16, pg, ptr);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za128,,,)(-1, -1, 0, pg, ptr, 1);
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za128,,,)(16, -1, 0, pg, ptr, 1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za8,,,)(0, -1, -1, pg, ptr, 1);
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za8,,,)(0, -1, 16, pg, ptr, 1);
}

ARM_STREAMING_ATTR
void test_constant(uint64_t u64, svbool_t pg, void *ptr) {
  SVE_ACLE_FUNC(svld1_hor_za8,,,)(u64, u64, 0, pg, ptr);  // expected-error {{argument to 'svld1_hor_za8' must be a constant integer}}
  SVE_ACLE_FUNC(svld1_ver_za16,,,)(0, u64, u64, pg, ptr); // expected-error {{argument to 'svld1_ver_za16' must be a constant integer}}
  SVE_ACLE_FUNC(svst1_hor_za32,,,)(u64, u64, 0, pg, ptr); // expected-error {{argument to 'svst1_hor_za32' must be a constant integer}}
  SVE_ACLE_FUNC(svst1_ver_za64,,,)(0, u64, u64, pg, ptr); // expected-error {{argument to 'svst1_ver_za64' must be a constant integer}}
  SVE_ACLE_FUNC(svld1_hor_vnum_za8,,,)(u64, u64, 0, pg, ptr, u64);  // expected-error {{argument to 'svld1_hor_vnum_za8' must be a constant integer}}
  SVE_ACLE_FUNC(svld1_ver_vnum_za16,,,)(0, u64, u64, pg, ptr, u64); // expected-error {{argument to 'svld1_ver_vnum_za16' must be a constant integer}}
  SVE_ACLE_FUNC(svst1_hor_vnum_za32,,,)(u64, u64, 0, pg, ptr, u64); // expected-error {{argument to 'svst1_hor_vnum_za32' must be a constant integer}}
  SVE_ACLE_FUNC(svst1_ver_vnum_za64,,,)(0, u64, u64, pg, ptr, u64); // expected-error {{argument to 'svst1_ver_vnum_za64' must be a constant integer}}
}
