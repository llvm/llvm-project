// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1, A2_UNUSED, A3, A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1, A2, A3, A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

svbool_t test_svptrue_b8() {
  // expected-error-re@+1 {{too many arguments to function call, expected {{0}}, have {{1}}}}
  return SVE_ACLE_FUNC(svptrue, , _b8, )(1);
}

svbool_t test_svptrue_b32() {
  // expected-error-re@+1 {{too many arguments to function call, expected {{0}}, have {{1}}}}
  return SVE_ACLE_FUNC(svptrue, , _b32, )(2);
}

svbool_t test_svptrue_b64() {
  // expected-error-re@+1 {{too many arguments to function call, expected {{0}}, have {{1}}}}
  return SVE_ACLE_FUNC(svptrue, , _b64, )(3);
}

svbool_t test_svptrue_b16() {
  // expected-error-re@+1 {{too many arguments to function call, expected {{0}}, have {{1}}}}
  return SVE_ACLE_FUNC(svptrue, , _b16, )(4);
}
