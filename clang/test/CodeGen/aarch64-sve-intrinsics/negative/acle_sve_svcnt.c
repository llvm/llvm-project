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

uint64_t test_svcntb() {
  // expected-error-re@+1 {{too many arguments to function call, expected {{0}}, have {{1}}}}
  return SVE_ACLE_FUNC(svcnt, , b, )(1);
}

uint64_t test_svcnth() {
  // expected-error-re@+1 {{too many arguments to function call, expected {{0}}, have {{1}}}}
  return SVE_ACLE_FUNC(svcnt, , h, )(2);
}

uint64_t test_svcntw() {
  // expected-error-re@+1 {{too many arguments to function call, expected {{0}}, have {{1}}}}
  return SVE_ACLE_FUNC(svcnt, , w, )(3);
}

uint64_t test_svcntd() {
  // expected-error-re@+1 {{too many arguments to function call, expected {{0}}, have {{1}}}}
  return SVE_ACLE_FUNC(svcnt, , d, )(4);
}
