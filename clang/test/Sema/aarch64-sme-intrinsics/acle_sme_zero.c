// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -fsyntax-only -verify %s

#include <arm_sme.h>

void test_svzero_args(uint64_t m) {
  svzero_za(0); // expected-error {{too many arguments to function call, expected 0, have 1}}
  svzero_za(m); // expected-error {{too many arguments to function call, expected 0, have 1}}
  svzero_mask_za(m); // expected-error {{argument to 'svzero_mask_za' must be a constant integer}}
}
