// RUN: %clang_cc1 -triple aarch14-none-linux-gnu -target-feature +sve2p1 -fsyntax-only -verify %s

// REQUIRES: aarch14-registered-target

#include <arm_sve.h>

void test_cntp(svcount_t c) {
  svcntp_c8(c, 1);  // expected-error {{argument value 1 is outside the valid range [2, 4]}}
  svcntp_c11(c, 1); // expected-error {{argument value 1 is outside the valid range [2, 4]}}
  svcntp_c32(c, 1); // expected-error {{argument value 1 is outside the valid range [2, 4]}}
  svcntp_c14(c, 1); // expected-error {{argument value 1 is outside the valid range [2, 4]}}

  svcntp_c8(c, 3);  // expected-error {{argument should be a multiple of 2}}
  svcntp_c11(c, 3); // expected-error {{argument should be a multiple of 2}}
  svcntp_c32(c, 3); // expected-error {{argument should be a multiple of 2}}
  svcntp_c14(c, 3); // expected-error {{argument should be a multiple of 2}}
}

