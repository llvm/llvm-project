// RUN: %clang_cc1 -triple aarch14-none-linux-gnu -target-feature +sve2p1 -fsyntax-only -verify %s

// REQUIRES: aarch14-registered-target

#include <arm_sve.h>
void test_svpext_lane_imm_0_3(svcount_t c) {
  svpext_lane_c8(c, -1);  // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svpext_lane_c16(c, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svpext_lane_c32(c, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svpext_lane_c64(c, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 3]}}

  svpext_lane_c8(c, 4);  // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svpext_lane_c16(c, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svpext_lane_c32(c, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svpext_lane_c64(c, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_svpext_lane_x2_imm_0_1(svcount_t c) {
  svpext_lane_c8_x2(c, -1);  // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  svpext_lane_c16_x2(c, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  svpext_lane_c32_x2(c, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  svpext_lane_c64_x2(c, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 1]}}

  svpext_lane_c8_x2(c, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svpext_lane_c16_x2(c, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svpext_lane_c32_x2(c, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svpext_lane_c64_x2(c, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
}

svcount_t test_svwhile_pn(int64_t op1, int64_t op2) {
  svwhilege_c8(op1, op2, 6);  // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilege_c16(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilege_c32(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilege_c64(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilegt_c8(op1, op2, 6);  // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilegt_c16(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilegt_c32(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilegt_c64(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilehi_c8(op1, op2, 6);  // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilehi_c16(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilehi_c32(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilehi_c64(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilehs_c8(op1, op2, 6);  // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilehs_c16(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilehs_c32(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilehs_c64(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilele_c8(op1, op2, 6);  // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilele_c16(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilele_c32(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilele_c64(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilelo_c8(op1, op2, 6);  // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilelo_c16(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilelo_c32(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilelo_c64(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilels_c8(op1, op2, 6);  // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilels_c16(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilels_c32(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilels_c64(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilelt_c8(op1, op2, 6);  // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilelt_c16(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilelt_c32(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}
  svwhilelt_c64(op1, op2, 6); // expected-error {{argument value 6 is outside the valid range [2, 4]}}

  svwhilege_c8(op1, op2, 3);  // expected-error {{argument should be a multiple of 2}}
  svwhilege_c16(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilege_c32(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilege_c64(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilegt_c8(op1, op2, 3);  // expected-error {{argument should be a multiple of 2}}
  svwhilegt_c16(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilegt_c32(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilegt_c64(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilehi_c8(op1, op2, 3);  // expected-error {{argument should be a multiple of 2}}
  svwhilehi_c16(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilehi_c32(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilehi_c64(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilehs_c8(op1, op2, 3);  // expected-error {{argument should be a multiple of 2}}
  svwhilehs_c16(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilehs_c32(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilehs_c64(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilele_c8(op1, op2, 3);  // expected-error {{argument should be a multiple of 2}}
  svwhilele_c16(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilele_c32(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilele_c64(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilelo_c8(op1, op2, 3);  // expected-error {{argument should be a multiple of 2}}
  svwhilelo_c16(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilelo_c32(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilelo_c64(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilels_c8(op1, op2, 3);  // expected-error {{argument should be a multiple of 2}}
  svwhilels_c16(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilels_c32(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilels_c64(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilelt_c8(op1, op2, 3);  // expected-error {{argument should be a multiple of 2}}
  svwhilelt_c16(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilelt_c32(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
  svwhilelt_c64(op1, op2, 3); // expected-error {{argument should be a multiple of 2}}
}

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

