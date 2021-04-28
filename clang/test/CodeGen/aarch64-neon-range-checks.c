// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +sha3 -target-feature +sm4 -verify %s

#include <arm_neon.h>

void test_range_check_vsm3tt1a(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  vsm3tt1aq_u32(a, b, c, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vsm3tt1aq_u32(a, b, c, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vsm3tt1aq_u32(a, b, c, 3);
  vsm3tt1aq_u32(a, b, c, 0);
}

void test_range_check_vsm3tt1b(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  vsm3tt1bq_u32(a, b, c, 4);// expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vsm3tt1bq_u32(a, b, c, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vsm3tt1bq_u32(a, b, c, 3);
  vsm3tt1bq_u32(a, b, c, 0);
}

void test_range_check_vsm3tt2a(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  vsm3tt2aq_u32(a, b, c, 4);// expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vsm3tt2aq_u32(a, b, c, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vsm3tt2aq_u32(a, b, c, 3);
  vsm3tt2aq_u32(a, b, c, 0);
}

void test_range_check_vsm3tt2b(uint32x4_t a, uint32x4_t b, uint32x4_t c) {
  vsm3tt2bq_u32(a, b, c, 4);// expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vsm3tt2bq_u32(a, b, c, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vsm3tt2bq_u32(a, b, c, 3);
  vsm3tt2bq_u32(a, b, c, 0);
}

void test_range_check_xar(uint64x2_t a, uint64x2_t b) {
  vxarq_u64(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 63]}}
  vxarq_u64(a, b, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  vxarq_u64(a, b, 0);
  vxarq_u64(a, b, 63);
}

