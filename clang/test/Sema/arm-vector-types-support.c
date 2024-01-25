// RUN: %clang_cc1 %s -triple armv7 -fsyntax-only -verify

typedef __attribute__((neon_vector_type(2))) int int32x2_t; // expected-error{{'neon_vector_type' attribute is not supported on targets missing 'neon', 'mve', 'sve' or 'sme'; specify an appropriate -march= or -mcpu=}}
typedef __attribute__((neon_polyvector_type(16))) short poly8x16_t; // expected-error{{'neon_polyvector_type' attribute is not supported on targets missing 'neon' or 'mve'; specify an appropriate -march= or -mcpu=}}
typedef __attribute__((arm_sve_vector_bits(256))) void nosveflag; // expected-error{{'arm_sve_vector_bits' attribute is not supported on targets missing 'sve'; specify an appropriate -march= or -mcpu=}}
