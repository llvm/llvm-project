// RUN: %clang_cc1 %s -triple armv8.1m.main -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple aarch64 -fsyntax-only -verify=sve-type
// RUN: %clang_cc1 %s -triple aarch64 -target-feature -fp-armv8 -target-abi aapcs-soft -fsyntax-only -verify=sve-type

typedef __attribute__((neon_vector_type(2))) int int32x2_t; // expected-error{{on M-profile architectures 'neon_vector_type' attribute is not supported on targets missing 'mve'; specify an appropriate -march= or -mcpu=}}
typedef __attribute__((neon_polyvector_type(16))) unsigned char poly8x16_t; // expected-error{{on M-profile architectures 'neon_polyvector_type' attribute is not supported on targets missing 'mve'; specify an appropriate -march= or -mcpu=}}
typedef __attribute__((arm_sve_vector_bits(256))) void nosveflag; // expected-error{{'arm_sve_vector_bits' attribute is not supported on targets missing 'sve'; specify an appropriate -march= or -mcpu=}}
                                                                  // sve-type-error@-1{{'arm_sve_vector_bits' attribute is not supported on targets missing 'sve'; specify an appropriate -march= or -mcpu=}}
