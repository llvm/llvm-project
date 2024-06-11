// RUN: %clang_cc1 %s -triple armv7 -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple aarch64 -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple aarch64 -target-feature -fp-armv8 -target-abi aapcs-soft -fsyntax-only -verify

typedef __attribute__((arm_sve_vector_bits(256))) void nosveflag; // expected-error{{'arm_sve_vector_bits' attribute is not supported on targets missing 'sve'; specify an appropriate -march= or -mcpu=}}
