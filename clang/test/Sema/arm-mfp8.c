// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64-arm-none-eabi -target-feature -fp8 %s

// REQUIRES: aarch64-registered-target
#include<arm_neon.h>
__mfp8 test_cast_from_float(unsigned in) {
  return (__mfp8)in; // expected-error {{used type '__mfp8' (aka '__MFloat8_t') where arithmetic or pointer type is required}}
}

unsigned test_cast_to_int(__mfp8 in) {
  return (unsigned)in; // expected-error {{operand of type '__mfp8' (aka '__MFloat8_t') where arithmetic or pointer type is required}}
}
