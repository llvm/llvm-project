// RUN: %clang_cc1 -verify -ffp-exception-behavior=strict -Wno-unknown-pragmas %s

// REQUIRES: x86-registered-target

#pragma STDC FENV_ROUND FE_DYNAMIC

// nextUp(1.F) == 0x1.000002p0F

float F1 = 0x1.000000p0F + 0x0.000002p0F;
float F2 = 0x1.000000p0F + 0x0.000001p0F; // expected-error{{initializer element is not a compile-time constant}}
