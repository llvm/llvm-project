// RUN: %clang_cc1 -triple thumbv7-windows -fsyntax-only -verify %s

// pcs("aapcs-vfp") is the default on Windows on ARM, so it is accepted
// silently. pcs("aapcs") would change the ABI and is still diagnosed.

float __attribute__((pcs("aapcs-vfp"))) with_pcs(float a, float b) { return a + b; }

// The attributed function has the plain function type.
float (*fp)(float, float) = with_pcs;

float __attribute__((pcs("aapcs"))) base_aapcs(float a, float b) { return a + b; } // expected-warning {{'pcs' calling convention is not supported for this target}}
