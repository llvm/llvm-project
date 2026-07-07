// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fsyntax-only -verify
// REQUIRES: systemz-registered-target
// Generate an error on the asm label because pragma map was previously seen
void rp(void);
#pragma map(rp, "a187") // expected-note {{previous declaration is here}}

void rp_ext(void);

void rp(void) asm("rp_ext"); // expected-error {{conflicting asm label}}

void rp_orig(void) asm("a187");

void rp_ext(void) {
  rp_orig();
}

