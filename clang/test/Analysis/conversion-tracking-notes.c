// RUN: %clang_analyze_cc1 %s \
// RUN:   -Wno-conversion -Wno-tautological-constant-compare \
// RUN:   -analyzer-checker=core,apiModeling,alpha.core.Conversion \
// RUN:   -analyzer-output=text \
// RUN:   -verify

unsigned char U8;
signed char S8;

void track_assign(void) {
  unsigned long L = 1000; // expected-note {{'L' initialized to 1000}}
  int I = -1;             // expected-note {{'I' initialized to -1}}
  U8 *= L; // expected-warning {{Loss of precision in implicit conversion}}
           // expected-note@-1 {{Loss of precision in implicit conversion}}
  L *= I;  // expected-warning {{Loss of sign in implicit conversion}}
           // expected-note@-1 {{Loss of sign in implicit conversion}}
}

void track_relational(unsigned U, signed S) {
  if (S < -10) { // expected-note    {{Taking true branch}}
                 // expected-note@-1 {{Assuming the condition is true}}
    if (U < S) { // expected-warning {{Loss of sign in implicit conversion}}
                 // expected-note@-1 {{Loss of sign in implicit conversion}}
    }
  }
}
