// RUN: %clang_cc1 -triple arm64-apple-macosx10.8.0 -fsyntax-only -verify %s

unsigned t, r, *p;

int foo (void) {
  __asm__ __volatile__( "stxr   %w[_t], %[_r], [%[_p]]" : [_t] "=&r" (t) : [_p] "p" (p), [_r] "r" (r) : "memory"); // expected-warning{{value size does not match register size specified by the constraint and modifier}} expected-note {{use constraint modifier "w"}}
  return 1;
}
