// RUN: %clang_cc1 -fsyntax-only -verify -std=c89 -Wfunction-effects %s

// Tests for a few cases involving C functions without prototypes.

void hasproto(void) __attribute__((blocking)); // expected-note {{function does not permit inference of 'nonblocking' because it is declared 'blocking'}}

// Has no prototype, inferably safe.
void nb1() {}

// Has no prototype, noreturn.
[[noreturn]]
void aborts();

void nb2(void) __attribute__((nonblocking)) {
  hasproto(); // expected-warning {{function with 'nonblocking' attribute must not call non-'nonblocking' function 'hasproto'}}
  nb1();
  aborts(); // no diagnostic because it's noreturn.
}
