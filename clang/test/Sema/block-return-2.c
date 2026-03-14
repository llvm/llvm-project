// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks

void foo(void) {
  ^ (void) __attribute__((noreturn)) { }(); // expected-error {{block declared 'noreturn' should not return}}
}
