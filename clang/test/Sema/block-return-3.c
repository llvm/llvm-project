// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks

void foo(void) {
  ^ int (void) { }(); // expected-error {{non-void block does not return a value}}
}
