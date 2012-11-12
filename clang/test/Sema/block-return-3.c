// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks

void foo() {
  ^ int (void) { }(); // expected-error {{control reaches end of non-void block}}
}
