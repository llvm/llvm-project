// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks

int j;
void foo() {
  ^ (void) { if (j) return 1; }(); // expected-error {{control may reach end of non-void block}}
}
