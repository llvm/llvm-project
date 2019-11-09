// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks

int j;
void foo() {
  ^ (void) { if (j) return 1; }(); // expected-error {{non-void block does not return a value in all control paths}}
}
