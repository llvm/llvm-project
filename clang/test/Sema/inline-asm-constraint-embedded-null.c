// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

// Regression test for issue173900.

void test_input(void) {
  __asm__("" : : "f\0001"(0.0)); // expected-error {{input constraint contains an embedded null character}}
}

void test_output(void) {
  double x;
  __asm__("" : "=r\0"(x)); // expected-error {{output constraint contains an embedded null character}}
}

void test_clobber(void) {
  __asm__("" : : : "rax\0"); // expected-error {{clobber contains an embedded null character}}
}
