// RUN: %clang_cc1 %s -triple armv7-apple-darwin -verify -fsyntax-only

void f (void) {
  int Val;
  asm volatile ("lw (r1), %0[val]": "=&b"(Val)); // expected-error {{invalid output constraint '=&b' in asm}}
  return;
}
