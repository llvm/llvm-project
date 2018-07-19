// RUN: %clang_cc1 %s -triple armv7-apple-darwin -verify -fsyntax-only

void f (void) {
  int Val;
  asm volatile ("lw (r1), %0[val]": "=&b"(Val)); // expected-error {{invalid output constraint '=&b' in asm}}
  return;
}

void test_64bit_r(void) {
  long long foo = 0, bar = 0;
  asm volatile("INST %0, %1" : "=r"(foo) : "r"(bar));
}
