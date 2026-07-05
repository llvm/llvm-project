// RUN: %clang_cc1 -triple arm64-apple-ios7.1 -fsyntax-only -verify %s

void foo(void) {
  asm volatile("USE(%0)" :: "z"(0LL));
  asm volatile("USE(%x0)" :: "z"(0LL));
  asm volatile("USE(%w0)" :: "z"(0));

  asm volatile("USE(%0)" :: "z"(0)); // expected-warning {{value size does not match register size specified by the constraint and modifier}} expected-note {{use constraint modifier "w"}}
}

void test_clobber_conflict(void) {
  register long x asm("x1");
  asm volatile("nop" :: "r"(x) : "%x1"); // expected-error {{conflicts with asm clobber list}}
  asm volatile("nop" : "=r"(x) :: "%x1"); // expected-error {{conflicts with asm clobber list}}
}
