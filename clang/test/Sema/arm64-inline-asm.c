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

// A 128-bit value takes two 64-bit registers, and %H names the second.
void test_int128(unsigned __int128 *p, unsigned __int128 x) {
  asm volatile("stp %1, %H1, [%0]" :: "r"(p), "r"(x) : "memory");
}
