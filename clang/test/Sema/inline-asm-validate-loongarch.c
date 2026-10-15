// RUN: %clang_cc1 -triple loongarch32 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple loongarch64 -fsyntax-only -verify %s

void test_clobber_conflict(void) {
  register long r4 asm("r4");
  asm volatile("" :: "r"(r4) : "$r4"); // expected-error {{conflicts with asm clobber list}}
  asm volatile("" :: "r"(r4) : "$a0"); // expected-error {{conflicts with asm clobber list}}
  asm volatile("" : "=r"(r4) :: "$r4"); // expected-error {{conflicts with asm clobber list}}
  asm volatile("" : "=r"(r4) :: "$a0"); // expected-error {{conflicts with asm clobber list}}
}
