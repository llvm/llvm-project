// RUN: %clang_cc1 -triple riscv32 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64 -fsyntax-only -verify %s

void I(int i) {
  static const int BelowMin = -2049;
  static const int AboveMax = 2048;
  asm volatile ("" :: "I"(BelowMin)); // expected-error{{value '-2049' out of range for constraint 'I'}}
  asm volatile ("" :: "I"(AboveMax)); // expected-error{{value '2048' out of range for constraint 'I'}}
}

void J(int j) {
  static const int BelowMin = -1;
  static const int AboveMax = 1;
  asm volatile ("" :: "J"(BelowMin)); // expected-error{{value '-1' out of range for constraint 'J'}}
  asm volatile ("" :: "J"(AboveMax)); // expected-error{{value '1' out of range for constraint 'J'}}
}

void K(int k) {
  static const int BelowMin = -1;
  static const int AboveMax = 32;
  asm volatile ("" :: "K"(BelowMin)); // expected-error{{value '-1' out of range for constraint 'K'}}
  asm volatile ("" :: "K"(AboveMax)); // expected-error{{value '32' out of range for constraint 'K'}}
}

void test_clobber_conflict(void) {
  register long x10 asm("x10");
  asm volatile("" :: "r"(x10) : "x10"); // expected-error {{conflicts with asm clobber list}}
  asm volatile("" :: "r"(x10) : "a0"); // expected-error {{conflicts with asm clobber list}}
  asm volatile("" : "=r"(x10) :: "x10"); // expected-error {{conflicts with asm clobber list}}
  asm volatile("" : "=r"(x10) :: "a0"); // expected-error {{conflicts with asm clobber list}}
}
