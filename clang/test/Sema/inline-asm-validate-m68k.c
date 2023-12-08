// REQUIRES: m68k-registered-target
// RUN: %clang_cc1 -triple m68k -fsyntax-only -verify %s -DINVALID
// RUN: %clang_cc1 -triple m68k -fsyntax-only -verify %s

#ifdef INVALID

// Invalid constraint usages that can be blocked by frontend

void I() {
  static const int BelowMin = 0;
  static const int AboveMax = 9;
  asm ("" :: "I"(BelowMin)); // expected-error{{value '0' out of range for constraint 'I'}}
  asm ("" :: "I"(AboveMax)); // expected-error{{value '9' out of range for constraint 'I'}}
}

void J() {
  static const int BelowMin = -0x8001;
  static const int AboveMax = 0x8000;
  asm ("" :: "J"(BelowMin)); // expected-error{{value '-32769' out of range for constraint 'J'}}
  asm ("" :: "J"(AboveMax)); // expected-error{{value '32768' out of range for constraint 'J'}}
}

void L() {
  static const int BelowMin = -9;
  static const int AboveMax = 0;
  asm ("" :: "L"(BelowMin)); // expected-error{{value '-9' out of range for constraint 'L'}}
  asm ("" :: "L"(AboveMax)); // expected-error{{value '0' out of range for constraint 'L'}}
}

void N() {
  static const int BelowMin = 23;
  static const int AboveMax = 32;
  asm ("" :: "N"(BelowMin)); // expected-error{{value '23' out of range for constraint 'N'}}
  asm ("" :: "N"(AboveMax)); // expected-error{{value '32' out of range for constraint 'N'}}
}

void O() {
  // Valid only if it's 16
  static const int IncorrectVal = 18;
  asm ("" :: "O"(IncorrectVal)); // expected-error{{value '18' out of range for constraint 'O'}}
}

void P() {
  static const int BelowMin = 7;
  static const int AboveMax = 16;
  asm ("" :: "P"(BelowMin)); // expected-error{{value '7' out of range for constraint 'P'}}
  asm ("" :: "P"(AboveMax)); // expected-error{{value '16' out of range for constraint 'P'}}
}

void C0() {
  // Valid only if it's 0
  static const int IncorrectVal = 1;
  asm ("" :: "C0"(IncorrectVal)); // expected-error{{value '1' out of range for constraint 'C0'}}
}

#else
// Valid constraint usages.
// Note that these constraints can not be fully validated by frontend.
// So we're only testing the availability of their letters here.
// expected-no-diagnostics

void K() {
  asm ("" :: "K"(0x80));
}

void M() {
  asm ("" :: "M"(0x100));
}
void Ci() {
  asm ("" :: "Ci"(0));
}

void Cj() {
  asm ("" :: "Cj"(0x8000));
}

// Register constraints
void a(int x) {
  asm ("" :: "a"(x));
}

void d(int x) {
  asm ("" :: "d"(x));
}

// Memory constraints
void mem() {
  int x;
  asm ("" :: "m"(x));
  asm ("" :: "Q"(x));
  asm ("" :: "U"(x));
}
#endif

