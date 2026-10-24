// RUN: %clang_cc1 -Wall -Wno-unused-but-set-variable -Wshift-sign-overflow -ffreestanding -fsyntax-only -verify=expected,cxx17 -std=c++17 %s
// RUN: %clang_cc1 -Wall -Wno-unused-but-set-variable -Wshift-sign-overflow -ffreestanding -fsyntax-only -verify=expected,cxx2a -std=c++2a %s

#include <limits.h>

#define WORD_BIT (sizeof(int) * CHAR_BIT)

enum {
  X = 1 << 0,
  Y = 1 << 1,
  Z = 1 << 2
};

void test() {
  char c;

  c = 0 << 0;
  c = 0 << 1;
  c = 1 << 0;
  c = 1 << -0;
  c = 1 >> -0;
  c = 1 << -1; // expected-warning {{shift count is negative}}
  c = 1 >> -1; // expected-warning {{shift count is negative}}
  c = 1 << (unsigned)-1; // expected-warning {{shift count >= width of type}}
                         // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from -2147483648 to 0}}
  c = 1 >> (unsigned)-1; // expected-warning {{shift count >= width of type}}
  c = 1 << c;
  c <<= 0;
  c >>= 0;
  c <<= 1;
  c >>= 1;
  c <<= -1; // expected-warning {{shift count is negative}}
  c >>= -1; // expected-warning {{shift count is negative}}
  c <<= 999999; // expected-warning {{shift count >= width of type}}
  c >>= 999999; // expected-warning {{shift count >= width of type}}
  // These do not warn: the shift count is checked against the promoted left
  // operand's width, not the LHS's storage width, so the compound forms are
  // diagnosed just like their non-compound equivalents.
  c <<= CHAR_BIT;
  c >>= CHAR_BIT;
  c <<= CHAR_BIT+1;
  c >>= CHAR_BIT+1;
  (void)((long)c << CHAR_BIT);

  int i;
  i = 1 << (WORD_BIT - 2);
  i = 2 << (WORD_BIT - 1); // cxx17-warning {{bits to represent, but 'int' only has}}
  i = 1 << (WORD_BIT - 1); // cxx17-warning {{sets the sign bit of the shift expression}}
  i = -1 << (WORD_BIT - 1); // cxx17-warning {{shifting a negative signed value is undefined}}
  i = -1 << 0; // cxx17-warning {{shifting a negative signed value is undefined}}
  i = 0 << (WORD_BIT - 1);
  i = (char)1 << (WORD_BIT - 2);

  unsigned u;
  u = 1U << (WORD_BIT - 1);
  u = 5U << (WORD_BIT - 1);

  long long int lli;
  lli = INT_MIN << 2; // cxx17-warning {{shifting a negative signed value is undefined}}
  lli = 1LL << (sizeof(long long) * CHAR_BIT - 2);
}

#define a 0
#define ashift 8
enum { b = (a << ashift) };

// Don't warn for negative shifts in code that is unreachable.
void test_pr5544() {
  (void) (((1) > 63 && (1) < 128 ? (((unsigned long long) 1)<<((1)-64)) : (unsigned long long) 0)); // no-warning
}

void test_shift_too_much(char x) {
  if (0)
    (void) (x >> 80); // no-warning
  (void) (x >> 80); // expected-warning {{shift count >= width of type}}
}

// For compound shift assignments, check the shift count against the width of
// the promoted LHS, just like regular shift assignments.
// https://github.com/llvm/llvm-project/issues/234217
unsigned char char_compound(unsigned char x) {
  x >>= 8;
  return x;
}

unsigned char char_ordinary(unsigned char x) {
  x = x >> 8;
  return x;
}

unsigned short short_compound(unsigned short x) {
  x >>= 16;
  return x;
}

unsigned short short_ordinary(unsigned short x) {
  x = x >> 16;
  return x;
}

unsigned char count_32(unsigned char x) {
  x >>= 32; // expected-warning {{shift count >= width of type}}
  x = x >> 32; // expected-warning {{shift count >= width of type}}
  return x;
}

unsigned char count_negative(unsigned char x) {
  x >>= -1; // expected-warning {{shift count is negative}}
  x = x >> -1; // expected-warning {{shift count is negative}}
  return x;
}

typedef unsigned vec16 __attribute__((vector_size(16)));
typedef unsigned vec8 __attribute__((vector_size(8)));

void vect_shift_1(vec16 *x) { *x = *x << 4; }

void vect_shift_2(vec16 *x, vec16 y) { *x = *x << y; }

void vect_shift_3(vec16 *x, vec8 y) {
  *x = *x << y; // expected-error {{vector operands do not have the same number of elements}}
}
