// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux-gnu -fsyntax-only --embed-dir=%S/Inputs -std=c23 %s -pedantic -Wpre-c23-compat

#include <limits.h>

#define GET_TYPE_INT(x) _Generic(x, \
  char: 1,\
  unsigned char: 2,\
  signed char: 3,\
  short: 4,\
  unsigned short: 5,\
  int: 6,\
  unsigned int: 7,\
  long: 8,\
  unsigned long: 9,\
  long long: 10,\
  unsigned long long: 11,\
  default: 0xFF\
  )\

enum x {
a = INT_MAX,
b = ULLONG_MAX, // expected-warning {{enumerator values exceeding range of 'int' are incompatible with C standards before C23}}
a_type = GET_TYPE_INT(a),
b_type = GET_TYPE_INT(b)
};

_Static_assert(GET_TYPE_INT(a) == GET_TYPE_INT(b), "ok"); 

extern enum x e_a;
extern __typeof(b) e_a;
extern __typeof(a) e_a;

enum a {
  a0 = 0xFFFFFFFFFFFFFFFFULL // expected-warning {{enumerator values exceeding range of 'int' are incompatible with C standards before C23}}
};

_Bool e () {
  return a0;
}

int f () {
  return a0; // expected-warning {{implicit conversion from 'unsigned long' to 'int' changes value from 18446744073709551615 to -1}}
}

unsigned long g () {
  return a0;
}

unsigned long long h () {
  return a0;
}

enum big_enum {
  big_enum_a = LONG_MAX, // expected-warning {{enumerator values exceeding range of 'int' are incompatible with C standards before C23}}
  big_enum_b = a + 1, // expected-warning {{enumerator values exceeding range of 'int' are incompatible with C standards before C23}}
  big_enum_c = ULLONG_MAX // expected-warning {{enumerator values exceeding range of 'int' are incompatible with C standards before C23}}
};

_Static_assert(GET_TYPE_INT(big_enum_a) == GET_TYPE_INT(big_enum_b), "ok");
