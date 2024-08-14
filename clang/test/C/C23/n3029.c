// RUN: %clang_cc1 -verify -fsyntax-only --embed-dir=%S/Inputs -std=c23 %s -pedantic

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
b = ULLONG_MAX,
a_type = GET_TYPE_INT(a),
b_type = GET_TYPE_INT(b)
};

static_assert(GET_TYPE_INT(a) == GET_TYPE_INT(b));

extern enum x e_a;
extern __typeof(b) e_a;
extern __typeof(a) e_a;

enum a {
  a0 = 0xFFFFFFFFFFFFFFFFULL
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
  big_enum_a = LONG_MAX,
  big_enum_b = a + 1,
  big_enum_c = ULLONG_MAX
};

static_assert(GET_TYPE_INT(big_enum_a) == GET_TYPE_INT(big_enum_b));
