// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

typedef __attribute__((ext_vector_type(2))) char char2;
typedef __attribute__((ext_vector_type(3))) char char3;

typedef __attribute__((ext_vector_type(2))) int int2;

typedef __attribute__((ext_vector_type(2))) float float2;

// ** Positive tests **

char2 ptest01(char2 c, char s) {
  return c << s;
}

char2 ptest02(char2 c, char2 s) {
  return c << s;
}

char2 ptest03(char2 c, int s) {
  return c << s;
}

char2 ptest04(char2 c, int2 s) {
  return c << s;
}

int2 ptest05(int2 c, char2 s) {
  return c << s;
}

char2 ptest06(char2 c) {
  return c << 1;
}

void ptest07() {
  char3 v = {1,1,1};
  char3 w = {1,2,3};

  v <<= w;
}

// ** Negative tests **

char2 ntest01(char c, char2 s) {
  return c << s; // expected-error {{requested shift is a vector of type '__private char2' (vector of 2 'char' values) but the first operand is not a vector ('__private char')}}
}

char3 ntest02(char3 c, char2 s) {
  return c << s; // expected-error {{vector operands do not have the same number of elements ('char3' (vector of 3 'char' values) and 'char2' (vector of 2 'char' values))}}
}

float2 ntest03(float2 c, char s) {
  return c << s; // expected-error {{used type 'float2' (vector of 2 'float' values) where integer is required}}
}

int2 ntest04(int2 c, float s) {
  return c << s; // expected-error {{used type 'float' where integer is required}}
}
