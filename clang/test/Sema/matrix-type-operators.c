// RUN: %clang_cc1 %s -fenable-matrix -pedantic -verify -triple=x86_64-apple-darwin9

typedef float sx5x10_t __attribute__((matrix_type(5, 10)));
typedef float sx10x5_t __attribute__((matrix_type(10, 5)));
typedef float sx10x10_t __attribute__((matrix_type(10, 10)));

void add(sx10x10_t a, sx5x10_t b, sx10x5_t c) {
  a = b + c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t' (aka 'float __attribute__((matrix_type(10, 5)))'))}}

  a = b + b; // expected-error {{assigning to 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') from incompatible type 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))')}}

  a = 10 + b;
  // expected-error@-1 {{assigning to 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') from incompatible type 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))')}}

  a = b + &c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*'))}}
  // expected-error@-2 {{casting 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*') to incompatible type 'float'}}
}

void sub(sx10x10_t a, sx5x10_t b, sx10x5_t c) {
  a = b - c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t' (aka 'float __attribute__((matrix_type(10, 5)))'))}}

  a = b - b; // expected-error {{assigning to 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') from incompatible type 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))')}}

  a = 10 - b;
  // expected-error@-1 {{assigning to 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') from incompatible type 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))')}}

  a = b - &c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*'))}}
  // expected-error@-2 {{casting 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*') to incompatible type 'float'}}
}
