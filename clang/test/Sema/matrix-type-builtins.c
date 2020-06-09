// RUN: %clang_cc1 %s -fenable-matrix -pedantic -verify -triple=x86_64-apple-darwin9

typedef float sx5x10_t __attribute__((matrix_type(5, 10)));
typedef int ix3x2_t __attribute__((matrix_type(3, 2)));
typedef double dx3x3 __attribute__((matrix_type(3, 3)));
typedef unsigned ix3x3 __attribute__((matrix_type(3, 3)));

void transpose(sx5x10_t a, ix3x2_t b, dx3x3 c, int *d, int e) {
  a = __builtin_matrix_transpose(b);
  // expected-error@-1 {{assigning to 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') from incompatible type 'int __attribute__((matrix_type(2, 3)))'}}
  b = __builtin_matrix_transpose(b);
  // expected-error@-1 {{assigning to 'ix3x2_t' (aka 'int __attribute__((matrix_type(3, 2)))') from incompatible type 'int __attribute__((matrix_type(2, 3)))'}}
  __builtin_matrix_transpose(d);
  // expected-error@-1 {{first argument must be a matrix}}
  __builtin_matrix_transpose(e);
  // expected-error@-1 {{first argument must be a matrix}}
  __builtin_matrix_transpose("test");
  // expected-error@-1 {{first argument must be a matrix}}

  ix3x3 m = __builtin_matrix_transpose(c);
  // expected-error@-1 {{initializing 'ix3x3' (aka 'unsigned int __attribute__((matrix_type(3, 3)))') with an expression of incompatible type 'double __attribute__((matrix_type(3, 3)))'}}
}
