// RUN: %clang_cc1 %s -fenable-matrix -pedantic -verify -triple=x86_64-apple-darwin9

typedef float sx5x10_t __attribute__((matrix_type(5, 10)));
typedef float sx10x5_t __attribute__((matrix_type(10, 5)));
typedef float sx10x10_t __attribute__((matrix_type(10, 10)));

void add(sx10x10_t a, sx5x10_t b, sx10x5_t c) {
  a = b + c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t' (aka 'float __attribute__((matrix_type(10, 5)))'))}}

  b += c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t' (aka 'float __attribute__((matrix_type(10, 5)))'))}}

  a = b + b; // expected-error {{assigning to 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') from incompatible type 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))')}}

  a = 10 + b;
  // expected-error@-1 {{assigning to 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') from incompatible type 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))')}}

  a = b + &c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*'))}}
  // expected-error@-2 {{casting 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*') to incompatible type 'float'}}

  b += &c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*'))}}
  // expected-error@-2 {{casting 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*') to incompatible type 'float'}}
}

void sub(sx10x10_t a, sx5x10_t b, sx10x5_t c) {
  a = b - c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t' (aka 'float __attribute__((matrix_type(10, 5)))'))}}

  b -= c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t' (aka 'float __attribute__((matrix_type(10, 5)))'))}}

  a = b - b; // expected-error {{assigning to 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') from incompatible type 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))')}}

  a = 10 - b;
  // expected-error@-1 {{assigning to 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') from incompatible type 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))')}}

  a = b - &c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*'))}}
  // expected-error@-2 {{casting 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*') to incompatible type 'float'}}

  b -= &c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*'))}}
  // expected-error@-2 {{casting 'sx10x5_t *' (aka 'float  __attribute__((matrix_type(10, 5)))*') to incompatible type 'float'}}
}

typedef int ix10x5_t __attribute__((matrix_type(10, 5)));
typedef int ix10x10_t __attribute__((matrix_type(10, 10)));

void matrix_matrix_multiply(sx10x10_t a, sx5x10_t b, ix10x5_t c, ix10x10_t d, float sf, char *p) {
  // Check dimension mismatches.
  a = a * b;
  // expected-error@-1 {{invalid operands to binary expression ('sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') and 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))'))}}
  a *= b;
  // expected-error@-1 {{invalid operands to binary expression ('sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') and 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))'))}}
  b = a * a;
  // expected-error@-1 {{assigning to 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') from incompatible type 'float __attribute__((matrix_type(10, 10)))'}}

  // Check element type mismatches.
  a = b * c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'ix10x5_t' (aka 'int __attribute__((matrix_type(10, 5)))'))}}
  b *= c;
  // expected-error@-1 {{invalid operands to binary expression ('sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') and 'ix10x5_t' (aka 'int __attribute__((matrix_type(10, 5)))'))}}
  d = a * a;
  // expected-error@-1 {{assigning to 'ix10x10_t' (aka 'int __attribute__((matrix_type(10, 10)))') from incompatible type 'float __attribute__((matrix_type(10, 10)))'}}

  p = a * a;
  // expected-error@-1 {{assigning to 'char *' from incompatible type 'float __attribute__((matrix_type(10, 10)))'}}
}

void mat_scalar_multiply(sx10x10_t a, sx5x10_t b, float sf, char *p) {
  // Shape of multiplication result does not match the type of b.
  b = a * sf;
  // expected-error@-1 {{assigning to 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') from incompatible type 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))')}}
  b = sf * a;
  // expected-error@-1 {{assigning to 'sx5x10_t' (aka 'float __attribute__((matrix_type(5, 10)))') from incompatible type 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))')}}

  a = a * p;
  // expected-error@-1 {{casting 'char *' to incompatible type 'float'}}
  // expected-error@-2 {{invalid operands to binary expression ('sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') and 'char *')}}
  a *= p;
  // expected-error@-1 {{casting 'char *' to incompatible type 'float'}}
  // expected-error@-2 {{invalid operands to binary expression ('sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))') and 'char *')}}
  a = p * a;
  // expected-error@-1 {{casting 'char *' to incompatible type 'float'}}
  // expected-error@-2 {{invalid operands to binary expression ('char *' and 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))'))}}
  p *= a;
  // expected-error@-1 {{casting 'char *' to incompatible type 'float'}}
  // expected-error@-2 {{invalid operands to binary expression ('char *' and 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))'))}}

  sf = a * sf;
  // expected-error@-1 {{assigning to 'float' from incompatible type 'sx10x10_t' (aka 'float __attribute__((matrix_type(10, 10)))')}}
}

sx5x10_t get_matrix();

void insert(sx5x10_t a, float f) {
  // Non integer indexes.
  a[3][f] = 0;
  // expected-error@-1 {{matrix column index is not an integer}}
  a[f][9] = 0;
  // expected-error@-1 {{matrix row index is not an integer}}
  a[f][f] = 0;
  // expected-error@-1 {{matrix row index is not an integer}}
  // expected-error@-2 {{matrix column index is not an integer}}
  a[0][f] = 0;
  // expected-error@-1 {{matrix column index is not an integer}}

  a[f][f] = 0;
  // expected-error@-1 {{matrix row index is not an integer}}
  // expected-error@-2 {{matrix column index is not an integer}}

  // Invalid element type.
  a[3][4] = &f;
  // expected-error@-1 {{assigning to 'float' from incompatible type 'float *'; remove &}}

  // Indexes outside allowed dimensions.
  a[-1][3] = 10.0;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  a[3][-1] = 10.0;
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 10)}}
  a[3][-1u] = 10.0;
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 10)}}
  a[-1u][3] = 10.0;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  a[5][2] = 10.0;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  a[4][10] = 10.0;
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 10)}}
  a[5][0] = f;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  (a[1])[1] = f;
  // expected-error@-1 {{matrix row and column subscripts cannot be separated by any expression}}

  a[3] = 5.0;
  // expected-error@-1 {{single subscript expressions are not allowed for matrix values}}

  (a[3]) = 5.0;
  // expected-error@-1 {{single subscript expressions are not allowed for matrix values}}

  get_matrix()[0][0] = f;
  // expected-error@-1 {{expression is not assignable}}
  get_matrix()[5][1] = f;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  get_matrix()[3] = 5.0;
  // expected-error@-1 {{single subscript expressions are not allowed for matrix values}}

  (get_matrix()[5])[10.0] = f;
  // expected-error@-1 {{matrix row and column subscripts cannot be separated by any expression}}
  (get_matrix()[3]) = 5.0;
  // expected-error@-1 {{single subscript expressions are not allowed for matrix values}}

  a([0])[0] = f;
  // expected-error@-1 {{expected expression}}
  a[0]([0]) = f;
  // expected-error@-1 {{expected expression}}
}

void extract(sx5x10_t a, float f) {
  // Non integer indexes.
  float v1 = a[3][f];
  // expected-error@-1 {{matrix column index is not an integer}}
  float v2 = a[f][9];
  // expected-error@-1 {{matrix row index is not an integer}}
  float v3 = a[f][f];
  // expected-error@-1 {{matrix row index is not an integer}}
  // expected-error@-2 {{matrix column index is not an integer}}

  // Invalid element type.
  char *v4 = a[3][4];
  // expected-error@-1 {{initializing 'char *' with an expression of incompatible type 'float'}}

  // Indexes outside allowed dimensions.
  float v5 = a[-1][3];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  float v6 = a[3][-1];
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 10)}}
  float v8 = a[-1u][3];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  float v9 = a[5][2];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  float v10 = a[4][10];
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 10)}}
  float v11 = a[5][9];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}

  float v12 = a[3];
  // expected-error@-1 {{single subscript expressions are not allowed for matrix values}}
}

float *address_of_element(sx5x10_t *a) {
  return &(*a)[0][1];
  // expected-error@-1 {{address of matrix element requested}}
}
