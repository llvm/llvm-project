// RUN: %clang_cc1 -fsyntax-only -fenable-matrix -std=c++11 -verify -triple x86_64-apple-darwin %s

using matrix_double_t = double __attribute__((matrix_type(6, 6)));
using matrix_float_t = float __attribute__((matrix_type(6, 6)));
using matrix_int_t = int __attribute__((matrix_type(6, 6)));

void matrix_var_dimensions(int Rows, unsigned Columns, char C) {
  using matrix1_t = int __attribute__((matrix_type(Rows, 1)));    // expected-error{{matrix_type attribute requires an integer constant}}
  using matrix2_t = int __attribute__((matrix_type(1, Columns))); // expected-error{{matrix_type attribute requires an integer constant}}
  using matrix3_t = int __attribute__((matrix_type(C, C)));       // expected-error{{matrix_type attribute requires an integer constant}}
  using matrix4_t = int __attribute__((matrix_type(-1, 1)));      // expected-error{{matrix row size too large}}
  using matrix5_t = int __attribute__((matrix_type(1, -1)));      // expected-error{{matrix column size too large}}
  using matrix6_t = int __attribute__((matrix_type(0, 1)));       // expected-error{{zero matrix size}}
  using matrix7_t = int __attribute__((matrix_type(1, 0)));       // expected-error{{zero matrix size}}
  using matrix7_t = int __attribute__((matrix_type(char, 0)));    // expected-error{{expected '(' for function-style cast or type construction}}
  using matrix8_t = int __attribute__((matrix_type(1048576, 1))); // expected-error{{matrix row size too large}}
}

struct S1 {};

enum TestEnum {
  A,
  B
};

void matrix_unsupported_element_type() {
  using matrix1_t = char *__attribute__((matrix_type(1, 1)));    // expected-error{{invalid matrix element type 'char *'}}
  using matrix2_t = S1 __attribute__((matrix_type(1, 1)));       // expected-error{{invalid matrix element type 'S1'}}
  using matrix3_t = bool __attribute__((matrix_type(1, 1)));     // expected-error{{invalid matrix element type 'bool'}}
  using matrix4_t = TestEnum __attribute__((matrix_type(1, 1))); // expected-error{{invalid matrix element type 'TestEnum'}}
}

void matrix_unsupported_bit_int() {
  using m1 = _BitInt(2) __attribute__((matrix_type(4, 4))); // expected-error{{'_BitInt' matrix element width must be at least as wide as 'CHAR_BIT'}}
  using m2 = _BitInt(7) __attribute__((matrix_type(4, 4))); // expected-error{{'_BitInt' matrix element width must be at least as wide as 'CHAR_BIT'}}
  using m3 = _BitInt(9) __attribute__((matrix_type(4, 4))); // expected-error{{'_BitInt' matrix element width must be a power of 2}}
  using m4 = _BitInt(12) __attribute__((matrix_type(4, 4))); // expected-error{{'_BitInt' matrix element width must be a power of 2}}
  using m5 = _BitInt(8) __attribute__((matrix_type(4, 4)));
  using m6 = _BitInt(64) __attribute__((matrix_type(4, 4)));
  using m7 = _BitInt(256) __attribute__((matrix_type(4, 4)));
}
