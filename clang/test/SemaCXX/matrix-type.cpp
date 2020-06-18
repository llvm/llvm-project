// RUN: %clang_cc1 -fsyntax-only -pedantic -fenable-matrix -std=c++11 -verify -triple x86_64-apple-darwin %s

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
