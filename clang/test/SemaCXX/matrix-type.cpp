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

template <typename T> // expected-note{{declared here}}
void matrix_template_1() {
  using matrix1_t = float __attribute__((matrix_type(T, T))); // expected-error{{'T' does not refer to a value}}
}

template <class C> // expected-note{{declared here}}
void matrix_template_2() {
  using matrix1_t = float __attribute__((matrix_type(C, C))); // expected-error{{'C' does not refer to a value}}
}

template <unsigned Rows, unsigned Cols>
void matrix_template_3() {
  using matrix1_t = float __attribute__((matrix_type(Rows, Cols))); // expected-error{{zero matrix size}}
}

void instantiate_template_3() {
  matrix_template_3<1, 10>();
  matrix_template_3<0, 10>(); // expected-note{{in instantiation of function template specialization 'matrix_template_3<0, 10>' requested here}}
}

template <int Rows, unsigned Cols>
void matrix_template_4() {
  using matrix1_t = float __attribute__((matrix_type(Rows, Cols))); // expected-error{{matrix row size too large}}
}

void instantiate_template_4() {
  matrix_template_4<2, 10>();
  matrix_template_4<-3, 10>(); // expected-note{{in instantiation of function template specialization 'matrix_template_4<-3, 10>' requested here}}
}

template <class T, unsigned long R, unsigned long C>
using matrix = T __attribute__((matrix_type(R, C)));

template <class T, unsigned long R>
void use_matrix(matrix<T, R, 10> &m) {}
// expected-note@-1 {{candidate function [with T = float, R = 10]}}

template <class T, unsigned long C>
void use_matrix(matrix<T, 10, C> &m) {}
// expected-note@-1 {{candidate function [with T = float, C = 10]}}

void test_ambigous_deduction1() {
  matrix<float, 10, 10> m;
  use_matrix(m);
  // expected-error@-1 {{call to 'use_matrix' is ambiguous}}
}

template <class T, long R>
void type_conflict(matrix<T, R, 10> &m, T x) {}
// expected-note@-1 {{candidate template ignored: deduced conflicting types for parameter 'T' ('float' vs. 'char *')}}

void test_type_conflict(char *p) {
  matrix<float, 10, 10> m;
  type_conflict(m, p);
  // expected-error@-1 {{no matching function for call to 'type_conflict'}}
}

template <unsigned long R, unsigned long C>
matrix<float, R + 1, C + 2> use_matrix_2(matrix<int, R, C> &m) {}
// expected-note@-1 {{candidate function template not viable: requires single argument 'm', but 2 arguments were provided}}
// expected-note@-2 {{candidate function template not viable: requires single argument 'm', but 2 arguments were provided}}

template <unsigned long R, unsigned long C>
void use_matrix_2(matrix<int, R + 2, C / 2> &m1, matrix<float, R, C> &m2) {}
// expected-note@-1 {{candidate function [with R = 3, C = 11] not viable: no known conversion from 'matrix<int, 5, 6>' (aka 'int __attribute__((matrix_type(5, 6)))') to 'matrix<int, 3UL + 2, 11UL / 2> &' (aka 'int  __attribute__((matrix_type(5, 5)))&') for 1st argument}}
// expected-note@-2 {{candidate template ignored: deduced type 'matrix<float, 3UL, 4UL>' of 2nd parameter does not match adjusted type 'matrix<int, 3, 4>' of argument [with R = 3, C = 4]}}

template <typename T, unsigned long R, unsigned long C>
void use_matrix_2(matrix<T, R + C, C> &m1, matrix<T, R, C - R> &m2) {}
// expected-note@-1 {{candidate template ignored: deduced conflicting types for parameter 'T' ('int' vs. 'float')}}
// expected-note@-2 {{candidate template ignored: deduced type 'matrix<[...], 3UL + 4UL, 4UL>' of 1st parameter does not match adjusted type 'matrix<[...], 3, 4>' of argument [with T = int, R = 3, C = 4]}}

template <typename T, unsigned long R>
void use_matrix_3(matrix<T, R - 2, R> &m) {}
// expected-note@-1 {{candidate template ignored: deduced type 'matrix<[...], 5UL - 2, 5UL>' of 1st parameter does not match adjusted type 'matrix<[...], 5, 5>' of argument [with T = unsigned int, R = 5]}}

void test_use_matrix_2() {
  matrix<int, 5, 6> m1;
  matrix<float, 5, 8> r1 = use_matrix_2(m1);
  // expected-error@-1 {{cannot initialize a variable of type 'matrix<[...], 5, 8>' with an rvalue of type 'matrix<[...], 5UL + 1, 6UL + 2>'}}

  matrix<int, 4, 5> m2;
  matrix<float, 5, 8> r2 = use_matrix_2(m2);
  // expected-error@-1 {{cannot initialize a variable of type 'matrix<[...], 5, 8>' with an rvalue of type 'matrix<[...], 4UL + 1, 5UL + 2>'}}

  matrix<float, 3, 11> m3;
  use_matrix_2(m1, m3);
  // expected-error@-1 {{no matching function for call to 'use_matrix_2'}}

  matrix<int, 3, 4> m4;
  use_matrix_2(m4, m4);
  // expected-error@-1 {{no matching function for call to 'use_matrix_2'}}

  matrix<unsigned, 5, 5> m5;
  use_matrix_3(m5);
  // expected-error@-1 {{no matching function for call to 'use_matrix_3'}}
}
