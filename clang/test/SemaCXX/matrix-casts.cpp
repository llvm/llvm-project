// RUN: %clang_cc1 -std=c++11 -fenable-matrix -fsyntax-only -verify %s

template <typename X>

using matrix_4_4 = X __attribute__((matrix_type(4, 4)));

template <typename Y>

using matrix_5_5 = Y __attribute__((matrix_type(5, 5)));

typedef struct test_struct {
} test_struct;

typedef int vec __attribute__((vector_size(4)));

void f1() {
  // TODO: Update this test once the support of C-style casts for C++ is implemented.
  matrix_4_4<char> m1;
  matrix_4_4<int> m2;
  matrix_4_4<short> m3;
  matrix_5_5<int> m4;
  int i;
  vec v;
  test_struct *s;

  (matrix_4_4<int>)m1;   // expected-error {{C-style cast from 'matrix_4_4<char>' (aka 'char __attribute__((matrix_type(4, \
4)))') to 'matrix_4_4<int>' (aka 'int __attribute__((matrix_type(4, 4)))') is not allowed}}
  (matrix_4_4<short>)m2; // expected-error {{C-style cast from 'matrix_4_4<int>' (aka 'int __attribute__((matrix_type(4, \
4)))') to 'matrix_4_4<short>' (aka 'short __attribute__((matrix_type(4, 4)))') is not allowed}}
  (matrix_5_5<int>)m3;   // expected-error {{C-style cast from 'matrix_4_4<short>' (aka 'short __attribute__((matrix_type(4, \
4)))') to 'matrix_5_5<int>' (aka 'int __attribute__((matrix_type(5, 5)))') is not allowed}}

  (int)m3;            // expected-error {{C-style cast from 'matrix_4_4<short>' (aka 'short __attribute__((matrix_type(4, \
4)))') to 'int'}}
  (matrix_4_4<int>)i; // expected-error {{C-style cast from 'int' to 'matrix_4_4<int>' (aka 'int __attribute__((\
matrix_type(4, 4)))') is not allowed}}

  (vec) m2;            // expected-error {{C-style cast from 'matrix_4_4<int>' (aka 'int __attribute__((matrix_type(4, 4)))') \
to 'vec' (vector of 1 'int' value) is not allowed}}
  (matrix_4_4<char>)v; // expected-error {{C-style cast from 'vec' (vector of 1 'int' value) to 'matrix_4_4<char>' \
(aka 'char __attribute__((matrix_type(4, 4)))') is not allowed}}

  (test_struct *)m1;    // expected-error {{cannot cast from type 'matrix_4_4<char>' (aka 'char __attribute__\
((matrix_type(4, 4)))') to pointer type 'test_struct *}}'
  (matrix_5_5<float>)s; // expected-error {{C-style cast from 'test_struct *' to 'matrix_5_5<float>' (aka 'float __attribute__\
((matrix_type(5, 5)))') is not allowed}}'
}

void f2() {
  // TODO: Update this test once the support of C-style casts for C++ is implemented.
  matrix_4_4<float> m1;
  matrix_5_5<double> m2;
  matrix_5_5<signed int> m3;
  matrix_4_4<unsigned int> m4;
  float f;

  (matrix_4_4<double>)m1;       // expected-error {{C-style cast from 'matrix_4_4<float>' (aka 'float __attribute__\
((matrix_type(4, 4)))') to 'matrix_4_4<double>' (aka 'double __attribute__((matrix_type(4, 4)))') is not allowed}}
  (matrix_5_5<float>)m2;        // expected-error {{C-style cast from 'matrix_5_5<double>' (aka 'double __attribute__\
((matrix_type(5, 5)))') to 'matrix_5_5<float>' (aka 'float __attribute__((matrix_type(5, 5)))') is not allowed}}
  (matrix_5_5<unsigned int>)m3; // expected-error {{C-style cast from 'matrix_5_5<int>' (aka 'int __attribute__\
((matrix_type(5, 5)))') to 'matrix_5_5<unsigned int>' (aka 'unsigned int __attribute__((matrix_type(5, 5)))') \
is not allowed}}
  (matrix_4_4<int>)m4;          // expected-error {{C-style cast from 'matrix_4_4<unsigned int>' (aka 'unsigned int \
__attribute__((matrix_type(4, 4)))') to 'matrix_4_4<int>' (aka 'int __attribute__((matrix_type(4, 4)))') is not \
allowed}}
}
