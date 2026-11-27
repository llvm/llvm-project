// RUN: %clang_cc1 -std=c++11 -fenable-matrix -fsyntax-only -verify %s

template <typename X>

using matrix_4_4 = X __attribute__((matrix_type(4, 4)));

template <typename Y>

using matrix_5_5 = Y __attribute__((matrix_type(5, 5)));

typedef struct test_struct {
} test_struct;

typedef int vec __attribute__((vector_size(4)));

void f1() {
  matrix_4_4<char> m1;
  matrix_4_4<int> m2;
  matrix_4_4<short> m3;
  matrix_5_5<int> m4;
  int i;
  vec v;
  test_struct *s;

  m2 = (matrix_4_4<int>)m1;
  m2 = m1; // expected-error {{assigning to 'matrix_4_4<int>' from incompatible type 'matrix_4_4<char>'}}
  m3 = (matrix_4_4<short>)m2;
  (matrix_5_5<int>)m3; // expected-error {{conversion between matrix types 'matrix_5_5<int>' (aka 'int __attribute__\
((matrix_type(5, 5)))') and 'short __attribute__((matrix_type(4, 4)))' of different size is not allowed}}

  (int)m3;            // expected-error {{C-style cast from 'matrix_4_4<short>' (aka 'short __attribute__((matrix_type(4, 4)))') to 'int' is not allowed}}
  (matrix_4_4<int>)i; // expected-error {{C-style cast from 'int' to 'matrix_4_4<int>' (aka 'int __attribute__((matrix_type(4, 4)))') is not allowed}}

  (vec) m2;            // expected-error {{C-style cast from 'matrix_4_4<int>' (aka 'int __attribute__((matrix_type(4, 4)))') to 'vec' (vector of 1 'int' value) is not allowed}}
  (matrix_4_4<char>)v; // expected-error {{C-style cast from 'vec' (vector of 1 'int' value) to 'matrix_4_4<char>' (aka 'char __attribute__((matrix_type(4, 4)))') is not allowed}}

  (test_struct *)m1;    // expected-error {{cannot cast from type 'matrix_4_4<char>' (aka 'char __attribute__((matrix_type(4, 4)))') to pointer type 'test_struct *'}}
  (matrix_5_5<float>)s; // expected-error {{C-style cast from 'test_struct *' (aka 'struct test_struct *') to 'matrix_5_5<float>' (aka 'float __attribute__((matrix_type(5, 5)))') is not allowed}}
}

void f2() {
  matrix_4_4<char> m1;
  matrix_4_4<int> m2;
  matrix_4_4<short> m3;
  matrix_5_5<int> m4;
  int i;
  vec v;
  test_struct *s;

  m2 = static_cast<matrix_4_4<int>>(m1);
  m3 = static_cast<matrix_4_4<short>>(m2);
  static_cast<matrix_5_5<int>>(m3); // expected-error {{conversion between matrix types 'matrix_5_5<int>' (aka 'int __attribute__\
((matrix_type(5, 5)))') and 'short __attribute__((matrix_type(4, 4)))' of different size is not allowed}}

  static_cast<int>(m3);            // expected-error {{static_cast from 'matrix_4_4<short>' (aka 'short __attribute__((matrix_type(4, 4)))') to 'int' is not allowed}}
  static_cast<matrix_4_4<int>>(i); // expected-error {{static_cast from 'int' to 'matrix_4_4<int>' (aka 'int __attribute__((matrix_type(4, 4)))') is not allowed}}

  static_cast<vec>(m2);             // expected-error {{static_cast from 'matrix_4_4<int>' (aka 'int __attribute__((matrix_type(4, 4)))') to 'vec' (vector of 1 'int' value) is not allowed}}
  static_cast<matrix_4_4<char>>(v); // expected-error {{static_cast from 'vec' (vector of 1 'int' value) to 'matrix_4_4<char>' (aka 'char __attribute__((matrix_type(4, 4)))') is not allowed}}

  static_cast<test_struct *>(m1);    // expected-error {{cannot cast from type 'matrix_4_4<char>' (aka 'char __attribute__((matrix_type(4, 4)))') to pointer type 'test_struct *'}}
  static_cast<matrix_5_5<float>>(s); // expected-error {{static_cast from 'test_struct *' (aka 'struct test_struct *') to 'matrix_5_5<float>' (aka 'float __attribute__((matrix_type(5, 5)))') is not allowed}}
}

void f3() {
  matrix_4_4<float> m1;
  matrix_4_4<double> m2;
  matrix_5_5<double> m3;
  matrix_5_5<signed int> m4;
  matrix_4_4<unsigned int> m5;
  matrix_5_5<unsigned int> m6;
  float f;

  m2 = (matrix_4_4<double>)m1;
  (matrix_5_5<double>)m1; // expected-error {{conversion between matrix types 'matrix_5_5<double>' (aka 'double __\
attribute__((matrix_type(5, 5)))') and 'float __attribute__((matrix_type(4, 4)))' of different\
 size is not allowed}}
  m4 = (matrix_5_5<signed int>)m3;
  m5 = (matrix_5_5<unsigned int>)m4; // expected-error {{assigning to 'matrix_4_4<unsigned int>' (aka 'unsigned int \
__attribute__((matrix_type(4, 4)))') from incompatible type 'matrix_5_5<unsigned int>' (aka 'unsigned int __attribute__\
((matrix_type(5, 5)))')}}
  m6 = (matrix_5_5<unsigned int>)m4;
  m4 = (matrix_5_5<signed int>)m6;
}

void f4() {
  matrix_4_4<float> m1;
  matrix_4_4<double> m2;
  matrix_5_5<double> m3;
  matrix_5_5<signed int> m4;
  matrix_4_4<unsigned int> m5;
  matrix_5_5<unsigned int> m6;
  float f;

  m2 = static_cast<matrix_4_4<double>>(m1);
  static_cast<matrix_5_5<double>>(m1); // expected-error {{conversion between matrix types 'matrix_5_5<double>' (aka 'double __\
attribute__((matrix_type(5, 5)))') and 'float __attribute__((matrix_type(4, 4)))' of different size is not allowed}}

  m4 = static_cast<matrix_5_5<signed int>>(m3);
  m5 = static_cast<matrix_5_5<unsigned int>>(m4); // expected-error {{assigning to 'matrix_4_4<unsigned int>' (aka 'unsigned int \
__attribute__((matrix_type(4, 4)))') from incompatible type 'matrix_5_5<unsigned int>' (aka 'unsigned int __attribute__\
((matrix_type(5, 5)))')}}
  m6 = static_cast<matrix_5_5<unsigned int>>(m4);
  m4 = static_cast<matrix_5_5<signed int>>(m6);
}

class Foo {
  // expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'matrix_4_4<float>' (aka 'float __attribute__((matrix_type(4, 4)))') to 'const Foo' for 1st argument}}
  // expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'matrix_4_4<float>' (aka 'float __attribute__((matrix_type(4, 4)))') to 'Foo' for 1st argument}}

  int x;

public:
  Foo();
  // expected-note@-1 {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  Foo(matrix_5_5<int> x);
  // expected-note@-1 {{candidate constructor not viable: no known conversion from 'matrix_4_4<float>' (aka 'float __attribute__((matrix_type(4, 4)))') to 'matrix_5_5<int>' (aka 'int __attribute__((matrix_type(5, 5)))') for 1st argument}}
};

struct Bar {
  // expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'matrix_4_4<unsigned int>' (aka 'unsigned int __attribute__((matrix_type(4, 4)))') to 'const Bar' for 1st argument}}
  // expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'matrix_4_4<unsigned int>' (aka 'unsigned int __attribute__((matrix_type(4, 4)))') to 'Bar' for 1st argument}}
  // expected-note@-3 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  float x;
};

void f5_constructors() {
  matrix_4_4<float> m1;
  matrix_4_4<unsigned int> m5;

  Foo F = Foo(m1);
  // expected-error@-1 {{no matching conversion for functional-style cast from 'matrix_4_4<float>' (aka 'float __attribute__((matrix_type(4, 4)))') to 'Foo'}}
  Bar B = Bar(m5);
  // expected-error@-1 {{no matching conversion for functional-style cast from 'matrix_4_4<unsigned int>' (aka 'unsigned int __attribute__((matrix_type(4, 4)))') to 'Bar'}}
}
