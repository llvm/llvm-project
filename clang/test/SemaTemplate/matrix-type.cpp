// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -fenable-matrix %s

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
  matrix_template_3<0, 10>(); // expected-note{{in instantiation of function template specialization 'matrix_template_3<0U, 10U>' requested here}}
}

template <int Rows, unsigned Cols>
void matrix_template_4() {
  using matrix1_t = float __attribute__((matrix_type(Rows, Cols))); // expected-error{{matrix row size too large}}
}

void instantiate_template_4() {
  matrix_template_4<2, 10>();
  matrix_template_4<-3, 10>(); // expected-note{{in instantiation of function template specialization 'matrix_template_4<-3, 10U>' requested here}}
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
template <typename T, unsigned R, unsigned C>
struct make1 {
  typedef T __attribute__((matrix_type(R, C))) type;
};

void test_make1() {
  make1<int, 5, 4>::type x;
}

template <typename T, unsigned R, unsigned C>
struct make2 {
  typedef T __attribute__((matrix_type(R, C))) type; // expected-error{{zero matrix size}}
};

int test_make2() {
  make2<int, 0, 1> x; // expected-note{{in instantiation of}}
}

template <typename T, unsigned R, unsigned C>
struct make3 {
  typedef T __attribute__((matrix_type(R, C))) type; // expected-error{{invalid matrix element type 's'}}
};

struct s {};

int test_make3() {
  make3<s, 3, 3> x; // expected-note{{in instantiation of}}
}

template <typename T, T R, unsigned C>
struct make4 {
  typedef T __attribute__((matrix_type(R, C))) type;
};

int test_make4() {
  make4<int, 4, 5>::type x;
}

typedef int *int_ptr;
template <unsigned R, unsigned C>
struct make5 {
  typedef int_ptr __attribute__((matrix_type(R, C))) type; // expected-error{{invalid matrix element type}}
};

template <int R, unsigned C>
struct make6 {
  typedef int __attribute__((matrix_type(R, C))) type;
};

int test_make6() {
  make6<4, 4>::type x;

  make6<2, 2>::type y;
}

namespace Deduction {
template <typename T>
struct X0;

template <typename T, unsigned N>
struct X0<T __attribute__((matrix_type(N, 3)))> {
  static const unsigned value = 0;
};

template <typename T>
struct X0<T __attribute__((matrix_type(4, 3)))> {
  static const unsigned value = 1;
};

template <unsigned N>
struct X0<float __attribute__((matrix_type(N, 3)))> {
  static const unsigned value = 2;
};

template <>
struct X0<float __attribute__((matrix_type(4, 3)))> {
  static const unsigned value = 3;
};

typedef int __attribute__((matrix_type(2, 3))) int2;
typedef int __attribute__((matrix_type(4, 3))) int4;
typedef float __attribute__((matrix_type(2, 3))) float2;
typedef float __attribute__((matrix_type(4, 3))) float4;

int array0[X0<int2>::value == 0 ? 1 : -1];
int array1[X0<int4>::value == 1 ? 1 : -1];
int array2[X0<float2>::value == 2 ? 1 : -1];
int array3[X0<float4>::value == 3 ? 1 : -1];
} // namespace Deduction
