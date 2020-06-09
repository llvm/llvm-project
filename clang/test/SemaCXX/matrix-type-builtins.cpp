// RUN: %clang_cc1 %s -fenable-matrix -pedantic -std=c++11 -verify -triple=x86_64-apple-darwin9

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  using matrix_t = EltTy __attribute__((matrix_type(Rows, Columns)));

  matrix_t value;
};

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1>
typename MyMatrix<EltTy1, R1, C1>::matrix_t transpose(MyMatrix<EltTy0, R0, C0> &A) {
  char *v1 = __builtin_matrix_transpose(A.value);
  // expected-error@-1 {{cannot initialize a variable of type 'char *' with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 2)))'}}
  // expected-error@-2 {{cannot initialize a variable of type 'char *' with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 3)))'}}
  // expected-error@-3 {{cannot initialize a variable of type 'char *' with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 3)))'}}

  __builtin_matrix_transpose(A);
  // expected-error@-1 {{first argument must be a matrix}}
  // expected-error@-2 {{first argument must be a matrix}}
  // expected-error@-3 {{first argument must be a matrix}}

  return __builtin_matrix_transpose(A.value);
  // expected-error@-1 {{cannot initialize return object of type 'typename MyMatrix<unsigned int, 2U, 3U>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 3)))') with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 2)))'}}
  // expected-error@-2 {{cannot initialize return object of type 'typename MyMatrix<unsigned int, 2U, 3U>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 3)))') with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 3)))'}}
  // expected-error@-3 {{cannot initialize return object of type 'typename MyMatrix<float, 3U, 3U>::matrix_t' (aka 'float __attribute__((matrix_type(3, 3)))') with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 3)))'}}
}

void test_transpose_template(unsigned *Ptr1, float *Ptr2) {
  MyMatrix<unsigned, 2, 3> Mat1;
  MyMatrix<unsigned, 3, 3> Mat2;
  Mat1.value = *((decltype(Mat1)::matrix_t *)Ptr1);
  Mat1.value = transpose<unsigned, 2, 3, unsigned, 2, 3>(Mat1);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 2, 3, unsigned int, 2, 3>' requested here}}

  Mat1.value = transpose<unsigned, 3, 3, unsigned, 2, 3>(Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 3, 3, unsigned int, 2, 3>' requested here}}

  MyMatrix<float, 3, 3> Mat3;
  Mat3.value = transpose<unsigned, 3, 3, float, 3, 3>(Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 3, 3, float, 3, 3>' requested here}}
}
