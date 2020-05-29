// RUN: %clang_cc1 %s -fenable-matrix -pedantic -std=c++11 -verify -triple=x86_64-apple-darwin9

typedef float sx5x10_t __attribute__((matrix_type(5, 10)));

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  using matrix_t = EltTy __attribute__((matrix_type(Rows, Columns)));

  matrix_t value;
};

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1, typename EltTy2, unsigned R2, unsigned C2>
typename MyMatrix<EltTy2, R2, C2>::matrix_t add(MyMatrix<EltTy0, R0, C0> &A, MyMatrix<EltTy1, R1, C1> &B) {
  char *v1 = A.value + B.value;
  // expected-error@-1 {{cannot initialize a variable of type 'char *' with an rvalue of type 'MyMatrix<unsigned int, 2, 2>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 2)))')}}
  // expected-error@-2 {{invalid operands to binary expression ('MyMatrix<unsigned int, 3, 3>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(3, 3)))') and 'MyMatrix<float, 2, 2>::matrix_t' (aka 'float __attribute__((matrix_type(2, 2)))'))}}
  // expected-error@-3 {{invalid operands to binary expression ('MyMatrix<unsigned int, 2, 2>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 2)))') and 'MyMatrix<unsigned int, 3, 3>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(3, 3)))'))}}

  return A.value + B.value;
  // expected-error@-1 {{invalid operands to binary expression ('MyMatrix<unsigned int, 3, 3>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(3, 3)))') and 'MyMatrix<float, 2, 2>::matrix_t' (aka 'float __attribute__((matrix_type(2, 2)))'))}}
  // expected-error@-2 {{invalid operands to binary expression ('MyMatrix<unsigned int, 2, 2>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 2)))') and 'MyMatrix<unsigned int, 3, 3>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(3, 3)))'))}}
}

void test_add_template(unsigned *Ptr1, float *Ptr2) {
  MyMatrix<unsigned, 2, 2> Mat1;
  MyMatrix<unsigned, 3, 3> Mat2;
  MyMatrix<float, 2, 2> Mat3;
  Mat1.value = *((decltype(Mat1)::matrix_t *)Ptr1);
  unsigned v1 = add<unsigned, 2, 2, unsigned, 2, 2, unsigned, 2, 2>(Mat1, Mat1);
  // expected-error@-1 {{cannot initialize a variable of type 'unsigned int' with an rvalue of type 'typename MyMatrix<unsigned int, 2U, 2U>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 2)))')}}
  // expected-note@-2 {{in instantiation of function template specialization 'add<unsigned int, 2, 2, unsigned int, 2, 2, unsigned int, 2, 2>' requested here}}

  Mat1.value = add<unsigned, 2, 2, unsigned, 3, 3, unsigned, 2, 2>(Mat1, Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'add<unsigned int, 2, 2, unsigned int, 3, 3, unsigned int, 2, 2>' requested here}}

  Mat1.value = add<unsigned, 3, 3, float, 2, 2, unsigned, 2, 2>(Mat2, Mat3);
  // expected-note@-1 {{in instantiation of function template specialization 'add<unsigned int, 3, 3, float, 2, 2, unsigned int, 2, 2>' requested here}}
}

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1, typename EltTy2, unsigned R2, unsigned C2>
typename MyMatrix<EltTy2, R2, C2>::matrix_t subtract(MyMatrix<EltTy0, R0, C0> &A, MyMatrix<EltTy1, R1, C1> &B) {
  char *v1 = A.value - B.value;
  // expected-error@-1 {{cannot initialize a variable of type 'char *' with an rvalue of type 'MyMatrix<unsigned int, 2, 2>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 2)))')}}
  // expected-error@-2 {{invalid operands to binary expression ('MyMatrix<unsigned int, 3, 3>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(3, 3)))') and 'MyMatrix<float, 2, 2>::matrix_t' (aka 'float __attribute__((matrix_type(2, 2)))')}}
  // expected-error@-3 {{invalid operands to binary expression ('MyMatrix<unsigned int, 2, 2>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 2)))') and 'MyMatrix<unsigned int, 3, 3>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(3, 3)))')}}

  return A.value - B.value;
  // expected-error@-1 {{invalid operands to binary expression ('MyMatrix<unsigned int, 3, 3>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(3, 3)))') and 'MyMatrix<float, 2, 2>::matrix_t' (aka 'float __attribute__((matrix_type(2, 2)))')}}
  // expected-error@-2 {{invalid operands to binary expression ('MyMatrix<unsigned int, 2, 2>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 2)))') and 'MyMatrix<unsigned int, 3, 3>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(3, 3)))')}}
}

void test_subtract_template(unsigned *Ptr1, float *Ptr2) {
  MyMatrix<unsigned, 2, 2> Mat1;
  MyMatrix<unsigned, 3, 3> Mat2;
  MyMatrix<float, 2, 2> Mat3;
  Mat1.value = *((decltype(Mat1)::matrix_t *)Ptr1);
  unsigned v1 = subtract<unsigned, 2, 2, unsigned, 2, 2, unsigned, 2, 2>(Mat1, Mat1);
  // expected-error@-1 {{cannot initialize a variable of type 'unsigned int' with an rvalue of type 'typename MyMatrix<unsigned int, 2U, 2U>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 2)))')}}
  // expected-note@-2 {{in instantiation of function template specialization 'subtract<unsigned int, 2, 2, unsigned int, 2, 2, unsigned int, 2, 2>' requested here}}

  Mat1.value = subtract<unsigned, 2, 2, unsigned, 3, 3, unsigned, 2, 2>(Mat1, Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'subtract<unsigned int, 2, 2, unsigned int, 3, 3, unsigned int, 2, 2>' requested here}}

  Mat1.value = subtract<unsigned, 3, 3, float, 2, 2, unsigned, 2, 2>(Mat2, Mat3);
  // expected-note@-1 {{in instantiation of function template specialization 'subtract<unsigned int, 3, 3, float, 2, 2, unsigned int, 2, 2>' requested here}}
}

struct UserT {};

struct StructWithC {
  operator UserT() {
    // expected-note@-1 4 {{candidate function}}
    return {};
  }
};

void test_DoubleWrapper(MyMatrix<double, 10, 9> &m, StructWithC &c) {
  m.value = m.value + c;
  // expected-error@-1 {{no viable conversion from 'StructWithC' to 'double'}}
  // expected-error@-2 {{invalid operands to binary expression ('MyMatrix<double, 10, 9>::matrix_t' (aka 'double __attribute__((matrix_type(10, 9)))') and 'StructWithC')}}

  m.value = c + m.value;
  // expected-error@-1 {{no viable conversion from 'StructWithC' to 'double'}}
  // expected-error@-2 {{invalid operands to binary expression ('StructWithC' and 'MyMatrix<double, 10, 9>::matrix_t' (aka 'double __attribute__((matrix_type(10, 9)))'))}}

  m.value = m.value - c;
  // expected-error@-1 {{no viable conversion from 'StructWithC' to 'double'}}
  // expected-error@-2 {{invalid operands to binary expression ('MyMatrix<double, 10, 9>::matrix_t' (aka 'double __attribute__((matrix_type(10, 9)))') and 'StructWithC')}}

  m.value = c - m.value;
  // expected-error@-1 {{no viable conversion from 'StructWithC' to 'double'}}
  // expected-error@-2 {{invalid operands to binary expression ('StructWithC' and 'MyMatrix<double, 10, 9>::matrix_t' (aka 'double __attribute__((matrix_type(10, 9)))'))}}
}
