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
  a[5][10.0] = f;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  // expected-error@-2 {{matrix column index is not an integer}}

  get_matrix()[0][0] = f;
  // expected-error@-1 {{expression is not assignable}}
  get_matrix()[5][10.0] = f;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  // expected-error@-2 {{matrix column index is not an integer}}
  get_matrix()[3] = 5.0;
  // expected-error@-1 {{single subscript expressions are not allowed for matrix values}}

  float &x = reinterpret_cast<float &>(a[3][3]);
  // expected-error@-1 {{reinterpret_cast of a matrix element to 'float &' needs its address, which is not allowed}}

  a[4, 5] = 5.0;
  // expected-error@-1 {{comma expressions are not allowed as indices in matrix subscript expressions}}
  // expected-warning@-2 {{expression result unused}}

  a[4, 5, 4] = 5.0;
  // expected-error@-1 {{comma expressions are not allowed as indices in matrix subscript expressions}}
  // expected-warning@-2 {{expression result unused}}
  // expected-warning@-3 {{expression result unused}}
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
  // expected-error@-1 {{cannot initialize a variable of type 'char *' with an lvalue of type 'float'}}

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
  float v11 = a[5][10.0];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  // expected-error@-2 {{matrix column index is not an integer}}

  float v12 = get_matrix()[0][0];
  float v13 = get_matrix()[5][10.0];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 5)}}
  // expected-error@-2 {{matrix column index is not an integer}}
}

const float &const_subscript_reference(sx5x10_t m) {
  return m[2][2];
  // expected-warning@-1 {{returning reference to local temporary object}}
}

const float &const_subscript_reference(const sx5x10_t &m) {
  return m[2][2];
  // expected-warning@-1 {{returning reference to local temporary object}}
}

float &nonconst_subscript_reference(sx5x10_t m) {
  return m[2][2];
  // expected-error@-1 {{non-const reference cannot bind to matrix element}}
}

void incomplete_matrix_index_expr(sx5x10_t a, float f) {
  float x = a[3];
  // expected-error@-1 {{single subscript expressions are not allowed for matrix values}}
  a[2] = f;
  // expected-error@-1 {{single subscript expressions are not allowed for matrix values}}
}
