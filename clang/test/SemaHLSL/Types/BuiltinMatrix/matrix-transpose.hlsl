// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -fnative-half-type -finclude-default-header -fsyntax-only %s -verify

void transpose(float3x4 a, int3x2 b, double3x3 c, int e) {

  a = __builtin_matrix_transpose(b);
  // expected-error@-1 {{assigning to 'float3x4' (aka 'matrix<float, 3, 4>') from incompatible type 'matrix<int, 2, 3>'}}
  b = __builtin_matrix_transpose(b);
  // expected-error@-1 {{assigning to 'int3x2' (aka 'matrix<int, 3, 2>') from incompatible type 'matrix<int, 2, 3>'}}
  __builtin_matrix_transpose(e);
  // expected-error@-1 {{1st argument must be a matrix}}
  __builtin_matrix_transpose("test");
  // expected-error@-1 {{1st argument must be a matrix}}

  uint3x3 m = __builtin_matrix_transpose(c);
  // expected-error@-1 {{cannot initialize a variable of type 'uint3x3' (aka 'matrix<uint, 3, 3>') with an rvalue of type 'matrix<double, 3, 3>'}}
}

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  using matrix_t = matrix<EltTy, Rows, Columns>;

  matrix_t value;
};

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1>
typename MyMatrix<EltTy1, R1, C1>::matrix_t transpose(inout MyMatrix<EltTy0, R0, C0> A) {
  uint16_t v1 = __builtin_matrix_transpose(A.value);
  // expected-error@-1 {{cannot initialize a variable of type 'uint16_t' (aka 'unsigned short') with an rvalue of type 'matrix<unsigned int, 3, 2>'}}
  // expected-error@-2 2 {{cannot initialize a variable of type 'uint16_t' (aka 'unsigned short') with an rvalue of type 'matrix<unsigned int, 3, 3>'}}

  __builtin_matrix_transpose(A);
  // expected-error@-1 3 {{1st argument must be a matrix}}

  return __builtin_matrix_transpose(A.value);
  // expected-error@-1 {{cannot initialize return object of type 'typename MyMatrix<unsigned int, 2U, 3U>::matrix_t' (aka 'matrix<unsigned int, 2, 3>') with an rvalue of type 'matrix<unsigned int, 3, 2>'}}
  // expected-error@-2 {{cannot initialize return object of type 'typename MyMatrix<unsigned int, 2U, 3U>::matrix_t' (aka 'matrix<unsigned int, 2, 3>') with an rvalue of type 'matrix<unsigned int, 3, 3>'}}
  // expected-error@-3 {{cannot initialize return object of type 'typename MyMatrix<float, 3U, 3U>::matrix_t' (aka 'matrix<float, 3, 3>') with an rvalue of type 'matrix<unsigned int, 3, 3>'}}
}

void test_transpose_template() {
  MyMatrix<unsigned, 2, 3> Mat1;
  MyMatrix<unsigned, 3, 3> Mat2;
  Mat1.value = transpose<unsigned, 2, 3, unsigned, 2, 3>(Mat1);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 2U, 3U, unsigned int, 2U, 3U>' requested here}}

  Mat1.value = transpose<unsigned, 3, 3, unsigned, 2, 3>(Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 3U, 3U, unsigned int, 2U, 3U>' requested here}}

  MyMatrix<float, 3, 3> Mat3;
  Mat3.value = transpose<unsigned, 3, 3, float, 3, 3>(Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 3U, 3U, float, 3U, 3U>' requested here}}
}

