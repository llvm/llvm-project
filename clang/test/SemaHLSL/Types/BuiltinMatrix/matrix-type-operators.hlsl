// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -fnative-half-type -finclude-default-header -fsyntax-only %s -verify

void add(float4x4 a, float3x4 b, float4x3 c) {
  a = b + c;
  // expected-error@-1 {{invalid operands to binary expression ('float3x4' (aka 'matrix<float, 3, 4>') and 'float4x3' (aka 'matrix<float, 4, 3>'))}}

  b += c;
  // expected-error@-1 {{invalid operands to binary expression ('float3x4' (aka 'matrix<float, 3, 4>') and 'float4x3' (aka 'matrix<float, 4, 3>'))}}

  a = b + b; // expected-error {{assigning to 'matrix<[...], 4, [...]>' from incompatible type 'matrix<[...], 3, [...]>'}}

  a = 10 + b;
  // expected-error@-1 {{assigning to 'matrix<[...], 4, [...]>' from incompatible type 'matrix<[...], 3, [...]>'}}
}

void sub(float4x4 a, float3x4 b, float4x3 c) {
  a = b - c;
  // expected-error@-1 {{invalid operands to binary expression ('float3x4' (aka 'matrix<float, 3, 4>') and 'float4x3' (aka 'matrix<float, 4, 3>'))}}

  b -= c;
  // expected-error@-1 {{invalid operands to binary expression ('float3x4' (aka 'matrix<float, 3, 4>') and 'float4x3' (aka 'matrix<float, 4, 3>'))}}

  a = b - b; // expected-error {{assigning to 'matrix<[...], 4, [...]>' from incompatible type 'matrix<[...], 3, [...]>'}}

  a = 10 - b;
  // expected-error@-1 {{assigning to 'matrix<[...], 4, [...]>' from incompatible type 'matrix<[...], 3, [...]>'}}

}

void matrix_matrix_multiply(float4x4 a, float3x4 b, int4x3 c, int4x4 d, float sf, inout uint16_t p) {
  // Check dimension mismatches.
  a = a * b;
  // expected-error@-1 {{invalid operands to binary expression ('float4x4' (aka 'matrix<float, 4, 4>') and 'float3x4' (aka 'matrix<float, 3, 4>'))}}
  a *= b;
  // expected-error@-1 {{invalid operands to binary expression ('float4x4' (aka 'matrix<float, 4, 4>') and 'float3x4' (aka 'matrix<float, 3, 4>'))}}
  b = a * a;
  // expected-error@-1 {{assigning to 'matrix<[...], 3, [...]>' from incompatible type 'matrix<[...], 4, [...]>'}}

  // Check element type mismatches.
  a = b * c;
  // expected-error@-1 {{invalid operands to binary expression ('float3x4' (aka 'matrix<float, 3, 4>') and 'int4x3' (aka 'matrix<int, 4, 3>'))}}
  b *= c;
  // expected-error@-1 {{invalid operands to binary expression ('float3x4' (aka 'matrix<float, 3, 4>') and 'int4x3' (aka 'matrix<int, 4, 3>'))}}
  d = a * a;
  // expected-error@-1 {{assigning to 'matrix<int, [2 * ...]>' from incompatible type 'matrix<float, [2 * ...]>'}}

  p = a * a;
  // expected-error@-1 {{assigning to 'uint16_t' (aka 'unsigned short') from incompatible type 'float4x4' (aka 'matrix<float, 4, 4>')}}
}

void mat_scalar_multiply(float4x4 a, float3x4 b, float sf, inout uint16_t p) {
  // Shape of multiplication result does not match the type of b.
  b = a * sf;
  // expected-error@-1 {{assigning to 'matrix<[...], 3, [...]>' from incompatible type 'matrix<[...], 4, [...]>'}}
  b = sf * a;
  // expected-error@-1 {{assigning to 'matrix<[...], 3, [...]>' from incompatible type 'matrix<[...], 4, [...]>'}}

  sf = a * sf;
  // expected-error@-1 {{assigning to 'float' from incompatible type 'float4x4' (aka 'matrix<float, 4, 4>')}}
}

void mat_scalar_divide(float4x4 a, float3x4 b, float sf, inout uint16_t p) {
  // Shape of multiplication result does not match the type of b.
  b = a / sf;
  // expected-error@-1 {{assigning to 'matrix<[...], 3, [...]>' from incompatible type 'matrix<[...], 4, [...]>'}}
  b = sf / a;
  // expected-error@-1 {{invalid operands to binary expression ('float' and 'float4x4' (aka 'matrix<float, 4, 4>'))}}

  a = p / a;
  // expected-error@-1 {{invalid operands to binary expression ('uint16_t' (aka 'unsigned short') and 'float4x4' (aka 'matrix<float, 4, 4>'))}}

  sf = a / sf;
  // expected-error@-1 {{assigning to 'float' from incompatible type 'float4x4' (aka 'matrix<float, 4, 4>')}}
}

void matrix_matrix_divide(float4x4 a, float3x4 b, int4x3 c, int4x4 d, float sf, uint16_t p) {
  // Matrix by matrix division is not supported.
  a = a / a;
  // expected-error@-1 {{invalid operands to binary expression ('float4x4' (aka 'matrix<float, 4, 4>') and 'float4x4')}}

  b = a / a;
  // expected-error@-1 {{invalid operands to binary expression ('float4x4' (aka 'matrix<float, 4, 4>') and 'float4x4')}}

  // Check element type mismatches.
  a = b / c;
  // expected-error@-1 {{invalid operands to binary expression ('float3x4' (aka 'matrix<float, 3, 4>') and 'int4x3' (aka 'matrix<int, 4, 3>'))}}
  d = a / a;
  // expected-error@-1 {{invalid operands to binary expression ('float4x4' (aka 'matrix<float, 4, 4>') and 'float4x4')}}

  p = a / a;
  // expected-error@-1 {{invalid operands to binary expression ('float4x4' (aka 'matrix<float, 4, 4>') and 'float4x4')}}
}

float3x4 get_matrix(void);

void insert(float3x4 a, float f) {
  // Non integer indexes.
  a[1][f] = 0;
  // expected-error@-1 {{matrix column index is not an integer}}
  a[f][2] = 0;
  // expected-error@-1 {{matrix row index is not an integer}}
  a[f][f] = 0;
  // expected-error@-1 {{matrix row index is not an integer}}
  // expected-error@-2 {{matrix column index is not an integer}}
  a[0][f] = 0;
  // expected-error@-1 {{matrix column index is not an integer}}

  a[f][f] = 0;
  // expected-error@-1 {{matrix row index is not an integer}}
  // expected-error@-2 {{matrix column index is not an integer}}

  // Indexes outside allowed dimensions.
  a[-1][3] = 10.0;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  a[2][-1] = 10.0;
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 4)}}
  a[2][-1u] = 10.0;
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 4)}}
  a[-1u][3] = 10.0;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  a[5][2] = 10.0;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  a[2][10] = 10.0;
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 4)}}
  a[3][2.0] = f;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  // expected-error@-2 {{matrix column index is not an integer}}
  (a[1])[1] = f;
  // expected-error@-1 {{matrix row and column subscripts cannot be separated by any expression}}

  get_matrix()[0][0] = f;
  // expected-error@-1 {{expression is not assignable}}
  get_matrix()[3][1.0] = f;
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  // expected-error@-2 {{matrix column index is not an integer}}

  (get_matrix()[0])[2] = f;
  // expected-error@-1 {{matrix row and column subscripts cannot be separated by any expression}}

  a[4, 5] = 5.0;
  // expected-error@-1 {{comma expressions are not allowed as indices in matrix subscript expressions}}
  // expected-warning@-2 {{left operand of comma operator has no effect}}

  a[4, 5, 4] = 5.0;
  // expected-error@-1 {{comma expressions are not allowed as indices in matrix subscript expressions}}
  // expected-warning@-2 {{left operand of comma operator has no effect}}
  // expected-warning@-3 {{left operand of comma operator has no effect}}
}

void extract(float3x4 a, float f) {
  // Non integer indexes.
  float v1 = a[2][f];
  // expected-error@-1 {{matrix column index is not an integer}}
  float v2 = a[f][3];
  // expected-error@-1 {{matrix row index is not an integer}}
  float v3 = a[f][f];
  // expected-error@-1 {{matrix row index is not an integer}}
  // expected-error@-2 {{matrix column index is not an integer}}

  // Indexes outside allowed dimensions.
  float v5 = a[-1][3];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  float v6 = a[2][-1];
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 4)}}
  float v8 = a[-1u][3];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  float v9 = a[5][2];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  float v10 = a[2][4];
  // expected-error@-1 {{matrix column index is outside the allowed range [0, 4)}}
  float v11 = a[3][2.0];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  // expected-error@-2 {{matrix column index is not an integer}}

  float v12 = get_matrix()[0][0];
  float v13 = get_matrix()[3][2.0];
  // expected-error@-1 {{matrix row index is outside the allowed range [0, 3)}}
  // expected-error@-2 {{matrix column index is not an integer}}

}

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  using matrix_t = matrix<EltTy, Rows, Columns>;

  matrix_t value;
};

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1, typename EltTy2, unsigned R2, unsigned C2>
typename MyMatrix<EltTy2, R2, C2>::matrix_t add(inout MyMatrix<EltTy0, R0, C0> A, inout MyMatrix<EltTy1, R1, C1> B) {
  uint16_t v1 = A.value + B.value;
  // expected-error@-1 {{cannot initialize a variable of type 'uint16_t' (aka 'unsigned short') with an rvalue of type 'matrix_t' (aka 'matrix<unsigned int, 2, 2>')}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 3, 3>') and 'matrix_t' (aka 'matrix<float, 2, 2>'))}}
  // expected-error@-3 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 2, 2>') and 'matrix_t' (aka 'matrix<unsigned int, 3, 3>'))}}

  return A.value + B.value;
  // expected-error@-1 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 3, 3>') and 'matrix_t' (aka 'matrix<float, 2, 2>'))}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 2, 2>') and 'matrix_t' (aka 'matrix<unsigned int, 3, 3>'))}}
}

void test_add_template() {
  MyMatrix<unsigned, 2, 2> Mat1;
  MyMatrix<unsigned, 3, 3> Mat2;
  MyMatrix<float, 2, 2> Mat3;
  unsigned v1 = add<unsigned, 2, 2, unsigned, 2, 2, unsigned, 2, 2>(Mat1, Mat1);
  // expected-error@-1 {{cannot initialize a variable of type 'unsigned int' with an rvalue of type 'typename MyMatrix<unsigned int, 2U, 2U>::matrix_t' (aka 'matrix<unsigned int, 2, 2>')}}
  // expected-note@-2 {{in instantiation of function template specialization 'add<unsigned int, 2U, 2U, unsigned int, 2U, 2U, unsigned int, 2U, 2U>' requested here}}

  Mat1.value = add<unsigned, 2, 2, unsigned, 3, 3, unsigned, 2, 2>(Mat1, Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'add<unsigned int, 2U, 2U, unsigned int, 3U, 3U, unsigned int, 2U, 2U>' requested here}}

  Mat1.value = add<unsigned, 3, 3, float, 2, 2, unsigned, 2, 2>(Mat2, Mat3);
  // expected-note@-1 {{in instantiation of function template specialization 'add<unsigned int, 3U, 3U, float, 2U, 2U, unsigned int, 2U, 2U>' requested here}}
}

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1, typename EltTy2, unsigned R2, unsigned C2>
typename MyMatrix<EltTy2, R2, C2>::matrix_t subtract(inout MyMatrix<EltTy0, R0, C0> A, inout MyMatrix<EltTy1, R1, C1> B) {
  uint16_t v1 = A.value - B.value;
  // expected-error@-1 {{cannot initialize a variable of type 'uint16_t' (aka 'unsigned short') with an rvalue of type 'matrix_t' (aka 'matrix<unsigned int, 2, 2>')}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 3, 3>') and 'matrix_t' (aka 'matrix<float, 2, 2>')}}
  // expected-error@-3 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 2, 2>') and 'matrix_t' (aka 'matrix<unsigned int, 3, 3>')}}

  return A.value - B.value;
  // expected-error@-1 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 3, 3>') and 'matrix_t' (aka 'matrix<float, 2, 2>')}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 2, 2>') and 'matrix_t' (aka 'matrix<unsigned int, 3, 3>')}}
}

void test_subtract_template() {
  MyMatrix<unsigned, 2, 2> Mat1;
  MyMatrix<unsigned, 3, 3> Mat2;
  MyMatrix<float, 2, 2> Mat3;
  unsigned v1 = subtract<unsigned, 2, 2, unsigned, 2, 2, unsigned, 2, 2>(Mat1, Mat1);
  // expected-error@-1 {{cannot initialize a variable of type 'unsigned int' with an rvalue of type 'typename MyMatrix<unsigned int, 2U, 2U>::matrix_t' (aka 'matrix<unsigned int, 2, 2>')}}
  // expected-note@-2 {{in instantiation of function template specialization 'subtract<unsigned int, 2U, 2U, unsigned int, 2U, 2U, unsigned int, 2U, 2U>' requested here}}

  Mat1.value = subtract<unsigned, 2, 2, unsigned, 3, 3, unsigned, 2, 2>(Mat1, Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'subtract<unsigned int, 2U, 2U, unsigned int, 3U, 3U, unsigned int, 2U, 2U>' requested here}}

  Mat1.value = subtract<unsigned, 3, 3, float, 2, 2, unsigned, 2, 2>(Mat2, Mat3);
  // expected-note@-1 {{in instantiation of function template specialization 'subtract<unsigned int, 3U, 3U, float, 2U, 2U, unsigned int, 2U, 2U>' requested here}}
}

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1, typename EltTy2, unsigned R2, unsigned C2>
typename MyMatrix<EltTy2, R2, C2>::matrix_t multiply(inout MyMatrix<EltTy0, R0, C0> A, inout MyMatrix<EltTy1, R1, C1> B) {
  uint16_t v1 = A.value * B.value;
  // expected-error@-1 {{cannot initialize a variable of type 'uint16_t' (aka 'unsigned short') with an rvalue of type 'matrix_t' (aka 'matrix<unsigned int, 2, 2>')}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 3, 2>') and 'matrix_t' (aka 'matrix<unsigned int, 3, 3>'))}}
  // expected-error@-3 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<float, 2, 2>') and 'matrix_t' (aka 'matrix<unsigned int, 2, 2>'))}}

  MyMatrix<int, 3, 4> m;
  B.value = m.value * A.value;
  // expected-error@-1 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<int, 3, 4>') and 'matrix_t' (aka 'matrix<unsigned int, 2, 2>'))}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<int, 3, 4>') and 'matrix_t' (aka 'matrix<unsigned int, 3, 2>'))}}
  // expected-error@-3 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<int, 3, 4>') and 'matrix_t' (aka 'matrix<float, 2, 2>'))}}

  return A.value * B.value;
  // expected-error@-1 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 3, 2>') and 'matrix_t' (aka 'matrix<unsigned int, 3, 3>'))}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<float, 2, 2>') and 'matrix_t' (aka 'matrix<unsigned int, 2, 2>'))}}
}

void test_multiply_template() {
  MyMatrix<unsigned, 2, 2> Mat1;
  MyMatrix<unsigned, 3, 3> Mat2;
  MyMatrix<float, 2, 2> Mat3;
  unsigned v1 = multiply<unsigned, 2, 2, unsigned, 2, 2, unsigned, 2, 2>(Mat1, Mat1);
  // expected-note@-1 {{in instantiation of function template specialization 'multiply<unsigned int, 2U, 2U, unsigned int, 2U, 2U, unsigned int, 2U, 2U>' requested here}}
  // expected-error@-2 {{cannot initialize a variable of type 'unsigned int' with an rvalue of type 'typename MyMatrix<unsigned int, 2U, 2U>::matrix_t' (aka 'matrix<unsigned int, 2, 2>')}}

  MyMatrix<unsigned, 3, 2> Mat4;
  Mat1.value = multiply<unsigned, 3, 2, unsigned, 3, 3, unsigned, 2, 2>(Mat4, Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'multiply<unsigned int, 3U, 2U, unsigned int, 3U, 3U, unsigned int, 2U, 2U>' requested here}}

  Mat1.value = multiply<float, 2, 2, unsigned, 2, 2, unsigned, 2, 2>(Mat3, Mat1);
  // expected-note@-1 {{in instantiation of function template specialization 'multiply<float, 2U, 2U, unsigned int, 2U, 2U, unsigned int, 2U, 2U>' requested here}}

  Mat4.value = Mat4.value * Mat1;
  // expected-error@-1 {{no viable conversion from 'MyMatrix<unsigned int, 2, 2>' to 'unsigned int'}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<unsigned int, 3, 2>') and 'MyMatrix<unsigned int, 2, 2>')}}
}

struct UserT {};

struct StructWithC {
  operator UserT() {
    // expected-note@-1 4 {{candidate function}}
    return {};
  }
};

void test_DoubleWrapper(inout MyMatrix<double, 4, 3> m, inout StructWithC c) {
  m.value = m.value + c;
  // expected-error@-1 {{no viable conversion from 'StructWithC' to 'double'}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<double, 4, 3>') and 'StructWithC')}}

  m.value = c + m.value;
  // expected-error@-1 {{no viable conversion from 'StructWithC' to 'double'}}
  // expected-error@-2 {{invalid operands to binary expression ('StructWithC' and 'matrix_t' (aka 'matrix<double, 4, 3>'))}}

  m.value = m.value - c;
  // expected-error@-1 {{no viable conversion from 'StructWithC' to 'double'}}
  // expected-error@-2 {{invalid operands to binary expression ('matrix_t' (aka 'matrix<double, 4, 3>') and 'StructWithC')}}

  m.value = c - m.value;
  // expected-error@-1 {{no viable conversion from 'StructWithC' to 'double'}}
  // expected-error@-2 {{invalid operands to binary expression ('StructWithC' and 'matrix_t' (aka 'matrix<double, 4, 3>'))}}
}

