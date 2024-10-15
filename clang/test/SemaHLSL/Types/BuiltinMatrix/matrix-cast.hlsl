// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -fnative-half-type -finclude-default-header -fsyntax-only %s -verify

typedef struct test_struct { // expected-note 1+ {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  // expected-note-re@-1 1+ {{candidate constructor (the implicit move constructor) not viable: no known conversion from '{{[^']*}}' (aka '{{[^']*}}') to 'test_struct' for 1st argument}}
  // expected-note-re@-2 1+ {{candidate constructor (the implicit copy constructor) not viable: no known conversion from '{{[^']*}}' (aka '{{[^']*}}') to 'const test_struct' for 1st argument}}
} test_struct;

void f1(void) {
  uint16_t3x3 u16_3x3;
  int3x3 i32_3x3;
  int16_t3x3 i16_3x3;
  int4x4 i32_4x4;
  float4x4 f32_4x4;
  int i;
  float4 v;
  test_struct s;

  i32_3x3 = (int3x3)u16_3x3;
  i16_3x3 = (int16_t3x3)i32_3x3;
  i32_4x4 = (int4x4)i16_3x3;        // expected-error {{conversion between matrix types 'int4x4' (aka 'matrix<int, 4, 4>') and 'matrix<short, 3, 3>' of different size is not allowed}}
  f32_4x4 = (int4x4)i32_4x4;        // expected-error {{assigning to 'matrix<float, [2 * ...]>' from incompatible type 'matrix<int, [2 * ...]>'}}
  i = (int)i32_4x4;           // expected-error {{C-style cast from 'int4x4' (aka 'matrix<int, 4, 4>') to 'int' is not allowed}}
  i32_4x4 = (int4x4)i;         // expected-error {{C-style cast from 'int' to 'int4x4' (aka 'matrix<int, 4, 4>') is not allowed}}
  v = (float4)i32_4x4;           // expected-error {{C-style cast from 'int4x4' (aka 'matrix<int, 4, 4>') to 'float4' (aka 'vector<float, 4>') is not allowed}}
  i32_4x4 = (int4x4)v;         // expected-error {{C-style cast from 'float4' (aka 'vector<float, 4>') to 'int4x4' (aka 'matrix<int, 4, 4>') is not allowed}}
  s = (test_struct)i16_3x3; // expected-error {{no matching conversion for C-style cast from 'int16_t3x3' (aka 'matrix<int16_t, 3, 3>') to 'test_struct'}}
  i16_3x3 = (int16_t3x3)s;         // expected-error {{cannot convert 'test_struct' to 'int16_t3x3' (aka 'matrix<int16_t, 3, 3>') without a conversion operator}}

  i32_4x4 = (int4x4)f32_4x4;
}

void f2(void) {
  float2x2 f32_2x2;
  double3x3 f64_3x3;
  double2x2 f64_2x2;
  int4x4 i32_4x4;
  uint4x4 u32_4x4;
  uint3x3 u32_3x3;
  float f;

  f64_3x3 = (double3x3)f32_2x2; // expected-error {{conversion between matrix types 'double3x3' (aka 'matrix<double, 3, 3>') and 'matrix<float, 2, 2>' of different size is not allowed}}
  f64_2x2 = (double2x2)f32_2x2;

  u32_4x4 = (uint4x4)i32_4x4;
  i32_4x4 = (int4x4)u32_4x4;
  u32_3x3 = (uint3x3)i32_4x4; // expected-error {{conversion between matrix types 'uint3x3' (aka 'matrix<uint, 3, 3>') and 'matrix<int, 4, 4>' of different size is not allowed}}
  f = (float)i32_4x4;    // expected-error {{C-style cast from 'int4x4' (aka 'matrix<int, 4, 4>') to 'float' is not allowed}}
  i32_4x4 = (int4x4)f;    // expected-error {{C-style cast from 'float' to 'int4x4' (aka 'matrix<int, 4, 4>') is not allowed}}
}

template <typename X>
using matrix_3_3 = matrix<X, 3, 3>;

template <typename Y>
using matrix_4_4 = matrix<Y, 4, 4>;

void f3() {
  matrix_3_3<uint16_t> u16_3x3;
  matrix_3_3<int> i32_3x3;
  matrix_3_3<int16_t> i16_3x3;
  matrix_4_4<int> i32_4x4;
  matrix_4_4<float> f32_4x4;
  int i;
  int4 v;
  test_struct s;

  i32_3x3 = (matrix_3_3<int>)u16_3x3;
  i32_3x3 = u16_3x3; // expected-error {{assigning to 'matrix_3_3<int>' from incompatible type 'matrix_3_3<uint16_t>'}}
  i16_3x3 = (matrix_3_3<int16_t>)i32_3x3;
  i32_4x4 = (matrix_4_4<int>)i16_3x3; // expected-error {{conversion between matrix types 'matrix_4_4<int>' (aka 'matrix<int, 4, 4>') and 'matrix<short, 3, 3>' of different size is not allowed}}

  i = (int)i16_3x3;            // expected-error {{C-style cast from 'matrix_3_3<int16_t>' (aka 'matrix<int16_t, 3, 3>') to 'int' is not allowed}}
  i32_3x3 = (matrix_3_3<int>)i; // expected-error {{C-style cast from 'int' to 'matrix_3_3<int>' (aka 'matrix<int, 3, 3>') is not allowed}}

  v = (int4)i32_3x3;            // expected-error {{C-style cast from 'matrix_3_3<int>' (aka 'matrix<int, 3, 3>') to 'int4' (aka 'vector<int, 4>') is not allowed}}
  u16_3x3 = (matrix_3_3<uint16_t>)v; // expected-error {{C-style cast from 'int4' (aka 'vector<int, 4>') to 'matrix_3_3<uint16_t>' (aka 'matrix<uint16_t, 3, 3>') is not allowed}}
  s = (test_struct)u16_3x3;    // expected-error {{no matching conversion for C-style cast from 'matrix_3_3<uint16_t>' (aka 'matrix<uint16_t, 3, 3>') to 'test_struct'}}
  f32_4x4 = (matrix_4_4<float>)s; // expected-error {{cannot convert 'test_struct' to 'matrix_4_4<float>' (aka 'matrix<float, 4, 4>') without a conversion operator}}
}

void f4() {
  matrix_3_3<uint16_t> u16_3x3;
  matrix_3_3<int> i32_3x3;
  matrix_3_3<int16_t> i16_3x3;
  matrix_4_4<int> i32_4x4;
  matrix_4_4<float> f32_4x4;
  int i;
  int4 v;
  test_struct s;

  i32_3x3 = static_cast<matrix_3_3<int>>(u16_3x3);
  i16_3x3 = static_cast<matrix_3_3<int16_t>>(i32_3x3);
  i32_4x4 = static_cast<matrix_4_4<int>>(i16_3x3); // expected-error {{conversion between matrix types 'matrix_4_4<int>' (aka 'matrix<int, 4, 4>') and 'matrix<short, 3, 3>' of different size is not allowed}}

  i = static_cast<int>(i16_3x3);            // expected-error {{static_cast from 'matrix_3_3<int16_t>' (aka 'matrix<int16_t, 3, 3>') to 'int' is not allowed}}
  i32_3x3 = static_cast<matrix_3_3<int>>(i); // expected-error {{static_cast from 'int' to 'matrix_3_3<int>' (aka 'matrix<int, 3, 3>') is not allowed}}

  v = static_cast<int4>(i32_3x3);             // expected-error {{static_cast from 'matrix_3_3<int>' (aka 'matrix<int, 3, 3>') to 'int4' (aka 'vector<int, 4>') is not allowed}}
  i16_3x3 = static_cast<matrix_3_3<uint16_t>>(v); // expected-error {{static_cast from 'int4' (aka 'vector<int, 4>') to 'matrix_3_3<uint16_t>' (aka 'matrix<uint16_t, 3, 3>') is not allowed}}

  s = static_cast<test_struct>(u16_3x3);    // expected-error {{no matching conversion for static_cast from 'matrix_3_3<uint16_t>' (aka 'matrix<uint16_t, 3, 3>') to 'test_struct'}}
  f32_4x4 = static_cast<matrix_4_4<float>>(s); // expected-error {{cannot convert 'test_struct' to 'matrix_4_4<float>' (aka 'matrix<float, 4, 4>') without a conversion operator}}
}

void f5() {
  matrix_3_3<float> f32_3x3;
  matrix_3_3<double> f64_3x3;
  matrix_4_4<double> f64_4x4;
  matrix_4_4<int> i32_4x4;
  matrix_3_3<uint> u32_3x3;
  matrix_4_4<uint> u32_4x4;
  float f;

  f64_3x3 = (matrix_3_3<double>)f32_3x3;
  f64_4x4 = (matrix_4_4<double>)f32_3x3; // expected-error {{conversion between matrix types 'matrix_4_4<double>' (aka 'matrix<double, 4, 4>') and 'matrix<float, 3, 3>' of different size is not allowed}}
  i32_4x4 = (matrix_4_4<int>)f64_4x4;
  u32_3x3 = (matrix_4_4<uint>)i32_4x4; // expected-error {{assigning to 'matrix<[...], 3, 3>' from incompatible type 'matrix<[...], 4, 4>'}}
  u32_4x4 = (matrix_4_4<uint>)i32_4x4;
  i32_4x4 = (matrix_4_4<int>)u32_4x4;
}

void f6() {
  matrix_3_3<float> f32_3x3;
  matrix_3_3<double> f64_3x3;
  matrix_4_4<double> f64_4x4;
  matrix_4_4<int> i32_4x4;
  matrix_3_3<uint> u32_3x3;
  matrix_4_4<uint> u32_4x4;
  float f;

  f64_3x3 = static_cast<matrix_3_3<double>>(f32_3x3);
  f64_4x4 = static_cast<matrix_4_4<double>>(f32_3x3); // expected-error {{conversion between matrix types 'matrix_4_4<double>' (aka 'matrix<double, 4, 4>') and 'matrix<float, 3, 3>' of different size is not allowed}}

  i32_4x4 = static_cast<matrix_4_4<int>>(f64_4x4);
  u32_3x3 = static_cast<matrix_4_4<uint>>(i32_4x4); // expected-error {{assigning to 'matrix<[...], 3, 3>' from incompatible type 'matrix<[...], 4, 4>'}}
  u32_4x4 = static_cast<matrix_4_4<uint>>(i32_4x4);
  i32_4x4 = static_cast<matrix_4_4<signed int>>(u32_4x4);
}
