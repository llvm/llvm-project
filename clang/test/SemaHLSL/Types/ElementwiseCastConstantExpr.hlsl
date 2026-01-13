// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -fnative-int16-type -std=hlsl202x -verify %s

// expected-no-diagnostics

struct Base {
  double D;
  uint64_t2 U;
  int16_t I : 5;
  uint16_t I2: 5;
};

struct R : Base {
  int G : 10;
  int : 30;
  float F;
};

struct B1 {
  float A;
  float B;
};

struct B2 : B1 {
  int C;
  int D;
  bool BB;
};

export void fn() {
  _Static_assert(((float4)(int[6]){1,2,3,4,5,6}).x == 1.0, "Woo!");

  // This compiling successfully verifies that the array constant expression
  // gets truncated to a float at compile time for instantiation via the
  // flat cast
  _Static_assert(((int)(int[2]){1,2}) == 1, "Woo!");

  // truncation tests
  // result type int
  // truncate from struct
  constexpr B1 SB1 = {1.0, 3.0};
  constexpr int X = (int)SB1;
  _Static_assert(X == 1, "Woo!");

  // result type float
  // truncate from array
  constexpr B1 Arr[2] = {4.0, 3.0, 2.0, 1.0};
  constexpr float F = (float)Arr;
  _Static_assert(F == 4.0, "Woo!");

  // result type vector
  // truncate from array of vector
  constexpr int2 Arr2[2] = {5,6,7,8};
  constexpr int2 I2 = (int2)Arr2;
  _Static_assert(I2[0] == 5, "Woo!");
  _Static_assert(I2[1] == 6, "Woo!");

  // lhs and rhs are same "size" tests
  
  // result type vector from  array
  constexpr int4 I4 = (int4)Arr;
  _Static_assert(I4[0] == 4, "Woo!");
  _Static_assert(I4[1] == 3, "Woo!");
  _Static_assert(I4[2] == 2, "Woo!");
  _Static_assert(I4[3] == 1, "Woo!");

  // result type array from vector
  constexpr double3 D3 = {100.11, 200.11, 300.11};
  constexpr float FArr[3] = (float[3])D3;
  _Static_assert(FArr[0] == 100.11, "Woo!");
  _Static_assert(FArr[1] == 200.11, "Woo!");
  _Static_assert(FArr[2] == 300.11, "Woo!");

  // result type struct from struct
  constexpr B2 SB2 = {5.5, 6.5, 1000, 5000, false};
  constexpr Base SB = (Base)SB2;
  _Static_assert(SB.D == 5.5, "Woo!");
  _Static_assert(SB.U[0] == 6, "Woo!");
  _Static_assert(SB.U[1] == 1000, "Woo!");
  _Static_assert(SB.I == 8, "Woo!");
  _Static_assert(SB.I2 == 0, "Woo!");

  // Make sure we read bitfields correctly
  constexpr Base BB = {222.22, {100, 200}, -2, 7};
  constexpr int Arr3[5] = (int[5])BB;
  _Static_assert(Arr3[0] == 222, "Woo!");
  _Static_assert(Arr3[1] == 100, "Woo!");
  _Static_assert(Arr3[2] == 200, "Woo!");
  _Static_assert(Arr3[3] == -2, "Woo!");
  _Static_assert(Arr3[4] == 7, "Woo!");
}
