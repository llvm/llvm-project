// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -std=hlsl202x -verify %s

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

  // truncation tests
  // result type int
  // truncate from struct
  constexpr B1 SB1 = {1.0, 3.0};
  constexpr float Blah = SB1.A;
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
}
