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

// tests for HLSLAggregateSplatCast
export void fn() {
  // result type vector
  // splat from a vector of size 1
  
  constexpr float1 Y = {1.0};
  constexpr float4 F4 = (float4)Y;
  _Static_assert(F4[0] == 1.0, "Woo!");
  _Static_assert(F4[1] == 1.0, "Woo!");
  _Static_assert(F4[2] == 1.0, "Woo!");
  _Static_assert(F4[3] == 1.0, "Woo!");

  // result type array
  // splat from a scalar
  constexpr float F = 3.33;
  constexpr int B6[6] = (int[6])F;
  _Static_assert(B6[0] == 3, "Woo!");
  _Static_assert(B6[1] == 3, "Woo!");
  _Static_assert(B6[2] == 3, "Woo!");
  _Static_assert(B6[3] == 3, "Woo!");
  _Static_assert(B6[4] == 3, "Woo!");
  _Static_assert(B6[5] == 3, "Woo!");

  // splat from a vector of size 1
  constexpr int1 A1 = {1};
  constexpr uint64_t2 A7[2] = (uint64_t2[2])A1;
  _Static_assert(A7[0][0] == 1, "Woo!");
  _Static_assert(A7[0][1] == 1, "Woo!");
  _Static_assert(A7[1][0] == 1, "Woo!");
  _Static_assert(A7[1][1] == 1, "Woo!");
}
