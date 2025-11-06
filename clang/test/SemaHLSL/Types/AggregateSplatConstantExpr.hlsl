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

  // result type struct
  // splat from a scalar
  constexpr double D = 97.6789;
  constexpr R SR = (R)(D + 3.0);
  _Static_assert(SR.D == 100.6789, "Woo!");
  _Static_assert(SR.U[0] == 100, "Woo!");
  _Static_assert(SR.U[1] == 100, "Woo!");
  _Static_assert(SR.I == 4, "Woo!");
  _Static_assert(SR.I2 == 4, "Woo!");
  _Static_assert(SR.G == 100, "Woo!");
  _Static_assert(SR.F == 100.6789, "Woo!");

  // splat from a vector of size 1
  constexpr float1 A100 = {1000.1111};
  constexpr B2 SB2 = (B2)A100;
  _Static_assert(SB2.A == 1000.1111, "Woo!");
  _Static_assert(SB2.B == 1000.1111, "Woo!");
  _Static_assert(SB2.C == 1000, "Woo!");
  _Static_assert(SB2.D == 1000, "Woo!");
  _Static_assert(SB2.BB == true, "Woo!");

  // splat from a bool to an int and float etc
  constexpr bool B = true;
  constexpr B2 SB3 = (B2)B;
  _Static_assert(SB3.A == 1.0, "Woo!");
  _Static_assert(SB3.B == 1.0, "Woo!");
  _Static_assert(SB3.C == 1, "Woo!");
  _Static_assert(SB3.D == 1, "Woo!");
  _Static_assert(SB3.BB == true, "Woo!");
}
