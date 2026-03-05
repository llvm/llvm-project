// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -fnative-half-type -fnative-int16-type -std=hlsl202x -verify %s

// expected-no-diagnostics
// XFAIL because of this issue: https://github.com/llvm/llvm-project/issues/188577
// These tests hit an assert.
// XFAIL: *

struct Base {
  double D;
  uint64_t2 U;
  int16_t I : 5;
  uint16_t I2: 5;
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

  // splat from a vector of size 1 in an init list
  constexpr int1 A1 = {1};
  constexpr uint64_t2 A72[2] = {(uint64_t2[2])A1};
  _Static_assert(A72[0][0] == 1, "Woo!");
  _Static_assert(A72[0][1] == 1, "Woo!");
  _Static_assert(A72[1][0] == 1, "Woo!");
  _Static_assert(A72[1][1] == 1, "Woo!");

  // splat from a scalar inside an init list
  constexpr double D = 97.6789;
  constexpr B1 SB0 = {(B1)(D + 3.0)};
  _Static_assert(SB0.A == 100.6789, "Woo!");
  _Static_assert(SB0.B == 100.6789, "Woo!");

  // result type struct from struct in an init list
  constexpr B2 SB2 = {5.5, 6.5, 1000, 5000, false};
  constexpr Base SB3 = {(Base)SB2};
  _Static_assert(SB3.D == 5.5, "Woo!");
  _Static_assert(SB3.U[0] == 6, "Woo!");
  _Static_assert(SB3.U[1] == 1000, "Woo!");
  _Static_assert(SB3.I == 8, "Woo!");
  _Static_assert(SB3.I2 == 0, "Woo!");

  // Test taken from ElementwiseCastConstantExpr.hlsl
  // because it crashes for the same reason as the other tests in this file
  // result type struct from struct
  constexpr B2 SB2 = {5.5, 6.5, 1000, 5000, false};
  constexpr Base SB = (Base)SB2;
  _Static_assert(SB.D == 5.5, "Woo!");
  _Static_assert(SB.U[0] == 6, "Woo!");
  _Static_assert(SB.U[1] == 1000, "Woo!");
  _Static_assert(SB.I == 8, "Woo!");
  _Static_assert(SB.I2 == 0, "Woo!");

  // The below tests were taken from AggregateSplatConstantExpr.hlsl
  // because they crash for the same reason as the other tests in this file
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
