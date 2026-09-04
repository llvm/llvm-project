// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -fexperimental-new-constant-interpreter -verify %s

// expected-no-diagnostics

// Tests for constexpr evaluation of element-wise vector casts.

export void fn() {
  // CK_IntegralToFloating: int4 -> float4
  constexpr float4 ItF = (float4)int4(1, 2, 3, 4);
  _Static_assert(ItF.x == 1.0f, "");
  _Static_assert(ItF.y == 2.0f, "");
  _Static_assert(ItF.z == 3.0f, "");
  _Static_assert(ItF.w == 4.0f, "");

  // CK_FloatingToIntegral: float4 -> int4 (truncation toward zero)
  constexpr int4 FtI = (int4)float4(1.9f, 2.1f, -3.9f, -4.1f);
  _Static_assert(FtI.x == 1, "");
  _Static_assert(FtI.y == 2, "");
  _Static_assert(FtI.z == -3, "");
  _Static_assert(FtI.w == -4, "");

  // CK_IntegralCast: int4 -> uint4
  constexpr uint4 IC = (uint4)int4(1, 2, 3, 4);
  _Static_assert(IC.x == 1u, "");
  _Static_assert(IC.y == 2u, "");
  _Static_assert(IC.z == 3u, "");
  _Static_assert(IC.w == 4u, "");

  // CK_FloatingCast: float4 -> double4
  constexpr double4 FC = (double4)float4(1.0f, 2.0f, 3.0f, 4.0f);
  _Static_assert(FC.x == 1.0, "");
  _Static_assert(FC.y == 2.0, "");
  _Static_assert(FC.z == 3.0, "");
  _Static_assert(FC.w == 4.0, "");

  // CK_IntegralToBoolean: int4 -> bool4
  constexpr bool4 ItB = (bool4)int4(1, 0, -1, 2);
  _Static_assert(ItB.x == true, "");
  _Static_assert(ItB.y == false, "");
  _Static_assert(ItB.z == true, "");
  _Static_assert(ItB.w == true, "");

  // CK_FloatingToBoolean: float4 -> bool4
  constexpr bool4 FtB = (bool4)float4(1.0f, 0.0f, -1.0f, 2.0f);
  _Static_assert(FtB.x == true, "");
  _Static_assert(FtB.y == false, "");
  _Static_assert(FtB.z == true, "");
  _Static_assert(FtB.w == true, "");

  // Bool source: CK_IntegralToFloating with bool4 -> float4
  constexpr float4 BtF = (float4)bool4(true, false, true, false);
  _Static_assert(BtF.x == 1.0f, "");
  _Static_assert(BtF.y == 0.0f, "");
  _Static_assert(BtF.z == 1.0f, "");
  _Static_assert(BtF.w == 0.0f, "");

  // Bool source: CK_IntegralCast with bool4 -> int4
  constexpr int4 BtI = (int4)bool4(true, false, true, false);
  _Static_assert(BtI.x == 1, "");
  _Static_assert(BtI.y == 0, "");
  _Static_assert(BtI.z == 1, "");
  _Static_assert(BtI.w == 0, "");
}
