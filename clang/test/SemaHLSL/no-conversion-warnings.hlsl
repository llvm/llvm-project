// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library -Wno-conversion -Wno-vector-conversion -Wno-matrix-conversion -verify %s

// expected-no-diagnostics

void test() {
  // vector truncation
  float3 F3 = 1.0;
  float2 F2 = F2;

  // matrix truncation
  int4x4 I4x4 = 1;
  int3x3 I3x3 = I4x4;

  // matrix to scalar
  float4x4 M4x4 = 2.0;
  float F = M4x4;

  // floating-point precision loss
  double4 D4 = F3.xyzx;
  float2 F2_2 = D4;

  // float to int
  int2 I2 = F2;

  // integer precision loss
  vector<long, 4> I64_4 = D4;
  int2 I2_2 = I64_4;
}
