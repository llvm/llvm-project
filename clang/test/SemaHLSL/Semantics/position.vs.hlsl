// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-vertex -x hlsl -finclude-default-header -o - %s -verify

// expected-error@+1 {{attribute 'SV_POSITION' is unsupported in 'vertex' shaders, requires pixel}}
float4 main(float4 a : SV_Position) {
  return a;
}
