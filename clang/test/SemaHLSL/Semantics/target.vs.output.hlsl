// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-vertex -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-vertex -finclude-default-header -x hlsl -verify -o - %s

float4 main(float4 a : SV_Position) : SV_Target {
// expected-error@-1 {{semantic 'SV_Target' is not supported in vertex shader outputs}}
  return a;
}
