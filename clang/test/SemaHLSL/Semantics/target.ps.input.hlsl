// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -x hlsl -verify -o - %s

float4 main(float4 a : SV_Target) : A {
// expected-error@-1 {{semantic 'SV_Target' is unsupported as input for stage pixel}}
  return a;
}
