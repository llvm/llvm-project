// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -x hlsl -verify -o - %s

float4 main(float4 a : SV_Target) : SV_Target {
// expected-error@-1 {{semantic 'SV_Target' is not supported as pixel shader inputs; it is only available as an output}}
  return a;
}
