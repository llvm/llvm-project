// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-vertex -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-vertex -finclude-default-header -x hlsl -verify -o - %s

float4 main(bool ff : SV_IsFrontFace) : SV_Position {
// expected-error@-1 {{semantic 'SV_IsFrontFace' is not supported in vertex shader inputs}}
  return ff ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}
