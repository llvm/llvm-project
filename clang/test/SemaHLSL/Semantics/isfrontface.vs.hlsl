// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-vertex -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-vertex -finclude-default-header -x hlsl -verify -o - %s

float4 main(bool ff : SV_IsFrontFace) : SV_Position {
// expected-error@-1 {{attribute 'SV_IsFrontFace' is unsupported in 'vertex' shaders, requires pixel}}
  return float4(1, 1, 1, 1);
}
