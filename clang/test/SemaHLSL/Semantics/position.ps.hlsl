// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -x hlsl -verify -o - %s

float4 main(float4 a : A) : SV_Position {
// expected-error@-1 {{attribute 'SV_Position' is unsupported in 'pixel' shaders, requires one of the following: pixel, vertex}}
  return a;
}
