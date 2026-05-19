// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -x hlsl -verify -o - %s

float4 main(uint id : SV_VertexID) : SV_Target {
// expected-error@-1 {{attribute 'SV_VertexID' is unsupported in 'pixel' shaders, requires vertex}}
  return float4(1, 1, 1, 1);
}


