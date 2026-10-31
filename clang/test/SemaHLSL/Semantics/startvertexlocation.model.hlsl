// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.7-library -finclude-default-header -x hlsl -verify=dxil -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-library -finclude-default-header -x hlsl -verify=ok -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify=ok -o - %s

// SV_StartVertexLocation requires shader model 6.8 for DXIL.
// The shader model restriction does not apply to SPIR-V, which is only run to check that no error is emitted.

[shader("vertex")]
float4 vs_in(int v : SV_StartVertexLocation) : SV_Position {
// dxil-error@-1 {{semantic 'SV_StartVertexLocation' requires shader model 6.8 or greater}}
  return float4((float)v, 0, 0, 1);
}

// ok-no-diagnostics
