// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_StartInstanceLocation: only a vertex input.
// Type and index checks are covered by startinstancelocation.vs.hlsl.

[shader("vertex")]
float4 ok_vs_in(int i : SV_StartInstanceLocation) : SV_Position {
  return float4((float)i, 0, 0, 1);
}

[shader("vertex")]
int bad_vs_out() : SV_StartInstanceLocation { return 0; }
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' is not supported in vertex shader outputs}}

[shader("pixel")]
void bad_ps_in(int i : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' is not supported in pixel shader inputs}}

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_cs_in(int i : SV_StartInstanceLocation) {}
// expected-error@-1 {{semantic 'SV_StartInstanceLocation' is not supported in compute shader inputs}}
