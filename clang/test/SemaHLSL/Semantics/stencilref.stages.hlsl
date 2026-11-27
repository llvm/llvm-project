// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_StencilRef: only a pixel output.
// Type and index checks are covered by stencilref.ps.hlsl.

struct PSOut {
  float4 c : SV_Target;
  uint s : SV_StencilRef;
};

[shader("pixel")]
PSOut ok_ps_out() { return (PSOut)0; }

[shader("pixel")]
void bad_ps_in(uint s : SV_StencilRef) {}
// expected-error@-1 {{semantic 'SV_StencilRef' is not supported in pixel shader inputs}}

[shader("vertex")]
uint bad_vs_out() : SV_StencilRef { return 0; }
// expected-error@-1 {{semantic 'SV_StencilRef' is not supported in vertex shader outputs}}

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_cs_in(uint s : SV_StencilRef) {}
// expected-error@-1 {{semantic 'SV_StencilRef' is not supported in compute shader inputs}}
