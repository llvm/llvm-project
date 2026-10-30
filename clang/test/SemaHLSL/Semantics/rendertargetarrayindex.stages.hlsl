// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_RenderTargetArrayIndex: a vertex output
// and a pixel input. Type, index and interpolation checks are covered by
// rendertargetarrayindex.vs-ps.hlsl.

struct VSOut {
  float4 p : SV_Position;
  uint i : SV_RenderTargetArrayIndex;
};

[shader("vertex")]
VSOut ok_vs_out() { return (VSOut)0; }

[shader("pixel")]
float4 ok_ps_in(uint i : SV_RenderTargetArrayIndex) : SV_Target {
  return float4((float)i, 0, 0, 1);
}

// A vertex input with this name is an arbitrary semantic, so any type works.
[shader("vertex")]
float4 ok_vs_in_arbitrary(float4 i : SV_RenderTargetArrayIndex) : SV_Position {
  return i;
}

[shader("pixel")]
uint bad_ps_out() : SV_RenderTargetArrayIndex { return 0; }
// expected-error@-1 {{semantic 'SV_RenderTargetArrayIndex' is not supported in pixel shader outputs}}

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_cs_in(uint i : SV_RenderTargetArrayIndex) {}
// expected-error@-1 {{semantic 'SV_RenderTargetArrayIndex' is not supported in compute shader inputs}}
