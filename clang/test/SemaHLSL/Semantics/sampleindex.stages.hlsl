// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_SampleIndex: only a pixel input.
// Type, index and interpolation checks are covered by sampleindex.ps.hlsl.

[shader("pixel")]
float4 ok_ps_in(uint s : SV_SampleIndex) : SV_Target {
  return float4((float)s, 0, 0, 1);
}

[shader("pixel")]
uint bad_ps_out() : SV_SampleIndex { return 0; }
// expected-error@-1 {{semantic 'SV_SampleIndex' is not supported in pixel shader outputs}}

[shader("vertex")]
float4 bad_vs_in(uint s : SV_SampleIndex) : SV_Position { return 0; }
// expected-error@-1 {{semantic 'SV_SampleIndex' is not supported in vertex shader inputs}}

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_cs_in(uint s : SV_SampleIndex) {}
// expected-error@-1 {{semantic 'SV_SampleIndex' is not supported in compute shader inputs}}
