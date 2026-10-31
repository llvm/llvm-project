// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_Barycentrics: only a pixel input.
// Type, index and interpolation checks are covered by barycentrics.ps.hlsl.

[shader("pixel")]
float4 ok_ps_in(float3 b : SV_Barycentrics) : SV_Target {
  return float4(b, 1);
}

[shader("vertex")]
float4 wrong_stage(float3 b : SV_Barycentrics) : SV_Position { return 0; }
// expected-error@-1 {{semantic 'SV_Barycentrics' is not supported in vertex shader inputs}}

[shader("pixel")]
float3 wrong_output() : SV_Barycentrics { return 0; }
// expected-error@-1 {{semantic 'SV_Barycentrics' is not supported in pixel shader outputs}}

static float3 g;

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_cs_in(float3 b : SV_Barycentrics) {
// expected-error@-1 {{semantic 'SV_Barycentrics' is not supported in compute shader inputs}}
  g = b;
}
