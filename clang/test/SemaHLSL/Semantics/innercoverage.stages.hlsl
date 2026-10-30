// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_InnerCoverage: only a pixel input.
// Type and index checks are covered by innercoverage.ps.hlsl.

[shader("pixel")]
float4 ok_ps_in(uint c : SV_InnerCoverage) : SV_Target {
  return float4((float)c, 0, 0, 1);
}

[shader("pixel")]
uint ps_output(float4 p : SV_Position) : SV_InnerCoverage {
// expected-error@-1 {{semantic 'SV_InnerCoverage' is not supported in pixel shader outputs}}
  return 0;
}

[shader("vertex")]
float4 vs_input(uint c : SV_InnerCoverage) : SV_Position {
// expected-error@-1 {{semantic 'SV_InnerCoverage' is not supported in vertex shader inputs}}
  return 0;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_input(uint c : SV_InnerCoverage) {}
// expected-error@-1 {{semantic 'SV_InnerCoverage' is not supported in compute shader inputs}}
