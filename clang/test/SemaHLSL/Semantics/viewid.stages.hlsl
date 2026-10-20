// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_ViewID: an input to every graphics
// stage except amplification shaders. Type, index and interpolation checks are
// covered by viewid.ps.hlsl.

[shader("vertex")]
float4 ok_vs_in(uint v : SV_ViewID) : SV_Position {
  return float4((float)v, 0, 0, 1);
}

[shader("pixel")]
float4 ok_ps_in(uint v : SV_ViewID) : SV_Target {
  return float4((float)v, 0, 0, 1);
}

[shader("vertex")]
uint bad_vs_out() : SV_ViewID { return 0; }
// expected-error@-1 {{semantic 'SV_ViewID' is not supported in vertex shader outputs}}

[shader("pixel")]
uint bad_ps_out() : SV_ViewID { return 0; }
// expected-error@-1 {{semantic 'SV_ViewID' is not supported in pixel shader outputs}}

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_cs_in(uint v : SV_ViewID) {}
// expected-error@-1 {{semantic 'SV_ViewID' is not supported in compute shader inputs}}

[shader("amplification")]
[numthreads(1, 1, 1)]
void bad_as_in(uint v : SV_ViewID) {}
// expected-error@-1 {{semantic 'SV_ViewID' is not supported in amplification shader inputs}}
