// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_IsFrontFace: only a pixel input.
// Type, index and interpolation checks are covered by isfrontface.ps.hlsl and
// isfrontface.ps.errors.hlsl.

[shader("pixel")]
float4 ok_ps_in(bool s : SV_IsFrontFace) : SV_Target {
  return s ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}

[shader("pixel")]
bool bad_ps_out() : SV_IsFrontFace { return false; }
// expected-error@-1 {{semantic 'SV_IsFrontFace' is not supported in pixel shader outputs}}

[shader("pixel")]
float4 bad_ps_out_param(out bool s : SV_IsFrontFace) : SV_Target {
// expected-error@-1 {{semantic 'SV_IsFrontFace' is not supported in pixel shader outputs}}
  s = true;
  return s ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}

[shader("vertex")]
float4 bad_vs_in(bool s : SV_IsFrontFace) : SV_Position { return 0; }
// expected-error@-1 {{semantic 'SV_IsFrontFace' is not supported in vertex shader inputs}}

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_cs_in(bool s : SV_IsFrontFace) {}
// expected-error@-1 {{semantic 'SV_IsFrontFace' is not supported in compute shader inputs}}
