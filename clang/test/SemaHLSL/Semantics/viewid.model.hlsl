// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -x hlsl -verify=dxil -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-library -finclude-default-header -x hlsl -verify=ok -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify=ok -o - %s

// SV_ViewID requires shader model 6.1 for DXIL.
// The shader model restriction does not apply to SPIR-V, which is only run to check that no error is emitted.

[shader("pixel")]
float4 ps_in(uint v : SV_ViewID) : SV_Target {
// dxil-error@-1 {{semantic 'SV_ViewID' requires shader model 6.1 or greater}}
  return float4((float)v, 0, 0, 1);
}

// ok-no-diagnostics
