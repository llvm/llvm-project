// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -x hlsl -verify=dxil -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-library -finclude-default-header -x hlsl -verify=ok -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify=ok -o - %s

// SV_Barycentrics requires shader model 6.1 for DXIL.
// The shader model does not apply to SPIR-V, which is only run to check that no error is emitted.

[shader("pixel")]
float4 ps_in(float3 b : SV_Barycentrics) : SV_Target {
// dxil-error@-1 {{semantic 'SV_Barycentrics' requires shader model 6.1 or greater}}
  return float4(b, 1);
}

// ok-no-diagnostics
