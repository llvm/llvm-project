// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -x hlsl -verify=dxil -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -finclude-default-header -x hlsl -verify=ok -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify=ok -o - %s

// SV_ShadingRate requires shader model 6.4 for DXIL.
// The shader model does not apply to SPIR-V, which is only run to check that no error is emitted.
// Behavior parity with DXC's hlsl/semantics/sv_shadingrate/shadingrate4.hlsl

[shader("vertex")]
float4 vs_out(out uint r : SV_ShadingRate) : SV_Position {
// dxil-error@-1 {{semantic 'SV_ShadingRate' requires shader model 6.4 or greater}}
  r = 5;
  return float4(0, 0, 0, 1);
}

[shader("pixel")]
float4 ps_in(uint r : SV_ShadingRate) : SV_Target {
// dxil-error@-1 {{semantic 'SV_ShadingRate' requires shader model 6.4 or greater}}
  return float4((float)r, 0, 0, 1);
}

// ok-no-diagnostics
