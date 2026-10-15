// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -finclude-default-header -fsyntax-only -verify=old %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-pixel -finclude-default-header -fsyntax-only -verify=valid %s
// valid-no-diagnostics

float4 main(float3 b : SV_Barycentrics) : SV_Target { return float4(b, 1); }
// old-error@-1 {{attribute 'SV_Barycentrics' requires shader model 6.1 or greater}}
