// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -finclude-default-header -x hlsl -verify -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// Stage and IO direction checks for SV_ShadingRate: a vertex output and a pixel input.
// Type and index checks are covered by shadingrate.ps.hlsl.

[shader("vertex")]
float4 ok_vs_out(out uint r : SV_ShadingRate) : SV_Position {
  r = 5;
  return float4(0, 0, 0, 1);
}

[shader("pixel")]
float4 ok_ps_in(uint r : SV_ShadingRate) : SV_Target {
  return float4((float)r, 0, 0, 1);
}

// User semantic next to SV_ShadingRate
[shader("pixel")]
float2 ok_ps_in_with_user_semantic(float2 v0 : Texcoord0,
                                   uint rate : SV_ShadingRate) : SV_Target {
  return v0 * (float)rate;
}

struct PSIn {
  float4 pos : SV_Position;
  uint rate : SV_ShadingRate;
};

[shader("pixel")]
float4 ok_ps_in_struct(PSIn i) : SV_Target {
  return float4((float)i.rate, 0, 0, 1);
}

struct VSOut {
  float4 pos : SV_Position;
  uint rate : SV_ShadingRate;
};

// User semantic named like the system value
[shader("vertex")]
VSOut ok_vs_out_struct(float4 pos : Pos, uint rate : ShadingRate) {
  VSOut o;
  o.pos = pos;
  o.rate = rate;
  return o;
}

[shader("vertex")]
float4 bad_vs_in(uint r : SV_ShadingRate) : SV_Position {
// expected-error@-1 {{semantic 'SV_ShadingRate' is not supported in vertex shader inputs}}
  return float4((float)r, 0, 0, 1);
}

[shader("pixel")]
float4 bad_ps_out(out uint r : SV_ShadingRate) : SV_Target {
// expected-error@-1 {{semantic 'SV_ShadingRate' is not supported in pixel shader outputs}}
  r = 0;
  return float4(0, 0, 0, 1);
}

static uint g;

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_cs_in(uint r : SV_ShadingRate) {
// expected-error@-1 {{semantic 'SV_ShadingRate' is not supported in compute shader inputs}}
  g = r;
}
