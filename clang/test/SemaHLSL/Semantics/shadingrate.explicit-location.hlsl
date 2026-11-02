// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify -o - %s

// SV_ShadingRate lowers to a BuiltIn, not a Location. It must therefore not
// count towards the explicit/implicit vk::location consistency check or
// advance the implicit location index.

// Pixel input.
[shader("pixel")]
float4 ps_main([[vk::location(0)]] float4 c : COLOR, uint r : SV_ShadingRate) : SV_Target {
  return c * (float)r;
}

// Vertex output.
struct VSOut {
  float4 pos : SV_Position;
  [[vk::location(1)]] float4 color : COLOR;
  uint rate : SV_ShadingRate;
};

[shader("vertex")]
VSOut vs_main(float4 p : A) {
  VSOut o;
  o.pos = p;
  o.color = p;
  o.rate = 5;
  return o;
}
// expected-no-diagnostics
