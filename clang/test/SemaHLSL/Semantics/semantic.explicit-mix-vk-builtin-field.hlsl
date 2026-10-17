// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-vertex -hlsl-entry main1 -fsyntax-only -verify %s

// vk::builtin fields are exempt from the "all or nothing" explicit-location
// check, so they can be mixed with vk::location annotated fields.
struct VSOutput {
  [[vk::location(0)]] float4 color : COLOR;
  [[vk::location(1)]] float2 uv : TEXCOORD0;
  [[vk::builtin("PointSize")]] float size : PSIZE;
  [[vk::location(2)]] float other : TEXCOORD1;
};

[shader("vertex")]
VSOutput main1() {
  VSOutput output = (VSOutput)0;
  return output;
}

// expected-no-diagnostics
