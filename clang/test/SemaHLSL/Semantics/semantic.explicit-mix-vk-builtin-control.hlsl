// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-vertex -hlsl-entry main1 -fsyntax-only -verify -verify-ignore-unexpected=note %s

// Control for semantic.explicit-mix-vk-builtin-field.hlsl: without the
// vk::builtin attribute, the same struct shape is still rejected.
struct VSOutput {
  [[vk::location(0)]] float4 color : COLOR;
  float size : PSIZE;
  // expected-error@-1 {{partial explicit stage input location assignment via vk::location(X) unsupported}}
};

[shader("vertex")]
VSOutput main1() {
  VSOutput output = (VSOutput)0;
  return output;
}
