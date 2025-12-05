// RUN: %clang_cc1 -triple spirv-linux-vulkan-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s -verify -verify-ignore-unexpected=note

// This is almost the same as semantic.explicit-mix-builtin.hlsl, except this
// time we build a vertex shader. This means the SV_Position semantic is not
// a BuiltIn anymore, but a Location decorated variable. This means we mix
// implicit and explicit location assignment.
struct S1 {
  float4 position : SV_Position;
  [[vk::location(3)]] float4 color : A;
  // expected-error@-1 {{partial explicit stage input location assignment via vk::location(X) unsupported}}
};

[shader("vertex")]
float4 main1(S1 p) : A {
  return p.position + p.color;
}
