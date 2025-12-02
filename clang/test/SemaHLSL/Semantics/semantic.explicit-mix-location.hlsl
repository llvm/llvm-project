// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-pixel %s -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected=note

// The following code is not legal: both semantics A and B will be lowered
// into a Location decoration. And mixing implicit and explicit Location
// assignment is not supported.
struct S1 {
  float4 position : A;
  [[vk::location(3)]] float4 color : B;
  // expected-error@-1 {{partial explicit stage input location assignment via vk::location(X) unsupported}}
};

[shader("pixel")]
float4 main1(S1 p) : SV_Target {
  return p.position + p.color;
}
