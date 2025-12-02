// RUN: %clang --driver-mode=dxc %s -T ps_6_8 -E main1 -O3 -spirv -Xclang -verify -Xclang -verify-ignore-unexpected=note

// The following code is not legal: both semantics A and B will be lowered
// into a Location decoration. And mixing implicit and explicit Location
// assignment is not supported.
struct S1 {
  [[vk::location(3)]] float4 color : B;
  float4 position : A;
  // expected-error@-1 {{partial explicit stage input location assignment via vk::location(X) unsupported}}
};

[shader("pixel")]
float4 main1(S1 p) : SV_Target {
  return p.position + p.color;
}
