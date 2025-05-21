// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify
// expected-no-diagnostics

#define NoOverlap0 "CBV(b0), CBV(b1)"

[RootSignature(NoOverlap0)]
void valid_root_signature_0() {}

#define NoOverlap1 "CBV(b0, visibility = SHADER_VISIBILITY_DOMAIN), CBV(b0, visibility = SHADER_VISIBILITY_PIXEL)"

[RootSignature(NoOverlap1)]
void valid_root_signature_1() {}

#define NoOverlap2 "CBV(b0, space = 1), CBV(b0, space = 2)"

[RootSignature(NoOverlap2)]
void valid_root_signature_2() {}

#define NoOverlap3 "CBV(b0), SRV(t0)"

[RootSignature(NoOverlap3)]
void valid_root_signature_3() {}
