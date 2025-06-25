// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

// Attr test

[RootSignature()] // expected-error {{expected string literal as argument of 'RootSignature' attribute}}
void bad_root_signature_0() {}

// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
[RootSignature("", "")]
void bad_root_signature_1() {}

[RootSignature(""), RootSignature("DescriptorTable()")] // expected-error {{attribute 'RootSignature' cannot appear more than once on a declaration}}
void bad_root_signature_2() {}

[RootSignature(""), RootSignature("")] // expected-warning {{attribute 'RootSignature' is already applied}}
void bad_root_signature_3() {}

[RootSignature("DescriptorTable(), invalid")] // expected-error {{expected end of stream to denote end of parameters, or, another valid parameter of RootSignature}}
void bad_root_signature_4() {}

// Basic validation of register value and space

// expected-error@+2 {{parameter value must be in the range [0, 4294967294]}}
// expected-error@+1 {{parameter value must be in the range [0, 4294967279]}}
[RootSignature("CBV(b4294967295, space = 4294967280)")]
void bad_root_signature_5() {}

// expected-error@+2 {{parameter value must be in the range [0, 4294967294]}}
// expected-error@+1 {{parameter value must be in the range [0, 4294967279]}}
[RootSignature("RootConstants(b4294967295, space = 4294967280, num32BitConstants = 1)")]
void bad_root_signature_6() {}

// expected-error@+2 {{parameter value must be in the range [0, 4294967294]}}
// expected-error@+1 {{parameter value must be in the range [0, 4294967279]}}
[RootSignature("StaticSampler(s4294967295, space = 4294967280)")]
void bad_root_signature_7() {}

// expected-error@+2 {{parameter value must be in the range [0, 4294967294]}}
// expected-error@+1 {{parameter value must be in the range [0, 4294967279]}}
[RootSignature("DescriptorTable(SRV(t4294967295, space = 4294967280))")]
void bad_root_signature_8() {}

// expected-error@+2 {{parameter value must be in the range [1, 4294967294]}}
// expected-error@+1 {{parameter value must be in the range [1, 4294967294]}}
[RootSignature("DescriptorTable(UAV(u0, numDescriptors = 0), Sampler(s0, numDescriptors = 0))")]
void bad_root_signature_9() {}

#define ErroneousStaticSampler \
  "StaticSampler(s0" \
  "  maxAnisotropy = 17," \
  ")"

// expected-error@+2 {{parameter value must be in the range [0, 16]}}
// expected-error@+1 {{parameter value must be in the range [-16.000000, 15.990000]}}
[RootSignature("StaticSampler(s0, maxAnisotropy = 17, mipLODBias = -16.000001)")]
void bad_root_signature_10() {}

// expected-error@+1 {{parameter value must be in the range [-16.000000, 15.990000]}}
[RootSignature("StaticSampler(s0, mipLODBias = 15.990001)")]
void bad_root_signature_11() {}
