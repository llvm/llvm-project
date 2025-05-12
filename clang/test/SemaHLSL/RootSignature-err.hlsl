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
