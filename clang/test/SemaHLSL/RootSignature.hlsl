// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only %s -verify

// expected-no-diagnostics

// Test that we have consistent behaviour for comma parsing. Namely:
// - a single trailing comma is allowed after any parameter
// - a trailing comma is not required

[RootSignature("CBV(b0, flags = DATA_VOLATILE,), DescriptorTable(Sampler(s0,),),")]
void maximum_commas() {}

[RootSignature("CBV(b0, flags = DATA_VOLATILE), DescriptorTable(Sampler(s0))")]
void minimal_commas() {}
