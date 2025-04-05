// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

// Attr test

[RootSignature()] // expected-error {{'RootSignature' attribute takes one argument}}
void bad_root_signature_0() {}

[RootSignature("Arg1", "Arg2")] // expected-error {{'RootSignature' attribute takes one argument}}
void bad_root_signature_1() {}
