// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only %s -verify
// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only %s 2>&1 | FileCheck %s

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

// expected-error@+1 {{expected ')' to denote end of parameters, or, another valid parameter of RootConstants}}
[RootSignature("RootConstants(b0, num32BitConstants = 1, invalid)")]
void bad_root_signature_5() {}

#define MultiLineRootSignature \
 "CBV(b0)," \
 "RootConstants(num32BitConstants = 3, b0, invalid)"

// CHECK: [[@LINE-2]]:42: note: expanded from macro 'MultiLineRootSignature'
// CHECK-NEXT: [[@LINE-3]] | "RootConstants(num32BitConstants = 3, b0, invalid)"
// CHECK-NEXT:             |                                         ^
// expected-error@+1 {{expected ')' to denote end of parameters, or, another valid parameter of RootConstants}}
[RootSignature(MultiLineRootSignature)]
void bad_root_signature_6() {}

// expected-error@+1 {{expected end of stream to denote end of parameters, or, another valid parameter of RootSignature}}
[RootSignature("RootFlags() RootConstants(b0, num32BitConstants = 1)")]
void bad_root_signature_7() {}
