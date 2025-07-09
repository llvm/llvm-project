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

// expected-error@+1 {{invalid parameter of RootSignature}}
[RootSignature("DescriptorTable(), invalid")]
void bad_root_signature_4() {}

// expected-error@+1 {{invalid parameter of RootConstants}}
[RootSignature("RootConstants(b0, num32BitConstants = 1, invalid)")]
void bad_root_signature_5() {}

#define MultiLineRootSignature \
 "CBV(b0)," \
 "RootConstants(num32BitConstants = 3, b0, invalid)"

// CHECK: [[@LINE-2]]:44: note: expanded from macro 'MultiLineRootSignature'
// CHECK-NEXT: [[@LINE-3]] | "RootConstants(num32BitConstants = 3, b0, invalid)"
// CHECK-NEXT:             |                                           ^
// expected-error@+1 {{invalid parameter of RootConstants}}
[RootSignature(MultiLineRootSignature)]
void bad_root_signature_6() {}

// expected-error@+1 {{expected end of stream}}
[RootSignature("RootFlags() RootConstants(b0, num32BitConstants = 1)")]
void bad_root_signature_7() {}

// expected-error@+1 {{invalid parameter of RootConstants}}
[RootSignature("RootConstants(b0, num32BitConstantsTypo = 1))")]
void bad_root_signature_8() {}

// expected-error@+1 {{invalid parameter of UAV}}
[RootSignature("UAV(b3")]
void bad_root_signature_9() {}

// expected-error@+1 {{invalid parameter of SRV}}
[RootSignature("DescriptorTable(SRV(s1, invalid))")]
void bad_root_signature_10() {}

// expected-error@+1 {{invalid parameter of DescriptorTable}}
[RootSignature("DescriptorTable(invalid))")]
void bad_root_signature_11() {}
