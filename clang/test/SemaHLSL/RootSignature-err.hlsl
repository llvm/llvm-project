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

// expected-error@+1 {{expected ')' or ','}}
[RootSignature("RootConstants(b0 num32BitConstants = 1)")]
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

// expected-error@+1 {{expected end of stream or ','}}
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

// expected-error@+1 {{expected integer literal after '+'}}
[RootSignature("CBV(space = +invalid))")]
void bad_root_signature_12() {}

// expected-error@+1 {{expected integer literal after '='}}
[RootSignature("CBV(space = invalid))")]
void bad_root_signature_13() {}

// expected-error@+1 {{expected '(' after UAV}}
[RootSignature("UAV invalid")]
void bad_root_signature_14() {}

// expected-error@+1 {{invalid value of visibility}}
[RootSignature("StaticSampler(s0, visibility = visibility_typo)")]
void bad_root_signature_15() {}

// expected-error@+1 {{invalid value of filter}}
[RootSignature("StaticSampler(s0, filter = filter_typo)")]
void bad_root_signature_16() {}

// expected-error@+1 {{invalid value of addressU}}
[RootSignature("StaticSampler(s0, addressU = addressU_typo)")]
void bad_root_signature_17() {}

// expected-error@+1 {{invalid value of addressV}}
[RootSignature("StaticSampler(s0, addressV = addressV_typo)")]
void bad_root_signature_18() {}

// expected-error@+1 {{invalid value of comparisonFunc}}
[RootSignature("StaticSampler(s0, comparisonFunc = comparisonFunc_typo)")]
void bad_root_signature_19() {}

// expected-error@+1 {{invalid value of borderColor}}
[RootSignature("StaticSampler(s0, borderColor = borderColor_typo)")]
void bad_root_signature_20() {}

// expected-error@+1 {{invalid value of flags}}
[RootSignature("CBV(b0, flags = DATA_VOLATILE | root_descriptor_flag_typo)")]
void bad_root_signature_21() {}

// expected-error@+1 {{invalid value of flags}}
[RootSignature("DescriptorTable(SRV(t0, flags = descriptor_range_flag_typo)")]
void bad_root_signature_22() {}

// expected-error@+1 {{invalid value of RootFlags}}
[RootSignature("RootFlags(local_root_signature | root_flag_typo)")]
void bad_root_signature_23() {}

// Basic validation of register value and space

// expected-error@+2 {{value must be in the range [0, 4294967294]}}
// expected-error@+1 {{value must be in the range [0, 4294967279]}}
[RootSignature("CBV(b4294967295, space = 4294967280)")]
void basic_validation_0() {}

// expected-error@+2 {{value must be in the range [0, 4294967294]}}
// expected-error@+1 {{value must be in the range [0, 4294967279]}}
[RootSignature("RootConstants(b4294967295, space = 4294967280, num32BitConstants = 1)")]
void basic_validation_1() {}

// expected-error@+2 {{value must be in the range [0, 4294967294]}}
// expected-error@+1 {{value must be in the range [0, 4294967279]}}
[RootSignature("StaticSampler(s4294967295, space = 4294967280)")]
void basic_validation_2() {}

// expected-error@+2 {{value must be in the range [0, 4294967294]}}
// expected-error@+1 {{value must be in the range [0, 4294967279]}}
[RootSignature("DescriptorTable(SRV(t4294967295, space = 4294967280))")]
void basic_validation_3() {}

// expected-error@+2 {{value must be in the range [1, 4294967294]}}
// expected-error@+1 {{value must be in the range [1, 4294967294]}}
[RootSignature("DescriptorTable(UAV(u0, numDescriptors = 0), Sampler(s0, numDescriptors = 0))")]
void basic_validation_4() {}

// expected-error@+2 {{value must be in the range [0, 16]}}
// expected-error@+1 {{value must be in the range [-16.00, 15.99]}}
[RootSignature("StaticSampler(s0, maxAnisotropy = 17, mipLODBias = -16.000001)")]
void basic_validation_5() {}

// expected-error@+1 {{value must be in the range [-16.00, 15.99]}}
[RootSignature("StaticSampler(s0, mipLODBias = 15.990001)")]
void basic_validation_6() {}
