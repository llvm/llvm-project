// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

// This file mirrors the diagnostics testing in ParseHLSLRootSignatureTest.cpp
// to verify that the correct diagnostics strings are output

// Lexer related tests

#define InvalidToken \
  "DescriptorTable( " \
  "  invalid " \
  ")"

[RootSignature(InvalidToken)] // expected-error {{expected one of the following token kinds: CBV, SRV, UAV, Sampler}}
void bad_root_signature_1() {}

#define InvalidEmptyNumber \
  "DescriptorTable( " \
  "  CBV(t32, space = +) " \
  ")"

// expected-error@+1 {{expected the following token kind: integer literal}}
[RootSignature(InvalidEmptyNumber)]
void bad_root_signature_2() {}

#define InvalidOverflowNumber \
  "DescriptorTable( " \
  "  CBV(t32, space = 98273498327498273487) " \
  ")"

// expected-error@+1 {{integer literal '98273498327498273487' is too large to be represented in a 32-bit integer type}}
[RootSignature(InvalidOverflowNumber)]
void bad_root_signature_3() {}

// Parser related tests

#define InvalidEOS \
  "DescriptorTable( " \
  "  CBV("

[RootSignature(InvalidEOS)] // expected-error {{expected one of the following token kinds: b register, t register, u register, s register}}
void bad_root_signature_4() {}

#define InvalidTokenKind \
  "DescriptorTable( " \
  "  SRV(s0, CBV())" \
  ")"

[RootSignature(InvalidTokenKind)] // expected-error {{expected one of the following token kinds: offset, numDescriptors, space, flags}}
void bad_root_signature_5() {}

#define InvalidRepeat \
  "DescriptorTable( " \
  "  CBV(t0, space = 1, space = 2)" \
  ")"

[RootSignature(InvalidRepeat)] // expected-error {{specified the same parameter 'space' multiple times}}
void bad_root_signature_6() {}
