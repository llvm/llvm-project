// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

// This file mirrors the diagnostics testing in ParseHLSLRootSignatureTest.cpp
// to verify that the correct diagnostics strings are output

// Lexer related tests

#define InvalidToken \
  "DescriptorTable( " \
  "  invalid " \
  ")"

[RootSignature(InvalidToken)] // expected-error {{unable to lex a valid Root Signature token}}
void bad_root_signature_1() {}

#define InvalidEmptyNumber \
  "DescriptorTable( " \
  "  CBV(t32, space = +) " \
  ")"

[RootSignature(InvalidEmptyNumber)] // expected-error {{expected number literal is not a supported number literal of unsigned integer or integer}}
void bad_root_signature_2() {}

#define InvalidOverflowNumber \
  "DescriptorTable( " \
  "  CBV(t32, space = 98273498327498273487) " \
  ")"

[RootSignature(InvalidOverflowNumber)] // expected-error {{provided unsigned integer literal '98273498327498273487' that overflows the maximum of 32 bits}}
void bad_root_signature_3() {}

#define InvalidEOS \
  "DescriptorTable( "

// Parser related tests

[RootSignature(InvalidEOS)] // expected-error {{unexpected end to token stream}}
void bad_root_signature_4() {}

#define InvalidTokenKind \
  "DescriptorTable( " \
  "  DescriptorTable()" \
  ")"

[RootSignature(InvalidTokenKind)] // expected-error {{expected the one of the following token kinds 'CBV, SRV, UAV, Sampler'}}
void bad_root_signature_5() {}

#define InvalidRepeat \
  "DescriptorTable( " \
  "  CBV(t0, space = 1, space = 2)" \
  ")"

[RootSignature(InvalidRepeat)] // expected-error {{specified the same parameter 'space' multiple times}}
void bad_root_signature_6() {}
