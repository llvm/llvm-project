//=== LexHLSLRootSignatureTest.cpp - Lex Root Signature tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/LexHLSLRootSignature.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

// The test fixture.
class LexHLSLRootSignatureTest : public ::testing::Test {
protected:
  LexHLSLRootSignatureTest() {}

  void CheckTokens(hlsl::RootSignatureLexer &Lexer,
                   SmallVector<hlsl::RootSignatureToken> &Computed,
                   SmallVector<hlsl::TokenKind> &Expected) {
    for (unsigned I = 0, E = Expected.size(); I != E; ++I) {
      // Skip these to help with the macro generated test
      if (Expected[I] == hlsl::TokenKind::invalid ||
          Expected[I] == hlsl::TokenKind::end_of_stream)
        continue;
      hlsl::RootSignatureToken Result = Lexer.ConsumeToken();
      ASSERT_EQ(Result.Kind, Expected[I]);
      Computed.push_back(Result);
    }
    hlsl::RootSignatureToken EndOfStream = Lexer.ConsumeToken();
    ASSERT_EQ(EndOfStream.Kind, hlsl::TokenKind::end_of_stream);
    ASSERT_TRUE(Lexer.EndOfBuffer());
  }
};

// Lexing Tests

TEST_F(LexHLSLRootSignatureTest, ValidLexNumbersTest) {
  // This test will check that we can lex different number tokens
  const llvm::StringLiteral Source = R"cc(
    -42 42 +42 +2147483648
  )cc";

  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  SmallVector<hlsl::TokenKind> Expected = {
      hlsl::TokenKind::pu_minus,    hlsl::TokenKind::int_literal,
      hlsl::TokenKind::int_literal, hlsl::TokenKind::pu_plus,
      hlsl::TokenKind::int_literal, hlsl::TokenKind::pu_plus,
      hlsl::TokenKind::int_literal,
  };
  CheckTokens(Lexer, Tokens, Expected);

  // Sample negative: int component
  hlsl::RootSignatureToken IntToken = Tokens[1];
  ASSERT_EQ(IntToken.NumSpelling, "42");

  // Sample unsigned int
  IntToken = Tokens[2];
  ASSERT_EQ(IntToken.NumSpelling, "42");

  // Sample positive: int component
  IntToken = Tokens[4];
  ASSERT_EQ(IntToken.NumSpelling, "42");

  // Sample positive int that would overflow the signed representation but
  // is treated as an unsigned integer instead
  IntToken = Tokens[6];
  ASSERT_EQ(IntToken.NumSpelling, "2147483648");
}

TEST_F(LexHLSLRootSignatureTest, ValidLexAllTokensTest) {
  // This test will check that we can lex all defined tokens as defined in
  // HLSLRootSignatureTokenKinds.def, plus some additional integer variations
  const llvm::StringLiteral Source = R"cc(
    42

    b0 t43 u987 s234

    (),|=+-

    DescriptorTable

    CBV SRV UAV Sampler
    space visibility flags
    numDescriptors offset

    DESCRIPTOR_RANGE_OFFSET_APPEND

    DATA_VOLATILE
    DATA_STATIC_WHILE_SET_AT_EXECUTE
    DATA_STATIC
    DESCRIPTORS_VOLATILE
    DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS

    shader_visibility_all
    shader_visibility_vertex
    shader_visibility_hull
    shader_visibility_domain
    shader_visibility_geometry
    shader_visibility_pixel
    shader_visibility_amplification
    shader_visibility_mesh
  )cc";
  auto TokLoc = SourceLocation();
  hlsl::RootSignatureLexer Lexer(Source, TokLoc);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  SmallVector<hlsl::TokenKind> Expected = {
#define TOK(NAME) hlsl::TokenKind::NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  CheckTokens(Lexer, Tokens, Expected);
}

TEST_F(LexHLSLRootSignatureTest, ValidLexPeekTest) {
  // This test will check that we the peek api is correctly used
  const llvm::StringLiteral Source = R"cc(
    )1
  )cc";
  auto TokLoc = SourceLocation();
  hlsl::RootSignatureLexer Lexer(Source, TokLoc);

  // Test basic peek
  hlsl::RootSignatureToken Res = Lexer.PeekNextToken();
  ASSERT_EQ(Res.Kind, hlsl::TokenKind::pu_r_paren);

  // Ensure it doesn't peek past one element
  Res = Lexer.PeekNextToken();
  ASSERT_EQ(Res.Kind, hlsl::TokenKind::pu_r_paren);

  Res = Lexer.ConsumeToken();
  ASSERT_EQ(Res.Kind, hlsl::TokenKind::pu_r_paren);

  // Invoke after reseting the NextToken
  Res = Lexer.PeekNextToken();
  ASSERT_EQ(Res.Kind, hlsl::TokenKind::int_literal);

  // Ensure we can still consume the second token
  Res = Lexer.ConsumeToken();
  ASSERT_EQ(Res.Kind, hlsl::TokenKind::int_literal);

  // Ensure end of stream token
  Res = Lexer.PeekNextToken();
  ASSERT_EQ(Res.Kind, hlsl::TokenKind::end_of_stream);
}

} // anonymous namespace
