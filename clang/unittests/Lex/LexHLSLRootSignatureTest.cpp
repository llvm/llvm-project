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
using TokenKind = hlsl::RootSignatureToken::Kind;

namespace {

// The test fixture.
class LexHLSLRootSignatureTest : public ::testing::Test {
protected:
  LexHLSLRootSignatureTest() {}

  void checkTokens(hlsl::RootSignatureLexer &Lexer,
                   SmallVector<hlsl::RootSignatureToken> &Computed,
                   SmallVector<TokenKind> &Expected) {
    for (unsigned I = 0, E = Expected.size(); I != E; ++I) {
      // Skip these to help with the macro generated test
      if (Expected[I] == TokenKind::invalid ||
          Expected[I] == TokenKind::end_of_stream)
        continue;
      hlsl::RootSignatureToken Result = Lexer.consumeToken();
      ASSERT_EQ(Result.TokKind, Expected[I]);
      Computed.push_back(Result);
    }
    hlsl::RootSignatureToken EndOfStream = Lexer.consumeToken();
    ASSERT_EQ(EndOfStream.TokKind, TokenKind::end_of_stream);
    ASSERT_TRUE(Lexer.isEndOfBuffer());
  }
};

// Lexing Tests

TEST_F(LexHLSLRootSignatureTest, ValidLexNumbersTest) {
  // This test will check that we can lex different number tokens
  const llvm::StringLiteral Source = R"cc(
    -42 42 +42 +2147483648
    42. 4.2 .42
    42f 4.2F
    .42e+3 4.2E-12
    42.e+10f
  )cc";

  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  SmallVector<TokenKind> Expected = {
      TokenKind::pu_minus,      TokenKind::int_literal,
      TokenKind::int_literal,   TokenKind::pu_plus,
      TokenKind::int_literal,   TokenKind::pu_plus,
      TokenKind::int_literal,   TokenKind::float_literal,
      TokenKind::float_literal, TokenKind::float_literal,
      TokenKind::float_literal, TokenKind::float_literal,
      TokenKind::float_literal, TokenKind::float_literal,
      TokenKind::float_literal,
  };
  checkTokens(Lexer, Tokens, Expected);

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

  // Sample decimal end
  hlsl::RootSignatureToken FloatToken = Tokens[7];
  ASSERT_EQ(FloatToken.NumSpelling, "42.");

  // Sample decimal middle
  FloatToken = Tokens[8];
  ASSERT_EQ(FloatToken.NumSpelling, "4.2");

  // Sample decimal start
  FloatToken = Tokens[9];
  ASSERT_EQ(FloatToken.NumSpelling, ".42");

  // Sample float lower
  FloatToken = Tokens[10];
  ASSERT_EQ(FloatToken.NumSpelling, "42f");

  // Sample float upper
  FloatToken = Tokens[11];
  ASSERT_EQ(FloatToken.NumSpelling, "4.2F");

  // Sample exp +
  FloatToken = Tokens[12];
  ASSERT_EQ(FloatToken.NumSpelling, ".42e+3");

  // Sample exp -
  FloatToken = Tokens[13];
  ASSERT_EQ(FloatToken.NumSpelling, "4.2E-12");

  // Sample all combined
  FloatToken = Tokens[14];
  ASSERT_EQ(FloatToken.NumSpelling, "42.e+10f");
}

TEST_F(LexHLSLRootSignatureTest, ValidLexAllTokensTest) {
  // This test will check that we can lex all defined tokens as defined in
  // HLSLRootSignatureTokenKinds.def, plus some additional integer variations
  const llvm::StringLiteral Source = R"cc(
    42 42.0f

    b0 t43 u987 s234

    (),|=+-

    RootSignature

    DescriptorTable

    CBV SRV UAV Sampler
    space visibility flags
    numDescriptors offset

    unbounded
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
  SmallVector<TokenKind> Expected = {
#define TOK(NAME, SPELLING) TokenKind::NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  checkTokens(Lexer, Tokens, Expected);
}

TEST_F(LexHLSLRootSignatureTest, ValidCaseInsensitiveKeywordsTest) {
  // This test will check that we can lex keywords in an case-insensitive
  // manner
  const llvm::StringLiteral Source = R"cc(
    DeScRiPtOrTaBlE

    CBV srv UAV sampler
    SPACE visibility FLAGS
    numDescriptors OFFSET
  )cc";
  auto TokLoc = SourceLocation();
  hlsl::RootSignatureLexer Lexer(Source, TokLoc);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  SmallVector<TokenKind> Expected = {
      TokenKind::kw_DescriptorTable,
      TokenKind::kw_CBV,
      TokenKind::kw_SRV,
      TokenKind::kw_UAV,
      TokenKind::kw_Sampler,
      TokenKind::kw_space,
      TokenKind::kw_visibility,
      TokenKind::kw_flags,
      TokenKind::kw_numDescriptors,
      TokenKind::kw_offset,
  };

  checkTokens(Lexer, Tokens, Expected);
}

TEST_F(LexHLSLRootSignatureTest, ValidLexPeekTest) {
  // This test will check that we the peek api is correctly used
  const llvm::StringLiteral Source = R"cc(
    )1
  )cc";
  auto TokLoc = SourceLocation();
  hlsl::RootSignatureLexer Lexer(Source, TokLoc);

  // Test basic peek
  hlsl::RootSignatureToken Res = Lexer.peekNextToken();
  ASSERT_EQ(Res.TokKind, TokenKind::pu_r_paren);

  // Ensure it doesn't peek past one element
  Res = Lexer.peekNextToken();
  ASSERT_EQ(Res.TokKind, TokenKind::pu_r_paren);

  Res = Lexer.consumeToken();
  ASSERT_EQ(Res.TokKind, TokenKind::pu_r_paren);

  // Invoke after reseting the NextToken
  Res = Lexer.peekNextToken();
  ASSERT_EQ(Res.TokKind, TokenKind::int_literal);

  // Ensure we can still consume the second token
  Res = Lexer.consumeToken();
  ASSERT_EQ(Res.TokKind, TokenKind::int_literal);

  // Ensure end of stream token
  Res = Lexer.peekNextToken();
  ASSERT_EQ(Res.TokKind, TokenKind::end_of_stream);
}

} // anonymous namespace
