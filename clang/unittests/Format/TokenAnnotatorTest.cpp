//===- unittest/Format/TokenAnnotatorTest.cpp - Formatting unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "FormatTestUtils.h"
#include "TestLexer.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {
namespace {

class TokenAnnotatorTest : public ::testing::Test {
protected:
  TokenList annotate(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    return TestLexer(Allocator, Buffers, Style).annotate(Code);
  }
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;
};

#define EXPECT_TOKEN_KIND(FormatTok, Kind)                                     \
  EXPECT_EQ((FormatTok)->Tok.getKind(), Kind) << *(FormatTok)
#define EXPECT_TOKEN_TYPE(FormatTok, Type)                                     \
  EXPECT_EQ((FormatTok)->getType(), Type) << *(FormatTok)
#define EXPECT_TOKEN(FormatTok, Kind, Type)                                    \
  do {                                                                         \
    EXPECT_TOKEN_KIND(FormatTok, Kind);                                        \
    EXPECT_TOKEN_TYPE(FormatTok, Type);                                        \
  } while (false);

TEST_F(TokenAnnotatorTest, UnderstandsUsesOfStarAndAmpInMacroDefinition) {
  // This is a regression test for mis-parsing the & after decltype as a binary
  // operator instead of a reference (when inside a macro definition).
  auto Tokens = annotate("auto x = [](const decltype(x) &ptr) {};");
  EXPECT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_decltype, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[9], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);
  // Same again with * instead of &:
  Tokens = annotate("auto x = [](const decltype(x) *ptr) {};");
  EXPECT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::star, TT_PointerOrReference);

  // Also check that we parse correctly within a macro definition:
  Tokens = annotate("#define lambda [](const decltype(x) &ptr) {}");
  EXPECT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_decltype, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[9], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);
  // Same again with * instead of &:
  Tokens = annotate("#define lambda [](const decltype(x) *ptr) {}");
  EXPECT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::star, TT_PointerOrReference);
}

TEST_F(TokenAnnotatorTest, UnderstandsClasses) {
  auto Tokens = annotate("class C {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_RecordLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsStructs) {
  auto Tokens = annotate("struct S {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_RecordLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsUnions) {
  auto Tokens = annotate("union U {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_RecordLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsEnums) {
  auto Tokens = annotate("enum E {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_RecordLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsLBracesInMacroDefinition) {
  auto Tokens = annotate("#define BEGIN NS {");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_Unknown);
}

} // namespace
} // namespace format
} // namespace clang
