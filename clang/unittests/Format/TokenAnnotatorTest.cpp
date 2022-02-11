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

// Not really the equality, but everything we need.
static bool operator==(const FormatToken &LHS,
                       const FormatToken &RHS) noexcept {
  return LHS.Tok.getKind() == RHS.Tok.getKind() &&
         LHS.getType() == RHS.getType();
}

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

TEST_F(TokenAnnotatorTest, UnderstandsDelete) {
  auto Tokens = annotate("delete (void *)p;");
  EXPECT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[] (void *)p;");
  EXPECT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[] /*comment*/ (void *)p;");
  EXPECT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[/*comment*/] (void *)p;");
  EXPECT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete/*comment*/[] (void *)p;");
  EXPECT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);
}

TEST_F(TokenAnnotatorTest, UnderstandsRequiresClausesAndConcepts) {
  auto Tokens = annotate("template <typename T>\n"
                         "concept C = (Foo && Bar) && (Bar && Baz);");

  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[13], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[16], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template <typename T>\n"
                    "concept C = requires(T t) {\n"
                    "  { t.foo() };\n"
                    "} && Bar<T> && Baz<T>;");
  ASSERT_EQ(Tokens.size(), 35u) << Tokens;
  EXPECT_TOKEN(Tokens[23], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[28], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template<typename T>\n"
                    "requires C1<T> && (C21<T> || C22<T> && C2e<T>) && C3<T>\n"
                    "struct Foo;");
  ASSERT_EQ(Tokens.size(), 36u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::identifier, TT_Unknown);
  EXPECT_EQ(Tokens[6]->FakeLParens.size(), 1u);
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[16], tok::pipepipe, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[21], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[27], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[31], tok::greater, TT_TemplateCloser);
  EXPECT_EQ(Tokens[31]->FakeRParens, 1u);
  EXPECT_TRUE(Tokens[31]->ClosesRequiresClause);

  Tokens =
      annotate("template<typename T>\n"
               "requires (C1<T> && (C21<T> || C22<T> && C2e<T>) && C3<T>)\n"
               "struct Foo;");
  ASSERT_EQ(Tokens.size(), 38u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::identifier, TT_Unknown);
  EXPECT_EQ(Tokens[7]->FakeLParens.size(), 1u);
  EXPECT_TOKEN(Tokens[11], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[17], tok::pipepipe, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[22], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[28], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[32], tok::greater, TT_TemplateCloser);
  EXPECT_EQ(Tokens[32]->FakeRParens, 1u);
  EXPECT_TOKEN(Tokens[33], tok::r_paren, TT_Unknown);
  EXPECT_TRUE(Tokens[33]->ClosesRequiresClause);
}

TEST_F(TokenAnnotatorTest, RequiresDoesNotChangeParsingOfTheRest) {
  auto NumberOfAdditionalRequiresClauseTokens = 5u;
  auto NumberOfTokensBeforeRequires = 5u;

  auto BaseTokens = annotate("template<typename T>\n"
                             "T Pi = 3.14;");
  auto ConstrainedTokens = annotate("template<typename T>\n"
                                    "  requires Foo<T>\n"
                                    "T Pi = 3.14;");

  auto NumberOfBaseTokens = 11u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;

  BaseTokens = annotate("template<typename T>\n"
                        "struct Bar;");
  ConstrainedTokens = annotate("template<typename T>\n"
                               "  requires Foo<T>\n"
                               "struct Bar;");
  NumberOfBaseTokens = 9u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;

  BaseTokens = annotate("template<typename T>\n"
                        "struct Bar {"
                        "  T foo();\n"
                        "  T bar();\n"
                        "};");
  ConstrainedTokens = annotate("template<typename T>\n"
                               "  requires Foo<T>\n"
                               "struct Bar {"
                               "  T foo();\n"
                               "  T bar();\n"
                               "};");
  NumberOfBaseTokens = 21u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;

  BaseTokens = annotate("template<typename T>\n"
                        "Bar(T) -> Bar<T>;");
  ConstrainedTokens = annotate("template<typename T>\n"
                               "  requires Foo<T>\n"
                               "Bar(T) -> Bar<T>;");
  NumberOfBaseTokens = 16u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;

  BaseTokens = annotate("template<typename T>\n"
                        "T foo();");
  ConstrainedTokens = annotate("template<typename T>\n"
                               "  requires Foo<T>\n"
                               "T foo();");
  NumberOfBaseTokens = 11u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;

  BaseTokens = annotate("template<typename T>\n"
                        "T foo() {\n"
                        "  auto bar = baz();\n"
                        "  return bar + T{};\n"
                        "}");
  ConstrainedTokens = annotate("template<typename T>\n"
                               "  requires Foo<T>\n"
                               "T foo() {\n"
                               "  auto bar = baz();\n"
                               "  return bar + T{};\n"
                               "}");
  NumberOfBaseTokens = 26u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;

  BaseTokens = annotate("template<typename T>\n"
                        "T foo();");
  ConstrainedTokens = annotate("template<typename T>\n"
                               "T foo() requires Foo<T>;");
  NumberOfBaseTokens = 11u;
  NumberOfTokensBeforeRequires = 9u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;

  BaseTokens = annotate("template<typename T>\n"
                        "T foo() {\n"
                        "  auto bar = baz();\n"
                        "  return bar + T{};\n"
                        "}");
  ConstrainedTokens = annotate("template<typename T>\n"
                               "T foo() requires Foo<T> {\n"
                               "  auto bar = baz();\n"
                               "  return bar + T{};\n"
                               "}");
  NumberOfBaseTokens = 26u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;

  BaseTokens = annotate("template<typename T>\n"
                        "Bar(T) -> Bar<typename T::I>;");
  ConstrainedTokens = annotate("template<typename T>\n"
                               "  requires requires(T &&t) {\n"
                               "             typename T::I;\n"
                               "           }\n"
                               "Bar(T) -> Bar<typename T::I>;");
  NumberOfBaseTokens = 19u;
  NumberOfAdditionalRequiresClauseTokens = 14u;
  NumberOfTokensBeforeRequires = 5u;

  ASSERT_EQ(BaseTokens.size(), NumberOfBaseTokens) << BaseTokens;
  ASSERT_EQ(ConstrainedTokens.size(),
            NumberOfBaseTokens + NumberOfAdditionalRequiresClauseTokens)
      << ConstrainedTokens;

  for (auto I = 0u; I < NumberOfBaseTokens; ++I)
    if (I < NumberOfTokensBeforeRequires)
      EXPECT_EQ(*BaseTokens[I], *ConstrainedTokens[I]) << I;
    else
      EXPECT_EQ(*BaseTokens[I],
                *ConstrainedTokens[I + NumberOfAdditionalRequiresClauseTokens])
          << I;
}

} // namespace
} // namespace format
} // namespace clang
