//===-- SourceCodeTests.cpp  ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Evaluating scoring functions isn't a great fit for assert-based tests.
// For interesting cases, both exact scores and "X beats Y" are too brittle to
// make good hard assertions.
//
// Here we test the signal extraction and sanity-check that signals point in
// the right direction. This should be supplemented by quality metrics which
// we can compute from a corpus of queries and preferred rankings.
//
//===----------------------------------------------------------------------===//

#include "Quality.h"
#include "TestTU.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(QualityTests, SymbolQualitySignalExtraction) {
  auto Header = TestTU::withHeaderCode(R"cpp(
    int x;

    [[deprecated]]
    int f() { return x; }
  )cpp");
  auto Symbols = Header.headerSymbols();
  auto AST = Header.build();

  SymbolQualitySignals Quality;
  Quality.merge(findSymbol(Symbols, "x"));
  EXPECT_FALSE(Quality.Deprecated);
  EXPECT_EQ(Quality.References, SymbolQualitySignals().References);
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Variable);

  Symbol F = findSymbol(Symbols, "f");
  F.References = 24; // TestTU doesn't count references, so fake it.
  Quality = {};
  Quality.merge(F);
  EXPECT_FALSE(Quality.Deprecated); // FIXME: Include deprecated bit in index.
  EXPECT_EQ(Quality.References, 24u);
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Function);

  Quality = {};
  Quality.merge(CodeCompletionResult(&findDecl(AST, "f"), /*Priority=*/42));
  EXPECT_TRUE(Quality.Deprecated);
  EXPECT_EQ(Quality.References, SymbolQualitySignals().References);
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Function);
}

TEST(QualityTests, SymbolRelevanceSignalExtraction) {
  TestTU Test;
  Test.HeaderCode = R"cpp(
    int header();
    int header_main();
    )cpp";
  Test.Code = R"cpp(
    int ::header_main() {}
    int main();

    [[deprecated]]
    int deprecated() { return 0; }

    namespace { struct X { void y() { int z; } }; }
  )cpp";
  auto AST = Test.build();

  SymbolRelevanceSignals Relevance;
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "deprecated"),
                                       /*Priority=*/42, nullptr, false,
                                       /*Accessible=*/false));
  EXPECT_EQ(Relevance.NameMatch, SymbolRelevanceSignals().NameMatch);
  EXPECT_TRUE(Relevance.Forbidden);
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::GlobalScope);

  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "main"), 42));
  EXPECT_FLOAT_EQ(Relevance.ProximityScore, 1.0) << "Decl in current file";
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "header"), 42));
  EXPECT_FLOAT_EQ(Relevance.ProximityScore, 0.0) << "Decl from header";
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "header_main"), 42));
  EXPECT_FLOAT_EQ(Relevance.ProximityScore, 1.0) << "Current file and header";

  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findAnyDecl(AST, "X"), 42));
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::FileScope);
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findAnyDecl(AST, "y"), 42));
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::ClassScope);
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findAnyDecl(AST, "z"), 42));
  EXPECT_EQ(Relevance.Scope, SymbolRelevanceSignals::FunctionScope);
}

// Do the signals move the scores in the direction we expect?
TEST(QualityTests, SymbolQualitySignalsSanity) {
  SymbolQualitySignals Default;
  EXPECT_EQ(Default.evaluate(), 1);

  SymbolQualitySignals Deprecated;
  Deprecated.Deprecated = true;
  EXPECT_LT(Deprecated.evaluate(), Default.evaluate());

  SymbolQualitySignals WithReferences, ManyReferences;
  WithReferences.References = 10;
  ManyReferences.References = 1000;
  EXPECT_GT(WithReferences.evaluate(), Default.evaluate());
  EXPECT_GT(ManyReferences.evaluate(), WithReferences.evaluate());

  SymbolQualitySignals Variable, Macro;
  Variable.Category = SymbolQualitySignals::Variable;
  Macro.Category = SymbolQualitySignals::Macro;
  EXPECT_GT(Variable.evaluate(), Default.evaluate());
  EXPECT_LT(Macro.evaluate(), Default.evaluate());
}

TEST(QualityTests, SymbolRelevanceSignalsSanity) {
  SymbolRelevanceSignals Default;
  EXPECT_EQ(Default.evaluate(), 1);

  SymbolRelevanceSignals Forbidden;
  Forbidden.Forbidden = true;
  EXPECT_LT(Forbidden.evaluate(), Default.evaluate());

  SymbolRelevanceSignals PoorNameMatch;
  PoorNameMatch.NameMatch = 0.2f;
  EXPECT_LT(PoorNameMatch.evaluate(), Default.evaluate());

  SymbolRelevanceSignals WithProximity;
  WithProximity.ProximityScore = 0.2f;
  EXPECT_GT(WithProximity.evaluate(), Default.evaluate());

  SymbolRelevanceSignals Scoped;
  Scoped.Scope = SymbolRelevanceSignals::FileScope;
  EXPECT_EQ(Scoped.evaluate(), Default.evaluate());
  Scoped.Query = SymbolRelevanceSignals::CodeComplete;
  EXPECT_GT(Scoped.evaluate(), Default.evaluate());
}

TEST(QualityTests, SortText) {
  EXPECT_LT(sortText(std::numeric_limits<float>::infinity()), sortText(1000.2f));
  EXPECT_LT(sortText(1000.2f), sortText(1));
  EXPECT_LT(sortText(1), sortText(0.3f));
  EXPECT_LT(sortText(0.3f), sortText(0));
  EXPECT_LT(sortText(0), sortText(-10));
  EXPECT_LT(sortText(-10), sortText(-std::numeric_limits<float>::infinity()));

  EXPECT_LT(sortText(1, "z"), sortText(0, "a"));
  EXPECT_LT(sortText(0, "a"), sortText(0, "z"));
}

} // namespace
} // namespace clangd
} // namespace clang
