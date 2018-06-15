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
#include "TestFS.h"
#include "TestTU.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

// Force the unittest URI scheme to be linked,
static int LLVM_ATTRIBUTE_UNUSED UnittestSchemeAnchorDest =
    UnittestSchemeAnchorSource;

namespace {

TEST(QualityTests, SymbolQualitySignalExtraction) {
  auto Header = TestTU::withHeaderCode(R"cpp(
    int _X;

    [[deprecated]]
    int _f() { return _X; }
  )cpp");
  auto Symbols = Header.headerSymbols();
  auto AST = Header.build();

  SymbolQualitySignals Quality;
  Quality.merge(findSymbol(Symbols, "_X"));
  EXPECT_FALSE(Quality.Deprecated);
  EXPECT_TRUE(Quality.ReservedName);
  EXPECT_EQ(Quality.References, SymbolQualitySignals().References);
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Variable);

  Symbol F = findSymbol(Symbols, "_f");
  F.References = 24; // TestTU doesn't count references, so fake it.
  Quality = {};
  Quality.merge(F);
  EXPECT_FALSE(Quality.Deprecated); // FIXME: Include deprecated bit in index.
  EXPECT_FALSE(Quality.ReservedName);
  EXPECT_EQ(Quality.References, 24u);
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Function);

  Quality = {};
  Quality.merge(CodeCompletionResult(&findDecl(AST, "_f"), /*Priority=*/42));
  EXPECT_TRUE(Quality.Deprecated);
  EXPECT_FALSE(Quality.ReservedName);
  EXPECT_EQ(Quality.References, SymbolQualitySignals().References);
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Function);

  Quality = {};
  Quality.merge(CodeCompletionResult("if"));
  EXPECT_EQ(Quality.Category, SymbolQualitySignals::Keyword);
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
  EXPECT_FLOAT_EQ(Relevance.SemaProximityScore, 1.0) << "Decl in current file";
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "header"), 42));
  EXPECT_FLOAT_EQ(Relevance.SemaProximityScore, 0.6) << "Decl from header";
  Relevance = {};
  Relevance.merge(CodeCompletionResult(&findDecl(AST, "header_main"), 42));
  EXPECT_FLOAT_EQ(Relevance.SemaProximityScore, 1.0)
      << "Current file and header";

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

  SymbolQualitySignals ReservedName;
  ReservedName.ReservedName = true;
  EXPECT_LT(ReservedName.evaluate(), Default.evaluate());

  SymbolQualitySignals WithReferences, ManyReferences;
  WithReferences.References = 10;
  ManyReferences.References = 1000;
  EXPECT_GT(WithReferences.evaluate(), Default.evaluate());
  EXPECT_GT(ManyReferences.evaluate(), WithReferences.evaluate());

  SymbolQualitySignals Keyword, Variable, Macro;
  Keyword.Category = SymbolQualitySignals::Keyword;
  Variable.Category = SymbolQualitySignals::Variable;
  Macro.Category = SymbolQualitySignals::Macro;
  EXPECT_GT(Variable.evaluate(), Default.evaluate());
  EXPECT_GT(Keyword.evaluate(), Variable.evaluate());
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
  WithProximity.SemaProximityScore = 0.2f;
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

// {a,b,c} becomes /clangd-test/a/b/c
std::string joinPaths(llvm::ArrayRef<StringRef> Parts) {
  return testPath(
      llvm::join(Parts.begin(), Parts.end(), llvm::sys::path::get_separator()));
}

static constexpr float ProximityBase = 0.7;

// Calculates a proximity score for an index symbol with declaration file
// SymPath with the given URI scheme.
float URIProximity(const FileProximityMatcher &Matcher, StringRef SymPath,
                     StringRef Scheme = "file") {
  auto U = URI::create(SymPath, Scheme);
  EXPECT_TRUE(static_cast<bool>(U)) << llvm::toString(U.takeError());
  return Matcher.uriProximity(U->toString());
}

TEST(QualityTests, URIProximityScores) {
  FileProximityMatcher Matcher(
      /*ProximityPaths=*/{joinPaths({"a", "b", "c", "d", "x"})});

  EXPECT_FLOAT_EQ(URIProximity(Matcher, joinPaths({"a", "b", "c", "d", "x"})),
                  1);
  EXPECT_FLOAT_EQ(URIProximity(Matcher, joinPaths({"a", "b", "c", "d", "y"})),
                  ProximityBase);
  EXPECT_FLOAT_EQ(URIProximity(Matcher, joinPaths({"a", "y", "z"})),
                  std::pow(ProximityBase, 5));
  EXPECT_FLOAT_EQ(
      URIProximity(Matcher, joinPaths({"a", "b", "c", "d", "e", "y"})),
      std::pow(ProximityBase, 2));
  EXPECT_FLOAT_EQ(
      URIProximity(Matcher, joinPaths({"a", "b", "m", "n", "o", "y"})),
      std::pow(ProximityBase, 5));
  EXPECT_FLOAT_EQ(
      URIProximity(Matcher, joinPaths({"a", "t", "m", "n", "o", "y"})),
      std::pow(ProximityBase, 6));
  // Note the common directory is /clang-test/
  EXPECT_FLOAT_EQ(URIProximity(Matcher, joinPaths({"m", "n", "o", "p", "y"})),
                  std::pow(ProximityBase, 7));
}

TEST(QualityTests, URIProximityScoresWithTestURI) {
  FileProximityMatcher Matcher(
      /*ProximityPaths=*/{joinPaths({"b", "c", "x"})});
  EXPECT_FLOAT_EQ(URIProximity(Matcher, joinPaths({"b", "c", "x"}), "unittest"),
                  1);
  EXPECT_FLOAT_EQ(URIProximity(Matcher, joinPaths({"b", "y"}), "unittest"),
                  std::pow(ProximityBase, 2));
  // unittest:///b/c/x vs unittest:///m/n/y. No common directory.
  EXPECT_FLOAT_EQ(URIProximity(Matcher, joinPaths({"m", "n", "y"}), "unittest"),
                  std::pow(ProximityBase, 4));
}

} // namespace
} // namespace clangd
} // namespace clang
