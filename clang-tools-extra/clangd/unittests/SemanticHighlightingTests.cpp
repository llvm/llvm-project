//==- SemanticHighlightingTests.cpp - SemanticHighlighting tests-*- C++ -* -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "SemanticHighlighting.h"
#include "TestTU.h"
#include "gmock/gmock.h"

namespace clang {
namespace clangd {
namespace {

std::vector<HighlightingToken>
makeHighlightingTokens(llvm::ArrayRef<Range> Ranges, HighlightingKind Kind) {
  std::vector<HighlightingToken> Tokens(Ranges.size());
  for (int I = 0, End = Ranges.size(); I < End; ++I) {
    Tokens[I].R = Ranges[I];
    Tokens[I].Kind = Kind;
  }

  return Tokens;
}

void checkHighlightings(llvm::StringRef Code) {
  Annotations Test(Code);
  auto AST = TestTU::withCode(Test.code()).build();
  static const std::map<HighlightingKind, std::string> KindToString{
      {HighlightingKind::Variable, "Variable"},
      {HighlightingKind::Function, "Function"}};
  std::vector<HighlightingToken> ExpectedTokens;
  for (const auto &KindString : KindToString) {
    std::vector<HighlightingToken> Toks =
        makeHighlightingTokens(Test.ranges(KindString.second), KindString.first);
    ExpectedTokens.insert(ExpectedTokens.end(), Toks.begin(), Toks.end());
  }

  auto ActualTokens = getSemanticHighlightings(AST);
  EXPECT_THAT(ActualTokens, testing::UnorderedElementsAreArray(ExpectedTokens));
}

TEST(SemanticHighlighting, GetsCorrectTokens) {
  const char *TestCases[] = {
      R"cpp(
    struct A {
      double SomeMember;
    };
    struct {
    }   $Variable[[HStruct]];
    void $Function[[foo]](int $Variable[[a]]) {
      auto $Variable[[VeryLongVariableName]] = 12312;
      A     $Variable[[aa]];
    }
  )cpp",
      R"cpp(
    void $Function[[foo]](int);
  )cpp"};
  for (const auto &TestCase : TestCases) {
    checkHighlightings(TestCase);
  }
}

} // namespace
} // namespace clangd
} // namespace clang
