//===-- ConfigYAMLTests.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ConfigFragment.h"
#include "Matchers.h"
#include "Protocol.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"

namespace clang {
namespace clangd {
namespace config {
template <typename T> void PrintTo(const Located<T> &V, std::ostream *OS) {
  *OS << ::testing::PrintToString(*V);
}

namespace {
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

MATCHER_P(Val, Value, "") {
  if (*arg == Value)
    return true;
  *result_listener << "value is " << *arg;
  return false;
}

Position toPosition(llvm::SMLoc L, const llvm::SourceMgr &SM) {
  auto LineCol = SM.getLineAndColumn(L);
  Position P;
  P.line = LineCol.first - 1;
  P.character = LineCol.second - 1;
  return P;
}

Range toRange(llvm::SMRange R, const llvm::SourceMgr &SM) {
  return Range{toPosition(R.Start, SM), toPosition(R.End, SM)};
}

struct CapturedDiags {
  std::function<void(const llvm::SMDiagnostic &)> callback() {
    return [this](const llvm::SMDiagnostic &D) {
      Diagnostics.emplace_back();
      Diag &Out = Diagnostics.back();
      Out.Message = D.getMessage().str();
      Out.Kind = D.getKind();
      Out.Pos.line = D.getLineNo() - 1;
      Out.Pos.character = D.getColumnNo(); // Zero-based - bug in SourceMgr?
      if (!D.getRanges().empty()) {
        const auto &R = D.getRanges().front();
        Out.Rng.emplace();
        Out.Rng->start.line = Out.Rng->end.line = Out.Pos.line;
        Out.Rng->start.character = R.first;
        Out.Rng->end.character = R.second;
      }
    };
  }
  struct Diag {
    std::string Message;
    llvm::SourceMgr::DiagKind Kind;
    Position Pos;
    llvm::Optional<Range> Rng;

    friend void PrintTo(const Diag &D, std::ostream *OS) {
      *OS << (D.Kind == llvm::SourceMgr::DK_Error ? "error: " : "warning: ")
          << D.Message << "@" << llvm::to_string(D.Pos);
    }
  };
  std::vector<Diag> Diagnostics;
};

MATCHER_P(DiagMessage, M, "") { return arg.Message == M; }
MATCHER_P(DiagKind, K, "") { return arg.Kind == K; }
MATCHER_P(DiagPos, P, "") { return arg.Pos == P; }
MATCHER_P(DiagRange, R, "") { return arg.Rng == R; }

TEST(ParseYAML, SyntacticForms) {
  CapturedDiags Diags;
  const char *YAML = R"yaml(
If:
  PathMatch:
    - 'abc'
CompileFlags: { Add: [foo, bar] }
---
CompileFlags:
  Add: |
    b
    az
  )yaml";
  auto Results = Fragment::parseYAML(YAML, "config.yaml", Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 2u);
  EXPECT_FALSE(Results.front().Condition.HasUnrecognizedCondition);
  EXPECT_THAT(Results.front().Condition.PathMatch, ElementsAre(Val("abc")));
  EXPECT_THAT(Results.front().CompileFlags.Add,
              ElementsAre(Val("foo"), Val("bar")));

  EXPECT_THAT(Results.back().CompileFlags.Add, ElementsAre(Val("b\naz\n")));
}

TEST(ParseYAML, Locations) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
If:
  PathMatch: [['???bad***regex(((']]
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  ASSERT_NE(Results.front().Source.Manager, nullptr);
  EXPECT_EQ(toRange(Results.front().Condition.PathMatch.front().Range,
                    *Results.front().Source.Manager),
            YAML.range());
}

TEST(ParseYAML, Diagnostics) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
If:
  [[UnknownCondition]]: "foo"
CompileFlags:
  Add: 'first'
---
CompileFlags: {^
)yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());

  ASSERT_THAT(
      Diags.Diagnostics,
      ElementsAre(AllOf(DiagMessage("Unknown Condition key UnknownCondition"),
                        DiagKind(llvm::SourceMgr::DK_Warning),
                        DiagPos(YAML.range().start), DiagRange(YAML.range())),
                  AllOf(DiagMessage("Unexpected token. Expected Key, Flow "
                                    "Entry, or Flow Mapping End."),
                        DiagKind(llvm::SourceMgr::DK_Error),
                        DiagPos(YAML.point()), DiagRange(llvm::None))));

  ASSERT_EQ(Results.size(), 2u);
  EXPECT_THAT(Results.front().CompileFlags.Add, ElementsAre(Val("first")));
  EXPECT_TRUE(Results.front().Condition.HasUnrecognizedCondition);
  EXPECT_THAT(Results.back().CompileFlags.Add, IsEmpty());
}

} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
