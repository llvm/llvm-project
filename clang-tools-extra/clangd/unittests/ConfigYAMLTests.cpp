//===-- ConfigYAMLTests.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ConfigFragment.h"
#include "ConfigTesting.h"
#include "Protocol.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

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
---
Index:
  Background: Skip
  )yaml";
  auto Results = Fragment::parseYAML(YAML, "config.yaml", Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 3u);
  EXPECT_FALSE(Results[0].If.HasUnrecognizedCondition);
  EXPECT_THAT(Results[0].If.PathMatch, ElementsAre(Val("abc")));
  EXPECT_THAT(Results[0].CompileFlags.Add, ElementsAre(Val("foo"), Val("bar")));

  EXPECT_THAT(Results[1].CompileFlags.Add, ElementsAre(Val("b\naz\n")));

  ASSERT_TRUE(Results[2].Index.Background);
  EXPECT_EQ("Skip", *Results[2].Index.Background.getValue());
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
  EXPECT_EQ(toRange(Results.front().If.PathMatch.front().Range,
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
      ElementsAre(AllOf(DiagMessage("Unknown If key UnknownCondition"),
                        DiagKind(llvm::SourceMgr::DK_Warning),
                        DiagPos(YAML.range().start), DiagRange(YAML.range())),
                  AllOf(DiagMessage("Unexpected token. Expected Key, Flow "
                                    "Entry, or Flow Mapping End."),
                        DiagKind(llvm::SourceMgr::DK_Error),
                        DiagPos(YAML.point()), DiagRange(llvm::None))));

  ASSERT_EQ(Results.size(), 1u); // invalid fragment discarded.
  EXPECT_THAT(Results.front().CompileFlags.Add, ElementsAre(Val("first")));
  EXPECT_TRUE(Results.front().If.HasUnrecognizedCondition);
}

TEST(ParseYAML, Invalid) {
  CapturedDiags Diags;
  const char *YAML = R"yaml(
If:

horrible
---
- 1
  )yaml";
  auto Results = Fragment::parseYAML(YAML, "config.yaml", Diags.callback());
  EXPECT_THAT(Diags.Diagnostics,
              ElementsAre(DiagMessage("If should be a dictionary"),
                          DiagMessage("Config should be a dictionary")));
  ASSERT_THAT(Results, IsEmpty());
}

} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
