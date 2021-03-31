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
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/SupportHelpers.h"
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

MATCHER_P2(PairVal, Value1, Value2, "") {
  if (*arg.first == Value1 && *arg.second == Value2)
    return true;
  *result_listener << "values are [" << *arg.first << ", " << *arg.second
                   << "]";
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
---
Diagnostics:
  ClangTidy:
    CheckOptions:
      IgnoreMacros: true
      example-check.ExampleOption: 0
  )yaml";
  auto Results = Fragment::parseYAML(YAML, "config.yaml", Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(Diags.Files, ElementsAre("config.yaml"));
  ASSERT_EQ(Results.size(), 4u);
  EXPECT_FALSE(Results[0].If.HasUnrecognizedCondition);
  EXPECT_THAT(Results[0].If.PathMatch, ElementsAre(Val("abc")));
  EXPECT_THAT(Results[0].CompileFlags.Add, ElementsAre(Val("foo"), Val("bar")));

  EXPECT_THAT(Results[1].CompileFlags.Add, ElementsAre(Val("b\naz\n")));

  ASSERT_TRUE(Results[2].Index.Background);
  EXPECT_EQ("Skip", *Results[2].Index.Background.getValue());
  EXPECT_THAT(Results[3].Diagnostics.ClangTidy.CheckOptions,
              ElementsAre(PairVal("IgnoreMacros", "true"),
                          PairVal("example-check.ExampleOption", "0")));
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

TEST(ParseYAML, ConfigDiagnostics) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
If:
  $unknown[[UnknownCondition]]: "foo"
CompileFlags:
  Add: 'first'
---
CompileFlags: {$unexpected^
)yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());

  ASSERT_THAT(
      Diags.Diagnostics,
      ElementsAre(AllOf(DiagMessage("Unknown If key 'UnknownCondition'"),
                        DiagKind(llvm::SourceMgr::DK_Warning),
                        DiagPos(YAML.range("unknown").start),
                        DiagRange(YAML.range("unknown"))),
                  AllOf(DiagMessage("Unexpected token. Expected Key, Flow "
                                    "Entry, or Flow Mapping End."),
                        DiagKind(llvm::SourceMgr::DK_Error),
                        DiagPos(YAML.point("unexpected")),
                        DiagRange(llvm::None))));

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

TEST(ParseYAML, ExternalBlock) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
Index:
  External:
    File: "foo"
    Server: ^"bar"
    MountPoint: "baz"
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_EQ(Results.size(), 1u);
  ASSERT_TRUE(Results[0].Index.External);
  EXPECT_THAT(*Results[0].Index.External.getValue()->File, Val("foo"));
  EXPECT_THAT(*Results[0].Index.External.getValue()->MountPoint, Val("baz"));
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(*Results[0].Index.External.getValue()->Server, Val("bar"));
}

TEST(ParseYAML, AllScopes) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
Completion:
  AllScopes: True
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].Completion.AllScopes, llvm::ValueIs(Val(true)));
}

TEST(ParseYAML, AllScopesWarn) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
Completion:
  AllScopes: $diagrange[[Truex]]
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  EXPECT_THAT(Diags.Diagnostics,
              ElementsAre(AllOf(DiagMessage("AllScopes should be a boolean"),
                                DiagKind(llvm::SourceMgr::DK_Warning),
                                DiagPos(YAML.range("diagrange").start),
                                DiagRange(YAML.range("diagrange")))));
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].Completion.AllScopes, testing::Eq(llvm::None));
}
} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
