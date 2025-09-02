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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace config {

// PrintTo is a magic identifier of GTest
// NOLINTNEXTLINE (readability-identifier-naming)
template <typename T> void PrintTo(const Located<T> &V, std::ostream *OS) {
  *OS << ::testing::PrintToString(*V);
}

namespace {
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

MATCHER_P(val, Value, "") {
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
  UnusedIncludes: Strict
  )yaml";
  auto Results = Fragment::parseYAML(YAML, "config.yaml", Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(Diags.Files, ElementsAre("config.yaml"));
  ASSERT_EQ(Results.size(), 4u);
  EXPECT_FALSE(Results[0].If.HasUnrecognizedCondition);
  EXPECT_THAT(Results[0].If.PathMatch, ElementsAre(val("abc")));
  EXPECT_THAT(Results[0].CompileFlags.Add, ElementsAre(val("foo"), val("bar")));

  EXPECT_THAT(Results[1].CompileFlags.Add, ElementsAre(val("b\naz\n")));

  ASSERT_TRUE(Results[2].Index.Background);
  EXPECT_EQ("Skip", **Results[2].Index.Background);
  EXPECT_THAT(Results[3].Diagnostics.ClangTidy.CheckOptions,
              ElementsAre(PairVal("IgnoreMacros", "true"),
                          PairVal("example-check.ExampleOption", "0")));
  EXPECT_TRUE(Results[3].Diagnostics.UnusedIncludes);
  EXPECT_EQ("Strict", **Results[3].Diagnostics.UnusedIncludes);
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
      ElementsAre(AllOf(diagMessage("Unknown If key 'UnknownCondition'"),
                        diagKind(llvm::SourceMgr::DK_Warning),
                        diagPos(YAML.range("unknown").start),
                        diagRange(YAML.range("unknown"))),
                  AllOf(diagMessage("Unexpected token. Expected Key, Flow "
                                    "Entry, or Flow Mapping End."),
                        diagKind(llvm::SourceMgr::DK_Error),
                        diagPos(YAML.point("unexpected")),
                        diagRange(std::nullopt))));

  ASSERT_EQ(Results.size(), 1u); // invalid fragment discarded.
  EXPECT_THAT(Results.front().CompileFlags.Add, ElementsAre(val("first")));
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
              ElementsAre(diagMessage("If should be a dictionary"),
                          diagMessage("Config should be a dictionary")));
  ASSERT_THAT(Results, IsEmpty());
}

TEST(ParseYAML, ExternalBlockNone) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
Index:
  External: None
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  ASSERT_TRUE(Results[0].Index.External);
  EXPECT_FALSE((*Results[0].Index.External)->File.has_value());
  EXPECT_FALSE((*Results[0].Index.External)->MountPoint.has_value());
  EXPECT_FALSE((*Results[0].Index.External)->Server.has_value());
  EXPECT_THAT(*(*Results[0].Index.External)->IsNone, testing::Eq(true));
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
  EXPECT_THAT(*(*Results[0].Index.External)->File, val("foo"));
  EXPECT_THAT(*(*Results[0].Index.External)->MountPoint, val("baz"));
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(*(*Results[0].Index.External)->Server, val("bar"));
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
  EXPECT_THAT(Results[0].Completion.AllScopes, llvm::ValueIs(val(true)));
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
              ElementsAre(AllOf(diagMessage("AllScopes should be a boolean"),
                                diagKind(llvm::SourceMgr::DK_Warning),
                                diagPos(YAML.range("diagrange").start),
                                diagRange(YAML.range("diagrange")))));
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].Completion.AllScopes, testing::Eq(std::nullopt));
}

TEST(ParseYAML, CodePatterns) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
    Completion:
      CodePatterns: None
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].Completion.CodePatterns, llvm::ValueIs(val("None")));
}

TEST(ParseYAML, Hover) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
Hover:
  ShowAKA: True
  MacroContentsLimit: 4096
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].Hover.ShowAKA, llvm::ValueIs(val(true)));
  EXPECT_THAT(Results[0].Hover.MacroContentsLimit, llvm::ValueIs(val(4096U)));
}

TEST(ParseYAML, InlayHints) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
InlayHints:
  Enabled: No
  ParameterNames: Yes
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].InlayHints.Enabled, llvm::ValueIs(val(false)));
  EXPECT_THAT(Results[0].InlayHints.ParameterNames, llvm::ValueIs(val(true)));
  EXPECT_EQ(Results[0].InlayHints.DeducedTypes, std::nullopt);
}

TEST(ParseYAML, SemanticTokens) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
SemanticTokens:
  DisabledKinds: [ Operator, InactiveCode]
  DisabledModifiers: Readonly
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].SemanticTokens.DisabledKinds,
              ElementsAre(val("Operator"), val("InactiveCode")));
  EXPECT_THAT(Results[0].SemanticTokens.DisabledModifiers,
              ElementsAre(val("Readonly")));
}

TEST(ParseYAML, IncludesIgnoreHeader) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
Diagnostics:
  Includes:
    IgnoreHeader: [foo, bar]
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].Diagnostics.Includes.IgnoreHeader,
              ElementsAre(val("foo"), val("bar")));
}

TEST(ParseYAML, IncludesAnalyzeAngledIncludes) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
Diagnostics:
  Includes:
    AnalyzeAngledIncludes: true
  )yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].Diagnostics.Includes.AnalyzeAngledIncludes,
              llvm::ValueIs(val(true)));
}

TEST(ParseYAML, Style) {
  CapturedDiags Diags;
  Annotations YAML(R"yaml(
Style:
  FullyQualifiedNamespaces: [foo, bar]
  AngledHeaders: ["foo", "bar"]
  QuotedHeaders: ["baz", "baar"])yaml");
  auto Results =
      Fragment::parseYAML(YAML.code(), "config.yaml", Diags.callback());
  ASSERT_THAT(Diags.Diagnostics, IsEmpty());
  ASSERT_EQ(Results.size(), 1u);
  EXPECT_THAT(Results[0].Style.FullyQualifiedNamespaces,
              ElementsAre(val("foo"), val("bar")));
  EXPECT_THAT(Results[0].Style.AngledHeaders,
              ElementsAre(val("foo"), val("bar")));
  EXPECT_THAT(Results[0].Style.QuotedHeaders,
              ElementsAre(val("baz"), val("baar")));
}
} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
