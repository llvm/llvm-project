//===- unittest/Format/CleanupTest.cpp - Code cleanup unit tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "../Tooling/RewriterTestContext.h"
#include "clang/Tooling/Core/Replacement.h"

#include "gtest/gtest.h"

namespace clang {
namespace format {
namespace {

class CleanupTest : public ::testing::Test {
protected:
  std::string cleanup(llvm::StringRef Code,
                      const std::vector<tooling::Range> &Ranges,
                      const FormatStyle &Style = getLLVMStyle()) {
    tooling::Replacements Replaces = format::cleanup(Style, Code, Ranges);

    std::string Result = applyAllReplacements(Code, Replaces);
    EXPECT_NE("", Result);
    return Result;
  }
};

TEST_F(CleanupTest, DeleteEmptyNamespaces) {
  std::string Code = "namespace A {\n"
                     "namespace B {\n"
                     "} // namespace B\n"
                     "} // namespace A\n\n"
                     "namespace C {\n"
                     "namespace D { int i; }\n"
                     "inline namespace E { namespace { } }\n"
                     "}";
  std::string Expected = "\n\n\n\n\nnamespace C {\n"
                         "namespace D { int i; }\n   \n"
                         "}";
  std::vector<tooling::Range> Ranges;
  Ranges.push_back(tooling::Range(28, 0));
  Ranges.push_back(tooling::Range(91, 6));
  Ranges.push_back(tooling::Range(132, 0));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, NamespaceWithSyntaxError) {
  std::string Code = "namespace A {\n"
                     "namespace B {\n" // missing r_brace
                     "} // namespace A\n\n"
                     "namespace C {\n"
                     "namespace D int i; }\n"
                     "inline namespace E { namespace { } }\n"
                     "}";
  std::string Expected = "namespace A {\n"
                         "\n\n\nnamespace C {\n"
                         "namespace D int i; }\n   \n"
                         "}";
  std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, EmptyNamespaceNotAffected) {
  std::string Code = "namespace A {\n\n"
                     "namespace {\n\n}}";
  // Even though the namespaces are empty, but the inner most empty namespace
  // block is not affected by the changed ranges.
  std::string Expected = "namespace A {\n\n"
                         "namespace {\n\n}}";
  // Set the changed range to be the second "\n".
  std::vector<tooling::Range> Ranges(1, tooling::Range(14, 0));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, EmptyNamespaceWithCommentsNoBreakBeforeBrace) {
  std::string Code = "namespace A {\n"
                     "namespace B {\n"
                     "// Yo\n"
                     "} // namespace B\n"
                     "} // namespace A\n"
                     "namespace C { // Yo\n"
                     "}";
  std::string Expected = "\n\n\n\n\n\n";
  std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, EmptyNamespaceWithCommentsBreakBeforeBrace) {
  std::string Code = "namespace A\n"
                     "/* Yo */ {\n"
                     "namespace B\n"
                     "{\n"
                     "// Yo\n"
                     "} // namespace B\n"
                     "} // namespace A\n"
                     "namespace C\n"
                     "{ // Yo\n"
                     "}\n";
  std::string Expected = "\n\n\n\n\n\n\n\n\n\n";
  std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
  FormatStyle Style = getLLVMStyle();
  Style.BraceWrapping.AfterNamespace = true;
  std::string Result = cleanup(Code, Ranges, Style);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, CtorInitializationSimpleRedundantComma) {
  std::string Code = "class A {\nA() : , {} };";
  std::string Expected = "class A {\nA()  {} };";
  std::vector<tooling::Range> Ranges;
  Ranges.push_back(tooling::Range(17, 0));
  Ranges.push_back(tooling::Range(19, 0));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);

  Code = "class A {\nA() : x(1), {} };";
  Expected = "class A {\nA() : x(1) {} };";
  Ranges.clear();
  Ranges.push_back(tooling::Range(23, 0));
  Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);

  Code = "class A {\nA() :,,,,{} };";
  Expected = "class A {\nA() {} };";
  Ranges.clear();
  Ranges.push_back(tooling::Range(15, 0));
  Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, ListSimpleRedundantComma) {
  std::string Code = "void f() { std::vector<int> v = {1,2,,,3,{4,5}}; }";
  std::string Expected = "void f() { std::vector<int> v = {1,2,3,{4,5}}; }";
  std::vector<tooling::Range> Ranges;
  Ranges.push_back(tooling::Range(40, 0));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);

  Code = "int main() { f(1,,2,3,,4);}";
  Expected = "int main() { f(1,2,3,4);}";
  Ranges.clear();
  Ranges.push_back(tooling::Range(17, 0));
  Ranges.push_back(tooling::Range(22, 0));
  Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, CtorInitializationBracesInParens) {
  std::string Code = "class A {\nA() : x({1}),, {} };";
  std::string Expected = "class A {\nA() : x({1}) {} };";
  std::vector<tooling::Range> Ranges;
  Ranges.push_back(tooling::Range(24, 0));
  Ranges.push_back(tooling::Range(26, 0));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, RedundantCommaNotInAffectedRanges) {
  std::string Code =
      "class A {\nA() : x({1}), /* comment */, { int x = 0; } };";
  std::string Expected =
      "class A {\nA() : x({1}), /* comment */, { int x = 0; } };";
  // Set the affected range to be "int x = 0", which does not intercept the
  // constructor initialization list.
  std::vector<tooling::Range> Ranges(1, tooling::Range(42, 9));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);

  Code = "class A {\nA() : x(1), {} };";
  Expected = "class A {\nA() : x(1), {} };";
  // No range. Fixer should do nothing.
  Ranges.clear();
  Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

// FIXME: delete comments too.
TEST_F(CleanupTest, CtorInitializationCommentAroundCommas) {
  // Remove redundant commas around comment.
  std::string Code = "class A {\nA() : x({1}), /* comment */, {} };";
  std::string Expected = "class A {\nA() : x({1}) /* comment */ {} };";
  std::vector<tooling::Range> Ranges;
  Ranges.push_back(tooling::Range(25, 0));
  Ranges.push_back(tooling::Range(40, 0));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);

  // Remove trailing comma and ignore comment.
  Code = "class A {\nA() : x({1}), // comment\n{} };";
  Expected = "class A {\nA() : x({1}) // comment\n{} };";
  Ranges = std::vector<tooling::Range>(1, tooling::Range(25, 0));
  Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);

  // Remove trailing comma and ignore comment.
  Code = "class A {\nA() : x({1}), // comment\n , y(1),{} };";
  Expected = "class A {\nA() : x({1}), // comment\n  y(1){} };";
  Ranges = std::vector<tooling::Range>(1, tooling::Range(38, 0));
  Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);

  // Remove trailing comma and ignore comment.
  Code = "class A {\nA() : x({1}), \n/* comment */, y(1),{} };";
  Expected = "class A {\nA() : x({1}), \n/* comment */ y(1){} };";
  Ranges = std::vector<tooling::Range>(1, tooling::Range(40, 0));
  Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);

  // Remove trailing comma and ignore comment.
  Code = "class A {\nA() : , // comment\n y(1),{} };";
  Expected = "class A {\nA() :  // comment\n y(1){} };";
  Ranges = std::vector<tooling::Range>(1, tooling::Range(17, 0));
  Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, CtorInitializerInNamespace) {
  std::string Code = "namespace A {\n"
                     "namespace B {\n" // missing r_brace
                     "} // namespace A\n\n"
                     "namespace C {\n"
                     "class A { A() : x(0),, {} };\n"
                     "inline namespace E { namespace { } }\n"
                     "}";
  std::string Expected = "namespace A {\n"
                         "\n\n\nnamespace C {\n"
                         "class A { A() : x(0) {} };\n   \n"
                         "}";
  std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
  std::string Result = cleanup(Code, Ranges);
  EXPECT_EQ(Expected, Result);
}

class CleanUpReplacementsTest : public ::testing::Test {
protected:
  tooling::Replacement createReplacement(SourceLocation Start, unsigned Length,
                                         llvm::StringRef ReplacementText) {
    return tooling::Replacement(Context.Sources, Start, Length,
                                ReplacementText);
  }

  RewriterTestContext Context;
};

TEST_F(CleanUpReplacementsTest, FixOnlyAffectedCodeAfterReplacements) {
  std::string Code = "namespace A {\n"
                     "namespace B {\n"
                     "  int x;\n"
                     "} // namespace B\n"
                     "} // namespace A\n"
                     "\n"
                     "namespace C {\n"
                     "namespace D { int i; }\n"
                     "inline namespace E { namespace { int y; } }\n"
                     "int x=     0;"
                     "}";
  std::string Expected = "\n\nnamespace C {\n"
                         "namespace D { int i; }\n\n"
                         "int x=     0;"
                         "}";
  FileID ID = Context.createInMemoryFile("fix.cpp", Code);
  tooling::Replacements Replaces;
  Replaces.insert(tooling::Replacement(Context.Sources,
                                       Context.getLocation(ID, 3, 3), 6, ""));
  Replaces.insert(tooling::Replacement(Context.Sources,
                                       Context.getLocation(ID, 9, 34), 6, ""));

  format::FormatStyle Style = format::getLLVMStyle();
  auto FinalReplaces = formatReplacements(
      Code, cleanupAroundReplacements(Code, Replaces, Style), Style);
  EXPECT_EQ(Expected, applyAllReplacements(Code, FinalReplaces));
}

} // end namespace
} // end namespace format
} // end namespace clang
