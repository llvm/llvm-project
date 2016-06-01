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
  tooling::Replacement createReplacement(unsigned Offset, unsigned Length,
                                         StringRef Text) {
    return tooling::Replacement(FileName, Offset, Length, Text);
  }

  tooling::Replacement createInsertion(StringRef HeaderName) {
    return createReplacement(UINT_MAX, 0, HeaderName);
  }

  inline std::string apply(StringRef Code,
                           const tooling::Replacements Replaces) {
    return applyAllReplacements(
        Code, cleanupAroundReplacements(Code, Replaces, Style));
  }

  inline std::string formatAndApply(StringRef Code,
                                    const tooling::Replacements Replaces) {
    return applyAllReplacements(
        Code,
        formatReplacements(
            Code, cleanupAroundReplacements(Code, Replaces, Style), Style));
  }

  int getOffset(StringRef Code, int Line, int Column) {
    RewriterTestContext Context;
    FileID ID = Context.createInMemoryFile(FileName, Code);
    auto DecomposedLocation =
        Context.Sources.getDecomposedLoc(Context.getLocation(ID, Line, Column));
    return DecomposedLocation.second;
  }

  const std::string FileName = "fix.cpp";
  FormatStyle Style = getLLVMStyle();
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
  tooling::Replacements Replaces = {
      createReplacement(getOffset(Code, 3, 3), 6, ""),
      createReplacement(getOffset(Code, 9, 34), 6, "")};

  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, NoExistingIncludeWithoutDefine) {
  std::string Code = "int main() {}";
  std::string Expected = "#include \"a.h\"\n"
                         "int main() {}";
  tooling::Replacements Replaces = {createInsertion("#include \"a.h\"")};
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, NoExistingIncludeWithDefine) {
  std::string Code = "#ifndef A_H\n"
                     "#define A_H\n"
                     "class A {};\n"
                     "#define MMM 123\n"
                     "#endif";
  std::string Expected = "#ifndef A_H\n"
                         "#define A_H\n"
                         "#include \"b.h\"\n"
                         "class A {};\n"
                         "#define MMM 123\n"
                         "#endif";

  tooling::Replacements Replaces = {createInsertion("#include \"b.h\"")};
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertBeforeCategoryWithLowerPriority) {
  std::string Code = "#ifndef A_H\n"
                     "#define A_H\n"
                     "\n"
                     "\n"
                     "\n"
                     "#include <vector>\n"
                     "class A {};\n"
                     "#define MMM 123\n"
                     "#endif";
  std::string Expected = "#ifndef A_H\n"
                         "#define A_H\n"
                         "\n"
                         "\n"
                         "\n"
                         "#include \"a.h\"\n"
                         "#include <vector>\n"
                         "class A {};\n"
                         "#define MMM 123\n"
                         "#endif";

  tooling::Replacements Replaces = {createInsertion("#include \"a.h\"")};
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertAfterMainHeader) {
  std::string Code = "#include \"fix.h\"\n"
                     "\n"
                     "int main() {}";
  std::string Expected = "#include \"fix.h\"\n"
                         "#include <a>\n"
                         "\n"
                         "int main() {}";
  tooling::Replacements Replaces = {createInsertion("#include <a>")};
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp);
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertBeforeSystemHeaderLLVM) {
  std::string Code = "#include <memory>\n"
                     "\n"
                     "int main() {}";
  std::string Expected = "#include \"z.h\"\n"
                         "#include <memory>\n"
                         "\n"
                         "int main() {}";
  tooling::Replacements Replaces = {createInsertion("#include \"z.h\"")};
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertAfterSystemHeaderGoogle) {
  std::string Code = "#include <memory>\n"
                     "\n"
                     "int main() {}";
  std::string Expected = "#include <memory>\n"
                         "#include \"z.h\"\n"
                         "\n"
                         "int main() {}";
  tooling::Replacements Replaces = {createInsertion("#include \"z.h\"")};
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp);
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertOneIncludeLLVMStyle) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "#include \"a.h\"\n"
                     "#include \"b.h\"\n"
                     "#include \"clang/Format/Format.h\"\n"
                     "#include <memory>\n";
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"d.h\"\n"
                         "#include \"clang/Format/Format.h\"\n"
                         "#include \"llvm/x/y.h\"\n"
                         "#include <memory>\n";
  tooling::Replacements Replaces = {createInsertion("#include \"d.h\""),
                                    createInsertion("#include \"llvm/x/y.h\"")};
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertMultipleIncludesLLVMStyle) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "#include \"a.h\"\n"
                     "#include \"b.h\"\n"
                     "#include \"clang/Format/Format.h\"\n"
                     "#include <memory>\n";
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"new/new.h\"\n"
                         "#include \"clang/Format/Format.h\"\n"
                         "#include <memory>\n"
                         "#include <list>\n";
  tooling::Replacements Replaces = {createInsertion("#include <list>"),
                                    createInsertion("#include \"new/new.h\"")};
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertNewSystemIncludeGoogleStyle) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "\n"
                     "#include \"y/a.h\"\n"
                     "#include \"z/b.h\"\n";
  // FIXME: inserting after the empty line following the main header might be
  // prefered.
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include <vector>\n"
                         "\n"
                         "#include \"y/a.h\"\n"
                         "#include \"z/b.h\"\n";
  tooling::Replacements Replaces = {createInsertion("#include <vector>")};
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp);
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertMultipleIncludesGoogleStyle) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "\n"
                     "#include <vector>\n"
                     "\n"
                     "#include \"y/a.h\"\n"
                     "#include \"z/b.h\"\n";
  std::string Expected = "#include \"x/fix.h\"\n"
                         "\n"
                         "#include <vector>\n"
                         "#include <list>\n"
                         "\n"
                         "#include \"y/a.h\"\n"
                         "#include \"z/b.h\"\n"
                         "#include \"x/x.h\"\n";
  tooling::Replacements Replaces = {createInsertion("#include <list>"),
                                    createInsertion("#include \"x/x.h\"")};
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp);
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertMultipleNewHeadersAndSortLLVM) {
  std::string Code = "\nint x;";
  std::string Expected = "#include \"fix.h\"\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"c.h\"\n"
                         "#include <list>\n"
                         "#include <vector>\n"
                         "\nint x;";
  tooling::Replacements Replaces = {createInsertion("#include \"a.h\""),
                                    createInsertion("#include \"c.h\""),
                                    createInsertion("#include \"b.h\""),
                                    createInsertion("#include <vector>"),
                                    createInsertion("#include <list>"),
                                    createInsertion("#include \"fix.h\"")};
  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertMultipleNewHeadersAndSortGoogle) {
  std::string Code = "\nint x;";
  std::string Expected = "#include \"fix.h\"\n"
                         "#include <list>\n"
                         "#include <vector>\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"c.h\"\n"
                         "\nint x;";
  tooling::Replacements Replaces = {createInsertion("#include \"a.h\""),
                                    createInsertion("#include \"c.h\""),
                                    createInsertion("#include \"b.h\""),
                                    createInsertion("#include <vector>"),
                                    createInsertion("#include <list>"),
                                    createInsertion("#include \"fix.h\"")};
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp);
  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, FormatCorrectLineWhenHeadersAreInserted) {
  std::string Code = "\n"
                     "int    a;\n"
                     "int    a;\n"
                     "int    a;";

  std::string Expected = "#include \"x.h\"\n"
                         "#include \"y.h\"\n"
                         "#include \"clang/x/x.h\"\n"
                         "#include <list>\n"
                         "#include <vector>\n"
                         "\n"
                         "int    a;\n"
                         "int b;\n"
                         "int    a;";
  tooling::Replacements Replaces = {
      createReplacement(getOffset(Code, 3, 8), 1, "b"),
      createInsertion("#include <vector>"),
      createInsertion("#include <list>"),
      createInsertion("#include \"clang/x/x.h\""),
      createInsertion("#include \"y.h\""),
      createInsertion("#include \"x.h\"")};
  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, NotConfusedByDefine) {
  std::string Code = "void f() {}\n"
                     "#define A \\\n"
                     "  int i;";
  std::string Expected = "#include <vector>\n"
                         "void f() {}\n"
                         "#define A \\\n"
                         "  int i;";
  tooling::Replacements Replaces = {createInsertion("#include <vector>")};
  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

} // end namespace
} // end namespace format
} // end namespace clang
