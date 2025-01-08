//===- unittest/Format/CleanupTest.cpp - Code cleanup unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "../Tooling/ReplacementTest.h"
#include "clang/Tooling/Core/Replacement.h"

#include "gtest/gtest.h"

using clang::tooling::ReplacementTest;
using clang::tooling::toReplacements;

namespace clang {
namespace format {
namespace {

class CleanupTest : public testing::Test {
protected:
  std::string cleanup(StringRef Code, const std::vector<tooling::Range> &Ranges,
                      const FormatStyle &Style = getLLVMStyle()) {
    tooling::Replacements Replaces = format::cleanup(Style, Code, Ranges);

    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    return *Result;
  }

  // Returns code after cleanup around \p Offsets.
  std::string cleanupAroundOffsets(ArrayRef<unsigned> Offsets, StringRef Code,
                                   const FormatStyle &Style = getLLVMStyle()) {
    std::vector<tooling::Range> Ranges;
    for (auto Offset : Offsets)
      Ranges.push_back(tooling::Range(Offset, 0));
    return cleanup(Code, Ranges, Style);
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
  EXPECT_EQ(Expected, cleanupAroundOffsets({28, 91, 132}, Code));
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
  EXPECT_EQ(Expected, cleanup(Code, Ranges));
}

TEST_F(CleanupTest, EmptyNamespaceNotAffected) {
  std::string Code = "namespace A {\n\n"
                     "namespace {\n\n}}";
  // Even though the namespaces are empty, but the inner most empty namespace
  // block is not affected by the changed ranges.
  std::string Expected = "namespace A {\n\n"
                         "namespace {\n\n}}";
  // Set the changed range to be the second "\n".
  EXPECT_EQ(Expected, cleanupAroundOffsets({14}, Code));
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

TEST_F(CleanupTest, EmptyNamespaceAroundConditionalCompilation) {
  std::string Code = "#ifdef A\n"
                     "int a;\n"
                     "int b;\n"
                     "#else\n"
                     "#endif\n"
                     "namespace {}";
  std::string Expected = "#ifdef A\n"
                         "int a;\n"
                         "int b;\n"
                         "#else\n"
                         "#endif\n";
  std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
  FormatStyle Style = getLLVMStyle();
  std::string Result = cleanup(Code, Ranges, Style);
  EXPECT_EQ(Expected, Result);
}

TEST_F(CleanupTest, CtorInitializationSimpleRedundantComma) {
  std::string Code = "class A {\nA() : , {} };";
  std::string Expected = "class A {\nA()  {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({17, 19}, Code));

  Code = "class A {\nA() : x(1), {} };";
  Expected = "class A {\nA() : x(1) {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({23}, Code));

  Code = "class A {\nA() :,,,,{} };";
  Expected = "class A {\nA() {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({15}, Code));
}

// regression test for bug 39310
TEST_F(CleanupTest, CtorInitializationSimpleRedundantCommaInFunctionTryBlock) {
  std::string Code = "class A {\nA() try : , {} };";
  std::string Expected = "class A {\nA() try  {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({21, 23}, Code));

  Code = "class A {\nA() try : x(1), {} };";
  Expected = "class A {\nA() try : x(1) {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({27}, Code));

  Code = "class A {\nA() try :,,,,{} };";
  Expected = "class A {\nA() try {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({19}, Code));

  Code = "class A {\nA() try : x(1),,, {} };";
  Expected = "class A {\nA() try : x(1) {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({27}, Code));

  // Do not remove every comma following a colon as it simply doesn't make
  // sense in some situations.
  Code = "try : , {}";
  Expected = "try : , {}";
  EXPECT_EQ(Expected, cleanupAroundOffsets({8}, Code));
}

TEST_F(CleanupTest, CtorInitializationSimpleRedundantColon) {
  std::string Code = "class A {\nA() : =default; };";
  std::string Expected = "class A {\nA()  =default; };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({15}, Code));

  Code = "class A {\nA() : , =default; };";
  Expected = "class A {\nA()  =default; };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({15}, Code));
}

TEST_F(CleanupTest, ListRedundantComma) {
  std::string Code = "void f() { std::vector<int> v = {1,2,,,3,{4,5}}; }";
  std::string Expected = "void f() { std::vector<int> v = {1,2,3,{4,5}}; }";
  EXPECT_EQ(Expected, cleanupAroundOffsets({40}, Code));

  Code = "int main() { f(1,,2,3,,4);}";
  Expected = "int main() { f(1,2,3,4);}";
  EXPECT_EQ(Expected, cleanupAroundOffsets({17, 22}, Code));
}

TEST_F(CleanupTest, NoCleanupsForJavaScript) {
  std::string Code = "function f() { var x = [a, b, , c]; }";
  std::string Expected = "function f() { var x = [a, b, , c]; }";
  const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_JavaScript);

  EXPECT_EQ(Expected, cleanupAroundOffsets({30}, Code, Style));
}

TEST_F(CleanupTest, TrailingCommaInParens) {
  std::string Code = "int main() { f(,1,,2,3,f(1,2,),4,,);}";
  std::string Expected = "int main() { f(1,2,3,f(1,2),4);}";
  EXPECT_EQ(Expected, cleanupAroundOffsets({15, 18, 29, 33}, Code));

  // Lambda contents are also checked for trailing commas.
  Code = "int main() { [](){f(,1,,2,3,f(1,2,),4,,);}();}";
  Expected = "int main() { [](){f(1,2,3,f(1,2),4);}();}";
  EXPECT_EQ(Expected, cleanupAroundOffsets({20, 23, 34, 38}, Code));
}

TEST_F(CleanupTest, TrailingCommaInBraces) {
  // Trailing comma is allowed in brace list.
  // If there was trailing comma in the original code, then trailing comma is
  // preserved. In this example, element between the last two commas is deleted
  // causing the second-last comma to be redundant.
  std::string Code = "void f() { std::vector<int> v = {1,2,3,,}; }";
  std::string Expected = "void f() { std::vector<int> v = {1,2,3,}; }";
  EXPECT_EQ(Expected, cleanupAroundOffsets({39}, Code));

  // If there was no trailing comma in the original code, then trailing comma
  // introduced by replacements should be cleaned up. In this example, the
  // element after the last comma is deleted causing the last comma to be
  // redundant.
  Code = "void f() { std::vector<int> v = {1,2,3,}; }";
  // FIXME: redundant trailing comma should be removed.
  Expected = "void f() { std::vector<int> v = {1,2,3,}; }";
  EXPECT_EQ(Expected, cleanupAroundOffsets({39}, Code));

  // Still no trailing comma in the original code, but two elements are deleted,
  // which makes it seems like there was trailing comma.
  Code = "void f() { std::vector<int> v = {1, 2, 3, , }; }";
  // FIXME: redundant trailing comma should also be removed.
  Expected = "void f() { std::vector<int> v = {1, 2, 3,  }; }";
  EXPECT_EQ(Expected, cleanupAroundOffsets({42, 44}, Code));
}

TEST_F(CleanupTest, CtorInitializationBracesInParens) {
  std::string Code = "class A {\nA() : x({1}),, {} };";
  std::string Expected = "class A {\nA() : x({1}) {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({24, 26}, Code));
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

TEST_F(CleanupTest, RemoveCommentsAroundDeleteCode) {
  std::string Code =
      "class A {\nA() : x({1}), /* comment */, /* comment */ {} };";
  std::string Expected = "class A {\nA() : x({1}) {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({25, 40}, Code));

  Code = "class A {\nA() : x({1}), // comment\n {} };";
  Expected = "class A {\nA() : x({1})\n {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({25}, Code));

  Code = "class A {\nA() : x({1}), // comment\n , y(1),{} };";
  Expected = "class A {\nA() : x({1}),  y(1){} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({38}, Code));

  Code = "class A {\nA() : x({1}), \n/* comment */, y(1),{} };";
  Expected = "class A {\nA() : x({1}), \n y(1){} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({40}, Code));

  Code = "class A {\nA() : , // comment\n y(1),{} };";
  Expected = "class A {\nA() :  // comment\n y(1){} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({17}, Code));

  Code = "class A {\nA() // comment\n : ,,{} };";
  Expected = "class A {\nA() // comment\n {} };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({30}, Code));

  Code = "class A {\nA() // comment\n : ,,=default; };";
  Expected = "class A {\nA() // comment\n =default; };";
  EXPECT_EQ(Expected, cleanupAroundOffsets({30}, Code));
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

class CleanUpReplacementsTest : public ReplacementTest {
protected:
  tooling::Replacement createReplacement(unsigned Offset, unsigned Length,
                                         StringRef Text) {
    return tooling::Replacement(FileName, Offset, Length, Text);
  }

  tooling::Replacement createInsertion(StringRef IncludeDirective) {
    return createReplacement(UINT_MAX, 0, IncludeDirective);
  }

  tooling::Replacement createDeletion(StringRef HeaderName) {
    return createReplacement(UINT_MAX, 1, HeaderName);
  }

  inline std::string apply(StringRef Code,
                           const tooling::Replacements &Replaces) {
    auto CleanReplaces = cleanupAroundReplacements(Code, Replaces, Style);
    EXPECT_TRUE(static_cast<bool>(CleanReplaces))
        << toString(CleanReplaces.takeError()) << "\n";
    auto Result = applyAllReplacements(Code, *CleanReplaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    return *Result;
  }

  inline std::string formatAndApply(StringRef Code,
                                    const tooling::Replacements &Replaces) {
    auto CleanReplaces = cleanupAroundReplacements(Code, Replaces, Style);
    EXPECT_TRUE(static_cast<bool>(CleanReplaces))
        << toString(CleanReplaces.takeError()) << "\n";
    auto FormattedReplaces = formatReplacements(Code, *CleanReplaces, Style);
    EXPECT_TRUE(static_cast<bool>(FormattedReplaces))
        << toString(FormattedReplaces.takeError()) << "\n";
    auto Result = applyAllReplacements(Code, *FormattedReplaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    return *Result;
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
  tooling::Replacements Replaces =
      toReplacements({createReplacement(getOffset(Code, 3, 3), 6, ""),
                      createReplacement(getOffset(Code, 9, 34), 6, "")});

  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertMultipleIncludesLLVMStyle) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "#include \"a.h\"\n"
                     "#include \"b.h\"\n"
                     "#include \"z.h\"\n"
                     "#include \"clang/Format/Format.h\"\n"
                     "#include <memory>\n";
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"new/new.h\"\n"
                         "#include \"z.h\"\n"
                         "#include \"clang/Format/Format.h\"\n"
                         "#include <list>\n"
                         "#include <memory>\n";
  tooling::Replacements Replaces =
      toReplacements({createInsertion("#include <list>"),
                      createInsertion("#include \"new/new.h\"")});
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
                         "#include <list>\n"
                         "#include <vector>\n"
                         "\n"
                         "#include \"x/x.h\"\n"
                         "#include \"y/a.h\"\n"
                         "#include \"z/b.h\"\n";
  tooling::Replacements Replaces =
      toReplacements({createInsertion("#include <list>"),
                      createInsertion("#include \"x/x.h\"")});
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp);
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertMultipleNewHeadersAndSortLLVM) {
  std::string Code = "\nint x;";
  std::string Expected = "\n#include \"fix.h\"\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"c.h\"\n"
                         "#include <list>\n"
                         "#include <vector>\n"
                         "int x;";
  tooling::Replacements Replaces = toReplacements(
      {createInsertion("#include \"a.h\""), createInsertion("#include \"c.h\""),
       createInsertion("#include \"b.h\""),
       createInsertion("#include <vector>"), createInsertion("#include <list>"),
       createInsertion("#include \"fix.h\"")});
  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertMultipleNewHeadersAndSortGoogle) {
  std::string Code = "\nint x;";
  std::string Expected = "\n#include \"fix.h\"\n"
                         "\n"
                         "#include <list>\n"
                         "#include <vector>\n"
                         "\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"c.h\"\n"
                         "int x;";
  tooling::Replacements Replaces = toReplacements(
      {createInsertion("#include \"a.h\""), createInsertion("#include \"c.h\""),
       createInsertion("#include \"b.h\""),
       createInsertion("#include <vector>"), createInsertion("#include <list>"),
       createInsertion("#include \"fix.h\"")});
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp);
  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, NoNewLineAtTheEndOfCodeMultipleInsertions) {
  std::string Code = "#include <map>";
  // FIXME: a better behavior is to only append on newline to Code, but this
  // case should be rare in practice.
  std::string Expected =
      "#include <map>\n#include <string>\n\n#include <vector>\n";
  tooling::Replacements Replaces =
      toReplacements({createInsertion("#include <string>"),
                      createInsertion("#include <vector>")});
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, FormatCorrectLineWhenHeadersAreInserted) {
  std::string Code = "\n"
                     "int x;\n"
                     "int    a;\n"
                     "int    a;\n"
                     "int    a;";

  std::string Expected = "\n#include \"x.h\"\n"
                         "#include \"y.h\"\n"
                         "#include \"clang/x/x.h\"\n"
                         "#include <list>\n"
                         "#include <vector>\n"
                         "int x;\n"
                         "int    a;\n"
                         "int b;\n"
                         "int    a;";
  tooling::Replacements Replaces = toReplacements(
      {createReplacement(getOffset(Code, 4, 8), 1, "b"),
       createInsertion("#include <vector>"), createInsertion("#include <list>"),
       createInsertion("#include \"clang/x/x.h\""),
       createInsertion("#include \"y.h\""),
       createInsertion("#include \"x.h\"")});
  EXPECT_EQ(Expected, formatAndApply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, SimpleDeleteIncludes) {
  std::string Code = "#include \"abc.h\"\n"
                     "#include \"xyz.h\" // comment\n"
                     "#include \"xyz\"\n"
                     "int x;\n";
  std::string Expected = "#include \"xyz\"\n"
                         "int x;\n";
  tooling::Replacements Replaces =
      toReplacements({createDeletion("abc.h"), createDeletion("xyz.h")});
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

TEST_F(CleanUpReplacementsTest, InsertionAndDeleteHeader) {
  std::string Code = "#include \"a.h\"\n"
                     "\n"
                     "#include <vector>\n";
  std::string Expected = "#include \"a.h\"\n"
                         "\n"
                         "#include <map>\n";
  tooling::Replacements Replaces = toReplacements(
      {createDeletion("<vector>"), createInsertion("#include <map>")});
  EXPECT_EQ(Expected, apply(Code, Replaces));
}

} // end namespace
} // end namespace format
} // end namespace clang
