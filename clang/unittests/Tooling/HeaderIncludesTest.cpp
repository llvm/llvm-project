//===- unittest/Tooling/CleanupTest.cpp - Include insertion/deletion tests ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Inclusions/HeaderIncludes.h"
#include "../Tooling/ReplacementTest.h"
#include "../Tooling/RewriterTestContext.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"

#include "gtest/gtest.h"

using clang::tooling::ReplacementTest;
using clang::tooling::toReplacements;

namespace clang {
namespace tooling {
namespace {

class HeaderIncludesTest : public ::testing::Test {
protected:
  std::string insert(llvm::StringRef Code, llvm::StringRef Header) {
    HeaderIncludes Includes(FileName, Code, Style);
    assert(Header.startswith("\"") || Header.startswith("<"));
    auto R = Includes.insert(Header.trim("\"<>"), Header.startswith("<"));
    if (!R)
      return Code;
    auto Result = applyAllReplacements(Code, Replacements(*R));
    EXPECT_TRUE(static_cast<bool>(Result));
    return *Result;
  }

  std::string remove(llvm::StringRef Code, llvm::StringRef Header) {
    HeaderIncludes Includes(FileName, Code, Style);
    assert(Header.startswith("\"") || Header.startswith("<"));
    auto Replaces = Includes.remove(Header.trim("\"<>"), Header.startswith("<"));
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    return *Result;
  }

  const std::string FileName = "fix.cpp";
  IncludeStyle Style = format::getLLVMStyle().IncludeStyle;
};

TEST_F(HeaderIncludesTest, NoExistingIncludeWithoutDefine) {
  std::string Code = "int main() {}";
  std::string Expected = "#include \"a.h\"\n"
                         "int main() {}";
  EXPECT_EQ(Expected, insert(Code, "\"a.h\""));
}

TEST_F(HeaderIncludesTest, NoExistingIncludeWithDefine) {
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

  EXPECT_EQ(Expected, insert(Code, "\"b.h\""));
}

TEST_F(HeaderIncludesTest, InsertBeforeCategoryWithLowerPriority) {
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

  EXPECT_EQ(Expected, insert(Code, "\"a.h\""));
}

TEST_F(HeaderIncludesTest, InsertAfterMainHeader) {
  std::string Code = "#include \"fix.h\"\n"
                     "\n"
                     "int main() {}";
  std::string Expected = "#include \"fix.h\"\n"
                         "#include <a>\n"
                         "\n"
                         "int main() {}";
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp)
              .IncludeStyle;
  EXPECT_EQ(Expected, insert(Code, "<a>"));
}

TEST_F(HeaderIncludesTest, InsertBeforeSystemHeaderLLVM) {
  std::string Code = "#include <memory>\n"
                     "\n"
                     "int main() {}";
  std::string Expected = "#include \"z.h\"\n"
                         "#include <memory>\n"
                         "\n"
                         "int main() {}";
  EXPECT_EQ(Expected, insert(Code, "\"z.h\""));
}

TEST_F(HeaderIncludesTest, InsertAfterSystemHeaderGoogle) {
  std::string Code = "#include <memory>\n"
                     "\n"
                     "int main() {}";
  std::string Expected = "#include <memory>\n"
                         "#include \"z.h\"\n"
                         "\n"
                         "int main() {}";
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp)
              .IncludeStyle;
  EXPECT_EQ(Expected, insert(Code, "\"z.h\""));
}

TEST_F(HeaderIncludesTest, InsertOneIncludeLLVMStyle) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "#include \"a.h\"\n"
                     "#include \"b.h\"\n"
                     "#include \"clang/Format/Format.h\"\n"
                     "#include <memory>\n";
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"clang/Format/Format.h\"\n"
                         "#include \"llvm/x/y.h\"\n"
                         "#include <memory>\n";
  EXPECT_EQ(Expected, insert(Code, "\"llvm/x/y.h\""));
}

TEST_F(HeaderIncludesTest, InsertIntoBlockSorted) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "#include \"a.h\"\n"
                     "#include \"c.h\"\n"
                     "#include <memory>\n";
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"c.h\"\n"
                         "#include <memory>\n";
  EXPECT_EQ(Expected, insert(Code, "\"b.h\""));
}

TEST_F(HeaderIncludesTest, InsertIntoFirstBlockOfSameKind) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "#include \"c.h\"\n"
                     "#include \"e.h\"\n"
                     "#include \"f.h\"\n"
                     "#include <memory>\n"
                     "#include <vector>\n"
                     "#include \"m.h\"\n"
                     "#include \"n.h\"\n";
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include \"c.h\"\n"
                         "#include \"d.h\"\n"
                         "#include \"e.h\"\n"
                         "#include \"f.h\"\n"
                         "#include <memory>\n"
                         "#include <vector>\n"
                         "#include \"m.h\"\n"
                         "#include \"n.h\"\n";
  EXPECT_EQ(Expected, insert(Code, "\"d.h\""));
}

TEST_F(HeaderIncludesTest, InsertIntoSystemBlockSorted) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "#include \"a.h\"\n"
                     "#include \"c.h\"\n"
                     "#include <a>\n"
                     "#include <z>\n";
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include \"a.h\"\n"
                         "#include \"c.h\"\n"
                         "#include <a>\n"
                         "#include <vector>\n"
                         "#include <z>\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, InsertNewSystemIncludeGoogleStyle) {
  std::string Code = "#include \"x/fix.h\"\n"
                     "\n"
                     "#include \"y/a.h\"\n"
                     "#include \"z/b.h\"\n";
  // FIXME: inserting after the empty line following the main header might be
  // preferred.
  std::string Expected = "#include \"x/fix.h\"\n"
                         "#include <vector>\n"
                         "\n"
                         "#include \"y/a.h\"\n"
                         "#include \"z/b.h\"\n";
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp)
              .IncludeStyle;
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, NotConfusedByDefine) {
  std::string Code = "void f() {}\n"
                     "#define A \\\n"
                     "  int i;";
  std::string Expected = "#include <vector>\n"
                         "void f() {}\n"
                         "#define A \\\n"
                         "  int i;";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, SkippedTopComment) {
  std::string Code = "// comment\n"
                     "\n"
                     "   // comment\n";
  std::string Expected = "// comment\n"
                         "\n"
                         "   // comment\n"
                         "#include <vector>\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, SkippedMixedComments) {
  std::string Code = "// comment\n"
                     "// comment \\\n"
                     " comment continued\n"
                     "/*\n"
                     "* comment\n"
                     "*/\n";
  std::string Expected = "// comment\n"
                         "// comment \\\n"
                         " comment continued\n"
                         "/*\n"
                         "* comment\n"
                         "*/\n"
                         "#include <vector>\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, MultipleBlockCommentsInOneLine) {
  std::string Code = "/*\n"
                     "* comment\n"
                     "*/ /* comment\n"
                     "*/\n"
                     "\n\n"
                     "/* c1 */ /*c2 */\n";
  std::string Expected = "/*\n"
                         "* comment\n"
                         "*/ /* comment\n"
                         "*/\n"
                         "\n\n"
                         "/* c1 */ /*c2 */\n"
                         "#include <vector>\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, CodeAfterComments) {
  std::string Code = "/*\n"
                     "* comment\n"
                     "*/ /* comment\n"
                     "*/\n"
                     "\n\n"
                     "/* c1 */ /*c2 */\n"
                     "\n"
                     "int x;\n";
  std::string Expected = "/*\n"
                         "* comment\n"
                         "*/ /* comment\n"
                         "*/\n"
                         "\n\n"
                         "/* c1 */ /*c2 */\n"
                         "\n"
                         "#include <vector>\n"
                         "int x;\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, FakeHeaderGuardIfDef) {
  std::string Code = "// comment \n"
                     "#ifdef X\n"
                     "#define X\n";
  std::string Expected = "// comment \n"
                         "#include <vector>\n"
                         "#ifdef X\n"
                         "#define X\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, RealHeaderGuardAfterComments) {
  std::string Code = "// comment \n"
                     "#ifndef X\n"
                     "#define X\n"
                     "int x;\n"
                     "#define Y 1\n";
  std::string Expected = "// comment \n"
                         "#ifndef X\n"
                         "#define X\n"
                         "#include <vector>\n"
                         "int x;\n"
                         "#define Y 1\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, IfNDefWithNoDefine) {
  std::string Code = "// comment \n"
                     "#ifndef X\n"
                     "int x;\n"
                     "#define Y 1\n";
  std::string Expected = "// comment \n"
                         "#include <vector>\n"
                         "#ifndef X\n"
                         "int x;\n"
                         "#define Y 1\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, FakeHeaderGuard) {
  std::string Code = "// comment \n"
                     "#ifndef X\n"
                     "#define 1\n";
  std::string Expected = "// comment \n"
                         "#include <vector>\n"
                         "#ifndef X\n"
                         "#define 1\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, HeaderGuardWithComment) {
  std::string Code = "// comment \n"
                     "#ifndef X // comment\n"
                     "// comment\n"
                     "/* comment\n"
                     "*/\n"
                     "/* comment */ #define X\n"
                     "int x;\n"
                     "#define Y 1\n";
  std::string Expected = "// comment \n"
                         "#ifndef X // comment\n"
                         "// comment\n"
                         "/* comment\n"
                         "*/\n"
                         "/* comment */ #define X\n"
                         "#include <vector>\n"
                         "int x;\n"
                         "#define Y 1\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, EmptyCode) {
  std::string Code = "";
  std::string Expected = "#include <vector>\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, NoNewLineAtTheEndOfCode) {
  std::string Code = "#include <map>";
  std::string Expected = "#include <map>\n#include <vector>\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
}

TEST_F(HeaderIncludesTest, SkipExistingHeaders) {
  std::string Code = "#include \"a.h\"\n"
                     "#include <vector>\n";
  std::string Expected = "#include \"a.h\"\n"
                         "#include <vector>\n";
  EXPECT_EQ(Expected, insert(Code, "<vector>"));
  EXPECT_EQ(Expected, insert(Code, "\"a.h\""));
}

TEST_F(HeaderIncludesTest, AddIncludesWithDifferentForms) {
  std::string Code = "#include <vector>\n";
  // FIXME: this might not be the best behavior.
  std::string Expected = "#include \"vector\"\n"
                         "#include <vector>\n";
  EXPECT_EQ(Expected, insert(Code, "\"vector\""));
}

TEST_F(HeaderIncludesTest, NoInsertionAfterCode) {
  std::string Code = "#include \"a.h\"\n"
                     "void f() {}\n"
                     "#include \"b.h\"\n";
  std::string Expected = "#include \"a.h\"\n"
                         "#include \"c.h\"\n"
                         "void f() {}\n"
                         "#include \"b.h\"\n";
  EXPECT_EQ(Expected, insert(Code, "\"c.h\""));
}

TEST_F(HeaderIncludesTest, NoInsertionInStringLiteral) {
  std::string Code = "#include \"a.h\"\n"
                     "const char[] = R\"(\n"
                     "#include \"b.h\"\n"
                     ")\";\n";
  std::string Expected = "#include \"a.h\"\n"
                         "#include \"c.h\"\n"
                         "const char[] = R\"(\n"
                         "#include \"b.h\"\n"
                         ")\";\n";
  EXPECT_EQ(Expected, insert(Code, "\"c.h\""));
}

TEST_F(HeaderIncludesTest, NoInsertionAfterOtherDirective) {
  std::string Code = "#include \"a.h\"\n"
                     "#ifdef X\n"
                     "#include \"b.h\"\n"
                     "#endif\n";
  std::string Expected = "#include \"a.h\"\n"
                         "#include \"c.h\"\n"
                         "#ifdef X\n"
                         "#include \"b.h\"\n"
                         "#endif\n";
  EXPECT_EQ(Expected, insert(Code, "\"c.h\""));
}

TEST_F(HeaderIncludesTest, CanInsertAfterLongSystemInclude) {
  std::string Code = "#include \"a.h\"\n"
                     "// comment\n\n"
                     "#include <a/b/c/d/e.h>\n";
  std::string Expected = "#include \"a.h\"\n"
                         "// comment\n\n"
                         "#include <a/b/c/d/e.h>\n"
                         "#include <x.h>\n";
  EXPECT_EQ(Expected, insert(Code, "<x.h>"));
}

TEST_F(HeaderIncludesTest, CanInsertAfterComment) {
  std::string Code = "#include \"a.h\"\n"
                     "// Comment\n"
                     "\n"
                     "/* Comment */\n"
                     "// Comment\n"
                     "\n"
                     "#include \"b.h\"\n";
  std::string Expected = "#include \"a.h\"\n"
                         "// Comment\n"
                         "\n"
                         "/* Comment */\n"
                         "// Comment\n"
                         "\n"
                         "#include \"b.h\"\n"
                         "#include \"c.h\"\n";
  EXPECT_EQ(Expected, insert(Code, "\"c.h\""));
}

TEST_F(HeaderIncludesTest, LongCommentsInTheBeginningOfFile) {
  std::string Code = "// Loooooooooooooooooooooooooong comment\n"
                     "// Loooooooooooooooooooooooooong comment\n"
                     "// Loooooooooooooooooooooooooong comment\n"
                     "#include <string>\n"
                     "#include <vector>\n"
                     "\n"
                     "#include \"a.h\"\n"
                     "#include \"b.h\"\n";
  std::string Expected = "// Loooooooooooooooooooooooooong comment\n"
                         "// Loooooooooooooooooooooooooong comment\n"
                         "// Loooooooooooooooooooooooooong comment\n"
                         "#include <string>\n"
                         "#include <vector>\n"
                         "\n"
                         "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"third.h\"\n";
  Style = format::getGoogleStyle(format::FormatStyle::LanguageKind::LK_Cpp)
              .IncludeStyle;
  EXPECT_EQ(Expected, insert(Code, "\"third.h\""));
}

TEST_F(HeaderIncludesTest, SimpleDeleteInclude) {
  std::string Code = "#include \"abc.h\"\n"
                     "#include \"xyz.h\" // comment\n"
                     "int x;\n";
  std::string Expected = "#include \"abc.h\"\n"
                         "int x;\n";
  EXPECT_EQ(Expected, remove(Code, "\"xyz.h\""));
}

TEST_F(HeaderIncludesTest, DeleteQuotedOnly) {
  std::string Code = "#include \"abc.h\"\n"
                     "#include <abc.h>\n"
                     "int x;\n";
  std::string Expected = "#include <abc.h>\n"
                         "int x;\n";
  EXPECT_EQ(Expected, remove(Code, "\"abc.h\""));
}

TEST_F(HeaderIncludesTest, DeleteAllCode) {
  std::string Code = "#include \"xyz.h\"\n";
  std::string Expected = "";
  EXPECT_EQ(Expected, remove(Code, "\"xyz.h\""));
}

TEST_F(HeaderIncludesTest, DeleteOnlyIncludesWithSameQuote) {
  std::string Code = "#include \"xyz.h\"\n"
                     "#include \"xyz\"\n"
                     "#include <xyz.h>\n";
  std::string Expected = "#include \"xyz.h\"\n"
                         "#include \"xyz\"\n";
  EXPECT_EQ(Expected, remove(Code, "<xyz.h>"));
}

TEST_F(HeaderIncludesTest, CanDeleteAfterCode) {
  std::string Code = "#include \"a.h\"\n"
                     "void f() {}\n"
                     "#include \"b.h\"\n";
  std::string Expected = "#include \"a.h\"\n"
                         "void f() {}\n";
  EXPECT_EQ(Expected, remove(Code, "\"b.h\""));
}

} // namespace
} // namespace tooling
} // namespace clang
