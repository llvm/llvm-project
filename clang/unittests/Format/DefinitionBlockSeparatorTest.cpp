//===- DefinitionBlockSeparatorTest.cpp - Formatting unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"

#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "definition-block-separator-test"

namespace clang {
namespace format {
namespace {

class DefinitionBlockSeparatorTest : public testing::Test {
protected:
  static std::string
  separateDefinitionBlocks(StringRef Code,
                           const std::vector<tooling::Range> &Ranges,
                           const FormatStyle &Style = getLLVMStyle()) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    tooling::Replacements Replaces = reformat(Style, Code, Ranges, "<stdin>");
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  static std::string
  separateDefinitionBlocks(StringRef Code,
                           const FormatStyle &Style = getLLVMStyle()) {
    return separateDefinitionBlocks(
        Code,
        /*Ranges=*/{1, tooling::Range(0, Code.size())}, Style);
  }

  static void _verifyFormat(const char *File, int Line, StringRef Code,
                            const FormatStyle &Style = getLLVMStyle(),
                            StringRef ExpectedCode = "", bool Inverse = true) {
    testing::ScopedTrace t(File, Line, testing::Message() << Code.str());
    bool HasOriginalCode = true;
    if (ExpectedCode == "") {
      ExpectedCode = Code;
      HasOriginalCode = false;
    }

    EXPECT_EQ(ExpectedCode, separateDefinitionBlocks(ExpectedCode, Style))
        << "Expected code is not stable";
    if (Inverse) {
      FormatStyle InverseStyle = Style;
      if (Style.SeparateDefinitionBlocks == FormatStyle::SDS_Always)
        InverseStyle.SeparateDefinitionBlocks = FormatStyle::SDS_Never;
      else
        InverseStyle.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
      EXPECT_NE(ExpectedCode,
                separateDefinitionBlocks(ExpectedCode, InverseStyle))
          << "Inverse formatting makes no difference";
    }
    std::string CodeToFormat =
        HasOriginalCode ? Code.str() : removeEmptyLines(Code);
    std::string Result = separateDefinitionBlocks(CodeToFormat, Style);
    EXPECT_EQ(ExpectedCode, Result) << "Test failed. Formatted:\n" << Result;
  }

  static std::string removeEmptyLines(StringRef Code) {
    std::string Result = "";
    for (auto Char : Code.str()) {
      if (Result.size()) {
        auto LastChar = Result.back();
        if ((Char == '\n' && LastChar == '\n') ||
            (Char == '\r' && (LastChar == '\r' || LastChar == '\n'))) {
          continue;
        }
      }
      Result.push_back(Char);
    }
    return Result;
  }
};

#define verifyFormat(...) _verifyFormat(__FILE__, __LINE__, __VA_ARGS__)

TEST_F(DefinitionBlockSeparatorTest, Basic) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  verifyFormat("int foo(int i, int j) {\n"
               "  int r = i + j;\n"
               "  return r;\n"
               "}\n"
               "\n"
               "int bar(int j, int k) {\n"
               "  int r = j + k;\n"
               "  return r;\n"
               "}",
               Style);

  verifyFormat("struct foo {\n"
               "  int i, j;\n"
               "};\n"
               "\n"
               "struct bar {\n"
               "  int j, k;\n"
               "};",
               Style);

  verifyFormat("union foo {\n"
               "  int i, j;\n"
               "};\n"
               "\n"
               "union bar {\n"
               "  int j, k;\n"
               "};",
               Style);

  verifyFormat("class foo {\n"
               "  int i, j;\n"
               "};\n"
               "\n"
               "class bar {\n"
               "  int j, k;\n"
               "};",
               Style);

  verifyFormat("namespace foo {\n"
               "int i, j;\n"
               "}\n"
               "\n"
               "namespace bar {\n"
               "int j, k;\n"
               "}",
               Style);

  verifyFormat("enum Foo { FOO, BAR };\n"
               "\n"
               "enum Bar { FOOBAR, BARFOO };",
               Style);

  FormatStyle BreakAfterReturnTypeStyle = Style;
  BreakAfterReturnTypeStyle.BreakAfterReturnType = FormatStyle::RTBS_All;
  // Test uppercased long typename
  verifyFormat("class Foo {\n"
               "  void\n"
               "  Bar(int t, int p) {\n"
               "    int r = t + p;\n"
               "    return r;\n"
               "  }\n"
               "\n"
               "  HRESULT\n"
               "  Foobar(int t, int p) {\n"
               "    int r = t * p;\n"
               "    return r;\n"
               "  }\n"
               "}",
               BreakAfterReturnTypeStyle);
}

TEST_F(DefinitionBlockSeparatorTest, FormatConflict) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  StringRef Code = "class Test {\n"
                   "public:\n"
                   "  static void foo() {\n"
                   "    int t;\n"
                   "    return 1;\n"
                   "  }\n"
                   "};";
  std::vector<tooling::Range> Ranges = {1, tooling::Range(0, Code.size())};
  EXPECT_EQ(reformat(Style, Code, Ranges, "<stdin>").size(), 0u);
}

TEST_F(DefinitionBlockSeparatorTest, CommentBlock) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  std::string Prefix = "enum Foo { FOO, BAR };\n"
                       "\n"
                       "/*\n"
                       "test1\n"
                       "test2\n"
                       "*/\n"
                       "int foo(int i, int j) {\n"
                       "  int r = i + j;\n"
                       "  return r;\n"
                       "}\n";
  std::string Suffix = "enum Bar { FOOBAR, BARFOO };\n"
                       "\n"
                       "/* Comment block in one line*/\n"
                       "int bar3(int j, int k) {\n"
                       "  // A comment\n"
                       "  int r = j % k;\n"
                       "  return r;\n"
                       "}\n";
  std::string CommentedCode = "/*\n"
                              "int bar2(int j, int k) {\n"
                              "  int r = j / k;\n"
                              "  return r;\n"
                              "}\n"
                              "*/\n";
  verifyFormat(removeEmptyLines(Prefix) + "\n" + CommentedCode + "\n" +
                   removeEmptyLines(Suffix),
               Style, Prefix + "\n" + CommentedCode + "\n" + Suffix);
  verifyFormat(removeEmptyLines(Prefix) + "\n" + CommentedCode +
                   removeEmptyLines(Suffix),
               Style, Prefix + "\n" + CommentedCode + Suffix);
}

TEST_F(DefinitionBlockSeparatorTest, UntouchBlockStartStyle) {
  // Returns a std::pair of two strings, with the first one for passing into
  // Always test and the second one be the expected result of the first string.
  auto MakeUntouchTest = [&](std::string BlockHeader, std::string BlockChanger,
                             std::string BlockFooter, bool BlockEndNewLine) {
    std::string CodePart1 = "enum Foo { FOO, BAR };\n"
                            "\n"
                            "/*\n"
                            "test1\n"
                            "test2\n"
                            "*/\n"
                            "int foo(int i, int j) {\n"
                            "  int r = i + j;\n"
                            "  return r;\n"
                            "}\n";
    std::string CodePart2 = "/* Comment block in one line*/\n"
                            "enum Bar { FOOBAR, BARFOO };\n"
                            "\n"
                            "int bar3(int j, int k) {\n"
                            "  // A comment\n"
                            "  int r = j % k;\n"
                            "  return r;\n"
                            "}\n";
    std::string CodePart3 = "int bar2(int j, int k) {\n"
                            "  int r = j / k;\n"
                            "  return r;\n"
                            "}\n";
    std::string ConcatAll = BlockHeader + CodePart1 + BlockChanger + CodePart2 +
                            BlockFooter + (BlockEndNewLine ? "\n" : "") +
                            CodePart3;
    return std::make_pair(BlockHeader + removeEmptyLines(CodePart1) +
                              BlockChanger + removeEmptyLines(CodePart2) +
                              BlockFooter + removeEmptyLines(CodePart3),
                          ConcatAll);
  };

  FormatStyle AlwaysStyle = getLLVMStyle();
  AlwaysStyle.SeparateDefinitionBlocks = FormatStyle::SDS_Always;

  FormatStyle NeverStyle = getLLVMStyle();
  NeverStyle.SeparateDefinitionBlocks = FormatStyle::SDS_Never;

  auto TestKit = MakeUntouchTest("/* FOOBAR */\n"
                                 "#ifdef FOO\n\n",
                                 "\n#elifndef BAR\n\n", "\n#endif\n\n", false);
  verifyFormat(TestKit.first, AlwaysStyle, TestKit.second);
  verifyFormat(TestKit.second, NeverStyle, removeEmptyLines(TestKit.second));

  TestKit = MakeUntouchTest("/* FOOBAR */\n"
                            "#ifdef FOO\n",
                            "#elifndef BAR\n", "#endif\n", false);
  verifyFormat(TestKit.first, AlwaysStyle, TestKit.second);
  verifyFormat(TestKit.second, NeverStyle, removeEmptyLines(TestKit.second));

  TestKit = MakeUntouchTest("namespace Ns {\n\n",
                            "\n} // namespace Ns\n\n"
                            "namespace {\n\n",
                            "\n} // namespace\n", true);
  verifyFormat(TestKit.first, AlwaysStyle, TestKit.second);
  verifyFormat(TestKit.second, NeverStyle, removeEmptyLines(TestKit.second));

  TestKit = MakeUntouchTest("namespace Ns {\n",
                            "} // namespace Ns\n\n"
                            "namespace {\n",
                            "} // namespace\n", true);
  verifyFormat(TestKit.first, AlwaysStyle, TestKit.second);
  verifyFormat(TestKit.second, NeverStyle, removeEmptyLines(TestKit.second));
}

TEST_F(DefinitionBlockSeparatorTest, Always) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;

  verifyFormat("// clang-format off\n"
               "template<class T>\n"
               "concept C = not A<S<T>>;\n"
               "// clang-format on\n"
               "\n"
               "struct E {};",
               Style);

  std::string Prefix = "namespace {\n";
  std::string Infix = "\n"
                      "// Enum test1\n"
                      "// Enum test2\n"
                      "enum Foo { FOO, BAR };\n"
                      "\n"
                      "/*\n"
                      "test1\n"
                      "test2\n"
                      "*/\n"
                      "/*const*/ int foo(int i, int j) {\n"
                      "  int r = i + j;\n"
                      "  return r;\n"
                      "}\n"
                      "\n"
                      "// Foobar\n"
                      "int i, j, k;\n"
                      "\n"
                      "// Comment for function\n"
                      "// Comment line 2\n"
                      "// Comment line 3\n"
                      "int bar(int j, int k) {\n"
                      "  {\n"
                      "    int r = j * k;\n"
                      "    return r;\n"
                      "  }\n"
                      "}\n"
                      "\n"
                      "int bar2(int j, int k) {\n"
                      "  int r = j / k;\n"
                      "  return r;\n"
                      "}\n"
                      "\n"
                      "/* Comment block in one line*/\n"
                      "enum Bar { FOOBAR, BARFOO };\n"
                      "\n"
                      "int bar3(int j, int k, const enum Bar b) {\n"
                      "  // A comment\n"
                      "  int r = j % k;\n"
                      "  if (struct S = getS()) {\n"
                      "    // if condition\n"
                      "  }\n"
                      "  return r;\n"
                      "}\n";
  std::string Postfix = "\n"
                        "} // namespace\n"
                        "\n"
                        "namespace T {\n"
                        "int i, j, k;\n"
                        "} // namespace T";
  verifyFormat(Prefix + removeEmptyLines(Infix) + removeEmptyLines(Postfix),
               Style, Prefix + Infix + Postfix);
}

TEST_F(DefinitionBlockSeparatorTest, Never) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Never;
  std::string Prefix = "namespace {\n";
  std::string Postfix = "// Enum test1\n"
                        "// Enum test2\n"
                        "enum Foo { FOO, BAR };\n"
                        "\n"
                        "/*\n"
                        "test1\n"
                        "test2\n"
                        "*/\n"
                        "/*const*/ int foo(int i, int j) {\n"
                        "  int r = i + j;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "// Foobar\n"
                        "int i, j, k;\n"
                        "\n"
                        "// Comment for function\n"
                        "// Comment line 2\n"
                        "// Comment line 3\n"
                        "int bar(int j, int k) {\n"
                        "  {\n"
                        "    int r = j * k;\n"
                        "    return r;\n"
                        "  }\n"
                        "}\n"
                        "\n"
                        "int bar2(int j, int k) {\n"
                        "  int r = j / k;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "/* Comment block in one line*/\n"
                        "enum Bar { FOOBAR, BARFOO };\n"
                        "\n"
                        "int bar3(int j, int k, const enum Bar b) {\n"
                        "  // A comment\n"
                        "  int r = j % k;\n"
                        "  if (struct S = getS()) {\n"
                        "    // if condition\n"
                        "  }\n"
                        "  return r;\n"
                        "}\n"
                        "} // namespace";
  verifyFormat(Prefix + "\n\n\n" + Postfix, Style,
               Prefix + removeEmptyLines(Postfix));
}

TEST_F(DefinitionBlockSeparatorTest, OpeningBracketOwnsLine) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  verifyFormat("namespace NS\n"
               "{\n"
               "// Enum test1\n"
               "// Enum test2\n"
               "enum Foo\n"
               "{\n"
               "  FOO,\n"
               "  BAR\n"
               "};\n"
               "\n"
               "/*\n"
               "test1\n"
               "test2\n"
               "*/\n"
               "/*const*/ int foo(int i, int j)\n"
               "{\n"
               "  int r = i + j;\n"
               "  return r;\n"
               "}\n"
               "\n"
               "// Foobar\n"
               "int i, j, k;\n"
               "\n"
               "// Comment for function\n"
               "// Comment line 2\n"
               "// Comment line 3\n"
               "int bar(int j, int k)\n"
               "{\n"
               "  {\n"
               "    int r = j * k;\n"
               "    return r;\n"
               "  }\n"
               "}\n"
               "\n"
               "int bar2(int j, int k)\n"
               "{\n"
               "  int r = j / k;\n"
               "  return r;\n"
               "}\n"
               "\n"
               "enum Bar\n"
               "{\n"
               "  FOOBAR,\n"
               "  BARFOO\n"
               "};\n"
               "\n"
               "int bar3(int j, int k, const enum Bar b)\n"
               "{\n"
               "  // A comment\n"
               "  int r = j % k;\n"
               "  if (struct S = getS())\n"
               "  {\n"
               "    // if condition\n"
               "  }\n"
               "  return r;\n"
               "}\n"
               "} // namespace NS",
               Style);
}

TEST_F(DefinitionBlockSeparatorTest, TryBlocks) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  verifyFormat("void FunctionWithInternalTry()\n"
               "{\n"
               "  try\n"
               "  {\n"
               "    return;\n"
               "  }\n"
               "  catch (const std::exception &)\n"
               "  {\n"
               "  }\n"
               "}",
               Style, "", /*Inverse=*/false);
  verifyFormat("void FunctionWithTryBlock()\n"
               "try\n"
               "{\n"
               "  return;\n"
               "}\n"
               "catch (const std::exception &)\n"
               "{\n"
               "}",
               Style, "", /*Inverse=*/false);
}

TEST_F(DefinitionBlockSeparatorTest, Leave) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Leave;
  Style.MaxEmptyLinesToKeep = 3;
  std::string LeaveAs = "namespace {\n"
                        "\n"
                        "// Enum test1\n"
                        "// Enum test2\n"
                        "enum Foo { FOO, BAR };\n"
                        "\n\n\n"
                        "/*\n"
                        "test1\n"
                        "test2\n"
                        "*/\n"
                        "/*const*/ int foo(int i, int j) {\n"
                        "  int r = i + j;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "// Foobar\n"
                        "int i, j, k;\n"
                        "\n"
                        "// Comment for function\n"
                        "// Comment line 2\n"
                        "// Comment line 3\n"
                        "int bar(int j, int k) {\n"
                        "  {\n"
                        "    int r = j * k;\n"
                        "    return r;\n"
                        "  }\n"
                        "}\n"
                        "\n"
                        "int bar2(int j, int k) {\n"
                        "  int r = j / k;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "// Comment for inline enum\n"
                        "enum Bar { FOOBAR, BARFOO };\n"
                        "int bar3(int j, int k, const enum Bar b) {\n"
                        "  // A comment\n"
                        "  int r = j % k;\n"
                        "  if (struct S = getS()) {\n"
                        "    // if condition\n"
                        "  }\n"
                        "  return r;\n"
                        "}\n"
                        "} // namespace";
  verifyFormat(LeaveAs, Style, LeaveAs);
}

TEST_F(DefinitionBlockSeparatorTest, CSharp) {
  FormatStyle Style = getLLVMStyle(FormatStyle::LK_CSharp);
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  Style.AllowShortEnumsOnASingleLine = false;
  verifyFormat("namespace {\r\n"
               "public class SomeTinyClass {\r\n"
               "  int X;\r\n"
               "}\r\n"
               "\r\n"
               "public class AnotherTinyClass {\r\n"
               "  int Y;\r\n"
               "}\r\n"
               "\r\n"
               "internal static String toString() {\r\n"
               "}\r\n"
               "\r\n"
               "// Comment for enum\r\n"
               "public enum var {\r\n"
               "  none,\r\n"
               "  @string,\r\n"
               "  bool,\r\n"
               "  @enum\r\n"
               "}\r\n"
               "\r\n"
               "// Test\r\n"
               "[STAThread]\r\n"
               "static void Main(string[] args) {\r\n"
               "  Console.WriteLine(\"HelloWorld\");\r\n"
               "}\r\n"
               "\r\n"
               "static decimal Test() {\r\n"
               "}\r\n"
               "}\r\n"
               "\r\n"
               "public class FoobarClass {\r\n"
               "  int foobar;\r\n"
               "}",
               Style);
}

TEST_F(DefinitionBlockSeparatorTest, JavaScript) {
  FormatStyle Style = getLLVMStyle(FormatStyle::LK_JavaScript);
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  Style.AllowShortEnumsOnASingleLine = false;
  verifyFormat("export const enum Foo {\n"
               "  A = 1,\n"
               "  B\n"
               "}\n"
               "\n"
               "export function A() {\n"
               "}\n"
               "\n"
               "export default function B() {\n"
               "}\n"
               "\n"
               "export function C() {\n"
               "}\n"
               "\n"
               "var t, p, q;\n"
               "\n"
               "export abstract class X {\n"
               "  y: number;\n"
               "}\n"
               "\n"
               "export const enum Bar {\n"
               "  D = 1,\n"
               "  E\n"
               "}",
               Style);
}
} // namespace
} // namespace format
} // namespace clang
