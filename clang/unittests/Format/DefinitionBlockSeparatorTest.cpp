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

class DefinitionBlockSeparatorTest : public ::testing::Test {
protected:
  static std::string
  separateDefinitionBlocks(llvm::StringRef Code,
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
  separateDefinitionBlocks(llvm::StringRef Code,
                           const FormatStyle &Style = getLLVMStyle()) {
    return separateDefinitionBlocks(
        Code,
        /*Ranges=*/{1, tooling::Range(0, Code.size())}, Style);
  }

  static void verifyFormat(llvm::StringRef Code,
                           const FormatStyle &Style = getLLVMStyle(),
                           llvm::StringRef ExpectedCode = "") {
    bool HasOriginalCode = true;
    if (ExpectedCode == "") {
      ExpectedCode = Code;
      HasOriginalCode = false;
    }

    FormatStyle InverseStyle = Style;
    if (Style.SeparateDefinitionBlocks == FormatStyle::SDS_Always)
      InverseStyle.SeparateDefinitionBlocks = FormatStyle::SDS_Never;
    else
      InverseStyle.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
    EXPECT_EQ(ExpectedCode.str(), separateDefinitionBlocks(ExpectedCode, Style))
        << "Expected code is not stable";
    std::string InverseResult = separateDefinitionBlocks(Code, InverseStyle);
    EXPECT_NE(Code.str(), InverseResult)
        << "Inverse formatting makes no difference";
    std::string CodeToFormat =
        HasOriginalCode ? Code.str() : removeEmptyLines(Code);
    std::string Result = separateDefinitionBlocks(CodeToFormat, Style);
    EXPECT_EQ(ExpectedCode.str(), Result) << "Test failed. Formatted:\n"
                                          << Result;
  }

  static std::string removeEmptyLines(llvm::StringRef Code) {
    std::string Result = "";
    for (auto Char : Code.str()) {
      if (Result.size()) {
        auto LastChar = Result.back();
        if ((Char == '\n' && LastChar == '\n') ||
            (Char == '\r' && (LastChar == '\r' || LastChar == '\n')))
          continue;
      }
      Result.push_back(Char);
    }
    return Result;
  }
};

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
               "enum Bar { FOOBAR, BARFOO };\n",
               Style);
}

TEST_F(DefinitionBlockSeparatorTest, Always) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  std::string Prefix = "namespace {\n";
  std::string Postfix = "enum Foo { FOO, BAR };\n"
                        "\n"
                        "int foo(int i, int j) {\n"
                        "  int r = i + j;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "int i, j, k;\n"
                        "\n"
                        "int bar(int j, int k) {\n"
                        "  int r = j * k;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "enum Bar { FOOBAR, BARFOO };\n"
                        "} // namespace";
  verifyFormat(Prefix + "\n\n\n" + removeEmptyLines(Postfix), Style,
               Prefix + Postfix);
}

TEST_F(DefinitionBlockSeparatorTest, Never) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Never;
  std::string Prefix = "namespace {\n";
  std::string Postfix = "enum Foo { FOO, BAR };\n"
                        "\n"
                        "int foo(int i, int j) {\n"
                        "  int r = i + j;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "int i, j, k;\n"
                        "\n"
                        "int bar(int j, int k) {\n"
                        "  int r = j * k;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "enum Bar { FOOBAR, BARFOO };\n"
                        "} // namespace";
  verifyFormat(Prefix + "\n\n\n" + Postfix, Style,
               Prefix + removeEmptyLines(Postfix));
}

TEST_F(DefinitionBlockSeparatorTest, OpeningBracketOwnsLine) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  verifyFormat("enum Foo\n"
               "{\n"
               "  FOO,\n"
               "  BAR\n"
               "};\n"
               "\n"
               "int foo(int i, int j)\n"
               "{\n"
               "  int r = i + j;\n"
               "  return r;\n"
               "}\n"
               "\n"
               "int i, j, k;\n"
               "\n"
               "int bar(int j, int k)\n"
               "{\n"
               "  int r = j * k;\n"
               "  return r;\n"
               "}\n"
               "\n"
               "enum Bar\n"
               "{\n"
               "  FOOBAR,\n"
               "  BARFOO\n"
               "};",
               Style);
}

TEST_F(DefinitionBlockSeparatorTest, Leave) {
  FormatStyle Style = getLLVMStyle();
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Leave;
  Style.MaxEmptyLinesToKeep = 3;
  std::string LeaveAs = "namespace {\n"
                        "\n"
                        "enum Foo { FOO, BAR };\n"
                        "\n\n\n"
                        "int foo(int i, int j) {\n"
                        "  int r = i + j;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "int i, j, k;\n"
                        "\n"
                        "int bar(int j, int k) {\n"
                        "  int r = j * k;\n"
                        "  return r;\n"
                        "}\n"
                        "\n"
                        "enum Bar { FOOBAR, BARFOO };\n"
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
               "public enum var {\r\n"
               "  none,\r\n"
               "  @string,\r\n"
               "  bool,\r\n"
               "  @enum\r\n"
               "}\r\n"
               "\r\n"
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
