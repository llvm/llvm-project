//===- unittest/Format/FormatTestCSharp.cpp - Formatting tests for CSharp -===//
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

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {

class FormatTestCSharp : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    tooling::Replacements Replaces = reformat(Style, Code, Ranges);
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  static std::string
  format(llvm::StringRef Code,
         const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_CSharp)) {
    return format(Code, 0, Code.size(), Style);
  }

  static FormatStyle getStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  static void verifyFormat(
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_CSharp)) {
    EXPECT_EQ(Code.str(), format(Code, Style)) << "Expected code is not stable";
    EXPECT_EQ(Code.str(), format(test::messUp(Code), Style));
  }
};

TEST_F(FormatTestCSharp, CSharpClass) {
  verifyFormat("public class SomeClass {\n"
               "  void f() {}\n"
               "  int g() { return 0; }\n"
               "  void h() {\n"
               "    while (true) f();\n"
               "    for (;;) f();\n"
               "    if (true) f();\n"
               "  }\n"
               "}");
}

TEST_F(FormatTestCSharp, AccessModifiers) {
  verifyFormat("public String toString() {}");
  verifyFormat("private String toString() {}");
  verifyFormat("protected String toString() {}");
  verifyFormat("internal String toString() {}");

  verifyFormat("public override String toString() {}");
  verifyFormat("private override String toString() {}");
  verifyFormat("protected override String toString() {}");
  verifyFormat("internal override String toString() {}");

  verifyFormat("internal static String toString() {}");
}

TEST_F(FormatTestCSharp, NoStringLiteralBreaks) {
  verifyFormat("foo("
               "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaaaaa\");");
}

TEST_F(FormatTestCSharp, CSharpVerbatiumStringLiterals) {
  verifyFormat("foo(@\"aaaaaaaa\\abc\\aaaa\");");
  // @"ABC\" + ToString("B") - handle embedded \ in literal string at
  // the end
  //
  /*
   * After removal of Lexer change we are currently not able
   * To handle these cases
   verifyFormat("string s = @\"ABC\\\" + ToString(\"B\");");
   verifyFormat("string s = @\"ABC\"\"DEF\"\"GHI\"");
   verifyFormat("string s = @\"ABC\"\"DEF\"\"\"");
   verifyFormat("string s = @\"ABC\"\"DEF\"\"\" + abc");
  */
}

TEST_F(FormatTestCSharp, CSharpInterpolatedStringLiterals) {
  verifyFormat("foo($\"aaaaaaaa{aaa}aaaa\");");
  verifyFormat("foo($\"aaaa{A}\");");
  verifyFormat(
      "foo($\"aaaa{A}"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\");");
  verifyFormat("Name = $\"{firstName} {lastName}\";");

  // $"ABC\" + ToString("B") - handle embedded \ in literal string at
  // the end
  verifyFormat("string s = $\"A{abc}BC\" + ToString(\"B\");");
  verifyFormat("$\"{domain}\\\\{user}\"");
  verifyFormat(
      "var verbatimInterpolated = $@\"C:\\Users\\{userName}\\Documents\\\";");
}

TEST_F(FormatTestCSharp, CSharpFatArrows) {
  verifyFormat("Task serverTask = Task.Run(async() => {");
  verifyFormat("public override string ToString() => \"{Name}\\{Age}\";");
}

TEST_F(FormatTestCSharp, CSharpNullConditional) {
  verifyFormat(
      "public Person(string firstName, string lastName, int? age=null)");

  verifyFormat("switch(args?.Length)");

  verifyFormat("public static void Main(string[] args) { string dirPath "
               "= args?[0]; }");
}

TEST_F(FormatTestCSharp, Attributes) {
  verifyFormat("[STAThread]\n"
               "static void\n"
               "Main(string[] args) {}");

  verifyFormat("[TestMethod]\n"
               "private class Test {}");

  verifyFormat("[TestMethod]\n"
               "protected class Test {}");

  verifyFormat("[TestMethod]\n"
               "internal class Test {}");

  verifyFormat("[TestMethod]\n"
               "class Test {}");

  verifyFormat("[TestMethod]\n"
               "[DeploymentItem(\"Test.txt\")]\n"
               "public class Test {}");

  verifyFormat("[System.AttributeUsage(System.AttributeTargets.Method)]\n"
               "[System.Runtime.InteropServices.ComVisible(true)]\n"
               "public sealed class STAThreadAttribute : Attribute {}");

  verifyFormat("[Verb(\"start\", HelpText = \"Starts the server listening on "
               "provided port\")]\n"
               "class Test {}");

  verifyFormat("[TestMethod]\n"
               "public string Host {\n  set;\n  get;\n}");

  verifyFormat("[TestMethod(\"start\", HelpText = \"Starts the server "
               "listening on provided host\")]\n"
               "public string Host {\n  set;\n  get;\n}");
}

TEST_F(FormatTestCSharp, CSharpRegions) {
  verifyFormat("#region aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaa "
               "aaaaaaaaaaaaaaa long region");
}

TEST_F(FormatTestCSharp, CSharpKeyWordEscaping) {
  verifyFormat("public enum var { none, @string, bool, @enum }");
}

TEST_F(FormatTestCSharp, CSharpNullCoalescing) {
  verifyFormat("var test = ABC ?? DEF");
  verifyFormat("string myname = name ?? \"ABC\";");
  verifyFormat("return _name ?? \"DEF\";");
}

} // namespace format
} // end namespace clang
