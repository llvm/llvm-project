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
         const FormatStyle &Style = getMicrosoftStyle(FormatStyle::LK_CSharp)) {
    return format(Code, 0, Code.size(), Style);
  }

  static FormatStyle getStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getMicrosoftStyle(FormatStyle::LK_CSharp);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  static void verifyFormat(
      llvm::StringRef Code,
      const FormatStyle &Style = getMicrosoftStyle(FormatStyle::LK_CSharp)) {
    EXPECT_EQ(Code.str(), format(Code, Style)) << "Expected code is not stable";
    EXPECT_EQ(Code.str(), format(test::messUp(Code), Style));
  }
};

TEST_F(FormatTestCSharp, CSharpClass) {
  verifyFormat("public class SomeClass\n"
               "{\n"
               "    void f()\n"
               "    {\n"
               "    }\n"
               "    int g()\n"
               "    {\n"
               "        return 0;\n"
               "    }\n"
               "    void h()\n"
               "    {\n"
               "        while (true)\n"
               "            f();\n"
               "        for (;;)\n"
               "            f();\n"
               "        if (true)\n"
               "            f();\n"
               "    }\n"
               "}");

  // Ensure that small and empty classes are handled correctly with condensed
  // (Google C++-like) brace-breaking style.
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.BreakBeforeBraces = FormatStyle::BS_Attach;

  verifyFormat("public class SomeEmptyClass {}", Style);

  verifyFormat("public class SomeTinyClass {\n"
               "  int X;\n"
               "}",
               Style);
  verifyFormat("private class SomeTinyClass {\n"
               "  int X;\n"
               "}",
               Style);
  verifyFormat("protected class SomeTinyClass {\n"
               "  int X;\n"
               "}",
               Style);
  verifyFormat("internal class SomeTinyClass {\n"
               "  int X;\n"
               "}",
               Style);
}

TEST_F(FormatTestCSharp, AccessModifiers) {
  verifyFormat("public String toString()\n"
               "{\n"
               "}");
  verifyFormat("private String toString()\n"
               "{\n"
               "}");
  verifyFormat("protected String toString()\n"
               "{\n"
               "}");
  verifyFormat("internal String toString()\n"
               "{\n"
               "}");

  verifyFormat("public override String toString()\n"
               "{\n"
               "}");
  verifyFormat("private override String toString()\n"
               "{\n"
               "}");
  verifyFormat("protected override String toString()\n"
               "{\n"
               "}");
  verifyFormat("internal override String toString()\n"
               "{\n"
               "}");

  verifyFormat("internal static String toString()\n"
               "{\n"
               "}");
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
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;

  verifyFormat(
      "public Person(string firstName, string lastName, int? age=null)");

  verifyFormat("foo () {\n"
               "  switch (args?.Length) {}\n"
               "}",
               Style);

  verifyFormat("switch (args?.Length) {}", Style);

  verifyFormat("public static void Main(string[] args)\n"
               "{\n"
               "    string dirPath = args?[0];\n"
               "}");

  Style.SpaceBeforeParens = FormatStyle::SBPO_Never;

  verifyFormat("switch(args?.Length) {}", Style);
}

TEST_F(FormatTestCSharp, Attributes) {
  verifyFormat("[STAThread]\n"
               "static void Main(string[] args)\n"
               "{\n"
               "}");

  verifyFormat("[TestMethod]\n"
               "private class Test\n"
               "{\n"
               "}");

  verifyFormat("[TestMethod]\n"
               "protected class Test\n"
               "{\n"
               "}");

  verifyFormat("[TestMethod]\n"
               "internal class Test\n"
               "{\n"
               "}");

  verifyFormat("[TestMethod]\n"
               "class Test\n"
               "{\n"
               "}");

  verifyFormat("[TestMethod]\n"
               "[DeploymentItem(\"Test.txt\")]\n"
               "public class Test\n"
               "{\n"
               "}");

  verifyFormat("[System.AttributeUsage(System.AttributeTargets.Method)]\n"
               "[System.Runtime.InteropServices.ComVisible(true)]\n"
               "public sealed class STAThreadAttribute : Attribute\n"
               "{\n"
               "}");

  verifyFormat("[Verb(\"start\", HelpText = \"Starts the server listening on "
               "provided port\")]\n"
               "class Test\n"
               "{\n"
               "}");

  verifyFormat("[TestMethod]\n"
               "public string Host\n"
               "{\n"
               "    set;\n"
               "    get;\n"
               "}");

  verifyFormat("[TestMethod(\"start\", HelpText = \"Starts the server "
               "listening on provided host\")]\n"
               "public string Host\n"
               "{\n"
               "    set;\n"
               "    get;\n"
               "}");

  verifyFormat(
      "[DllImport(\"Hello\", EntryPoint = \"hello_world\")]\n"
      "// The const char* returned by hello_world must not be deleted.\n"
      "private static extern IntPtr HelloFromCpp();)");

  // Class attributes go on their own line and do not affect layout of
  // interfaces. Line wrapping decisions previously caused each interface to be
  // on its own line.
  verifyFormat("[SomeAttribute]\n"
               "[SomeOtherAttribute]\n"
               "public class A : IShape, IAnimal, IVehicle\n"
               "{\n"
               "    int X;\n"
               "}");

  // Attributes in a method declaration do not cause line wrapping.
  verifyFormat("void MethodA([In][Out] ref double x)\n"
               "{\n"
               "}");

  //  Unwrappable lines go on a line of their own.
  // 'target:' is not treated as a label.
  // Modify Style to enforce a column limit.
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.ColumnLimit = 10;
  verifyFormat(R"([assembly:InternalsVisibleTo(
    "SomeAssembly, PublicKey=SomePublicKeyThatExceedsTheColumnLimit")])",
               Style);
}

TEST_F(FormatTestCSharp, CSharpUsing) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;
  verifyFormat("public void foo () {\n"
               "  using (StreamWriter sw = new StreamWriter (filenameA)) {}\n"
               "  using () {}\n"
               "}",
               Style);

  // Ensure clang-format affects top-level snippets correctly.
  verifyFormat("using (StreamWriter sw = new StreamWriter (filenameB)) {}",
               Style);

  Style.SpaceBeforeParens = FormatStyle::SBPO_Never;
  verifyFormat("public void foo() {\n"
               "  using(StreamWriter sw = new StreamWriter(filenameB)) {}\n"
               "  using() {}\n"
               "}",
               Style);

  // Ensure clang-format affects top-level snippets correctly.
  verifyFormat("using(StreamWriter sw = new StreamWriter(filenameB)) {}",
               Style);

  Style.SpaceBeforeParens = FormatStyle::SBPO_ControlStatements;
  verifyFormat("public void foo() {\n"
               "  using (StreamWriter sw = new StreamWriter(filenameA)) {}\n"
               "  using () {}\n"
               "}",
               Style);

  // Ensure clang-format affects top-level snippets correctly.
  verifyFormat("using (StreamWriter sw = new StreamWriter(filenameB)) {}",
               Style);

  Style.SpaceBeforeParens = FormatStyle::SBPO_NonEmptyParentheses;
  verifyFormat("public void foo() {\n"
               "  using (StreamWriter sw = new StreamWriter (filenameA)) {}\n"
               "  using() {}\n"
               "}",
               Style);

  // Ensure clang-format affects top-level snippets correctly.
  verifyFormat("using (StreamWriter sw = new StreamWriter (filenameB)) {}",
               Style);
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

TEST_F(FormatTestCSharp, AttributesIndentation) {
  FormatStyle Style = getMicrosoftStyle(FormatStyle::LK_CSharp);
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_None;

  verifyFormat("[STAThread]\n"
               "static void Main(string[] args)\n"
               "{\n"
               "}",
               Style);

  verifyFormat("[STAThread]\n"
               "void "
               "veryLooooooooooooooongFunctionName(string[] args)\n"
               "{\n"
               "}",
               Style);

  verifyFormat("[STAThread]\n"
               "veryLoooooooooooooooooooongReturnType "
               "veryLooooooooooooooongFunctionName(string[] args)\n"
               "{\n"
               "}",
               Style);

  verifyFormat("[SuppressMessage(\"A\", \"B\", Justification = \"C\")]\n"
               "public override X Y()\n"
               "{\n"
               "}\n",
               Style);

  verifyFormat("[SuppressMessage]\n"
               "public X Y()\n"
               "{\n"
               "}\n",
               Style);

  verifyFormat("[SuppressMessage]\n"
               "public override X Y()\n"
               "{\n"
               "}\n",
               Style);

  verifyFormat("public A(B b) : base(b)\n"
               "{\n"
               "    [SuppressMessage]\n"
               "    public override X Y()\n"
               "    {\n"
               "    }\n"
               "}\n",
               Style);

  verifyFormat("public A : Base\n"
               "{\n"
               "}\n"
               "[Test]\n"
               "public Foo()\n"
               "{\n"
               "}\n",
               Style);

  verifyFormat("namespace\n"
               "{\n"
               "public A : Base\n"
               "{\n"
               "}\n"
               "[Test]\n"
               "public Foo()\n"
               "{\n"
               "}\n"
               "}\n",
               Style);
}

TEST_F(FormatTestCSharp, CSharpSpaceBefore) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;

  verifyFormat("List<string> list;", Style);
  verifyFormat("Dictionary<string, string> dict;", Style);

  verifyFormat("for (int i = 0; i < size (); i++) {\n"
               "}",
               Style);
  verifyFormat("foreach (var x in y) {\n"
               "}",
               Style);
  verifyFormat("switch (x) {}", Style);
  verifyFormat("do {\n"
               "} while (x);",
               Style);

  Style.SpaceBeforeParens = FormatStyle::SBPO_Never;

  verifyFormat("List<string> list;", Style);
  verifyFormat("Dictionary<string, string> dict;", Style);

  verifyFormat("for(int i = 0; i < size(); i++) {\n"
               "}",
               Style);
  verifyFormat("foreach(var x in y) {\n"
               "}",
               Style);
  verifyFormat("switch(x) {}", Style);
  verifyFormat("do {\n"
               "} while(x);",
               Style);
}

TEST_F(FormatTestCSharp, CSharpSpaceAfterCStyleCast) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  verifyFormat("(int)x / y;", Style);

  Style.SpaceAfterCStyleCast = true;
  verifyFormat("(int) x / y;", Style);
}

TEST_F(FormatTestCSharp, CSharpEscapedQuotesInVerbatimStrings) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  verifyFormat(R"(string str = @"""";)", Style);
  verifyFormat(R"(string str = @"""Hello world""";)", Style);
  verifyFormat(R"(string str = $@"""Hello {friend}""";)", Style);
}

TEST_F(FormatTestCSharp, CSharpQuotesInInterpolatedStrings) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  verifyFormat(R"(string str1 = $"{null ?? "null"}";)", Style);
  verifyFormat(R"(string str2 = $"{{{braceCount} braces";)", Style);
  verifyFormat(R"(string str3 = $"{braceCount}}} braces";)", Style);
}

TEST_F(FormatTestCSharp, CSharpNewlinesInVerbatimStrings) {
  // Use MS style as Google Style inserts a line break before multiline strings.

  // verifyFormat does not understand multiline C# string-literals
  // so check the format explicitly.

  FormatStyle Style = getMicrosoftStyle(FormatStyle::LK_CSharp);

  std::string Code = R"(string s1 = $@"some code:
  class {className} {{
    {className}() {{}}
  }}";)";

  EXPECT_EQ(Code, format(Code, Style));

  // Multiline string in the middle of a function call.
  Code = R"(
var x = foo(className, $@"some code:
  class {className} {{
    {className}() {{}}
  }}",
            y);)"; // y aligned with `className` arg.

  EXPECT_EQ(Code, format(Code, Style));

  // Interpolated string with embedded multiline string.
  Code = R"(Console.WriteLine($"{string.Join(@",
		", values)}");)";

  EXPECT_EQ(Code, format(Code, Style));
}

TEST_F(FormatTestCSharp, CSharpObjectInitializers) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  // Start code fragemnts with a comment line so that C++ raw string literals
  // as seen are identical to expected formatted code.

  verifyFormat(R"(//
Shape[] shapes = new[] {
    new Circle {
        Radius = 2.7281,
        Colour = Colours.Red,
    },
    new Square {
        Side = 101.1,
        Colour = Colours.Yellow,
    },
};)",
               Style);

  // Omitted final `,`s will change the formatting.
  verifyFormat(R"(//
Shape[] shapes = new[] {new Circle {Radius = 2.7281, Colour = Colours.Red},
                        new Square {
                            Side = 101.1,
                            Colour = Colours.Yellow,
                        }};)",
               Style);
}

} // namespace format
} // end namespace clang
