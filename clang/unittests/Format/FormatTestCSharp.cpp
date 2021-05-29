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

TEST_F(FormatTestCSharp, CSharpConditionalExpressions) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  // conditional expression is not seen as a NullConditional.
  verifyFormat("var y = A < B ? -1 : 1;", Style);
}

TEST_F(FormatTestCSharp, CSharpNullConditional) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;

  verifyFormat(
      "public Person(string firstName, string lastName, int? age = null)");

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
               "public string Host { set; get; }");

  // Adjacent properties should not cause line wrapping issues
  verifyFormat("[JsonProperty(\"foo\")]\n"
               "public string Foo { set; get; }\n"
               "[JsonProperty(\"bar\")]\n"
               "public string Bar { set; get; }\n"
               "[JsonProperty(\"bar\")]\n"
               "protected string Bar { set; get; }\n"
               "[JsonProperty(\"bar\")]\n"
               "internal string Bar { set; get; }");

  // Multiple attributes should always be split (not just the first ones)
  verifyFormat("[XmlIgnore]\n"
               "[JsonProperty(\"foo\")]\n"
               "public string Foo { set; get; }");

  verifyFormat("[XmlIgnore]\n"
               "[JsonProperty(\"foo\")]\n"
               "public string Foo { set; get; }\n"
               "[XmlIgnore]\n"
               "[JsonProperty(\"bar\")]\n"
               "public string Bar { set; get; }");

  verifyFormat("[XmlIgnore]\n"
               "[ScriptIgnore]\n"
               "[JsonProperty(\"foo\")]\n"
               "public string Foo { set; get; }\n"
               "[XmlIgnore]\n"
               "[ScriptIgnore]\n"
               "[JsonProperty(\"bar\")]\n"
               "public string Bar { set; get; }");

  verifyFormat("[TestMethod(\"start\", HelpText = \"Starts the server "
               "listening on provided host\")]\n"
               "public string Host { set; get; }");

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

  verifyFormat("void MethodA([In, Out] ref double x)\n"
               "{\n"
               "}");

  verifyFormat("void MethodA([In, Out] double[] x)\n"
               "{\n"
               "}");

  verifyFormat("void MethodA([In] double[] x)\n"
               "{\n"
               "}");

  verifyFormat("void MethodA(int[] x)\n"
               "{\n"
               "}");
  verifyFormat("void MethodA(int[][] x)\n"
               "{\n"
               "}");
  verifyFormat("void MethodA([] x)\n"
               "{\n"
               "}");

  verifyFormat("public void Log([CallerLineNumber] int line = -1, "
               "[CallerFilePath] string path = null,\n"
               "                [CallerMemberName] string name = null)\n"
               "{\n"
               "}");

  // [] in an attribute do not cause premature line wrapping or indenting.
  verifyFormat(R"(//
public class A
{
    [SomeAttribute(new[] { RED, GREEN, BLUE }, -1.0f, 1.0f)]
    [DoNotSerialize]
    public Data MemberVariable;
})");

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
  verifyFormat("public enum var\n"
               "{\n"
               "    none,\n"
               "    @string,\n"
               "    bool,\n"
               "    @enum\n"
               "}");
}

TEST_F(FormatTestCSharp, CSharpNullCoalescing) {
  verifyFormat("var test = ABC ?? DEF");
  verifyFormat("string myname = name ?? \"ABC\";");
  verifyFormat("return _name ?? \"DEF\";");
}

TEST_F(FormatTestCSharp, CSharpNullCoalescingAssignment) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.SpaceBeforeAssignmentOperators = true;

  verifyFormat(R"(test ??= ABC;)", Style);
  verifyFormat(R"(test ??= true;)", Style);

  Style.SpaceBeforeAssignmentOperators = false;

  verifyFormat(R"(test??= ABC;)", Style);
  verifyFormat(R"(test??= true;)", Style);
}

TEST_F(FormatTestCSharp, CSharpNullForgiving) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  verifyFormat("var test = null!;", Style);
  verifyFormat("string test = someFunctionCall()! + \"ABC\"!", Style);
  verifyFormat("int test = (1! + 2 + bar! + foo())!", Style);
  verifyFormat(R"(test ??= !foo!;)", Style);
  verifyFormat("test = !bar! ?? !foo!;", Style);
  verifyFormat("bool test = !(!true && !true! || !null && !null! || !false && "
               "!false! && !bar()! + (!foo()))!",
               Style);

  // Check that line break keeps identifier with the bang.
  Style.ColumnLimit = 14;

  verifyFormat("var test =\n"
               "    foo!;",
               Style);
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

TEST_F(FormatTestCSharp, CSharpLambdas) {
  FormatStyle GoogleStyle = getGoogleStyle(FormatStyle::LK_CSharp);
  FormatStyle MicrosoftStyle = getMicrosoftStyle(FormatStyle::LK_CSharp);

  verifyFormat(R"(//
class MyClass {
  Action<string> greet = name => {
    string greeting = $"Hello {name}!";
    Console.WriteLine(greeting);
  };
})",
               GoogleStyle);

  // Microsoft Style:
  // https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/statements-expressions-operators/lambda-expressions#statement-lambdas
  verifyFormat(R"(//
class MyClass
{
    Action<string> greet = name =>
    {
        string greeting = $"Hello {name}!";
        Console.WriteLine(greeting);
    };
})",
               MicrosoftStyle);
}

TEST_F(FormatTestCSharp, CSharpObjectInitializers) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  // Start code fragments with a comment line so that C++ raw string literals
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
Shape[] shapes = new[] { new Circle { Radius = 2.7281, Colour = Colours.Red },
                         new Square { Side = 101.1, Colour = Colours.Yellow } };)",
               Style);

  // Lambdas can be supplied as initialiser arguments.
  verifyFormat(R"(//
private Transformer _transformer = new X.Y {
  Filler = (Shape shape) => { return new Transform.Fill(shape, RED); },
  Scaler = (Shape shape) => { return new Transform.Resize(shape, 0.1); },
};)",
               Style);

  // Dictionary initialisation.
  verifyFormat(R"(//
var myDict = new Dictionary<string, string> {
  ["name"] = _donald,
  ["age"] = Convert.ToString(DateTime.Today.Year - 1934),
  ["type"] = _duck,
};)",
               Style);
}

TEST_F(FormatTestCSharp, CSharpArrayInitializers) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  verifyFormat(R"(//
private MySet<Node>[] setPoints = {
  new Point<Node>(),
  new Point<Node>(),
};)",
               Style);
}

TEST_F(FormatTestCSharp, CSharpNamedArguments) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  verifyFormat(R"(//
PrintOrderDetails(orderNum: 31, productName: "Red Mug", sellerName: "Gift Shop");)",
               Style);

  // Ensure that trailing comments do not cause problems.
  verifyFormat(R"(//
PrintOrderDetails(orderNum: 31, productName: "Red Mug",  // comment
                  sellerName: "Gift Shop");)",
               Style);

  verifyFormat(R"(foreach (var tickCount in task.Begin(seed: 0)) {)", Style);
}

TEST_F(FormatTestCSharp, CSharpPropertyAccessors) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  verifyFormat("int Value { get }", Style);
  verifyFormat("int Value { get; }", Style);
  verifyFormat("int Value { internal get; }", Style);
  verifyFormat("int Value { get; } = 0", Style);
  verifyFormat("int Value { set }", Style);
  verifyFormat("int Value { set; }", Style);
  verifyFormat("int Value { internal set; }", Style);
  verifyFormat("int Value { set; } = 0", Style);
  verifyFormat("int Value { get; set }", Style);
  verifyFormat("int Value { set; get }", Style);
  verifyFormat("int Value { get; private set; }", Style);
  verifyFormat("int Value { get; set; }", Style);
  verifyFormat("int Value { get; set; } = 0", Style);
  verifyFormat("int Value { internal get; internal set; }", Style);

  // Do not wrap expression body definitions.
  verifyFormat(R"(//
public string Name {
  get => _name;
  set => _name = value;
})",
               Style);

  // Examples taken from
  // https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/classes-and-structs/properties
  verifyFormat(R"(
// Expression body definitions
public class SaleItem {
  public decimal Price {
    get => _cost;
    set => _cost = value;
  }
})",
               Style);

  verifyFormat(R"(
// Properties with backing fields
class TimePeriod {
  public double Hours {
    get { return _seconds / 3600; }
    set {
      if (value < 0 || value > 24)
        throw new ArgumentOutOfRangeException($"{nameof(value)} must be between 0 and 24.");
      _seconds = value * 3600;
    }
  }
})",
               Style);

  verifyFormat(R"(
// Auto-implemented properties
public class SaleItem {
  public decimal Price { get; set; }
})",
               Style);

  // Add column limit to wrap long lines.
  Style.ColumnLimit = 100;

  // Examples with assignment to default value.
  verifyFormat(R"(
// Long assignment to default value
class MyClass {
  public override VeryLongNamedTypeIndeed VeryLongNamedValue { get; set } =
      VeryLongNamedTypeIndeed.Create(DefaultFirstArgument, DefaultSecondArgument,
                                     DefaultThirdArgument);
})",
               Style);

  verifyFormat(R"(
// Long assignment to default value with expression body
class MyClass {
  public override VeryLongNamedTypeIndeed VeryLongNamedValue {
    get => veryLongNamedField;
    set => veryLongNamedField = value;
  } = VeryLongNamedTypeIndeed.Create(DefaultFirstArgument, DefaultSecondArgument,
                                     DefaultThirdArgument);
})",
               Style);

  // Brace wrapping and single-lining of accessor can be controlled by config.
  Style.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Never;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterFunction = true;

  verifyFormat(R"(//
class TimePeriod {
  public double Hours
  {
    get {
      return _seconds / 3600;
    }
    set {
      _seconds = value * 3600;
    }
  }
})",
               Style);

  // Microsoft style trivial property accessors have no line break before the
  // opening brace.
  auto MicrosoftStyle = getMicrosoftStyle(FormatStyle::LK_CSharp);
  verifyFormat(R"(//
public class SaleItem
{
    public decimal Price { get; set; }
})",
               MicrosoftStyle);
}

TEST_F(FormatTestCSharp, CSharpSpaces) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.SpaceBeforeSquareBrackets = false;
  Style.SpacesInSquareBrackets = false;
  Style.SpaceBeforeCpp11BracedList = true;
  Style.Cpp11BracedListStyle = false;
  Style.SpacesInContainerLiterals = false;
  Style.SpaceAfterCStyleCast = false;

  verifyFormat(R"(new Car { "Door", 0.1 })", Style);
  verifyFormat(R"(new Car { 0.1, "Door" })", Style);
  verifyFormat(R"(new string[] { "A" })", Style);
  verifyFormat(R"(new string[] {})", Style);
  verifyFormat(R"(new Car { someVariableName })", Style);
  verifyFormat(R"(new Car { someVariableName })", Style);
  verifyFormat(R"(new Dictionary<string, string> { ["Key"] = "Value" };)",
               Style);
  verifyFormat(R"(Apply(x => x.Name, x => () => x.ID);)", Style);
  verifyFormat(R"(bool[] xs = { true, true };)", Style);
  verifyFormat(R"(taskContext.Factory.Run(async () => doThing(args);)", Style);
  verifyFormat(R"(catch (TestException) when (innerFinallyExecuted))", Style);
  verifyFormat(R"(private float[,] Values;)", Style);
  verifyFormat(R"(Result this[Index x] => Foo(x);)", Style);

  verifyFormat(R"(char[,,] rawCharArray = MakeCharacterGrid();)", Style);
  verifyFormat(R"(var (key, value))", Style);

  // `&&` is not seen as a reference.
  verifyFormat(R"(A == typeof(X) && someBool)", Style);

  // Not seen as a C-style cast.
  verifyFormat(R"(//
foreach ((A a, B b) in someList) {
})",
               Style);

  // space after lock in `lock (processes)`.
  verifyFormat("lock (process)", Style);

  Style.SpacesInSquareBrackets = true;
  verifyFormat(R"(private float[ , ] Values;)", Style);
  verifyFormat(R"(string dirPath = args?[ 0 ];)", Style);
  verifyFormat(R"(char[ ,, ] rawCharArray = MakeCharacterGrid();)", Style);

  // Method returning tuple
  verifyFormat(R"(public (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(private (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(protected (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(virtual (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(extern (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(static (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(internal (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(abstract (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(sealed (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(override (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(async (string name, int age) methodTuple() {})", Style);
  verifyFormat(R"(unsafe (string name, int age) methodTuple() {})", Style);
}

TEST_F(FormatTestCSharp, CSharpNullableTypes) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);
  Style.SpacesInSquareBrackets = false;

  verifyFormat(R"(//
public class A {
  void foo() {
    int? value = some.bar();
  }
})",
               Style); // int? is nullable not a conditional expression.

  verifyFormat(R"(void foo(int? x, int? y, int? z) {})",
               Style); // Nullables in function definitions.

  verifyFormat(R"(public float? Value;)", Style); // no space before `?`.

  verifyFormat(R"(int?[] arr = new int?[10];)",
               Style); // An array of a nullable type.

  verifyFormat(R"(var x = (int?)y;)", Style); // Cast to a nullable type.

  verifyFormat(R"(var x = new MyContainer<int?>();)", Style); // Generics.

  verifyFormat(R"(//
public interface I {
  int? Function();
})",
               Style); // Interface methods.

  Style.ColumnLimit = 10;
  verifyFormat(R"(//
public VeryLongType? Function(
    int arg1,
    int arg2) {
  //
})",
               Style); // ? sticks with identifier.
}

TEST_F(FormatTestCSharp, CSharpArraySubscripts) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  // Do not format array subscript operators as attributes.
  verifyFormat(R"(//
if (someThings[index].Contains(myThing)) {
})",
               Style);

  verifyFormat(R"(//
if (someThings[i][j][k].Contains(myThing)) {
})",
               Style);
}

TEST_F(FormatTestCSharp, CSharpGenericTypeConstraints) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  EXPECT_TRUE(Style.BraceWrapping.SplitEmptyRecord);

  verifyFormat("class ItemFactory<T>\n"
               "    where T : new() {\n"
               "}",
               Style);

  verifyFormat("class Dictionary<TKey, TVal>\n"
               "    where TKey : IComparable<TKey>\n"
               "    where TVal : IMyInterface {\n"
               "  public void MyMethod<T>(T t)\n"
               "      where T : IMyInterface {\n"
               "    doThing();\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("class ItemFactory<T>\n"
               "    where T : new(), IAnInterface<T>, IAnotherInterface<T>, "
               "IAnotherInterfaceStill<T> {\n"
               "}",
               Style);

  Style.ColumnLimit = 50; // Force lines to be wrapped.
  verifyFormat(R"(//
class ItemFactory<T, U>
    where T : new(),
              IAnInterface<T>,
              IAnotherInterface<T, U>,
              IAnotherInterfaceStill<T, U> {
})",
               Style);

  // In other languages `where` can be used as a normal identifier.
  // This example is in C++!
  verifyFormat(R"(//
class A {
  int f(int where) {}
};)",
               getGoogleStyle(FormatStyle::LK_Cpp));
}

} // namespace format
} // end namespace clang
