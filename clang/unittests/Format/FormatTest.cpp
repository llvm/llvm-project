//===- unittest/Format/FormatTest.cpp - Formatting unit tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {
namespace test {
namespace {

class FormatTest : public test::FormatTestBase {};

TEST_F(FormatTest, MessUp) {
  EXPECT_EQ("1 2 3", messUp("1 2 3"));
  EXPECT_EQ("1 2 3", messUp("1\n2\n3"));
  EXPECT_EQ("a\n//b\nc", messUp("a\n//b\nc"));
  EXPECT_EQ("a\n#b\nc", messUp("a\n#b\nc"));
  EXPECT_EQ("a\n#b c d\ne", messUp("a\n#b\\\nc\\\nd\ne"));
}

TEST_F(FormatTest, DefaultLLVMStyleIsCpp) {
  EXPECT_EQ(FormatStyle::LK_Cpp, getLLVMStyle().Language);
}

TEST_F(FormatTest, LLVMStyleOverride) {
  EXPECT_EQ(FormatStyle::LK_Proto,
            getLLVMStyle(FormatStyle::LK_Proto).Language);
}

//===----------------------------------------------------------------------===//
// Basic function tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, DoesNotChangeCorrectlyFormattedCode) { verifyFormat(";"); }

TEST_F(FormatTest, FormatsGlobalStatementsAt0) {
  verifyFormat("int i;", "  int i;");
  verifyFormat("\nint i;", " \n\t \v \f  int i;");
  verifyFormat("int i;\nint j;", "    int i; int j;");
  verifyFormat("int i;\nint j;", "    int i;\n  int j;");
}

TEST_F(FormatTest, FormatsUnwrappedLinesAtFirstFormat) {
  verifyFormat("int i;", "int\ni;");
}

TEST_F(FormatTest, FormatsNestedBlockStatements) {
  verifyFormat("{\n  {\n    {}\n  }\n}", "{{{}}}");
}

TEST_F(FormatTest, FormatsNestedCall) {
  verifyFormat("Method(f1, f2(f3));");
  verifyFormat("Method(f1(f2, f3()));");
  verifyFormat("Method(f1(f2, (f3())));");
}

TEST_F(FormatTest, NestedNameSpecifiers) {
  verifyFormat("vector<::Type> v;");
  verifyFormat("::ns::SomeFunction(::ns::SomeOtherFunction())");
  verifyFormat("static constexpr bool Bar = decltype(bar())::value;");
  verifyFormat("static constexpr bool Bar = typeof(bar())::value;");
  verifyFormat("static constexpr bool Bar = __underlying_type(bar())::value;");
  verifyFormat("static constexpr bool Bar = _Atomic(bar())::value;");
  verifyFormat("bool a = 2 < ::SomeFunction();");
  verifyFormat("ALWAYS_INLINE ::std::string getName();");
  verifyFormat("some::string getName();");
}

TEST_F(FormatTest, OnlyGeneratesNecessaryReplacements) {
  verifyFormat("if (a) {\n"
               "  f();\n"
               "}",
               "if(a){f();}");
  EXPECT_EQ(4, ReplacementCount);
  verifyNoChange("if (a) {\n"
                 "  f();\n"
                 "}");
  EXPECT_EQ(0, ReplacementCount);
  verifyNoChange("/*\r\n"
                 "\r\n"
                 "*/");
  EXPECT_EQ(0, ReplacementCount);
}

TEST_F(FormatTest, RemovesEmptyLines) {
  verifyFormat("class C {\n"
               "  int i;\n"
               "};",
               "class C {\n"
               " int i;\n"
               "\n"
               "};");

  // Don't remove empty lines at the start of namespaces or extern "C" blocks.
  verifyFormat("namespace N {\n"
               "\n"
               "int i;\n"
               "}",
               "namespace N {\n"
               "\n"
               "int    i;\n"
               "}",
               getGoogleStyle());
  verifyFormat("/* something */ namespace N {\n"
               "\n"
               "int i;\n"
               "}",
               "/* something */ namespace N {\n"
               "\n"
               "int    i;\n"
               "}",
               getGoogleStyle());
  verifyFormat("inline namespace N {\n"
               "\n"
               "int i;\n"
               "}",
               "inline namespace N {\n"
               "\n"
               "int    i;\n"
               "}",
               getGoogleStyle());
  verifyFormat("/* something */ inline namespace N {\n"
               "\n"
               "int i;\n"
               "}",
               "/* something */ inline namespace N {\n"
               "\n"
               "int    i;\n"
               "}",
               getGoogleStyle());
  verifyFormat("export namespace N {\n"
               "\n"
               "int i;\n"
               "}",
               "export namespace N {\n"
               "\n"
               "int    i;\n"
               "}",
               getGoogleStyle());
  verifyFormat("extern /**/ \"C\" /**/ {\n"
               "\n"
               "int i;\n"
               "}",
               "extern /**/ \"C\" /**/ {\n"
               "\n"
               "int    i;\n"
               "}",
               getGoogleStyle());

  auto CustomStyle = getLLVMStyle();
  CustomStyle.BreakBeforeBraces = FormatStyle::BS_Custom;
  CustomStyle.BraceWrapping.AfterNamespace = true;
  CustomStyle.KeepEmptyLinesAtTheStartOfBlocks = false;
  verifyFormat("namespace N\n"
               "{\n"
               "\n"
               "int i;\n"
               "}",
               "namespace N\n"
               "{\n"
               "\n"
               "\n"
               "int    i;\n"
               "}",
               CustomStyle);
  verifyFormat("/* something */ namespace N\n"
               "{\n"
               "\n"
               "int i;\n"
               "}",
               "/* something */ namespace N {\n"
               "\n"
               "\n"
               "int    i;\n"
               "}",
               CustomStyle);
  verifyFormat("inline namespace N\n"
               "{\n"
               "\n"
               "int i;\n"
               "}",
               "inline namespace N\n"
               "{\n"
               "\n"
               "\n"
               "int    i;\n"
               "}",
               CustomStyle);
  verifyFormat("/* something */ inline namespace N\n"
               "{\n"
               "\n"
               "int i;\n"
               "}",
               "/* something */ inline namespace N\n"
               "{\n"
               "\n"
               "int    i;\n"
               "}",
               CustomStyle);
  verifyFormat("export namespace N\n"
               "{\n"
               "\n"
               "int i;\n"
               "}",
               "export namespace N\n"
               "{\n"
               "\n"
               "int    i;\n"
               "}",
               CustomStyle);
  verifyFormat("namespace a\n"
               "{\n"
               "namespace b\n"
               "{\n"
               "\n"
               "class AA {};\n"
               "\n"
               "} // namespace b\n"
               "} // namespace a",
               "namespace a\n"
               "{\n"
               "namespace b\n"
               "{\n"
               "\n"
               "\n"
               "class AA {};\n"
               "\n"
               "\n"
               "}\n"
               "}",
               CustomStyle);
  verifyFormat("namespace A /* comment */\n"
               "{\n"
               "class B {}\n"
               "} // namespace A",
               "namespace A /* comment */ { class B {} }", CustomStyle);
  verifyFormat("namespace A\n"
               "{ /* comment */\n"
               "class B {}\n"
               "} // namespace A",
               "namespace A {/* comment */ class B {} }", CustomStyle);
  verifyFormat("namespace A\n"
               "{ /* comment */\n"
               "\n"
               "class B {}\n"
               "\n"
               ""
               "} // namespace A",
               "namespace A { /* comment */\n"
               "\n"
               "\n"
               "class B {}\n"
               "\n"
               "\n"
               "}",
               CustomStyle);
  verifyFormat("namespace A /* comment */\n"
               "{\n"
               "\n"
               "class B {}\n"
               "\n"
               "} // namespace A",
               "namespace A/* comment */ {\n"
               "\n"
               "\n"
               "class B {}\n"
               "\n"
               "\n"
               "}",
               CustomStyle);

  // ...but do keep inlining and removing empty lines for non-block extern "C"
  // functions.
  verifyGoogleFormat("extern \"C\" int f() { return 42; }");
  verifyFormat("extern \"C\" int f() {\n"
               "  int i = 42;\n"
               "  return i;\n"
               "}",
               "extern \"C\" int f() {\n"
               "\n"
               "  int i = 42;\n"
               "  return i;\n"
               "}",
               getGoogleStyle());

  // Remove empty lines at the beginning and end of blocks.
  verifyFormat("void f() {\n"
               "\n"
               "  if (a) {\n"
               "\n"
               "    f();\n"
               "  }\n"
               "}",
               "void f() {\n"
               "\n"
               "  if (a) {\n"
               "\n"
               "    f();\n"
               "\n"
               "  }\n"
               "\n"
               "}");
  verifyFormat("void f() {\n"
               "  if (a) {\n"
               "    f();\n"
               "  }\n"
               "}",
               "void f() {\n"
               "\n"
               "  if (a) {\n"
               "\n"
               "    f();\n"
               "\n"
               "  }\n"
               "\n"
               "}",
               getGoogleStyle());

  // Don't remove empty lines in more complex control statements.
  verifyFormat("void f() {\n"
               "  if (a) {\n"
               "    f();\n"
               "\n"
               "  } else if (b) {\n"
               "    f();\n"
               "  }\n"
               "}",
               "void f() {\n"
               "  if (a) {\n"
               "    f();\n"
               "\n"
               "  } else if (b) {\n"
               "    f();\n"
               "\n"
               "  }\n"
               "\n"
               "}");

  // Don't remove empty lines before namespace endings.
  FormatStyle LLVMWithNoNamespaceFix = getLLVMStyle();
  LLVMWithNoNamespaceFix.FixNamespaceComments = false;
  verifyNoChange("namespace {\n"
                 "int i;\n"
                 "\n"
                 "}",
                 LLVMWithNoNamespaceFix);
  verifyFormat("namespace {\n"
               "int i;\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyNoChange("namespace {\n"
                 "int i;\n"
                 "\n"
                 "};",
                 LLVMWithNoNamespaceFix);
  verifyFormat("namespace {\n"
               "int i;\n"
               "};",
               LLVMWithNoNamespaceFix);
  verifyNoChange("namespace {\n"
                 "int i;\n"
                 "\n"
                 "}");
  verifyFormat("namespace {\n"
               "int i;\n"
               "\n"
               "} // namespace",
               "namespace {\n"
               "int i;\n"
               "\n"
               "}  // namespace");

  FormatStyle Style = getLLVMStyle();
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  Style.MaxEmptyLinesToKeep = 2;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterClass = true;
  Style.BraceWrapping.AfterFunction = true;
  Style.KeepEmptyLinesAtTheStartOfBlocks = false;

  verifyFormat("class Foo\n"
               "{\n"
               "  Foo() {}\n"
               "\n"
               "  void funk() {}\n"
               "};",
               "class Foo\n"
               "{\n"
               "  Foo()\n"
               "  {\n"
               "  }\n"
               "\n"
               "  void funk() {}\n"
               "};",
               Style);
}

TEST_F(FormatTest, RecognizesBinaryOperatorKeywords) {
  verifyFormat("x = (a) and (b);");
  verifyFormat("x = (a) or (b);");
  verifyFormat("x = (a) bitand (b);");
  verifyFormat("x = (a) bitor (b);");
  verifyFormat("x = (a) not_eq (b);");
  verifyFormat("x = (a) and_eq (b);");
  verifyFormat("x = (a) or_eq (b);");
  verifyFormat("x = (a) xor (b);");
}

TEST_F(FormatTest, RecognizesUnaryOperatorKeywords) {
  verifyFormat("x = compl(a);");
  verifyFormat("x = not(a);");
  verifyFormat("x = bitand(a);");
  // Unary operator must not be merged with the next identifier
  verifyFormat("x = compl a;");
  verifyFormat("x = not a;");
  verifyFormat("x = bitand a;");
}

//===----------------------------------------------------------------------===//
// Tests for control statements.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, FormatIfWithoutCompoundStatement) {
  verifyFormat("if (true)\n  f();\ng();");
  verifyFormat("if (a)\n  if (b)\n    if (c)\n      g();\nh();");
  verifyFormat("if (a)\n  if (b) {\n    f();\n  }\ng();");
  verifyFormat("if constexpr (true)\n"
               "  f();\ng();");
  verifyFormat("if CONSTEXPR (true)\n"
               "  f();\ng();");
  verifyFormat("if constexpr (a)\n"
               "  if constexpr (b)\n"
               "    if constexpr (c)\n"
               "      g();\n"
               "h();");
  verifyFormat("if CONSTEXPR (a)\n"
               "  if CONSTEXPR (b)\n"
               "    if CONSTEXPR (c)\n"
               "      g();\n"
               "h();");
  verifyFormat("if constexpr (a)\n"
               "  if constexpr (b) {\n"
               "    f();\n"
               "  }\n"
               "g();");
  verifyFormat("if CONSTEXPR (a)\n"
               "  if CONSTEXPR (b) {\n"
               "    f();\n"
               "  }\n"
               "g();");

  verifyFormat("if consteval {\n}");
  verifyFormat("if !consteval {\n}");
  verifyFormat("if not consteval {\n}");
  verifyFormat("if consteval {\n} else {\n}");
  verifyFormat("if !consteval {\n} else {\n}");
  verifyFormat("if consteval {\n"
               "  f();\n"
               "}");
  verifyFormat("if !consteval {\n"
               "  f();\n"
               "}");
  verifyFormat("if consteval {\n"
               "  f();\n"
               "} else {\n"
               "  g();\n"
               "}");
  verifyFormat("if CONSTEVAL {\n"
               "  f();\n"
               "}");
  verifyFormat("if !CONSTEVAL {\n"
               "  f();\n"
               "}");

  verifyFormat("if (a)\n"
               "  g();");
  verifyFormat("if (a) {\n"
               "  g()\n"
               "};");
  verifyFormat("if (a)\n"
               "  g();\n"
               "else\n"
               "  g();");
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else\n"
               "  g();");
  verifyFormat("if (a)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}");
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}");
  verifyFormat("if (a)\n"
               "  g();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();");
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();");
  verifyFormat("if (a)\n"
               "  g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else\n"
               "  g();");
  verifyFormat("if (a)\n"
               "  g();\n"
               "else if (b)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}");
  verifyFormat("if (a)\n"
               "  g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}");
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}");

  FormatStyle AllowsMergedIf = getLLVMStyle();
  AllowsMergedIf.IfMacros.push_back("MYIF");
  AllowsMergedIf.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  AllowsMergedIf.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_WithoutElse;
  verifyFormat("if (a)\n"
               "  // comment\n"
               "  f();",
               AllowsMergedIf);
  verifyFormat("{\n"
               "  if (a)\n"
               "  label:\n"
               "    f();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("#define A \\\n"
               "  if (a)  \\\n"
               "  label:  \\\n"
               "    f()",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  ;",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  if (b) return;",
               AllowsMergedIf);

  verifyFormat("if (a) // Can't merge this\n"
               "  f();",
               AllowsMergedIf);
  verifyFormat("if (a) /* still don't merge */\n"
               "  f();",
               AllowsMergedIf);
  verifyFormat("if (a) { // Never merge this\n"
               "  f();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) { /* Never merge this */\n"
               "  f();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  // comment\n"
               "  f();",
               AllowsMergedIf);
  verifyFormat("{\n"
               "  MYIF (a)\n"
               "  label:\n"
               "    f();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("#define A  \\\n"
               "  MYIF (a) \\\n"
               "  label:   \\\n"
               "    f()",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  ;",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  MYIF (b) return;",
               AllowsMergedIf);

  verifyFormat("MYIF (a) // Can't merge this\n"
               "  f();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) /* still don't merge */\n"
               "  f();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) { // Never merge this\n"
               "  f();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) { /* Never merge this */\n"
               "  f();\n"
               "}",
               AllowsMergedIf);

  AllowsMergedIf.ColumnLimit = 14;
  // Where line-lengths matter, a 2-letter synonym that maintains line length.
  // Not IF to avoid any confusion that IF is somehow special.
  AllowsMergedIf.IfMacros.push_back("FI");
  verifyFormat("if (a) return;", AllowsMergedIf);
  verifyFormat("if (aaaaaaaaa)\n"
               "  return;",
               AllowsMergedIf);
  verifyFormat("FI (a) return;", AllowsMergedIf);
  verifyFormat("FI (aaaaaaaaa)\n"
               "  return;",
               AllowsMergedIf);

  AllowsMergedIf.ColumnLimit = 13;
  verifyFormat("if (a)\n  return;", AllowsMergedIf);
  verifyFormat("FI (a)\n  return;", AllowsMergedIf);

  FormatStyle AllowsMergedIfElse = getLLVMStyle();
  AllowsMergedIfElse.IfMacros.push_back("MYIF");
  AllowsMergedIfElse.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_AllIfsAndElse;
  verifyFormat("if (a)\n"
               "  // comment\n"
               "  f();\n"
               "else\n"
               "  // comment\n"
               "  f();",
               AllowsMergedIfElse);
  verifyFormat("{\n"
               "  if (a)\n"
               "  label:\n"
               "    f();\n"
               "  else\n"
               "  label:\n"
               "    f();\n"
               "}",
               AllowsMergedIfElse);
  verifyFormat("if (a)\n"
               "  ;\n"
               "else\n"
               "  ;",
               AllowsMergedIfElse);
  verifyFormat("if (a) {\n"
               "} else {\n"
               "}",
               AllowsMergedIfElse);
  verifyFormat("if (a) return;\n"
               "else if (b) return;\n"
               "else return;",
               AllowsMergedIfElse);
  verifyFormat("if (a) {\n"
               "} else return;",
               AllowsMergedIfElse);
  verifyFormat("if (a) {\n"
               "} else if (b) return;\n"
               "else return;",
               AllowsMergedIfElse);
  verifyFormat("if (a) return;\n"
               "else if (b) {\n"
               "} else return;",
               AllowsMergedIfElse);
  verifyFormat("if (a)\n"
               "  if (b) return;\n"
               "  else return;",
               AllowsMergedIfElse);
  verifyFormat("if constexpr (a)\n"
               "  if constexpr (b) return;\n"
               "  else if constexpr (c) return;\n"
               "  else return;",
               AllowsMergedIfElse);
  verifyFormat("MYIF (a)\n"
               "  // comment\n"
               "  f();\n"
               "else\n"
               "  // comment\n"
               "  f();",
               AllowsMergedIfElse);
  verifyFormat("{\n"
               "  MYIF (a)\n"
               "  label:\n"
               "    f();\n"
               "  else\n"
               "  label:\n"
               "    f();\n"
               "}",
               AllowsMergedIfElse);
  verifyFormat("MYIF (a)\n"
               "  ;\n"
               "else\n"
               "  ;",
               AllowsMergedIfElse);
  verifyFormat("MYIF (a) {\n"
               "} else {\n"
               "}",
               AllowsMergedIfElse);
  verifyFormat("MYIF (a) return;\n"
               "else MYIF (b) return;\n"
               "else return;",
               AllowsMergedIfElse);
  verifyFormat("MYIF (a) {\n"
               "} else return;",
               AllowsMergedIfElse);
  verifyFormat("MYIF (a) {\n"
               "} else MYIF (b) return;\n"
               "else return;",
               AllowsMergedIfElse);
  verifyFormat("MYIF (a) return;\n"
               "else MYIF (b) {\n"
               "} else return;",
               AllowsMergedIfElse);
  verifyFormat("MYIF (a)\n"
               "  MYIF (b) return;\n"
               "  else return;",
               AllowsMergedIfElse);
  verifyFormat("MYIF constexpr (a)\n"
               "  MYIF constexpr (b) return;\n"
               "  else MYIF constexpr (c) return;\n"
               "  else return;",
               AllowsMergedIfElse);
}

TEST_F(FormatTest, FormatIfWithoutCompoundStatementButElseWith) {
  FormatStyle AllowsMergedIf = getLLVMStyle();
  AllowsMergedIf.IfMacros.push_back("MYIF");
  AllowsMergedIf.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  AllowsMergedIf.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_WithoutElse;
  verifyFormat("if (a)\n"
               "  f();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  f();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);

  verifyFormat("if (a) g();", AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g()\n"
               "};",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  g();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  g();\n"
               "else if (b)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  f();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  f();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);

  verifyFormat("MYIF (a) g();", AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g()\n"
               "};",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else MYIF (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else MYIF (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else MYIF (b) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else MYIF (b)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else if (b)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else MYIF (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a)\n"
               "  g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else MYIF (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);

  AllowsMergedIf.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_OnlyFirstIf;

  verifyFormat("if (a) f();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) f();\n"
               "else {\n"
               "  if (a) f();\n"
               "  else {\n"
               "    g();\n"
               "  }\n"
               "  g();\n"
               "}",
               AllowsMergedIf);

  verifyFormat("if (a) g();", AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g()\n"
               "};",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else if (b)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) f();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) f();\n"
               "else {\n"
               "  if (a) f();\n"
               "  else {\n"
               "    g();\n"
               "  }\n"
               "  g();\n"
               "}",
               AllowsMergedIf);

  verifyFormat("MYIF (a) g();", AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g()\n"
               "};",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else MYIF (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else MYIF (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else if (b)\n"
               "  g();\n"
               "else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else MYIF (b) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else\n"
               "  g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else MYIF (b)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else if (b)\n"
               "  g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else MYIF (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else MYIF (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);

  AllowsMergedIf.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_AllIfsAndElse;

  verifyFormat("if (a) f();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) f();\n"
               "else {\n"
               "  if (a) f();\n"
               "  else {\n"
               "    g();\n"
               "  }\n"
               "  g();\n"
               "}",
               AllowsMergedIf);

  verifyFormat("if (a) g();", AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g()\n"
               "};",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else g();",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else g();",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else if (b) g();\n"
               "else g();",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else if (b) g();\n"
               "else g();",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else g();",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else if (b) g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) {\n"
               "  g();\n"
               "} else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) f();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) f();\n"
               "else {\n"
               "  if (a) f();\n"
               "  else {\n"
               "    g();\n"
               "  }\n"
               "  g();\n"
               "}",
               AllowsMergedIf);

  verifyFormat("MYIF (a) g();", AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g()\n"
               "};",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else MYIF (b) g();\n"
               "else g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else if (b) g();\n"
               "else g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else MYIF (b) g();\n"
               "else g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else if (b) g();\n"
               "else g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else MYIF (b) {\n"
               "  g();\n"
               "} else g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else g();",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else MYIF (b) g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else if (b) g();\n"
               "else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else MYIF (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) g();\n"
               "else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else MYIF (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("MYIF (a) {\n"
               "  g();\n"
               "} else if (b) {\n"
               "  g();\n"
               "} else {\n"
               "  g();\n"
               "}",
               AllowsMergedIf);
}

TEST_F(FormatTest, FormatLoopsWithoutCompoundStatement) {
  FormatStyle AllowsMergedLoops = getLLVMStyle();
  AllowsMergedLoops.AllowShortLoopsOnASingleLine = true;
  verifyFormat("while (true) continue;", AllowsMergedLoops);
  verifyFormat("for (;;) continue;", AllowsMergedLoops);
  verifyFormat("for (int &v : vec) v *= 2;", AllowsMergedLoops);
  verifyFormat("BOOST_FOREACH (int &v, vec) v *= 2;", AllowsMergedLoops);
  verifyFormat("while (true)\n"
               "  ;",
               AllowsMergedLoops);
  verifyFormat("for (;;)\n"
               "  ;",
               AllowsMergedLoops);
  verifyFormat("for (;;)\n"
               "  for (;;) continue;",
               AllowsMergedLoops);
  verifyFormat("for (;;)\n"
               "  while (true) continue;",
               AllowsMergedLoops);
  verifyFormat("while (true)\n"
               "  for (;;) continue;",
               AllowsMergedLoops);
  verifyFormat("BOOST_FOREACH (int &v, vec)\n"
               "  for (;;) continue;",
               AllowsMergedLoops);
  verifyFormat("for (;;)\n"
               "  BOOST_FOREACH (int &v, vec) continue;",
               AllowsMergedLoops);
  verifyFormat("for (;;) // Can't merge this\n"
               "  continue;",
               AllowsMergedLoops);
  verifyFormat("for (;;) /* still don't merge */\n"
               "  continue;",
               AllowsMergedLoops);
  verifyFormat("do a++;\n"
               "while (true);",
               AllowsMergedLoops);
  verifyFormat("do /* Don't merge */\n"
               "  a++;\n"
               "while (true);",
               AllowsMergedLoops);
  verifyFormat("do // Don't merge\n"
               "  a++;\n"
               "while (true);",
               AllowsMergedLoops);
  verifyFormat("do\n"
               "  // Don't merge\n"
               "  a++;\n"
               "while (true);",
               AllowsMergedLoops);
  // Without braces labels are interpreted differently.
  verifyFormat("{\n"
               "  do\n"
               "  label:\n"
               "    a++;\n"
               "  while (true);\n"
               "}",
               AllowsMergedLoops);
}

TEST_F(FormatTest, FormatShortBracedStatements) {
  FormatStyle AllowSimpleBracedStatements = getLLVMStyle();
  EXPECT_EQ(AllowSimpleBracedStatements.AllowShortBlocksOnASingleLine, false);
  EXPECT_EQ(AllowSimpleBracedStatements.AllowShortIfStatementsOnASingleLine,
            FormatStyle::SIS_Never);
  EXPECT_EQ(AllowSimpleBracedStatements.AllowShortLoopsOnASingleLine, false);
  EXPECT_EQ(AllowSimpleBracedStatements.BraceWrapping.AfterFunction, false);
  verifyFormat("for (;;) {\n"
               "  f();\n"
               "}");
  verifyFormat("/*comment*/ for (;;) {\n"
               "  f();\n"
               "}");
  verifyFormat("BOOST_FOREACH (int v, vec) {\n"
               "  f();\n"
               "}");
  verifyFormat("/*comment*/ BOOST_FOREACH (int v, vec) {\n"
               "  f();\n"
               "}");
  verifyFormat("while (true) {\n"
               "  f();\n"
               "}");
  verifyFormat("/*comment*/ while (true) {\n"
               "  f();\n"
               "}");
  verifyFormat("if (true) {\n"
               "  f();\n"
               "}");
  verifyFormat("/*comment*/ if (true) {\n"
               "  f();\n"
               "}");

  AllowSimpleBracedStatements.AllowShortBlocksOnASingleLine =
      FormatStyle::SBS_Empty;
  AllowSimpleBracedStatements.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_WithoutElse;
  verifyFormat("if (true) {}", AllowSimpleBracedStatements);
  verifyFormat("if (i) break;", AllowSimpleBracedStatements);
  verifyFormat("if (i > 0) {\n"
               "  return i;\n"
               "}",
               AllowSimpleBracedStatements);

  AllowSimpleBracedStatements.IfMacros.push_back("MYIF");
  // Where line-lengths matter, a 2-letter synonym that maintains line length.
  // Not IF to avoid any confusion that IF is somehow special.
  AllowSimpleBracedStatements.IfMacros.push_back("FI");
  AllowSimpleBracedStatements.ColumnLimit = 40;
  AllowSimpleBracedStatements.AllowShortBlocksOnASingleLine =
      FormatStyle::SBS_Always;
  AllowSimpleBracedStatements.AllowShortLoopsOnASingleLine = true;
  AllowSimpleBracedStatements.BreakBeforeBraces = FormatStyle::BS_Custom;
  AllowSimpleBracedStatements.BraceWrapping.AfterFunction = true;
  AllowSimpleBracedStatements.BraceWrapping.SplitEmptyRecord = false;

  verifyFormat("if (true) {}", AllowSimpleBracedStatements);
  verifyFormat("if constexpr (true) {}", AllowSimpleBracedStatements);
  verifyFormat("if CONSTEXPR (true) {}", AllowSimpleBracedStatements);
  verifyFormat("if consteval {}", AllowSimpleBracedStatements);
  verifyFormat("if !consteval {}", AllowSimpleBracedStatements);
  verifyFormat("if CONSTEVAL {}", AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {}", AllowSimpleBracedStatements);
  verifyFormat("MYIF constexpr (true) {}", AllowSimpleBracedStatements);
  verifyFormat("MYIF CONSTEXPR (true) {}", AllowSimpleBracedStatements);
  verifyFormat("while (true) {}", AllowSimpleBracedStatements);
  verifyFormat("for (;;) {}", AllowSimpleBracedStatements);
  verifyFormat("if (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if constexpr (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if CONSTEXPR (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if consteval { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if CONSTEVAL { f(); }", AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("MYIF constexpr (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("MYIF CONSTEXPR (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("MYIF consteval { f(); }", AllowSimpleBracedStatements);
  verifyFormat("MYIF CONSTEVAL { f(); }", AllowSimpleBracedStatements);
  verifyFormat("while (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("for (;;) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if (true) { fffffffffffffffffffffff(); }",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  ffffffffffffffffffffffff();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  ffffffffffffffffffffffffffffffffffffffffffffffffffffff();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) { //\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  f();\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  f();\n"
               "} else {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("FI (true) { fffffffffffffffffffffff(); }",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {\n"
               "  ffffffffffffffffffffffff();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {\n"
               "  ffffffffffffffffffffffffffffffffffffffffffffffffffffff();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) { //\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {\n"
               "  f();\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {\n"
               "  f();\n"
               "} else {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);

  verifyFormat("struct A2 {\n"
               "  int X;\n"
               "};",
               AllowSimpleBracedStatements);
  verifyFormat("typedef struct A2 {\n"
               "  int X;\n"
               "} A2_t;",
               AllowSimpleBracedStatements);
  verifyFormat("template <int> struct A2 {\n"
               "  struct B {};\n"
               "};",
               AllowSimpleBracedStatements);

  AllowSimpleBracedStatements.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_Never;
  verifyFormat("if (true) {}", AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  f();\n"
               "} else {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {}", AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {\n"
               "  f();\n"
               "} else {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);

  AllowSimpleBracedStatements.AllowShortLoopsOnASingleLine = false;
  verifyFormat("while (true) {}", AllowSimpleBracedStatements);
  verifyFormat("while (true) {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("for (;;) {}", AllowSimpleBracedStatements);
  verifyFormat("for (;;) {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("BOOST_FOREACH (int v, vec) {}", AllowSimpleBracedStatements);
  verifyFormat("BOOST_FOREACH (int v, vec) {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);

  AllowSimpleBracedStatements.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_WithoutElse;
  AllowSimpleBracedStatements.AllowShortLoopsOnASingleLine = true;
  AllowSimpleBracedStatements.BraceWrapping.AfterControlStatement =
      FormatStyle::BWACS_Always;

  verifyFormat("if (true) {}", AllowSimpleBracedStatements);
  verifyFormat("if constexpr (true) {}", AllowSimpleBracedStatements);
  verifyFormat("if CONSTEXPR (true) {}", AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {}", AllowSimpleBracedStatements);
  verifyFormat("MYIF constexpr (true) {}", AllowSimpleBracedStatements);
  verifyFormat("MYIF CONSTEXPR (true) {}", AllowSimpleBracedStatements);
  verifyFormat("while (true) {}", AllowSimpleBracedStatements);
  verifyFormat("for (;;) {}", AllowSimpleBracedStatements);
  verifyFormat("if (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if constexpr (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if CONSTEXPR (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("MYIF constexpr (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("MYIF CONSTEXPR (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("while (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("for (;;) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if (true) { fffffffffffffffffffffff(); }",
               AllowSimpleBracedStatements);
  verifyFormat("if (true)\n"
               "{\n"
               "  ffffffffffffffffffffffff();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true)\n"
               "{\n"
               "  ffffffffffffffffffffffffffffffffffffffffffffffffffffff();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true)\n"
               "{ //\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true)\n"
               "{\n"
               "  f();\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true)\n"
               "{\n"
               "  f();\n"
               "} else\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("FI (true) { fffffffffffffffffffffff(); }",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true)\n"
               "{\n"
               "  ffffffffffffffffffffffff();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true)\n"
               "{\n"
               "  ffffffffffffffffffffffffffffffffffffffffffffffffffffff();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true)\n"
               "{ //\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true)\n"
               "{\n"
               "  f();\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true)\n"
               "{\n"
               "  f();\n"
               "} else\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);

  AllowSimpleBracedStatements.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_Never;
  verifyFormat("if (true) {}", AllowSimpleBracedStatements);
  verifyFormat("if (true)\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true)\n"
               "{\n"
               "  f();\n"
               "} else\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true) {}", AllowSimpleBracedStatements);
  verifyFormat("MYIF (true)\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("MYIF (true)\n"
               "{\n"
               "  f();\n"
               "} else\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);

  AllowSimpleBracedStatements.AllowShortLoopsOnASingleLine = false;
  verifyFormat("while (true) {}", AllowSimpleBracedStatements);
  verifyFormat("while (true)\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("for (;;) {}", AllowSimpleBracedStatements);
  verifyFormat("for (;;)\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("BOOST_FOREACH (int v, vec) {}", AllowSimpleBracedStatements);
  verifyFormat("BOOST_FOREACH (int v, vec)\n"
               "{\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);

  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;

  verifyFormat("while (i > 0)\n"
               "{\n"
               "  --i;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "{\n"
               "  ++b;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "{\n"
               "  b = 1;\n"
               "} else\n"
               "{\n"
               "  b = 0;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "{\n"
               "  b = 1;\n"
               "} else if (c)\n"
               "{\n"
               "  b = 2;\n"
               "} else\n"
               "{\n"
               "  b = 0;\n"
               "}",
               Style);

  Style.BraceWrapping.BeforeElse = true;

  verifyFormat("if (a)\n"
               "{\n"
               "  b = 1;\n"
               "}\n"
               "else\n"
               "{\n"
               "  b = 0;\n"
               "}",
               Style);

  verifyFormat("if (a)\n"
               "{\n"
               "  b = 1;\n"
               "}\n"
               "else if (c)\n"
               "{\n"
               "  b = 2;\n"
               "}\n"
               "else\n"
               "{\n"
               "  b = 0;\n"
               "}",
               Style);
}

TEST_F(FormatTest, UnderstandsMacros) {
  verifyFormat("#define A (parentheses)");
  verifyFormat("/* comment */ #define A (parentheses)");
  verifyFormat("/* comment */ /* another comment */ #define A (parentheses)");
  // Even the partial code should never be merged.
  verifyNoChange("/* comment */ #define A (parentheses)\n"
                 "#");
  verifyFormat("/* comment */ #define A (parentheses)\n"
               "#\n");
  verifyFormat("/* comment */ #define A (parentheses)\n"
               "#define B (parentheses)");
  verifyFormat("#define true ((int)1)");
  verifyFormat("#define and(x)");
  verifyFormat("#define if(x) x");
  verifyFormat("#define return(x) (x)");
  verifyFormat("#define while(x) for (; x;)");
  verifyFormat("#define xor(x) (^(x))");
  verifyFormat("#define __except(x)");
  verifyFormat("#define __try(x)");

  // https://llvm.org/PR54348.
  verifyFormat(
      "#define A"
      "                                                                      "
      "\\\n"
      "  class & {}");

  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterFunction = true;
  // Test that a macro definition never gets merged with the following
  // definition.
  // FIXME: The AAA macro definition probably should not be split into 3 lines.
  verifyFormat("#define AAA                                                    "
               "                \\\n"
               "  N                                                            "
               "                \\\n"
               "  {\n"
               "#define BBB }",
               Style);
  // verifyFormat("#define AAA N { //", Style);

  verifyFormat("MACRO(return)");
  verifyFormat("MACRO(co_await)");
  verifyFormat("MACRO(co_return)");
  verifyFormat("MACRO(co_yield)");
  verifyFormat("MACRO(return, something)");
  verifyFormat("MACRO(co_return, something)");
  verifyFormat("MACRO(something##something)");
  verifyFormat("MACRO(return##something)");
  verifyFormat("MACRO(co_return##something)");
}

TEST_F(FormatTest, ShortBlocksInMacrosDontMergeWithCodeAfterMacro) {
  FormatStyle Style = getLLVMStyleWithColumns(60);
  Style.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Always;
  Style.AllowShortIfStatementsOnASingleLine = FormatStyle::SIS_WithoutElse;
  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  verifyFormat("#define A                                                  \\\n"
               "  if (HANDLEwernufrnuLwrmviferuvnierv)                     \\\n"
               "  {                                                        \\\n"
               "    RET_ERR1_ANUIREUINERUIFNIOAerwfwrvnuier;               \\\n"
               "  }\n"
               "X;",
               "#define A \\\n"
               "   if (HANDLEwernufrnuLwrmviferuvnierv) { \\\n"
               "      RET_ERR1_ANUIREUINERUIFNIOAerwfwrvnuier; \\\n"
               "   }\n"
               "X;",
               Style);
}

TEST_F(FormatTest, ParseIfElse) {
  verifyFormat("if (true)\n"
               "  if (true)\n"
               "    if (true)\n"
               "      f();\n"
               "    else\n"
               "      g();\n"
               "  else\n"
               "    h();\n"
               "else\n"
               "  i();");
  verifyFormat("if (true)\n"
               "  if (true)\n"
               "    if (true) {\n"
               "      if (true)\n"
               "        f();\n"
               "    } else {\n"
               "      g();\n"
               "    }\n"
               "  else\n"
               "    h();\n"
               "else {\n"
               "  i();\n"
               "}");
  verifyFormat("if (true)\n"
               "  if constexpr (true)\n"
               "    if (true) {\n"
               "      if constexpr (true)\n"
               "        f();\n"
               "    } else {\n"
               "      g();\n"
               "    }\n"
               "  else\n"
               "    h();\n"
               "else {\n"
               "  i();\n"
               "}");
  verifyFormat("if (true)\n"
               "  if CONSTEXPR (true)\n"
               "    if (true) {\n"
               "      if CONSTEXPR (true)\n"
               "        f();\n"
               "    } else {\n"
               "      g();\n"
               "    }\n"
               "  else\n"
               "    h();\n"
               "else {\n"
               "  i();\n"
               "}");
  verifyFormat("void f() {\n"
               "  if (a) {\n"
               "  } else {\n"
               "  }\n"
               "}");
}

TEST_F(FormatTest, ElseIf) {
  verifyFormat("if (a) {\n} else if (b) {\n}");
  verifyFormat("if (a)\n"
               "  f();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  h();");
  verifyFormat("if (a)\n"
               "  f();\n"
               "else // comment\n"
               "  if (b) {\n"
               "    g();\n"
               "    h();\n"
               "  }");
  verifyFormat("if constexpr (a)\n"
               "  f();\n"
               "else if constexpr (b)\n"
               "  g();\n"
               "else\n"
               "  h();");
  verifyFormat("if CONSTEXPR (a)\n"
               "  f();\n"
               "else if CONSTEXPR (b)\n"
               "  g();\n"
               "else\n"
               "  h();");
  verifyFormat("if (a) {\n"
               "  f();\n"
               "}\n"
               "// or else ..\n"
               "else {\n"
               "  g()\n"
               "}");

  verifyFormat("if (a) {\n"
               "} else if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaa)) {\n"
               "}");
  verifyFormat("if (a) {\n"
               "} else if constexpr (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                         aaaaaaaaaaaaaaaaaaaaaaaaaaaa)) {\n"
               "}");
  verifyFormat("if (a) {\n"
               "} else if CONSTEXPR (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                         aaaaaaaaaaaaaaaaaaaaaaaaaaaa)) {\n"
               "}");
  verifyFormat("if (a) {\n"
               "} else if (\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
               "}",
               getLLVMStyleWithColumns(62));
  verifyFormat("if (a) {\n"
               "} else if constexpr (\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
               "}",
               getLLVMStyleWithColumns(62));
  verifyFormat("if (a) {\n"
               "} else if CONSTEXPR (\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
               "}",
               getLLVMStyleWithColumns(62));
}

TEST_F(FormatTest, SeparatePointerReferenceAlignment) {
  FormatStyle Style = getLLVMStyle();
  EXPECT_EQ(Style.PointerAlignment, FormatStyle::PAS_Right);
  EXPECT_EQ(Style.ReferenceAlignment, FormatStyle::RAS_Pointer);
  verifyFormat("int *f1(int *a, int &b, int &&c);", Style);
  verifyFormat("int &f2(int &&c, int *a, int &b);", Style);
  verifyFormat("int &&f3(int &b, int &&c, int *a);", Style);
  verifyFormat("int *f1(int &a) const &;", Style);
  verifyFormat("int *f1(int &a) const & = 0;", Style);
  verifyFormat("int *a = f1();", Style);
  verifyFormat("int &b = f2();", Style);
  verifyFormat("int &&c = f3();", Style);
  verifyFormat("int f3() { return sizeof(Foo &); }", Style);
  verifyFormat("int f4() { return sizeof(Foo &&); }", Style);
  verifyFormat("void f5() { int f6(Foo &, Bar &); }", Style);
  verifyFormat("void f5() { int f6(Foo &&, Bar &&); }", Style);
  verifyFormat("for (auto a = 0, b = 0; const auto &c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const int &c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const Foo &c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const Foo *c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const auto &c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const int &c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const Foo &c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const auto &c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const int &c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const Foo &c : {1, 2, 3})", Style);
  verifyFormat("for (auto x = 0; auto &c : {1, 2, 3})", Style);
  verifyFormat("for (auto x = 0; int &c : {1, 2, 3})", Style);
  verifyFormat("for (int x = 0; auto &c : {1, 2, 3})", Style);
  verifyFormat("for (int x = 0; int &c : {1, 2, 3})", Style);
  verifyFormat("for (f(); auto &c : {1, 2, 3})", Style);
  verifyFormat("for (f(); int &c : {1, 2, 3})", Style);
  verifyFormat(
      "function<int(int &)> res1 = [](int &a) { return 0000000000000; },\n"
      "                     res2 = [](int &a) { return 0000000000000; };",
      Style);

  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("Const unsigned int *c;\n"
               "const unsigned int *d;\n"
               "Const unsigned int &e;\n"
               "const unsigned int &f;\n"
               "int                *f1(int *a, int &b, int &&c);\n"
               "double             *(*f2)(int *a, double &&b);\n"
               "const unsigned    &&g;\n"
               "Const unsigned      h;",
               Style);

  Style.PointerAlignment = FormatStyle::PAS_Left;
  Style.ReferenceAlignment = FormatStyle::RAS_Pointer;
  verifyFormat("int* f1(int* a, int& b, int&& c);", Style);
  verifyFormat("int& f2(int&& c, int* a, int& b);", Style);
  verifyFormat("int&& f3(int& b, int&& c, int* a);", Style);
  verifyFormat("int* f1(int& a) const& = 0;", Style);
  verifyFormat("int* a = f1();", Style);
  verifyFormat("int& b = f2();", Style);
  verifyFormat("int&& c = f3();", Style);
  verifyFormat("int f3() { return sizeof(Foo&); }", Style);
  verifyFormat("int f4() { return sizeof(Foo&&); }", Style);
  verifyFormat("void f5() { int f6(Foo&, Bar&); }", Style);
  verifyFormat("void f5() { int f6(Foo&&, Bar&&); }", Style);
  verifyFormat("for (auto a = 0, b = 0; const auto& c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const int& c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const Foo& c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const Foo* c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const auto& c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const int& c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const Foo& c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const Foo* c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const auto& c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const int& c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const Foo& c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const Foo* c : {1, 2, 3})", Style);
  verifyFormat("for (auto x = 0; auto& c : {1, 2, 3})", Style);
  verifyFormat("for (auto x = 0; int& c : {1, 2, 3})", Style);
  verifyFormat("for (int x = 0; auto& c : {1, 2, 3})", Style);
  verifyFormat("for (int x = 0; int& c : {1, 2, 3})", Style);
  verifyFormat("for (f(); auto& c : {1, 2, 3})", Style);
  verifyFormat("for (f(); int& c : {1, 2, 3})", Style);
  verifyFormat(
      "function<int(int&)> res1 = [](int& a) { return 0000000000000; },\n"
      "                    res2 = [](int& a) { return 0000000000000; };",
      Style);

  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("Const unsigned int* c;\n"
               "const unsigned int* d;\n"
               "Const unsigned int& e;\n"
               "const unsigned int& f;\n"
               "int*                f1(int* a, int& b, int&& c);\n"
               "double*             (*f2)(int* a, double&& b);\n"
               "const unsigned&&    g;\n"
               "Const unsigned      h;",
               Style);

  Style.PointerAlignment = FormatStyle::PAS_Right;
  Style.ReferenceAlignment = FormatStyle::RAS_Left;
  verifyFormat("int *f1(int *a, int& b, int&& c);", Style);
  verifyFormat("int& f2(int&& c, int *a, int& b);", Style);
  verifyFormat("int&& f3(int& b, int&& c, int *a);", Style);
  verifyFormat("int *a = f1();", Style);
  verifyFormat("int& b = f2();", Style);
  verifyFormat("int&& c = f3();", Style);
  verifyFormat("int f3() { return sizeof(Foo&); }", Style);
  verifyFormat("int f4() { return sizeof(Foo&&); }", Style);
  verifyFormat("void f5() { int f6(Foo&, Bar&); }", Style);
  verifyFormat("void f5() { int f6(Foo&&, Bar&&); }", Style);
  verifyFormat("for (auto a = 0, b = 0; const Foo *c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const Foo *c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const Foo *c : {1, 2, 3})", Style);

  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("Const unsigned int *c;\n"
               "const unsigned int *d;\n"
               "Const unsigned int& e;\n"
               "const unsigned int& f;\n"
               "int                *f1(int *a, int& b, int&& c);\n"
               "double             *(*f2)(int *a, double&& b);\n"
               "const unsigned      g;\n"
               "Const unsigned      h;",
               Style);

  Style.PointerAlignment = FormatStyle::PAS_Left;
  Style.ReferenceAlignment = FormatStyle::RAS_Middle;
  verifyFormat("int* f1(int* a, int & b, int && c);", Style);
  verifyFormat("int & f2(int && c, int* a, int & b);", Style);
  verifyFormat("int && f3(int & b, int && c, int* a);", Style);
  verifyFormat("int* a = f1();", Style);
  verifyFormat("int & b = f2();", Style);
  verifyFormat("int && c = f3();", Style);
  verifyFormat("int f3() { return sizeof(Foo &); }", Style);
  verifyFormat("int f4() { return sizeof(Foo &&); }", Style);
  verifyFormat("void f5() { int f6(Foo &, Bar &); }", Style);
  verifyFormat("void f5() { int f6(Foo &&, Bar &&); }", Style);
  verifyFormat("for (auto a = 0, b = 0; const auto & c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const int & c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const Foo & c : {1, 2, 3})", Style);
  verifyFormat("for (auto a = 0, b = 0; const Foo* c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const auto & c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const int & c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const Foo & c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const Foo* c : {1, 2, 3})", Style);
  verifyFormat("for (auto x = 0; auto & c : {1, 2, 3})", Style);
  verifyFormat("for (auto x = 0; int & c : {1, 2, 3})", Style);
  verifyFormat("for (int x = 0; auto & c : {1, 2, 3})", Style);
  verifyFormat("for (int x = 0; int & c : {1, 2, 3})", Style);
  verifyFormat("for (f(); auto & c : {1, 2, 3})", Style);
  verifyFormat("for (f(); int & c : {1, 2, 3})", Style);
  verifyFormat(
      "function<int(int &)> res1 = [](int & a) { return 0000000000000; },\n"
      "                     res2 = [](int & a) { return 0000000000000; };",
      Style);

  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("Const unsigned int*  c;\n"
               "const unsigned int*  d;\n"
               "Const unsigned int & e;\n"
               "const unsigned int & f;\n"
               "int*                 f1(int* a, int & b, int && c);\n"
               "double*              (*f2)(int* a, double && b);\n"
               "const unsigned &&    g;\n"
               "Const unsigned       h;",
               Style);

  Style.PointerAlignment = FormatStyle::PAS_Middle;
  Style.ReferenceAlignment = FormatStyle::RAS_Right;
  verifyFormat("int * f1(int * a, int &b, int &&c);", Style);
  verifyFormat("int &f2(int &&c, int * a, int &b);", Style);
  verifyFormat("int &&f3(int &b, int &&c, int * a);", Style);
  verifyFormat("int * a = f1();", Style);
  verifyFormat("int &b = f2();", Style);
  verifyFormat("int &&c = f3();", Style);
  verifyFormat("int f3() { return sizeof(Foo &); }", Style);
  verifyFormat("int f4() { return sizeof(Foo &&); }", Style);
  verifyFormat("void f5() { int f6(Foo &, Bar &); }", Style);
  verifyFormat("void f5() { int f6(Foo &&, Bar &&); }", Style);
  verifyFormat("for (auto a = 0, b = 0; const Foo * c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b = 0; const Foo * c : {1, 2, 3})", Style);
  verifyFormat("for (int a = 0, b++; const Foo * c : {1, 2, 3})", Style);

  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("Const unsigned int * c;\n"
               "const unsigned int * d;\n"
               "Const unsigned int  &e;\n"
               "const unsigned int  &f;\n"
               "int *                f1(int * a, int &b, int &&c);\n"
               "double *             (*f2)(int * a, double &&b);\n"
               "const unsigned     &&g;\n"
               "Const unsigned       h;",
               Style);

  // FIXME: we don't handle this yet, so output may be arbitrary until it's
  // specifically handled
  // verifyFormat("int Add2(BTree * &Root, char * szToAdd)", Style);
}

TEST_F(FormatTest, FormatsForLoop) {
  verifyFormat(
      "for (int VeryVeryLongLoopVariable = 0; VeryVeryLongLoopVariable < 10;\n"
      "     ++VeryVeryLongLoopVariable)\n"
      "  ;");
  verifyFormat("for (;;)\n"
               "  f();");
  verifyFormat("for (;;) {\n}");
  verifyFormat("for (;;) {\n"
               "  f();\n"
               "}");
  verifyFormat("for (int i = 0; (i < 10); ++i) {\n}");

  verifyFormat(
      "for (std::vector<UnwrappedLine>::iterator I = UnwrappedLines.begin(),\n"
      "                                          E = UnwrappedLines.end();\n"
      "     I != E; ++I) {\n}");

  verifyFormat(
      "for (MachineFun::iterator IIII = PrevIt, EEEE = F.end(); IIII != EEEE;\n"
      "     ++IIIII) {\n}");
  verifyFormat("for (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaa =\n"
               "         aaaaaaaaaaaaaaaa.aaaaaaaaaaaaaaa;\n"
               "     aaaaaaaaaaa != aaaaaaaaaaaaaaaaaaa; ++aaaaaaaaaaa) {\n}");
  verifyFormat("for (llvm::ArrayRef<NamedDecl *>::iterator\n"
               "         I = FD->getDeclsInPrototypeScope().begin(),\n"
               "         E = FD->getDeclsInPrototypeScope().end();\n"
               "     I != E; ++I) {\n}");
  verifyFormat("for (SmallVectorImpl<TemplateIdAnnotationn *>::iterator\n"
               "         I = Container.begin(),\n"
               "         E = Container.end();\n"
               "     I != E; ++I) {\n}",
               getLLVMStyleWithColumns(76));

  verifyFormat(
      "for (aaaaaaaaaaaaaaaaa aaaaaaaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;\n"
      "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa !=\n"
      "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
      "     ++aaaaaaaaaaa) {\n}");
  verifyFormat("for (int i = 0; i < aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "                bbbbbbbbbbbbbbbbbbbb < ccccccccccccccc;\n"
               "     ++i) {\n}");
  verifyFormat("for (int aaaaaaaaaaa = 1; aaaaaaaaaaa <= bbbbbbbbbbbbbbb;\n"
               "     aaaaaaaaaaa++, bbbbbbbbbbbbbbbbb++) {\n"
               "}");
  verifyFormat("for (some_namespace::SomeIterator iter( // force break\n"
               "         aaaaaaaaaa);\n"
               "     iter; ++iter) {\n"
               "}");
  verifyFormat("for (auto aaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaa != bbbbbbbbbbbbbbbbbbbbbbb;\n"
               "     ++aaaaaaaaaaaaaaaaaaaaaaaaaaa) {");

  // These should not be formatted as Objective-C for-in loops.
  verifyFormat("for (Foo *x = 0; x != in; x++) {\n}");
  verifyFormat("Foo *x;\nfor (x = 0; x != in; x++) {\n}");
  verifyFormat("Foo *x;\nfor (x in y) {\n}");
  verifyFormat(
      "for (const Foo<Bar> &baz = in.value(); !baz.at_end(); ++baz) {\n}");

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = false;
  verifyFormat("for (int aaaaaaaaaaa = 1;\n"
               "     aaaaaaaaaaa <= aaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaa,\n"
               "                                           aaaaaaaaaaaaaaaa,\n"
               "                                           aaaaaaaaaaaaaaaa,\n"
               "                                           aaaaaaaaaaaaaaaa);\n"
               "     aaaaaaaaaaa++, bbbbbbbbbbbbbbbbb++) {\n"
               "}",
               NoBinPacking);
  verifyFormat(
      "for (std::vector<UnwrappedLine>::iterator I = UnwrappedLines.begin(),\n"
      "                                          E = UnwrappedLines.end();\n"
      "     I != E;\n"
      "     ++I) {\n}",
      NoBinPacking);

  FormatStyle AlignLeft = getLLVMStyle();
  AlignLeft.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("for (A* a = start; a < end; ++a, ++value) {\n}", AlignLeft);
}

TEST_F(FormatTest, RangeBasedForLoops) {
  verifyFormat("for (auto aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat("for (auto aaaaaaaaaaaaaaaaaaaaa :\n"
               "     aaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaa, aaaaaaaaaaaaa)) {\n}");
  verifyFormat("for (const aaaaaaaaaaaaaaaaaaaaa &aaaaaaaaa :\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat("for (aaaaaaaaa aaaaaaaaaaaaaaaaaaaaa :\n"
               "     aaaaaaaaaaaa.aaaaaaaaaaaa().aaaaaaaaa().a()) {\n}");
}

TEST_F(FormatTest, ForEachLoops) {
  FormatStyle Style = getLLVMStyle();
  EXPECT_EQ(Style.AllowShortBlocksOnASingleLine, FormatStyle::SBS_Never);
  EXPECT_EQ(Style.AllowShortLoopsOnASingleLine, false);
  verifyFormat("void f() {\n"
               "  for (;;) {\n"
               "  }\n"
               "  foreach (Item *item, itemlist) {\n"
               "  }\n"
               "  Q_FOREACH (Item *item, itemlist) {\n"
               "  }\n"
               "  BOOST_FOREACH (Item *item, itemlist) {\n"
               "  }\n"
               "  UNKNOWN_FOREACH(Item * item, itemlist) {}\n"
               "}",
               Style);
  verifyFormat("void f() {\n"
               "  for (;;)\n"
               "    int j = 1;\n"
               "  Q_FOREACH (int v, vec)\n"
               "    v *= 2;\n"
               "  for (;;) {\n"
               "    int j = 1;\n"
               "  }\n"
               "  Q_FOREACH (int v, vec) {\n"
               "    v *= 2;\n"
               "  }\n"
               "}",
               Style);

  FormatStyle ShortBlocks = getLLVMStyle();
  ShortBlocks.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Always;
  EXPECT_EQ(ShortBlocks.AllowShortLoopsOnASingleLine, false);
  verifyFormat("void f() {\n"
               "  for (;;)\n"
               "    int j = 1;\n"
               "  Q_FOREACH (int &v, vec)\n"
               "    v *= 2;\n"
               "  for (;;) {\n"
               "    int j = 1;\n"
               "  }\n"
               "  Q_FOREACH (int &v, vec) {\n"
               "    int j = 1;\n"
               "  }\n"
               "}",
               ShortBlocks);

  FormatStyle ShortLoops = getLLVMStyle();
  ShortLoops.AllowShortLoopsOnASingleLine = true;
  EXPECT_EQ(ShortLoops.AllowShortBlocksOnASingleLine, FormatStyle::SBS_Never);
  verifyFormat("void f() {\n"
               "  for (;;) int j = 1;\n"
               "  Q_FOREACH (int &v, vec) int j = 1;\n"
               "  for (;;) {\n"
               "    int j = 1;\n"
               "  }\n"
               "  Q_FOREACH (int &v, vec) {\n"
               "    int j = 1;\n"
               "  }\n"
               "}",
               ShortLoops);

  FormatStyle ShortBlocksAndLoops = getLLVMStyle();
  ShortBlocksAndLoops.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Always;
  ShortBlocksAndLoops.AllowShortLoopsOnASingleLine = true;
  verifyFormat("void f() {\n"
               "  for (;;) int j = 1;\n"
               "  Q_FOREACH (int &v, vec) int j = 1;\n"
               "  for (;;) { int j = 1; }\n"
               "  Q_FOREACH (int &v, vec) { int j = 1; }\n"
               "}",
               ShortBlocksAndLoops);

  Style.SpaceBeforeParens =
      FormatStyle::SBPO_ControlStatementsExceptControlMacros;
  verifyFormat("void f() {\n"
               "  for (;;) {\n"
               "  }\n"
               "  foreach(Item *item, itemlist) {\n"
               "  }\n"
               "  Q_FOREACH(Item *item, itemlist) {\n"
               "  }\n"
               "  BOOST_FOREACH(Item *item, itemlist) {\n"
               "  }\n"
               "  UNKNOWN_FOREACH(Item * item, itemlist) {}\n"
               "}",
               Style);

  // As function-like macros.
  verifyFormat("#define foreach(x, y)\n"
               "#define Q_FOREACH(x, y)\n"
               "#define BOOST_FOREACH(x, y)\n"
               "#define UNKNOWN_FOREACH(x, y)");

  // Not as function-like macros.
  verifyFormat("#define foreach (x, y)\n"
               "#define Q_FOREACH (x, y)\n"
               "#define BOOST_FOREACH (x, y)\n"
               "#define UNKNOWN_FOREACH (x, y)");

  // handle microsoft non standard extension
  verifyFormat("for each (char c in x->MyStringProperty)");
}

TEST_F(FormatTest, FormatsWhileLoop) {
  verifyFormat("while (true) {\n}");
  verifyFormat("while (true)\n"
               "  f();");
  verifyFormat("while () {\n}");
  verifyFormat("while () {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, FormatsDoWhile) {
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while (something());");
  verifyFormat("do\n"
               "  do_something();\n"
               "while (something());");
}

TEST_F(FormatTest, FormatsSwitchStatement) {
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "  f();\n"
               "  break;\n"
               "case kFoo:\n"
               "case ns::kBar:\n"
               "case kBaz:\n"
               "  break;\n"
               "default:\n"
               "  g();\n"
               "  break;\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  break;\n"
               "}\n"
               "case 2: {\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  {\n"
               "    g();\n"
               "    h();\n"
               "  }\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  if (foo) {\n"
               "    g();\n"
               "    h();\n"
               "  }\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  g();\n"
               "} break;\n"
               "}");
  verifyFormat("switch (test)\n"
               "  ;");
  verifyFormat("switch (x) {\n"
               "default: {\n"
               "  // Do nothing.\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "// comment\n"
               "// if 1, do f()\n"
               "case 1:\n"
               "  f();\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "  // Do amazing stuff\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "  break;\n"
               "}");
  verifyFormat("#define A          \\\n"
               "  switch (x) {     \\\n"
               "  case a:          \\\n"
               "    foo = b;       \\\n"
               "  }",
               getLLVMStyleWithColumns(20));
  verifyFormat("#define OPERATION_CASE(name)           \\\n"
               "  case OP_name:                        \\\n"
               "    return operations::Operation##name",
               getLLVMStyleWithColumns(40));
  verifyFormat("switch (x) {\n"
               "case 1:;\n"
               "default:;\n"
               "  int i;\n"
               "}");

  verifyGoogleFormat("switch (x) {\n"
                     "  case 1:\n"
                     "    f();\n"
                     "    break;\n"
                     "  case kFoo:\n"
                     "  case ns::kBar:\n"
                     "  case kBaz:\n"
                     "    break;\n"
                     "  default:\n"
                     "    g();\n"
                     "    break;\n"
                     "}");
  verifyGoogleFormat("switch (x) {\n"
                     "  case 1: {\n"
                     "    f();\n"
                     "    break;\n"
                     "  }\n"
                     "}");
  verifyGoogleFormat("switch (test)\n"
                     "  ;");

  verifyGoogleFormat("#define OPERATION_CASE(name) \\\n"
                     "  case OP_name:              \\\n"
                     "    return operations::Operation##name");
  verifyGoogleFormat("Operation codeToOperation(OperationCode OpCode) {\n"
                     "  // Get the correction operation class.\n"
                     "  switch (OpCode) {\n"
                     "    CASE(Add);\n"
                     "    CASE(Subtract);\n"
                     "    default:\n"
                     "      return operations::Unknown;\n"
                     "  }\n"
                     "#undef OPERATION_CASE\n"
                     "}");
  verifyFormat("DEBUG({\n"
               "  switch (x) {\n"
               "  case A:\n"
               "    f();\n"
               "    break;\n"
               "    // fallthrough\n"
               "  case B:\n"
               "    g();\n"
               "    break;\n"
               "  }\n"
               "});");
  verifyNoChange("DEBUG({\n"
                 "  switch (x) {\n"
                 "  case A:\n"
                 "    f();\n"
                 "    break;\n"
                 "  // On B:\n"
                 "  case B:\n"
                 "    g();\n"
                 "    break;\n"
                 "  }\n"
                 "});");
  verifyFormat("switch (n) {\n"
               "case 0: {\n"
               "  return false;\n"
               "}\n"
               "default: {\n"
               "  return true;\n"
               "}\n"
               "}",
               "switch (n)\n"
               "{\n"
               "case 0: {\n"
               "  return false;\n"
               "}\n"
               "default: {\n"
               "  return true;\n"
               "}\n"
               "}");
  verifyFormat("switch (a) {\n"
               "case (b):\n"
               "  return;\n"
               "}");

  verifyFormat("switch (a) {\n"
               "case some_namespace::\n"
               "    some_constant:\n"
               "  return;\n"
               "}",
               getLLVMStyleWithColumns(34));

  verifyFormat("switch (a) {\n"
               "[[likely]] case 1:\n"
               "  return;\n"
               "}");
  verifyFormat("switch (a) {\n"
               "[[likely]] [[other::likely]] case 1:\n"
               "  return;\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "  return;\n"
               "[[likely]] case 2:\n"
               "  return;\n"
               "}");
  verifyFormat("switch (a) {\n"
               "case 1:\n"
               "[[likely]] case 2:\n"
               "  return;\n"
               "}");
  FormatStyle Attributes = getLLVMStyle();
  Attributes.AttributeMacros.push_back("LIKELY");
  Attributes.AttributeMacros.push_back("OTHER_LIKELY");
  verifyFormat("switch (a) {\n"
               "LIKELY case b:\n"
               "  return;\n"
               "}",
               Attributes);
  verifyFormat("switch (a) {\n"
               "LIKELY OTHER_LIKELY() case b:\n"
               "  return;\n"
               "}",
               Attributes);
  verifyFormat("switch (a) {\n"
               "case 1:\n"
               "  return;\n"
               "LIKELY case 2:\n"
               "  return;\n"
               "}",
               Attributes);
  verifyFormat("switch (a) {\n"
               "case 1:\n"
               "LIKELY case 2:\n"
               "  return;\n"
               "}",
               Attributes);

  FormatStyle Style = getLLVMStyle();
  Style.IndentCaseLabels = true;
  Style.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Never;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterCaseLabel = true;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  verifyFormat("switch (n)\n"
               "{\n"
               "  case 0:\n"
               "  {\n"
               "    return false;\n"
               "  }\n"
               "  default:\n"
               "  {\n"
               "    return true;\n"
               "  }\n"
               "}",
               "switch (n) {\n"
               "  case 0: {\n"
               "    return false;\n"
               "  }\n"
               "  default: {\n"
               "    return true;\n"
               "  }\n"
               "}",
               Style);
  Style.BraceWrapping.AfterCaseLabel = false;
  verifyFormat("switch (n)\n"
               "{\n"
               "  case 0: {\n"
               "    return false;\n"
               "  }\n"
               "  default: {\n"
               "    return true;\n"
               "  }\n"
               "}",
               "switch (n) {\n"
               "  case 0:\n"
               "  {\n"
               "    return false;\n"
               "  }\n"
               "  default:\n"
               "  {\n"
               "    return true;\n"
               "  }\n"
               "}",
               Style);
  Style.IndentCaseLabels = false;
  Style.IndentCaseBlocks = true;
  verifyFormat("switch (n)\n"
               "{\n"
               "case 0:\n"
               "  {\n"
               "    return false;\n"
               "  }\n"
               "case 1:\n"
               "  break;\n"
               "default:\n"
               "  {\n"
               "    return true;\n"
               "  }\n"
               "}",
               "switch (n) {\n"
               "case 0: {\n"
               "  return false;\n"
               "}\n"
               "case 1:\n"
               "  break;\n"
               "default: {\n"
               "  return true;\n"
               "}\n"
               "}",
               Style);
  Style.IndentCaseLabels = true;
  Style.IndentCaseBlocks = true;
  verifyFormat("switch (n)\n"
               "{\n"
               "  case 0:\n"
               "    {\n"
               "      return false;\n"
               "    }\n"
               "  case 1:\n"
               "    break;\n"
               "  default:\n"
               "    {\n"
               "      return true;\n"
               "    }\n"
               "}",
               "switch (n) {\n"
               "case 0: {\n"
               "  return false;\n"
               "}\n"
               "case 1:\n"
               "  break;\n"
               "default: {\n"
               "  return true;\n"
               "}\n"
               "}",
               Style);
}

TEST_F(FormatTest, CaseRanges) {
  verifyFormat("switch (x) {\n"
               "case 'A' ... 'Z':\n"
               "case 1 ... 5:\n"
               "case a ... b:\n"
               "  break;\n"
               "}");
}

TEST_F(FormatTest, ShortEnums) {
  FormatStyle Style = getLLVMStyle();
  EXPECT_TRUE(Style.AllowShortEnumsOnASingleLine);
  EXPECT_FALSE(Style.BraceWrapping.AfterEnum);
  verifyFormat("enum { A, B, C } ShortEnum1, ShortEnum2;", Style);
  verifyFormat("typedef enum { A, B, C } ShortEnum1, ShortEnum2;", Style);
  Style.AllowShortEnumsOnASingleLine = false;
  verifyFormat("enum {\n"
               "  A,\n"
               "  B,\n"
               "  C\n"
               "} ShortEnum1, ShortEnum2;",
               Style);
  verifyFormat("typedef enum {\n"
               "  A,\n"
               "  B,\n"
               "  C\n"
               "} ShortEnum1, ShortEnum2;",
               Style);
  verifyFormat("enum {\n"
               "  A,\n"
               "} ShortEnum1, ShortEnum2;",
               Style);
  verifyFormat("typedef enum {\n"
               "  A,\n"
               "} ShortEnum1, ShortEnum2;",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterEnum = true;
  verifyFormat("enum\n"
               "{\n"
               "  A,\n"
               "  B,\n"
               "  C\n"
               "} ShortEnum1, ShortEnum2;",
               Style);
  verifyFormat("typedef enum\n"
               "{\n"
               "  A,\n"
               "  B,\n"
               "  C\n"
               "} ShortEnum1, ShortEnum2;",
               Style);
}

TEST_F(FormatTest, ShortCaseLabels) {
  FormatStyle Style = getLLVMStyle();
  Style.AllowShortCaseLabelsOnASingleLine = true;
  verifyFormat("switch (a) {\n"
               "case 1: x = 1; break;\n"
               "case 2: return;\n"
               "case 3:\n"
               "case 4:\n"
               "case 5: return;\n"
               "case 6: // comment\n"
               "  return;\n"
               "case 7:\n"
               "  // comment\n"
               "  return;\n"
               "case 8:\n"
               "  x = 8; // comment\n"
               "  break;\n"
               "default: y = 1; break;\n"
               "}",
               Style);
  verifyFormat("switch (a) {\n"
               "case 0: return; // comment\n"
               "case 1: break;  // comment\n"
               "case 2: return;\n"
               "// comment\n"
               "case 3: return;\n"
               "// comment 1\n"
               "// comment 2\n"
               "// comment 3\n"
               "case 4: break; /* comment */\n"
               "case 5:\n"
               "  // comment\n"
               "  break;\n"
               "case 6: /* comment */ x = 1; break;\n"
               "case 7: x = /* comment */ 1; break;\n"
               "case 8:\n"
               "  x = 1; /* comment */\n"
               "  break;\n"
               "case 9:\n"
               "  break; // comment line 1\n"
               "         // comment line 2\n"
               "}",
               Style);
  verifyFormat("switch (a) {\n"
               "case 1:\n"
               "  x = 8;\n"
               "  // fall through\n"
               "case 2: x = 8;\n"
               "// comment\n"
               "case 3:\n"
               "  return; /* comment line 1\n"
               "           * comment line 2 */\n"
               "case 4: i = 8;\n"
               "// something else\n"
               "#if FOO\n"
               "case 5: break;\n"
               "#endif\n"
               "}",
               "switch (a) {\n"
               "case 1: x = 8;\n"
               "  // fall through\n"
               "case 2:\n"
               "  x = 8;\n"
               "// comment\n"
               "case 3:\n"
               "  return; /* comment line 1\n"
               "           * comment line 2 */\n"
               "case 4:\n"
               "  i = 8;\n"
               "// something else\n"
               "#if FOO\n"
               "case 5: break;\n"
               "#endif\n"
               "}",
               Style);
  verifyFormat("switch (a) {\n"
               "case 0:\n"
               "  return; // long long long long long long long long long long "
               "long long comment\n"
               "          // line\n"
               "}",
               "switch (a) {\n"
               "case 0: return; // long long long long long long long long "
               "long long long long comment line\n"
               "}",
               Style);
  verifyFormat("switch (a) {\n"
               "case 0:\n"
               "  return; /* long long long long long long long long long long "
               "long long comment\n"
               "             line */\n"
               "}",
               "switch (a) {\n"
               "case 0: return; /* long long long long long long long long "
               "long long long long comment line */\n"
               "}",
               Style);
  verifyFormat("switch (a) {\n"
               "#if FOO\n"
               "case 0: return 0;\n"
               "#endif\n"
               "}",
               Style);
  verifyFormat("switch (a) {\n"
               "case 1: {\n"
               "}\n"
               "case 2: {\n"
               "  return;\n"
               "}\n"
               "case 3: {\n"
               "  x = 1;\n"
               "  return;\n"
               "}\n"
               "case 4:\n"
               "  if (x)\n"
               "    return;\n"
               "}",
               Style);
  Style.ColumnLimit = 21;
  verifyFormat("#define X           \\\n"
               "  case 0: break;\n"
               "#include \"f\"",
               Style);
  verifyFormat("switch (a) {\n"
               "case 1: x = 1; break;\n"
               "case 2: return;\n"
               "case 3:\n"
               "case 4:\n"
               "case 5: return;\n"
               "default:\n"
               "  y = 1;\n"
               "  break;\n"
               "}",
               Style);
  Style.ColumnLimit = 80;
  Style.AllowShortCaseLabelsOnASingleLine = false;
  Style.IndentCaseLabels = true;
  verifyFormat("switch (n) {\n"
               "  default /*comments*/:\n"
               "    return true;\n"
               "  case 0:\n"
               "    return false;\n"
               "}",
               "switch (n) {\n"
               "default/*comments*/:\n"
               "  return true;\n"
               "case 0:\n"
               "  return false;\n"
               "}",
               Style);
  Style.AllowShortCaseLabelsOnASingleLine = true;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterCaseLabel = true;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  verifyFormat("switch (n)\n"
               "{\n"
               "  case 0:\n"
               "  {\n"
               "    return false;\n"
               "  }\n"
               "  default:\n"
               "  {\n"
               "    return true;\n"
               "  }\n"
               "}",
               "switch (n) {\n"
               "  case 0: {\n"
               "    return false;\n"
               "  }\n"
               "  default:\n"
               "  {\n"
               "    return true;\n"
               "  }\n"
               "}",
               Style);
}

TEST_F(FormatTest, FormatsLabels) {
  verifyFormat("void f() {\n"
               "  some_code();\n"
               "test_label:\n"
               "  some_other_code();\n"
               "  {\n"
               "    some_more_code();\n"
               "  another_label:\n"
               "    some_more_code();\n"
               "  }\n"
               "}");
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label:\n"
               "  some_other_code();\n"
               "}");
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label:;\n"
               "  int i = 0;\n"
               "}");
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label: { some_other_code(); }\n"
               "}");
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label: {\n"
               "  some_other_code();\n"
               "  some_other_code();\n"
               "}\n"
               "}");
  verifyFormat("{\n"
               "L0:\n"
               "[[foo]] L1:\n"
               "[[bar]] [[baz]] L2:\n"
               "  g();\n"
               "}");
  verifyFormat("{\n"
               "[[foo]] L1: {\n"
               "[[bar]] [[baz]] L2:\n"
               "  g();\n"
               "}\n"
               "}");
  verifyFormat("{\n"
               "[[foo]] L1:\n"
               "  f();\n"
               "  {\n"
               "  [[bar]] [[baz]] L2:\n"
               "    g();\n"
               "  }\n"
               "}");
  FormatStyle Style = getLLVMStyle();
  Style.IndentGotoLabels = false;
  verifyFormat("void f() {\n"
               "  some_code();\n"
               "test_label:\n"
               "  some_other_code();\n"
               "  {\n"
               "    some_more_code();\n"
               "another_label:\n"
               "    some_more_code();\n"
               "  }\n"
               "}",
               Style);
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label:\n"
               "  some_other_code();\n"
               "}",
               Style);
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label:;\n"
               "  int i = 0;\n"
               "}",
               Style);
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label: { some_other_code(); }\n"
               "}",
               Style);
  verifyFormat("{\n"
               "[[foo]] L1:\n"
               "  f();\n"
               "  {\n"
               "[[bar]] [[baz]] L2:\n"
               "    g();\n"
               "  }\n"
               "}",
               Style);
  // The opening brace may either be on the same unwrapped line as the colon or
  // on a separate one. The formatter should recognize both.
  Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BraceBreakingStyle::BS_Allman;
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label:\n"
               "{\n"
               "  some_other_code();\n"
               "}\n"
               "}",
               Style);
  verifyFormat("{\n"
               "[[foo]] L1:\n"
               "{\n"
               "[[bar]] [[baz]] L2:\n"
               "  g();\n"
               "}\n"
               "}",
               Style);
}

TEST_F(FormatTest, MultiLineControlStatements) {
  FormatStyle Style = getLLVMStyleWithColumns(20);
  Style.BreakBeforeBraces = FormatStyle::BraceBreakingStyle::BS_Custom;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_MultiLine;
  // Short lines should keep opening brace on same line.
  verifyFormat("if (foo) {\n"
               "  bar();\n"
               "}",
               "if(foo){bar();}", Style);
  verifyFormat("if (foo) {\n"
               "  bar();\n"
               "} else {\n"
               "  baz();\n"
               "}",
               "if(foo){bar();}else{baz();}", Style);
  verifyFormat("if (foo && bar) {\n"
               "  baz();\n"
               "}",
               "if(foo&&bar){baz();}", Style);
  verifyFormat("if (foo) {\n"
               "  bar();\n"
               "} else if (baz) {\n"
               "  quux();\n"
               "}",
               "if(foo){bar();}else if(baz){quux();}", Style);
  verifyFormat("if (foo) {\n"
               "  bar();\n"
               "} else if (baz) {\n"
               "  quux();\n"
               "} else {\n"
               "  foobar();\n"
               "}",
               "if(foo){bar();}else if(baz){quux();}else{foobar();}", Style);
  verifyFormat("for (;;) {\n"
               "  foo();\n"
               "}",
               "for(;;){foo();}");
  verifyFormat("while (1) {\n"
               "  foo();\n"
               "}",
               "while(1){foo();}", Style);
  verifyFormat("switch (foo) {\n"
               "case bar:\n"
               "  return;\n"
               "}",
               "switch(foo){case bar:return;}", Style);
  verifyFormat("try {\n"
               "  foo();\n"
               "} catch (...) {\n"
               "  bar();\n"
               "}",
               "try{foo();}catch(...){bar();}", Style);
  verifyFormat("do {\n"
               "  foo();\n"
               "} while (bar &&\n"
               "         baz);",
               "do{foo();}while(bar&&baz);", Style);
  // Long lines should put opening brace on new line.
  verifyFormat("void f() {\n"
               "  if (a1 && a2 &&\n"
               "      a3)\n"
               "  {\n"
               "    quux();\n"
               "  }\n"
               "}",
               "void f(){if(a1&&a2&&a3){quux();}}", Style);
  verifyFormat("if (foo && bar &&\n"
               "    baz)\n"
               "{\n"
               "  quux();\n"
               "}",
               "if(foo&&bar&&baz){quux();}", Style);
  verifyFormat("if (foo && bar &&\n"
               "    baz)\n"
               "{\n"
               "  quux();\n"
               "}",
               "if (foo && bar &&\n"
               "    baz) {\n"
               "  quux();\n"
               "}",
               Style);
  verifyFormat("if (foo) {\n"
               "  bar();\n"
               "} else if (baz ||\n"
               "           quux)\n"
               "{\n"
               "  foobar();\n"
               "}",
               "if(foo){bar();}else if(baz||quux){foobar();}", Style);
  verifyFormat("if (foo) {\n"
               "  bar();\n"
               "} else if (baz ||\n"
               "           quux)\n"
               "{\n"
               "  foobar();\n"
               "} else {\n"
               "  barbaz();\n"
               "}",
               "if(foo){bar();}else if(baz||quux){foobar();}else{barbaz();}",
               Style);
  verifyFormat("for (int i = 0;\n"
               "     i < 10; ++i)\n"
               "{\n"
               "  foo();\n"
               "}",
               "for(int i=0;i<10;++i){foo();}", Style);
  verifyFormat("foreach (int i,\n"
               "         list)\n"
               "{\n"
               "  foo();\n"
               "}",
               "foreach(int i, list){foo();}", Style);
  Style.ColumnLimit =
      40; // to concentrate at brace wrapping, not line wrap due to column limit
  verifyFormat("foreach (int i, list) {\n"
               "  foo();\n"
               "}",
               "foreach(int i, list){foo();}", Style);
  Style.ColumnLimit =
      20; // to concentrate at brace wrapping, not line wrap due to column limit
  verifyFormat("while (foo || bar ||\n"
               "       baz)\n"
               "{\n"
               "  quux();\n"
               "}",
               "while(foo||bar||baz){quux();}", Style);
  verifyFormat("switch (\n"
               "    foo = barbaz)\n"
               "{\n"
               "case quux:\n"
               "  return;\n"
               "}",
               "switch(foo=barbaz){case quux:return;}", Style);
  verifyFormat("try {\n"
               "  foo();\n"
               "} catch (\n"
               "    Exception &bar)\n"
               "{\n"
               "  baz();\n"
               "}",
               "try{foo();}catch(Exception&bar){baz();}", Style);
  Style.ColumnLimit =
      40; // to concentrate at brace wrapping, not line wrap due to column limit
  verifyFormat("try {\n"
               "  foo();\n"
               "} catch (Exception &bar) {\n"
               "  baz();\n"
               "}",
               "try{foo();}catch(Exception&bar){baz();}", Style);
  Style.ColumnLimit =
      20; // to concentrate at brace wrapping, not line wrap due to column limit

  Style.BraceWrapping.BeforeElse = true;
  verifyFormat("if (foo) {\n"
               "  bar();\n"
               "}\n"
               "else if (baz ||\n"
               "         quux)\n"
               "{\n"
               "  foobar();\n"
               "}\n"
               "else {\n"
               "  barbaz();\n"
               "}",
               "if(foo){bar();}else if(baz||quux){foobar();}else{barbaz();}",
               Style);

  Style.BraceWrapping.BeforeCatch = true;
  verifyFormat("try {\n"
               "  foo();\n"
               "}\n"
               "catch (...) {\n"
               "  baz();\n"
               "}",
               "try{foo();}catch(...){baz();}", Style);

  Style.BraceWrapping.AfterFunction = true;
  Style.BraceWrapping.AfterStruct = false;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_MultiLine;
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  Style.ColumnLimit = 80;
  verifyFormat("void shortfunction() { bar(); }", Style);
  verifyFormat("struct T shortfunction() { return bar(); }", Style);
  verifyFormat("struct T {};", Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  verifyFormat("void shortfunction()\n"
               "{\n"
               "  bar();\n"
               "}",
               Style);
  verifyFormat("struct T shortfunction()\n"
               "{\n"
               "  return bar();\n"
               "}",
               Style);
  verifyFormat("struct T {};", Style);

  Style.BraceWrapping.AfterFunction = false;
  Style.BraceWrapping.AfterStruct = true;
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  verifyFormat("void shortfunction() { bar(); }", Style);
  verifyFormat("struct T shortfunction() { return bar(); }", Style);
  verifyFormat("struct T\n"
               "{\n"
               "};",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  verifyFormat("void shortfunction() {\n"
               "  bar();\n"
               "}",
               Style);
  verifyFormat("struct T shortfunction() {\n"
               "  return bar();\n"
               "}",
               Style);
  verifyFormat("struct T\n"
               "{\n"
               "};",
               Style);
}

TEST_F(FormatTest, BeforeWhile) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BraceBreakingStyle::BS_Custom;

  verifyFormat("do {\n"
               "  foo();\n"
               "} while (1);",
               Style);
  Style.BraceWrapping.BeforeWhile = true;
  verifyFormat("do {\n"
               "  foo();\n"
               "}\n"
               "while (1);",
               Style);
}

//===----------------------------------------------------------------------===//
// Tests for classes, namespaces, etc.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, DoesNotBreakSemiAfterClassDecl) {
  verifyFormat("class A {};");
}

TEST_F(FormatTest, UnderstandsAccessSpecifiers) {
  verifyFormat("class A {\n"
               "public:\n"
               "public: // comment\n"
               "protected:\n"
               "private:\n"
               "  void f() {}\n"
               "};");
  verifyFormat("export class A {\n"
               "public:\n"
               "public: // comment\n"
               "protected:\n"
               "private:\n"
               "  void f() {}\n"
               "};");
  verifyGoogleFormat("class A {\n"
                     " public:\n"
                     " protected:\n"
                     " private:\n"
                     "  void f() {}\n"
                     "};");
  verifyGoogleFormat("export class A {\n"
                     " public:\n"
                     " protected:\n"
                     " private:\n"
                     "  void f() {}\n"
                     "};");
  verifyFormat("class A {\n"
               "public slots:\n"
               "  void f1() {}\n"
               "public Q_SLOTS:\n"
               "  void f2() {}\n"
               "protected slots:\n"
               "  void f3() {}\n"
               "protected Q_SLOTS:\n"
               "  void f4() {}\n"
               "private slots:\n"
               "  void f5() {}\n"
               "private Q_SLOTS:\n"
               "  void f6() {}\n"
               "signals:\n"
               "  void g1();\n"
               "Q_SIGNALS:\n"
               "  void g2();\n"
               "};");

  // Don't interpret 'signals' the wrong way.
  verifyFormat("signals.set();");
  verifyFormat("for (Signals signals : f()) {\n}");
  verifyFormat("{\n"
               "  signals.set(); // This needs indentation.\n"
               "}");
  verifyFormat("void f() {\n"
               "label:\n"
               "  signals.baz();\n"
               "}");
  verifyFormat("private[1];");
  verifyFormat("testArray[public] = 1;");
  verifyFormat("public();");
  verifyFormat("myFunc(public);");
  verifyFormat("std::vector<int> testVec = {private};");
  verifyFormat("private.p = 1;");
  verifyFormat("void function(private...){};");
  verifyFormat("if (private && public)");
  verifyFormat("private &= true;");
  verifyFormat("int x = private * public;");
  verifyFormat("public *= private;");
  verifyFormat("int x = public + private;");
  verifyFormat("private++;");
  verifyFormat("++private;");
  verifyFormat("public += private;");
  verifyFormat("public = public - private;");
  verifyFormat("public->foo();");
  verifyFormat("private--;");
  verifyFormat("--private;");
  verifyFormat("public -= 1;");
  verifyFormat("if (!private && !public)");
  verifyFormat("public != private;");
  verifyFormat("int x = public / private;");
  verifyFormat("public /= 2;");
  verifyFormat("public = public % 2;");
  verifyFormat("public %= 2;");
  verifyFormat("if (public < private)");
  verifyFormat("public << private;");
  verifyFormat("public <<= private;");
  verifyFormat("if (public > private)");
  verifyFormat("public >> private;");
  verifyFormat("public >>= private;");
  verifyFormat("public ^ private;");
  verifyFormat("public ^= private;");
  verifyFormat("public | private;");
  verifyFormat("public |= private;");
  verifyFormat("auto x = private ? 1 : 2;");
  verifyFormat("if (public == private)");
  verifyFormat("void foo(public, private)");
  verifyFormat("public::foo();");

  verifyFormat("class A {\n"
               "public:\n"
               "  std::unique_ptr<int *[]> b() { return nullptr; }\n"
               "\n"
               "private:\n"
               "  int c;\n"
               "};\n"
               "class B {\n"
               "public:\n"
               "  std::unique_ptr<int *[] /* okay */> b() { return nullptr; }\n"
               "\n"
               "private:\n"
               "  int c;\n"
               "};");
}

TEST_F(FormatTest, SeparatesLogicalBlocks) {
  verifyFormat("class A {\n"
               "public:\n"
               "  void f();\n"
               "\n"
               "private:\n"
               "  void g() {}\n"
               "  // test\n"
               "protected:\n"
               "  int h;\n"
               "};",
               "class A {\n"
               "public:\n"
               "void f();\n"
               "private:\n"
               "void g() {}\n"
               "// test\n"
               "protected:\n"
               "int h;\n"
               "};");
  verifyFormat("class A {\n"
               "protected:\n"
               "public:\n"
               "  void f();\n"
               "};",
               "class A {\n"
               "protected:\n"
               "\n"
               "public:\n"
               "\n"
               "  void f();\n"
               "};");

  // Even ensure proper spacing inside macros.
  verifyFormat("#define B     \\\n"
               "  class A {   \\\n"
               "   protected: \\\n"
               "   public:    \\\n"
               "    void f(); \\\n"
               "  };",
               "#define B     \\\n"
               "  class A {   \\\n"
               "   protected: \\\n"
               "              \\\n"
               "   public:    \\\n"
               "              \\\n"
               "    void f(); \\\n"
               "  };",
               getGoogleStyle());
  // But don't remove empty lines after macros ending in access specifiers.
  verifyFormat("#define A private:\n"
               "\n"
               "int i;",
               "#define A         private:\n"
               "\n"
               "int              i;");
}

TEST_F(FormatTest, FormatsClasses) {
  verifyFormat("class A : public B {};");
  verifyFormat("class A : public ::B {};");

  verifyFormat(
      "class AAAAAAAAAAAAAAAAAAAA : public BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,\n"
      "                             public CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC {};");
  verifyFormat("class AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
               "    : public BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,\n"
               "      public CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC {};");
  verifyFormat(
      "class A : public B, public C, public D, public E, public F {};");
  verifyFormat("class AAAAAAAAAAAA : public B,\n"
               "                     public C,\n"
               "                     public D,\n"
               "                     public E,\n"
               "                     public F,\n"
               "                     public G {};");

  verifyFormat("class\n"
               "    ReallyReallyLongClassName {\n"
               "  int i;\n"
               "};",
               getLLVMStyleWithColumns(32));
  verifyFormat("struct aaaaaaaaaaaaa : public aaaaaaaaaaaaaaaaaaa< // break\n"
               "                           aaaaaaaaaaaaaaaa> {};");
  verifyFormat("struct aaaaaaaaaaaaaaaaaaaa\n"
               "    : public aaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaaaaaa,\n"
               "                                 aaaaaaaaaaaaaaaaaaaaaa> {};");
  verifyFormat("template <class R, class C>\n"
               "struct Aaaaaaaaaaaaaaaaa<R (C::*)(int) const>\n"
               "    : Aaaaaaaaaaaaaaaaa<R (C::*)(int)> {};");
  verifyFormat("class ::A::B {};");
}

TEST_F(FormatTest, BreakInheritanceStyle) {
  FormatStyle StyleWithInheritanceBreakBeforeComma = getLLVMStyle();
  StyleWithInheritanceBreakBeforeComma.BreakInheritanceList =
      FormatStyle::BILS_BeforeComma;
  verifyFormat("class MyClass : public X {};",
               StyleWithInheritanceBreakBeforeComma);
  verifyFormat("class MyClass\n"
               "    : public X\n"
               "    , public Y {};",
               StyleWithInheritanceBreakBeforeComma);
  verifyFormat("class AAAAAAAAAAAAAAAAAAAAAA\n"
               "    : public BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n"
               "    , public CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC {};",
               StyleWithInheritanceBreakBeforeComma);
  verifyFormat("struct aaaaaaaaaaaaa\n"
               "    : public aaaaaaaaaaaaaaaaaaa< // break\n"
               "          aaaaaaaaaaaaaaaa> {};",
               StyleWithInheritanceBreakBeforeComma);

  FormatStyle StyleWithInheritanceBreakAfterColon = getLLVMStyle();
  StyleWithInheritanceBreakAfterColon.BreakInheritanceList =
      FormatStyle::BILS_AfterColon;
  verifyFormat("class MyClass : public X {};",
               StyleWithInheritanceBreakAfterColon);
  verifyFormat("class MyClass : public X, public Y {};",
               StyleWithInheritanceBreakAfterColon);
  verifyFormat("class AAAAAAAAAAAAAAAAAAAAAA :\n"
               "    public BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,\n"
               "    public CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC {};",
               StyleWithInheritanceBreakAfterColon);
  verifyFormat("struct aaaaaaaaaaaaa :\n"
               "    public aaaaaaaaaaaaaaaaaaa< // break\n"
               "        aaaaaaaaaaaaaaaa> {};",
               StyleWithInheritanceBreakAfterColon);

  FormatStyle StyleWithInheritanceBreakAfterComma = getLLVMStyle();
  StyleWithInheritanceBreakAfterComma.BreakInheritanceList =
      FormatStyle::BILS_AfterComma;
  verifyFormat("class MyClass : public X {};",
               StyleWithInheritanceBreakAfterComma);
  verifyFormat("class MyClass : public X,\n"
               "                public Y {};",
               StyleWithInheritanceBreakAfterComma);
  verifyFormat(
      "class AAAAAAAAAAAAAAAAAAAAAA : public BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,\n"
      "                               public CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC "
      "{};",
      StyleWithInheritanceBreakAfterComma);
  verifyFormat("struct aaaaaaaaaaaaa : public aaaaaaaaaaaaaaaaaaa< // break\n"
               "                           aaaaaaaaaaaaaaaa> {};",
               StyleWithInheritanceBreakAfterComma);
  verifyFormat("class AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
               "    : public OnceBreak,\n"
               "      public AlwaysBreak,\n"
               "      EvenBasesFitInOneLine {};",
               StyleWithInheritanceBreakAfterComma);
}

TEST_F(FormatTest, FormatsVariableDeclarationsAfterRecord) {
  verifyFormat("class A {\n} a, b;");
  verifyFormat("struct A {\n} a, b;");
  verifyFormat("union A {\n} a, b;");

  verifyFormat("constexpr class A {\n} a, b;");
  verifyFormat("constexpr struct A {\n} a, b;");
  verifyFormat("constexpr union A {\n} a, b;");

  verifyFormat("namespace {\nclass A {\n} a, b;\n} // namespace");
  verifyFormat("namespace {\nstruct A {\n} a, b;\n} // namespace");
  verifyFormat("namespace {\nunion A {\n} a, b;\n} // namespace");

  verifyFormat("namespace {\nconstexpr class A {\n} a, b;\n} // namespace");
  verifyFormat("namespace {\nconstexpr struct A {\n} a, b;\n} // namespace");
  verifyFormat("namespace {\nconstexpr union A {\n} a, b;\n} // namespace");

  verifyFormat("namespace ns {\n"
               "class {\n"
               "} a, b;\n"
               "} // namespace ns");
  verifyFormat("namespace ns {\n"
               "const class {\n"
               "} a, b;\n"
               "} // namespace ns");
  verifyFormat("namespace ns {\n"
               "constexpr class C {\n"
               "} a, b;\n"
               "} // namespace ns");
  verifyFormat("namespace ns {\n"
               "class { /* comment */\n"
               "} a, b;\n"
               "} // namespace ns");
  verifyFormat("namespace ns {\n"
               "const class { /* comment */\n"
               "} a, b;\n"
               "} // namespace ns");
}

TEST_F(FormatTest, FormatsEnum) {
  verifyFormat("enum {\n"
               "  Zero,\n"
               "  One = 1,\n"
               "  Two = One + 1,\n"
               "  Three = (One + Two),\n"
               "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
               "  Five = (One, Two, Three, Four, 5)\n"
               "};");
  verifyGoogleFormat("enum {\n"
                     "  Zero,\n"
                     "  One = 1,\n"
                     "  Two = One + 1,\n"
                     "  Three = (One + Two),\n"
                     "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
                     "  Five = (One, Two, Three, Four, 5)\n"
                     "};");
  verifyFormat("enum Enum {};");
  verifyFormat("enum {};");
  verifyFormat("enum X E {} d;");
  verifyFormat("enum __attribute__((...)) E {} d;");
  verifyFormat("enum __declspec__((...)) E {} d;");
  verifyFormat("enum [[nodiscard]] E {} d;");
  verifyFormat("enum {\n"
               "  Bar = Foo<int, int>::value\n"
               "};",
               getLLVMStyleWithColumns(30));

  verifyFormat("enum ShortEnum { A, B, C };");
  verifyGoogleFormat("enum ShortEnum { A, B, C };");

  verifyFormat("enum KeepEmptyLines {\n"
               "  ONE,\n"
               "\n"
               "  TWO,\n"
               "\n"
               "  THREE\n"
               "}",
               "enum KeepEmptyLines {\n"
               "  ONE,\n"
               "\n"
               "  TWO,\n"
               "\n"
               "\n"
               "  THREE\n"
               "}");
  verifyFormat("enum E { // comment\n"
               "  ONE,\n"
               "  TWO\n"
               "};\n"
               "int i;");

  FormatStyle EightIndent = getLLVMStyle();
  EightIndent.IndentWidth = 8;
  verifyFormat("enum {\n"
               "        VOID,\n"
               "        CHAR,\n"
               "        SHORT,\n"
               "        INT,\n"
               "        LONG,\n"
               "        SIGNED,\n"
               "        UNSIGNED,\n"
               "        BOOL,\n"
               "        FLOAT,\n"
               "        DOUBLE,\n"
               "        COMPLEX\n"
               "};",
               EightIndent);

  verifyFormat("enum [[nodiscard]] E {\n"
               "  ONE,\n"
               "  TWO,\n"
               "};");
  verifyFormat("enum [[nodiscard]] E {\n"
               "  // Comment 1\n"
               "  ONE,\n"
               "  // Comment 2\n"
               "  TWO,\n"
               "};");

  // Not enums.
  verifyFormat("enum X f() {\n"
               "  a();\n"
               "  return 42;\n"
               "}");
  verifyFormat("enum X Type::f() {\n"
               "  a();\n"
               "  return 42;\n"
               "}");
  verifyFormat("enum ::X f() {\n"
               "  a();\n"
               "  return 42;\n"
               "}");
  verifyFormat("enum ns::X f() {\n"
               "  a();\n"
               "  return 42;\n"
               "}");
}

TEST_F(FormatTest, FormatsEnumsWithErrors) {
  verifyFormat("enum Type {\n"
               "  One = 0; // These semicolons should be commas.\n"
               "  Two = 1;\n"
               "};");
  verifyFormat("namespace n {\n"
               "enum Type {\n"
               "  One,\n"
               "  Two, // missing };\n"
               "  int i;\n"
               "}\n"
               "void g() {}");
}

TEST_F(FormatTest, FormatsEnumStruct) {
  verifyFormat("enum struct {\n"
               "  Zero,\n"
               "  One = 1,\n"
               "  Two = One + 1,\n"
               "  Three = (One + Two),\n"
               "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
               "  Five = (One, Two, Three, Four, 5)\n"
               "};");
  verifyFormat("enum struct Enum {};");
  verifyFormat("enum struct {};");
  verifyFormat("enum struct X E {} d;");
  verifyFormat("enum struct __attribute__((...)) E {} d;");
  verifyFormat("enum struct __declspec__((...)) E {} d;");
  verifyFormat("enum struct [[nodiscard]] E {} d;");
  verifyFormat("enum struct X f() {\n  a();\n  return 42;\n}");

  verifyFormat("enum struct [[nodiscard]] E {\n"
               "  ONE,\n"
               "  TWO,\n"
               "};");
  verifyFormat("enum struct [[nodiscard]] E {\n"
               "  // Comment 1\n"
               "  ONE,\n"
               "  // Comment 2\n"
               "  TWO,\n"
               "};");
}

TEST_F(FormatTest, FormatsEnumClass) {
  verifyFormat("enum class {\n"
               "  Zero,\n"
               "  One = 1,\n"
               "  Two = One + 1,\n"
               "  Three = (One + Two),\n"
               "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
               "  Five = (One, Two, Three, Four, 5)\n"
               "};");
  verifyFormat("enum class Enum {};");
  verifyFormat("enum class {};");
  verifyFormat("enum class X E {} d;");
  verifyFormat("enum class __attribute__((...)) E {} d;");
  verifyFormat("enum class __declspec__((...)) E {} d;");
  verifyFormat("enum class [[nodiscard]] E {} d;");
  verifyFormat("enum class X f() {\n  a();\n  return 42;\n}");

  verifyFormat("enum class [[nodiscard]] E {\n"
               "  ONE,\n"
               "  TWO,\n"
               "};");
  verifyFormat("enum class [[nodiscard]] E {\n"
               "  // Comment 1\n"
               "  ONE,\n"
               "  // Comment 2\n"
               "  TWO,\n"
               "};");
}

TEST_F(FormatTest, FormatsEnumTypes) {
  verifyFormat("enum X : int {\n"
               "  A, // Force multiple lines.\n"
               "  B\n"
               "};");
  verifyFormat("enum X : int { A, B };");
  verifyFormat("enum X : std::uint32_t { A, B };");
}

TEST_F(FormatTest, FormatsTypedefEnum) {
  FormatStyle Style = getLLVMStyleWithColumns(40);
  verifyFormat("typedef enum {} EmptyEnum;");
  verifyFormat("typedef enum { A, B, C } ShortEnum;");
  verifyFormat("typedef enum {\n"
               "  ZERO = 0,\n"
               "  ONE = 1,\n"
               "  TWO = 2,\n"
               "  THREE = 3\n"
               "} LongEnum;",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterEnum = true;
  verifyFormat("typedef enum {} EmptyEnum;");
  verifyFormat("typedef enum { A, B, C } ShortEnum;");
  verifyFormat("typedef enum\n"
               "{\n"
               "  ZERO = 0,\n"
               "  ONE = 1,\n"
               "  TWO = 2,\n"
               "  THREE = 3\n"
               "} LongEnum;",
               Style);
}

TEST_F(FormatTest, FormatsNSEnums) {
  verifyGoogleFormat("typedef NS_ENUM(NSInteger, SomeName) { AAA, BBB }");
  verifyGoogleFormat(
      "typedef NS_CLOSED_ENUM(NSInteger, SomeName) { AAA, BBB }");
  verifyGoogleFormat("typedef NS_ENUM(NSInteger, MyType) {\n"
                     "  // Information about someDecentlyLongValue.\n"
                     "  someDecentlyLongValue,\n"
                     "  // Information about anotherDecentlyLongValue.\n"
                     "  anotherDecentlyLongValue,\n"
                     "  // Information about aThirdDecentlyLongValue.\n"
                     "  aThirdDecentlyLongValue\n"
                     "};");
  verifyGoogleFormat("typedef NS_CLOSED_ENUM(NSInteger, MyType) {\n"
                     "  // Information about someDecentlyLongValue.\n"
                     "  someDecentlyLongValue,\n"
                     "  // Information about anotherDecentlyLongValue.\n"
                     "  anotherDecentlyLongValue,\n"
                     "  // Information about aThirdDecentlyLongValue.\n"
                     "  aThirdDecentlyLongValue\n"
                     "};");
  verifyGoogleFormat("typedef NS_OPTIONS(NSInteger, MyType) {\n"
                     "  a = 1,\n"
                     "  b = 2,\n"
                     "  c = 3,\n"
                     "};");
  verifyGoogleFormat("typedef CF_ENUM(NSInteger, MyType) {\n"
                     "  a = 1,\n"
                     "  b = 2,\n"
                     "  c = 3,\n"
                     "};");
  verifyGoogleFormat("typedef CF_CLOSED_ENUM(NSInteger, MyType) {\n"
                     "  a = 1,\n"
                     "  b = 2,\n"
                     "  c = 3,\n"
                     "};");
  verifyGoogleFormat("typedef CF_OPTIONS(NSInteger, MyType) {\n"
                     "  a = 1,\n"
                     "  b = 2,\n"
                     "  c = 3,\n"
                     "};");
}

TEST_F(FormatTest, FormatsBitfields) {
  verifyFormat("struct Bitfields {\n"
               "  unsigned sClass : 8;\n"
               "  unsigned ValueKind : 2;\n"
               "};");
  verifyFormat("struct A {\n"
               "  int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa : 1,\n"
               "      bbbbbbbbbbbbbbbbbbbbbbbbb;\n"
               "};");
  verifyFormat("struct MyStruct {\n"
               "  uchar data;\n"
               "  uchar : 8;\n"
               "  uchar : 8;\n"
               "  uchar other;\n"
               "};");
  FormatStyle Style = getLLVMStyle();
  Style.BitFieldColonSpacing = FormatStyle::BFCS_None;
  verifyFormat("struct Bitfields {\n"
               "  unsigned sClass:8;\n"
               "  unsigned ValueKind:2;\n"
               "  uchar other;\n"
               "};",
               Style);
  verifyFormat("struct A {\n"
               "  int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1,\n"
               "      bbbbbbbbbbbbbbbbbbbbbbbbb:2;\n"
               "};",
               Style);
  Style.BitFieldColonSpacing = FormatStyle::BFCS_Before;
  verifyFormat("struct Bitfields {\n"
               "  unsigned sClass :8;\n"
               "  unsigned ValueKind :2;\n"
               "  uchar other;\n"
               "};",
               Style);
  Style.BitFieldColonSpacing = FormatStyle::BFCS_After;
  verifyFormat("struct Bitfields {\n"
               "  unsigned sClass: 8;\n"
               "  unsigned ValueKind: 2;\n"
               "  uchar other;\n"
               "};",
               Style);
}

TEST_F(FormatTest, FormatsNamespaces) {
  FormatStyle LLVMWithNoNamespaceFix = getLLVMStyle();
  LLVMWithNoNamespaceFix.FixNamespaceComments = false;

  verifyFormat("namespace some_namespace {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("#define M(x) x##x\n"
               "namespace M(x) {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("#define M(x) x##x\n"
               "namespace N::inline M(x) {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("#define M(x) x##x\n"
               "namespace M(x)::inline N {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("#define M(x) x##x\n"
               "namespace N::M(x) {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("#define M(x) x##x\n"
               "namespace M::N(x) {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("namespace N::inline D {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("namespace N::inline D::E {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("namespace [[deprecated(\"foo[bar\")]] some_namespace {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("/* something */ namespace some_namespace {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("namespace {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("/* something */ namespace {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("inline namespace X {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("/* something */ inline namespace X {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("export namespace X {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}",
               LLVMWithNoNamespaceFix);
  verifyFormat("using namespace some_namespace;\n"
               "class A {};\n"
               "void f() { f(); }",
               LLVMWithNoNamespaceFix);

  // This code is more common than we thought; if we
  // layout this correctly the semicolon will go into
  // its own line, which is undesirable.
  verifyFormat("namespace {};", LLVMWithNoNamespaceFix);
  verifyFormat("namespace {\n"
               "class A {};\n"
               "};",
               LLVMWithNoNamespaceFix);

  verifyFormat("namespace {\n"
               "int SomeVariable = 0; // comment\n"
               "} // namespace",
               LLVMWithNoNamespaceFix);
  verifyFormat("#ifndef HEADER_GUARD\n"
               "#define HEADER_GUARD\n"
               "namespace my_namespace {\n"
               "int i;\n"
               "} // my_namespace\n"
               "#endif // HEADER_GUARD",
               "#ifndef HEADER_GUARD\n"
               " #define HEADER_GUARD\n"
               "   namespace my_namespace {\n"
               "int i;\n"
               "}    // my_namespace\n"
               "#endif    // HEADER_GUARD",
               LLVMWithNoNamespaceFix);

  verifyFormat("namespace A::B {\n"
               "class C {};\n"
               "}",
               LLVMWithNoNamespaceFix);

  FormatStyle Style = getLLVMStyle();
  Style.NamespaceIndentation = FormatStyle::NI_All;
  verifyFormat("namespace out {\n"
               "  int i;\n"
               "  namespace in {\n"
               "    int i;\n"
               "  } // namespace in\n"
               "} // namespace out",
               "namespace out {\n"
               "int i;\n"
               "namespace in {\n"
               "int i;\n"
               "} // namespace in\n"
               "} // namespace out",
               Style);

  FormatStyle ShortInlineFunctions = getLLVMStyle();
  ShortInlineFunctions.NamespaceIndentation = FormatStyle::NI_All;
  ShortInlineFunctions.AllowShortFunctionsOnASingleLine =
      FormatStyle::SFS_Inline;
  verifyFormat("namespace {\n"
               "  void f() {\n"
               "    return;\n"
               "  }\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace { /* comment */\n"
               "  void f() {\n"
               "    return;\n"
               "  }\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace { // comment\n"
               "  void f() {\n"
               "    return;\n"
               "  }\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  int some_int;\n"
               "  void f() {\n"
               "    return;\n"
               "  }\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace interface {\n"
               "  void f() {\n"
               "    return;\n"
               "  }\n"
               "} // namespace interface",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  class X {\n"
               "    void f() { return; }\n"
               "  };\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  class X { /* comment */\n"
               "    void f() { return; }\n"
               "  };\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  class X { // comment\n"
               "    void f() { return; }\n"
               "  };\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  struct X {\n"
               "    void f() { return; }\n"
               "  };\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  union X {\n"
               "    void f() { return; }\n"
               "  };\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("extern \"C\" {\n"
               "void f() {\n"
               "  return;\n"
               "}\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  class X {\n"
               "    void f() { return; }\n"
               "  } x;\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  [[nodiscard]] class X {\n"
               "    void f() { return; }\n"
               "  };\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  static class X {\n"
               "    void f() { return; }\n"
               "  } x;\n"
               "} // namespace",
               ShortInlineFunctions);
  verifyFormat("namespace {\n"
               "  constexpr class X {\n"
               "    void f() { return; }\n"
               "  } x;\n"
               "} // namespace",
               ShortInlineFunctions);

  ShortInlineFunctions.IndentExternBlock = FormatStyle::IEBS_Indent;
  verifyFormat("extern \"C\" {\n"
               "  void f() {\n"
               "    return;\n"
               "  }\n"
               "} // namespace",
               ShortInlineFunctions);

  Style.NamespaceIndentation = FormatStyle::NI_Inner;
  verifyFormat("namespace out {\n"
               "int i;\n"
               "namespace in {\n"
               "  int i;\n"
               "} // namespace in\n"
               "} // namespace out",
               "namespace out {\n"
               "int i;\n"
               "namespace in {\n"
               "int i;\n"
               "} // namespace in\n"
               "} // namespace out",
               Style);

  Style.NamespaceIndentation = FormatStyle::NI_None;
  verifyFormat("template <class T>\n"
               "concept a_concept = X<>;\n"
               "namespace B {\n"
               "struct b_struct {};\n"
               "} // namespace B",
               Style);
  verifyFormat("template <int I>\n"
               "constexpr void foo()\n"
               "  requires(I == 42)\n"
               "{}\n"
               "namespace ns {\n"
               "void foo() {}\n"
               "} // namespace ns",
               Style);

  FormatStyle LLVMWithCompactInnerNamespace = getLLVMStyle();
  LLVMWithCompactInnerNamespace.CompactNamespaces = true;
  LLVMWithCompactInnerNamespace.NamespaceIndentation = FormatStyle::NI_Inner;
  verifyFormat("namespace ns1 { namespace ns2 { namespace ns3 {\n"
               "// block for debug mode\n"
               "#ifndef NDEBUG\n"
               "#endif\n"
               "}}} // namespace ns1::ns2::ns3",
               LLVMWithCompactInnerNamespace);
}

TEST_F(FormatTest, NamespaceMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.NamespaceMacros.push_back("TESTSUITE");

  verifyFormat("TESTSUITE(A) {\n"
               "int foo();\n"
               "} // TESTSUITE(A)",
               Style);

  verifyFormat("TESTSUITE(A, B) {\n"
               "int foo();\n"
               "} // TESTSUITE(A)",
               Style);

  // Properly indent according to NamespaceIndentation style
  Style.NamespaceIndentation = FormatStyle::NI_All;
  verifyFormat("TESTSUITE(A) {\n"
               "  int foo();\n"
               "} // TESTSUITE(A)",
               Style);
  verifyFormat("TESTSUITE(A) {\n"
               "  namespace B {\n"
               "    int foo();\n"
               "  } // namespace B\n"
               "} // TESTSUITE(A)",
               Style);
  verifyFormat("namespace A {\n"
               "  TESTSUITE(B) {\n"
               "    int foo();\n"
               "  } // TESTSUITE(B)\n"
               "} // namespace A",
               Style);

  Style.NamespaceIndentation = FormatStyle::NI_Inner;
  verifyFormat("TESTSUITE(A) {\n"
               "TESTSUITE(B) {\n"
               "  int foo();\n"
               "} // TESTSUITE(B)\n"
               "} // TESTSUITE(A)",
               Style);
  verifyFormat("TESTSUITE(A) {\n"
               "namespace B {\n"
               "  int foo();\n"
               "} // namespace B\n"
               "} // TESTSUITE(A)",
               Style);
  verifyFormat("namespace A {\n"
               "TESTSUITE(B) {\n"
               "  int foo();\n"
               "} // TESTSUITE(B)\n"
               "} // namespace A",
               Style);

  // Properly merge namespace-macros blocks in CompactNamespaces mode
  Style.NamespaceIndentation = FormatStyle::NI_None;
  Style.CompactNamespaces = true;
  verifyFormat("TESTSUITE(A) { TESTSUITE(B) {\n"
               "}} // TESTSUITE(A::B)",
               Style);

  verifyFormat("TESTSUITE(out) { TESTSUITE(in) {\n"
               "}} // TESTSUITE(out::in)",
               "TESTSUITE(out) {\n"
               "TESTSUITE(in) {\n"
               "} // TESTSUITE(in)\n"
               "} // TESTSUITE(out)",
               Style);

  verifyFormat("TESTSUITE(out) { TESTSUITE(in) {\n"
               "}} // TESTSUITE(out::in)",
               "TESTSUITE(out) {\n"
               "TESTSUITE(in) {\n"
               "} // TESTSUITE(in)\n"
               "} // TESTSUITE(out)",
               Style);

  // Do not merge different namespaces/macros
  verifyFormat("namespace out {\n"
               "TESTSUITE(in) {\n"
               "} // TESTSUITE(in)\n"
               "} // namespace out",
               Style);
  verifyFormat("TESTSUITE(out) {\n"
               "namespace in {\n"
               "} // namespace in\n"
               "} // TESTSUITE(out)",
               Style);
  Style.NamespaceMacros.push_back("FOOBAR");
  verifyFormat("TESTSUITE(out) {\n"
               "FOOBAR(in) {\n"
               "} // FOOBAR(in)\n"
               "} // TESTSUITE(out)",
               Style);
}

TEST_F(FormatTest, FormatsCompactNamespaces) {
  FormatStyle Style = getLLVMStyle();
  Style.CompactNamespaces = true;
  Style.NamespaceMacros.push_back("TESTSUITE");

  verifyFormat("namespace A { namespace B {\n"
               "}} // namespace A::B",
               Style);

  verifyFormat("namespace out { namespace in {\n"
               "}} // namespace out::in",
               "namespace out {\n"
               "namespace in {\n"
               "} // namespace in\n"
               "} // namespace out",
               Style);

  // Only namespaces which have both consecutive opening and end get compacted
  verifyFormat("namespace out {\n"
               "namespace in1 {\n"
               "} // namespace in1\n"
               "namespace in2 {\n"
               "} // namespace in2\n"
               "} // namespace out",
               Style);

  verifyFormat("namespace out {\n"
               "int i;\n"
               "namespace in {\n"
               "int j;\n"
               "} // namespace in\n"
               "int k;\n"
               "} // namespace out",
               "namespace out { int i;\n"
               "namespace in { int j; } // namespace in\n"
               "int k; } // namespace out",
               Style);

  verifyFormat("namespace A { namespace B { namespace C {\n"
               "}}} // namespace A::B::C",
               "namespace A { namespace B {\n"
               "namespace C {\n"
               "}} // namespace B::C\n"
               "} // namespace A",
               Style);

  Style.ColumnLimit = 40;
  verifyFormat("namespace aaaaaaaaaa {\n"
               "namespace bbbbbbbbbb {\n"
               "}} // namespace aaaaaaaaaa::bbbbbbbbbb",
               "namespace aaaaaaaaaa {\n"
               "namespace bbbbbbbbbb {\n"
               "} // namespace bbbbbbbbbb\n"
               "} // namespace aaaaaaaaaa",
               Style);

  verifyFormat("namespace aaaaaa { namespace bbbbbb {\n"
               "namespace cccccc {\n"
               "}}} // namespace aaaaaa::bbbbbb::cccccc",
               "namespace aaaaaa {\n"
               "namespace bbbbbb {\n"
               "namespace cccccc {\n"
               "} // namespace cccccc\n"
               "} // namespace bbbbbb\n"
               "} // namespace aaaaaa",
               Style);
  Style.ColumnLimit = 80;

  // Extra semicolon after 'inner' closing brace prevents merging
  verifyFormat("namespace out { namespace in {\n"
               "}; } // namespace out::in",
               "namespace out {\n"
               "namespace in {\n"
               "}; // namespace in\n"
               "} // namespace out",
               Style);

  // Extra semicolon after 'outer' closing brace is conserved
  verifyFormat("namespace out { namespace in {\n"
               "}}; // namespace out::in",
               "namespace out {\n"
               "namespace in {\n"
               "} // namespace in\n"
               "}; // namespace out",
               Style);

  Style.NamespaceIndentation = FormatStyle::NI_All;
  verifyFormat("namespace out { namespace in {\n"
               "  int i;\n"
               "}} // namespace out::in",
               "namespace out {\n"
               "namespace in {\n"
               "int i;\n"
               "} // namespace in\n"
               "} // namespace out",
               Style);
  verifyFormat("namespace out { namespace mid {\n"
               "  namespace in {\n"
               "    int j;\n"
               "  } // namespace in\n"
               "  int k;\n"
               "}} // namespace out::mid",
               "namespace out { namespace mid {\n"
               "namespace in { int j; } // namespace in\n"
               "int k; }} // namespace out::mid",
               Style);

  verifyFormat("namespace A { namespace B { namespace C {\n"
               "  int i;\n"
               "}}} // namespace A::B::C\n"
               "int main() {\n"
               "  if (true)\n"
               "    return 0;\n"
               "}",
               "namespace A { namespace B {\n"
               "namespace C {\n"
               "  int i;\n"
               "}} // namespace B::C\n"
               "} // namespace A\n"
               "int main() {\n"
               "  if (true)\n"
               "    return 0;\n"
               "}",
               Style);

  verifyFormat("namespace A { namespace B { namespace C {\n"
               "#ifdef FOO\n"
               "  int i;\n"
               "#endif\n"
               "}}} // namespace A::B::C\n"
               "int main() {\n"
               "  if (true)\n"
               "    return 0;\n"
               "}",
               "namespace A { namespace B {\n"
               "namespace C {\n"
               "#ifdef FOO\n"
               "  int i;\n"
               "#endif\n"
               "}} // namespace B::C\n"
               "} // namespace A\n"
               "int main() {\n"
               "  if (true)\n"
               "    return 0;\n"
               "}",
               Style);

  Style.NamespaceIndentation = FormatStyle::NI_Inner;
  verifyFormat("namespace out { namespace in {\n"
               "  int i;\n"
               "}} // namespace out::in",
               "namespace out {\n"
               "namespace in {\n"
               "int i;\n"
               "} // namespace in\n"
               "} // namespace out",
               Style);
  verifyFormat("namespace out { namespace mid { namespace in {\n"
               "  int i;\n"
               "}}} // namespace out::mid::in",
               "namespace out {\n"
               "namespace mid {\n"
               "namespace in {\n"
               "int i;\n"
               "} // namespace in\n"
               "} // namespace mid\n"
               "} // namespace out",
               Style);

  Style.CompactNamespaces = true;
  Style.AllowShortLambdasOnASingleLine = FormatStyle::SLS_None;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.BeforeLambdaBody = true;
  verifyFormat("namespace out { namespace in {\n"
               "}} // namespace out::in",
               Style);
  verifyFormat("namespace out { namespace in {\n"
               "}} // namespace out::in",
               "namespace out {\n"
               "namespace in {\n"
               "} // namespace in\n"
               "} // namespace out",
               Style);
}

TEST_F(FormatTest, FormatsExternC) {
  verifyFormat("extern \"C\" {\nint a;");
  verifyFormat("extern \"C\" {}");
  verifyFormat("extern \"C\" {\n"
               "int foo();\n"
               "}");
  verifyFormat("extern \"C\" int foo() {}");
  verifyFormat("extern \"C\" int foo();");
  verifyFormat("extern \"C\" int foo() {\n"
               "  int i = 42;\n"
               "  return i;\n"
               "}");

  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterFunction = true;
  verifyFormat("extern \"C\" int foo() {}", Style);
  verifyFormat("extern \"C\" int foo();", Style);
  verifyFormat("extern \"C\" int foo()\n"
               "{\n"
               "  int i = 42;\n"
               "  return i;\n"
               "}",
               Style);

  Style.BraceWrapping.AfterExternBlock = true;
  Style.BraceWrapping.SplitEmptyRecord = false;
  verifyFormat("extern \"C\"\n"
               "{}",
               Style);
  verifyFormat("extern \"C\"\n"
               "{\n"
               "  int foo();\n"
               "}",
               Style);
}

TEST_F(FormatTest, IndentExternBlockStyle) {
  FormatStyle Style = getLLVMStyle();
  Style.IndentWidth = 2;

  Style.IndentExternBlock = FormatStyle::IEBS_Indent;
  verifyFormat("extern \"C\" { /*9*/\n"
               "}",
               Style);
  verifyFormat("extern \"C\" {\n"
               "  int foo10();\n"
               "}",
               Style);

  Style.IndentExternBlock = FormatStyle::IEBS_NoIndent;
  verifyFormat("extern \"C\" { /*11*/\n"
               "}",
               Style);
  verifyFormat("extern \"C\" {\n"
               "int foo12();\n"
               "}",
               Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterExternBlock = true;
  Style.IndentExternBlock = FormatStyle::IEBS_Indent;
  verifyFormat("extern \"C\"\n"
               "{ /*13*/\n"
               "}",
               Style);
  verifyFormat("extern \"C\"\n{\n"
               "  int foo14();\n"
               "}",
               Style);

  Style.BraceWrapping.AfterExternBlock = false;
  Style.IndentExternBlock = FormatStyle::IEBS_NoIndent;
  verifyFormat("extern \"C\" { /*15*/\n"
               "}",
               Style);
  verifyFormat("extern \"C\" {\n"
               "int foo16();\n"
               "}",
               Style);

  Style.BraceWrapping.AfterExternBlock = true;
  verifyFormat("extern \"C\"\n"
               "{ /*13*/\n"
               "}",
               Style);
  verifyFormat("extern \"C\"\n"
               "{\n"
               "int foo14();\n"
               "}",
               Style);

  Style.IndentExternBlock = FormatStyle::IEBS_Indent;
  verifyFormat("extern \"C\"\n"
               "{ /*13*/\n"
               "}",
               Style);
  verifyFormat("extern \"C\"\n"
               "{\n"
               "  int foo14();\n"
               "}",
               Style);
}

TEST_F(FormatTest, FormatsInlineASM) {
  verifyFormat("asm(\"xyz\" : \"=a\"(a), \"=d\"(b) : \"a\"(data));");
  verifyFormat("asm(\"nop\" ::: \"memory\");");
  verifyFormat(
      "asm(\"movq\\t%%rbx, %%rsi\\n\\t\"\n"
      "    \"cpuid\\n\\t\"\n"
      "    \"xchgq\\t%%rbx, %%rsi\\n\\t\"\n"
      "    : \"=a\"(*rEAX), \"=S\"(*rEBX), \"=c\"(*rECX), \"=d\"(*rEDX)\n"
      "    : \"a\"(value));");
  verifyFormat(
      "void NS_InvokeByIndex(void *that, unsigned int methodIndex) {\n"
      "  __asm {\n"
      "        mov     edx,[that] // vtable in edx\n"
      "        mov     eax,methodIndex\n"
      "        call    [edx][eax*4] // stdcall\n"
      "  }\n"
      "}",
      "void NS_InvokeByIndex(void *that,   unsigned int methodIndex) {\n"
      "    __asm {\n"
      "        mov     edx,[that] // vtable in edx\n"
      "        mov     eax,methodIndex\n"
      "        call    [edx][eax*4] // stdcall\n"
      "    }\n"
      "}");
  verifyNoChange("_asm {\n"
                 "  xor eax, eax;\n"
                 "  cpuid;\n"
                 "}");
  verifyFormat("void function() {\n"
               "  // comment\n"
               "  asm(\"\");\n"
               "}");
  verifyFormat("__asm {\n"
               "}\n"
               "int i;",
               "__asm   {\n"
               "}\n"
               "int   i;");

  auto Style = getLLVMStyleWithColumns(0);
  const StringRef Code1{"asm(\"xyz\" : \"=a\"(a), \"=d\"(b) : \"a\"(data));"};
  const StringRef Code2{"asm(\"xyz\"\n"
                        "    : \"=a\"(a), \"=d\"(b)\n"
                        "    : \"a\"(data));"};
  const StringRef Code3{"asm(\"xyz\" : \"=a\"(a), \"=d\"(b)\n"
                        "    : \"a\"(data));"};

  Style.BreakBeforeInlineASMColon = FormatStyle::BBIAS_OnlyMultiline;
  verifyFormat(Code1, Style);
  verifyNoChange(Code2, Style);
  verifyNoChange(Code3, Style);

  Style.BreakBeforeInlineASMColon = FormatStyle::BBIAS_Always;
  verifyFormat(Code2, Code1, Style);
  verifyNoChange(Code2, Style);
  verifyFormat(Code2, Code3, Style);
}

TEST_F(FormatTest, FormatTryCatch) {
  verifyFormat("try {\n"
               "  throw a * b;\n"
               "} catch (int a) {\n"
               "  // Do nothing.\n"
               "} catch (...) {\n"
               "  exit(42);\n"
               "}");

  // Function-level try statements.
  verifyFormat("int f() try { return 4; } catch (...) {\n"
               "  return 5;\n"
               "}");
  verifyFormat("class A {\n"
               "  int a;\n"
               "  A() try : a(0) {\n"
               "  } catch (...) {\n"
               "    throw;\n"
               "  }\n"
               "};");
  verifyFormat("class A {\n"
               "  int a;\n"
               "  A() try : a(0), b{1} {\n"
               "  } catch (...) {\n"
               "    throw;\n"
               "  }\n"
               "};");
  verifyFormat("class A {\n"
               "  int a;\n"
               "  A() try : a(0), b{1}, c{2} {\n"
               "  } catch (...) {\n"
               "    throw;\n"
               "  }\n"
               "};");
  verifyFormat("class A {\n"
               "  int a;\n"
               "  A() try : a(0), b{1}, c{2} {\n"
               "    { // New scope.\n"
               "    }\n"
               "  } catch (...) {\n"
               "    throw;\n"
               "  }\n"
               "};");

  // Incomplete try-catch blocks.
  verifyIncompleteFormat("try {} catch (");
}

TEST_F(FormatTest, FormatTryAsAVariable) {
  verifyFormat("int try;");
  verifyFormat("int try, size;");
  verifyFormat("try = foo();");
  verifyFormat("if (try < size) {\n  return true;\n}");

  verifyFormat("int catch;");
  verifyFormat("int catch, size;");
  verifyFormat("catch = foo();");
  verifyFormat("if (catch < size) {\n  return true;\n}");

  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterFunction = true;
  Style.BraceWrapping.BeforeCatch = true;
  verifyFormat("try {\n"
               "  int bar = 1;\n"
               "}\n"
               "catch (...) {\n"
               "  int bar = 1;\n"
               "}",
               Style);
  verifyFormat("#if NO_EX\n"
               "try\n"
               "#endif\n"
               "{\n"
               "}\n"
               "#if NO_EX\n"
               "catch (...) {\n"
               "}",
               Style);
  verifyFormat("try /* abc */ {\n"
               "  int bar = 1;\n"
               "}\n"
               "catch (...) {\n"
               "  int bar = 1;\n"
               "}",
               Style);
  verifyFormat("try\n"
               "// abc\n"
               "{\n"
               "  int bar = 1;\n"
               "}\n"
               "catch (...) {\n"
               "  int bar = 1;\n"
               "}",
               Style);
}

TEST_F(FormatTest, FormatSEHTryCatch) {
  verifyFormat("__try {\n"
               "  int a = b * c;\n"
               "} __except (EXCEPTION_EXECUTE_HANDLER) {\n"
               "  // Do nothing.\n"
               "}");

  verifyFormat("__try {\n"
               "  int a = b * c;\n"
               "} __finally {\n"
               "  // Do nothing.\n"
               "}");

  verifyFormat("DEBUG({\n"
               "  __try {\n"
               "  } __finally {\n"
               "  }\n"
               "});");
}

TEST_F(FormatTest, IncompleteTryCatchBlocks) {
  verifyFormat("try {\n"
               "  f();\n"
               "} catch {\n"
               "  g();\n"
               "}");
  verifyFormat("try {\n"
               "  f();\n"
               "} catch (A a) MACRO(x) {\n"
               "  g();\n"
               "} catch (B b) MACRO(x) {\n"
               "  g();\n"
               "}");
}

TEST_F(FormatTest, FormatTryCatchBraceStyles) {
  FormatStyle Style = getLLVMStyle();
  for (auto BraceStyle : {FormatStyle::BS_Attach, FormatStyle::BS_Mozilla,
                          FormatStyle::BS_WebKit}) {
    Style.BreakBeforeBraces = BraceStyle;
    verifyFormat("try {\n"
                 "  // something\n"
                 "} catch (...) {\n"
                 "  // something\n"
                 "}",
                 Style);
  }
  Style.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  verifyFormat("try {\n"
               "  // something\n"
               "}\n"
               "catch (...) {\n"
               "  // something\n"
               "}",
               Style);
  verifyFormat("__try {\n"
               "  // something\n"
               "}\n"
               "__finally {\n"
               "  // something\n"
               "}",
               Style);
  verifyFormat("@try {\n"
               "  // something\n"
               "}\n"
               "@finally {\n"
               "  // something\n"
               "}",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  verifyFormat("try\n"
               "{\n"
               "  // something\n"
               "}\n"
               "catch (...)\n"
               "{\n"
               "  // something\n"
               "}",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Whitesmiths;
  verifyFormat("try\n"
               "  {\n"
               "  // something white\n"
               "  }\n"
               "catch (...)\n"
               "  {\n"
               "  // something white\n"
               "  }",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_GNU;
  verifyFormat("try\n"
               "  {\n"
               "    // something\n"
               "  }\n"
               "catch (...)\n"
               "  {\n"
               "    // something\n"
               "  }",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.BeforeCatch = true;
  verifyFormat("try {\n"
               "  // something\n"
               "}\n"
               "catch (...) {\n"
               "  // something\n"
               "}",
               Style);
}

TEST_F(FormatTest, StaticInitializers) {
  verifyFormat("static SomeClass SC = {1, 'a'};");

  verifyFormat("static SomeClass WithALoooooooooooooooooooongName = {\n"
               "    100000000, "
               "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"};");

  // Here, everything other than the "}" would fit on a line.
  verifyFormat("static int LooooooooooooooooooooooooongVariable[1] = {\n"
               "    10000000000000000000000000};");
  verifyFormat("S s = {a,\n"
               "\n"
               "       b};",
               "S s = {\n"
               "  a,\n"
               "\n"
               "  b\n"
               "};");

  // FIXME: This would fit into the column limit if we'd fit "{ {" on the first
  // line. However, the formatting looks a bit off and this probably doesn't
  // happen often in practice.
  verifyFormat("static int Variable[1] = {\n"
               "    {1000000000000000000000000000000000000}};",
               getLLVMStyleWithColumns(40));
}

TEST_F(FormatTest, DesignatedInitializers) {
  verifyFormat("const struct A a = {.a = 1, .b = 2};");
  verifyFormat("const struct A a = {.aaaaaaaaaa = 1,\n"
               "                    .bbbbbbbbbb = 2,\n"
               "                    .cccccccccc = 3,\n"
               "                    .dddddddddd = 4,\n"
               "                    .eeeeeeeeee = 5};");
  verifyFormat("const struct Aaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaa = {\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaa = 1,\n"
               "    .bbbbbbbbbbbbbbbbbbbbbbbbbbb = 2,\n"
               "    .ccccccccccccccccccccccccccc = 3,\n"
               "    .ddddddddddddddddddddddddddd = 4,\n"
               "    .eeeeeeeeeeeeeeeeeeeeeeeeeee = 5};");

  verifyGoogleFormat("const struct A a = {.a = 1, .b = 2};");

  verifyFormat("const struct A a = {[0] = 1, [1] = 2};");
  verifyFormat("const struct A a = {[1] = aaaaaaaaaa,\n"
               "                    [2] = bbbbbbbbbb,\n"
               "                    [3] = cccccccccc,\n"
               "                    [4] = dddddddddd,\n"
               "                    [5] = eeeeeeeeee};");
  verifyFormat("const struct Aaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaa = {\n"
               "    [1] = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    [2] = bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "    [3] = cccccccccccccccccccccccccccccccccccccc,\n"
               "    [4] = dddddddddddddddddddddddddddddddddddddd,\n"
               "    [5] = eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee};");
}

TEST_F(FormatTest, BracedInitializerIndentWidth) {
  auto Style = getLLVMStyleWithColumns(60);
  Style.BinPackArguments = true;
  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  Style.BracedInitializerIndentWidth = 6;

  // Non-initializing braces are unaffected by BracedInitializerIndentWidth.
  verifyFormat("enum class {\n"
               "  One,\n"
               "  Two,\n"
               "};",
               Style);
  verifyFormat("class Foo {\n"
               "  Foo() {}\n"
               "  void bar();\n"
               "};",
               Style);
  verifyFormat("void foo() {\n"
               "  auto bar = baz;\n"
               "  return baz;\n"
               "};",
               Style);
  verifyFormat("auto foo = [&] {\n"
               "  auto bar = baz;\n"
               "  return baz;\n"
               "};",
               Style);
  verifyFormat("{\n"
               "  auto bar = baz;\n"
               "  return baz;\n"
               "};",
               Style);
  // Non-brace initialization is unaffected by BracedInitializerIndentWidth.
  verifyFormat("SomeClass clazz(\n"
               "    \"xxxxxxxxxxxxxxxxxx\", \"yyyyyyyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzzzzzzz\");",
               Style);

  // The following types of initialization are all affected by
  // BracedInitializerIndentWidth. Aggregate initialization.
  verifyFormat("int LooooooooooooooooooooooooongVariable[2] = {\n"
               "      10000000, 20000000};",
               Style);
  verifyFormat("SomeStruct s{\n"
               "      \"xxxxxxxxxxxxxxxx\", \"yyyyyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzzzzz\"};",
               Style);
  // Designated initializers.
  verifyFormat("int LooooooooooooooooooooooooongVariable[2] = {\n"
               "      [0] = 10000000, [1] = 20000000};",
               Style);
  verifyFormat("SomeStruct s{\n"
               "      .foo = \"xxxxxxxxxxxxx\",\n"
               "      .bar = \"yyyyyyyyyyyyy\",\n"
               "      .baz = \"zzzzzzzzzzzzz\"};",
               Style);
  // List initialization.
  verifyFormat("SomeStruct s{\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  verifyFormat("SomeStruct{\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  verifyFormat("new SomeStruct{\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Member initializer.
  verifyFormat("class SomeClass {\n"
               "  SomeStruct s{\n"
               "        \"xxxxxxxxxxxxx\",\n"
               "        \"yyyyyyyyyyyyy\",\n"
               "        \"zzzzzzzzzzzzz\",\n"
               "  };\n"
               "};",
               Style);
  // Constructor member initializer.
  verifyFormat("SomeClass::SomeClass : strct{\n"
               "                             \"xxxxxxxxxxxxx\",\n"
               "                             \"yyyyyyyyyyyyy\",\n"
               "                             \"zzzzzzzzzzzzz\",\n"
               "                       } {}",
               Style);
  // Copy initialization.
  verifyFormat("SomeStruct s = SomeStruct{\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Copy list initialization.
  verifyFormat("SomeStruct s = {\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Assignment operand initialization.
  verifyFormat("s = {\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Returned object initialization.
  verifyFormat("return {\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Initializer list.
  verifyFormat("auto initializerList = {\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Function parameter initialization.
  verifyFormat("func({\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "});",
               Style);
  // Nested init lists.
  verifyFormat("SomeStruct s = {\n"
               "      {{init1, init2, init3, init4, init5},\n"
               "       {init1, init2, init3, init4, init5}}};",
               Style);
  verifyFormat("SomeStruct s = {\n"
               "      {{\n"
               "             .init1 = 1,\n"
               "             .init2 = 2,\n"
               "             .init3 = 3,\n"
               "             .init4 = 4,\n"
               "             .init5 = 5,\n"
               "       },\n"
               "       {init1, init2, init3, init4, init5}}};",
               Style);
  verifyFormat("SomeArrayT a[3] = {\n"
               "      {\n"
               "            foo,\n"
               "            bar,\n"
               "      },\n"
               "      {\n"
               "            foo,\n"
               "            bar,\n"
               "      },\n"
               "      SomeArrayT{},\n"
               "};",
               Style);
  verifyFormat("SomeArrayT a[3] = {\n"
               "      {foo},\n"
               "      {\n"
               "            {\n"
               "                  init1,\n"
               "                  init2,\n"
               "                  init3,\n"
               "            },\n"
               "            {\n"
               "                  init1,\n"
               "                  init2,\n"
               "                  init3,\n"
               "            },\n"
               "      },\n"
               "      {baz},\n"
               "};",
               Style);

  // Aligning after open braces unaffected by BracedInitializerIndentWidth.
  Style.AlignAfterOpenBracket = FormatStyle::BAS_Align;
  verifyFormat("SomeStruct s{\"xxxxxxxxxxxxx\", \"yyyyyyyyyyyyy\",\n"
               "             \"zzzzzzzzzzzzz\"};",
               Style);
}

TEST_F(FormatTest, NestedStaticInitializers) {
  verifyFormat("static A x = {{{}}};");
  verifyFormat("static A x = {{{init1, init2, init3, init4},\n"
               "               {init1, init2, init3, init4}}};",
               getLLVMStyleWithColumns(50));

  verifyFormat("somes Status::global_reps[3] = {\n"
               "    {kGlobalRef, OK_CODE, NULL, NULL, NULL},\n"
               "    {kGlobalRef, CANCELLED_CODE, NULL, NULL, NULL},\n"
               "    {kGlobalRef, UNKNOWN_CODE, NULL, NULL, NULL}};",
               getLLVMStyleWithColumns(60));
  verifyGoogleFormat("SomeType Status::global_reps[3] = {\n"
                     "    {kGlobalRef, OK_CODE, NULL, NULL, NULL},\n"
                     "    {kGlobalRef, CANCELLED_CODE, NULL, NULL, NULL},\n"
                     "    {kGlobalRef, UNKNOWN_CODE, NULL, NULL, NULL}};");
  verifyFormat("CGRect cg_rect = {{rect.fLeft, rect.fTop},\n"
               "                  {rect.fRight - rect.fLeft, rect.fBottom - "
               "rect.fTop}};");

  verifyFormat(
      "SomeArrayOfSomeType a = {\n"
      "    {{1, 2, 3},\n"
      "     {1, 2, 3},\n"
      "     {111111111111111111111111111111, 222222222222222222222222222222,\n"
      "      333333333333333333333333333333},\n"
      "     {1, 2, 3},\n"
      "     {1, 2, 3}}};");
  verifyFormat(
      "SomeArrayOfSomeType a = {\n"
      "    {{1, 2, 3}},\n"
      "    {{1, 2, 3}},\n"
      "    {{111111111111111111111111111111, 222222222222222222222222222222,\n"
      "      333333333333333333333333333333}},\n"
      "    {{1, 2, 3}},\n"
      "    {{1, 2, 3}}};");

  verifyFormat("struct {\n"
               "  unsigned bit;\n"
               "  const char *const name;\n"
               "} kBitsToOs[] = {{kOsMac, \"Mac\"},\n"
               "                 {kOsWin, \"Windows\"},\n"
               "                 {kOsLinux, \"Linux\"},\n"
               "                 {kOsCrOS, \"Chrome OS\"}};");
  verifyFormat("struct {\n"
               "  unsigned bit;\n"
               "  const char *const name;\n"
               "} kBitsToOs[] = {\n"
               "    {kOsMac, \"Mac\"},\n"
               "    {kOsWin, \"Windows\"},\n"
               "    {kOsLinux, \"Linux\"},\n"
               "    {kOsCrOS, \"Chrome OS\"},\n"
               "};");
}

TEST_F(FormatTest, FormatsSmallMacroDefinitionsInSingleLine) {
  verifyFormat("#define ALooooooooooooooooooooooooooooooooooooooongMacro("
               "                      \\\n"
               "    aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)");
}

TEST_F(FormatTest, DoesNotBreakPureVirtualFunctionDefinition) {
  verifyFormat("virtual void write(ELFWriter *writerrr,\n"
               "                   OwningPtr<FileOutputBuffer> &buffer) = 0;");

  // Do break defaulted and deleted functions.
  verifyFormat("virtual void ~Deeeeeeeestructor() =\n"
               "    default;",
               getLLVMStyleWithColumns(40));
  verifyFormat("virtual void ~Deeeeeeeestructor() =\n"
               "    delete;",
               getLLVMStyleWithColumns(40));
}

TEST_F(FormatTest, BreaksStringLiteralsOnlyInDefine) {
  verifyFormat("# 1111 \"/aaaaaaaaa/aaaaaaaaaaaaaaaaaaa/aaaaaaaa.cpp\" 2 3",
               getLLVMStyleWithColumns(40));
  verifyFormat("#line 11111 \"/aaaaaaaaa/aaaaaaaaaaaaaaaaaaa/aaaaaaaa.cpp\"",
               getLLVMStyleWithColumns(40));
  verifyFormat("#define Q                              \\\n"
               "  \"/aaaaaaaaa/aaaaaaaaaaaaaaaaaaa/\"    \\\n"
               "  \"aaaaaaaa.cpp\"",
               "#define Q \"/aaaaaaaaa/aaaaaaaaaaaaaaaaaaa/aaaaaaaa.cpp\"",
               getLLVMStyleWithColumns(40));
}

TEST_F(FormatTest, UnderstandsLinePPDirective) {
  verifyFormat("# 123 \"A string literal\"",
               "   #     123    \"A string literal\"");
}

TEST_F(FormatTest, LayoutUnknownPPDirective) {
  verifyFormat("#;");
  verifyFormat("#\n;\n;\n;");
}

TEST_F(FormatTest, UnescapedEndOfLineEndsPPDirective) {
  verifyFormat("#line 42 \"test\"", "#  \\\n  line  \\\n  42  \\\n  \"test\"");
  verifyFormat("#define A B", "#  \\\n define  \\\n    A  \\\n       B",
               getLLVMStyleWithColumns(12));
}

TEST_F(FormatTest, EndOfFileEndsPPDirective) {
  verifyFormat("#line 42 \"test\"", "#  \\\n  line  \\\n  42  \\\n  \"test\"");
  verifyFormat("#define A B", "#  \\\n define  \\\n    A  \\\n       B");
}

TEST_F(FormatTest, DoesntRemoveUnknownTokens) {
  verifyFormat("#define A \\x20");
  verifyFormat("#define A \\ x20");
  verifyFormat("#define A \\ x20", "#define A \\   x20");
  verifyFormat("#define A ''");
  verifyFormat("#define A ''qqq");
  verifyFormat("#define A `qqq");
  verifyFormat("f(\"aaaa, bbbb, \"\\\"ccccc\\\"\");");
  verifyFormat("const char *c = STRINGIFY(\n"
               "\\na : b);",
               "const char * c = STRINGIFY(\n"
               "\\na : b);");

  verifyFormat("a\r\\");
  verifyFormat("a\v\\");
  verifyFormat("a\f\\");
}

TEST_F(FormatTest, IndentsPPDirectiveWithPPIndentWidth) {
  FormatStyle style = getChromiumStyle(FormatStyle::LK_Cpp);
  style.IndentWidth = 4;
  style.PPIndentWidth = 1;

  style.IndentPPDirectives = FormatStyle::PPDIS_None;
  verifyFormat("#ifdef __linux__\n"
               "void foo() {\n"
               "    int x = 0;\n"
               "}\n"
               "#define FOO\n"
               "#endif\n"
               "void bar() {\n"
               "    int y = 0;\n"
               "}",
               style);

  style.IndentPPDirectives = FormatStyle::PPDIS_AfterHash;
  verifyFormat("#ifdef __linux__\n"
               "void foo() {\n"
               "    int x = 0;\n"
               "}\n"
               "# define FOO foo\n"
               "#endif\n"
               "void bar() {\n"
               "    int y = 0;\n"
               "}",
               style);

  style.IndentPPDirectives = FormatStyle::PPDIS_BeforeHash;
  verifyFormat("#ifdef __linux__\n"
               "void foo() {\n"
               "    int x = 0;\n"
               "}\n"
               " #define FOO foo\n"
               "#endif\n"
               "void bar() {\n"
               "    int y = 0;\n"
               "}",
               style);
  verifyFormat("#if 1\n"
               " // some comments\n"
               " // another\n"
               " #define foo 1\n"
               "// not a define comment\n"
               "void bar() {\n"
               "    // comment\n"
               "    int y = 0;\n"
               "}",
               "#if 1\n"
               "// some comments\n"
               "// another\n"
               "#define foo 1\n"
               "// not a define comment\n"
               "void bar() {\n"
               "  // comment\n"
               "  int y = 0;\n"
               "}",
               style);

  style.IndentPPDirectives = FormatStyle::PPDIS_None;
  verifyFormat("#ifdef foo\n"
               "#define bar() \\\n"
               "    if (A) {  \\\n"
               "        B();  \\\n"
               "    }         \\\n"
               "    C();\n"
               "#endif",
               style);
  verifyFormat("if (emacs) {\n"
               "#ifdef is\n"
               "#define lit           \\\n"
               "    if (af) {         \\\n"
               "        return duh(); \\\n"
               "    }\n"
               "#endif\n"
               "}",
               style);
  verifyFormat("#if abc\n"
               "#ifdef foo\n"
               "#define bar()    \\\n"
               "    if (A) {     \\\n"
               "        if (B) { \\\n"
               "            C(); \\\n"
               "        }        \\\n"
               "    }            \\\n"
               "    D();\n"
               "#endif\n"
               "#endif",
               style);
  verifyFormat("#ifndef foo\n"
               "#define foo\n"
               "if (emacs) {\n"
               "#ifdef is\n"
               "#define lit           \\\n"
               "    if (af) {         \\\n"
               "        return duh(); \\\n"
               "    }\n"
               "#endif\n"
               "}\n"
               "#endif",
               style);
  verifyFormat("#if 1\n"
               "#define X  \\\n"
               "    {      \\\n"
               "        x; \\\n"
               "        x; \\\n"
               "    }\n"
               "#endif",
               style);
  verifyFormat("#define X  \\\n"
               "    {      \\\n"
               "        x; \\\n"
               "        x; \\\n"
               "    }",
               style);

  style.PPIndentWidth = 2;
  verifyFormat("#ifdef foo\n"
               "#define bar() \\\n"
               "    if (A) {  \\\n"
               "        B();  \\\n"
               "    }         \\\n"
               "    C();\n"
               "#endif",
               style);
  style.IndentWidth = 8;
  verifyFormat("#ifdef foo\n"
               "#define bar()        \\\n"
               "        if (A) {     \\\n"
               "                B(); \\\n"
               "        }            \\\n"
               "        C();\n"
               "#endif",
               style);

  style.IndentWidth = 1;
  style.PPIndentWidth = 4;
  verifyFormat("#if 1\n"
               "#define X \\\n"
               " {        \\\n"
               "  x;      \\\n"
               "  x;      \\\n"
               " }\n"
               "#endif",
               style);
  verifyFormat("#define X \\\n"
               " {        \\\n"
               "  x;      \\\n"
               "  x;      \\\n"
               " }",
               style);

  style.IndentWidth = 4;
  style.PPIndentWidth = 1;
  style.IndentPPDirectives = FormatStyle::PPDIS_AfterHash;
  verifyFormat("#ifdef foo\n"
               "# define bar() \\\n"
               "     if (A) {  \\\n"
               "         B();  \\\n"
               "     }         \\\n"
               "     C();\n"
               "#endif",
               style);
  verifyFormat("#if abc\n"
               "# ifdef foo\n"
               "#  define bar()    \\\n"
               "      if (A) {     \\\n"
               "          if (B) { \\\n"
               "              C(); \\\n"
               "          }        \\\n"
               "      }            \\\n"
               "      D();\n"
               "# endif\n"
               "#endif",
               style);
  verifyFormat("#ifndef foo\n"
               "#define foo\n"
               "if (emacs) {\n"
               "#ifdef is\n"
               "# define lit           \\\n"
               "     if (af) {         \\\n"
               "         return duh(); \\\n"
               "     }\n"
               "#endif\n"
               "}\n"
               "#endif",
               style);
  verifyFormat("#define X  \\\n"
               "    {      \\\n"
               "        x; \\\n"
               "        x; \\\n"
               "    }",
               style);

  style.PPIndentWidth = 2;
  style.IndentWidth = 8;
  verifyFormat("#ifdef foo\n"
               "#  define bar()        \\\n"
               "          if (A) {     \\\n"
               "                  B(); \\\n"
               "          }            \\\n"
               "          C();\n"
               "#endif",
               style);

  style.PPIndentWidth = 4;
  style.IndentWidth = 1;
  verifyFormat("#define X \\\n"
               " {        \\\n"
               "  x;      \\\n"
               "  x;      \\\n"
               " }",
               style);

  style.IndentWidth = 4;
  style.PPIndentWidth = 1;
  style.IndentPPDirectives = FormatStyle::PPDIS_BeforeHash;
  verifyFormat("if (emacs) {\n"
               "#ifdef is\n"
               " #define lit           \\\n"
               "     if (af) {         \\\n"
               "         return duh(); \\\n"
               "     }\n"
               "#endif\n"
               "}",
               style);
  verifyFormat("#if abc\n"
               " #ifdef foo\n"
               "  #define bar() \\\n"
               "      if (A) {  \\\n"
               "          B();  \\\n"
               "      }         \\\n"
               "      C();\n"
               " #endif\n"
               "#endif",
               style);
  verifyFormat("#if 1\n"
               " #define X  \\\n"
               "     {      \\\n"
               "         x; \\\n"
               "         x; \\\n"
               "     }\n"
               "#endif",
               style);

  style.PPIndentWidth = 2;
  verifyFormat("#ifdef foo\n"
               "  #define bar() \\\n"
               "      if (A) {  \\\n"
               "          B();  \\\n"
               "      }         \\\n"
               "      C();\n"
               "#endif",
               style);

  style.PPIndentWidth = 4;
  style.IndentWidth = 1;
  verifyFormat("#if 1\n"
               "    #define X \\\n"
               "     {        \\\n"
               "      x;      \\\n"
               "      x;      \\\n"
               "     }\n"
               "#endif",
               style);
}

TEST_F(FormatTest, IndentsPPDirectiveInReducedSpace) {
  verifyFormat("#define A(BB)", getLLVMStyleWithColumns(13));
  verifyFormat("#define A( \\\n    BB)", getLLVMStyleWithColumns(12));
  verifyFormat("#define A( \\\n    A, B)", getLLVMStyleWithColumns(12));
  // FIXME: We never break before the macro name.
  verifyFormat("#define AA( \\\n    B)", getLLVMStyleWithColumns(12));

  verifyFormat("#define A A\n#define A A");
  verifyFormat("#define A(X) A\n#define A A");

  verifyFormat("#define Something Other", getLLVMStyleWithColumns(23));
  verifyFormat("#define Something    \\\n  Other", getLLVMStyleWithColumns(22));
}

TEST_F(FormatTest, HandlePreprocessorDirectiveContext) {
  verifyFormat("// somecomment\n"
               "#include \"a.h\"\n"
               "#define A(  \\\n"
               "    A, B)\n"
               "#include \"b.h\"\n"
               "// somecomment",
               "  // somecomment\n"
               "  #include \"a.h\"\n"
               "#define A(A,\\\n"
               "    B)\n"
               "    #include \"b.h\"\n"
               " // somecomment",
               getLLVMStyleWithColumns(13));
}

TEST_F(FormatTest, LayoutSingleHash) { verifyFormat("#\na;"); }

TEST_F(FormatTest, LayoutCodeInMacroDefinitions) {
  verifyFormat("#define A    \\\n"
               "  c;         \\\n"
               "  e;\n"
               "f;",
               "#define A c; e;\n"
               "f;",
               getLLVMStyleWithColumns(14));
}

TEST_F(FormatTest, LayoutRemainingTokens) { verifyFormat("{}"); }

TEST_F(FormatTest, MacroDefinitionInsideStatement) {
  verifyFormat("int x,\n"
               "#define A\n"
               "    y;",
               "int x,\n#define A\ny;");
}

TEST_F(FormatTest, HashInMacroDefinition) {
  verifyFormat("#define A(c) L#c");
  verifyFormat("#define A(c) u#c");
  verifyFormat("#define A(c) U#c");
  verifyFormat("#define A(c) u8#c");
  verifyFormat("#define A(c) LR#c");
  verifyFormat("#define A(c) uR#c");
  verifyFormat("#define A(c) UR#c");
  verifyFormat("#define A(c) u8R#c");
  verifyFormat("#define A \\\n  b #c;", getLLVMStyleWithColumns(11));
  verifyFormat("#define A  \\\n"
               "  {        \\\n"
               "    f(#c); \\\n"
               "  }",
               getLLVMStyleWithColumns(11));

  verifyFormat("#define A(X)         \\\n"
               "  void function##X()",
               getLLVMStyleWithColumns(22));

  verifyFormat("#define A(a, b, c)   \\\n"
               "  void a##b##c()",
               getLLVMStyleWithColumns(22));

  verifyFormat("#define A void # ## #", getLLVMStyleWithColumns(22));
}

TEST_F(FormatTest, RespectWhitespaceInMacroDefinitions) {
  verifyFormat("#define A (x)");
  verifyFormat("#define A(x)");

  FormatStyle Style = getLLVMStyle();
  Style.SpaceBeforeParens = FormatStyle::SBPO_Never;
  verifyFormat("#define true ((foo)1)", Style);
  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;
  verifyFormat("#define false((foo)0)", Style);
}

TEST_F(FormatTest, EmptyLinesInMacroDefinitions) {
  verifyFormat("#define A b;",
               "#define A \\\n"
               "          \\\n"
               "  b;",
               getLLVMStyleWithColumns(25));
  verifyNoChange("#define A \\\n"
                 "          \\\n"
                 "  a;      \\\n"
                 "  b;",
                 getLLVMStyleWithColumns(11));
  verifyNoChange("#define A \\\n"
                 "  a;      \\\n"
                 "          \\\n"
                 "  b;",
                 getLLVMStyleWithColumns(11));
}

TEST_F(FormatTest, MacroDefinitionsWithIncompleteCode) {
  verifyIncompleteFormat("#define A :");
  verifyFormat("#define SOMECASES  \\\n"
               "  case 1:          \\\n"
               "  case 2",
               getLLVMStyleWithColumns(20));
  verifyFormat("#define MACRO(a) \\\n"
               "  if (a)         \\\n"
               "    f();         \\\n"
               "  else           \\\n"
               "    g()",
               getLLVMStyleWithColumns(18));
  verifyFormat("#define A template <typename T>");
  verifyIncompleteFormat("#define STR(x) #x\n"
                         "f(STR(this_is_a_string_literal{));");
  verifyFormat("#pragma omp threadprivate( \\\n"
               "        y)), // expected-warning",
               getLLVMStyleWithColumns(28));
  verifyFormat("#d, = };");
  verifyFormat("#if \"a");
  verifyIncompleteFormat("({\n"
                         "#define b     \\\n"
                         "  }           \\\n"
                         "  a\n"
                         "a",
                         getLLVMStyleWithColumns(15));
  verifyFormat("#define A     \\\n"
               "  {           \\\n"
               "    {\n"
               "#define B     \\\n"
               "  }           \\\n"
               "  }",
               getLLVMStyleWithColumns(15));
  verifyNoCrash("#if a\na(\n#else\n#endif\n{a");
  verifyNoCrash("a={0,1\n#if a\n#else\n;\n#endif\n}");
  verifyNoCrash("#if a\na(\n#else\n#endif\n) a {a,b,c,d,f,g};");
  verifyNoCrash("#ifdef A\n a(\n #else\n #endif\n) = []() {      \n)}");
  verifyNoCrash("#else\n"
                "#else\n"
                "#endif\n"
                "#endif");
  verifyNoCrash("#else\n"
                "#if X\n"
                "#endif\n"
                "#endif");
  verifyNoCrash("#else\n"
                "#endif\n"
                "#if X\n"
                "#endif");
  verifyNoCrash("#if X\n"
                "#else\n"
                "#else\n"
                "#endif\n"
                "#endif");
  verifyNoCrash("#if X\n"
                "#elif Y\n"
                "#elif Y\n"
                "#endif\n"
                "#endif");
  verifyNoCrash("#endif\n"
                "#endif");
  verifyNoCrash("#endif\n"
                "#else");
  verifyNoCrash("#endif\n"
                "#elif Y");
}

TEST_F(FormatTest, MacrosWithoutTrailingSemicolon) {
  verifyFormat("SOME_TYPE_NAME abc;"); // Gated on the newline.
  verifyFormat("class A : public QObject {\n"
               "  Q_OBJECT\n"
               "\n"
               "  A() {}\n"
               "};",
               "class A  :  public QObject {\n"
               "     Q_OBJECT\n"
               "\n"
               "  A() {\n}\n"
               "}  ;");
  verifyFormat("MACRO\n"
               "/*static*/ int i;",
               "MACRO\n"
               " /*static*/ int   i;");
  verifyFormat("SOME_MACRO\n"
               "namespace {\n"
               "void f();\n"
               "} // namespace",
               "SOME_MACRO\n"
               "  namespace    {\n"
               "void   f(  );\n"
               "} // namespace");
  // Only if the identifier contains at least 5 characters.
  verifyFormat("HTTP f();", "HTTP\nf();");
  verifyNoChange("MACRO\nf();");
  // Only if everything is upper case.
  verifyFormat("class A : public QObject {\n"
               "  Q_Object A() {}\n"
               "};",
               "class A  :  public QObject {\n"
               "     Q_Object\n"
               "  A() {\n}\n"
               "}  ;");

  // Only if the next line can actually start an unwrapped line.
  verifyFormat("SOME_WEIRD_LOG_MACRO << SomeThing;", "SOME_WEIRD_LOG_MACRO\n"
                                                     "<< SomeThing;");

  verifyFormat("VISIT_GL_CALL(GenBuffers, void, (GLsizei n, GLuint* buffers), "
               "(n, buffers))",
               getChromiumStyle(FormatStyle::LK_Cpp));

  // See PR41483
  verifyNoChange("/**/ FOO(a)\n"
                 "FOO(b)");
}

TEST_F(FormatTest, MacroCallsWithoutTrailingSemicolon) {
  verifyFormat("INITIALIZE_PASS_BEGIN(ScopDetection, \"polly-detect\")\n"
               "INITIALIZE_AG_DEPENDENCY(AliasAnalysis)\n"
               "INITIALIZE_PASS_DEPENDENCY(DominatorTree)\n"
               "class X {};\n"
               "INITIALIZE_PASS_END(ScopDetection, \"polly-detect\")\n"
               "int *createScopDetectionPass() { return 0; }",
               "  INITIALIZE_PASS_BEGIN(ScopDetection, \"polly-detect\")\n"
               "  INITIALIZE_AG_DEPENDENCY(AliasAnalysis)\n"
               "  INITIALIZE_PASS_DEPENDENCY(DominatorTree)\n"
               "  class X {};\n"
               "  INITIALIZE_PASS_END(ScopDetection, \"polly-detect\")\n"
               "  int *createScopDetectionPass() { return 0; }");
  // FIXME: We could probably treat IPC_BEGIN_MESSAGE_MAP/IPC_END_MESSAGE_MAP as
  // braces, so that inner block is indented one level more.
  verifyFormat("int q() {\n"
               "  IPC_BEGIN_MESSAGE_MAP(WebKitTestController, message)\n"
               "  IPC_MESSAGE_HANDLER(xxx, qqq)\n"
               "  IPC_END_MESSAGE_MAP()\n"
               "}",
               "int q() {\n"
               "  IPC_BEGIN_MESSAGE_MAP(WebKitTestController, message)\n"
               "    IPC_MESSAGE_HANDLER(xxx, qqq)\n"
               "  IPC_END_MESSAGE_MAP()\n"
               "}");

  // Same inside macros.
  verifyFormat("#define LIST(L) \\\n"
               "  L(A)          \\\n"
               "  L(B)          \\\n"
               "  L(C)",
               "#define LIST(L) \\\n"
               "  L(A) \\\n"
               "  L(B) \\\n"
               "  L(C)",
               getGoogleStyle());

  // These must not be recognized as macros.
  verifyFormat("int q() {\n"
               "  f(x);\n"
               "  f(x) {}\n"
               "  f(x)->g();\n"
               "  f(x)->*g();\n"
               "  f(x).g();\n"
               "  f(x) = x;\n"
               "  f(x) += x;\n"
               "  f(x) -= x;\n"
               "  f(x) *= x;\n"
               "  f(x) /= x;\n"
               "  f(x) %= x;\n"
               "  f(x) &= x;\n"
               "  f(x) |= x;\n"
               "  f(x) ^= x;\n"
               "  f(x) >>= x;\n"
               "  f(x) <<= x;\n"
               "  f(x)[y].z();\n"
               "  LOG(INFO) << x;\n"
               "  ifstream(x) >> x;\n"
               "}",
               "int q() {\n"
               "  f(x)\n;\n"
               "  f(x)\n {}\n"
               "  f(x)\n->g();\n"
               "  f(x)\n->*g();\n"
               "  f(x)\n.g();\n"
               "  f(x)\n = x;\n"
               "  f(x)\n += x;\n"
               "  f(x)\n -= x;\n"
               "  f(x)\n *= x;\n"
               "  f(x)\n /= x;\n"
               "  f(x)\n %= x;\n"
               "  f(x)\n &= x;\n"
               "  f(x)\n |= x;\n"
               "  f(x)\n ^= x;\n"
               "  f(x)\n >>= x;\n"
               "  f(x)\n <<= x;\n"
               "  f(x)\n[y].z();\n"
               "  LOG(INFO)\n << x;\n"
               "  ifstream(x)\n >> x;\n"
               "}");
  verifyFormat("int q() {\n"
               "  F(x)\n"
               "  if (1) {\n"
               "  }\n"
               "  F(x)\n"
               "  while (1) {\n"
               "  }\n"
               "  F(x)\n"
               "  G(x);\n"
               "  F(x)\n"
               "  try {\n"
               "    Q();\n"
               "  } catch (...) {\n"
               "  }\n"
               "}",
               "int q() {\n"
               "F(x)\n"
               "if (1) {}\n"
               "F(x)\n"
               "while (1) {}\n"
               "F(x)\n"
               "G(x);\n"
               "F(x)\n"
               "try { Q(); } catch (...) {}\n"
               "}");
  verifyFormat("class A {\n"
               "  A() : t(0) {}\n"
               "  A(int i) noexcept() : {}\n"
               "  A(X x)\n" // FIXME: function-level try blocks are broken.
               "  try : t(0) {\n"
               "  } catch (...) {\n"
               "  }\n"
               "};",
               "class A {\n"
               "  A()\n : t(0) {}\n"
               "  A(int i)\n noexcept() : {}\n"
               "  A(X x)\n"
               "  try : t(0) {} catch (...) {}\n"
               "};");
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  Style.BraceWrapping.AfterFunction = true;
  verifyFormat("void f()\n"
               "try\n"
               "{\n"
               "}",
               "void f() try {\n"
               "}",
               Style);
  verifyFormat("class SomeClass {\n"
               "public:\n"
               "  SomeClass() EXCLUSIVE_LOCK_FUNCTION(mu_);\n"
               "};",
               "class SomeClass {\n"
               "public:\n"
               "  SomeClass()\n"
               "  EXCLUSIVE_LOCK_FUNCTION(mu_);\n"
               "};");
  verifyFormat("class SomeClass {\n"
               "public:\n"
               "  SomeClass()\n"
               "      EXCLUSIVE_LOCK_FUNCTION(mu_);\n"
               "};",
               "class SomeClass {\n"
               "public:\n"
               "  SomeClass()\n"
               "  EXCLUSIVE_LOCK_FUNCTION(mu_);\n"
               "};",
               getLLVMStyleWithColumns(40));

  verifyFormat("MACRO(>)");

  // Some macros contain an implicit semicolon.
  Style = getLLVMStyle();
  Style.StatementMacros.push_back("FOO");
  verifyFormat("FOO(a) int b = 0;");
  verifyFormat("FOO(a)\n"
               "int b = 0;",
               Style);
  verifyFormat("FOO(a);\n"
               "int b = 0;",
               Style);
  verifyFormat("FOO(argc, argv, \"4.0.2\")\n"
               "int b = 0;",
               Style);
  verifyFormat("FOO()\n"
               "int b = 0;",
               Style);
  verifyFormat("FOO\n"
               "int b = 0;",
               Style);
  verifyFormat("void f() {\n"
               "  FOO(a)\n"
               "  return a;\n"
               "}",
               Style);
  verifyFormat("FOO(a)\n"
               "FOO(b)",
               Style);
  verifyFormat("int a = 0;\n"
               "FOO(b)\n"
               "int c = 0;",
               Style);
  verifyFormat("int a = 0;\n"
               "int x = FOO(a)\n"
               "int b = 0;",
               Style);
  verifyFormat("void foo(int a) { FOO(a) }\n"
               "uint32_t bar() {}",
               Style);
}

TEST_F(FormatTest, FormatsMacrosWithZeroColumnWidth) {
  FormatStyle ZeroColumn = getLLVMStyleWithColumns(0);

  verifyFormat("#define A LOOOOOOOOOOOOOOOOOOONG() LOOOOOOOOOOOOOOOOOOONG()",
               ZeroColumn);
}

TEST_F(FormatTest, LayoutMacroDefinitionsStatementsSpanningBlocks) {
  verifyFormat("#define A \\\n"
               "  f({     \\\n"
               "    g();  \\\n"
               "  });",
               getLLVMStyleWithColumns(11));
}

TEST_F(FormatTest, IndentPreprocessorDirectives) {
  FormatStyle Style = getLLVMStyleWithColumns(40);
  Style.IndentPPDirectives = FormatStyle::PPDIS_None;
  verifyFormat("#ifdef _WIN32\n"
               "#define A 0\n"
               "#ifdef VAR2\n"
               "#define B 1\n"
               "#include <someheader.h>\n"
               "#define MACRO                          \\\n"
               "  some_very_long_func_aaaaaaaaaa();\n"
               "#endif\n"
               "#else\n"
               "#define A 1\n"
               "#endif",
               Style);
  Style.IndentPPDirectives = FormatStyle::PPDIS_AfterHash;
  verifyFormat("#if 1\n"
               "#  define __STR(x) #x\n"
               "#endif",
               Style);
  verifyFormat("#ifdef _WIN32\n"
               "#  define A 0\n"
               "#  ifdef VAR2\n"
               "#    define B 1\n"
               "#    include <someheader.h>\n"
               "#    define MACRO                      \\\n"
               "      some_very_long_func_aaaaaaaaaa();\n"
               "#  endif\n"
               "#else\n"
               "#  define A 1\n"
               "#endif",
               Style);
  verifyFormat("#if A\n"
               "#  define MACRO                        \\\n"
               "    void a(int x) {                    \\\n"
               "      b();                             \\\n"
               "      c();                             \\\n"
               "      d();                             \\\n"
               "      e();                             \\\n"
               "      f();                             \\\n"
               "    }\n"
               "#endif",
               Style);
  // Comments before include guard.
  verifyFormat("// file comment\n"
               "// file comment\n"
               "#ifndef HEADER_H\n"
               "#define HEADER_H\n"
               "code();\n"
               "#endif",
               Style);
  // Test with include guards.
  verifyFormat("#ifndef HEADER_H\n"
               "#define HEADER_H\n"
               "code();\n"
               "#endif",
               Style);
  // Include guards must have a #define with the same variable immediately
  // after #ifndef.
  verifyFormat("#ifndef NOT_GUARD\n"
               "#  define FOO\n"
               "code();\n"
               "#endif",
               Style);

  // Include guards must cover the entire file.
  verifyFormat("code();\n"
               "code();\n"
               "#ifndef NOT_GUARD\n"
               "#  define NOT_GUARD\n"
               "code();\n"
               "#endif",
               Style);
  verifyFormat("#ifndef NOT_GUARD\n"
               "#  define NOT_GUARD\n"
               "code();\n"
               "#endif\n"
               "code();",
               Style);
  // Test with trailing blank lines.
  verifyFormat("#ifndef HEADER_H\n"
               "#define HEADER_H\n"
               "code();\n"
               "#endif",
               Style);
  // Include guards don't have #else.
  verifyFormat("#ifndef NOT_GUARD\n"
               "#  define NOT_GUARD\n"
               "code();\n"
               "#else\n"
               "#endif",
               Style);
  verifyFormat("#ifndef NOT_GUARD\n"
               "#  define NOT_GUARD\n"
               "code();\n"
               "#elif FOO\n"
               "#endif",
               Style);
  // Non-identifier #define after potential include guard.
  verifyFormat("#ifndef FOO\n"
               "#  define 1\n"
               "#endif",
               Style);
  // #if closes past last non-preprocessor line.
  verifyFormat("#ifndef FOO\n"
               "#define FOO\n"
               "#if 1\n"
               "int i;\n"
               "#  define A 0\n"
               "#endif\n"
               "#endif",
               Style);
  // Don't crash if there is an #elif directive without a condition.
  verifyFormat("#if 1\n"
               "int x;\n"
               "#elif\n"
               "int y;\n"
               "#else\n"
               "int z;\n"
               "#endif",
               Style);
  // FIXME: This doesn't handle the case where there's code between the
  // #ifndef and #define but all other conditions hold. This is because when
  // the #define line is parsed, UnwrappedLineParser::Lines doesn't hold the
  // previous code line yet, so we can't detect it.
  verifyFormat("#ifndef NOT_GUARD\n"
               "code();\n"
               "#define NOT_GUARD\n"
               "code();\n"
               "#endif",
               "#ifndef NOT_GUARD\n"
               "code();\n"
               "#  define NOT_GUARD\n"
               "code();\n"
               "#endif",
               Style);
  // FIXME: This doesn't handle cases where legitimate preprocessor lines may
  // be outside an include guard. Examples are #pragma once and
  // #pragma GCC diagnostic, or anything else that does not change the meaning
  // of the file if it's included multiple times.
  verifyFormat("#ifdef WIN32\n"
               "#  pragma once\n"
               "#endif\n"
               "#ifndef HEADER_H\n"
               "#  define HEADER_H\n"
               "code();\n"
               "#endif",
               "#ifdef WIN32\n"
               "#  pragma once\n"
               "#endif\n"
               "#ifndef HEADER_H\n"
               "#define HEADER_H\n"
               "code();\n"
               "#endif",
               Style);
  // FIXME: This does not detect when there is a single non-preprocessor line
  // in front of an include-guard-like structure where other conditions hold
  // because ScopedLineState hides the line.
  verifyFormat("code();\n"
               "#ifndef HEADER_H\n"
               "#define HEADER_H\n"
               "code();\n"
               "#endif",
               "code();\n"
               "#ifndef HEADER_H\n"
               "#  define HEADER_H\n"
               "code();\n"
               "#endif",
               Style);
  // Keep comments aligned with #, otherwise indent comments normally. These
  // tests cannot use verifyFormat because messUp manipulates leading
  // whitespace.
  {
    const char *Expected = ""
                           "void f() {\n"
                           "#if 1\n"
                           "// Preprocessor aligned.\n"
                           "#  define A 0\n"
                           "  // Code. Separated by blank line.\n"
                           "\n"
                           "#  define B 0\n"
                           "  // Code. Not aligned with #\n"
                           "#  define C 0\n"
                           "#endif";
    const char *ToFormat = ""
                           "void f() {\n"
                           "#if 1\n"
                           "// Preprocessor aligned.\n"
                           "#  define A 0\n"
                           "// Code. Separated by blank line.\n"
                           "\n"
                           "#  define B 0\n"
                           "   // Code. Not aligned with #\n"
                           "#  define C 0\n"
                           "#endif";
    verifyFormat(Expected, ToFormat, Style);
    verifyNoChange(Expected, Style);
  }
  // Keep block quotes aligned.
  {
    const char *Expected = ""
                           "void f() {\n"
                           "#if 1\n"
                           "/* Preprocessor aligned. */\n"
                           "#  define A 0\n"
                           "  /* Code. Separated by blank line. */\n"
                           "\n"
                           "#  define B 0\n"
                           "  /* Code. Not aligned with # */\n"
                           "#  define C 0\n"
                           "#endif";
    const char *ToFormat = ""
                           "void f() {\n"
                           "#if 1\n"
                           "/* Preprocessor aligned. */\n"
                           "#  define A 0\n"
                           "/* Code. Separated by blank line. */\n"
                           "\n"
                           "#  define B 0\n"
                           "   /* Code. Not aligned with # */\n"
                           "#  define C 0\n"
                           "#endif";
    verifyFormat(Expected, ToFormat, Style);
    verifyNoChange(Expected, Style);
  }
  // Keep comments aligned with un-indented directives.
  {
    const char *Expected = ""
                           "void f() {\n"
                           "// Preprocessor aligned.\n"
                           "#define A 0\n"
                           "  // Code. Separated by blank line.\n"
                           "\n"
                           "#define B 0\n"
                           "  // Code. Not aligned with #\n"
                           "#define C 0\n";
    const char *ToFormat = ""
                           "void f() {\n"
                           "// Preprocessor aligned.\n"
                           "#define A 0\n"
                           "// Code. Separated by blank line.\n"
                           "\n"
                           "#define B 0\n"
                           "   // Code. Not aligned with #\n"
                           "#define C 0\n";
    verifyFormat(Expected, ToFormat, Style);
    verifyNoChange(Expected, Style);
  }
  // Test AfterHash with tabs.
  {
    FormatStyle Tabbed = Style;
    Tabbed.UseTab = FormatStyle::UT_Always;
    Tabbed.IndentWidth = 8;
    Tabbed.TabWidth = 8;
    verifyFormat("#ifdef _WIN32\n"
                 "#\tdefine A 0\n"
                 "#\tifdef VAR2\n"
                 "#\t\tdefine B 1\n"
                 "#\t\tinclude <someheader.h>\n"
                 "#\t\tdefine MACRO          \\\n"
                 "\t\t\tsome_very_long_func_aaaaaaaaaa();\n"
                 "#\tendif\n"
                 "#else\n"
                 "#\tdefine A 1\n"
                 "#endif",
                 Tabbed);
  }

  // Regression test: Multiline-macro inside include guards.
  verifyFormat("#ifndef HEADER_H\n"
               "#define HEADER_H\n"
               "#define A()        \\\n"
               "  int i;           \\\n"
               "  int j;\n"
               "#endif // HEADER_H",
               getLLVMStyleWithColumns(20));

  Style.IndentPPDirectives = FormatStyle::PPDIS_BeforeHash;
  // Basic before hash indent tests
  verifyFormat("#ifdef _WIN32\n"
               "  #define A 0\n"
               "  #ifdef VAR2\n"
               "    #define B 1\n"
               "    #include <someheader.h>\n"
               "    #define MACRO                      \\\n"
               "      some_very_long_func_aaaaaaaaaa();\n"
               "  #endif\n"
               "#else\n"
               "  #define A 1\n"
               "#endif",
               Style);
  verifyFormat("#if A\n"
               "  #define MACRO                        \\\n"
               "    void a(int x) {                    \\\n"
               "      b();                             \\\n"
               "      c();                             \\\n"
               "      d();                             \\\n"
               "      e();                             \\\n"
               "      f();                             \\\n"
               "    }\n"
               "#endif",
               Style);
  // Keep comments aligned with indented directives. These
  // tests cannot use verifyFormat because messUp manipulates leading
  // whitespace.
  {
    const char *Expected = "void f() {\n"
                           "// Aligned to preprocessor.\n"
                           "#if 1\n"
                           "  // Aligned to code.\n"
                           "  int a;\n"
                           "  #if 1\n"
                           "    // Aligned to preprocessor.\n"
                           "    #define A 0\n"
                           "  // Aligned to code.\n"
                           "  int b;\n"
                           "  #endif\n"
                           "#endif\n"
                           "}";
    const char *ToFormat = "void f() {\n"
                           "// Aligned to preprocessor.\n"
                           "#if 1\n"
                           "// Aligned to code.\n"
                           "int a;\n"
                           "#if 1\n"
                           "// Aligned to preprocessor.\n"
                           "#define A 0\n"
                           "// Aligned to code.\n"
                           "int b;\n"
                           "#endif\n"
                           "#endif\n"
                           "}";
    verifyFormat(Expected, ToFormat, Style);
    verifyNoChange(Expected, Style);
  }
  {
    const char *Expected = "void f() {\n"
                           "/* Aligned to preprocessor. */\n"
                           "#if 1\n"
                           "  /* Aligned to code. */\n"
                           "  int a;\n"
                           "  #if 1\n"
                           "    /* Aligned to preprocessor. */\n"
                           "    #define A 0\n"
                           "  /* Aligned to code. */\n"
                           "  int b;\n"
                           "  #endif\n"
                           "#endif\n"
                           "}";
    const char *ToFormat = "void f() {\n"
                           "/* Aligned to preprocessor. */\n"
                           "#if 1\n"
                           "/* Aligned to code. */\n"
                           "int a;\n"
                           "#if 1\n"
                           "/* Aligned to preprocessor. */\n"
                           "#define A 0\n"
                           "/* Aligned to code. */\n"
                           "int b;\n"
                           "#endif\n"
                           "#endif\n"
                           "}";
    verifyFormat(Expected, ToFormat, Style);
    verifyNoChange(Expected, Style);
  }

  // Test single comment before preprocessor
  verifyFormat("// Comment\n"
               "\n"
               "#if 1\n"
               "#endif",
               Style);
}

TEST_F(FormatTest, FormatAlignInsidePreprocessorElseBlock) {
  FormatStyle Style = getLLVMStyle();
  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.AlignConsecutiveDeclarations.Enabled = true;

  // Test with just #if blocks.
  verifyFormat("void f1() {\n"
               "#if 1\n"
               "  int foo    = 1;\n"
               "  int foobar = 2;\n"
               "#endif\n"
               "}\n"
               "#if 1\n"
               "int baz = 3;\n"
               "#endif\n"
               "void f2() {\n"
               "#if 1\n"
               "  char *foobarbaz = \"foobarbaz\";\n"
               "  int   quux      = 4;\n"
               "}",
               Style);

  // Test with just #else blocks.
  verifyFormat("void f1() {\n"
               "#if 1\n"
               "#else\n"
               "  int foo    = 1;\n"
               "  int foobar = 2;\n"
               "#endif\n"
               "}\n"
               "#if 1\n"
               "#else\n"
               "int baz = 3;\n"
               "#endif\n"
               "void f2() {\n"
               "#if 1\n"
               "#else\n"
               "  char *foobarbaz = \"foobarbaz\";\n"
               "  int   quux      = 4;\n"
               "}",
               Style);

  // Test with a mix of #if and #else blocks.
  verifyFormat("void f1() {\n"
               "#if 1\n"
               "#else\n"
               "  int foo    = 1;\n"
               "  int foobar = 2;\n"
               "#endif\n"
               "}\n"
               "#if 1\n"
               "int baz = 3;\n"
               "#endif\n"
               "void f2() {\n"
               "#if 1\n"
               "#else\n"
               "  // prevent alignment with #else in f1\n"
               "  char *foobarbaz = \"foobarbaz\";\n"
               "  int   quux      = 4;\n"
               "}",
               Style);

  // Test with nested #if and #else blocks.
  verifyFormat("void f1() {\n"
               "#if 1\n"
               "#else\n"
               "#if 2\n"
               "#else\n"
               "  int foo    = 1;\n"
               "  int foobar = 2;\n"
               "#endif\n"
               "#endif\n"
               "}\n"
               "#if 1\n"
               "#else\n"
               "#if 2\n"
               "int baz = 3;\n"
               "#endif\n"
               "#endif\n"
               "void f2() {\n"
               "#if 1\n"
               "#if 2\n"
               "#else\n"
               "  // prevent alignment with #else in f1\n"
               "  char *foobarbaz = \"foobarbaz\";\n"
               "  int   quux      = 4;\n"
               "#endif\n"
               "#endif\n"
               "}",
               Style);

  verifyFormat("#if FOO\n"
               "int a = 1;\n"
               "#else\n"
               "int ab = 2;\n"
               "#endif\n"
               "#ifdef BAR\n"
               "int abc = 3;\n"
               "#elifdef BAZ\n"
               "int abcd = 4;\n"
               "#endif",
               Style);

  verifyFormat("void f() {\n"
               "  if (foo) {\n"
               "#if FOO\n"
               "    int a = 1;\n"
               "#else\n"
               "    bool a = true;\n"
               "#endif\n"
               "    int abc = 3;\n"
               "#ifndef BAR\n"
               "    int abcd = 4;\n"
               "#elif BAZ\n"
               "    bool abcd = true;\n"
               "#endif\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("void f() {\n"
               "#if FOO\n"
               "  a = 1;\n"
               "#else\n"
               "  ab = 2;\n"
               "#endif\n"
               "}\n"
               "void g() {\n"
               "#if BAR\n"
               "  abc = 3;\n"
               "#elifndef BAZ\n"
               "  abcd = 4;\n"
               "#endif\n"
               "}",
               Style);
}

TEST_F(FormatTest, FormatHashIfNotAtStartOfLine) {
  verifyFormat("{\n  { a #c; }\n}");
}

TEST_F(FormatTest, FormatUnbalancedStructuralElements) {
  verifyFormat("#define A \\\n  {       \\\n    {\nint i;",
               "#define A { {\nint i;", getLLVMStyleWithColumns(11));
  verifyFormat("#define A \\\n  }       \\\n  }\nint i;",
               "#define A } }\nint i;", getLLVMStyleWithColumns(11));
}

TEST_F(FormatTest, EscapedNewlines) {
  FormatStyle Narrow = getLLVMStyleWithColumns(11);
  verifyFormat("#define A \\\n  int i;  \\\n  int j;",
               "#define A \\\nint i;\\\n  int j;", Narrow);
  verifyFormat("#define A\n\nint i;", "#define A \\\n\n int i;");
  verifyFormat("template <class T> f();", "\\\ntemplate <class T> f();");
  verifyFormat("/* \\  \\  \\\n */", "\\\n/* \\  \\  \\\n */");
  verifyNoChange("<a\n\\\\\n>");

  FormatStyle AlignLeft = getLLVMStyle();
  AlignLeft.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  verifyFormat("#define MACRO(x) \\\n"
               "private:         \\\n"
               "  int x(int a);",
               AlignLeft);

  // CRLF line endings
  verifyFormat("#define A \\\r\n  int i;  \\\r\n  int j;",
               "#define A \\\r\nint i;\\\r\n  int j;", Narrow);
  verifyFormat("#define A\r\n\r\nint i;", "#define A \\\r\n\r\n int i;");
  verifyFormat("template <class T> f();", "\\\ntemplate <class T> f();");
  verifyFormat("/* \\  \\  \\\r\n */", "\\\r\n/* \\  \\  \\\r\n */");
  verifyNoChange("<a\r\n\\\\\r\n>");
  verifyFormat("#define MACRO(x) \\\r\n"
               "private:         \\\r\n"
               "  int x(int a);",
               AlignLeft);

  FormatStyle DontAlign = getLLVMStyle();
  DontAlign.AlignEscapedNewlines = FormatStyle::ENAS_DontAlign;
  DontAlign.MaxEmptyLinesToKeep = 3;
  // FIXME: can't use verifyFormat here because the newline before
  // "public:" is not inserted the first time it's reformatted
  verifyNoChange("#define A \\\n"
                 "  class Foo { \\\n"
                 "    void bar(); \\\n"
                 "\\\n"
                 "\\\n"
                 "\\\n"
                 "  public: \\\n"
                 "    void baz(); \\\n"
                 "  };",
                 DontAlign);
}

TEST_F(FormatTest, CalculateSpaceOnConsecutiveLinesInMacro) {
  verifyFormat("#define A \\\n"
               "  int v(  \\\n"
               "      a); \\\n"
               "  int i;",
               getLLVMStyleWithColumns(11));
}

TEST_F(FormatTest, MixingPreprocessorDirectivesAndNormalCode) {
  verifyFormat("#define ALooooooooooooooooooooooooooooooooooooooongMacro("
               "                      \\\n"
               "    aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)\n"
               "\n"
               "AlooooooooooooooooooooooooooooooooooooooongCaaaaaaaaaal(\n"
               "    aLooooooooooooooooooooooonPaaaaaaaaaaaaaaaaaaaaarmmmm);",
               "  #define   ALooooooooooooooooooooooooooooooooooooooongMacro("
               "\\\n"
               "aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)\n"
               "  \n"
               "   AlooooooooooooooooooooooooooooooooooooooongCaaaaaaaaaal(\n"
               "  aLooooooooooooooooooooooonPaaaaaaaaaaaaaaaaaaaaarmmmm);");
}

TEST_F(FormatTest, LayoutStatementsAroundPreprocessorDirectives) {
  verifyFormat("int\n"
               "#define A\n"
               "    a;",
               "int\n#define A\na;");
  verifyFormat("functionCallTo(\n"
               "    someOtherFunction(\n"
               "        withSomeParameters, whichInSequence,\n"
               "        areLongerThanALine(andAnotherCall,\n"
               "#define A B\n"
               "                           withMoreParamters,\n"
               "                           whichStronglyInfluenceTheLayout),\n"
               "        andMoreParameters),\n"
               "    trailing);",
               getLLVMStyleWithColumns(69));
  verifyFormat("Foo::Foo()\n"
               "#ifdef BAR\n"
               "    : baz(0)\n"
               "#endif\n"
               "{\n"
               "}");
  verifyFormat("void f() {\n"
               "  if (true)\n"
               "#ifdef A\n"
               "    f(42);\n"
               "  x();\n"
               "#else\n"
               "    g();\n"
               "  x();\n"
               "#endif\n"
               "}");
  verifyFormat("void f(param1, param2,\n"
               "       param3,\n"
               "#ifdef A\n"
               "       param4(param5,\n"
               "#ifdef A1\n"
               "              param6,\n"
               "#ifdef A2\n"
               "              param7),\n"
               "#else\n"
               "              param8),\n"
               "       param9,\n"
               "#endif\n"
               "       param10,\n"
               "#endif\n"
               "       param11)\n"
               "#else\n"
               "       param12)\n"
               "#endif\n"
               "{\n"
               "  x();\n"
               "}",
               getLLVMStyleWithColumns(28));
  verifyFormat("#if 1\n"
               "int i;");
  verifyFormat("#if 1\n"
               "#endif\n"
               "#if 1\n"
               "#else\n"
               "#endif");
  verifyFormat("DEBUG({\n"
               "  return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;\n"
               "});\n"
               "#if a\n"
               "#else\n"
               "#endif");

  verifyIncompleteFormat("void f(\n"
                         "#if A\n"
                         ");\n"
                         "#else\n"
                         "#endif");

  // Verify that indentation is correct when there is an `#if 0` with an
  // `#else`.
  verifyFormat("#if 0\n"
               "{\n"
               "#else\n"
               "{\n"
               "#endif\n"
               "  x;\n"
               "}");

  verifyFormat("#if 0\n"
               "#endif\n"
               "#if X\n"
               "int something_fairly_long; // Align here please\n"
               "#endif                     // Should be aligned");

  verifyFormat("#if 0\n"
               "#endif\n"
               "#if X\n"
               "#else  // Align\n"
               ";\n"
               "#endif // Align");

  verifyFormat("void SomeFunction(int param1,\n"
               "                  template <\n"
               "#ifdef A\n"
               "#if 0\n"
               "#endif\n"
               "                      MyType<Some>>\n"
               "#else\n"
               "                      Type1, Type2>\n"
               "#endif\n"
               "                  param2,\n"
               "                  param3) {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, GraciouslyHandleIncorrectPreprocessorConditions) {
  verifyFormat("#endif\n"
               "#if B");
}

TEST_F(FormatTest, FormatsJoinedLinesOnSubsequentRuns) {
  FormatStyle SingleLine = getLLVMStyle();
  SingleLine.AllowShortIfStatementsOnASingleLine = FormatStyle::SIS_WithoutElse;
  verifyFormat("#if 0\n"
               "#elif 1\n"
               "#endif\n"
               "void foo() {\n"
               "  if (test) foo2();\n"
               "}",
               SingleLine);
}

TEST_F(FormatTest, LayoutBlockInsideParens) {
  verifyFormat("functionCall({ int i; });");
  verifyFormat("functionCall({\n"
               "  int i;\n"
               "  int j;\n"
               "});");
  verifyFormat("functionCall(\n"
               "    {\n"
               "      int i;\n"
               "      int j;\n"
               "    },\n"
               "    aaaa, bbbb, cccc);");
  verifyFormat("functionA(functionB({\n"
               "            int i;\n"
               "            int j;\n"
               "          }),\n"
               "          aaaa, bbbb, cccc);");
  verifyFormat("functionCall(\n"
               "    {\n"
               "      int i;\n"
               "      int j;\n"
               "    },\n"
               "    aaaa, bbbb, // comment\n"
               "    cccc);");
  verifyFormat("functionA(functionB({\n"
               "            int i;\n"
               "            int j;\n"
               "          }),\n"
               "          aaaa, bbbb, // comment\n"
               "          cccc);");
  verifyFormat("functionCall(aaaa, bbbb, { int i; });");
  verifyFormat("functionCall(aaaa, bbbb, {\n"
               "  int i;\n"
               "  int j;\n"
               "});");
  verifyFormat(
      "Aaa(\n" // FIXME: There shouldn't be a linebreak here.
      "    {\n"
      "      int i; // break\n"
      "    },\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
      "                                     ccccccccccccccccc));");
  verifyFormat("DEBUG({\n"
               "  if (a)\n"
               "    f();\n"
               "});");
}

TEST_F(FormatTest, LayoutBlockInsideStatement) {
  verifyFormat("SOME_MACRO { int i; }\n"
               "int i;",
               "  SOME_MACRO  {int i;}  int i;");
}

TEST_F(FormatTest, LayoutNestedBlocks) {
  verifyFormat("void AddOsStrings(unsigned bitmask) {\n"
               "  struct s {\n"
               "    int i;\n"
               "  };\n"
               "  s kBitsToOs[] = {{10}};\n"
               "  for (int i = 0; i < 10; ++i)\n"
               "    return;\n"
               "}");
  verifyFormat("call(parameter, {\n"
               "  something();\n"
               "  // Comment using all columns.\n"
               "  somethingelse();\n"
               "});",
               getLLVMStyleWithColumns(40));
  verifyFormat("DEBUG( //\n"
               "    { f(); }, a);");
  verifyFormat("DEBUG( //\n"
               "    {\n"
               "      f(); //\n"
               "    },\n"
               "    a);");

  verifyFormat("call(parameter, {\n"
               "  something();\n"
               "  // Comment too\n"
               "  // looooooooooong.\n"
               "  somethingElse();\n"
               "});",
               "call(parameter, {\n"
               "  something();\n"
               "  // Comment too looooooooooong.\n"
               "  somethingElse();\n"
               "});",
               getLLVMStyleWithColumns(29));
  verifyFormat("DEBUG({ int i; });", "DEBUG({ int   i; });");
  verifyFormat("DEBUG({ // comment\n"
               "  int i;\n"
               "});",
               "DEBUG({ // comment\n"
               "int  i;\n"
               "});");
  verifyFormat("DEBUG({\n"
               "  int i;\n"
               "\n"
               "  // comment\n"
               "  int j;\n"
               "});",
               "DEBUG({\n"
               "  int  i;\n"
               "\n"
               "  // comment\n"
               "  int  j;\n"
               "});");

  verifyFormat("DEBUG({\n"
               "  if (a)\n"
               "    return;\n"
               "});");
  verifyGoogleFormat("DEBUG({\n"
                     "  if (a) return;\n"
                     "});");
  FormatStyle Style = getGoogleStyle();
  Style.ColumnLimit = 45;
  verifyFormat("Debug(\n"
               "    aaaaa,\n"
               "    {\n"
               "      if (aaaaaaaaaaaaaaaaaaaaaaaa) return;\n"
               "    },\n"
               "    a);",
               Style);

  verifyFormat("SomeFunction({MACRO({ return output; }), b});");

  verifyNoCrash("^{v^{a}}");
}

TEST_F(FormatTest, FormatNestedBlocksInMacros) {
  verifyFormat("#define MACRO()                     \\\n"
               "  Debug(aaa, /* force line break */ \\\n"
               "        {                           \\\n"
               "          int i;                    \\\n"
               "          int j;                    \\\n"
               "        })",
               "#define   MACRO()   Debug(aaa,  /* force line break */ \\\n"
               "          {  int   i;  int  j;   })",
               getGoogleStyle());

  verifyFormat("#define A                                       \\\n"
               "  [] {                                          \\\n"
               "    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(        \\\n"
               "        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx); \\\n"
               "  }",
               "#define A [] { xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx( \\\n"
               "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx); }",
               getGoogleStyle());
}

TEST_F(FormatTest, PutEmptyBlocksIntoOneLine) {
  verifyFormat("{}");
  verifyFormat("enum E {};");
  verifyFormat("enum E {}");
  FormatStyle Style = getLLVMStyle();
  Style.SpaceInEmptyBlock = true;
  verifyFormat("void f() { }", "void f() {}", Style);
  Style.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Empty;
  verifyFormat("while (true) { }", "while (true) {}", Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.BeforeElse = false;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  verifyFormat("if (a)\n"
               "{\n"
               "} else if (b)\n"
               "{\n"
               "} else\n"
               "{ }",
               Style);
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Never;
  verifyFormat("if (a) {\n"
               "} else if (b) {\n"
               "} else {\n"
               "}",
               Style);
  Style.BraceWrapping.BeforeElse = true;
  verifyFormat("if (a) { }\n"
               "else if (b) { }\n"
               "else { }",
               Style);
}

TEST_F(FormatTest, FormatBeginBlockEndMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.MacroBlockBegin = "^[A-Z_]+_BEGIN$";
  Style.MacroBlockEnd = "^[A-Z_]+_END$";
  verifyFormat("FOO_BEGIN\n"
               "  FOO_ENTRY\n"
               "FOO_END",
               Style);
  verifyFormat("FOO_BEGIN\n"
               "  NESTED_FOO_BEGIN\n"
               "    NESTED_FOO_ENTRY\n"
               "  NESTED_FOO_END\n"
               "FOO_END",
               Style);
  verifyFormat("FOO_BEGIN(Foo, Bar)\n"
               "  int x;\n"
               "  x = 1;\n"
               "FOO_END(Baz)",
               Style);

  Style.RemoveBracesLLVM = true;
  verifyNoCrash("for (;;)\n"
                "  FOO_BEGIN\n"
                "    foo();\n"
                "  FOO_END",
                Style);
}

//===----------------------------------------------------------------------===//
// Line break tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, PreventConfusingIndents) {
  verifyFormat(
      "void f() {\n"
      "  SomeLongMethodName(SomeReallyLongMethod(CallOtherReallyLongMethod(\n"
      "                         parameter, parameter, parameter)),\n"
      "                     SecondLongCall(parameter));\n"
      "}");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    [aaaaaaaaaaaaaaaaaaaaaaaa\n"
      "         [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]\n"
      "         [aaaaaaaaaaaaaaaaaaaaaaaa]];");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaa<\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>,\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaa>;");
  verifyFormat("int a = bbbb && ccc &&\n"
               "        fffff(\n"
               "#define A Just forcing a new line\n"
               "            ddd);");
}

TEST_F(FormatTest, LineBreakingInBinaryExpressions) {
  verifyFormat(
      "bool aaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaa).aaaaaaaaaaaaaaaaaaa() ||\n"
      "    bbbbbbbb();");
  verifyFormat(
      "bool aaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaa).aaaaaaaaaaaaaaaaaaa() or\n"
      "    bbbbbbbb();");

  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa != bbbbbbbbbbbbbbbbbb &&\n"
               "    ccccccccc == ddddddddddd;");
  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa != bbbbbbbbbbbbbbbbbb and\n"
               "    ccccccccc == ddddddddddd;");
  verifyFormat(
      "bool aaaaaaaaaaaaaaaaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa not_eq bbbbbbbbbbbbbbbbbb and\n"
      "    ccccccccc == ddddddddddd;");

  verifyFormat("aaaaaa = aaaaaaa(aaaaaaa, // break\n"
               "                 aaaaaa) &&\n"
               "         bbbbbb && cccccc;");
  verifyFormat("aaaaaa = aaaaaaa(aaaaaaa, // break\n"
               "                 aaaaaa) >>\n"
               "         bbbbbb;");
  verifyFormat("aa = Whitespaces.addUntouchableComment(\n"
               "    SourceMgr.getSpellingColumnNumber(\n"
               "        TheLine.Last->FormatTok.Tok.getLocation()) -\n"
               "    1);");

  verifyFormat("if ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "     bbbbbbbbbbbbbbbbbb) && // aaaaaaaaaaaaaaaa\n"
               "    cccccc) {\n}");
  verifyFormat("if constexpr ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "               bbbbbbbbbbbbbbbbbb) && // aaaaaaaaaaa\n"
               "              cccccc) {\n}");
  verifyFormat("if CONSTEXPR ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "               bbbbbbbbbbbbbbbbbb) && // aaaaaaaaaaa\n"
               "              cccccc) {\n}");
  verifyFormat("b = a &&\n"
               "    // Comment\n"
               "    b.c && d;");

  // If the LHS of a comparison is not a binary expression itself, the
  // additional linebreak confuses many people.
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) > 5) {\n"
      "}");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) == 5) {\n"
      "}");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaa.aaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) == 5) {\n"
      "}");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) <=> 5) {\n"
      "}");
  // Even explicit parentheses stress the precedence enough to make the
  // additional break unnecessary.
  verifyFormat("if ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) == 5) {\n"
               "}");
  // This cases is borderline, but with the indentation it is still readable.
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaa) > aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
      "}",
      getLLVMStyleWithColumns(75));

  // If the LHS is a binary expression, we should still use the additional break
  // as otherwise the formatting hides the operator precedence.
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "    5) {\n"
               "}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa <=>\n"
               "    5) {\n"
               "}");

  FormatStyle OnePerLine = getLLVMStyle();
  OnePerLine.BinPackParameters = false;
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}",
      OnePerLine);

  verifyFormat("int i = someFunction(aaaaaaa, 0)\n"
               "                .aaa(aaaaaaaaaaaaa) *\n"
               "            aaaaaaa +\n"
               "        aaaaaaa;",
               getLLVMStyleWithColumns(40));
}

TEST_F(FormatTest, ExpressionIndentation) {
  verifyFormat("bool value = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "                         bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb +\n"
               "                     bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb &&\n"
               "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa >\n"
               "                 ccccccccccccccccccccccccccccccccccccccccc;");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}");
  verifyFormat("if () {\n"
               "} else if (aaaaa && bbbbb > // break\n"
               "                        ccccc) {\n"
               "}");
  verifyFormat("if () {\n"
               "} else if constexpr (aaaaa && bbbbb > // break\n"
               "                                  ccccc) {\n"
               "}");
  verifyFormat("if () {\n"
               "} else if CONSTEXPR (aaaaa && bbbbb > // break\n"
               "                                  ccccc) {\n"
               "}");
  verifyFormat("if () {\n"
               "} else if (aaaaa &&\n"
               "           bbbbb > // break\n"
               "               ccccc &&\n"
               "           ddddd) {\n"
               "}");

  // Presence of a trailing comment used to change indentation of b.
  verifyFormat("return aaaaaaaaaaaaaaaaaaa +\n"
               "       b;\n"
               "return aaaaaaaaaaaaaaaaaaa +\n"
               "       b; //",
               getLLVMStyleWithColumns(30));
}

TEST_F(FormatTest, ExpressionIndentationBreakingBeforeOperators) {
  // Not sure what the best system is here. Like this, the LHS can be found
  // immediately above an operator (everything with the same or a higher
  // indent). The RHS is aligned right of the operator and so compasses
  // everything until something with the same indent as the operator is found.
  // FIXME: Is this a good system?
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat(
      "bool value = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                     + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                     + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                 == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                            * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
      "                        + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
      "             && aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                        * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                    > ccccccccccccccccccccccccccccccccccccccccc;",
      Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "              * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "               * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "           + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if () {\n"
               "} else if (aaaaa\n"
               "           && bbbbb // break\n"
               "                  > ccccc) {\n"
               "}",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "       && bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               Style);
  verifyFormat("return (a)\n"
               "       // comment\n"
               "       + b;",
               Style);
  verifyFormat(
      "int aaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                 * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
      "             + cc;",
      Style);

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    = aaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);

  // Forced by comments.
  verifyFormat(
      "unsigned ContentSize =\n"
      "    sizeof(int16_t)   // DWARF ARange version number\n"
      "    + sizeof(int32_t) // Offset of CU in the .debug_info section\n"
      "    + sizeof(int8_t)  // Pointer Size (in bytes)\n"
      "    + sizeof(int8_t); // Segment Size (in bytes)");

  verifyFormat("return boost::fusion::at_c<0>(iiii).second\n"
               "       == boost::fusion::at_c<1>(iiii).second;",
               Style);

  Style.ColumnLimit = 60;
  verifyFormat("zzzzzzzzzz\n"
               "    = bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "      >> aaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);

  Style.ColumnLimit = 80;
  Style.IndentWidth = 4;
  Style.TabWidth = 4;
  Style.UseTab = FormatStyle::UT_Always;
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  Style.AlignOperands = FormatStyle::OAS_DontAlign;
  verifyFormat("return someVeryVeryLongConditionThatBarelyFitsOnALine\n"
               "\t&& (someOtherLongishConditionPart1\n"
               "\t\t|| someOtherEvenLongerNestedConditionPart2);",
               "return someVeryVeryLongConditionThatBarelyFitsOnALine && "
               "(someOtherLongishConditionPart1 || "
               "someOtherEvenLongerNestedConditionPart2);",
               Style);
}

TEST_F(FormatTest, ExpressionIndentationStrictAlign) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  Style.AlignOperands = FormatStyle::OAS_AlignAfterOperator;

  verifyFormat("bool value = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                   + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                   + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "              == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                         * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "                     + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "          && aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                     * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                 > ccccccccccccccccccccccccccccccccccccccccc;",
               Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "              * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "               * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "           + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if () {\n"
               "} else if (aaaaa\n"
               "           && bbbbb // break\n"
               "                  > ccccc) {\n"
               "}",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    && bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               Style);
  verifyFormat("return (a)\n"
               "     // comment\n"
               "     + b;",
               Style);
  verifyFormat(
      "int aaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "               * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
      "           + cc;",
      Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111\n"
               "     : bbbbbbbbbbbbbbbb ? 2222222222222222\n"
               "                        : 3333333333333333;",
               Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaaa    ? bbbbbbbbbbbbbbbbbb\n"
      "                           : ccccccccccccccc ? dddddddddddddddddd\n"
      "                                             : eeeeeeeeeeeeeeeeee)\n"
      "     : bbbbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    = aaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);

  verifyFormat("return boost::fusion::at_c<0>(iiii).second\n"
               "    == boost::fusion::at_c<1>(iiii).second;",
               Style);

  Style.ColumnLimit = 60;
  verifyFormat("zzzzzzzzzzzzz\n"
               "    = bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "   >> aaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);

  // Forced by comments.
  Style.ColumnLimit = 80;
  verifyFormat(
      "unsigned ContentSize\n"
      "    = sizeof(int16_t) // DWARF ARange version number\n"
      "    + sizeof(int32_t) // Offset of CU in the .debug_info section\n"
      "    + sizeof(int8_t)  // Pointer Size (in bytes)\n"
      "    + sizeof(int8_t); // Segment Size (in bytes)",
      Style);

  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
  verifyFormat(
      "unsigned ContentSize =\n"
      "    sizeof(int16_t)   // DWARF ARange version number\n"
      "    + sizeof(int32_t) // Offset of CU in the .debug_info section\n"
      "    + sizeof(int8_t)  // Pointer Size (in bytes)\n"
      "    + sizeof(int8_t); // Segment Size (in bytes)",
      Style);

  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  verifyFormat(
      "unsigned ContentSize =\n"
      "    sizeof(int16_t)   // DWARF ARange version number\n"
      "    + sizeof(int32_t) // Offset of CU in the .debug_info section\n"
      "    + sizeof(int8_t)  // Pointer Size (in bytes)\n"
      "    + sizeof(int8_t); // Segment Size (in bytes)",
      Style);
}

TEST_F(FormatTest, EnforcedOperatorWraps) {
  // Here we'd like to wrap after the || operators, but a comment is forcing an
  // earlier wrap.
  verifyFormat("bool x = aaaaa //\n"
               "         || bbbbb\n"
               "         //\n"
               "         || cccc;");
}

TEST_F(FormatTest, NoOperandAlignment) {
  FormatStyle Style = getLLVMStyle();
  Style.AlignOperands = FormatStyle::OAS_DontAlign;
  verifyFormat("aaaaaaaaaaaaaa(aaaaaaaaaaaa,\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "                   aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
  verifyFormat("bool value = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "            + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "    && aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        > ccccccccccccccccccccccccccccccccccccccccc;",
               Style);

  verifyFormat("int aaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "    + cc;",
               Style);
  verifyFormat("int a = aa\n"
               "    + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "        * cccccccccccccccccccccccccccccccccccc;",
               Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  verifyFormat("return (a > b\n"
               "    // comment1\n"
               "    // comment2\n"
               "    || c);",
               Style);
}

TEST_F(FormatTest, BreakingBeforeNonAssigmentOperators) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               Style);
}

TEST_F(FormatTest, AllowBinPackingInsideArguments) {
  FormatStyle Style = getLLVMStyleWithColumns(40);
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
  Style.BinPackArguments = false;
  verifyFormat("void test() {\n"
               "  someFunction(\n"
               "      this + argument + is + quite\n"
               "      + long + so + it + gets + wrapped\n"
               "      + but + remains + bin - packed);\n"
               "}",
               Style);
  verifyFormat("void test() {\n"
               "  someFunction(arg1,\n"
               "               this + argument + is\n"
               "                   + quite + long + so\n"
               "                   + it + gets + wrapped\n"
               "                   + but + remains + bin\n"
               "                   - packed,\n"
               "               arg3);\n"
               "}",
               Style);
  verifyFormat("void test() {\n"
               "  someFunction(\n"
               "      arg1,\n"
               "      this + argument + has\n"
               "          + anotherFunc(nested,\n"
               "                        calls + whose\n"
               "                            + arguments\n"
               "                            + are + also\n"
               "                            + wrapped,\n"
               "                        in + addition)\n"
               "          + to + being + bin - packed,\n"
               "      arg3);\n"
               "}",
               Style);

  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  verifyFormat("void test() {\n"
               "  someFunction(\n"
               "      arg1,\n"
               "      this + argument + has +\n"
               "          anotherFunc(nested,\n"
               "                      calls + whose +\n"
               "                          arguments +\n"
               "                          are + also +\n"
               "                          wrapped,\n"
               "                      in + addition) +\n"
               "          to + being + bin - packed,\n"
               "      arg3);\n"
               "}",
               Style);
}

TEST_F(FormatTest, BreakBinaryOperatorsInPresenceOfTemplates) {
  auto Style = getLLVMStyleWithColumns(45);
  EXPECT_EQ(Style.BreakBeforeBinaryOperators, FormatStyle::BOS_None);
  verifyFormat("bool b =\n"
               "    is_default_constructible_v<hash<T>> and\n"
               "    is_copy_constructible_v<hash<T>> and\n"
               "    is_move_constructible_v<hash<T>> and\n"
               "    is_copy_assignable_v<hash<T>> and\n"
               "    is_move_assignable_v<hash<T>> and\n"
               "    is_destructible_v<hash<T>> and\n"
               "    is_swappable_v<hash<T>> and\n"
               "    is_callable_v<hash<T>(T)>;",
               Style);

  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
  verifyFormat("bool b = is_default_constructible_v<hash<T>>\n"
               "         and is_copy_constructible_v<hash<T>>\n"
               "         and is_move_constructible_v<hash<T>>\n"
               "         and is_copy_assignable_v<hash<T>>\n"
               "         and is_move_assignable_v<hash<T>>\n"
               "         and is_destructible_v<hash<T>>\n"
               "         and is_swappable_v<hash<T>>\n"
               "         and is_callable_v<hash<T>(T)>;",
               Style);

  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat("bool b = is_default_constructible_v<hash<T>>\n"
               "         and is_copy_constructible_v<hash<T>>\n"
               "         and is_move_constructible_v<hash<T>>\n"
               "         and is_copy_assignable_v<hash<T>>\n"
               "         and is_move_assignable_v<hash<T>>\n"
               "         and is_destructible_v<hash<T>>\n"
               "         and is_swappable_v<hash<T>>\n"
               "         and is_callable_v<hash<T>(T)>;",
               Style);
}

TEST_F(FormatTest, ConstructorInitializers) {
  verifyFormat("Constructor() : Initializer(FitsOnTheLine) {}");
  verifyFormat("Constructor() : Inttializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(45));
  verifyFormat("Constructor()\n"
               "    : Inttializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(44));
  verifyFormat("Constructor()\n"
               "    : Inttializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(43));

  verifyFormat("template <typename T>\n"
               "Constructor() : Initializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(45));

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}");

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}");
  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}");
  verifyFormat("Constructor(aaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    : aaaaaaaaaa(aaaaaa) {}");

  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                               aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaa() {}");

  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}");

  verifyFormat("Constructor(int Parameter = 0)\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaa(aaaaaaaaaaaaaaaaa) {}");
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbbbbb(b) {\n"
               "}",
               getLLVMStyleWithColumns(60));
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaa(aaaa, aaaa)) {}");

  // Here a line could be saved by splitting the second initializer onto two
  // lines, but that is not desirable.
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaa(aaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaat(aaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}");

  FormatStyle OnePerLine = getLLVMStyle();
  OnePerLine.PackConstructorInitializers = FormatStyle::PCIS_Never;
  verifyFormat("MyClass::MyClass()\n"
               "    : a(a),\n"
               "      b(b),\n"
               "      c(c) {}",
               OnePerLine);
  verifyFormat("MyClass::MyClass()\n"
               "    : a(a), // comment\n"
               "      b(b),\n"
               "      c(c) {}",
               OnePerLine);
  verifyFormat("MyClass::MyClass(int a)\n"
               "    : b(a),      // comment\n"
               "      c(a + 1) { // lined up\n"
               "}",
               OnePerLine);
  verifyFormat("Constructor()\n"
               "    : a(b, b, b) {}",
               OnePerLine);
  OnePerLine.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  OnePerLine.AllowAllParametersOfDeclarationOnNextLine = false;
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
               OnePerLine);
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), // Some comment\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
               OnePerLine);
  verifyFormat("MyClass::MyClass(int var)\n"
               "    : some_var_(var),            // 4 space indent\n"
               "      some_other_var_(var + 1) { // lined up\n"
               "}",
               OnePerLine);
  verifyFormat("Constructor()\n"
               "    : aaaaa(aaaaaa),\n"
               "      aaaaa(aaaaaa),\n"
               "      aaaaa(aaaaaa),\n"
               "      aaaaa(aaaaaa),\n"
               "      aaaaa(aaaaaa) {}",
               OnePerLine);
  verifyFormat("Constructor()\n"
               "    : aaaaa(aaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaaaaaaaaaaaaaaaaaa) {}",
               OnePerLine);
  OnePerLine.BinPackParameters = false;
  verifyFormat(
      "Constructor()\n"
      "    : aaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "          aaaaaaaaaaa().aaa(),\n"
      "          aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
      OnePerLine);
  OnePerLine.ColumnLimit = 60;
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a),\n"
               "      bbbbbbbbbbbbbbbbbbbbbbbb(b) {}",
               OnePerLine);

  verifyFormat("Constructor()\n"
               "    : // Comment forcing unwanted break.\n"
               "      aaaa(aaaa) {}",
               "Constructor() :\n"
               "    // Comment forcing unwanted break.\n"
               "    aaaa(aaaa) {}");
}

TEST_F(FormatTest, AllowAllConstructorInitializersOnNextLine) {
  FormatStyle Style = getLLVMStyleWithColumns(60);
  Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeComma;
  Style.BinPackParameters = false;

  for (int i = 0; i < 4; ++i) {
    // Test all combinations of parameters that should not have an effect.
    Style.AllowAllParametersOfDeclarationOnNextLine = i & 1;
    Style.AllowAllArgumentsOnNextLine = i & 2;

    Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
    Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeComma;
    verifyFormat("Constructor()\n"
                 "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);
    verifyFormat("Constructor() : a(a), b(b) {}", Style);

    Style.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
    verifyFormat("Constructor()\n"
                 "    : aaaaaaaaaaaaaaaaaaaa(a)\n"
                 "    , bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);
    verifyFormat("Constructor() : a(a), b(b) {}", Style);

    Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
    verifyFormat("Constructor()\n"
                 "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);
    verifyFormat("Constructor()\n"
                 "    : a(a), b(b) {}",
                 Style);
    verifyFormat("Constructor()\n"
                 "    : aaaaaaaaaaaaaaaaaaaa(a)\n"
                 "    , bbbbbbbbbbbbbbbbbbbbb(b)\n"
                 "    , cccccccccccccccccccccc(c) {}",
                 Style);

    Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeColon;
    Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
    verifyFormat("Constructor()\n"
                 "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);

    Style.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
    verifyFormat("Constructor()\n"
                 "    : aaaaaaaaaaaaaaaaaaaa(a),\n"
                 "      bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);

    Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
    verifyFormat("Constructor()\n"
                 "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);
    verifyFormat("Constructor()\n"
                 "    : a(a), b(b) {}",
                 Style);
    verifyFormat("Constructor()\n"
                 "    : aaaaaaaaaaaaaaaaaaaa(a),\n"
                 "      bbbbbbbbbbbbbbbbbbbbb(b),\n"
                 "      cccccccccccccccccccccc(c) {}",
                 Style);

    Style.BreakConstructorInitializers = FormatStyle::BCIS_AfterColon;
    Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
    verifyFormat("Constructor() :\n"
                 "    aaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);

    Style.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
    verifyFormat("Constructor() :\n"
                 "    aaaaaaaaaaaaaaaaaa(a),\n"
                 "    bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);

    Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
    verifyFormat("Constructor() :\n"
                 "    aaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
                 Style);
    verifyFormat("Constructor() :\n"
                 "    a(a), b(b) {}",
                 Style);
    verifyFormat("Constructor() :\n"
                 "    aaaaaaaaaaaaaaaaaaaa(a),\n"
                 "    bbbbbbbbbbbbbbbbbbbbb(b),\n"
                 "    cccccccccccccccccccccc(c) {}",
                 Style);
  }

  // Test interactions between AllowAllParametersOfDeclarationOnNextLine and
  // AllowAllConstructorInitializersOnNextLine in all
  // BreakConstructorInitializers modes
  Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeComma;
  Style.AllowAllParametersOfDeclarationOnNextLine = true;
  Style.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa, int bbbbbbbbbbbbb)\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a)\n"
               "    , bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb,\n"
               "    int cccccccccccccccc)\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb,\n"
               "    int cccccccccccccccc)\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.AllowAllParametersOfDeclarationOnNextLine = false;
  Style.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb)\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a)\n"
               "    , bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeColon;

  Style.AllowAllParametersOfDeclarationOnNextLine = true;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa, int bbbbbbbbbbbbb)\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a),\n"
               "      bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb,\n"
               "    int cccccccccccccccc)\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb,\n"
               "    int cccccccccccccccc)\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.AllowAllParametersOfDeclarationOnNextLine = false;
  Style.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb)\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a),\n"
               "      bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.BreakConstructorInitializers = FormatStyle::BCIS_AfterColon;
  Style.AllowAllParametersOfDeclarationOnNextLine = true;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa, int bbbbbbbbbbbbb) :\n"
               "    aaaaaaaaaaaaaaaaaaaa(a),\n"
               "    bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb,\n"
               "    int cccccccccccccccc) :\n"
               "    aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb,\n"
               "    int cccccccccccccccc) :\n"
               "    aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style.AllowAllParametersOfDeclarationOnNextLine = false;
  Style.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
  verifyFormat("SomeClassWithALongName::Constructor(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb) :\n"
               "    aaaaaaaaaaaaaaaaaaaa(a),\n"
               "    bbbbbbbbbbbbbbbbbbbbb(b) {}",
               Style);

  Style = getLLVMStyleWithColumns(0);
  Style.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
  verifyFormat("Foo(Bar bar, Baz baz) : bar(bar), baz(baz) {}", Style);
  verifyNoChange("Foo(Bar bar, Baz baz)\n"
                 "    : bar(bar), baz(baz) {}",
                 Style);
}

TEST_F(FormatTest, AllowAllArgumentsOnNextLine) {
  FormatStyle Style = getLLVMStyleWithColumns(60);
  Style.BinPackArguments = false;
  for (int i = 0; i < 4; ++i) {
    // Test all combinations of parameters that should not have an effect.
    Style.AllowAllParametersOfDeclarationOnNextLine = i & 1;
    Style.PackConstructorInitializers =
        i & 2 ? FormatStyle::PCIS_BinPack : FormatStyle::PCIS_Never;

    Style.AllowAllArgumentsOnNextLine = true;
    verifyFormat("void foo() {\n"
                 "  FunctionCallWithReallyLongName(\n"
                 "      aaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbb);\n"
                 "}",
                 Style);
    Style.AllowAllArgumentsOnNextLine = false;
    verifyFormat("void foo() {\n"
                 "  FunctionCallWithReallyLongName(\n"
                 "      aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
                 "      bbbbbbbbbbbb);\n"
                 "}",
                 Style);

    Style.AllowAllArgumentsOnNextLine = true;
    verifyFormat("void foo() {\n"
                 "  auto VariableWithReallyLongName = {\n"
                 "      aaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbb};\n"
                 "}",
                 Style);
    Style.AllowAllArgumentsOnNextLine = false;
    verifyFormat("void foo() {\n"
                 "  auto VariableWithReallyLongName = {\n"
                 "      aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
                 "      bbbbbbbbbbbb};\n"
                 "}",
                 Style);
  }

  // This parameter should not affect declarations.
  Style.BinPackParameters = false;
  Style.AllowAllArgumentsOnNextLine = false;
  Style.AllowAllParametersOfDeclarationOnNextLine = true;
  verifyFormat("void FunctionCallWithReallyLongName(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaa, int bbbbbbbbbbbb);",
               Style);
  Style.AllowAllParametersOfDeclarationOnNextLine = false;
  verifyFormat("void FunctionCallWithReallyLongName(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbb);",
               Style);
}

TEST_F(FormatTest, AllowAllArgumentsOnNextLineDontAlign) {
  // Check that AllowAllArgumentsOnNextLine is respected for both BAS_DontAlign
  // and BAS_Align.
  FormatStyle Style = getLLVMStyleWithColumns(35);
  StringRef Input = "functionCall(paramA, paramB, paramC);\n"
                    "void functionDecl(int A, int B, int C);";
  Style.AllowAllArgumentsOnNextLine = false;
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  verifyFormat(StringRef("functionCall(paramA, paramB,\n"
                         "    paramC);\n"
                         "void functionDecl(int A, int B,\n"
                         "    int C);"),
               Input, Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_Align;
  verifyFormat(StringRef("functionCall(paramA, paramB,\n"
                         "             paramC);\n"
                         "void functionDecl(int A, int B,\n"
                         "                  int C);"),
               Input, Style);
  // However, BAS_AlwaysBreak and BAS_BlockIndent should take precedence over
  // AllowAllArgumentsOnNextLine.
  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  verifyFormat(StringRef("functionCall(\n"
                         "    paramA, paramB, paramC);\n"
                         "void functionDecl(\n"
                         "    int A, int B, int C);"),
               Input, Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;
  verifyFormat("functionCall(\n"
               "    paramA, paramB, paramC\n"
               ");\n"
               "void functionDecl(\n"
               "    int A, int B, int C\n"
               ");",
               Input, Style);

  // When AllowAllArgumentsOnNextLine is set, we prefer breaking before the
  // first argument.
  Style.AllowAllArgumentsOnNextLine = true;
  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  verifyFormat(StringRef("functionCall(\n"
                         "    paramA, paramB, paramC);\n"
                         "void functionDecl(\n"
                         "    int A, int B, int C);"),
               Input, Style);
  // It wouldn't fit on one line with aligned parameters so this setting
  // doesn't change anything for BAS_Align.
  Style.AlignAfterOpenBracket = FormatStyle::BAS_Align;
  verifyFormat(StringRef("functionCall(paramA, paramB,\n"
                         "             paramC);\n"
                         "void functionDecl(int A, int B,\n"
                         "                  int C);"),
               Input, Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  verifyFormat(StringRef("functionCall(\n"
                         "    paramA, paramB, paramC);\n"
                         "void functionDecl(\n"
                         "    int A, int B, int C);"),
               Input, Style);
}

TEST_F(FormatTest, BreakBeforeInlineASMColon) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeInlineASMColon = FormatStyle::BBIAS_Never;
  /* Test the behaviour with long lines */
  Style.ColumnLimit = 40;
  verifyFormat("asm volatile(\"loooooooooooooooooooong\",\n"
               "             : : val);",
               Style);
  verifyFormat("asm volatile(\"loooooooooooooooooooong\",\n"
               "             : val1 : val2);",
               Style);
  verifyFormat("asm(\"movq\\t%%rbx, %%rsi\\n\\t\"\n"
               "    \"cpuid\\n\\t\"\n"
               "    \"xchgq\\t%%rbx %%rsi\\n\\t\",\n"
               "    : \"=a\" : \"a\");",
               Style);
  Style.ColumnLimit = 80;
  verifyFormat("asm volatile(\"string\", : : val);", Style);
  verifyFormat("asm volatile(\"string\", : val1 : val2);", Style);

  Style.BreakBeforeInlineASMColon = FormatStyle::BBIAS_Always;
  verifyFormat("asm volatile(\"string\",\n"
               "             :\n"
               "             : val);",
               Style);
  verifyFormat("asm volatile(\"string\",\n"
               "             : val1\n"
               "             : val2);",
               Style);
  /* Test the behaviour with long lines */
  Style.ColumnLimit = 40;
  verifyFormat("asm(\"movq\\t%%rbx, %%rsi\\n\\t\"\n"
               "    \"cpuid\\n\\t\"\n"
               "    \"xchgq\\t%%rbx, %%rsi\\n\\t\"\n"
               "    : \"=a\"(*rEAX)\n"
               "    : \"a\"(value));",
               Style);
  verifyFormat("asm(\"movq\\t%%rbx, %%rsi\\n\\t\"\n"
               "    \"cpuid\\n\\t\"\n"
               "    \"xchgq\\t%%rbx, %%rsi\\n\\t\"\n"
               "    :\n"
               "    : \"a\"(value));",
               Style);
  verifyFormat("asm volatile(\"loooooooooooooooooooong\",\n"
               "             :\n"
               "             : val);",
               Style);
  verifyFormat("asm volatile(\"loooooooooooooooooooong\",\n"
               "             : val1\n"
               "             : val2);",
               Style);
}

TEST_F(FormatTest, BreakConstructorInitializersAfterColon) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakConstructorInitializers = FormatStyle::BCIS_AfterColon;

  verifyFormat("Constructor() : Initializer(FitsOnTheLine) {}");
  verifyFormat("Constructor() : Initializer(FitsOnTheLine) {}",
               getStyleWithColumns(Style, 45));
  verifyFormat("Constructor() :\n"
               "    Initializer(FitsOnTheLine) {}",
               getStyleWithColumns(Style, 44));
  verifyFormat("Constructor() :\n"
               "    Initializer(FitsOnTheLine) {}",
               getStyleWithColumns(Style, 43));

  verifyFormat("template <typename T>\n"
               "Constructor() : Initializer(FitsOnTheLine) {}",
               getStyleWithColumns(Style, 50));
  verifyFormat(
      "Class::Class(int some, int arguments, int loooooooooooooooooooong,\n"
      "             int mooooooooooooore) noexcept :\n"
      "    Super{some, arguments}, Member{5}, Member2{2} {}",
      Style);
  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  verifyFormat(
      "SomeClass::Constructor() :\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}",
      Style);
  verifyFormat(
      "SomeClass::Constructor() : // NOLINT\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}",
      Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
  verifyFormat(
      "SomeClass::Constructor() :\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}",
      Style);
  verifyFormat(
      "SomeClass::Constructor() : // NOLINT\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}",
      Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_BinPack;
  verifyFormat(
      "SomeClass::Constructor() :\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}",
      Style);

  verifyFormat(
      "SomeClass::Constructor() :\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
      Style);
  verifyFormat(
      "SomeClass::Constructor() :\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "    aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}",
      Style);
  verifyFormat(
      "Ctor(aaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "     aaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) : aaaaaaaaaa(aaaaaa) {}",
      Style);

  verifyFormat("Constructor() :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                             aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "    aaaaaaaaaaaaaaaaaaaaaaa() {}",
               Style);

  verifyFormat("Constructor() :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
               Style);

  verifyFormat("Constructor(int Parameter = 0) :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa),\n"
               "    aaaaaaaaaaaa(aaaaaaaaaaaaaaaaa) {}",
               Style);
  verifyFormat("Constructor() :\n"
               "    aaaaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbbbbb(b) {\n"
               "}",
               getStyleWithColumns(Style, 60));
  verifyFormat("Constructor() :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaa(aaaa, aaaa)) {}",
               Style);

  // Here a line could be saved by splitting the second initializer onto two
  // lines, but that is not desirable.
  verifyFormat("Constructor() :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "    aaaaaaaaaaa(aaaaaaaaaaa),\n"
               "    aaaaaaaaaaaaaaaaaaaaat(aaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
               Style);

  FormatStyle OnePerLine = Style;
  OnePerLine.PackConstructorInitializers = FormatStyle::PCIS_CurrentLine;
  verifyFormat("SomeClass::Constructor() :\n"
               "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
               OnePerLine);
  verifyFormat("SomeClass::Constructor() :\n"
               "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa), // Some comment\n"
               "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
               OnePerLine);
  verifyFormat("Foo::Foo(int i, int j) : // NOLINT\n"
               "    i(i),                // comment\n"
               "    j(j) {}",
               OnePerLine);
  verifyFormat("MyClass::MyClass(int var) :\n"
               "    some_var_(var),            // 4 space indent\n"
               "    some_other_var_(var + 1) { // lined up\n"
               "}",
               OnePerLine);
  verifyFormat("Constructor() :\n"
               "    aaaaa(aaaaaa),\n"
               "    aaaaa(aaaaaa),\n"
               "    aaaaa(aaaaaa),\n"
               "    aaaaa(aaaaaa),\n"
               "    aaaaa(aaaaaa) {}",
               OnePerLine);
  verifyFormat("Constructor() :\n"
               "    aaaaa(aaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaa,\n"
               "          aaaaaaaaaaaaaaaaaaaaaa) {}",
               OnePerLine);
  OnePerLine.BinPackParameters = false;
  verifyFormat("Constructor() :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaa().aaa(),\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
               OnePerLine);
  OnePerLine.ColumnLimit = 60;
  verifyFormat("Constructor() :\n"
               "    aaaaaaaaaaaaaaaaaaaa(a),\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbb(b) {}",
               OnePerLine);

  verifyFormat("Constructor() :\n"
               "    // Comment forcing unwanted break.\n"
               "    aaaa(aaaa) {}",
               Style);
  verifyFormat("Constructor() : // NOLINT\n"
               "    aaaa(aaaa) {}",
               Style);
  verifyFormat("Constructor() : // A very long trailing comment that cannot fit"
               " on a single\n"
               "                // line.\n"
               "    aaaa(aaaa) {}",
               "Constructor() : // A very long trailing comment that cannot fit"
               " on a single line.\n"
               "    aaaa(aaaa) {}",
               Style);

  Style.ColumnLimit = 0;
  verifyFormat("SomeClass::Constructor() :\n"
               "    a(a) {}",
               Style);
  verifyFormat("SomeClass::Constructor() noexcept :\n"
               "    a(a) {}",
               Style);
  verifyFormat("SomeClass::Constructor() :\n"
               "    a(a), b(b), c(c) {}",
               Style);
  verifyFormat("SomeClass::Constructor() :\n"
               "    a(a) {\n"
               "  foo();\n"
               "  bar();\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  verifyFormat("SomeClass::Constructor() :\n"
               "    a(a), b(b), c(c) {\n"
               "}",
               Style);
  verifyFormat("SomeClass::Constructor() :\n"
               "    a(a) {\n"
               "}",
               Style);

  Style.ColumnLimit = 80;
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  Style.ConstructorInitializerIndentWidth = 2;
  verifyFormat("SomeClass::Constructor() : a(a), b(b), c(c) {}", Style);
  verifyFormat("SomeClass::Constructor() :\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "  bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb {}",
               Style);

  // `ConstructorInitializerIndentWidth` actually applies to InheritanceList as
  // well
  Style.BreakInheritanceList = FormatStyle::BILS_BeforeColon;
  verifyFormat(
      "class SomeClass\n"
      "  : public aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    public bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb {};",
      Style);
  Style.BreakInheritanceList = FormatStyle::BILS_BeforeComma;
  verifyFormat(
      "class SomeClass\n"
      "  : public aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "  , public bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb {};",
      Style);
  Style.BreakInheritanceList = FormatStyle::BILS_AfterColon;
  verifyFormat(
      "class SomeClass :\n"
      "  public aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "  public bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb {};",
      Style);
  Style.BreakInheritanceList = FormatStyle::BILS_AfterComma;
  verifyFormat(
      "class SomeClass\n"
      "  : public aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    public bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb {};",
      Style);
}

#ifndef EXPENSIVE_CHECKS
// Expensive checks enables libstdc++ checking which includes validating the
// state of ranges used in std::priority_queue - this blows out the
// runtime/scalability of the function and makes this test unacceptably slow.
TEST_F(FormatTest, MemoizationTests) {
  // This breaks if the memoization lookup does not take \c Indent and
  // \c LastSpace into account.
  verifyFormat(
      "extern CFRunLoopTimerRef\n"
      "CFRunLoopTimerCreate(CFAllocatorRef allocato, CFAbsoluteTime fireDate,\n"
      "                     CFTimeInterval interval, CFOptionFlags flags,\n"
      "                     CFIndex order, CFRunLoopTimerCallBack callout,\n"
      "                     CFRunLoopTimerContext *context) {}");

  // Deep nesting somewhat works around our memoization.
  verifyFormat(
      "aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(\n"
      "    aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(\n"
      "        aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(\n"
      "            aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(\n"
      "                aaaaa())))))))))))))))))))))))))))))))))))))));",
      getLLVMStyleWithColumns(65));
  verifyFormat(
      "aaaaa(\n"
      "    aaaaa,\n"
      "    aaaaa(\n"
      "        aaaaa,\n"
      "        aaaaa(\n"
      "            aaaaa,\n"
      "            aaaaa(\n"
      "                aaaaa,\n"
      "                aaaaa(\n"
      "                    aaaaa,\n"
      "                    aaaaa(\n"
      "                        aaaaa,\n"
      "                        aaaaa(\n"
      "                            aaaaa,\n"
      "                            aaaaa(\n"
      "                                aaaaa,\n"
      "                                aaaaa(\n"
      "                                    aaaaa,\n"
      "                                    aaaaa(\n"
      "                                        aaaaa,\n"
      "                                        aaaaa(\n"
      "                                            aaaaa,\n"
      "                                            aaaaa(\n"
      "                                                aaaaa,\n"
      "                                                aaaaa))))))))))));",
      getLLVMStyleWithColumns(65));
  verifyFormat(
      "a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(), a), a), a), a),\n"
      "                                  a),\n"
      "                                a),\n"
      "                              a),\n"
      "                            a),\n"
      "                          a),\n"
      "                        a),\n"
      "                      a),\n"
      "                    a),\n"
      "                  a),\n"
      "                a),\n"
      "              a),\n"
      "            a),\n"
      "          a),\n"
      "        a),\n"
      "      a),\n"
      "    a),\n"
      "  a)",
      getLLVMStyleWithColumns(65));

  // This test takes VERY long when memoization is broken.
  FormatStyle OnePerLine = getLLVMStyle();
  OnePerLine.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  OnePerLine.BinPackParameters = false;
  std::string input = "Constructor()\n"
                      "    : aaaa(a,\n";
  for (unsigned i = 0, e = 80; i != e; ++i)
    input += "           a,\n";
  input += "           a) {}";
  verifyFormat(input, OnePerLine);
  OnePerLine.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
  verifyFormat(input, OnePerLine);
}
#endif

TEST_F(FormatTest, BreaksAsHighAsPossible) {
  verifyFormat(
      "void f() {\n"
      "  if ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaa && aaaaaaaaaaaaaaaaaaaaaaaaaa) ||\n"
      "      (bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb && bbbbbbbbbbbbbbbbbbbbbbbbbb))\n"
      "    f();\n"
      "}");
  verifyFormat("if (Intervals[i].getRange().getFirst() <\n"
               "    Intervals[i - 1].getRange().getLast()) {\n}");
}

TEST_F(FormatTest, BreaksFunctionDeclarations) {
  // Principially, we break function declarations in a certain order:
  // 1) break amongst arguments.
  verifyFormat("Aaaaaaaaaaaaaa bbbbbbbbbbbbbb(Cccccccccccccc cccccccccccccc,\n"
               "                              Cccccccccccccc cccccccccccccc);");
  verifyFormat("template <class TemplateIt>\n"
               "SomeReturnType SomeFunction(TemplateIt begin, TemplateIt end,\n"
               "                            TemplateIt *stop) {}");

  // 2) break after return type.
  verifyGoogleFormat(
      "Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "bbbbbbbbbbbbbb(Cccccccccccccc cccccccccccccccccccccccccc);");

  // 3) break after (.
  verifyGoogleFormat(
      "Aaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbb(\n"
      "    Cccccccccccccccccccccccccccccc cccccccccccccccccccccccccccccccc);");

  // 4) break before after nested name specifiers.
  verifyGoogleFormat(
      "Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "SomeClasssssssssssssssssssssssssssssssssssssss::\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(Cccccccccccccc cccccccccc);");

  // However, there are exceptions, if a sufficient amount of lines can be
  // saved.
  // FIXME: The precise cut-offs wrt. the number of saved lines might need some
  // more adjusting.
  verifyFormat("Aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb(Cccccccccccccc cccccccccc,\n"
               "                                  Cccccccccccccc cccccccccc,\n"
               "                                  Cccccccccccccc cccccccccc,\n"
               "                                  Cccccccccccccc cccccccccc,\n"
               "                                  Cccccccccccccc cccccccccc);");
  verifyGoogleFormat(
      "Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "bbbbbbbbbbb(Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
      "            Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
      "            Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc);");
  verifyFormat(
      "Aaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc);");
  verifyFormat("Aaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(\n"
               "    Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
               "    Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
               "    Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
               "    Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc);");

  // Break after multi-line parameters.
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    bbbb bbbb);");
  verifyFormat("void SomeLoooooooooooongFunction(\n"
               "    std::unique_ptr<aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb);");

  // Treat overloaded operators like other functions.
  verifyFormat("SomeLoooooooooooooooooooooooooogType\n"
               "operator>(const SomeLoooooooooooooooooooooooooogType &other);");
  verifyFormat("SomeLoooooooooooooooooooooooooogType\n"
               "operator>>(const SomeLooooooooooooooooooooooooogType &other);");
  verifyFormat("SomeLoooooooooooooooooooooooooogType\n"
               "operator<<(const SomeLooooooooooooooooooooooooogType &other);");
  verifyGoogleFormat(
      "SomeLoooooooooooooooooooooooooooooogType operator>>(\n"
      "    const SomeLooooooooogType &a, const SomeLooooooooogType &b);");
  verifyGoogleFormat(
      "SomeLoooooooooooooooooooooooooooooogType operator<<(\n"
      "    const SomeLooooooooogType &a, const SomeLooooooooogType &b);");
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = 1);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaa\n"
               "aaaaaaaaaaaaaaaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaa = 1);");
  verifyGoogleFormat(
      "typename aaaaaaaaaa<aaaaaa>::aaaaaaaaaaa\n"
      "aaaaaaaaaa<aaaaaa>::aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    bool *aaaaaaaaaaaaaaaaaa, bool *aa) {}");
  verifyGoogleFormat("template <typename T>\n"
                     "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
                     "aaaaaaaaaaaaaaaaaaaaaaa<T>::aaaaaaaaaaaaa(\n"
                     "    aaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaa);");

  FormatStyle Style = getLLVMStyle();
  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaa* const aaaaaaaaaaaa) {}",
               Style);
  verifyFormat("void aaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
               Style);
}

TEST_F(FormatTest, DontBreakBeforeQualifiedOperator) {
  // Regression test for https://bugs.llvm.org/show_bug.cgi?id=40516:
  // Prefer keeping `::` followed by `operator` together.
  verifyFormat("const aaaa::bbbbbbb &\n"
               "ccccccccc::operator++() {\n"
               "  stuff();\n"
               "}",
               "const aaaa::bbbbbbb\n"
               "&ccccccccc::operator++() { stuff(); }",
               getLLVMStyleWithColumns(40));
}

TEST_F(FormatTest, TrailingReturnType) {
  verifyFormat("auto foo() -> int;");
  // correct trailing return type spacing
  verifyFormat("auto operator->() -> int;");
  verifyFormat("auto operator++(int) -> int;");

  verifyFormat("struct S {\n"
               "  auto bar() const -> int;\n"
               "};");
  verifyFormat("template <size_t Order, typename T>\n"
               "auto load_img(const std::string &filename)\n"
               "    -> alias::tensor<Order, T, mem::tag::cpu> {}");
  verifyFormat("auto SomeFunction(A aaaaaaaaaaaaaaaaaaaaa) const\n"
               "    -> decltype(f(aaaaaaaaaaaaaaaaaaaaa)) {}");
  verifyFormat("auto doSomething(Aaaaaa *aaaaaa) -> decltype(aaaaaa->f()) {}");
  verifyFormat("template <typename T>\n"
               "auto aaaaaaaaaaaaaaaaaaaaaa(T t)\n"
               "    -> decltype(eaaaaaaaaaaaaaaa<T>(t.a).aaaaaaaa());");

  FormatStyle Style = getLLVMStyleWithColumns(60);
  verifyFormat("#define MAKE_DEF(NAME)                                     \\\n"
               "  auto NAME() -> int { return 42; }",
               Style);

  // Not trailing return types.
  verifyFormat("void f() { auto a = b->c(); }");
  verifyFormat("auto a = p->foo();");
  verifyFormat("int a = p->foo();");
  verifyFormat("auto lmbd = [] NOEXCEPT -> int { return 0; };");
}

TEST_F(FormatTest, DeductionGuides) {
  verifyFormat("template <class T> A(const T &, const T &) -> A<T &>;");
  verifyFormat("template <class T> explicit A(T &, T &&) -> A<T>;");
  verifyFormat("template <class... Ts> S(Ts...) -> S<Ts...>;");
  verifyFormat(
      "template <class... T>\n"
      "array(T &&...t) -> array<std::common_type_t<T...>, sizeof...(T)>;");
  verifyFormat("template <class T> A() -> A<decltype(p->foo<3>())>;");
  verifyFormat("template <class T> A() -> A<decltype(foo<traits<1>>)>;");
  verifyFormat("template <class T> A() -> A<sizeof(p->foo<1>)>;");
  verifyFormat("template <class T> A() -> A<(3 < 2)>;");
  verifyFormat("template <class T> A() -> A<((3) < (2))>;");
  verifyFormat("template <class T> x() -> x<1>;");
  verifyFormat("template <class T> explicit x(T &) -> x<1>;");

  verifyFormat("A(const char *) -> A<string &>;");
  verifyFormat("A() -> A<int>;");

  // Ensure not deduction guides.
  verifyFormat("c()->f<int>();");
  verifyFormat("x()->foo<1>;");
  verifyFormat("x = p->foo<3>();");
  verifyFormat("x()->x<1>();");
}

TEST_F(FormatTest, BreaksFunctionDeclarationsWithTrailingTokens) {
  // Avoid breaking before trailing 'const' or other trailing annotations, if
  // they are not function-like.
  FormatStyle Style = getGoogleStyleWithColumns(47);
  verifyFormat("void someLongFunction(\n"
               "    int someLoooooooooooooongParameter) const {\n}",
               getLLVMStyleWithColumns(47));
  verifyFormat("LoooooongReturnType\n"
               "someLoooooooongFunction() const {}",
               getLLVMStyleWithColumns(47));
  verifyFormat("LoooooongReturnType someLoooooooongFunction()\n"
               "    const {}",
               Style);
  verifyFormat("void SomeFunction(aaaaa aaaaaaaaaaaaaaaaaaaa,\n"
               "                  aaaaa aaaaaaaaaaaaaaaaaaaa) OVERRIDE;");
  verifyFormat("void SomeFunction(aaaaa aaaaaaaaaaaaaaaaaaaa,\n"
               "                  aaaaa aaaaaaaaaaaaaaaaaaaa) OVERRIDE FINAL;");
  verifyFormat("void SomeFunction(aaaaa aaaaaaaaaaaaaaaaaaaa,\n"
               "                  aaaaa aaaaaaaaaaaaaaaaaaaa) override final;");
  verifyFormat("virtual void aaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaa aaaa,\n"
               "                   aaaaaaaaaaa aaaaa) const override;");
  verifyGoogleFormat(
      "virtual void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
      "    const override;");

  // Even if the first parameter has to be wrapped.
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) const {}",
               getLLVMStyleWithColumns(46));
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) const {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) override {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) OVERRIDE {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) final {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) FINAL {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int parameter) const override {}",
               Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) const\n"
               "{\n"
               "}",
               Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Whitesmiths;
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) const\n"
               "  {\n"
               "  }",
               Style);

  // Unless these are unknown annotations.
  verifyFormat("void SomeFunction(aaaaaaaaaa aaaaaaaaaaaaaaa,\n"
               "                  aaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    LONG_AND_UGLY_ANNOTATION;");

  // Breaking before function-like trailing annotations is fine to keep them
  // close to their arguments.
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    LOCKS_EXCLUDED(aaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) const\n"
               "    LOCKS_EXCLUDED(aaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) const\n"
               "    LOCKS_EXCLUDED(aaaaaaaaaaaaa) {}");
  verifyGoogleFormat("void aaaaaaaaaaaaaa(aaaaaaaa aaa) override\n"
                     "    AAAAAAAAAAAAAAAAAAAAAAAA(aaaaaaaaaaaaaaa);");
  verifyFormat("SomeFunction([](int i) LOCKS_EXCLUDED(a) {});");

  verifyFormat(
      "void aaaaaaaaaaaaaaaaaa()\n"
      "    __attribute__((aaaaaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaaaaaaaaaaa));");
  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    __attribute__((unused));");
  verifyGoogleFormat(
      "bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    GUARDED_BY(aaaaaaaaaaaa);");
  verifyGoogleFormat(
      "bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    GUARDED_BY(aaaaaaaaaaaa);");
  verifyGoogleFormat(
      "bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa GUARDED_BY(aaaaaaaaaaaa) =\n"
      "    aaaaaaaa::aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyGoogleFormat(
      "bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa GUARDED_BY(aaaaaaaaaaaa) =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaa;");
}

TEST_F(FormatTest, FunctionAnnotations) {
  verifyFormat("DEPRECATED(\"Use NewClass::NewFunction instead.\")\n"
               "int OldFunction(const string &parameter) {}");
  verifyFormat("DEPRECATED(\"Use NewClass::NewFunction instead.\")\n"
               "string OldFunction(const string &parameter) {}");
  verifyFormat("template <typename T>\n"
               "DEPRECATED(\"Use NewClass::NewFunction instead.\")\n"
               "string OldFunction(const string &parameter) {}");

  // Not function annotations.
  verifyFormat("ASSERT(\"aaaaa\") << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                << bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
  verifyFormat("TEST_F(ThisIsATestFixtureeeeeeeeeeeee,\n"
               "       ThisIsATestWithAReallyReallyReallyReallyLongName) {}");
  verifyFormat("MACRO(abc).function() // wrap\n"
               "    << abc;");
  verifyFormat("MACRO(abc)->function() // wrap\n"
               "    << abc;");
  verifyFormat("MACRO(abc)::function() // wrap\n"
               "    << abc;");
}

TEST_F(FormatTest, BreaksDesireably) {
  verifyFormat("if (aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa) ||\n"
               "    aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa) ||\n"
               "    aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa)) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)) {\n"
               "}");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa));");

  verifyFormat(
      "aaaaaaaa(aaaaaaaaaaaaa,\n"
      "         aaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)),\n"
      "         aaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)));");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "    (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat(
      "void f() {\n"
      "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa &&\n"
      "                                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
      "}");
  verifyFormat(
      "aaaaaa(new Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaa));");
  verifyFormat(
      "aaaaaa(aaa, new Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "                aaaaaaaaaaaaaaaaaaaaaaaaaaaaa));");
  verifyFormat(
      "aaaaaa(aaa,\n"
      "       new Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "       aaaa);");
  verifyFormat("aaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  // Indent consistently independent of call expression and unary operator.
  verifyFormat("aaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(\n"
               "    dddddddddddddddddddddddddddddd));");
  verifyFormat("aaaaaaaaaaa(!bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(\n"
               "    dddddddddddddddddddddddddddddd));");
  verifyFormat("aaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbb.ccccccccccccccccc(\n"
               "    dddddddddddddddddddddddddddddd));");

  // This test case breaks on an incorrect memoization, i.e. an optimization not
  // taking into account the StopAt value.
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       aaaaaaaaaaa(aaaaaaaaa) || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       aaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("{\n  {\n    {\n"
               "      Annotation.SpaceRequiredBefore =\n"
               "          Line.Tokens[i - 1].Tok.isNot(tok::l_paren) &&\n"
               "          Line.Tokens[i - 1].Tok.isNot(tok::l_square);\n"
               "    }\n  }\n}");

  // Break on an outer level if there was a break on an inner level.
  verifyFormat("f(g(h(a, // comment\n"
               "      b, c),\n"
               "    d, e),\n"
               "  x, y);",
               "f(g(h(a, // comment\n"
               "    b, c), d, e), x, y);");

  // Prefer breaking similar line breaks.
  verifyFormat(
      "const int kTrackingOptions = NSTrackingMouseMoved |\n"
      "                             NSTrackingMouseEnteredAndExited |\n"
      "                             NSTrackingActiveAlways;");
}

TEST_F(FormatTest, FormatsDeclarationsOnePerLine) {
  FormatStyle NoBinPacking = getGoogleStyle();
  NoBinPacking.BinPackParameters = false;
  NoBinPacking.BinPackArguments = true;
  verifyFormat("void f() {\n"
               "  f(aaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
               "}",
               NoBinPacking);
  verifyFormat("void f(int aaaaaaaaaaaaaaaaaaaa,\n"
               "       int aaaaaaaaaaaaaaaaaaaa,\n"
               "       int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
               NoBinPacking);

  NoBinPacking.AllowAllParametersOfDeclarationOnNextLine = false;
  verifyFormat("void aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                        vector<int> bbbbbbbbbbbbbbb);",
               NoBinPacking);
  // FIXME: This behavior difference is probably not wanted. However, currently
  // we cannot distinguish BreakBeforeParameter being set because of the wrapped
  // template arguments from BreakBeforeParameter being set because of the
  // one-per-line formatting.
  verifyFormat(
      "void fffffffffff(aaaaaaaaaaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                                             aaaaaaaaaa> aaaaaaaaaa);",
      NoBinPacking);
  verifyFormat(
      "void fffffffffff(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaa>\n"
      "        aaaaaaaaaa);");
}

TEST_F(FormatTest, FormatsOneParameterPerLineIfNecessary) {
  FormatStyle NoBinPacking = getGoogleStyle();
  NoBinPacking.BinPackParameters = false;
  NoBinPacking.BinPackArguments = false;
  verifyFormat("f(aaaaaaaaaaaaaaaaaaaa,\n"
               "  aaaaaaaaaaaaaaaaaaaa,\n"
               "  aaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaa);",
               NoBinPacking);
  verifyFormat("aaaaaaa(aaaaaaaaaaaaa,\n"
               "        aaaaaaaaaaaaa,\n"
               "        aaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa));",
               NoBinPacking);
  verifyFormat(
      "aaaaaaaa(aaaaaaaaaaaaa,\n"
      "         aaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)),\n"
      "         aaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)));",
      NoBinPacking);
  verifyFormat("aaaaaaaaaaaaaaa(aaaaaaaaa, aaaaaaaaa, aaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaaaaaaaaaaaaaa();",
               NoBinPacking);
  verifyFormat("void f() {\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "      aaaaaaaaaa, aaaaaaaaaa, aaaaaaaaaa, aaaaaaaaaaa);\n"
               "}",
               NoBinPacking);

  verifyFormat(
      "aaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "             aaaaaaaaaaaa,\n"
      "             aaaaaaaaaaaa);",
      NoBinPacking);
  verifyFormat(
      "somefunction(someotherFunction(ddddddddddddddddddddddddddddddddddd,\n"
      "                               ddddddddddddddddddddddddddddd),\n"
      "             test);",
      NoBinPacking);

  verifyFormat("std::vector<aaaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaaaaaaaaaaaaaaaaaaa>\n"
               "    aaaaaaaaaaaaaaaaaa;",
               NoBinPacking);
  verifyFormat("a(\"a\"\n"
               "  \"a\",\n"
               "  a);");

  NoBinPacking.AllowAllParametersOfDeclarationOnNextLine = false;
  verifyFormat("void aaaaaaaaaa(aaaaaaaaa,\n"
               "                aaaaaaaaa,\n"
               "                aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               NoBinPacking);
  verifyFormat(
      "void f() {\n"
      "  aaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaa, aaaaaaaaa, aaaaaaaaaaaaaaaaaaaaa)\n"
      "      .aaaaaaa();\n"
      "}",
      NoBinPacking);
  verifyFormat(
      "template <class SomeType, class SomeOtherType>\n"
      "SomeType SomeFunction(SomeType Type, SomeOtherType OtherType) {}",
      NoBinPacking);
}

TEST_F(FormatTest, AdaptiveOnePerLineFormatting) {
  FormatStyle Style = getLLVMStyleWithColumns(15);
  Style.ExperimentalAutoDetectBinPacking = true;
  verifyFormat("aaa(aaaa,\n"
               "    aaaa,\n"
               "    aaaa);\n"
               "aaa(aaaa,\n"
               "    aaaa,\n"
               "    aaaa);",
               "aaa(aaaa,\n" // one-per-line
               "  aaaa,\n"
               "    aaaa  );\n"
               "aaa(aaaa,  aaaa,  aaaa);", // inconclusive
               Style);
  verifyFormat("aaa(aaaa, aaaa,\n"
               "    aaaa);\n"
               "aaa(aaaa, aaaa,\n"
               "    aaaa);",
               "aaa(aaaa,  aaaa,\n" // bin-packed
               "    aaaa  );\n"
               "aaa(aaaa,  aaaa,  aaaa);", // inconclusive
               Style);
}

TEST_F(FormatTest, FormatsBuilderPattern) {
  verifyFormat("return llvm::StringSwitch<Reference::Kind>(name)\n"
               "    .StartsWith(\".eh_frame_hdr\", ORDER_EH_FRAMEHDR)\n"
               "    .StartsWith(\".eh_frame\", ORDER_EH_FRAME)\n"
               "    .StartsWith(\".init\", ORDER_INIT)\n"
               "    .StartsWith(\".fini\", ORDER_FINI)\n"
               "    .StartsWith(\".hash\", ORDER_HASH)\n"
               "    .Default(ORDER_TEXT);");

  verifyFormat("return aaaaaaaaaaaaaaaaa->aaaaa().aaaaaaaaaaaaa().aaaaaa() <\n"
               "       aaaaaaaaaaaaaaa->aaaaa().aaaaaaaaaaaaa().aaaaaa();");
  verifyFormat("aaaaaaa->aaaaaaa\n"
               "    ->aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    ->aaaaaaaa(aaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaa->aaaaaaa\n"
      "    ->aaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "    ->aaaaaaaa(aaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaa()->aaaaaa(bbbbb)->aaaaaaaaaaaaaaaaaaa( // break\n"
      "    aaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaa *aaaaaaaaa =\n"
      "    aaaaaa->aaaaaaaaaaaa()\n"
      "        ->aaaaaaaaaaaaaaaa(\n"
      "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "        ->aaaaaaaaaaaaaaaaa();");
  verifyGoogleFormat(
      "void f() {\n"
      "  someo->Add((new util::filetools::Handler(dir))\n"
      "                 ->OnEvent1(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBA))\n"
      "                 ->OnEvent2(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBB))\n"
      "                 ->OnEvent3(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBC))\n"
      "                 ->OnEvent5(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBD))\n"
      "                 ->OnEvent6(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBE)));\n"
      "}");

  verifyFormat(
      "aaaaaaaaaaa().aaaaaaaaaaa().aaaaaaaaaaa().aaaaaaaaaaa().aaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaaaaaaa.aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaaaaaaa.aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa.aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaaaaa->aaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    ->aaaaaaaaaaaaaae(0)\n"
               "    ->aaaaaaaaaaaaaaa();");

  // Don't linewrap after very short segments.
  verifyFormat("a().aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("aa().aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("aaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaa.aaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .has<bbbbbbbbbbbbbbbbbbbbb>();");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaa.aaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>();");

  // Prefer not to break after empty parentheses.
  verifyFormat("FirstToken->WhitespaceRange.getBegin().getLocWithOffset(\n"
               "    First->LastNewlineOffset);");

  // Prefer not to create "hanging" indents.
  verifyFormat(
      "return !soooooooooooooome_map\n"
      "            .insert(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "            .second;");
  verifyFormat(
      "return aaaaaaaaaaaaaaaa\n"
      "    .aaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa)\n"
      "    .aaaa(aaaaaaaaaaaaaa);");
  // No hanging indent here.
  verifyFormat("aaaaaaaaaaaaaaaa.aaaaaaaaaaaaaa.aaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaaaaaaaaaaaaaa.aaaaaaaaaaaaaa().aaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaaaaaaaaaaaaaaaa.aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               getLLVMStyleWithColumns(60));
  verifyFormat("aaaaaaaaaaaaaaaaaa\n"
               "    .aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               getLLVMStyleWithColumns(59));
  verifyFormat("aaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  // Dont break if only closing statements before member call
  verifyFormat("test() {\n"
               "  ([]() -> {\n"
               "    int b = 32;\n"
               "    return 3;\n"
               "  }).foo();\n"
               "}");
  verifyFormat("test() {\n"
               "  (\n"
               "      []() -> {\n"
               "        int b = 32;\n"
               "        return 3;\n"
               "      },\n"
               "      foo, bar)\n"
               "      .foo();\n"
               "}");
  verifyFormat("test() {\n"
               "  ([]() -> {\n"
               "    int b = 32;\n"
               "    return 3;\n"
               "  })\n"
               "      .foo()\n"
               "      .bar();\n"
               "}");
  verifyFormat("test() {\n"
               "  ([]() -> {\n"
               "    int b = 32;\n"
               "    return 3;\n"
               "  })\n"
               "      .foo(\"aaaaaaaaaaaaaaaaa\"\n"
               "           \"bbbb\");\n"
               "}",
               getLLVMStyleWithColumns(30));
}

TEST_F(FormatTest, BreaksAccordingToOperatorPrecedence) {
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbb && ccccccccccccccccccccccccc) {\n}");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaa or\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbb and cccccccccccccccccccccccc) {\n}");

  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa && bbbbbbbbbbbbbbbbbbbbbbbbb ||\n"
               "    ccccccccccccccccccccccccc) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa and bbbbbbbbbbbbbbbbbbbbbbbb or\n"
               "    ccccccccccccccccccccccccc) {\n}");

  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa || bbbbbbbbbbbbbbbbbbbbbbbbb ||\n"
               "    ccccccccccccccccccccccccc) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa or bbbbbbbbbbbbbbbbbbbbbbbbb or\n"
               "    ccccccccccccccccccccccccc) {\n}");

  verifyFormat(
      "if ((aaaaaaaaaaaaaaaaaaaaaaaaa || bbbbbbbbbbbbbbbbbbbbbbbbb) &&\n"
      "    ccccccccccccccccccccccccc) {\n}");
  verifyFormat(
      "if ((aaaaaaaaaaaaaaaaaaaaaaaaa or bbbbbbbbbbbbbbbbbbbbbbbbb) and\n"
      "    ccccccccccccccccccccccccc) {\n}");

  verifyFormat("return aaaa & AAAAAAAAAAAAAAAAAAAAAAAAAAAAA ||\n"
               "       bbbb & BBBBBBBBBBBBBBBBBBBBBBBBBBBBB ||\n"
               "       cccc & CCCCCCCCCCCCCCCCCCCCCCCCCC ||\n"
               "       dddd & DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD;");
  verifyFormat("return aaaa & AAAAAAAAAAAAAAAAAAAAAAAAAAAAA or\n"
               "       bbbb & BBBBBBBBBBBBBBBBBBBBBBBBBBBBB or\n"
               "       cccc & CCCCCCCCCCCCCCCCCCCCCCCCCC or\n"
               "       dddd & DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD;");

  verifyFormat("if ((aaaaaaaaaa != aaaaaaaaaaaaaaa ||\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaa() >= aaaaaaaaaaaaaaaaaaaa) &&\n"
               "    aaaaaaaaaaaaaaa != aa) {\n}");
  verifyFormat("if ((aaaaaaaaaa != aaaaaaaaaaaaaaa or\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaa() >= aaaaaaaaaaaaaaaaaaaa) and\n"
               "    aaaaaaaaaaaaaaa != aa) {\n}");
}

TEST_F(FormatTest, BreaksAfterAssignments) {
  verifyFormat(
      "unsigned Cost =\n"
      "    TTI.getMemoryOpCost(I->getOpcode(), VectorTy, SI->getAlignment(),\n"
      "                        SI->getPointerAddressSpaceee());");
  verifyFormat(
      "CharSourceRange LineRange = CharSourceRange::getTokenRange(\n"
      "    Line.Tokens.front().Tok.getLo(), Line.Tokens.back().Tok.getLoc());");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaa aaaa = aaaaaaaaaaaaaa(0).aaaa().aaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaa::aaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("unsigned OriginalStartColumn =\n"
               "    SourceMgr.getSpellingColumnNumber(\n"
               "        Current.FormatTok.getStartOfNonWhitespace()) -\n"
               "    1;");
}

TEST_F(FormatTest, ConfigurableBreakAssignmentPenalty) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbb + cccccccccccccccccccccccccc;",
               Style);

  Style.PenaltyBreakAssignment = 20;
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaa = bbbbbbbbbbbbbbbbbbbbbbbbbb +\n"
               "                                 cccccccccccccccccccccccccc;",
               Style);
}

TEST_F(FormatTest, AlignsAfterAssignments) {
  verifyFormat(
      "int Result = aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "Result += aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "          aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "Result >>= aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "           aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "int Result = (aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "              aaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "double LooooooooooooooooooooooooongResult = aaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaaaaa;");
}

TEST_F(FormatTest, AlignsAfterReturn) {
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "       aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "return (aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa >=\n"
      "       aaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat(
      "return (aaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa >=\n"
      "        aaaaaaaaaaaaaaaaaaaaaa());");
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) &&\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("return\n"
               "    // true if code is one of a or b.\n"
               "    code == a || code == b;");
}

TEST_F(FormatTest, AlignsAfterOpenBracket) {
  verifyFormat(
      "void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaa aaaaaaaa,\n"
      "                                                aaaaaaaaa aaaaaaa) {}");
  verifyFormat(
      "SomeLongVariableName->someVeryLongFunctionName(aaaaaaaaaaa aaaaaaaaa,\n"
      "                                               aaaaaaaaaaa aaaaaaaaa);");
  verifyFormat(
      "SomeLongVariableName->someFunction(foooooooo(aaaaaaaaaaaaaaa,\n"
      "                                             aaaaaaaaaaaaaaaaaaaaa));");
  FormatStyle Style = getLLVMStyle();
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaa aaaaaaaa, aaaaaaaaa aaaaaaa) {}",
               Style);
  verifyFormat("SomeLongVariableName->someVeryLongFunctionName(\n"
               "    aaaaaaaaaaa aaaaaaaaa, aaaaaaaaaaa aaaaaaaaa);",
               Style);
  verifyFormat("SomeLongVariableName->someFunction(\n"
               "    foooooooo(aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaa));",
               Style);
  verifyFormat(
      "void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaa aaaaaaaa,\n"
      "    aaaaaaaaa aaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
      Style);
  verifyFormat(
      "SomeLongVariableName->someVeryLongFunctionName(aaaaaaaaaaa aaaaaaaaa,\n"
      "    aaaaaaaaaaa aaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "SomeLongVariableName->someFunction(foooooooo(aaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa));",
      Style);

  verifyFormat("bbbbbbbbbbbb(aaaaaaaaaaaaaaaaaaaaaaaa, //\n"
               "    ccccccc(aaaaaaaaaaaaaaaaa,         //\n"
               "        b));",
               Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  Style.BinPackArguments = false;
  Style.BinPackParameters = false;
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaa aaaaaaaa,\n"
               "    aaaaaaaaa aaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
               Style);
  verifyFormat("SomeLongVariableName->someVeryLongFunctionName(\n"
               "    aaaaaaaaaaa aaaaaaaaa,\n"
               "    aaaaaaaaaaa aaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  verifyFormat("SomeLongVariableName->someFunction(foooooooo(\n"
               "    aaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa));",
               Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa)));",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaa.aaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa)));",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa)),\n"
      "    aaaaaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa)) &&\n"
      "    aaaaaaaaaaaaaaaa);",
      Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;
  Style.BinPackArguments = false;
  Style.BinPackParameters = false;
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaa aaaaaaaa,\n"
               "    aaaaaaaaa aaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               ") {}",
               Style);
  verifyFormat("SomeLongVariableName->someVeryLongFunctionName(\n"
               "    aaaaaaaaaaa aaaaaaaaa,\n"
               "    aaaaaaaaaaa aaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               ");",
               Style);
  verifyFormat("SomeLongVariableName->someFunction(foooooooo(\n"
               "    aaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "));",
               Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa)\n"
               "));",
               Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaa.aaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa)\n"
               "));",
               Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa)\n"
      "    ),\n"
      "    aaaaaaaaaaaaaaaa\n"
      ");",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa)\n"
      "    ) &&\n"
      "    aaaaaaaaaaaaaaaa\n"
      ");",
      Style);
}

TEST_F(FormatTest, ParenthesesAndOperandAlignment) {
  FormatStyle Style = getLLVMStyleWithColumns(40);
  verifyFormat("int a = f(aaaaaaaaaaaaaaaaaaaaaa &&\n"
               "          bbbbbbbbbbbbbbbbbbbbbb);",
               Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_Align;
  Style.AlignOperands = FormatStyle::OAS_DontAlign;
  verifyFormat("int a = f(aaaaaaaaaaaaaaaaaaaaaa &&\n"
               "          bbbbbbbbbbbbbbbbbbbbbb);",
               Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  Style.AlignOperands = FormatStyle::OAS_Align;
  verifyFormat("int a = f(aaaaaaaaaaaaaaaaaaaaaa &&\n"
               "          bbbbbbbbbbbbbbbbbbbbbb);",
               Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  Style.AlignOperands = FormatStyle::OAS_DontAlign;
  verifyFormat("int a = f(aaaaaaaaaaaaaaaaaaaaaa &&\n"
               "    bbbbbbbbbbbbbbbbbbbbbb);",
               Style);
}

TEST_F(FormatTest, BreaksConditionalExpressions) {
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                               ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                               : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaa(aaaaaaaaaa, aaaaaaaa,\n"
      "     aaaaaaaaaaaaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                   : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaa(aaaaaaaaa, aaaaaaaaa,\n"
               "     aaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "             : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa ? aaaa(aaaaaa)\n"
      "                                                    : aaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                    : aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaa ?: aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaa);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "           ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "           : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "           ?: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    ? aaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        : aaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    ? aaaaaaaaaaaaaaa\n"
      "    : aaaaaaaaaaaaaaa;");
  verifyFormat("f(aaaaaaaaaaaaaaaa == // force break\n"
               "          aaaaaaaaa\n"
               "      ? b\n"
               "      : c);");
  verifyFormat("return aaaa == bbbb\n"
               "           // comment\n"
               "           ? aaaa\n"
               "           : bbbb;");
  verifyFormat("unsigned Indent =\n"
               "    format(TheLine.First,\n"
               "           IndentForLevel[TheLine.Level] >= 0\n"
               "               ? IndentForLevel[TheLine.Level]\n"
               "               : TheLine * 2,\n"
               "           TheLine.InPPDirective, PreviousEndOfLineColumn);",
               getLLVMStyleWithColumns(60));
  verifyFormat("bool aaaaaa = aaaaaaaaaaaaa //\n"
               "                  ? aaaaaaaaaaaaaaa\n"
               "                  : bbbbbbbbbbbbbbb //\n"
               "                        ? ccccccccccccccc\n"
               "                        : ddddddddddddddd;");
  verifyFormat("bool aaaaaa = aaaaaaaaaaaaa //\n"
               "                  ? aaaaaaaaaaaaaaa\n"
               "                  : (bbbbbbbbbbbbbbb //\n"
               "                         ? ccccccccccccccc\n"
               "                         : ddddddddddddddd);");
  verifyFormat(
      "int aaaaaaaaaaaaaaaaaaaaaaaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                      ? aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaa\n"
      "                                      : aaaaaaaaaa;");
  verifyFormat(
      "aaaaaa = aaaaaaaaaaaa ? aaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                   : aaaaaaaaaaaaaaaaaaaaaa\n"
      "                      : aaaaaaaaaaaaaaaaaaaaaaaaaaaa;");

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackArguments = false;
  verifyFormat(
      "void f() {\n"
      "  g(aaa,\n"
      "    aaaaaaaaaa == aaaaaaaaaa ? aaaa : aaaaa,\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "        ? aaaaaaaaaaaaaaa\n"
      "        : aaaaaaaaaaaaaaa);\n"
      "}",
      NoBinPacking);
  verifyFormat(
      "void f() {\n"
      "  g(aaa,\n"
      "    aaaaaaaaaa == aaaaaaaaaa ? aaaa : aaaaa,\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "        ?: aaaaaaaaaaaaaaa);\n"
      "}",
      NoBinPacking);

  verifyFormat("SomeFunction(aaaaaaaaaaaaaaaaa,\n"
               "             // comment.\n"
               "             ccccccccccccccccccccccccccccccccccccccc\n"
               "                 ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                 : bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);");

  // Assignments in conditional expressions. Apparently not uncommon :-(.
  verifyFormat("return a != b\n"
               "           // comment\n"
               "           ? a = b\n"
               "           : a = b;");
  verifyFormat("return a != b\n"
               "           // comment\n"
               "           ? a = a != b\n"
               "                     // comment\n"
               "                     ? a = b\n"
               "                     : a\n"
               "           : a;");
  verifyFormat("return a != b\n"
               "           // comment\n"
               "           ? a\n"
               "           : a = a != b\n"
               "                     // comment\n"
               "                     ? a = b\n"
               "                     : a;");

  // Chained conditionals
  FormatStyle Style = getLLVMStyleWithColumns(70);
  Style.AlignOperands = FormatStyle::OAS_Align;
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111\n"
               "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
               "                        : 3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111\n"
               "       : bbbbbbbbbb     ? 2222222222222222\n"
               "                        : 3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaa         ? 1111111111111111\n"
               "       : bbbbbbbbbbbbbbbb ? 2222222222222222\n"
               "                          : 3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111\n"
               "       : bbbbbbbbbbbbbb ? 222222\n"
               "                        : 333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111\n"
               "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
               "       : cccccccccccccc ? 3333333333333333\n"
               "                        : 4444444444444444;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? (aaa ? bbb : ccc)\n"
               "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
               "                        : 3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111\n"
               "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
               "                        : (aaa ? bbb : ccc);",
               Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                                             : cccccccccccccccccc)\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaa        ? (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                                             : cccccccccccccccccc)\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaa        ? a = (aaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                                             : dddddddddddddddddd)\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaa        ? a + (aaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                                             : dddddddddddddddddd)\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaa        ? 1111111111111111\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : a + (aaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                                             : dddddddddddddddddd)",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? 1111111111111111\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                                             : cccccccccccccccccc);",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                           : ccccccccccccccc ? dddddddddddddddddd\n"
      "                                             : eeeeeeeeeeeeeeeeee)\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaaa    ? bbbbbbbbbbbbbbbbbb\n"
      "                           : ccccccccccccccc ? dddddddddddddddddd\n"
      "                                             : eeeeeeeeeeeeeeeeee)\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                           : cccccccccccc    ? dddddddddddddddddd\n"
      "                                             : eeeeeeeeeeeeeeeeee)\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                                             : cccccccccccccccccc\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
      "                          : cccccccccccccccc ? dddddddddddddddddd\n"
      "                                             : eeeeeeeeeeeeeeeeee\n"
      "       : bbbbbbbbbbbbbb ? 2222222222222222\n"
      "                        : 3333333333333333;",
      Style);
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaa\n"
               "           ? (aaaaaaaaaaaaaaaaaa   ? bbbbbbbbbbbbbbbbbb\n"
               "              : cccccccccccccccccc ? dddddddddddddddddd\n"
               "                                   : eeeeeeeeeeeeeeeeee)\n"
               "       : bbbbbbbbbbbbbbbbbbb ? 2222222222222222\n"
               "                             : 3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "           ? aaaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb\n"
               "             : cccccccccccccccc ? dddddddddddddddddd\n"
               "                                : eeeeeeeeeeeeeeeeee\n"
               "       : bbbbbbbbbbbbbbbbbbbbbbb ? 2222222222222222\n"
               "                                 : 3333333333333333;",
               Style);

  Style.AlignOperands = FormatStyle::OAS_DontAlign;
  Style.BreakBeforeTernaryOperators = false;
  // FIXME: Aligning the question marks is weird given DontAlign.
  // Consider disabling this alignment in this case. Also check whether this
  // will render the adjustment from https://reviews.llvm.org/D82199
  // unnecessary.
  verifyFormat("int x = aaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaa :\n"
               "    bbbb                ? cccccccccccccccccc :\n"
               "                          ddddd;",
               Style);

  verifyFormat(
      "MMMMMMMMMMMMMMMMMMMMMMMMMMM = A ?\n"
      "    /*\n"
      "     */\n"
      "    function() {\n"
      "      try {\n"
      "        return JJJJJJJJJJJJJJ(\n"
      "            pppppppppppppppppppppppppppppppppppppppppppppppppp);\n"
      "      }\n"
      "    } :\n"
      "    function() {};",
      "MMMMMMMMMMMMMMMMMMMMMMMMMMM = A ?\n"
      "     /*\n"
      "      */\n"
      "     function() {\n"
      "      try {\n"
      "        return JJJJJJJJJJJJJJ(\n"
      "            pppppppppppppppppppppppppppppppppppppppppppppppppp);\n"
      "      }\n"
      "    } :\n"
      "    function() {};",
      getGoogleStyle(FormatStyle::LK_JavaScript));
}

TEST_F(FormatTest, BreaksConditionalExpressionsAfterOperator) {
  FormatStyle Style = getLLVMStyleWithColumns(70);
  Style.BreakBeforeTernaryOperators = false;
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
      "                               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
      "                               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaa(aaaaaaaaaa, aaaaaaaa,\n"
      "     aaaaaaaaaaaaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
      "                                  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
      "                                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
      Style);
  verifyFormat("aaaa(aaaaaaaa, aaaaaaaaaa,\n"
               "     aaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa ? aaaa(aaaaaa) :\n"
      "                                                      aaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
      "                                      aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaa ?: aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaa);",
      Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  verifyFormat("aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) :\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  verifyFormat("aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?:\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
      "    aaaaaaaaaaaaaaa :\n"
      "    aaaaaaaaaaaaaaa;",
      Style);
  verifyFormat("f(aaaaaaaaaaaaaaaa == // force break\n"
               "          aaaaaaaaa ?\n"
               "      b :\n"
               "      c);",
               Style);
  verifyFormat("unsigned Indent =\n"
               "    format(TheLine.First,\n"
               "           IndentForLevel[TheLine.Level] >= 0 ?\n"
               "               IndentForLevel[TheLine.Level] :\n"
               "               TheLine * 2,\n"
               "           TheLine.InPPDirective, PreviousEndOfLineColumn);",
               Style);
  verifyFormat("bool aaaaaa = aaaaaaaaaaaaa ? //\n"
               "                  aaaaaaaaaaaaaaa :\n"
               "                  bbbbbbbbbbbbbbb ? //\n"
               "                      ccccccccccccccc :\n"
               "                      ddddddddddddddd;",
               Style);
  verifyFormat("bool aaaaaa = aaaaaaaaaaaaa ? //\n"
               "                  aaaaaaaaaaaaaaa :\n"
               "                  (bbbbbbbbbbbbbbb ? //\n"
               "                       ccccccccccccccc :\n"
               "                       ddddddddddddddd);",
               Style);
  verifyFormat("int i = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "            /*bbbbbbbbbbbbbbb=*/bbbbbbbbbbbbbbbbbbbbbbbbb :\n"
               "            ccccccccccccccccccccccccccc;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "           aaaaa :\n"
               "           bbbbbbbbbbbbbbb + cccccccccccccccc;",
               Style);

  // Chained conditionals
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111 :\n"
               "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
               "                          3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111 :\n"
               "       bbbbbbbbbb       ? 2222222222222222 :\n"
               "                          3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaa       ? 1111111111111111 :\n"
               "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
               "                          3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111 :\n"
               "       bbbbbbbbbbbbbbbb ? 222222 :\n"
               "                          333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111 :\n"
               "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
               "       cccccccccccccccc ? 3333333333333333 :\n"
               "                          4444444444444444;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? (aaa ? bbb : ccc) :\n"
               "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
               "                          3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 1111111111111111 :\n"
               "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
               "                          (aaa ? bbb : ccc);",
               Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                                               cccccccccccccccccc) :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaa        ? (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                                               cccccccccccccccccc) :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaa        ? a = (aaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                                               dddddddddddddddddd) :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaa        ? a + (aaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                                               dddddddddddddddddd) :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaa        ? 1111111111111111 :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          a + (aaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                                               dddddddddddddddddd)",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? 1111111111111111 :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                                               cccccccccccccccccc);",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                           ccccccccccccccccc ? dddddddddddddddddd :\n"
      "                                               eeeeeeeeeeeeeeeeee) :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                           ccccccccccccc     ? dddddddddddddddddd :\n"
      "                                               eeeeeeeeeeeeeeeeee) :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? (aaaaaaaaaaaaa     ? bbbbbbbbbbbbbbbbbb :\n"
      "                           ccccccccccccccccc ? dddddddddddddddddd :\n"
      "                                               eeeeeeeeeeeeeeeeee) :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                                               cccccccccccccccccc :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat(
      "return aaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
      "                          cccccccccccccccccc ? dddddddddddddddddd :\n"
      "                                               eeeeeeeeeeeeeeeeee :\n"
      "       bbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
      "                          3333333333333333;",
      Style);
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaa ?\n"
               "           (aaaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
               "            cccccccccccccccccc ? dddddddddddddddddd :\n"
               "                                 eeeeeeeeeeeeeeeeee) :\n"
               "       bbbbbbbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
               "                               3333333333333333;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaa ?\n"
               "           aaaaaaaaaaaaaaaaaaaa ? bbbbbbbbbbbbbbbbbb :\n"
               "           cccccccccccccccccccc ? dddddddddddddddddd :\n"
               "                                  eeeeeeeeeeeeeeeeee :\n"
               "       bbbbbbbbbbbbbbbbbbbbb ? 2222222222222222 :\n"
               "                               3333333333333333;",
               Style);
}

TEST_F(FormatTest, DeclarationsOfMultipleVariables) {
  verifyFormat("bool aaaaaaaaaaaaaaaaa = aaaaaa->aaaaaaaaaaaaaaaaa(),\n"
               "     aaaaaaaaaaa = aaaaaa->aaaaaaaaaaa();");
  verifyFormat("bool a = true, b = false;");

  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaa),\n"
               "     bbbbbbbbbbbbbbbbbbbbbbbbb =\n"
               "         bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(bbbbbbbbbbbbbbbb);");
  verifyFormat(
      "bool aaaaaaaaaaaaaaaaaaaaa =\n"
      "         bbbbbbbbbbbbbbbbbbbbbbbbbbbb && cccccccccccccccccccccccccccc,\n"
      "     d = e && f;");
  verifyFormat("aaaaaaaaa a = aaaaaaaaaaaaaaaaaaaa, b = bbbbbbbbbbbbbbbbbbbb,\n"
               "          c = cccccccccccccccccccc, d = dddddddddddddddddddd;");
  verifyFormat("aaaaaaaaa *a = aaaaaaaaaaaaaaaaaaa, *b = bbbbbbbbbbbbbbbbbbb,\n"
               "          *c = ccccccccccccccccccc, *d = ddddddddddddddddddd;");
  verifyFormat("aaaaaaaaa ***a = aaaaaaaaaaaaaaaaaaa, ***b = bbbbbbbbbbbbbbb,\n"
               "          ***c = ccccccccccccccccccc, ***d = ddddddddddddddd;");

  FormatStyle Style = getGoogleStyle();
  Style.PointerAlignment = FormatStyle::PAS_Left;
  Style.DerivePointerAlignment = false;
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    *aaaaaaaaaaaaaaaaaaaaaaaaaaaaa = aaaaaaaaaaaaaaaaaaa,\n"
               "    *b = bbbbbbbbbbbbbbbbbbb;",
               Style);
  verifyFormat("aaaaaaaaa *a = aaaaaaaaaaaaaaaaaaa, *b = bbbbbbbbbbbbbbbbbbb,\n"
               "          *b = bbbbbbbbbbbbbbbbbbb, *d = ddddddddddddddddddd;",
               Style);
  verifyFormat("vector<int*> a, b;", Style);
  verifyFormat("for (int *p, *q; p != q; p = p->next) {\n}", Style);
  verifyFormat("/*comment*/ for (int *p, *q; p != q; p = p->next) {\n}", Style);
  verifyFormat("if (int *p, *q; p != q) {\n  p = p->next;\n}", Style);
  verifyFormat("/*comment*/ if (int *p, *q; p != q) {\n  p = p->next;\n}",
               Style);
  verifyFormat("switch (int *p, *q; p != q) {\n  default:\n    break;\n}",
               Style);
  verifyFormat(
      "/*comment*/ switch (int *p, *q; p != q) {\n  default:\n    break;\n}",
      Style);

  verifyFormat("if ([](int* p, int* q) {}()) {\n}", Style);
  verifyFormat("for ([](int* p, int* q) {}();;) {\n}", Style);
  verifyFormat("for (; [](int* p, int* q) {}();) {\n}", Style);
  verifyFormat("for (;; [](int* p, int* q) {}()) {\n}", Style);
  verifyFormat("switch ([](int* p, int* q) {}()) {\n  default:\n    break;\n}",
               Style);
}

TEST_F(FormatTest, ConditionalExpressionsInBrackets) {
  verifyFormat("arr[foo ? bar : baz];");
  verifyFormat("f()[foo ? bar : baz];");
  verifyFormat("(a + b)[foo ? bar : baz];");
  verifyFormat("arr[foo ? (4 > 5 ? 4 : 5) : 5 < 5 ? 5 : 7];");
}

TEST_F(FormatTest, AlignsStringLiterals) {
  verifyFormat("loooooooooooooooooooooooooongFunction(\"short literal \"\n"
               "                                      \"short literal\");");
  verifyFormat(
      "looooooooooooooooooooooooongFunction(\n"
      "    \"short literal\"\n"
      "    \"looooooooooooooooooooooooooooooooooooooooooooooooong literal\");");
  verifyFormat("someFunction(\"Always break between multi-line\"\n"
               "             \" string literals\",\n"
               "             also, other, parameters);");
  verifyFormat("fun + \"1243\" /* comment */\n"
               "      \"5678\";",
               "fun + \"1243\" /* comment */\n"
               "    \"5678\";",
               getLLVMStyleWithColumns(28));
  verifyFormat(
      "aaaaaa = \"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaa \"\n"
      "         \"aaaaaaaaaaaaaaaaaaaaa\"\n"
      "         \"aaaaaaaaaaaaaaaa\";",
      "aaaaaa ="
      "\"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaa "
      "aaaaaaaaaaaaaaaaaaaaa\" "
      "\"aaaaaaaaaaaaaaaa\";");
  verifyFormat("a = a + \"a\"\n"
               "        \"a\"\n"
               "        \"a\";");
  verifyFormat("f(\"a\", \"b\"\n"
               "       \"c\");");

  verifyFormat(
      "#define LL_FORMAT \"ll\"\n"
      "printf(\"aaaaa: %d, bbbbbb: %\" LL_FORMAT \"d, cccccccc: %\" LL_FORMAT\n"
      "       \"d, ddddddddd: %\" LL_FORMAT \"d\");");

  verifyFormat("#define A(X)          \\\n"
               "  \"aaaaa\" #X \"bbbbbb\" \\\n"
               "  \"ccccc\"",
               getLLVMStyleWithColumns(23));
  verifyFormat("#define A \"def\"\n"
               "f(\"abc\" A \"ghi\"\n"
               "  \"jkl\");");

  verifyFormat("f(L\"a\"\n"
               "  L\"b\");");
  verifyFormat("#define A(X)            \\\n"
               "  L\"aaaaa\" #X L\"bbbbbb\" \\\n"
               "  L\"ccccc\"",
               getLLVMStyleWithColumns(25));

  verifyFormat("f(@\"a\"\n"
               "  @\"b\");");
  verifyFormat("NSString s = @\"a\"\n"
               "             @\"b\"\n"
               "             @\"c\";");
  verifyFormat("NSString s = @\"a\"\n"
               "              \"b\"\n"
               "              \"c\";");
}

TEST_F(FormatTest, ReturnTypeBreakingStyle) {
  FormatStyle Style = getLLVMStyle();
  // No declarations or definitions should be moved to own line.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_None;
  verifyFormat("class A {\n"
               "  int f() { return 1; }\n"
               "  int g();\n"
               "};\n"
               "int f() { return 1; }\n"
               "int g();",
               Style);

  // All declarations and definitions should have the return type moved to its
  // own line.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_All;
  Style.TypenameMacros = {"LIST"};
  verifyFormat("SomeType\n"
               "funcdecl(LIST(uint64_t));",
               Style);
  verifyFormat("class E {\n"
               "  int\n"
               "  f() {\n"
               "    return 1;\n"
               "  }\n"
               "  int\n"
               "  g();\n"
               "};\n"
               "int\n"
               "f() {\n"
               "  return 1;\n"
               "}\n"
               "int\n"
               "g();",
               Style);

  // Top-level definitions, and no kinds of declarations should have the
  // return type moved to its own line.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_TopLevelDefinitions;
  verifyFormat("class B {\n"
               "  int f() { return 1; }\n"
               "  int g();\n"
               "};\n"
               "int\n"
               "f() {\n"
               "  return 1;\n"
               "}\n"
               "int g();",
               Style);

  // Top-level definitions and declarations should have the return type moved
  // to its own line.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_TopLevel;
  verifyFormat("class C {\n"
               "  int f() { return 1; }\n"
               "  int g();\n"
               "};\n"
               "int\n"
               "f() {\n"
               "  return 1;\n"
               "}\n"
               "int\n"
               "g();",
               Style);

  // All definitions should have the return type moved to its own line, but no
  // kinds of declarations.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_AllDefinitions;
  verifyFormat("class D {\n"
               "  int\n"
               "  f() {\n"
               "    return 1;\n"
               "  }\n"
               "  int g();\n"
               "};\n"
               "int\n"
               "f() {\n"
               "  return 1;\n"
               "}\n"
               "int g();",
               Style);
  verifyFormat("const char *\n"
               "f(void) {\n" // Break here.
               "  return \"\";\n"
               "}\n"
               "const char *bar(void);", // No break here.
               Style);
  verifyFormat("template <class T>\n"
               "T *\n"
               "f(T &c) {\n" // Break here.
               "  return NULL;\n"
               "}\n"
               "template <class T> T *f(T &c);", // No break here.
               Style);
  verifyFormat("class C {\n"
               "  int\n"
               "  operator+() {\n"
               "    return 1;\n"
               "  }\n"
               "  int\n"
               "  operator()() {\n"
               "    return 1;\n"
               "  }\n"
               "};",
               Style);
  verifyFormat("void\n"
               "A::operator()() {}\n"
               "void\n"
               "A::operator>>() {}\n"
               "void\n"
               "A::operator+() {}\n"
               "void\n"
               "A::operator*() {}\n"
               "void\n"
               "A::operator->() {}\n"
               "void\n"
               "A::operator void *() {}\n"
               "void\n"
               "A::operator void &() {}\n"
               "void\n"
               "A::operator void &&() {}\n"
               "void\n"
               "A::operator char *() {}\n"
               "void\n"
               "A::operator[]() {}\n"
               "void\n"
               "A::operator!() {}\n"
               "void\n"
               "A::operator**() {}\n"
               "void\n"
               "A::operator<Foo> *() {}\n"
               "void\n"
               "A::operator<Foo> **() {}\n"
               "void\n"
               "A::operator<Foo> &() {}\n"
               "void\n"
               "A::operator void **() {}",
               Style);
  verifyFormat("constexpr auto\n"
               "operator()() const -> reference {}\n"
               "constexpr auto\n"
               "operator>>() const -> reference {}\n"
               "constexpr auto\n"
               "operator+() const -> reference {}\n"
               "constexpr auto\n"
               "operator*() const -> reference {}\n"
               "constexpr auto\n"
               "operator->() const -> reference {}\n"
               "constexpr auto\n"
               "operator++() const -> reference {}\n"
               "constexpr auto\n"
               "operator void *() const -> reference {}\n"
               "constexpr auto\n"
               "operator void **() const -> reference {}\n"
               "constexpr auto\n"
               "operator void *() const -> reference {}\n"
               "constexpr auto\n"
               "operator void &() const -> reference {}\n"
               "constexpr auto\n"
               "operator void &&() const -> reference {}\n"
               "constexpr auto\n"
               "operator char *() const -> reference {}\n"
               "constexpr auto\n"
               "operator!() const -> reference {}\n"
               "constexpr auto\n"
               "operator[]() const -> reference {}",
               Style);
  verifyFormat("void *operator new(std::size_t s);", // No break here.
               Style);
  verifyFormat("void *\n"
               "operator new(std::size_t s) {}",
               Style);
  verifyFormat("void *\n"
               "operator delete[](void *ptr) {}",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  verifyFormat("const char *\n"
               "f(void)\n" // Break here.
               "{\n"
               "  return \"\";\n"
               "}\n"
               "const char *bar(void);", // No break here.
               Style);
  verifyFormat("template <class T>\n"
               "T *\n"     // Problem here: no line break
               "f(T &c)\n" // Break here.
               "{\n"
               "  return NULL;\n"
               "}\n"
               "template <class T> T *f(T &c);", // No break here.
               Style);
  verifyFormat("int\n"
               "foo(A<bool> a)\n"
               "{\n"
               "  return a;\n"
               "}",
               Style);
  verifyFormat("int\n"
               "foo(A<8> a)\n"
               "{\n"
               "  return a;\n"
               "}",
               Style);
  verifyFormat("int\n"
               "foo(A<B<bool>, 8> a)\n"
               "{\n"
               "  return a;\n"
               "}",
               Style);
  verifyFormat("int\n"
               "foo(A<B<8>, bool> a)\n"
               "{\n"
               "  return a;\n"
               "}",
               Style);
  verifyFormat("int\n"
               "foo(A<B<bool>, bool> a)\n"
               "{\n"
               "  return a;\n"
               "}",
               Style);
  verifyFormat("int\n"
               "foo(A<B<8>, 8> a)\n"
               "{\n"
               "  return a;\n"
               "}",
               Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterFunction = true;
  verifyFormat("int f(i);\n" // No break here.
               "int\n"       // Break here.
               "f(i)\n"
               "{\n"
               "  return i + 1;\n"
               "}\n"
               "int\n" // Break here.
               "f(i)\n"
               "{\n"
               "  return i + 1;\n"
               "};",
               Style);
  verifyFormat("int f(a, b, c);\n" // No break here.
               "int\n"             // Break here.
               "f(a, b, c)\n"      // Break here.
               "short a, b;\n"
               "float c;\n"
               "{\n"
               "  return a + b < c;\n"
               "}\n"
               "int\n"        // Break here.
               "f(a, b, c)\n" // Break here.
               "short a, b;\n"
               "float c;\n"
               "{\n"
               "  return a + b < c;\n"
               "};",
               Style);
  verifyFormat("byte *\n" // Break here.
               "f(a)\n"   // Break here.
               "byte a[];\n"
               "{\n"
               "  return a;\n"
               "}",
               Style);
  verifyFormat("byte *\n"
               "f(a)\n"
               "byte /* K&R C */ a[];\n"
               "{\n"
               "  return a;\n"
               "}\n"
               "byte *\n"
               "g(p)\n"
               "byte /* K&R C */ *p;\n"
               "{\n"
               "  return p;\n"
               "}",
               Style);
  verifyFormat("bool f(int a, int) override;\n"
               "Bar g(int a, Bar) final;\n"
               "Bar h(a, Bar) final;",
               Style);
  verifyFormat("int\n"
               "f(a)",
               Style);
  verifyFormat("bool\n"
               "f(size_t = 0, bool b = false)\n"
               "{\n"
               "  return !b;\n"
               "}",
               Style);

  // The return breaking style doesn't affect:
  // * function and object definitions with attribute-like macros
  verifyFormat("Tttttttttttttttttttttttt ppppppppppppppp\n"
               "    ABSL_GUARDED_BY(mutex) = {};",
               getGoogleStyleWithColumns(40));
  verifyFormat("Tttttttttttttttttttttttt ppppppppppppppp\n"
               "    ABSL_GUARDED_BY(mutex);  // comment",
               getGoogleStyleWithColumns(40));
  verifyFormat("Tttttttttttttttttttttttt ppppppppppppppp\n"
               "    ABSL_GUARDED_BY(mutex1)\n"
               "        ABSL_GUARDED_BY(mutex2);",
               getGoogleStyleWithColumns(40));
  verifyFormat("Tttttt f(int a, int b)\n"
               "    ABSL_GUARDED_BY(mutex1)\n"
               "        ABSL_GUARDED_BY(mutex2);",
               getGoogleStyleWithColumns(40));
  // * typedefs
  verifyGoogleFormat("typedef ATTR(X) char x;");

  Style = getGNUStyle();

  // Test for comments at the end of function declarations.
  verifyFormat("void\n"
               "foo (int a, /*abc*/ int b) // def\n"
               "{\n"
               "}",
               Style);

  verifyFormat("void\n"
               "foo (int a, /* abc */ int b) /* def */\n"
               "{\n"
               "}",
               Style);

  // Definitions that should not break after return type
  verifyFormat("void foo (int a, int b); // def", Style);
  verifyFormat("void foo (int a, int b); /* def */", Style);
  verifyFormat("void foo (int a, int b);", Style);
}

TEST_F(FormatTest, AlwaysBreakBeforeMultilineStrings) {
  FormatStyle NoBreak = getLLVMStyle();
  NoBreak.AlwaysBreakBeforeMultilineStrings = false;
  FormatStyle Break = getLLVMStyle();
  Break.AlwaysBreakBeforeMultilineStrings = true;
  verifyFormat("aaaa = \"bbbb\"\n"
               "       \"cccc\";",
               NoBreak);
  verifyFormat("aaaa =\n"
               "    \"bbbb\"\n"
               "    \"cccc\";",
               Break);
  verifyFormat("aaaa(\"bbbb\"\n"
               "     \"cccc\");",
               NoBreak);
  verifyFormat("aaaa(\n"
               "    \"bbbb\"\n"
               "    \"cccc\");",
               Break);
  verifyFormat("aaaa(qqq, \"bbbb\"\n"
               "          \"cccc\");",
               NoBreak);
  verifyFormat("aaaa(qqq,\n"
               "     \"bbbb\"\n"
               "     \"cccc\");",
               Break);
  verifyFormat("aaaa(qqq,\n"
               "     L\"bbbb\"\n"
               "     L\"cccc\");",
               Break);
  verifyFormat("aaaaa(aaaaaa, aaaaaaa(\"aaaa\"\n"
               "                      \"bbbb\"));",
               Break);
  verifyFormat("string s = someFunction(\n"
               "    \"abc\"\n"
               "    \"abc\");",
               Break);

  // As we break before unary operators, breaking right after them is bad.
  verifyFormat("string foo = abc ? \"x\"\n"
               "                   \"blah blah blah blah blah blah\"\n"
               "                 : \"y\";",
               Break);

  // Don't break if there is no column gain.
  verifyFormat("f(\"aaaa\"\n"
               "  \"bbbb\");",
               Break);

  // Treat literals with escaped newlines like multi-line string literals.
  verifyNoChange("x = \"a\\\n"
                 "b\\\n"
                 "c\";",
                 NoBreak);
  verifyFormat("xxxx =\n"
               "    \"a\\\n"
               "b\\\n"
               "c\";",
               "xxxx = \"a\\\n"
               "b\\\n"
               "c\";",
               Break);

  verifyFormat("NSString *const kString =\n"
               "    @\"aaaa\"\n"
               "    @\"bbbb\";",
               "NSString *const kString = @\"aaaa\"\n"
               "@\"bbbb\";",
               Break);

  Break.ColumnLimit = 0;
  verifyFormat("const char *hello = \"hello llvm\";", Break);
}

TEST_F(FormatTest, AlignsPipes) {
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaa\n"
      "                     << aaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                 << aaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
      "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "llvm::outs() << \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
      "                \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"\n"
      "             << \"ccccccccccccccccccccccccccccccccccccccccccccccccc\";");
  verifyFormat(
      "aaaaaaaa << (aaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                 << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "         << aaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("llvm::errs() << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "             << bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;");
  verifyFormat("llvm::errs() << \"aaaaaaaaaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaaaaaaaaaaa(aaaaaaaa, aaaaaaaaaaa);");
  verifyFormat(
      "llvm::errs() << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "auto Diag = diag() << aaaaaaaaaaaaaaaa(aaaaaaaaaaaa, aaaaaaaaaaaaa,\n"
      "                                       aaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("llvm::outs() << \"aaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaa.aaaaaaaaaaaa(aaa)->aaaaaaaaaaaaaa();");
  verifyFormat("llvm::errs() << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                    aaaaaaaaaaaaaaaaaaaaa)\n"
               "             << aaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("LOG_IF(aaa == //\n"
               "       bbb)\n"
               "    << a << b;");

  // But sometimes, breaking before the first "<<" is desirable.
  verifyFormat("Diag(aaaaaaaaaaaaaaaaaaaa, aaaaaaaa)\n"
               "    << aaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaa);");
  verifyFormat("Diag(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbb)\n"
               "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("SemaRef.Diag(Loc, diag::note_for_range_begin_end)\n"
               "    << BEF << IsTemplate << Description << E->getType();");
  verifyFormat("Diag(aaaaaaaaaaaaaaaaaaaa, aaaaaaaa)\n"
               "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("Diag(aaaaaaaaaaaaaaaaaaaa, aaaaaaaa)\n"
               "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    << aaa;");

  verifyFormat(
      "llvm::errs() << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");

  // Incomplete string literal.
  verifyFormat("llvm::errs() << \"\n"
               "             << a;",
               "llvm::errs() << \"\n<<a;");

  verifyFormat("void f() {\n"
               "  CHECK_EQ(aaaa, (*bbbbbbbbb)->cccccc)\n"
               "      << \"qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\";\n"
               "}");

  // Handle 'endl'.
  verifyFormat("llvm::errs() << aaaaaaaaaaaaaaaaaaaaaa << endl\n"
               "             << bbbbbbbbbbbbbbbbbbbbbb << endl;");
  verifyFormat("llvm::errs() << endl << bbbbbbbbbbbbbbbbbbbbbb << endl;");

  // Handle '\n'.
  verifyFormat("llvm::errs() << aaaaaaaaaaaaaaaaaaaaaa << \"\\n\"\n"
               "             << bbbbbbbbbbbbbbbbbbbbbb << \"\\n\";");
  verifyFormat("llvm::errs() << aaaaaaaaaaaaaaaaaaaaaa << \'\\n\'\n"
               "             << bbbbbbbbbbbbbbbbbbbbbb << \'\\n\';");
  verifyFormat("llvm::errs() << aaaa << \"aaaaaaaaaaaaaaaaaa\\n\"\n"
               "             << bbbb << \"bbbbbbbbbbbbbbbbbb\\n\";");
  verifyFormat("llvm::errs() << \"\\n\" << bbbbbbbbbbbbbbbbbbbbbb << \"\\n\";");
}

TEST_F(FormatTest, KeepStringLabelValuePairsOnALine) {
  verifyFormat("return out << \"somepacket = {\\n\"\n"
               "           << \" aaaaaa = \" << pkt.aaaaaa << \"\\n\"\n"
               "           << \" bbbb = \" << pkt.bbbb << \"\\n\"\n"
               "           << \" cccccc = \" << pkt.cccccc << \"\\n\"\n"
               "           << \" ddd = [\" << pkt.ddd << \"]\\n\"\n"
               "           << \"}\";");

  verifyFormat("llvm::outs() << \"aaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaa\n"
               "             << \"aaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaa\n"
               "             << \"aaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaa;");
  verifyFormat(
      "llvm::outs() << \"aaaaaaaaaaaaaaaaa = \" << aaaaaaaaaaaaaaaaa\n"
      "             << \"bbbbbbbbbbbbbbbbb = \" << bbbbbbbbbbbbbbbbb\n"
      "             << \"ccccccccccccccccc = \" << ccccccccccccccccc\n"
      "             << \"ddddddddddddddddd = \" << ddddddddddddddddd\n"
      "             << \"eeeeeeeeeeeeeeeee = \" << eeeeeeeeeeeeeeeee;");
  verifyFormat("llvm::outs() << aaaaaaaaaaaaaaaaaaaaaaaa << \"=\"\n"
               "             << bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;");
  verifyFormat(
      "void f() {\n"
      "  llvm::outs() << \"aaaaaaaaaaaaaaaaaaaa: \"\n"
      "               << aaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
      "}");

  // Breaking before the first "<<" is generally not desirable.
  verifyFormat(
      "llvm::errs()\n"
      "    << \"aaaaaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << \"aaaaaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << \"aaaaaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << \"aaaaaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
      getLLVMStyleWithColumns(70));
  verifyFormat("llvm::errs() << \"aaaaaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "             << \"aaaaaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "             << \"aaaaaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               getLLVMStyleWithColumns(70));

  verifyFormat("string v = \"aaaaaaaaaaaaaaaa: \" + aaaaaaaaaaaaaaaa +\n"
               "           \"aaaaaaaaaaaaaaaa: \" + aaaaaaaaaaaaaaaa +\n"
               "           \"aaaaaaaaaaaaaaaa: \" + aaaaaaaaaaaaaaaa;");
  verifyFormat("string v = StrCat(\"aaaaaaaaaaaaaaaa: \", aaaaaaaaaaaaaaaa,\n"
               "                  \"aaaaaaaaaaaaaaaa: \", aaaaaaaaaaaaaaaa,\n"
               "                  \"aaaaaaaaaaaaaaaa: \", aaaaaaaaaaaaaaaa);");
  verifyFormat("string v = \"aaaaaaaaaaaaaaaa: \" +\n"
               "           (aaaa + aaaa);",
               getLLVMStyleWithColumns(40));
  verifyFormat("string v = StrCat(\"aaaaaaaaaaaa: \" +\n"
               "                  (aaaaaaa + aaaaa));",
               getLLVMStyleWithColumns(40));
  verifyFormat(
      "string v = StrCat(\"aaaaaaaaaaaaaaaaaaaaaaaaaaa: \",\n"
      "                  SomeFunction(aaaaaaaaaaaa, aaaaaaaa.aaaaaaa),\n"
      "                  bbbbbbbbbbbbbbbbbbbbbbb);");
}

TEST_F(FormatTest, UnderstandsEquals) {
  verifyFormat(
      "aaaaaaaaaaaaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat(
      "if (a) {\n"
      "  f();\n"
      "} else if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
      "}");

  verifyFormat("if (int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "        100000000 + 10000000) {\n}");
}

TEST_F(FormatTest, WrapsAtFunctionCallsIfNecessary) {
  verifyFormat("LoooooooooooooooooooooooooooooooooooooongObject\n"
               "    .looooooooooooooooooooooooooooooooooooooongFunction();");

  verifyFormat("LoooooooooooooooooooooooooooooooooooooongObject\n"
               "    ->looooooooooooooooooooooooooooooooooooooongFunction();");

  verifyFormat(
      "LooooooooooooooooooooooooooooooooongObject->shortFunction(Parameter1,\n"
      "                                                          Parameter2);");

  verifyFormat(
      "ShortObject->shortFunction(\n"
      "    LooooooooooooooooooooooooooooooooooooooooooooooongParameter1,\n"
      "    LooooooooooooooooooooooooooooooooooooooooooooooongParameter2);");

  verifyFormat("loooooooooooooongFunction(\n"
               "    LoooooooooooooongObject->looooooooooooooooongFunction());");

  verifyFormat(
      "function(LoooooooooooooooooooooooooooooooooooongObject\n"
      "             ->loooooooooooooooooooooooooooooooooooooooongFunction());");

  verifyFormat("EXPECT_CALL(SomeObject, SomeFunction(Parameter))\n"
               "    .WillRepeatedly(Return(SomeValue));");
  verifyFormat("void f() {\n"
               "  EXPECT_CALL(SomeObject, SomeFunction(Parameter))\n"
               "      .Times(2)\n"
               "      .WillRepeatedly(Return(SomeValue));\n"
               "}");
  verifyFormat("SomeMap[std::pair(aaaaaaaaaaaa, bbbbbbbbbbbbbbb)].insert(\n"
               "    ccccccccccccccccccccccc);");
  verifyFormat("aaaaa(aaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "          .aaaaa(aaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("void f() {\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa)->aaaaaaaaa());\n"
               "}");
  verifyFormat("aaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaaaaaaaaaaa(aa(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                        aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                        aaaaaaaaaaaaaaaaaaaaaaaaaaa));");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()) {\n"
               "}");

  // Here, it is not necessary to wrap at "." or "->".
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaa) ||\n"
               "    aaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat(
      "aaaaaaaaaaa->aaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaaa->aaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa));");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa().aaaaaaaaaaaaaaaaa());");
  verifyFormat("a->aaaaaa()->aaaaaaaaaaa(aaaaaaaa()->aaaaaa()->aaaaa() *\n"
               "                         aaaaaaaaa()->aaaaaa()->aaaaa());");
  verifyFormat("a->aaaaaa()->aaaaaaaaaaa(aaaaaaaa()->aaaaaa()->aaaaa() ||\n"
               "                         aaaaaaaaa()->aaaaaa()->aaaaa());");

  verifyFormat("aaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    .a();");

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = false;
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaa,\n"
               "                         aaaaaaaaaaaaaaaaaaa,\n"
               "                         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               NoBinPacking);

  // If there is a subsequent call, change to hanging indentation.
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "                         aaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa))\n"
      "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa));");
  verifyFormat("aaaaaaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "                 .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "               .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa());");
}

TEST_F(FormatTest, WrapsTemplateDeclarations) {
  verifyFormat("template <typename T>\n"
               "virtual void loooooooooooongFunction(int Param1, int Param2);");
  verifyFormat("template <typename T>\n"
               "// T should be one of {A, B}.\n"
               "virtual void loooooooooooongFunction(int Param1, int Param2);");
  verifyFormat(
      "template <typename T>\n"
      "using comment_to_xml_conversion = comment_to_xml_conversion<T, int>;");
  verifyFormat("template <typename T>\n"
               "void f(int Paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaram1,\n"
               "       int Paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaram2);");
  verifyFormat(
      "template <typename T>\n"
      "void looooooooooooooooooooongFunction(int Paaaaaaaaaaaaaaaaaaaaram1,\n"
      "                                      int Paaaaaaaaaaaaaaaaaaaaram2);");
  verifyFormat(
      "template <typename T>\n"
      "aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaa<T>::aaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("template <typename T>\n"
               "void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "template <typename T1, typename T2 = char, typename T3 = char,\n"
      "          typename T4 = char>\n"
      "void f();");
  verifyFormat("template <typename aaaaaaaaaaa, typename bbbbbbbbbbbbb,\n"
               "          template <typename> class cccccccccccccccccccccc,\n"
               "          typename ddddddddddddd>\n"
               "class C {};");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa>(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("void f() {\n"
               "  a<aaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaa>(\n"
               "      a(aaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa));\n"
               "}");

  verifyFormat("template <typename T> class C {};");
  verifyFormat("template <typename T> void f();");
  verifyFormat("template <typename T> void f() {}");
  verifyFormat(
      "aaaaaaaaaaaaa<aaaaaaaaaa, aaaaaaaaaaa,\n"
      "              aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "              aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa> *aaaa =\n"
      "    new aaaaaaaaaaaaa<aaaaaaaaaa, aaaaaaaaaaa,\n"
      "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>(\n"
      "        bbbbbbbbbbbbbbbbbbbbbbbb);",
      getLLVMStyleWithColumns(72));
  verifyFormat("static_cast<A< //\n"
               "    B> *>(\n"
               "\n"
               ");",
               "static_cast<A<//\n"
               "    B>*>(\n"
               "\n"
               "    );");
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    const typename aaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaa);");

  FormatStyle AlwaysBreak = getLLVMStyle();
  AlwaysBreak.AlwaysBreakTemplateDeclarations = FormatStyle::BTDS_Yes;
  verifyFormat("template <typename T>\nclass C {};", AlwaysBreak);
  verifyFormat("template <typename T>\nvoid f();", AlwaysBreak);
  verifyFormat("template <typename T>\nvoid f() {}", AlwaysBreak);
  verifyFormat("void aaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                         bbbbbbbbbbbbbbbbbbbbbbbbbbbb>(\n"
               "    ccccccccccccccccccccccccccccccccccccccccccccccc);");
  verifyFormat("template <template <typename> class Fooooooo,\n"
               "          template <typename> class Baaaaaaar>\n"
               "struct C {};",
               AlwaysBreak);
  verifyFormat("template <typename T> // T can be A, B or C.\n"
               "struct C {};",
               AlwaysBreak);
  verifyFormat("template <typename T>\n"
               "C(T) noexcept;",
               AlwaysBreak);
  verifyFormat("template <typename T>\n"
               "ClassName(T) noexcept;",
               AlwaysBreak);
  verifyFormat("template <typename T>\n"
               "POOR_NAME(T) noexcept;",
               AlwaysBreak);
  verifyFormat("template <enum E> class A {\n"
               "public:\n"
               "  E *f();\n"
               "};");

  FormatStyle NeverBreak = getLLVMStyle();
  NeverBreak.AlwaysBreakTemplateDeclarations = FormatStyle::BTDS_No;
  verifyFormat("template <typename T> class C {};", NeverBreak);
  verifyFormat("template <typename T> void f();", NeverBreak);
  verifyFormat("template <typename T> void f() {}", NeverBreak);
  verifyFormat("template <typename T> C(T) noexcept;", NeverBreak);
  verifyFormat("template <typename T> ClassName(T) noexcept;", NeverBreak);
  verifyFormat("template <typename T> POOR_NAME(T) noexcept;", NeverBreak);
  verifyFormat("template <typename T>\nvoid foo(aaaaaaaaaaaaaaaaaaaaaaaaaa "
               "bbbbbbbbbbbbbbbbbbbb) {}",
               NeverBreak);
  verifyFormat("void aaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                         bbbbbbbbbbbbbbbbbbbbbbbbbbbb>(\n"
               "    ccccccccccccccccccccccccccccccccccccccccccccccc);",
               NeverBreak);
  verifyFormat("template <template <typename> class Fooooooo,\n"
               "          template <typename> class Baaaaaaar>\n"
               "struct C {};",
               NeverBreak);
  verifyFormat("template <typename T> // T can be A, B or C.\n"
               "struct C {};",
               NeverBreak);
  verifyFormat("template <enum E> class A {\n"
               "public:\n"
               "  E *f();\n"
               "};",
               NeverBreak);
  NeverBreak.PenaltyBreakTemplateDeclaration = 100;
  verifyFormat("template <typename T> void\nfoo(aaaaaaaaaaaaaaaaaaaaaaaaaa "
               "bbbbbbbbbbbbbbbbbbbb) {}",
               NeverBreak);
}

TEST_F(FormatTest, WrapsTemplateDeclarationsWithComments) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_Cpp);
  Style.ColumnLimit = 60;
  verifyFormat("// Baseline - no comments.\n"
               "template <\n"
               "    typename aaaaaaaaaaaaaaaaaaaaaa<bbbbbbbbbbbb>::value>\n"
               "void f() {}",
               Style);

  verifyFormat("template <\n"
               "    typename aaaaaaaaaa<bbbbbbbbbbbb>::value>  // trailing\n"
               "void f() {}",
               "template <\n"
               "    typename aaaaaaaaaa<bbbbbbbbbbbb>::value> // trailing\n"
               "void f() {}",
               Style);

  verifyFormat(
      "template <\n"
      "    typename aaaaaaaaaa<bbbbbbbbbbbb>::value> /* line */\n"
      "void f() {}",
      "template <typename aaaaaaaaaa<bbbbbbbbbbbb>::value>  /* line */\n"
      "void f() {}",
      Style);

  verifyFormat("template <\n"
               "    typename aaaaaaaaaa<bbbbbbbbbbbb>::value>  // trailing\n"
               "                                               // multiline\n"
               "void f() {}",
               "template <\n"
               "    typename aaaaaaaaaa<bbbbbbbbbbbb>::value> // trailing\n"
               "                                              // multiline\n"
               "void f() {}",
               Style);

  verifyFormat(
      "template <typename aaaaaaaaaa<\n"
      "    bbbbbbbbbbbb>::value>  // trailing loooong\n"
      "void f() {}",
      "template <\n"
      "    typename aaaaaaaaaa<bbbbbbbbbbbb>::value> // trailing loooong\n"
      "void f() {}",
      Style);
}

TEST_F(FormatTest, WrapsTemplateParameters) {
  FormatStyle Style = getLLVMStyle();
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  verifyFormat(
      "template <typename... a> struct q {};\n"
      "extern q<aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa>\n"
      "    y;",
      Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat(
      "template <typename... a> struct r {};\n"
      "extern r<aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa>\n"
      "    y;",
      Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  verifyFormat("template <typename... a> struct s {};\n"
               "extern s<\n"
               "    aaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaa, "
               "aaaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaa, "
               "aaaaaaaaaaaaaaaaaaaaaa>\n"
               "    y;",
               Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat("template <typename... a> struct t {};\n"
               "extern t<\n"
               "    aaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaa, "
               "aaaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaa, "
               "aaaaaaaaaaaaaaaaaaaaaa>\n"
               "    y;",
               Style);
}

TEST_F(FormatTest, WrapsAtNestedNameSpecifiers) {
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa());");

  // FIXME: Should we have the extra indent after the second break?
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");

  verifyFormat(
      "aaaaaaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb::\n"
      "                    cccccccccccccccccccccccccccccccccccccccccccccc());");

  // Breaking at nested name specifiers is generally not desirable.
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("aaaaaaaaaaaaaaaaaa(aaaaaaaa,\n"
               "                   aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
               "                       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                   aaaaaaaaaaaaaaaaaaaaa);",
               getLLVMStyleWithColumns(74));

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
}

TEST_F(FormatTest, UnderstandsTemplateParameters) {
  verifyFormat("A<int> a;");
  verifyFormat("A<A<A<int>>> a;");
  verifyFormat("A<A<A<int, 2>, 3>, 4> a;");
  verifyFormat("bool x = a < 1 || 2 > a;");
  verifyFormat("bool x = 5 < f<int>();");
  verifyFormat("bool x = f<int>() > 5;");
  verifyFormat("bool x = 5 < a<int>::x;");
  verifyFormat("bool x = a < 4 ? a > 2 : false;");
  verifyFormat("bool x = f() ? a < 2 : a > 2;");

  verifyGoogleFormat("A<A<int>> a;");
  verifyGoogleFormat("A<A<A<int>>> a;");
  verifyGoogleFormat("A<A<A<A<int>>>> a;");
  verifyGoogleFormat("A<A<int> > a;");
  verifyGoogleFormat("A<A<A<int> > > a;");
  verifyGoogleFormat("A<A<A<A<int> > > > a;");
  verifyGoogleFormat("A<::A<int>> a;");
  verifyGoogleFormat("A<::A> a;");
  verifyGoogleFormat("A< ::A> a;");
  verifyGoogleFormat("A< ::A<int> > a;");
  verifyFormat("A<A<A<A>>> a;", "A<A<A<A> >> a;", getGoogleStyle());
  verifyFormat("A<A<A<A>>> a;", "A<A<A<A>> > a;", getGoogleStyle());
  verifyFormat("A<::A<int>> a;", "A< ::A<int>> a;", getGoogleStyle());
  verifyFormat("A<::A<int>> a;", "A<::A<int> > a;", getGoogleStyle());
  verifyFormat("auto x = [] { A<A<A<A>>> a; };", "auto x=[]{A<A<A<A> >> a;};",
               getGoogleStyle());

  verifyFormat("A<A<int>> a;", getChromiumStyle(FormatStyle::LK_Cpp));

  // template closer followed by a token that starts with > or =
  verifyFormat("bool b = a<1> > 1;");
  verifyFormat("bool b = a<1> >= 1;");
  verifyFormat("int i = a<1> >> 1;");
  FormatStyle Style = getLLVMStyle();
  Style.SpaceBeforeAssignmentOperators = false;
  verifyFormat("bool b= a<1> == 1;", Style);
  verifyFormat("a<int> = 1;", Style);
  verifyFormat("a<int> >>= 1;", Style);

  verifyFormat("test < a | b >> c;");
  verifyFormat("test<test<a | b>> c;");
  verifyFormat("test >> a >> b;");
  verifyFormat("test << a >> b;");

  verifyFormat("f<int>();");
  verifyFormat("template <typename T> void f() {}");
  verifyFormat("struct A<std::enable_if<sizeof(T2) < sizeof(int32)>::type>;");
  verifyFormat("struct A<std::enable_if<sizeof(T2) ? sizeof(int32) : "
               "sizeof(char)>::type>;");
  verifyFormat("template <class T> struct S<std::is_arithmetic<T>{}> {};");
  verifyFormat("f(a.operator()<A>());");
  verifyFormat("f(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "      .template operator()<A>());",
               getLLVMStyleWithColumns(35));
  verifyFormat("bool_constant<a && noexcept(f())>");
  verifyFormat("bool_constant<a || noexcept(f())>");

  verifyFormat("if (std::tuple_size_v<T> > 0)");

  // Not template parameters.
  verifyFormat("return a < b && c > d;");
  verifyFormat("a < 0 ? b : a > 0 ? c : d;");
  verifyFormat("ratio{-1, 2} < ratio{-1, 3} == -1 / 3 > -1 / 2;");
  verifyFormat("void f() {\n"
               "  while (a < b && c > d) {\n"
               "  }\n"
               "}");
  verifyFormat("template <typename... Types>\n"
               "typename enable_if<0 < sizeof...(Types)>::type Foo() {}");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaa >> aaaaa);",
               getLLVMStyleWithColumns(60));
  verifyFormat("static_assert(is_convertible<A &&, B>::value, \"AAA\");");
  verifyFormat("Constructor(A... a) : a_(X<A>{std::forward<A>(a)}...) {}");
  verifyFormat("< < < < < < < < < < < < < < < < < < < < < < < < < < < < < <");
  verifyFormat("some_templated_type<decltype([](int i) { return i; })>");

  verifyFormat("#define FOO(typeName, realClass)                           \\\n"
               "  { #typeName, foo<FooType>(new foo<realClass>(#typeName)) }",
               getLLVMStyleWithColumns(60));
}

TEST_F(FormatTest, UnderstandsShiftOperators) {
  verifyFormat("if (i < x >> 1)");
  verifyFormat("while (i < x >> 1)");
  verifyFormat("for (unsigned i = 0; i < i; ++i, v = v >> 1)");
  verifyFormat("for (unsigned i = 0; i < x >> 1; ++i, v = v >> 1)");
  verifyFormat(
      "for (std::vector<int>::iterator i = 0; i < x >> 1; ++i, v = v >> 1)");
  verifyFormat("Foo.call<Bar<Function>>()");
  verifyFormat("if (Foo.call<Bar<Function>>() == 0)");
  verifyFormat("for (std::vector<std::pair<int>>::iterator i = 0; i < x >> 1; "
               "++i, v = v >> 1)");
  verifyFormat("if (w<u<v<x>>, 1>::t)");
}

TEST_F(FormatTest, BitshiftOperatorWidth) {
  verifyFormat("int a = 1 << 2; /* foo\n"
               "                   bar */",
               "int    a=1<<2;  /* foo\n"
               "                   bar */");

  verifyFormat("int b = 256 >> 1; /* foo\n"
               "                     bar */",
               "int  b  =256>>1 ;  /* foo\n"
               "                      bar */");
}

TEST_F(FormatTest, UnderstandsBinaryOperators) {
  verifyFormat("COMPARE(a, ==, b);");
  verifyFormat("auto s = sizeof...(Ts) - 1;");
}

TEST_F(FormatTest, UnderstandsPointersToMembers) {
  verifyFormat("int A::*x;");
  verifyFormat("int (S::*func)(void *);");
  verifyFormat("void f() { int (S::*func)(void *); }");
  verifyFormat("typedef bool *(Class::*Member)() const;");
  verifyFormat("void f() {\n"
               "  (a->*f)();\n"
               "  a->*x;\n"
               "  (a.*f)();\n"
               "  ((*a).*f)();\n"
               "  a.*x;\n"
               "}");
  verifyFormat("void f() {\n"
               "  (a->*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)(\n"
               "      aaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);\n"
               "}");
  verifyFormat(
      "(aaaaaaaaaa->*bbbbbbb)(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa));");
  FormatStyle Style = getLLVMStyle();
  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("typedef bool* (Class::*Member)() const;", Style);
}

TEST_F(FormatTest, UnderstandsUnaryOperators) {
  verifyFormat("int a = -2;");
  verifyFormat("f(-1, -2, -3);");
  verifyFormat("a[-1] = 5;");
  verifyFormat("int a = 5 + -2;");
  verifyFormat("if (i == -1) {\n}");
  verifyFormat("if (i != -1) {\n}");
  verifyFormat("if (i > -1) {\n}");
  verifyFormat("if (i < -1) {\n}");
  verifyFormat("++(a->f());");
  verifyFormat("--(a->f());");
  verifyFormat("(a->f())++;");
  verifyFormat("a[42]++;");
  verifyFormat("if (!(a->f())) {\n}");
  verifyFormat("if (!+i) {\n}");
  verifyFormat("~&a;");
  verifyFormat("for (x = 0; -10 < x; --x) {\n}");
  verifyFormat("sizeof -x");
  verifyFormat("sizeof +x");
  verifyFormat("sizeof *x");
  verifyFormat("sizeof &x");
  verifyFormat("delete +x;");
  verifyFormat("co_await +x;");
  verifyFormat("case *x:");
  verifyFormat("case &x:");

  verifyFormat("a-- > b;");
  verifyFormat("b ? -a : c;");
  verifyFormat("n * sizeof char16;");
  verifyGoogleFormat("n * alignof char16;");
  verifyFormat("sizeof(char);");
  verifyGoogleFormat("alignof(char);");

  verifyFormat("return -1;");
  verifyFormat("throw -1;");
  verifyFormat("switch (a) {\n"
               "case -1:\n"
               "  break;\n"
               "}");
  verifyFormat("#define X -1");
  verifyFormat("#define X -kConstant");

  verifyFormat("const NSPoint kBrowserFrameViewPatternOffset = {-5, +3};");
  verifyFormat("const NSPoint kBrowserFrameViewPatternOffset = {+5, -3};");

  verifyFormat("int a = /* confusing comment */ -1;");
  // FIXME: The space after 'i' is wrong, but hopefully, this is a rare case.
  verifyFormat("int a = i /* confusing comment */++;");

  verifyFormat("co_yield -1;");
  verifyFormat("co_return -1;");

  // Check that * is not treated as a binary operator when we set
  // PointerAlignment as PAS_Left after a keyword and not a declaration.
  FormatStyle PASLeftStyle = getLLVMStyle();
  PASLeftStyle.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("co_return *a;", PASLeftStyle);
  verifyFormat("co_await *a;", PASLeftStyle);
  verifyFormat("co_yield *a", PASLeftStyle);
  verifyFormat("return *a;", PASLeftStyle);
}

TEST_F(FormatTest, DoesNotIndentRelativeToUnaryOperators) {
  verifyFormat("if (!aaaaaaaaaa( // break\n"
               "        aaaaa)) {\n"
               "}");
  verifyFormat("aaaaaaaaaa(!aaaaaaaaaa( // break\n"
               "    aaaaa));");
  verifyFormat("*aaa = aaaaaaa( // break\n"
               "    bbbbbb);");
}

TEST_F(FormatTest, UnderstandsOverloadedOperators) {
  verifyFormat("bool operator<();");
  verifyFormat("bool operator>();");
  verifyFormat("bool operator=();");
  verifyFormat("bool operator==();");
  verifyFormat("bool operator!=();");
  verifyFormat("int operator+();");
  verifyFormat("int operator++();");
  verifyFormat("int operator++(int) volatile noexcept;");
  verifyFormat("bool operator,();");
  verifyFormat("bool operator();");
  verifyFormat("bool operator()();");
  verifyFormat("bool operator[]();");
  verifyFormat("operator bool();");
  verifyFormat("operator int();");
  verifyFormat("operator void *();");
  verifyFormat("operator SomeType<int>();");
  verifyFormat("operator SomeType<int, int>();");
  verifyFormat("operator SomeType<SomeType<int>>();");
  verifyFormat("operator< <>();");
  verifyFormat("operator<< <>();");
  verifyFormat("< <>");

  verifyFormat("void *operator new(std::size_t size);");
  verifyFormat("void *operator new[](std::size_t size);");
  verifyFormat("void operator delete(void *ptr);");
  verifyFormat("void operator delete[](void *ptr);");
  verifyFormat("template <typename AAAAAAA, typename BBBBBBB>\n"
               "AAAAAAA operator/(const AAAAAAA &a, BBBBBBB &b);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaa operator,(\n"
               "    aaaaaaaaaaaaaaaaaaaaa &aaaaaaaaaaaaaaaaaaaaaaaaaa) const;");

  verifyFormat(
      "ostream &operator<<(ostream &OutputStream,\n"
      "                    SomeReallyLongType WithSomeReallyLongValue);");
  verifyFormat("bool operator<(const aaaaaaaaaaaaaaaaaaaaa &left,\n"
               "               const aaaaaaaaaaaaaaaaaaaaa &right) {\n"
               "  return left.group < right.group;\n"
               "}");
  verifyFormat("SomeType &operator=(const SomeType &S);");
  verifyFormat("f.template operator()<int>();");

  verifyGoogleFormat("operator void*();");
  verifyGoogleFormat("operator SomeType<SomeType<int>>();");
  verifyGoogleFormat("operator ::A();");

  verifyFormat("using A::operator+;");
  verifyFormat("inline A operator^(const A &lhs, const A &rhs) {}\n"
               "int i;");

  // Calling an operator as a member function.
  verifyFormat("void f() { a.operator*(); }");
  verifyFormat("void f() { a.operator*(b & b); }");
  verifyFormat("void f() { a->operator&(a * b); }");
  verifyFormat("void f() { NS::a.operator+(*b * *b); }");
  // TODO: Calling an operator as a non-member function is hard to distinguish.
  // https://llvm.org/PR50629
  // verifyFormat("void f() { operator*(a & a); }");
  // verifyFormat("void f() { operator&(a, b * b); }");

  verifyFormat("void f() { return operator()(x) * b; }");
  verifyFormat("void f() { return operator[](x) * b; }");
  verifyFormat("void f() { return operator\"\"_a(x) * b; }");
  verifyFormat("void f() { return operator\"\" _a(x) * b; }");
  verifyFormat("void f() { return operator\"\"s(x) * b; }");
  verifyFormat("void f() { return operator\"\" s(x) * b; }");
  verifyFormat("void f() { return operator\"\"if(x) * b; }");

  verifyFormat("::operator delete(foo);");
  verifyFormat("::operator new(n * sizeof(foo));");
  verifyFormat("foo() { ::operator delete(foo); }");
  verifyFormat("foo() { ::operator new(n * sizeof(foo)); }");
}

TEST_F(FormatTest, SpaceBeforeTemplateCloser) {
  verifyFormat("C<&operator- > minus;");
  verifyFormat("C<&operator> > gt;");
  verifyFormat("C<&operator>= > ge;");
  verifyFormat("C<&operator<= > le;");
  verifyFormat("C<&operator< <X>> lt;");
}

TEST_F(FormatTest, UnderstandsFunctionRefQualification) {
  verifyFormat("void A::b() && {}");
  verifyFormat("void A::b() && noexcept {}");
  verifyFormat("Deleted &operator=(const Deleted &) & = default;");
  verifyFormat("Deleted &operator=(const Deleted &) && = delete;");
  verifyFormat("Deleted &operator=(const Deleted &) & noexcept = default;");
  verifyFormat("SomeType MemberFunction(const Deleted &) & = delete;");
  verifyFormat("SomeType MemberFunction(const Deleted &) && = delete;");
  verifyFormat("Deleted &operator=(const Deleted &) &;");
  verifyFormat("Deleted &operator=(const Deleted &) &&;");
  verifyFormat("SomeType MemberFunction(const Deleted &) &;");
  verifyFormat("SomeType MemberFunction(const Deleted &) &&;");
  verifyFormat("SomeType MemberFunction(const Deleted &) && {}");
  verifyFormat("SomeType MemberFunction(const Deleted &) && final {}");
  verifyFormat("SomeType MemberFunction(const Deleted &) && override {}");
  verifyFormat("SomeType MemberFunction(const Deleted &) && noexcept {}");
  verifyFormat("void Fn(T const &) const &;");
  verifyFormat("void Fn(T const volatile &&) const volatile &&;");
  verifyFormat("void Fn(T const volatile &&) const volatile && noexcept;");
  verifyGoogleFormat("template <typename T>\n"
                     "void F(T) && = delete;");
  verifyFormat("template <typename T> void operator=(T) &;");
  verifyFormat("template <typename T> void operator=(T) const &;");
  verifyFormat("template <typename T> void operator=(T) & noexcept;");
  verifyFormat("template <typename T> void operator=(T) & = default;");
  verifyFormat("template <typename T> void operator=(T) &&;");
  verifyFormat("template <typename T> void operator=(T) && = delete;");
  verifyFormat("template <typename T> void operator=(T) & {}");
  verifyFormat("template <typename T> void operator=(T) && {}");

  FormatStyle AlignLeft = getLLVMStyle();
  AlignLeft.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("void A::b() && {}", AlignLeft);
  verifyFormat("void A::b() && noexcept {}", AlignLeft);
  verifyFormat("Deleted& operator=(const Deleted&) & = default;", AlignLeft);
  verifyFormat("Deleted& operator=(const Deleted&) & noexcept = default;",
               AlignLeft);
  verifyFormat("SomeType MemberFunction(const Deleted&) & = delete;",
               AlignLeft);
  verifyFormat("Deleted& operator=(const Deleted&) &;", AlignLeft);
  verifyFormat("SomeType MemberFunction(const Deleted&) &;", AlignLeft);
  verifyFormat("auto Function(T t) & -> void {}", AlignLeft);
  verifyFormat("auto Function(T... t) & -> void {}", AlignLeft);
  verifyFormat("auto Function(T) & -> void {}", AlignLeft);
  verifyFormat("auto Function(T) & -> void;", AlignLeft);
  verifyFormat("void Fn(T const&) const&;", AlignLeft);
  verifyFormat("void Fn(T const volatile&&) const volatile&&;", AlignLeft);
  verifyFormat("void Fn(T const volatile&&) const volatile&& noexcept;",
               AlignLeft);
  verifyFormat("template <typename T> void operator=(T) &;", AlignLeft);
  verifyFormat("template <typename T> void operator=(T) const&;", AlignLeft);
  verifyFormat("template <typename T> void operator=(T) & noexcept;",
               AlignLeft);
  verifyFormat("template <typename T> void operator=(T) & = default;",
               AlignLeft);
  verifyFormat("template <typename T> void operator=(T) &&;", AlignLeft);
  verifyFormat("template <typename T> void operator=(T) && = delete;",
               AlignLeft);
  verifyFormat("template <typename T> void operator=(T) & {}", AlignLeft);
  verifyFormat("template <typename T> void operator=(T) && {}", AlignLeft);

  FormatStyle AlignMiddle = getLLVMStyle();
  AlignMiddle.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("void A::b() && {}", AlignMiddle);
  verifyFormat("void A::b() && noexcept {}", AlignMiddle);
  verifyFormat("Deleted & operator=(const Deleted &) & = default;",
               AlignMiddle);
  verifyFormat("Deleted & operator=(const Deleted &) & noexcept = default;",
               AlignMiddle);
  verifyFormat("SomeType MemberFunction(const Deleted &) & = delete;",
               AlignMiddle);
  verifyFormat("Deleted & operator=(const Deleted &) &;", AlignMiddle);
  verifyFormat("SomeType MemberFunction(const Deleted &) &;", AlignMiddle);
  verifyFormat("auto Function(T t) & -> void {}", AlignMiddle);
  verifyFormat("auto Function(T... t) & -> void {}", AlignMiddle);
  verifyFormat("auto Function(T) & -> void {}", AlignMiddle);
  verifyFormat("auto Function(T) & -> void;", AlignMiddle);
  verifyFormat("void Fn(T const &) const &;", AlignMiddle);
  verifyFormat("void Fn(T const volatile &&) const volatile &&;", AlignMiddle);
  verifyFormat("void Fn(T const volatile &&) const volatile && noexcept;",
               AlignMiddle);
  verifyFormat("template <typename T> void operator=(T) &;", AlignMiddle);
  verifyFormat("template <typename T> void operator=(T) const &;", AlignMiddle);
  verifyFormat("template <typename T> void operator=(T) & noexcept;",
               AlignMiddle);
  verifyFormat("template <typename T> void operator=(T) & = default;",
               AlignMiddle);
  verifyFormat("template <typename T> void operator=(T) &&;", AlignMiddle);
  verifyFormat("template <typename T> void operator=(T) && = delete;",
               AlignMiddle);
  verifyFormat("template <typename T> void operator=(T) & {}", AlignMiddle);
  verifyFormat("template <typename T> void operator=(T) && {}", AlignMiddle);

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInParens = FormatStyle::SIPO_Custom;
  Spaces.SpacesInParensOptions = {};
  Spaces.SpacesInParensOptions.InCStyleCasts = true;
  verifyFormat("Deleted &operator=(const Deleted &) & = default;", Spaces);
  verifyFormat("SomeType MemberFunction(const Deleted &) & = delete;", Spaces);
  verifyFormat("Deleted &operator=(const Deleted &) &;", Spaces);
  verifyFormat("SomeType MemberFunction(const Deleted &) &;", Spaces);

  Spaces.SpacesInParensOptions.InCStyleCasts = false;
  Spaces.SpacesInParensOptions.Other = true;
  verifyFormat("Deleted &operator=( const Deleted & ) & = default;", Spaces);
  verifyFormat("SomeType MemberFunction( const Deleted & ) & = delete;",
               Spaces);
  verifyFormat("Deleted &operator=( const Deleted & ) &;", Spaces);
  verifyFormat("SomeType MemberFunction( const Deleted & ) &;", Spaces);

  FormatStyle BreakTemplate = getLLVMStyle();
  BreakTemplate.AlwaysBreakTemplateDeclarations = FormatStyle::BTDS_Yes;

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  int &foo(const std::string &str) & noexcept {}\n"
               "};",
               BreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  int &foo(const std::string &str) && noexcept {}\n"
               "};",
               BreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  int &foo(const std::string &str) const & noexcept {}\n"
               "};",
               BreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  int &foo(const std::string &str) const & noexcept {}\n"
               "};",
               BreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  auto foo(const std::string &str) && noexcept -> int & {}\n"
               "};",
               BreakTemplate);

  FormatStyle AlignLeftBreakTemplate = getLLVMStyle();
  AlignLeftBreakTemplate.AlwaysBreakTemplateDeclarations =
      FormatStyle::BTDS_Yes;
  AlignLeftBreakTemplate.PointerAlignment = FormatStyle::PAS_Left;

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  int& foo(const std::string& str) & noexcept {}\n"
               "};",
               AlignLeftBreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  int& foo(const std::string& str) && noexcept {}\n"
               "};",
               AlignLeftBreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  int& foo(const std::string& str) const& noexcept {}\n"
               "};",
               AlignLeftBreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  int& foo(const std::string& str) const&& noexcept {}\n"
               "};",
               AlignLeftBreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  auto foo(const std::string& str) && noexcept -> int& {}\n"
               "};",
               AlignLeftBreakTemplate);

  // The `&` in `Type&` should not be confused with a trailing `&` of
  // DEPRECATED(reason) member function.
  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  DEPRECATED(reason)\n"
               "  Type &foo(arguments) {}\n"
               "};",
               BreakTemplate);

  verifyFormat("struct f {\n"
               "  template <class T>\n"
               "  DEPRECATED(reason)\n"
               "  Type& foo(arguments) {}\n"
               "};",
               AlignLeftBreakTemplate);

  verifyFormat("void (*foopt)(int) = &func;");

  FormatStyle DerivePointerAlignment = getLLVMStyle();
  DerivePointerAlignment.DerivePointerAlignment = true;
  // There's always a space between the function and its trailing qualifiers.
  // This isn't evidence for PAS_Right (or for PAS_Left).
  std::string Prefix = "void a() &;\n"
                       "void b() &;\n";
  verifyFormat(Prefix + "int* x;", DerivePointerAlignment);
  verifyFormat(Prefix + "int *x;", DerivePointerAlignment);
  // Same if the function is an overloaded operator, and with &&.
  Prefix = "void operator()() &&;\n"
           "void operator()() &&;\n";
  verifyFormat(Prefix + "int* x;", DerivePointerAlignment);
  verifyFormat(Prefix + "int *x;", DerivePointerAlignment);
  // However a space between cv-qualifiers and ref-qualifiers *is* evidence.
  Prefix = "void a() const &;\n"
           "void b() const &;\n";
  verifyFormat(Prefix + "int *x;", Prefix + "int* x;", DerivePointerAlignment);
}

TEST_F(FormatTest, PointerAlignmentFallback) {
  FormatStyle Style = getLLVMStyle();
  Style.DerivePointerAlignment = true;

  const StringRef Code("int* p;\n"
                       "int *q;\n"
                       "int * r;");

  EXPECT_EQ(Style.PointerAlignment, FormatStyle::PAS_Right);
  verifyFormat("int *p;\n"
               "int *q;\n"
               "int *r;",
               Code, Style);

  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("int* p;\n"
               "int* q;\n"
               "int* r;",
               Code, Style);

  Style.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("int * p;\n"
               "int * q;\n"
               "int * r;",
               Code, Style);
}

TEST_F(FormatTest, UnderstandsNewAndDelete) {
  verifyFormat("void f() {\n"
               "  A *a = new A;\n"
               "  A *a = new (placement) A;\n"
               "  delete a;\n"
               "  delete (A *)a;\n"
               "}");
  verifyFormat("new (aaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaa))\n"
               "    typename aaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    new (aaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaa))\n"
               "        typename aaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("delete[] h->p;");
  verifyFormat("delete[] (void *)p;");

  verifyFormat("void operator delete(void *foo) ATTRIB;");
  verifyFormat("void operator new(void *foo) ATTRIB;");
  verifyFormat("void operator delete[](void *foo) ATTRIB;");
  verifyFormat("void operator delete(void *ptr) noexcept;");

  verifyFormat("void new(link p);\n"
               "void delete(link p);",
               "void new (link p);\n"
               "void delete (link p);");
}

TEST_F(FormatTest, UnderstandsUsesOfStarAndAmp) {
  verifyFormat("int *f(int *a) {}");
  verifyFormat("int main(int argc, char **argv) {}");
  verifyFormat("Test::Test(int b) : a(b * b) {}");
  verifyIndependentOfContext("f(a, *a);");
  verifyFormat("void g() { f(*a); }");
  verifyIndependentOfContext("int a = b * 10;");
  verifyIndependentOfContext("int a = 10 * b;");
  verifyIndependentOfContext("int a = b * c;");
  verifyIndependentOfContext("int a += b * c;");
  verifyIndependentOfContext("int a -= b * c;");
  verifyIndependentOfContext("int a *= b * c;");
  verifyIndependentOfContext("int a /= b * c;");
  verifyIndependentOfContext("int a = *b;");
  verifyIndependentOfContext("int a = *b * c;");
  verifyIndependentOfContext("int a = b * *c;");
  verifyIndependentOfContext("int a = b * (10);");
  verifyIndependentOfContext("S << b * (10);");
  verifyIndependentOfContext("return 10 * b;");
  verifyIndependentOfContext("return *b * *c;");
  verifyIndependentOfContext("return a & ~b;");
  verifyIndependentOfContext("f(b ? *c : *d);");
  verifyIndependentOfContext("int a = b ? *c : *d;");
  verifyIndependentOfContext("*b = a;");
  verifyIndependentOfContext("a * ~b;");
  verifyIndependentOfContext("a * !b;");
  verifyIndependentOfContext("a * +b;");
  verifyIndependentOfContext("a * -b;");
  verifyIndependentOfContext("a * ++b;");
  verifyIndependentOfContext("a * --b;");
  verifyIndependentOfContext("a[4] * b;");
  verifyIndependentOfContext("a[a * a] = 1;");
  verifyIndependentOfContext("f() * b;");
  verifyIndependentOfContext("a * [self dostuff];");
  verifyIndependentOfContext("int x = a * (a + b);");
  verifyIndependentOfContext("(a *)(a + b);");
  verifyIndependentOfContext("*(int *)(p & ~3UL) = 0;");
  verifyIndependentOfContext("int *pa = (int *)&a;");
  verifyIndependentOfContext("return sizeof(int **);");
  verifyIndependentOfContext("return sizeof(int ******);");
  verifyIndependentOfContext("return (int **&)a;");
  verifyIndependentOfContext("f((*PointerToArray)[10]);");
  verifyFormat("void f(Type (*parameter)[10]) {}");
  verifyFormat("void f(Type (&parameter)[10]) {}");
  verifyGoogleFormat("return sizeof(int**);");
  verifyIndependentOfContext("Type **A = static_cast<Type **>(P);");
  verifyGoogleFormat("Type** A = static_cast<Type**>(P);");
  verifyFormat("auto a = [](int **&, int ***) {};");
  verifyFormat("auto PointerBinding = [](const char *S) {};");
  verifyFormat("typedef typeof(int(int, int)) *MyFunc;");
  verifyFormat("[](const decltype(*a) &value) {}");
  verifyFormat("[](const typeof(*a) &value) {}");
  verifyFormat("[](const _Atomic(a *) &value) {}");
  verifyFormat("[](const __underlying_type(a) &value) {}");
  verifyFormat("decltype(a * b) F();");
  verifyFormat("typeof(a * b) F();");
  verifyFormat("#define MACRO() [](A *a) { return 1; }");
  verifyFormat("Constructor() : member([](A *a, B *b) {}) {}");
  verifyIndependentOfContext("typedef void (*f)(int *a);");
  verifyIndependentOfContext("typedef void (*f)(Type *a);");
  verifyIndependentOfContext("int i{a * b};");
  verifyIndependentOfContext("aaa && aaa->f();");
  verifyIndependentOfContext("int x = ~*p;");
  verifyFormat("Constructor() : a(a), area(width * height) {}");
  verifyFormat("Constructor() : a(a), area(a, width * height) {}");
  verifyGoogleFormat("MACRO Constructor(const int& i) : a(a), b(b) {}");
  verifyFormat("void f() { f(a, c * d); }");
  verifyFormat("void f() { f(new a(), c * d); }");
  verifyFormat("void f(const MyOverride &override);");
  verifyFormat("void f(const MyFinal &final);");
  verifyIndependentOfContext("bool a = f() && override.f();");
  verifyIndependentOfContext("bool a = f() && final.f();");

  verifyIndependentOfContext("InvalidRegions[*R] = 0;");

  verifyIndependentOfContext("A<int *> a;");
  verifyIndependentOfContext("A<int **> a;");
  verifyIndependentOfContext("A<int *, int *> a;");
  verifyIndependentOfContext("A<int *[]> a;");
  verifyIndependentOfContext(
      "const char *const p = reinterpret_cast<const char *const>(q);");
  verifyIndependentOfContext("A<int **, int **> a;");
  verifyIndependentOfContext("void f(int *a = d * e, int *b = c * d);");
  verifyFormat("for (char **a = b; *a; ++a) {\n}");
  verifyFormat("for (; a && b;) {\n}");
  verifyFormat("bool foo = true && [] { return false; }();");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa, *aaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyGoogleFormat("int const* a = &b;");
  verifyGoogleFormat("**outparam = 1;");
  verifyGoogleFormat("*outparam = a * b;");
  verifyGoogleFormat("int main(int argc, char** argv) {}");
  verifyGoogleFormat("A<int*> a;");
  verifyGoogleFormat("A<int**> a;");
  verifyGoogleFormat("A<int*, int*> a;");
  verifyGoogleFormat("A<int**, int**> a;");
  verifyGoogleFormat("f(b ? *c : *d);");
  verifyGoogleFormat("int a = b ? *c : *d;");
  verifyGoogleFormat("Type* t = **x;");
  verifyGoogleFormat("Type* t = *++*x;");
  verifyGoogleFormat("*++*x;");
  verifyGoogleFormat("Type* t = const_cast<T*>(&*x);");
  verifyGoogleFormat("Type* t = x++ * y;");
  verifyGoogleFormat(
      "const char* const p = reinterpret_cast<const char* const>(q);");
  verifyGoogleFormat("void f(int i = 0, SomeType** temps = NULL);");
  verifyGoogleFormat("void f(Bar* a = nullptr, Bar* b);");
  verifyGoogleFormat("template <typename T>\n"
                     "void f(int i = 0, SomeType** temps = NULL);");

  FormatStyle Left = getLLVMStyle();
  Left.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("x = *a(x) = *a(y);", Left);
  verifyFormat("for (;; *a = b) {\n}", Left);
  verifyFormat("return *this += 1;", Left);
  verifyFormat("throw *x;", Left);
  verifyFormat("delete *x;", Left);
  verifyFormat("typedef typeof(int(int, int))* MyFuncPtr;", Left);
  verifyFormat("[](const decltype(*a)* ptr) {}", Left);
  verifyFormat("[](const typeof(*a)* ptr) {}", Left);
  verifyFormat("[](const _Atomic(a*)* ptr) {}", Left);
  verifyFormat("[](const __underlying_type(a)* ptr) {}", Left);
  verifyFormat("typedef typeof /*comment*/ (int(int, int))* MyFuncPtr;", Left);
  verifyFormat("auto x(A&&, B&&, C&&) -> D;", Left);
  verifyFormat("auto x = [](A&&, B&&, C&&) -> D {};", Left);
  verifyFormat("template <class T> X(T&&, T&&, T&&) -> X<T>;", Left);

  verifyIndependentOfContext("a = *(x + y);");
  verifyIndependentOfContext("a = &(x + y);");
  verifyIndependentOfContext("*(x + y).call();");
  verifyIndependentOfContext("&(x + y)->call();");
  verifyFormat("void f() { &(*I).first; }");

  verifyIndependentOfContext("f(b * /* confusing comment */ ++c);");
  verifyFormat("f(* /* confusing comment */ foo);");
  verifyFormat("void (* /*deleter*/)(const Slice &key, void *value)");
  verifyFormat("void foo(int * // this is the first paramters\n"
               "         ,\n"
               "         int second);");
  verifyFormat("double term = a * // first\n"
               "              b;");
  verifyFormat(
      "int *MyValues = {\n"
      "    *A, // Operator detection might be confused by the '{'\n"
      "    *BB // Operator detection might be confused by previous comment\n"
      "};");

  verifyIndependentOfContext("if (int *a = &b)");
  verifyIndependentOfContext("if (int &a = *b)");
  verifyIndependentOfContext("if (a & b[i])");
  verifyIndependentOfContext("if constexpr (a & b[i])");
  verifyIndependentOfContext("if CONSTEXPR (a & b[i])");
  verifyIndependentOfContext("if (a * (b * c))");
  verifyIndependentOfContext("if constexpr (a * (b * c))");
  verifyIndependentOfContext("if CONSTEXPR (a * (b * c))");
  verifyIndependentOfContext("if (a::b::c::d & b[i])");
  verifyIndependentOfContext("if (*b[i])");
  verifyIndependentOfContext("if (int *a = (&b))");
  verifyIndependentOfContext("while (int *a = &b)");
  verifyIndependentOfContext("while (a * (b * c))");
  verifyIndependentOfContext("size = sizeof *a;");
  verifyIndependentOfContext("if (a && (b = c))");
  verifyFormat("void f() {\n"
               "  for (const int &v : Values) {\n"
               "  }\n"
               "}");
  verifyFormat("for (int i = a * a; i < 10; ++i) {\n}");
  verifyFormat("for (int i = 0; i < a * a; ++i) {\n}");
  verifyGoogleFormat("for (int i = 0; i * 2 < z; i *= 2) {\n}");

  verifyFormat("#define A (!a * b)");
  verifyFormat("#define MACRO     \\\n"
               "  int *i = a * b; \\\n"
               "  void f(a *b);",
               getLLVMStyleWithColumns(19));

  verifyIndependentOfContext("A = new SomeType *[Length];");
  verifyIndependentOfContext("A = new SomeType *[Length]();");
  verifyIndependentOfContext("T **t = new T *;");
  verifyIndependentOfContext("T **t = new T *();");
  verifyGoogleFormat("A = new SomeType*[Length]();");
  verifyGoogleFormat("A = new SomeType*[Length];");
  verifyGoogleFormat("T** t = new T*;");
  verifyGoogleFormat("T** t = new T*();");

  verifyFormat("STATIC_ASSERT((a & b) == 0);");
  verifyFormat("STATIC_ASSERT(0 == (a & b));");
  verifyFormat("template <bool a, bool b> "
               "typename t::if<x && y>::type f() {}");
  verifyFormat("template <int *y> f() {}");
  verifyFormat("vector<int *> v;");
  verifyFormat("vector<int *const> v;");
  verifyFormat("vector<int *const **const *> v;");
  verifyFormat("vector<int *volatile> v;");
  verifyFormat("vector<a *_Nonnull> v;");
  verifyFormat("vector<a *_Nullable> v;");
  verifyFormat("vector<a *_Null_unspecified> v;");
  verifyFormat("vector<a *__ptr32> v;");
  verifyFormat("vector<a *__ptr64> v;");
  verifyFormat("vector<a *__capability> v;");
  FormatStyle TypeMacros = getLLVMStyle();
  TypeMacros.TypenameMacros = {"LIST"};
  verifyFormat("vector<LIST(uint64_t)> v;", TypeMacros);
  verifyFormat("vector<LIST(uint64_t) *> v;", TypeMacros);
  verifyFormat("vector<LIST(uint64_t) **> v;", TypeMacros);
  verifyFormat("vector<LIST(uint64_t) *attr> v;", TypeMacros);
  verifyFormat("vector<A(uint64_t) * attr> v;", TypeMacros); // multiplication

  FormatStyle CustomQualifier = getLLVMStyle();
  // Add identifiers that should not be parsed as a qualifier by default.
  CustomQualifier.AttributeMacros.push_back("__my_qualifier");
  CustomQualifier.AttributeMacros.push_back("_My_qualifier");
  CustomQualifier.AttributeMacros.push_back("my_other_qualifier");
  verifyFormat("vector<a * __my_qualifier> parse_as_multiply;");
  verifyFormat("vector<a *__my_qualifier> v;", CustomQualifier);
  verifyFormat("vector<a * _My_qualifier> parse_as_multiply;");
  verifyFormat("vector<a *_My_qualifier> v;", CustomQualifier);
  verifyFormat("vector<a * my_other_qualifier> parse_as_multiply;");
  verifyFormat("vector<a *my_other_qualifier> v;", CustomQualifier);
  verifyFormat("vector<a * _NotAQualifier> v;");
  verifyFormat("vector<a * __not_a_qualifier> v;");
  verifyFormat("vector<a * b> v;");
  verifyFormat("foo<b && false>();");
  verifyFormat("foo<b & 1>();");
  verifyFormat("foo<b & (1)>();");
  verifyFormat("foo<b & (~0)>();");
  verifyFormat("foo<b & (true)>();");
  verifyFormat("foo<b & ((1))>();");
  verifyFormat("foo<b & (/*comment*/ 1)>();");
  verifyFormat("decltype(*::std::declval<const T &>()) void F();");
  verifyFormat("typeof(*::std::declval<const T &>()) void F();");
  verifyFormat("_Atomic(*::std::declval<const T &>()) void F();");
  verifyFormat("__underlying_type(*::std::declval<const T &>()) void F();");
  verifyFormat(
      "template <class T, class = typename std::enable_if<\n"
      "                       std::is_integral<T>::value &&\n"
      "                       (sizeof(T) > 1 || sizeof(T) < 8)>::type>\n"
      "void F();",
      getLLVMStyleWithColumns(70));
  verifyFormat("template <class T,\n"
               "          class = typename std::enable_if<\n"
               "              std::is_integral<T>::value &&\n"
               "              (sizeof(T) > 1 || sizeof(T) < 8)>::type,\n"
               "          class U>\n"
               "void F();",
               getLLVMStyleWithColumns(70));
  verifyFormat(
      "template <class T,\n"
      "          class = typename ::std::enable_if<\n"
      "              ::std::is_array<T>{} && ::std::is_array<T>{}>::type>\n"
      "void F();",
      getGoogleStyleWithColumns(68));

  FormatStyle Style = getLLVMStyle();
  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("struct {\n"
               "}* ptr;",
               Style);
  verifyFormat("union {\n"
               "}* ptr;",
               Style);
  verifyFormat("class {\n"
               "}* ptr;",
               Style);
  // Don't confuse a multiplication after a brace-initialized expression with
  // a class pointer.
  verifyFormat("int i = int{42} * 34;", Style);
  verifyFormat("struct {\n"
               "}&& ptr = {};",
               Style);
  verifyFormat("union {\n"
               "}&& ptr = {};",
               Style);
  verifyFormat("class {\n"
               "}&& ptr = {};",
               Style);
  verifyFormat("bool b = 3 == int{3} && true;");

  Style.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("struct {\n"
               "} * ptr;",
               Style);
  verifyFormat("union {\n"
               "} * ptr;",
               Style);
  verifyFormat("class {\n"
               "} * ptr;",
               Style);
  verifyFormat("struct {\n"
               "} && ptr = {};",
               Style);
  verifyFormat("union {\n"
               "} && ptr = {};",
               Style);
  verifyFormat("class {\n"
               "} && ptr = {};",
               Style);

  Style.PointerAlignment = FormatStyle::PAS_Right;
  verifyFormat("struct {\n"
               "} *ptr;",
               Style);
  verifyFormat("union {\n"
               "} *ptr;",
               Style);
  verifyFormat("class {\n"
               "} *ptr;",
               Style);
  verifyFormat("struct {\n"
               "} &&ptr = {};",
               Style);
  verifyFormat("union {\n"
               "} &&ptr = {};",
               Style);
  verifyFormat("class {\n"
               "} &&ptr = {};",
               Style);

  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("delete[] *ptr;", Style);
  verifyFormat("delete[] **ptr;", Style);
  verifyFormat("delete[] *(ptr);", Style);

  verifyIndependentOfContext("MACRO(int *i);");
  verifyIndependentOfContext("MACRO(auto *a);");
  verifyIndependentOfContext("MACRO(const A *a);");
  verifyIndependentOfContext("MACRO(_Atomic(A) *a);");
  verifyIndependentOfContext("MACRO(decltype(A) *a);");
  verifyIndependentOfContext("MACRO(typeof(A) *a);");
  verifyIndependentOfContext("MACRO(__underlying_type(A) *a);");
  verifyIndependentOfContext("MACRO(A *const a);");
  verifyIndependentOfContext("MACRO(A *restrict a);");
  verifyIndependentOfContext("MACRO(A *__restrict__ a);");
  verifyIndependentOfContext("MACRO(A *__restrict a);");
  verifyIndependentOfContext("MACRO(A *volatile a);");
  verifyIndependentOfContext("MACRO(A *__volatile a);");
  verifyIndependentOfContext("MACRO(A *__volatile__ a);");
  verifyIndependentOfContext("MACRO(A *_Nonnull a);");
  verifyIndependentOfContext("MACRO(A *_Nullable a);");
  verifyIndependentOfContext("MACRO(A *_Null_unspecified a);");
  verifyIndependentOfContext("MACRO(A *__attribute__((foo)) a);");
  verifyIndependentOfContext("MACRO(A *__attribute((foo)) a);");
  verifyIndependentOfContext("MACRO(A *[[clang::attr]] a);");
  verifyIndependentOfContext("MACRO(A *[[clang::attr(\"foo\")]] a);");
  verifyIndependentOfContext("MACRO(A *__ptr32 a);");
  verifyIndependentOfContext("MACRO(A *__ptr64 a);");
  verifyIndependentOfContext("MACRO(A *__capability);");
  verifyIndependentOfContext("MACRO(A &__capability);");
  verifyFormat("MACRO(A *__my_qualifier);");               // type declaration
  verifyFormat("void f() { MACRO(A * __my_qualifier); }"); // multiplication
  // If we add __my_qualifier to AttributeMacros it should always be parsed as
  // a type declaration:
  verifyFormat("MACRO(A *__my_qualifier);", CustomQualifier);
  verifyFormat("void f() { MACRO(A *__my_qualifier); }", CustomQualifier);
  // Also check that TypenameMacros prevents parsing it as multiplication:
  verifyIndependentOfContext("MACRO(LIST(uint64_t) * a);"); // multiplication
  verifyIndependentOfContext("MACRO(LIST(uint64_t) *a);", TypeMacros); // type

  verifyIndependentOfContext("MACRO('0' <= c && c <= '9');");
  verifyFormat("void f() { f(float{1}, a * a); }");
  verifyFormat("void f() { f(float(1), a * a); }");

  verifyFormat("f((void (*)(int))g);");
  verifyFormat("f((void (&)(int))g);");
  verifyFormat("f((void (^)(int))g);");

  // FIXME: Is there a way to make this work?
  // verifyIndependentOfContext("MACRO(A *a);");
  verifyFormat("MACRO(A &B);");
  verifyFormat("MACRO(A *B);");
  verifyFormat("void f() { MACRO(A * B); }");
  verifyFormat("void f() { MACRO(A & B); }");

  // This lambda was mis-formatted after D88956 (treating it as a binop):
  verifyFormat("auto x = [](const decltype(x) &ptr) {};");
  verifyFormat("auto x = [](const decltype(x) *ptr) {};");
  verifyFormat("#define lambda [](const decltype(x) &ptr) {}");
  verifyFormat("#define lambda [](const decltype(x) *ptr) {}");

  verifyFormat("DatumHandle const *operator->() const { return input_; }");
  verifyFormat("return options != nullptr && operator==(*options);");

  verifyFormat("#define OP(x)                                    \\\n"
               "  ostream &operator<<(ostream &s, const A &a) {  \\\n"
               "    return s << a.DebugString();                 \\\n"
               "  }",
               "#define OP(x) \\\n"
               "  ostream &operator<<(ostream &s, const A &a) { \\\n"
               "    return s << a.DebugString(); \\\n"
               "  }",
               getLLVMStyleWithColumns(50));

  verifyFormat("#define FOO             \\\n"
               "  void foo() {          \\\n"
               "    operator+(a * b);   \\\n"
               "  }",
               getLLVMStyleWithColumns(25));

  // FIXME: We cannot handle this case yet; we might be able to figure out that
  // foo<x> d > v; doesn't make sense.
  verifyFormat("foo<a<b && c> d> v;");

  FormatStyle PointerMiddle = getLLVMStyle();
  PointerMiddle.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("delete *x;", PointerMiddle);
  verifyFormat("int * x;", PointerMiddle);
  verifyFormat("int *[] x;", PointerMiddle);
  verifyFormat("template <int * y> f() {}", PointerMiddle);
  verifyFormat("int * f(int * a) {}", PointerMiddle);
  verifyFormat("int main(int argc, char ** argv) {}", PointerMiddle);
  verifyFormat("Test::Test(int b) : a(b * b) {}", PointerMiddle);
  verifyFormat("A<int *> a;", PointerMiddle);
  verifyFormat("A<int **> a;", PointerMiddle);
  verifyFormat("A<int *, int *> a;", PointerMiddle);
  verifyFormat("A<int *[]> a;", PointerMiddle);
  verifyFormat("A = new SomeType *[Length]();", PointerMiddle);
  verifyFormat("A = new SomeType *[Length];", PointerMiddle);
  verifyFormat("T ** t = new T *;", PointerMiddle);

  // Member function reference qualifiers aren't binary operators.
  verifyFormat("string // break\n"
               "operator()() & {}");
  verifyFormat("string // break\n"
               "operator()() && {}");
  verifyGoogleFormat("template <typename T>\n"
                     "auto x() & -> int {}");

  // Should be binary operators when used as an argument expression (overloaded
  // operator invoked as a member function).
  verifyFormat("void f() { a.operator()(a * a); }");
  verifyFormat("void f() { a->operator()(a & a); }");
  verifyFormat("void f() { a.operator()(*a & *a); }");
  verifyFormat("void f() { a->operator()(*a * *a); }");

  verifyFormat("int operator()(T (&&)[N]) { return 1; }");
  verifyFormat("int operator()(T (&)[N]) { return 0; }");

  verifyFormat("val1 & val2;");
  verifyFormat("val1 & val2 & val3;");
  verifyFormat("class c {\n"
               "  void func(type &a) { a & member; }\n"
               "  anotherType &member;\n"
               "}");
}

TEST_F(FormatTest, UnderstandsAttributes) {
  verifyFormat("SomeType s __attribute__((unused)) (InitValue);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa __attribute__((unused))\n"
               "aaaaaaaaaaaaaaaaaaaaaaa(int i);");
  verifyFormat("__attribute__((nodebug)) ::qualified_type f();");
  FormatStyle AfterType = getLLVMStyle();
  AfterType.AlwaysBreakAfterReturnType = FormatStyle::RTBS_All;
  verifyFormat("__attribute__((nodebug)) void\n"
               "foo() {}",
               AfterType);
  verifyFormat("__unused void\n"
               "foo() {}",
               AfterType);

  FormatStyle CustomAttrs = getLLVMStyle();
  CustomAttrs.AttributeMacros.push_back("__unused");
  CustomAttrs.AttributeMacros.push_back("__attr1");
  CustomAttrs.AttributeMacros.push_back("__attr2");
  CustomAttrs.AttributeMacros.push_back("no_underscore_attr");
  verifyFormat("vector<SomeType *__attribute((foo))> v;");
  verifyFormat("vector<SomeType *__attribute__((foo))> v;");
  verifyFormat("vector<SomeType * __not_attribute__((foo))> v;");
  // Check that it is parsed as a multiplication without AttributeMacros and
  // as a pointer qualifier when we add __attr1/__attr2 to AttributeMacros.
  verifyFormat("vector<SomeType * __attr1> v;");
  verifyFormat("vector<SomeType __attr1 *> v;");
  verifyFormat("vector<SomeType __attr1 *const> v;");
  verifyFormat("vector<SomeType __attr1 * __attr2> v;");
  verifyFormat("vector<SomeType *__attr1> v;", CustomAttrs);
  verifyFormat("vector<SomeType *__attr2> v;", CustomAttrs);
  verifyFormat("vector<SomeType *no_underscore_attr> v;", CustomAttrs);
  verifyFormat("vector<SomeType __attr1 *> v;", CustomAttrs);
  verifyFormat("vector<SomeType __attr1 *const> v;", CustomAttrs);
  verifyFormat("vector<SomeType __attr1 *__attr2> v;", CustomAttrs);
  verifyFormat("vector<SomeType __attr1 *no_underscore_attr> v;", CustomAttrs);

  // Check that these are not parsed as function declarations:
  CustomAttrs.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  CustomAttrs.BreakBeforeBraces = FormatStyle::BS_Allman;
  verifyFormat("SomeType s(InitValue);", CustomAttrs);
  verifyFormat("SomeType s{InitValue};", CustomAttrs);
  verifyFormat("SomeType *__unused s(InitValue);", CustomAttrs);
  verifyFormat("SomeType *__unused s{InitValue};", CustomAttrs);
  verifyFormat("SomeType s __unused(InitValue);", CustomAttrs);
  verifyFormat("SomeType s __unused{InitValue};", CustomAttrs);
  verifyFormat("SomeType *__capability s(InitValue);", CustomAttrs);
  verifyFormat("SomeType *__capability s{InitValue};", CustomAttrs);
}

TEST_F(FormatTest, UnderstandsPointerQualifiersInCast) {
  // Check that qualifiers on pointers don't break parsing of casts.
  verifyFormat("x = (foo *const)*v;");
  verifyFormat("x = (foo *volatile)*v;");
  verifyFormat("x = (foo *restrict)*v;");
  verifyFormat("x = (foo *__attribute__((foo)))*v;");
  verifyFormat("x = (foo *_Nonnull)*v;");
  verifyFormat("x = (foo *_Nullable)*v;");
  verifyFormat("x = (foo *_Null_unspecified)*v;");
  verifyFormat("x = (foo *_Nonnull)*v;");
  verifyFormat("x = (foo *[[clang::attr]])*v;");
  verifyFormat("x = (foo *[[clang::attr(\"foo\")]])*v;");
  verifyFormat("x = (foo *__ptr32)*v;");
  verifyFormat("x = (foo *__ptr64)*v;");
  verifyFormat("x = (foo *__capability)*v;");

  // Check that we handle multiple trailing qualifiers and skip them all to
  // determine that the expression is a cast to a pointer type.
  FormatStyle LongPointerRight = getLLVMStyleWithColumns(999);
  FormatStyle LongPointerLeft = getLLVMStyleWithColumns(999);
  LongPointerLeft.PointerAlignment = FormatStyle::PAS_Left;
  StringRef AllQualifiers =
      "const volatile restrict __attribute__((foo)) _Nonnull _Null_unspecified "
      "_Nonnull [[clang::attr]] __ptr32 __ptr64 __capability";
  verifyFormat(("x = (foo *" + AllQualifiers + ")*v;").str(), LongPointerRight);
  verifyFormat(("x = (foo* " + AllQualifiers + ")*v;").str(), LongPointerLeft);

  // Also check that address-of is not parsed as a binary bitwise-and:
  verifyFormat("x = (foo *const)&v;");
  verifyFormat(("x = (foo *" + AllQualifiers + ")&v;").str(), LongPointerRight);
  verifyFormat(("x = (foo* " + AllQualifiers + ")&v;").str(), LongPointerLeft);

  // Check custom qualifiers:
  FormatStyle CustomQualifier = getLLVMStyleWithColumns(999);
  CustomQualifier.AttributeMacros.push_back("__my_qualifier");
  verifyFormat("x = (foo * __my_qualifier) * v;"); // not parsed as qualifier.
  verifyFormat("x = (foo *__my_qualifier)*v;", CustomQualifier);
  verifyFormat(("x = (foo *" + AllQualifiers + " __my_qualifier)*v;").str(),
               CustomQualifier);
  verifyFormat(("x = (foo *" + AllQualifiers + " __my_qualifier)&v;").str(),
               CustomQualifier);

  // Check that unknown identifiers result in binary operator parsing:
  verifyFormat("x = (foo * __unknown_qualifier) * v;");
  verifyFormat("x = (foo * __unknown_qualifier) & v;");
}

TEST_F(FormatTest, UnderstandsSquareAttributes) {
  verifyFormat("SomeType s [[unused]] (InitValue);");
  verifyFormat("SomeType s [[gnu::unused]] (InitValue);");
  verifyFormat("SomeType s [[using gnu: unused]] (InitValue);");
  verifyFormat("[[gsl::suppress(\"clang-tidy-check-name\")]] void f() {}");
  verifyFormat("void f() [[deprecated(\"so sorry\")]];");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    [[unused]] aaaaaaaaaaaaaaaaaaaaaaa(int i);");
  verifyFormat("[[nodiscard]] bool f() { return false; }");
  verifyFormat("class [[nodiscard]] f {\npublic:\n  f() {}\n}");
  verifyFormat("class [[deprecated(\"so sorry\")]] f {\npublic:\n  f() {}\n}");
  verifyFormat("class [[gnu::unused]] f {\npublic:\n  f() {}\n}");
  verifyFormat("[[nodiscard]] ::qualified_type f();");

  // Make sure we do not mistake attributes for array subscripts.
  verifyFormat("int a() {}\n"
               "[[unused]] int b() {}");
  verifyFormat("NSArray *arr;\n"
               "arr[[Foo() bar]];");

  // On the other hand, we still need to correctly find array subscripts.
  verifyFormat("int a = std::vector<int>{1, 2, 3}[0];");

  // Make sure that we do not mistake Objective-C method inside array literals
  // as attributes, even if those method names are also keywords.
  verifyFormat("@[ [foo bar] ];");
  verifyFormat("@[ [NSArray class] ];");
  verifyFormat("@[ [foo enum] ];");

  verifyFormat("template <typename T> [[nodiscard]] int a() { return 1; }");

  // Make sure we do not parse attributes as lambda introducers.
  FormatStyle MultiLineFunctions = getLLVMStyle();
  MultiLineFunctions.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  verifyFormat("[[unused]] int b() {\n"
               "  return 42;\n"
               "}",
               MultiLineFunctions);
}

TEST_F(FormatTest, AttributeClass) {
  FormatStyle Style = getChromiumStyle(FormatStyle::LK_Cpp);
  verifyFormat("class S {\n"
               "  S(S&&) = default;\n"
               "};",
               Style);
  verifyFormat("class [[nodiscard]] S {\n"
               "  S(S&&) = default;\n"
               "};",
               Style);
  verifyFormat("class __attribute((maybeunused)) S {\n"
               "  S(S&&) = default;\n"
               "};",
               Style);
  verifyFormat("struct S {\n"
               "  S(S&&) = default;\n"
               "};",
               Style);
  verifyFormat("struct [[nodiscard]] S {\n"
               "  S(S&&) = default;\n"
               "};",
               Style);
}

TEST_F(FormatTest, AttributesAfterMacro) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("MACRO;\n"
               "__attribute__((maybe_unused)) int foo() {\n"
               "  //...\n"
               "}");

  verifyFormat("MACRO;\n"
               "[[nodiscard]] int foo() {\n"
               "  //...\n"
               "}");

  verifyNoChange("MACRO\n\n"
                 "__attribute__((maybe_unused)) int foo() {\n"
                 "  //...\n"
                 "}");

  verifyNoChange("MACRO\n\n"
                 "[[nodiscard]] int foo() {\n"
                 "  //...\n"
                 "}");
}

TEST_F(FormatTest, AttributePenaltyBreaking) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("void ABCDEFGH::ABCDEFGHIJKLMN(\n"
               "    [[maybe_unused]] const shared_ptr<ALongTypeName> &C d) {}",
               Style);
  verifyFormat("void ABCDEFGH::ABCDEFGHIJK(\n"
               "    [[maybe_unused]] const shared_ptr<ALongTypeName> &C d) {}",
               Style);
  verifyFormat("void ABCDEFGH::ABCDEFGH([[maybe_unused]] const "
               "shared_ptr<ALongTypeName> &C d) {\n}",
               Style);
}

TEST_F(FormatTest, UnderstandsEllipsis) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("int printf(const char *fmt, ...);");
  verifyFormat("template <class... Ts> void Foo(Ts... ts) { Foo(ts...); }");
  verifyFormat("template <class... Ts> void Foo(Ts *...ts) {}");

  verifyFormat("template <int *...PP> a;", Style);

  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("template <class... Ts> void Foo(Ts*... ts) {}", Style);

  verifyFormat("template <int*... PP> a;", Style);

  Style.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("template <int *... PP> a;", Style);
}

TEST_F(FormatTest, AdaptivelyFormatsPointersAndReferences) {
  verifyFormat("int *a;\n"
               "int *a;\n"
               "int *a;",
               "int *a;\n"
               "int* a;\n"
               "int *a;",
               getGoogleStyle());
  verifyFormat("int* a;\n"
               "int* a;\n"
               "int* a;",
               "int* a;\n"
               "int* a;\n"
               "int *a;",
               getGoogleStyle());
  verifyFormat("int *a;\n"
               "int *a;\n"
               "int *a;",
               "int *a;\n"
               "int * a;\n"
               "int *  a;",
               getGoogleStyle());
  verifyFormat("auto x = [] {\n"
               "  int *a;\n"
               "  int *a;\n"
               "  int *a;\n"
               "};",
               "auto x=[]{int *a;\n"
               "int * a;\n"
               "int *  a;};",
               getGoogleStyle());
}

TEST_F(FormatTest, UnderstandsRvalueReferences) {
  verifyFormat("int f(int &&a) {}");
  verifyFormat("int f(int a, char &&b) {}");
  verifyFormat("void f() { int &&a = b; }");
  verifyGoogleFormat("int f(int a, char&& b) {}");
  verifyGoogleFormat("void f() { int&& a = b; }");

  verifyIndependentOfContext("A<int &&> a;");
  verifyIndependentOfContext("A<int &&, int &&> a;");
  verifyGoogleFormat("A<int&&> a;");
  verifyGoogleFormat("A<int&&, int&&> a;");

  // Not rvalue references:
  verifyFormat("template <bool B, bool C> class A {\n"
               "  static_assert(B && C, \"Something is wrong\");\n"
               "};");
  verifyFormat("template <typename T> void swap() noexcept(Bar<T> && Foo<T>);");
  verifyFormat("template <typename T> struct S {\n"
               "  explicit(Bar<T> && Foo<T>) S(const S &);\n"
               "};");
  verifyGoogleFormat("#define IF(a, b, c) if (a && (b == c))");
  verifyGoogleFormat("#define WHILE(a, b, c) while (a && (b == c))");
  verifyFormat("#define A(a, b) (a && b)");
}

TEST_F(FormatTest, FormatsBinaryOperatorsPrecedingEquals) {
  verifyFormat("void f() {\n"
               "  x[aaaaaaaaa -\n"
               "    b] = 23;\n"
               "}",
               getLLVMStyleWithColumns(15));
}

TEST_F(FormatTest, FormatsCasts) {
  verifyFormat("Type *A = static_cast<Type *>(P);");
  verifyFormat("static_cast<Type *>(P);");
  verifyFormat("static_cast<Type &>(Fun)(Args);");
  verifyFormat("static_cast<Type &>(*Fun)(Args);");
  verifyFormat("if (static_cast<int>(A) + B >= 0)\n  ;");
  // Check that static_cast<...>(...) does not require the next token to be on
  // the same line.
  verifyFormat("some_loooong_output << something_something__ << "
               "static_cast<const void *>(R)\n"
               "                    << something;");
  verifyFormat("a = static_cast<Type &>(*Fun)(Args);");
  verifyFormat("const_cast<Type &>(*Fun)(Args);");
  verifyFormat("dynamic_cast<Type &>(*Fun)(Args);");
  verifyFormat("reinterpret_cast<Type &>(*Fun)(Args);");
  verifyFormat("Type *A = (Type *)P;");
  verifyFormat("Type *A = (vector<Type *, int *>)P;");
  verifyFormat("int a = (int)(2.0f);");
  verifyFormat("int a = (int)2.0f;");
  verifyFormat("x[(int32)y];");
  verifyFormat("x = (int32)y;");
  verifyFormat("#define AA(X) sizeof(((X *)NULL)->a)");
  verifyFormat("int a = (int)*b;");
  verifyFormat("int a = (int)2.0f;");
  verifyFormat("int a = (int)~0;");
  verifyFormat("int a = (int)++a;");
  verifyFormat("int a = (int)sizeof(int);");
  verifyFormat("int a = (int)+2;");
  verifyFormat("my_int a = (my_int)2.0f;");
  verifyFormat("my_int a = (my_int)sizeof(int);");
  verifyFormat("return (my_int)aaa;");
  verifyFormat("throw (my_int)aaa;");
  verifyFormat("#define x ((int)-1)");
  verifyFormat("#define LENGTH(x, y) (x) - (y) + 1");
  verifyFormat("#define p(q) ((int *)&q)");
  verifyFormat("fn(a)(b) + 1;");

  verifyFormat("void f() { my_int a = (my_int)*b; }");
  verifyFormat("void f() { return P ? (my_int)*P : (my_int)0; }");
  verifyFormat("my_int a = (my_int)~0;");
  verifyFormat("my_int a = (my_int)++a;");
  verifyFormat("my_int a = (my_int)-2;");
  verifyFormat("my_int a = (my_int)1;");
  verifyFormat("my_int a = (my_int *)1;");
  verifyFormat("my_int a = (const my_int)-1;");
  verifyFormat("my_int a = (const my_int *)-1;");
  verifyFormat("my_int a = (my_int)(my_int)-1;");
  verifyFormat("my_int a = (ns::my_int)-2;");
  verifyFormat("case (my_int)ONE:");
  verifyFormat("auto x = (X)this;");
  // Casts in Obj-C style calls used to not be recognized as such.
  verifyGoogleFormat("int a = [(type*)[((type*)val) arg] arg];");

  // FIXME: single value wrapped with paren will be treated as cast.
  verifyFormat("void f(int i = (kValue)*kMask) {}");

  verifyFormat("{ (void)F; }");

  // Don't break after a cast's
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    (aaaaaaaaaaaaaaaaaaaaaaaaaa *)(aaaaaaaaaaaaaaaaaaaaaa +\n"
               "                                   bbbbbbbbbbbbbbbbbbbbbb);");

  verifyFormat("#define CONF_BOOL(x) (bool *)(void *)(x)");
  verifyFormat("#define CONF_BOOL(x) (bool *)(x)");
  verifyFormat("#define CONF_BOOL(x) (bool)(x)");
  verifyFormat("bool *y = (bool *)(void *)(x);");
  verifyFormat("#define CONF_BOOL(x) (bool *)(void *)(int)(x)");
  verifyFormat("bool *y = (bool *)(void *)(int)(x);");
  verifyFormat("#define CONF_BOOL(x) (bool *)(void *)(int)foo(x)");
  verifyFormat("bool *y = (bool *)(void *)(int)foo(x);");

  // These are not casts.
  verifyFormat("void f(int *) {}");
  verifyFormat("f(foo)->b;");
  verifyFormat("f(foo).b;");
  verifyFormat("f(foo)(b);");
  verifyFormat("f(foo)[b];");
  verifyFormat("[](foo) { return 4; }(bar);");
  verifyFormat("(*funptr)(foo)[4];");
  verifyFormat("funptrs[4](foo)[4];");
  verifyFormat("void f(int *);");
  verifyFormat("void f(int *) = 0;");
  verifyFormat("void f(SmallVector<int>) {}");
  verifyFormat("void f(SmallVector<int>);");
  verifyFormat("void f(SmallVector<int>) = 0;");
  verifyFormat("void f(int i = (kA * kB) & kMask) {}");
  verifyFormat("int a = sizeof(int) * b;");
  verifyGoogleFormat("int a = alignof(int) * b;");
  verifyFormat("template <> void f<int>(int i) SOME_ANNOTATION;");
  verifyFormat("f(\"%\" SOME_MACRO(ll) \"d\");");
  verifyFormat("aaaaa &operator=(const aaaaa &) LLVM_DELETED_FUNCTION;");

  // These are not casts, but at some point were confused with casts.
  verifyFormat("virtual void foo(int *) override;");
  verifyFormat("virtual void foo(char &) const;");
  verifyFormat("virtual void foo(int *a, char *) const;");
  verifyFormat("int a = sizeof(int *) + b;");
  verifyGoogleFormat("int a = alignof(int *) + b;");
  verifyFormat("bool b = f(g<int>) && c;");
  verifyFormat("typedef void (*f)(int i) func;");
  verifyFormat("void operator++(int) noexcept;");
  verifyFormat("void operator++(int &) noexcept;");
  verifyFormat("void operator delete(void *, std::size_t, const std::nothrow_t "
               "&) noexcept;");
  verifyFormat(
      "void operator delete(std::size_t, const std::nothrow_t &) noexcept;");
  verifyFormat("void operator delete(const std::nothrow_t &) noexcept;");
  verifyFormat("void operator delete(std::nothrow_t &) noexcept;");
  verifyFormat("void operator delete(nothrow_t &) noexcept;");
  verifyFormat("void operator delete(foo &) noexcept;");
  verifyFormat("void operator delete(foo) noexcept;");
  verifyFormat("void operator delete(int) noexcept;");
  verifyFormat("void operator delete(int &) noexcept;");
  verifyFormat("void operator delete(int &) volatile noexcept;");
  verifyFormat("void operator delete(int &) const");
  verifyFormat("void operator delete(int &) = default");
  verifyFormat("void operator delete(int &) = delete");
  verifyFormat("void operator delete(int &) [[noreturn]]");
  verifyFormat("void operator delete(int &) throw();");
  verifyFormat("void operator delete(int &) throw(int);");
  verifyFormat("auto operator delete(int &) -> int;");
  verifyFormat("auto operator delete(int &) override");
  verifyFormat("auto operator delete(int &) final");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *foo = (aaaaaaaaaaaaaaaaa *)\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;");
  // FIXME: The indentation here is not ideal.
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    [bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] = (*cccccccccccccccc)\n"
      "        [dddddddddddddddddddddddddddddddddddddddddddddddddddddddd];");
}

TEST_F(FormatTest, FormatsFunctionTypes) {
  verifyFormat("A<bool()> a;");
  verifyFormat("A<SomeType()> a;");
  verifyFormat("A<void (*)(int, std::string)> a;");
  verifyFormat("A<void *(int)>;");
  verifyFormat("void *(*a)(int *, SomeType *);");
  verifyFormat("int (*func)(void *);");
  verifyFormat("void f() { int (*func)(void *); }");
  verifyFormat("template <class CallbackClass>\n"
               "using MyCallback = void (CallbackClass::*)(SomeObject *Data);");

  verifyGoogleFormat("A<void*(int*, SomeType*)>;");
  verifyGoogleFormat("void* (*a)(int);");
  verifyGoogleFormat(
      "template <class CallbackClass>\n"
      "using MyCallback = void (CallbackClass::*)(SomeObject* Data);");

  // Other constructs can look somewhat like function types:
  verifyFormat("A<sizeof(*x)> a;");
  verifyFormat("#define DEREF_AND_CALL_F(x) f(*x)");
  verifyFormat("some_var = function(*some_pointer_var)[0];");
  verifyFormat("void f() { function(*some_pointer_var)[0] = 10; }");
  verifyFormat("int x = f(&h)();");
  verifyFormat("returnsFunction(&param1, &param2)(param);");
  verifyFormat("std::function<\n"
               "    LooooooooooongTemplatedType<\n"
               "        SomeType>*(\n"
               "        LooooooooooooooooongType type)>\n"
               "    function;",
               getGoogleStyleWithColumns(40));
}

TEST_F(FormatTest, FormatsPointersToArrayTypes) {
  verifyFormat("A (*foo_)[6];");
  verifyFormat("vector<int> (*foo_)[6];");
}

TEST_F(FormatTest, BreaksLongVariableDeclarations) {
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable;");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType const\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable;");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    *LoooooooooooooooooooooooooooooooooooooooongVariable;");

  // Different ways of ()-initializiation.
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable(1);");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable(a);");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable({});");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooongVariable([A a]);");

  // Lambdas should not confuse the variable declaration heuristic.
  verifyFormat("LooooooooooooooooongType\n"
               "    variable(nullptr, [](A *a) {});",
               getLLVMStyleWithColumns(40));
}

TEST_F(FormatTest, BreaksLongDeclarations) {
  verifyFormat("typedef LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    AnotherNameForTheLongType;");
  verifyFormat("typedef LongTemplateType<aaaaaaaaaaaaaaaaaaa()>\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
               "LoooooooooooooooooooooooooooooooongFunctionDeclaration();");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType *\n"
               "LoooooooooooooooooooooooooooooooongFunctionDeclaration();");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType MACRO\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType const\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("decltype(LoooooooooooooooooooooooooooooooooooooooongName)\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("typeof(LoooooooooooooooooooooooooooooooooooooooooongName)\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("_Atomic(LooooooooooooooooooooooooooooooooooooooooongName)\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("__underlying_type(LooooooooooooooooooooooooooooooongName)\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
               "LooooooooooooooooooooooooooongFunctionDeclaration(T... t);");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
               "LooooooooooooooooooooooooooongFunctionDeclaration(T /*t*/) {}");
  FormatStyle Indented = getLLVMStyle();
  Indented.IndentWrappedFunctionNames = true;
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
               "    LoooooooooooooooooooooooooooooooongFunctionDeclaration();",
               Indented);
  verifyFormat(
      "LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
      "    LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}",
      Indented);
  verifyFormat(
      "LoooooooooooooooooooooooooooooooooooooooongReturnType const\n"
      "    LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}",
      Indented);
  verifyFormat(
      "decltype(LoooooooooooooooooooooooooooooooooooooooongName)\n"
      "    LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}",
      Indented);

  // FIXME: Without the comment, this breaks after "(".
  verifyGoogleFormat(
      "LoooooooooooooooooooooooooooooooooooooooongType  // break\n"
      "    (*LoooooooooooooooooooooooooooongFunctionTypeVarialbe)();");

  verifyFormat("int *someFunction(int LoooooooooooooooooooongParam1,\n"
               "                  int LoooooooooooooooooooongParam2) {}");
  verifyFormat(
      "TypeSpecDecl *TypeSpecDecl::Create(ASTContext &C, DeclContext *DC,\n"
      "                                   SourceLocation L, IdentifierIn *II,\n"
      "                                   Type *T) {}");
  verifyFormat("ReallyLongReturnType<TemplateParam1, TemplateParam2>\n"
               "ReallyReaaallyLongFunctionName(\n"
               "    const std::string &SomeParameter,\n"
               "    const SomeType<string, SomeOtherTemplateParameter>\n"
               "        &ReallyReallyLongParameterName,\n"
               "    const SomeType<string, SomeOtherTemplateParameter>\n"
               "        &AnotherLongParameterName) {}");
  verifyFormat("template <typename A>\n"
               "SomeLoooooooooooooooooooooongType<\n"
               "    typename some_namespace::SomeOtherType<A>::Type>\n"
               "Function() {}");

  verifyGoogleFormat(
      "aaaaaaaaaaaaaaaa::aaaaaaaaaaaaaaaa<aaaaaaaaaaaaa, aaaaaaaaaaaa>\n"
      "    aaaaaaaaaaaaaaaaaaaaaaa;");
  verifyGoogleFormat(
      "TypeSpecDecl* TypeSpecDecl::Create(ASTContext& C, DeclContext* DC,\n"
      "                                   SourceLocation L) {}");
  verifyGoogleFormat(
      "some_namespace::LongReturnType\n"
      "long_namespace::SomeVeryLongClass::SomeVeryLongFunction(\n"
      "    int first_long_parameter, int second_parameter) {}");

  verifyGoogleFormat("template <typename T>\n"
                     "aaaaaaaa::aaaaa::aaaaaa<T, aaaaaaaaaaaaaaaaaaaaaaaaa>\n"
                     "aaaaaaaaaaaaaaaaaaaaaaaa<T>::aaaaaaa() {}");
  verifyGoogleFormat("A<A<A>> aaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
                     "                   int aaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("typedef size_t (*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)(\n"
               "    const aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        *aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    vector<aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    vector<aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>>\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("template <typename T> // Templates on own line.\n"
               "static int            // Some comment.\n"
               "MyFunction(int a);");
}

TEST_F(FormatTest, FormatsAccessModifiers) {
  FormatStyle Style = getLLVMStyle();
  EXPECT_EQ(Style.EmptyLineBeforeAccessModifier,
            FormatStyle::ELBAMS_LogicalBlock);
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "private:\n"
               "  int i;\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo { /* comment */\n"
               "private:\n"
               "  int i;\n"
               "  // comment\n"
               "private:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "#ifdef FOO\n"
               "#endif\n"
               "private:\n"
               "  int i;\n"
               "#ifdef FOO\n"
               "private:\n"
               "#endif\n"
               "  int j;\n"
               "};",
               Style);
  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Never;
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "private:\n"
               "  int i;\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "private:\n"
               "  int i;\n"
               "protected:\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo { /* comment */\n"
               "private:\n"
               "  int i;\n"
               "  // comment\n"
               "private:\n"
               "  int j;\n"
               "};",
               "struct foo { /* comment */\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "  // comment\n"
               "\n"
               "private:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "#ifdef FOO\n"
               "#endif\n"
               "private:\n"
               "  int i;\n"
               "#ifdef FOO\n"
               "private:\n"
               "#endif\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "#ifdef FOO\n"
               "#endif\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "#ifdef FOO\n"
               "\n"
               "private:\n"
               "#endif\n"
               "  int j;\n"
               "};",
               Style);
  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Always;
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "private:\n"
               "  int i;\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo { /* comment */\n"
               "private:\n"
               "  int i;\n"
               "  // comment\n"
               "\n"
               "private:\n"
               "  int j;\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "#ifdef FOO\n"
               "#endif\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "#ifdef FOO\n"
               "\n"
               "private:\n"
               "#endif\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "#ifdef FOO\n"
               "#endif\n"
               "private:\n"
               "  int i;\n"
               "#ifdef FOO\n"
               "private:\n"
               "#endif\n"
               "  int j;\n"
               "};",
               Style);
  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Leave;
  verifyNoChange("struct foo {\n"
                 "\n"
                 "private:\n"
                 "  void f() {}\n"
                 "\n"
                 "private:\n"
                 "  int i;\n"
                 "\n"
                 "protected:\n"
                 "  int j;\n"
                 "};",
                 Style);
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "private:\n"
               "  int i;\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);
  verifyNoChange("struct foo { /* comment */\n"
                 "\n"
                 "private:\n"
                 "  int i;\n"
                 "  // comment\n"
                 "\n"
                 "private:\n"
                 "  int j;\n"
                 "};",
                 Style);
  verifyFormat("struct foo { /* comment */\n"
               "private:\n"
               "  int i;\n"
               "  // comment\n"
               "private:\n"
               "  int j;\n"
               "};",
               Style);
  verifyNoChange("struct foo {\n"
                 "#ifdef FOO\n"
                 "#endif\n"
                 "\n"
                 "private:\n"
                 "  int i;\n"
                 "#ifdef FOO\n"
                 "\n"
                 "private:\n"
                 "#endif\n"
                 "  int j;\n"
                 "};",
                 Style);
  verifyFormat("struct foo {\n"
               "#ifdef FOO\n"
               "#endif\n"
               "private:\n"
               "  int i;\n"
               "#ifdef FOO\n"
               "private:\n"
               "#endif\n"
               "  int j;\n"
               "};",
               Style);

  FormatStyle NoEmptyLines = getLLVMStyle();
  NoEmptyLines.MaxEmptyLinesToKeep = 0;
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "public:\n"
               "protected:\n"
               "  int j;\n"
               "};",
               NoEmptyLines);

  NoEmptyLines.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Never;
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "private:\n"
               "  int i;\n"
               "public:\n"
               "protected:\n"
               "  int j;\n"
               "};",
               NoEmptyLines);

  NoEmptyLines.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Always;
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "public:\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               NoEmptyLines);
}

TEST_F(FormatTest, FormatsAfterAccessModifiers) {

  FormatStyle Style = getLLVMStyle();
  EXPECT_EQ(Style.EmptyLineAfterAccessModifier, FormatStyle::ELAAMS_Never);
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);

  // Check if lines are removed.
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "\n"
               "  int j;\n"
               "};",
               Style);

  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Always;
  verifyFormat("struct foo {\n"
               "private:\n"
               "\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "\n"
               "  int j;\n"
               "};",
               Style);

  // Check if lines are added.
  verifyFormat("struct foo {\n"
               "private:\n"
               "\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);

  // Leave tests rely on the code layout, test::messUp can not be used.
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Leave;
  Style.MaxEmptyLinesToKeep = 0u;
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);

  // Check if MaxEmptyLinesToKeep is respected.
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n\n\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "\n\n\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "\n\n\n"
               "  int j;\n"
               "};",
               Style);

  Style.MaxEmptyLinesToKeep = 1u;
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n"
                 "  void f() {}\n"
                 "\n"
                 "private:\n"
                 "\n"
                 "  int i;\n"
                 "\n"
                 "protected:\n"
                 "\n"
                 "  int j;\n"
                 "};",
                 Style);
  // Check if no lines are kept.
  verifyFormat("struct foo {\n"
               "private:\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "  int j;\n"
               "};",
               Style);
  // Check if MaxEmptyLinesToKeep is respected.
  verifyFormat("struct foo {\n"
               "private:\n"
               "\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "\n"
               "  int j;\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n\n\n"
               "  void f() {}\n"
               "\n"
               "private:\n"
               "\n\n\n"
               "  int i;\n"
               "\n"
               "protected:\n"
               "\n\n\n"
               "  int j;\n"
               "};",
               Style);

  Style.MaxEmptyLinesToKeep = 10u;
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n\n\n"
                 "  void f() {}\n"
                 "\n"
                 "private:\n"
                 "\n\n\n"
                 "  int i;\n"
                 "\n"
                 "protected:\n"
                 "\n\n\n"
                 "  int j;\n"
                 "};",
                 Style);

  // Test with comments.
  Style = getLLVMStyle();
  verifyFormat("struct foo {\n"
               "private:\n"
               "  // comment\n"
               "  void f() {}\n"
               "\n"
               "private: /* comment */\n"
               "  int i;\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "private:\n"
               "  // comment\n"
               "  void f() {}\n"
               "\n"
               "private: /* comment */\n"
               "  int i;\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n"
               "  // comment\n"
               "  void f() {}\n"
               "\n"
               "private: /* comment */\n"
               "\n"
               "  int i;\n"
               "};",
               Style);

  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Always;
  verifyFormat("struct foo {\n"
               "private:\n"
               "\n"
               "  // comment\n"
               "  void f() {}\n"
               "\n"
               "private: /* comment */\n"
               "\n"
               "  int i;\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "  // comment\n"
               "  void f() {}\n"
               "\n"
               "private: /* comment */\n"
               "  int i;\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "private:\n"
               "\n"
               "  // comment\n"
               "  void f() {}\n"
               "\n"
               "private: /* comment */\n"
               "\n"
               "  int i;\n"
               "};",
               Style);

  // Test with preprocessor defines.
  Style = getLLVMStyle();
  verifyFormat("struct foo {\n"
               "private:\n"
               "#ifdef FOO\n"
               "#endif\n"
               "  void f() {}\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "private:\n"
               "#ifdef FOO\n"
               "#endif\n"
               "  void f() {}\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n"
               "#ifdef FOO\n"
               "#endif\n"
               "  void f() {}\n"
               "};",
               Style);
  verifyNoChange("struct foo {\n"
                 "#ifdef FOO\n"
                 "#else\n"
                 "private:\n"
                 "\n"
                 "#endif\n"
                 "};",
                 Style);
  verifyFormat("struct foo {\n"
               "#ifdef FOO\n"
               "#else\n"
               "private:\n"
               "\n"
               "#endif\n"
               "};",
               "struct foo {\n"
               "#ifdef FOO\n"
               "#else\n"
               "private:\n"
               "\n"
               "\n"
               "#endif\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "#ifdef FOO\n"
               "private:\n"
               "#else\n"
               "#endif\n"
               "};",
               "struct foo {\n"
               "#ifdef FOO\n"
               "private:\n"
               "\n"
               "\n"
               "#else\n"
               "#endif\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "#if 0\n"
               "#else\n"
               "#endif\n"
               "#ifdef FOO\n"
               "private:\n"
               "#endif\n"
               "};",
               "struct foo {\n"
               "#if 0\n"
               "#else\n"
               "#endif\n"
               "#ifdef FOO\n"
               "private:\n"
               "\n"
               "\n"
               "#endif\n"
               "};",
               Style);

  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Always;
  verifyFormat("struct foo {\n"
               "private:\n"
               "\n"
               "#ifdef FOO\n"
               "#endif\n"
               "  void f() {}\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "#ifdef FOO\n"
               "#endif\n"
               "  void f() {}\n"
               "};",
               Style);
  verifyFormat("struct foo {\n"
               "private:\n"
               "\n"
               "#ifdef FOO\n"
               "#endif\n"
               "  void f() {}\n"
               "};",
               Style);
}

TEST_F(FormatTest, FormatsAfterAndBeforeAccessModifiersInteraction) {
  // Combined tests of EmptyLineAfterAccessModifier and
  // EmptyLineBeforeAccessModifier.
  FormatStyle Style = getLLVMStyle();
  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Always;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Always;
  verifyFormat("struct foo {\n"
               "private:\n"
               "\n"
               "protected:\n"
               "};",
               Style);

  Style.MaxEmptyLinesToKeep = 10u;
  // Both remove all new lines.
  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Never;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Never;
  verifyFormat("struct foo {\n"
               "private:\n"
               "protected:\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n\n\n"
               "protected:\n"
               "};",
               Style);

  // Leave tests rely on the code layout, test::messUp can not be used.
  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Leave;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Leave;
  Style.MaxEmptyLinesToKeep = 10u;
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n\n\n"
                 "protected:\n"
                 "};",
                 Style);
  Style.MaxEmptyLinesToKeep = 3u;
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n\n\n"
                 "protected:\n"
                 "};",
                 Style);
  Style.MaxEmptyLinesToKeep = 1u;
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n\n\n"
                 "protected:\n"
                 "};",
                 Style); // Based on new lines in original document and not
                         // on the setting.

  Style.MaxEmptyLinesToKeep = 10u;
  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Always;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Leave;
  // Newlines are kept if they are greater than zero,
  // test::messUp removes all new lines which changes the logic
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n\n\n"
                 "protected:\n"
                 "};",
                 Style);

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Leave;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Always;
  // test::messUp removes all new lines which changes the logic
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n\n\n"
                 "protected:\n"
                 "};",
                 Style);

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Leave;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Never;
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n\n\n"
                 "protected:\n"
                 "};",
                 Style); // test::messUp removes all new lines which changes
                         // the logic.

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Never;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Leave;
  verifyFormat("struct foo {\n"
               "private:\n"
               "protected:\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n\n\n"
               "protected:\n"
               "};",
               Style);

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Always;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Never;
  verifyNoChange("struct foo {\n"
                 "private:\n"
                 "\n\n\n"
                 "protected:\n"
                 "};",
                 Style); // test::messUp removes all new lines which changes
                         // the logic.

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_Never;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Always;
  verifyFormat("struct foo {\n"
               "private:\n"
               "protected:\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n\n\n"
               "protected:\n"
               "};",
               Style);

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_LogicalBlock;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Always;
  verifyFormat("struct foo {\n"
               "private:\n"
               "protected:\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n\n\n"
               "protected:\n"
               "};",
               Style);

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_LogicalBlock;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Leave;
  verifyFormat("struct foo {\n"
               "private:\n"
               "protected:\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n\n\n"
               "protected:\n"
               "};",
               Style);

  Style.EmptyLineBeforeAccessModifier = FormatStyle::ELBAMS_LogicalBlock;
  Style.EmptyLineAfterAccessModifier = FormatStyle::ELAAMS_Never;
  verifyFormat("struct foo {\n"
               "private:\n"
               "protected:\n"
               "};",
               "struct foo {\n"
               "private:\n"
               "\n\n\n"
               "protected:\n"
               "};",
               Style);
}

TEST_F(FormatTest, FormatsArrays) {
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaa[aaaaaaaaaaaaaaaaaaaaaaaaa]\n"
               "                         [bbbbbbbbbbbbbbbbbbbbbbbbb] = c;");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaa[aaaaaaaaaaa(aaaaaaaaaaaa)]\n"
               "                         [bbbbbbbbbbb(bbbbbbbbbbbb)] = c;");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaa &&\n"
               "    aaaaaaaaaaaaaaaaaaa[aaaaaaaaaaaaa][aaaaaaaaaaaaa]) {\n}");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    [bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] = ccccccccccc;");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    [a][bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] = cccccccc;");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]\n"
               "    [bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] = ccccccccccc;");
  verifyFormat(
      "llvm::outs() << \"aaaaaaaaaaaa: \"\n"
      "             << (*aaaaaaaiaaaaaaa)[aaaaaaaaaaaaaaaaaaaaaaaaa]\n"
      "                                  [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa];");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa[aaaaaaaaaaaaaaaaa][a]\n"
               "    .aaaaaaaaaaaaaaaaaaaaaa();");

  verifyGoogleFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<int>\n"
                     "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa[aaaaaaaaaaaa];");
  verifyFormat(
      "aaaaaaaaaaa aaaaaaaaaaaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaa->aaaaaaaaa[0]\n"
      "                                  .aaaaaaa[0]\n"
      "                                  .aaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("a[::b::c];");

  verifyNoCrash("a[,Y?)]", getLLVMStyleWithColumns(10));

  FormatStyle NoColumnLimit = getLLVMStyleWithColumns(0);
  verifyFormat("aaaaa[bbbbbb].cccccc()", NoColumnLimit);
}

TEST_F(FormatTest, LineStartsWithSpecialCharacter) {
  verifyFormat("(a)->b();");
  verifyFormat("--a;");
}

TEST_F(FormatTest, HandlesIncludeDirectives) {
  verifyFormat("#include <string>\n"
               "#include <a/b/c.h>\n"
               "#include \"a/b/string\"\n"
               "#include \"string.h\"\n"
               "#include \"string.h\"\n"
               "#include <a-a>\n"
               "#include < path with space >\n"
               "#include_next <test.h>"
               "#include \"abc.h\" // this is included for ABC\n"
               "#include \"some long include\" // with a comment\n"
               "#include \"some very long include path\"\n"
               "#include <some/very/long/include/path>",
               getLLVMStyleWithColumns(35));
  verifyFormat("#include \"a.h\"", "#include  \"a.h\"");
  verifyFormat("#include <a>", "#include<a>");

  verifyFormat("#import <string>");
  verifyFormat("#import <a/b/c.h>");
  verifyFormat("#import \"a/b/string\"");
  verifyFormat("#import \"string.h\"");
  verifyFormat("#import \"string.h\"");
  verifyFormat("#if __has_include(<strstream>)\n"
               "#include <strstream>\n"
               "#endif");

  verifyFormat("#define MY_IMPORT <a/b>");

  verifyFormat("#if __has_include(<a/b>)");
  verifyFormat("#if __has_include_next(<a/b>)");
  verifyFormat("#define F __has_include(<a/b>)");
  verifyFormat("#define F __has_include_next(<a/b>)");

  // Protocol buffer definition or missing "#".
  verifyFormat("import \"aaaaaaaaaaaaaaaaa/aaaaaaaaaaaaaaa\";",
               getLLVMStyleWithColumns(30));

  FormatStyle Style = getLLVMStyle();
  Style.AlwaysBreakBeforeMultilineStrings = true;
  Style.ColumnLimit = 0;
  verifyFormat("#import \"abc.h\"", Style);

  // But 'import' might also be a regular C++ namespace.
  verifyFormat("import::SomeFunction(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("import::Bar foo(val ? 2 : 1);");
}

//===----------------------------------------------------------------------===//
// Error recovery tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, IncompleteParameterLists) {
  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = false;
  verifyFormat("void aaaaaaaaaaaaaaaaaa(int level,\n"
               "                        double *min_x,\n"
               "                        double *max_x,\n"
               "                        double *min_y,\n"
               "                        double *max_y,\n"
               "                        double *min_z,\n"
               "                        double *max_z, ) {}",
               NoBinPacking);
}

TEST_F(FormatTest, IncorrectCodeTrailingStuff) {
  verifyFormat("void f() { return; }\n42");
  verifyFormat("void f() {\n"
               "  if (0)\n"
               "    return;\n"
               "}\n"
               "42");
  verifyFormat("void f() { return }\n42");
  verifyFormat("void f() {\n"
               "  if (0)\n"
               "    return\n"
               "}\n"
               "42");
}

TEST_F(FormatTest, IncorrectCodeMissingSemicolon) {
  verifyFormat("void f() { return }", "void  f ( )  {  return  }");
  verifyFormat("void f() {\n"
               "  if (a)\n"
               "    return\n"
               "}",
               "void  f  (  )  {  if  ( a )  return  }");
  verifyFormat("namespace N {\n"
               "void f()\n"
               "}",
               "namespace  N  {  void f()  }");
  verifyFormat("namespace N {\n"
               "void f() {}\n"
               "void g()\n"
               "} // namespace N",
               "namespace N  { void f( ) { } void g( ) }");
}

TEST_F(FormatTest, IndentationWithinColumnLimitNotPossible) {
  verifyFormat("int aaaaaaaa =\n"
               "    // Overlylongcomment\n"
               "    b;",
               getLLVMStyleWithColumns(20));
  verifyFormat("function(\n"
               "    ShortArgument,\n"
               "    LoooooooooooongArgument);",
               getLLVMStyleWithColumns(20));
}

TEST_F(FormatTest, IncorrectAccessSpecifier) {
  verifyFormat("public:");
  verifyFormat("class A {\n"
               "public\n"
               "  void f() {}\n"
               "};");
  verifyFormat("public\n"
               "int qwerty;");
  verifyFormat("public\n"
               "B {}");
  verifyFormat("public\n"
               "{}");
  verifyFormat("public\n"
               "B { int x; }");
}

TEST_F(FormatTest, IncorrectCodeUnbalancedBraces) {
  verifyFormat("{");
  verifyFormat("#})");
  verifyNoCrash("(/**/[:!] ?[).");
}

TEST_F(FormatTest, IncorrectUnbalancedBracesInMacrosWithUnicode) {
  // Found by oss-fuzz:
  // https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=8212
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_Cpp);
  Style.ColumnLimit = 60;
  verifyNoCrash(
      "\x23\x47\xff\x20\x28\xff\x3c\xff\x3f\xff\x20\x2f\x7b\x7a\xff\x20"
      "\xff\xff\xff\xca\xb5\xff\xff\xff\xff\x3a\x7b\x7d\xff\x20\xff\x20"
      "\xff\x74\xff\x20\x7d\x7d\xff\x7b\x3a\xff\x20\x71\xff\x20\xff\x0a",
      Style);
}

TEST_F(FormatTest, IncorrectCodeDoNoWhile) {
  verifyFormat("do {\n}");
  verifyFormat("do {\n}\n"
               "f();");
  verifyFormat("do {\n}\n"
               "wheeee(fun);");
  verifyFormat("do {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, IncorrectCodeMissingParens) {
  verifyFormat("if {\n  foo;\n  foo();\n}");
  verifyFormat("switch {\n  foo;\n  foo();\n}");
  verifyIncompleteFormat("for {\n  foo;\n  foo();\n}");
  verifyIncompleteFormat("ERROR: for target;");
  verifyFormat("while {\n  foo;\n  foo();\n}");
  verifyFormat("do {\n  foo;\n  foo();\n} while;");
}

TEST_F(FormatTest, DoesNotTouchUnwrappedLinesWithErrors) {
  verifyIncompleteFormat("namespace {\n"
                         "class Foo { Foo (\n"
                         "};\n"
                         "} // namespace");
}

TEST_F(FormatTest, IncorrectCodeErrorDetection) {
  verifyFormat("{\n  {}", "{\n{\n}");
  verifyFormat("{\n  {}", "{\n  {\n}");
  verifyFormat("{\n  {}", "{\n  {\n  }");
  verifyFormat("{\n  {}\n}\n}", "{\n  {\n    }\n  }\n}");

  verifyFormat("{\n"
               "  {\n"
               "    breakme(\n"
               "        qwe);\n"
               "  }",
               "{\n"
               "    {\n"
               " breakme(qwe);\n"
               "}",
               getLLVMStyleWithColumns(10));
}

TEST_F(FormatTest, LayoutCallsInsideBraceInitializers) {
  verifyFormat("int x = {\n"
               "    avariable,\n"
               "    b(alongervariable)};",
               getLLVMStyleWithColumns(25));
}

TEST_F(FormatTest, LayoutBraceInitializersInReturnStatement) {
  verifyFormat("return (a)(b){1, 2, 3};");
}

TEST_F(FormatTest, LayoutCxx11BraceInitializers) {
  verifyFormat("vector<int> x{1, 2, 3, 4};");
  verifyFormat("vector<int> x{\n"
               "    1,\n"
               "    2,\n"
               "    3,\n"
               "    4,\n"
               "};");
  verifyFormat("vector<T> x{{}, {}, {}, {}};");
  verifyFormat("f({1, 2});");
  verifyFormat("auto v = Foo{-1};");
  verifyFormat("f({1, 2}, {{2, 3}, {4, 5}}, c, {d});");
  verifyFormat("Class::Class : member{1, 2, 3} {}");
  verifyFormat("new vector<int>{1, 2, 3};");
  verifyFormat("new int[3]{1, 2, 3};");
  verifyFormat("new int{1};");
  verifyFormat("return {arg1, arg2};");
  verifyFormat("return {arg1, SomeType{parameter}};");
  verifyFormat("int count = set<int>{f(), g(), h()}.size();");
  verifyFormat("new T{arg1, arg2};");
  verifyFormat("f(MyMap[{composite, key}]);");
  verifyFormat("class Class {\n"
               "  T member = {arg1, arg2};\n"
               "};");
  verifyFormat("vector<int> foo = {::SomeGlobalFunction()};");
  verifyFormat("const struct A a = {.a = 1, .b = 2};");
  verifyFormat("const struct A a = {[0] = 1, [1] = 2};");
  verifyFormat("static_assert(std::is_integral<int>{} + 0, \"\");");
  verifyFormat("int a = std::is_integral<int>{} + 0;");

  verifyFormat("int foo(int i) { return fo1{}(i); }");
  verifyFormat("int foo(int i) { return fo1{}(i); }");
  verifyFormat("auto i = decltype(x){};");
  verifyFormat("auto i = typeof(x){};");
  verifyFormat("auto i = _Atomic(x){};");
  verifyFormat("std::vector<int> v = {1, 0 /* comment */};");
  verifyFormat("Node n{1, Node{1000}, //\n"
               "       2};");
  verifyFormat("Aaaa aaaaaaa{\n"
               "    {\n"
               "        aaaa,\n"
               "    },\n"
               "};");
  verifyFormat("class C : public D {\n"
               "  SomeClass SC{2};\n"
               "};");
  verifyFormat("class C : public A {\n"
               "  class D : public B {\n"
               "    void f() { int i{2}; }\n"
               "  };\n"
               "};");
  verifyFormat("#define A {a, a},");
  // Don't confuse braced list initializers with compound statements.
  verifyFormat(
      "class A {\n"
      "  A() : a{} {}\n"
      "  A() : Base<int>{} {}\n"
      "  A() : Base<Foo<int>>{} {}\n"
      "  A(int b) : b(b) {}\n"
      "  A(int a, int b) : a(a), bs{{bs...}} { f(); }\n"
      "  int a, b;\n"
      "  explicit Expr(const Scalar<Result> &x) : u{Constant<Result>{x}} {}\n"
      "  explicit Expr(Scalar<Result> &&x) : u{Constant<Result>{std::move(x)}} "
      "{}\n"
      "};");

  // Avoid breaking between equal sign and opening brace
  FormatStyle AvoidBreakingFirstArgument = getLLVMStyle();
  AvoidBreakingFirstArgument.PenaltyBreakBeforeFirstCallParameter = 200;
  verifyFormat("const std::unordered_map<std::string, int> MyHashTable =\n"
               "    {{\"aaaaaaaaaaaaaaaaaaaaa\", 0},\n"
               "     {\"bbbbbbbbbbbbbbbbbbbbb\", 1},\n"
               "     {\"ccccccccccccccccccccc\", 2}};",
               AvoidBreakingFirstArgument);

  // Binpacking only if there is no trailing comma
  verifyFormat("const Aaaaaa aaaaa = {aaaaaaaaaa, bbbbbbbbbb,\n"
               "                      cccccccccc, dddddddddd};",
               getLLVMStyleWithColumns(50));
  verifyFormat("const Aaaaaa aaaaa = {\n"
               "    aaaaaaaaaaa,\n"
               "    bbbbbbbbbbb,\n"
               "    ccccccccccc,\n"
               "    ddddddddddd,\n"
               "};",
               getLLVMStyleWithColumns(50));

  // Cases where distinguising braced lists and blocks is hard.
  verifyFormat("vector<int> v{12} GUARDED_BY(mutex);");
  verifyFormat("void f() {\n"
               "  return; // comment\n"
               "}\n"
               "SomeType t;");
  verifyFormat("void f() {\n"
               "  if (a) {\n"
               "    f();\n"
               "  }\n"
               "}\n"
               "SomeType t;");

  // In combination with BinPackArguments = false.
  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackArguments = false;
  verifyFormat("const Aaaaaa aaaaa = {aaaaa,\n"
               "                      bbbbb,\n"
               "                      ccccc,\n"
               "                      ddddd,\n"
               "                      eeeee,\n"
               "                      ffffff,\n"
               "                      ggggg,\n"
               "                      hhhhhh,\n"
               "                      iiiiii,\n"
               "                      jjjjjj,\n"
               "                      kkkkkk};",
               NoBinPacking);
  verifyFormat("const Aaaaaa aaaaa = {\n"
               "    aaaaa,\n"
               "    bbbbb,\n"
               "    ccccc,\n"
               "    ddddd,\n"
               "    eeeee,\n"
               "    ffffff,\n"
               "    ggggg,\n"
               "    hhhhhh,\n"
               "    iiiiii,\n"
               "    jjjjjj,\n"
               "    kkkkkk,\n"
               "};",
               NoBinPacking);
  verifyFormat(
      "const Aaaaaa aaaaa = {\n"
      "    aaaaa,  bbbbb,  ccccc,  ddddd,  eeeee,  ffffff, ggggg, hhhhhh,\n"
      "    iiiiii, jjjjjj, kkkkkk, aaaaa,  bbbbb,  ccccc,  ddddd, eeeee,\n"
      "    ffffff, ggggg,  hhhhhh, iiiiii, jjjjjj, kkkkkk,\n"
      "};",
      NoBinPacking);

  NoBinPacking.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  verifyFormat("static uint8 CddDp83848Reg[] = {\n"
               "    CDDDP83848_BMCR_REGISTER,\n"
               "    CDDDP83848_BMSR_REGISTER,\n"
               "    CDDDP83848_RBR_REGISTER};",
               "static uint8 CddDp83848Reg[] = {CDDDP83848_BMCR_REGISTER,\n"
               "                                CDDDP83848_BMSR_REGISTER,\n"
               "                                CDDDP83848_RBR_REGISTER};",
               NoBinPacking);

  // FIXME: The alignment of these trailing comments might be bad. Then again,
  // this might be utterly useless in real code.
  verifyFormat("Constructor::Constructor()\n"
               "    : some_value{         //\n"
               "                 aaaaaaa, //\n"
               "                 bbbbbbb} {}");

  // In braced lists, the first comment is always assumed to belong to the
  // first element. Thus, it can be moved to the next or previous line as
  // appropriate.
  verifyFormat("function({// First element:\n"
               "          1,\n"
               "          // Second element:\n"
               "          2});",
               "function({\n"
               "    // First element:\n"
               "    1,\n"
               "    // Second element:\n"
               "    2});");
  verifyFormat("std::vector<int> MyNumbers{\n"
               "    // First element:\n"
               "    1,\n"
               "    // Second element:\n"
               "    2};",
               "std::vector<int> MyNumbers{// First element:\n"
               "                           1,\n"
               "                           // Second element:\n"
               "                           2};",
               getLLVMStyleWithColumns(30));
  // A trailing comma should still lead to an enforced line break and no
  // binpacking.
  verifyFormat("vector<int> SomeVector = {\n"
               "    // aaa\n"
               "    1,\n"
               "    2,\n"
               "};",
               "vector<int> SomeVector = { // aaa\n"
               "    1, 2, };");

  // C++11 brace initializer list l-braces should not be treated any differently
  // when breaking before lambda bodies is enabled
  FormatStyle BreakBeforeLambdaBody = getLLVMStyle();
  BreakBeforeLambdaBody.BreakBeforeBraces = FormatStyle::BS_Custom;
  BreakBeforeLambdaBody.BraceWrapping.BeforeLambdaBody = true;
  BreakBeforeLambdaBody.AlwaysBreakBeforeMultilineStrings = true;
  verifyFormat(
      "std::runtime_error{\n"
      "    \"Long string which will force a break onto the next line...\"};",
      BreakBeforeLambdaBody);

  FormatStyle ExtraSpaces = getLLVMStyle();
  ExtraSpaces.Cpp11BracedListStyle = false;
  ExtraSpaces.ColumnLimit = 75;
  verifyFormat("vector<int> x{ 1, 2, 3, 4 };", ExtraSpaces);
  verifyFormat("vector<T> x{ {}, {}, {}, {} };", ExtraSpaces);
  verifyFormat("f({ 1, 2 });", ExtraSpaces);
  verifyFormat("auto v = Foo{ 1 };", ExtraSpaces);
  verifyFormat("f({ 1, 2 }, { { 2, 3 }, { 4, 5 } }, c, { d });", ExtraSpaces);
  verifyFormat("Class::Class : member{ 1, 2, 3 } {}", ExtraSpaces);
  verifyFormat("new vector<int>{ 1, 2, 3 };", ExtraSpaces);
  verifyFormat("new int[3]{ 1, 2, 3 };", ExtraSpaces);
  verifyFormat("return { arg1, arg2 };", ExtraSpaces);
  verifyFormat("return { arg1, SomeType{ parameter } };", ExtraSpaces);
  verifyFormat("int count = set<int>{ f(), g(), h() }.size();", ExtraSpaces);
  verifyFormat("new T{ arg1, arg2 };", ExtraSpaces);
  verifyFormat("f(MyMap[{ composite, key }]);", ExtraSpaces);
  verifyFormat("class Class {\n"
               "  T member = { arg1, arg2 };\n"
               "};",
               ExtraSpaces);
  verifyFormat(
      "foo = aaaaaaaaaaa ? vector<int>{ aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                                 aaaaaaaaaaaaaaaaaaaa, aaaaa }\n"
      "                  : vector<int>{ bbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
      "                                 bbbbbbbbbbbbbbbbbbbb, bbbbb };",
      ExtraSpaces);
  verifyFormat("DoSomethingWithVector({} /* No data */);", ExtraSpaces);
  verifyFormat("DoSomethingWithVector({ {} /* No data */ }, { { 1, 2 } });",
               ExtraSpaces);
  verifyFormat(
      "someFunction(OtherParam,\n"
      "             BracedList{ // comment 1 (Forcing interesting break)\n"
      "                         param1, param2,\n"
      "                         // comment 2\n"
      "                         param3, param4 });",
      ExtraSpaces);
  verifyFormat(
      "std::this_thread::sleep_for(\n"
      "    std::chrono::nanoseconds{ std::chrono::seconds{ 1 } } / 5);",
      ExtraSpaces);
  verifyFormat("std::vector<MyValues> aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa{\n"
               "    aaaaaaa,\n"
               "    aaaaaaaaaa,\n"
               "    aaaaa,\n"
               "    aaaaaaaaaaaaaaa,\n"
               "    aaa,\n"
               "    aaaaaaaaaa,\n"
               "    a,\n"
               "    aaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaa,\n"
               "    a};");
  verifyFormat("vector<int> foo = { ::SomeGlobalFunction() };", ExtraSpaces);
  verifyFormat("const struct A a = { .a = 1, .b = 2 };", ExtraSpaces);
  verifyFormat("const struct A a = { [0] = 1, [1] = 2 };", ExtraSpaces);

  // Avoid breaking between initializer/equal sign and opening brace
  ExtraSpaces.PenaltyBreakBeforeFirstCallParameter = 200;
  verifyFormat("const std::unordered_map<std::string, int> MyHashTable = {\n"
               "  { \"aaaaaaaaaaaaaaaaaaaaa\", 0 },\n"
               "  { \"bbbbbbbbbbbbbbbbbbbbb\", 1 },\n"
               "  { \"ccccccccccccccccccccc\", 2 }\n"
               "};",
               ExtraSpaces);
  verifyFormat("const std::unordered_map<std::string, int> MyHashTable{\n"
               "  { \"aaaaaaaaaaaaaaaaaaaaa\", 0 },\n"
               "  { \"bbbbbbbbbbbbbbbbbbbbb\", 1 },\n"
               "  { \"ccccccccccccccccccccc\", 2 }\n"
               "};",
               ExtraSpaces);

  FormatStyle SpaceBeforeBrace = getLLVMStyle();
  SpaceBeforeBrace.SpaceBeforeCpp11BracedList = true;
  verifyFormat("vector<int> x {1, 2, 3, 4};", SpaceBeforeBrace);
  verifyFormat("f({}, {{}, {}}, MyMap[{k, v}]);", SpaceBeforeBrace);

  FormatStyle SpaceBetweenBraces = getLLVMStyle();
  SpaceBetweenBraces.SpacesInAngles = FormatStyle::SIAS_Always;
  SpaceBetweenBraces.SpacesInParens = FormatStyle::SIPO_Custom;
  SpaceBetweenBraces.SpacesInParensOptions.Other = true;
  SpaceBetweenBraces.SpacesInSquareBrackets = true;
  verifyFormat("vector< int > x{ 1, 2, 3, 4 };", SpaceBetweenBraces);
  verifyFormat("f( {}, { {}, {} }, MyMap[ { k, v } ] );", SpaceBetweenBraces);
  verifyFormat("vector< int > x{ // comment 1\n"
               "                 1, 2, 3, 4 };",
               SpaceBetweenBraces);
  SpaceBetweenBraces.ColumnLimit = 20;
  verifyFormat("vector< int > x{\n"
               "    1, 2, 3, 4 };",
               "vector<int>x{1,2,3,4};", SpaceBetweenBraces);
  SpaceBetweenBraces.ColumnLimit = 24;
  verifyFormat("vector< int > x{ 1, 2,\n"
               "                 3, 4 };",
               "vector<int>x{1,2,3,4};", SpaceBetweenBraces);
  verifyFormat("vector< int > x{\n"
               "    1,\n"
               "    2,\n"
               "    3,\n"
               "    4,\n"
               "};",
               "vector<int>x{1,2,3,4,};", SpaceBetweenBraces);
  verifyFormat("vector< int > x{};", SpaceBetweenBraces);
  SpaceBetweenBraces.SpacesInParens = FormatStyle::SIPO_Custom;
  SpaceBetweenBraces.SpacesInParensOptions.InEmptyParentheses = true;
  verifyFormat("vector< int > x{ };", SpaceBetweenBraces);
}

TEST_F(FormatTest, FormatsBracedListsInColumnLayout) {
  verifyFormat("vector<int> x = {1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777};");
  verifyFormat("vector<int> x = {1, 22, 333, 4444, 55555, 666666, 7777777, //\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, //\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777};");
  verifyFormat(
      "vector<int> x = {1,       22, 333, 4444, 55555, 666666, 7777777,\n"
      "                 1,       22, 333, 4444, 55555, 666666, 7777777,\n"
      "                 1,       22, 333, 4444, 55555, 666666, // comment\n"
      "                 7777777, 1,  22,  333,  4444,  55555,  666666,\n"
      "                 7777777, 1,  22,  333,  4444,  55555,  666666,\n"
      "                 7777777, 1,  22,  333,  4444,  55555,  666666,\n"
      "                 7777777};");
  verifyFormat("static const uint16_t CallerSavedRegs64Bittttt[] = {\n"
               "    X86::RAX, X86::RDX, X86::RCX, X86::RSI, X86::RDI,\n"
               "    X86::R8,  X86::R9,  X86::R10, X86::R11, 0};");
  verifyFormat("static const uint16_t CallerSavedRegs64Bittttt[] = {\n"
               "    X86::RAX, X86::RDX, X86::RCX, X86::RSI, X86::RDI,\n"
               "    // Separating comment.\n"
               "    X86::R8, X86::R9, X86::R10, X86::R11, 0};");
  verifyFormat("static const uint16_t CallerSavedRegs64Bittttt[] = {\n"
               "    // Leading comment\n"
               "    X86::RAX, X86::RDX, X86::RCX, X86::RSI, X86::RDI,\n"
               "    X86::R8,  X86::R9,  X86::R10, X86::R11, 0};");
  verifyFormat("vector<int> x = {1, 1, 1, 1,\n"
               "                 1, 1, 1, 1};",
               getLLVMStyleWithColumns(39));
  verifyFormat("vector<int> x = {1, 1, 1, 1,\n"
               "                 1, 1, 1, 1};",
               getLLVMStyleWithColumns(38));
  verifyFormat("vector<int> aaaaaaaaaaaaaaaaaaaaaa = {\n"
               "    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};",
               getLLVMStyleWithColumns(43));
  verifyFormat(
      "static unsigned SomeValues[10][3] = {\n"
      "    {1, 4, 0},  {4, 9, 0},  {4, 5, 9},  {8, 5, 4}, {1, 8, 4},\n"
      "    {10, 1, 6}, {11, 0, 9}, {2, 11, 9}, {5, 2, 9}, {11, 2, 7}};");
  verifyFormat("static auto fields = new vector<string>{\n"
               "    \"aaaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaaaaaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\n"
               "};");
  verifyFormat("vector<int> x = {1, 2, 3, 4, aaaaaaaaaaaaaaaaa, 6};");
  verifyFormat("vector<int> x = {1, aaaaaaaaaaaaaaaaaaaaaa,\n"
               "                 2, bbbbbbbbbbbbbbbbbbbbbb,\n"
               "                 3, cccccccccccccccccccccc};",
               getLLVMStyleWithColumns(60));

  // Trailing commas.
  verifyFormat("vector<int> x = {\n"
               "    1, 1, 1, 1, 1, 1, 1, 1,\n"
               "};",
               getLLVMStyleWithColumns(39));
  verifyFormat("vector<int> x = {\n"
               "    1, 1, 1, 1, 1, 1, 1, 1, //\n"
               "};",
               getLLVMStyleWithColumns(39));
  verifyFormat("vector<int> x = {1, 1, 1, 1,\n"
               "                 1, 1, 1, 1,\n"
               "                 /**/ /**/};",
               getLLVMStyleWithColumns(39));

  // Trailing comment in the first line.
  verifyFormat("vector<int> iiiiiiiiiiiiiii = {                      //\n"
               "    1111111111, 2222222222, 33333333333, 4444444444, //\n"
               "    111111111,  222222222,  3333333333,  444444444,  //\n"
               "    11111111,   22222222,   333333333,   44444444};");
  // Trailing comment in the last line.
  verifyFormat("int aaaaa[] = {\n"
               "    1, 2, 3, // comment\n"
               "    4, 5, 6  // comment\n"
               "};");

  // With nested lists, we should either format one item per line or all nested
  // lists one on line.
  // FIXME: For some nested lists, we can do better.
  verifyFormat("return {{aaaaaaaaaaaaaaaaaaaaa},\n"
               "        {aaaaaaaaaaaaaaaaaaa},\n"
               "        {aaaaaaaaaaaaaaaaaaaaa},\n"
               "        {aaaaaaaaaaaaaaaaa}};",
               getLLVMStyleWithColumns(60));
  verifyFormat(
      "SomeStruct my_struct_array = {\n"
      "    {aaaaaa, aaaaaaaa, aaaaaaaaaa, aaaaaaaaa, aaaaaaaaa, aaaaaaaaaa,\n"
      "     aaaaaaaaaaaaa, aaaaaaa, aaa},\n"
      "    {aaa, aaa},\n"
      "    {aaa, aaa},\n"
      "    {aaaa, aaaa, aaaa, aaaa, aaaa, aaaa, aaaa, aaa},\n"
      "    {aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "     aaaaaaaaaaaa, a, aaaaaaaaaa, aaaaaaaaa, aaa}};");

  // No column layout should be used here.
  verifyFormat("aaaaaaaaaaaaaaa = {aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, 0, 0,\n"
               "                   bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb};");

  verifyNoCrash("a<,");

  // No braced initializer here.
  verifyFormat("void f() {\n"
               "  struct Dummy {};\n"
               "  f(v);\n"
               "}");
  verifyFormat("void foo() {\n"
               "  { // asdf\n"
               "    { int a; }\n"
               "  }\n"
               "  {\n"
               "    { int b; }\n"
               "  }\n"
               "}");
  verifyFormat("namespace n {\n"
               "void foo() {\n"
               "  {\n"
               "    {\n"
               "      statement();\n"
               "      if (false) {\n"
               "      }\n"
               "    }\n"
               "  }\n"
               "  {}\n"
               "}\n"
               "} // namespace n");

  // Long lists should be formatted in columns even if they are nested.
  verifyFormat(
      "vector<int> x = function({1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777});");

  // Allow "single-column" layout even if that violates the column limit. There
  // isn't going to be a better way.
  verifyFormat("std::vector<int> a = {\n"
               "    aaaaaaaa,\n"
               "    aaaaaaaa,\n"
               "    aaaaaaaa,\n"
               "    aaaaaaaa,\n"
               "    aaaaaaaaaa,\n"
               "    aaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaa};",
               getLLVMStyleWithColumns(30));
  verifyFormat("vector<int> aaaa = {\n"
               "    aaaaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaa.aaaaaaa,\n"
               "    aaaaaa.aaaaaaa,\n"
               "    aaaaaa.aaaaaaa,\n"
               "    aaaaaa.aaaaaaa,\n"
               "};");

  // Don't create hanging lists.
  verifyFormat("someFunction(Param, {List1, List2,\n"
               "                     List3});",
               getLLVMStyleWithColumns(35));
  verifyFormat("someFunction(Param, Param,\n"
               "             {List1, List2,\n"
               "              List3});",
               getLLVMStyleWithColumns(35));
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaa, {},\n"
               "                               aaaaaaaaaaaaaaaaaaaaaaa);");
}

TEST_F(FormatTest, PullTrivialFunctionDefinitionsIntoSingleLine) {
  FormatStyle DoNotMerge = getLLVMStyle();
  DoNotMerge.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;

  verifyFormat("void f() { return 42; }");
  verifyFormat("void f() {\n"
               "  return 42;\n"
               "}",
               DoNotMerge);
  verifyFormat("void f() {\n"
               "  // Comment\n"
               "}");
  verifyFormat("{\n"
               "#error {\n"
               "  int a;\n"
               "}");
  verifyFormat("{\n"
               "  int a;\n"
               "#error {\n"
               "}");
  verifyFormat("void f() {} // comment");
  verifyFormat("void f() { int a; } // comment");
  verifyFormat("void f() {\n"
               "} // comment",
               DoNotMerge);
  verifyFormat("void f() {\n"
               "  int a;\n"
               "} // comment",
               DoNotMerge);
  verifyFormat("void f() {\n"
               "} // comment",
               getLLVMStyleWithColumns(15));

  verifyFormat("void f() { return 42; }", getLLVMStyleWithColumns(23));
  verifyFormat("void f() {\n  return 42;\n}", getLLVMStyleWithColumns(22));

  verifyFormat("void f() {}", getLLVMStyleWithColumns(11));
  verifyFormat("void f() {\n}", getLLVMStyleWithColumns(10));
  verifyGoogleFormat("class C {\n"
                     "  C()\n"
                     "      : iiiiiiii(nullptr),\n"
                     "        kkkkkkk(nullptr),\n"
                     "        mmmmmmm(nullptr),\n"
                     "        nnnnnnn(nullptr) {}\n"
                     "};");

  FormatStyle NoColumnLimit = getLLVMStyleWithColumns(0);
  verifyFormat("A() : b(0) {}", "A():b(0){}", NoColumnLimit);
  verifyFormat("class C {\n"
               "  A() : b(0) {}\n"
               "};",
               "class C{A():b(0){}};", NoColumnLimit);
  verifyFormat("A()\n"
               "    : b(0) {\n"
               "}",
               "A()\n:b(0)\n{\n}", NoColumnLimit);

  FormatStyle NoColumnLimitWrapAfterFunction = NoColumnLimit;
  NoColumnLimitWrapAfterFunction.BreakBeforeBraces = FormatStyle::BS_Custom;
  NoColumnLimitWrapAfterFunction.BraceWrapping.AfterFunction = true;
  verifyFormat("class C {\n"
               "#pragma foo\n"
               "  int foo { return 0; }\n"
               "};",
               NoColumnLimitWrapAfterFunction);
  verifyFormat("class C {\n"
               "#pragma foo\n"
               "  void foo {}\n"
               "};",
               NoColumnLimitWrapAfterFunction);

  FormatStyle DoNotMergeNoColumnLimit = NoColumnLimit;
  DoNotMergeNoColumnLimit.AllowShortFunctionsOnASingleLine =
      FormatStyle::SFS_None;
  verifyFormat("A()\n"
               "    : b(0) {\n"
               "}",
               "A():b(0){}", DoNotMergeNoColumnLimit);
  verifyFormat("A()\n"
               "    : b(0) {\n"
               "}",
               "A()\n:b(0)\n{\n}", DoNotMergeNoColumnLimit);

  verifyFormat("#define A          \\\n"
               "  void f() {       \\\n"
               "    int i;         \\\n"
               "  }",
               getLLVMStyleWithColumns(20));
  verifyFormat("#define A           \\\n"
               "  void f() { int i; }",
               getLLVMStyleWithColumns(21));
  verifyFormat("#define A            \\\n"
               "  void f() {         \\\n"
               "    int i;           \\\n"
               "  }                  \\\n"
               "  int j;",
               getLLVMStyleWithColumns(22));
  verifyFormat("#define A             \\\n"
               "  void f() { int i; } \\\n"
               "  int j;",
               getLLVMStyleWithColumns(23));
}

TEST_F(FormatTest, PullEmptyFunctionDefinitionsIntoSingleLine) {
  FormatStyle MergeEmptyOnly = getLLVMStyle();
  MergeEmptyOnly.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Empty;
  verifyFormat("class C {\n"
               "  int f() {}\n"
               "};",
               MergeEmptyOnly);
  verifyFormat("class C {\n"
               "  int f() {\n"
               "    return 42;\n"
               "  }\n"
               "};",
               MergeEmptyOnly);
  verifyFormat("int f() {}", MergeEmptyOnly);
  verifyFormat("int f() {\n"
               "  return 42;\n"
               "}",
               MergeEmptyOnly);

  // Also verify behavior when BraceWrapping.AfterFunction = true
  MergeEmptyOnly.BreakBeforeBraces = FormatStyle::BS_Custom;
  MergeEmptyOnly.BraceWrapping.AfterFunction = true;
  verifyFormat("int f() {}", MergeEmptyOnly);
  verifyFormat("class C {\n"
               "  int f() {}\n"
               "};",
               MergeEmptyOnly);
}

TEST_F(FormatTest, PullInlineFunctionDefinitionsIntoSingleLine) {
  FormatStyle MergeInlineOnly = getLLVMStyle();
  MergeInlineOnly.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  verifyFormat("class C {\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("int f() {\n"
               "  return 42;\n"
               "}",
               MergeInlineOnly);

  // SFS_Inline implies SFS_Empty
  verifyFormat("class C {\n"
               "  int f() {}\n"
               "};",
               MergeInlineOnly);
  verifyFormat("int f() {}", MergeInlineOnly);
  // https://llvm.org/PR54147
  verifyFormat("auto lambda = []() {\n"
               "  // comment\n"
               "  f();\n"
               "  g();\n"
               "};",
               MergeInlineOnly);

  verifyFormat("class C {\n"
               "#ifdef A\n"
               "  int f() { return 42; }\n"
               "#endif\n"
               "};",
               MergeInlineOnly);

  verifyFormat("struct S {\n"
               "// comment\n"
               "#ifdef FOO\n"
               "  int foo() { bar(); }\n"
               "#endif\n"
               "};",
               MergeInlineOnly);

  // Also verify behavior when BraceWrapping.AfterFunction = true
  MergeInlineOnly.BreakBeforeBraces = FormatStyle::BS_Custom;
  MergeInlineOnly.BraceWrapping.AfterFunction = true;
  verifyFormat("class C {\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("int f()\n"
               "{\n"
               "  return 42;\n"
               "}",
               MergeInlineOnly);

  // SFS_Inline implies SFS_Empty
  verifyFormat("int f() {}", MergeInlineOnly);
  verifyFormat("class C {\n"
               "  int f() {}\n"
               "};",
               MergeInlineOnly);

  MergeInlineOnly.BraceWrapping.AfterClass = true;
  MergeInlineOnly.BraceWrapping.AfterStruct = true;
  verifyFormat("class C\n"
               "{\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("struct C\n"
               "{\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("int f()\n"
               "{\n"
               "  return 42;\n"
               "}",
               MergeInlineOnly);
  verifyFormat("int f() {}", MergeInlineOnly);
  verifyFormat("class C\n"
               "{\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("struct C\n"
               "{\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("struct C\n"
               "// comment\n"
               "/* comment */\n"
               "// comment\n"
               "{\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("/* comment */ struct C\n"
               "{\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
}

TEST_F(FormatTest, PullInlineOnlyFunctionDefinitionsIntoSingleLine) {
  FormatStyle MergeInlineOnly = getLLVMStyle();
  MergeInlineOnly.AllowShortFunctionsOnASingleLine =
      FormatStyle::SFS_InlineOnly;
  verifyFormat("class C {\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("int f() {\n"
               "  return 42;\n"
               "}",
               MergeInlineOnly);

  // SFS_InlineOnly does not imply SFS_Empty
  verifyFormat("class C {\n"
               "  int f() {}\n"
               "};",
               MergeInlineOnly);
  verifyFormat("int f() {\n"
               "}",
               MergeInlineOnly);

  // Also verify behavior when BraceWrapping.AfterFunction = true
  MergeInlineOnly.BreakBeforeBraces = FormatStyle::BS_Custom;
  MergeInlineOnly.BraceWrapping.AfterFunction = true;
  verifyFormat("class C {\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("int f()\n"
               "{\n"
               "  return 42;\n"
               "}",
               MergeInlineOnly);

  // SFS_InlineOnly does not imply SFS_Empty
  verifyFormat("int f()\n"
               "{\n"
               "}",
               MergeInlineOnly);
  verifyFormat("class C {\n"
               "  int f() {}\n"
               "};",
               MergeInlineOnly);
}

TEST_F(FormatTest, SplitEmptyFunction) {
  FormatStyle Style = getLLVMStyleWithColumns(40);
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterFunction = true;
  Style.BraceWrapping.SplitEmptyFunction = false;

  verifyFormat("int f()\n"
               "{}",
               Style);
  verifyFormat("int f()\n"
               "{\n"
               "  return 42;\n"
               "}",
               Style);
  verifyFormat("int f()\n"
               "{\n"
               "  // some comment\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Empty;
  verifyFormat("int f() {}", Style);
  verifyFormat("int aaaaaaaaaaaaaa(int bbbbbbbbbbbbbb)\n"
               "{}",
               Style);
  verifyFormat("int f()\n"
               "{\n"
               "  return 0;\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  verifyFormat("class Foo {\n"
               "  int f() {}\n"
               "};",
               Style);
  verifyFormat("class Foo {\n"
               "  int f() { return 0; }\n"
               "};",
               Style);
  verifyFormat("class Foo {\n"
               "  int aaaaaaaaaaaaaa(int bbbbbbbbbbbbbb)\n"
               "  {}\n"
               "};",
               Style);
  verifyFormat("class Foo {\n"
               "  int aaaaaaaaaaaaaa(int bbbbbbbbbbbbbb)\n"
               "  {\n"
               "    return 0;\n"
               "  }\n"
               "};",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  verifyFormat("int f() {}", Style);
  verifyFormat("int f() { return 0; }", Style);
  verifyFormat("int aaaaaaaaaaaaaa(int bbbbbbbbbbbbbb)\n"
               "{}",
               Style);
  verifyFormat("int aaaaaaaaaaaaaa(int bbbbbbbbbbbbbb)\n"
               "{\n"
               "  return 0;\n"
               "}",
               Style);
}

TEST_F(FormatTest, SplitEmptyFunctionButNotRecord) {
  FormatStyle Style = getLLVMStyleWithColumns(40);
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterFunction = true;
  Style.BraceWrapping.SplitEmptyFunction = true;
  Style.BraceWrapping.SplitEmptyRecord = false;

  verifyFormat("class C {};", Style);
  verifyFormat("struct C {};", Style);
  verifyFormat("void f(int aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       int bbbbbbbbbbbbbbbbbbbbbbbb)\n"
               "{\n"
               "}",
               Style);
  verifyFormat("class C {\n"
               "  C()\n"
               "      : aaaaaaaaaaaaaaaaaaaaaaaaaaaa(),\n"
               "        bbbbbbbbbbbbbbbbbbb()\n"
               "  {\n"
               "  }\n"
               "  void\n"
               "  m(int aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbbbbbbbbbbbbb)\n"
               "  {\n"
               "  }\n"
               "};",
               Style);
}

TEST_F(FormatTest, KeepShortFunctionAfterPPElse) {
  FormatStyle Style = getLLVMStyle();
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  verifyFormat("#ifdef A\n"
               "int f() {}\n"
               "#else\n"
               "int g() {}\n"
               "#endif",
               Style);
}

TEST_F(FormatTest, SplitEmptyClass) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterClass = true;
  Style.BraceWrapping.SplitEmptyRecord = false;

  verifyFormat("class Foo\n"
               "{};",
               Style);
  verifyFormat("/* something */ class Foo\n"
               "{};",
               Style);
  verifyFormat("template <typename X> class Foo\n"
               "{};",
               Style);
  verifyFormat("class Foo\n"
               "{\n"
               "  Foo();\n"
               "};",
               Style);
  verifyFormat("typedef class Foo\n"
               "{\n"
               "} Foo_t;",
               Style);

  Style.BraceWrapping.SplitEmptyRecord = true;
  Style.BraceWrapping.AfterStruct = true;
  verifyFormat("class rep\n"
               "{\n"
               "};",
               Style);
  verifyFormat("struct rep\n"
               "{\n"
               "};",
               Style);
  verifyFormat("template <typename T> class rep\n"
               "{\n"
               "};",
               Style);
  verifyFormat("template <typename T> struct rep\n"
               "{\n"
               "};",
               Style);
  verifyFormat("class rep\n"
               "{\n"
               "  int x;\n"
               "};",
               Style);
  verifyFormat("struct rep\n"
               "{\n"
               "  int x;\n"
               "};",
               Style);
  verifyFormat("template <typename T> class rep\n"
               "{\n"
               "  int x;\n"
               "};",
               Style);
  verifyFormat("template <typename T> struct rep\n"
               "{\n"
               "  int x;\n"
               "};",
               Style);
  verifyFormat("template <typename T> class rep // Foo\n"
               "{\n"
               "  int x;\n"
               "};",
               Style);
  verifyFormat("template <typename T> struct rep // Bar\n"
               "{\n"
               "  int x;\n"
               "};",
               Style);

  verifyFormat("template <typename T> class rep<T>\n"
               "{\n"
               "  int x;\n"
               "};",
               Style);

  verifyFormat("template <typename T> class rep<std::complex<T>>\n"
               "{\n"
               "  int x;\n"
               "};",
               Style);
  verifyFormat("template <typename T> class rep<std::complex<T>>\n"
               "{\n"
               "};",
               Style);

  verifyFormat("#include \"stdint.h\"\n"
               "namespace rep {}",
               Style);
  verifyFormat("#include <stdint.h>\n"
               "namespace rep {}",
               Style);
  verifyFormat("#include <stdint.h>\n"
               "namespace rep {}",
               "#include <stdint.h>\n"
               "namespace rep {\n"
               "\n"
               "\n"
               "}",
               Style);
}

TEST_F(FormatTest, SplitEmptyStruct) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterStruct = true;
  Style.BraceWrapping.SplitEmptyRecord = false;

  verifyFormat("struct Foo\n"
               "{};",
               Style);
  verifyFormat("/* something */ struct Foo\n"
               "{};",
               Style);
  verifyFormat("template <typename X> struct Foo\n"
               "{};",
               Style);
  verifyFormat("struct Foo\n"
               "{\n"
               "  Foo();\n"
               "};",
               Style);
  verifyFormat("typedef struct Foo\n"
               "{\n"
               "} Foo_t;",
               Style);
  // typedef struct Bar {} Bar_t;
}

TEST_F(FormatTest, SplitEmptyUnion) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterUnion = true;
  Style.BraceWrapping.SplitEmptyRecord = false;

  verifyFormat("union Foo\n"
               "{};",
               Style);
  verifyFormat("/* something */ union Foo\n"
               "{};",
               Style);
  verifyFormat("union Foo\n"
               "{\n"
               "  A,\n"
               "};",
               Style);
  verifyFormat("typedef union Foo\n"
               "{\n"
               "} Foo_t;",
               Style);
}

TEST_F(FormatTest, SplitEmptyNamespace) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterNamespace = true;
  Style.BraceWrapping.SplitEmptyNamespace = false;

  verifyFormat("namespace Foo\n"
               "{};",
               Style);
  verifyFormat("/* something */ namespace Foo\n"
               "{};",
               Style);
  verifyFormat("inline namespace Foo\n"
               "{};",
               Style);
  verifyFormat("/* something */ inline namespace Foo\n"
               "{};",
               Style);
  verifyFormat("export namespace Foo\n"
               "{};",
               Style);
  verifyFormat("namespace Foo\n"
               "{\n"
               "void Bar();\n"
               "};",
               Style);
}

TEST_F(FormatTest, NeverMergeShortRecords) {
  FormatStyle Style = getLLVMStyle();

  verifyFormat("class Foo {\n"
               "  Foo();\n"
               "};",
               Style);
  verifyFormat("typedef class Foo {\n"
               "  Foo();\n"
               "} Foo_t;",
               Style);
  verifyFormat("struct Foo {\n"
               "  Foo();\n"
               "};",
               Style);
  verifyFormat("typedef struct Foo {\n"
               "  Foo();\n"
               "} Foo_t;",
               Style);
  verifyFormat("union Foo {\n"
               "  A,\n"
               "};",
               Style);
  verifyFormat("typedef union Foo {\n"
               "  A,\n"
               "} Foo_t;",
               Style);
  verifyFormat("namespace Foo {\n"
               "void Bar();\n"
               "};",
               Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterClass = true;
  Style.BraceWrapping.AfterStruct = true;
  Style.BraceWrapping.AfterUnion = true;
  Style.BraceWrapping.AfterNamespace = true;
  verifyFormat("class Foo\n"
               "{\n"
               "  Foo();\n"
               "};",
               Style);
  verifyFormat("typedef class Foo\n"
               "{\n"
               "  Foo();\n"
               "} Foo_t;",
               Style);
  verifyFormat("struct Foo\n"
               "{\n"
               "  Foo();\n"
               "};",
               Style);
  verifyFormat("typedef struct Foo\n"
               "{\n"
               "  Foo();\n"
               "} Foo_t;",
               Style);
  verifyFormat("union Foo\n"
               "{\n"
               "  A,\n"
               "};",
               Style);
  verifyFormat("typedef union Foo\n"
               "{\n"
               "  A,\n"
               "} Foo_t;",
               Style);
  verifyFormat("namespace Foo\n"
               "{\n"
               "void Bar();\n"
               "};",
               Style);
}

TEST_F(FormatTest, UnderstandContextOfRecordTypeKeywords) {
  // Elaborate type variable declarations.
  verifyFormat("struct foo a = {bar};\nint n;");
  verifyFormat("class foo a = {bar};\nint n;");
  verifyFormat("union foo a = {bar};\nint n;");

  // Elaborate types inside function definitions.
  verifyFormat("struct foo f() {}\nint n;");
  verifyFormat("class foo f() {}\nint n;");
  verifyFormat("union foo f() {}\nint n;");

  // Templates.
  verifyFormat("template <class X> void f() {}\nint n;");
  verifyFormat("template <struct X> void f() {}\nint n;");
  verifyFormat("template <union X> void f() {}\nint n;");

  // Actual definitions...
  verifyFormat("struct {\n} n;");
  verifyFormat(
      "template <template <class T, class Y>, class Z> class X {\n} n;");
  verifyFormat("union Z {\n  int n;\n} x;");
  verifyFormat("class MACRO Z {\n} n;");
  verifyFormat("class MACRO(X) Z {\n} n;");
  verifyFormat("class __attribute__(X) Z {\n} n;");
  verifyFormat("class __declspec(X) Z {\n} n;");
  verifyFormat("class A##B##C {\n} n;");
  verifyFormat("class alignas(16) Z {\n} n;");
  verifyFormat("class MACRO(X) alignas(16) Z {\n} n;");
  verifyFormat("class MACROA MACRO(X) Z {\n} n;");

  // Redefinition from nested context:
  verifyFormat("class A::B::C {\n} n;");

  // Template definitions.
  verifyFormat(
      "template <typename F>\n"
      "Matcher(const Matcher<F> &Other,\n"
      "        typename enable_if_c<is_base_of<F, T>::value &&\n"
      "                             !is_same<F, T>::value>::type * = 0)\n"
      "    : Implementation(new ImplicitCastMatcher<F>(Other)) {}");

  // FIXME: This is still incorrectly handled at the formatter side.
  verifyFormat("template <> struct X < 15, i<3 && 42 < 50 && 33 < 28> {};");
  verifyFormat("int i = SomeFunction(a<b, a> b);");

  // FIXME:
  // This now gets parsed incorrectly as class definition.
  // verifyFormat("class A<int> f() {\n}\nint n;");

  // Elaborate types where incorrectly parsing the structural element would
  // break the indent.
  verifyFormat("if (true)\n"
               "  class X x;\n"
               "else\n"
               "  f();");

  // This is simply incomplete. Formatting is not important, but must not crash.
  verifyFormat("class A:");
}

TEST_F(FormatTest, DoNotInterfereWithErrorAndWarning) {
  verifyNoChange("#error Leave     all         white!!!!! space* alone!");
  verifyNoChange("#warning Leave     all         white!!!!! space* alone!");
  verifyFormat("#error 1", "  #  error   1");
  verifyFormat("#warning 1", "  #  warning 1");
}

TEST_F(FormatTest, FormatHashIfExpressions) {
  verifyFormat("#if AAAA && BBBB");
  verifyFormat("#if (AAAA && BBBB)");
  verifyFormat("#elif (AAAA && BBBB)");
  // FIXME: Come up with a better indentation for #elif.
  verifyFormat(
      "#if !defined(AAAAAAA) && (defined CCCCCC || defined DDDDDD) &&  \\\n"
      "    defined(BBBBBBBB)\n"
      "#elif !defined(AAAAAA) && (defined CCCCC || defined DDDDDD) &&  \\\n"
      "    defined(BBBBBBBB)\n"
      "#endif",
      getLLVMStyleWithColumns(65));
}

TEST_F(FormatTest, MergeHandlingInTheFaceOfPreprocessorDirectives) {
  FormatStyle AllowsMergedIf = getGoogleStyle();
  AllowsMergedIf.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_WithoutElse;
  verifyFormat("void f() { f(); }\n#error E", AllowsMergedIf);
  verifyFormat("if (true) return 42;\n#error E", AllowsMergedIf);
  verifyFormat("if (true)\n#error E\n  return 42;", AllowsMergedIf);
  verifyFormat("if (true) return 42;", "if (true)\nreturn 42;", AllowsMergedIf);
  FormatStyle ShortMergedIf = AllowsMergedIf;
  ShortMergedIf.ColumnLimit = 25;
  verifyFormat("#define A \\\n"
               "  if (true) return 42;",
               ShortMergedIf);
  verifyFormat("#define A \\\n"
               "  f();    \\\n"
               "  if (true)\n"
               "#define B",
               ShortMergedIf);
  verifyFormat("#define A \\\n"
               "  f();    \\\n"
               "  if (true)\n"
               "g();",
               ShortMergedIf);
  verifyFormat("{\n"
               "#ifdef A\n"
               "  // Comment\n"
               "  if (true) continue;\n"
               "#endif\n"
               "  // Comment\n"
               "  if (true) continue;\n"
               "}",
               ShortMergedIf);
  ShortMergedIf.ColumnLimit = 33;
  verifyFormat("#define A \\\n"
               "  if constexpr (true) return 42;",
               ShortMergedIf);
  verifyFormat("#define A \\\n"
               "  if CONSTEXPR (true) return 42;",
               ShortMergedIf);
  ShortMergedIf.ColumnLimit = 29;
  verifyFormat("#define A                   \\\n"
               "  if (aaaaaaaaaa) return 1; \\\n"
               "  return 2;",
               ShortMergedIf);
  ShortMergedIf.ColumnLimit = 28;
  verifyFormat("#define A         \\\n"
               "  if (aaaaaaaaaa) \\\n"
               "    return 1;     \\\n"
               "  return 2;",
               ShortMergedIf);
  verifyFormat("#define A                \\\n"
               "  if constexpr (aaaaaaa) \\\n"
               "    return 1;            \\\n"
               "  return 2;",
               ShortMergedIf);
  verifyFormat("#define A                \\\n"
               "  if CONSTEXPR (aaaaaaa) \\\n"
               "    return 1;            \\\n"
               "  return 2;",
               ShortMergedIf);

  verifyFormat("//\n"
               "#define a \\\n"
               "  if      \\\n"
               "  0",
               getChromiumStyle(FormatStyle::LK_Cpp));
}

TEST_F(FormatTest, FormatStarDependingOnContext) {
  verifyFormat("void f(int *a);");
  verifyFormat("void f() { f(fint * b); }");
  verifyFormat("class A {\n  void f(int *a);\n};");
  verifyFormat("class A {\n  int *a;\n};");
  verifyFormat("namespace a {\n"
               "namespace b {\n"
               "class A {\n"
               "  void f() {}\n"
               "  int *a;\n"
               "};\n"
               "} // namespace b\n"
               "} // namespace a");
}

TEST_F(FormatTest, SpecialTokensAtEndOfLine) {
  verifyFormat("while");
  verifyFormat("operator");
}

TEST_F(FormatTest, SkipsDeeplyNestedLines) {
  // This code would be painfully slow to format if we didn't skip it.
  std::string Code("A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(\n" // 20x
                   "A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(\n"
                   "A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(\n"
                   "A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(\n"
                   "A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(A(\n"
                   "A(1, 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n" // 10x
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1)\n"
                   ", 1), 1), 1), 1), 1), 1), 1), 1), 1), 1);\n");
  // Deeply nested part is untouched, rest is formatted.
  EXPECT_EQ(std::string("int i;") + Code + "int j;",
            format(std::string("int    i;") + Code + "int    j;",
                   getLLVMStyle(), SC_ExpectIncomplete));
}

//===----------------------------------------------------------------------===//
// Objective-C tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, FormatForObjectiveCMethodDecls) {
  verifyFormat("- (void)sendAction:(SEL)aSelector to:(BOOL)anObject;");
  verifyFormat("- (NSUInteger)indexOfObject:(id)anObject;",
               "-(NSUInteger)indexOfObject:(id)anObject;");
  verifyFormat("- (NSInteger)Mthod1;", "-(NSInteger)Mthod1;");
  verifyFormat("+ (id)Mthod2;", "+(id)Mthod2;");
  verifyFormat("- (NSInteger)Method3:(id)anObject;",
               "-(NSInteger)Method3:(id)anObject;");
  verifyFormat("- (NSInteger)Method4:(id)anObject;",
               "-(NSInteger)Method4:(id)anObject;");
  verifyFormat("- (NSInteger)Method5:(id)anObject:(id)AnotherObject;",
               "-(NSInteger)Method5:(id)anObject:(id)AnotherObject;");
  verifyFormat("- (id)Method6:(id)A:(id)B:(id)C:(id)D;");
  verifyFormat("- (void)sendAction:(SEL)aSelector to:(id)anObject "
               "forAllCells:(BOOL)flag;");

  // Very long objectiveC method declaration.
  verifyFormat("- (void)aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:\n"
               "    (SoooooooooooooooooooooomeType *)bbbbbbbbbb;");
  verifyFormat("- (NSUInteger)indexOfObject:(id)anObject\n"
               "                    inRange:(NSRange)range\n"
               "                   outRange:(NSRange)out_range\n"
               "                  outRange1:(NSRange)out_range1\n"
               "                  outRange2:(NSRange)out_range2\n"
               "                  outRange3:(NSRange)out_range3\n"
               "                  outRange4:(NSRange)out_range4\n"
               "                  outRange5:(NSRange)out_range5\n"
               "                  outRange6:(NSRange)out_range6\n"
               "                  outRange7:(NSRange)out_range7\n"
               "                  outRange8:(NSRange)out_range8\n"
               "                  outRange9:(NSRange)out_range9;");

  // When the function name has to be wrapped.
  FormatStyle Style = getLLVMStyle();
  // ObjC ignores IndentWrappedFunctionNames when wrapping methods
  // and always indents instead.
  Style.IndentWrappedFunctionNames = false;
  verifyFormat("- (SomeLooooooooooooooooooooongType *)\n"
               "    veryLooooooooooongName:(NSString)aaaaaaaaaaaaaa\n"
               "               anotherName:(NSString)bbbbbbbbbbbbbb {\n"
               "}",
               Style);
  Style.IndentWrappedFunctionNames = true;
  verifyFormat("- (SomeLooooooooooooooooooooongType *)\n"
               "    veryLooooooooooongName:(NSString)cccccccccccccc\n"
               "               anotherName:(NSString)dddddddddddddd {\n"
               "}",
               Style);

  verifyFormat("- (int)sum:(vector<int>)numbers;");
  verifyGoogleFormat("- (void)setDelegate:(id<Protocol>)delegate;");
  // FIXME: In LLVM style, there should be a space in front of a '<' for ObjC
  // protocol lists (but not for template classes):
  // verifyFormat("- (void)setDelegate:(id <Protocol>)delegate;");

  verifyFormat("- (int (*)())foo:(int (*)())f;");
  verifyGoogleFormat("- (int (*)())foo:(int (*)())foo;");

  // If there's no return type (very rare in practice!), LLVM and Google style
  // agree.
  verifyFormat("- foo;");
  verifyFormat("- foo:(int)f;");
  verifyGoogleFormat("- foo:(int)foo;");
}

TEST_F(FormatTest, BreaksStringLiterals) {
  // FIXME: unstable test case
  EXPECT_EQ("\"some text \"\n"
            "\"other\";",
            format("\"some text other\";", getLLVMStyleWithColumns(12)));
  // FIXME: unstable test case
  EXPECT_EQ("\"some text \"\n"
            "\"other\";",
            format("\\\n\"some text other\";", getLLVMStyleWithColumns(12)));
  verifyFormat("#define A  \\\n"
               "  \"some \"  \\\n"
               "  \"text \"  \\\n"
               "  \"other\";",
               "#define A \"some text other\";", getLLVMStyleWithColumns(12));
  verifyFormat("#define A  \\\n"
               "  \"so \"    \\\n"
               "  \"text \"  \\\n"
               "  \"other\";",
               "#define A \"so text other\";", getLLVMStyleWithColumns(12));

  verifyFormat("\"some text\"", getLLVMStyleWithColumns(1));
  verifyFormat("\"some text\"", getLLVMStyleWithColumns(11));
  // FIXME: unstable test case
  EXPECT_EQ("\"some \"\n"
            "\"text\"",
            format("\"some text\"", getLLVMStyleWithColumns(10)));
  // FIXME: unstable test case
  EXPECT_EQ("\"some \"\n"
            "\"text\"",
            format("\"some text\"", getLLVMStyleWithColumns(7)));
  // FIXME: unstable test case
  EXPECT_EQ("\"some\"\n"
            "\" tex\"\n"
            "\"t\"",
            format("\"some text\"", getLLVMStyleWithColumns(6)));
  // FIXME: unstable test case
  EXPECT_EQ("\"some\"\n"
            "\" tex\"\n"
            "\" and\"",
            format("\"some tex and\"", getLLVMStyleWithColumns(6)));
  // FIXME: unstable test case
  EXPECT_EQ("\"some\"\n"
            "\"/tex\"\n"
            "\"/and\"",
            format("\"some/tex/and\"", getLLVMStyleWithColumns(6)));

  verifyFormat("variable =\n"
               "    \"long string \"\n"
               "    \"literal\";",
               "variable = \"long string literal\";",
               getLLVMStyleWithColumns(20));

  verifyFormat("variable = f(\n"
               "    \"long string \"\n"
               "    \"literal\",\n"
               "    short,\n"
               "    loooooooooooooooooooong);",
               "variable = f(\"long string literal\", short, "
               "loooooooooooooooooooong);",
               getLLVMStyleWithColumns(20));

  verifyFormat("f(g(\"long string \"\n"
               "    \"literal\"),\n"
               "  b);",
               "f(g(\"long string literal\"), b);",
               getLLVMStyleWithColumns(20));
  verifyFormat("f(g(\"long string \"\n"
               "    \"literal\",\n"
               "    a),\n"
               "  b);",
               "f(g(\"long string literal\", a), b);",
               getLLVMStyleWithColumns(20));
  verifyFormat("f(\"one two\".split(\n"
               "    variable));",
               "f(\"one two\".split(variable));", getLLVMStyleWithColumns(20));
  verifyFormat("f(\"one two three four five six \"\n"
               "  \"seven\".split(\n"
               "      really_looooong_variable));",
               "f(\"one two three four five six seven\"."
               "split(really_looooong_variable));",
               getLLVMStyleWithColumns(33));

  verifyFormat("f(\"some \"\n"
               "  \"text\",\n"
               "  other);",
               "f(\"some text\", other);", getLLVMStyleWithColumns(10));

  // Only break as a last resort.
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaa,\n"
      "    aaaaaa(\"aaa aaaaa aaa aaa aaaaa aaa aaaaa aaa aaa aaaaaa\"));");

  // FIXME: unstable test case
  EXPECT_EQ("\"splitmea\"\n"
            "\"trandomp\"\n"
            "\"oint\"",
            format("\"splitmeatrandompoint\"", getLLVMStyleWithColumns(10)));

  // FIXME: unstable test case
  EXPECT_EQ("\"split/\"\n"
            "\"pathat/\"\n"
            "\"slashes\"",
            format("\"split/pathat/slashes\"", getLLVMStyleWithColumns(10)));

  // FIXME: unstable test case
  EXPECT_EQ("\"split/\"\n"
            "\"pathat/\"\n"
            "\"slashes\"",
            format("\"split/pathat/slashes\"", getLLVMStyleWithColumns(10)));
  // FIXME: unstable test case
  EXPECT_EQ("\"split at \"\n"
            "\"spaces/at/\"\n"
            "\"slashes.at.any$\"\n"
            "\"non-alphanumeric%\"\n"
            "\"1111111111characte\"\n"
            "\"rs\"",
            format("\"split at "
                   "spaces/at/"
                   "slashes.at."
                   "any$non-"
                   "alphanumeric%"
                   "1111111111characte"
                   "rs\"",
                   getLLVMStyleWithColumns(20)));

  // Verify that splitting the strings understands
  // Style::AlwaysBreakBeforeMultilineStrings.
  verifyFormat("aaaaaaaaaaaa(\n"
               "    \"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaa \"\n"
               "    \"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaa\");",
               "aaaaaaaaaaaa(\"aaaaaaaaaaaaaaaaaaaaaaaaaa "
               "aaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaa "
               "aaaaaaaaaaaaaaaaaaaaaa\");",
               getGoogleStyle());
  verifyFormat("return \"aaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaa \"\n"
               "       \"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaa\";",
               "return \"aaaaaaaaaaaaaaaaaaaaaa "
               "aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaa "
               "aaaaaaaaaaaaaaaaaaaaaa\";",
               getGoogleStyle());
  verifyFormat("llvm::outs() << \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \"\n"
               "                \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";",
               "llvm::outs() << "
               "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaa"
               "aaaaaaaaaaaaaaaaaaa\";");
  verifyFormat("ffff(\n"
               "    {\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \"\n"
               "     \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"});",
               "ffff({\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"});",
               getGoogleStyle());

  FormatStyle Style = getLLVMStyleWithColumns(12);
  Style.BreakStringLiterals = false;
  verifyFormat("\"some text other\";", Style);

  FormatStyle AlignLeft = getLLVMStyleWithColumns(12);
  AlignLeft.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  verifyFormat("#define A \\\n"
               "  \"some \" \\\n"
               "  \"text \" \\\n"
               "  \"other\";",
               "#define A \"some text other\";", AlignLeft);
}

TEST_F(FormatTest, BreaksStringLiteralsAtColumnLimit) {
  verifyFormat("C a = \"some more \"\n"
               "      \"text\";",
               "C a = \"some more text\";", getLLVMStyleWithColumns(18));
}

TEST_F(FormatTest, FullyRemoveEmptyLines) {
  FormatStyle NoEmptyLines = getLLVMStyleWithColumns(80);
  NoEmptyLines.MaxEmptyLinesToKeep = 0;
  verifyFormat("int i = a(b());", "int i=a(\n\n b(\n\n\n )\n\n);",
               NoEmptyLines);
}

TEST_F(FormatTest, BreaksStringLiteralsWithTabs) {
  // FIXME: unstable test case
  EXPECT_EQ(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "(\n"
      "    \"x\t\");",
      format("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
             "aaaaaaa("
             "\"x\t\");"));
}

TEST_F(FormatTest, BreaksWideAndNSStringLiterals) {
  // FIXME: unstable test case
  EXPECT_EQ(
      "u8\"utf8 string \"\n"
      "u8\"literal\";",
      format("u8\"utf8 string literal\";", getGoogleStyleWithColumns(16)));
  // FIXME: unstable test case
  EXPECT_EQ(
      "u\"utf16 string \"\n"
      "u\"literal\";",
      format("u\"utf16 string literal\";", getGoogleStyleWithColumns(16)));
  // FIXME: unstable test case
  EXPECT_EQ(
      "U\"utf32 string \"\n"
      "U\"literal\";",
      format("U\"utf32 string literal\";", getGoogleStyleWithColumns(16)));
  // FIXME: unstable test case
  EXPECT_EQ("L\"wide string \"\n"
            "L\"literal\";",
            format("L\"wide string literal\";", getGoogleStyleWithColumns(16)));
  verifyFormat("@\"NSString \"\n"
               "@\"literal\";",
               "@\"NSString literal\";", getGoogleStyleWithColumns(19));
  verifyFormat(R"(NSString *s = @"那那那那";)", getLLVMStyleWithColumns(26));

  // This input makes clang-format try to split the incomplete unicode escape
  // sequence, which used to lead to a crasher.
  verifyNoCrash(
      "aaaaaaaaaaaaaaaaaaaa = L\"\\udff\"'; // aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      getLLVMStyleWithColumns(60));
}

TEST_F(FormatTest, DoesNotBreakRawStringLiterals) {
  FormatStyle Style = getGoogleStyleWithColumns(15);
  verifyFormat("R\"x(raw literal)x\";", Style);
  verifyFormat("uR\"x(raw literal)x\";", Style);
  verifyFormat("LR\"x(raw literal)x\";", Style);
  verifyFormat("UR\"x(raw literal)x\";", Style);
  verifyFormat("u8R\"x(raw literal)x\";", Style);
}

TEST_F(FormatTest, BreaksStringLiteralsWithin_TMacro) {
  FormatStyle Style = getLLVMStyleWithColumns(20);
  // FIXME: unstable test case
  EXPECT_EQ(
      "_T(\"aaaaaaaaaaaaaa\")\n"
      "_T(\"aaaaaaaaaaaaaa\")\n"
      "_T(\"aaaaaaaaaaaa\")",
      format("  _T(\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\")", Style));
  verifyFormat("f(x,\n"
               "  _T(\"aaaaaaaaaaaa\")\n"
               "  _T(\"aaa\"),\n"
               "  z);",
               "f(x, _T(\"aaaaaaaaaaaaaaa\"), z);", Style);

  // FIXME: Handle embedded spaces in one iteration.
  //  EXPECT_EQ("_T(\"aaaaaaaaaaaaa\")\n"
  //            "_T(\"aaaaaaaaaaaaa\")\n"
  //            "_T(\"aaaaaaaaaaaaa\")\n"
  //            "_T(\"a\")",
  //            format("  _T ( \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" )",
  //                   getLLVMStyleWithColumns(20)));
  verifyFormat("_T ( \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" )",
               "  _T ( \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" )", Style);
  verifyFormat("f(\n"
               "#if !TEST\n"
               "    _T(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXn\")\n"
               "#endif\n"
               ");",
               "f(\n"
               "#if !TEST\n"
               "_T(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXn\")\n"
               "#endif\n"
               ");");
  verifyFormat("f(\n"
               "\n"
               "    _T(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXn\"));",
               "f(\n"
               "\n"
               "_T(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXn\"));");
  // Regression test for accessing tokens past the end of a vector in the
  // TokenLexer.
  verifyNoCrash(R"(_T(
"
)
)");
}

TEST_F(FormatTest, BreaksStringLiteralOperands) {
  // In a function call with two operands, the second can be broken with no line
  // break before it.
  verifyFormat("func(a, \"long long \"\n"
               "        \"long long\");",
               "func(a, \"long long long long\");",
               getLLVMStyleWithColumns(24));
  // In a function call with three operands, the second must be broken with a
  // line break before it.
  verifyFormat("func(a,\n"
               "     \"long long long \"\n"
               "     \"long\",\n"
               "     c);",
               "func(a, \"long long long long\", c);",
               getLLVMStyleWithColumns(24));
  // In a function call with three operands, the third must be broken with a
  // line break before it.
  verifyFormat("func(a, b,\n"
               "     \"long long long \"\n"
               "     \"long\");",
               "func(a, b, \"long long long long\");",
               getLLVMStyleWithColumns(24));
  // In a function call with three operands, both the second and the third must
  // be broken with a line break before them.
  verifyFormat("func(a,\n"
               "     \"long long long \"\n"
               "     \"long\",\n"
               "     \"long long long \"\n"
               "     \"long\");",
               "func(a, \"long long long long\", \"long long long long\");",
               getLLVMStyleWithColumns(24));
  // In a chain of << with two operands, the second can be broken with no line
  // break before it.
  verifyFormat("a << \"line line \"\n"
               "     \"line\";",
               "a << \"line line line\";", getLLVMStyleWithColumns(20));
  // In a chain of << with three operands, the second can be broken with no line
  // break before it.
  verifyFormat("abcde << \"line \"\n"
               "         \"line line\"\n"
               "      << c;",
               "abcde << \"line line line\" << c;",
               getLLVMStyleWithColumns(20));
  // In a chain of << with three operands, the third must be broken with a line
  // break before it.
  verifyFormat("a << b\n"
               "  << \"line line \"\n"
               "     \"line\";",
               "a << b << \"line line line\";", getLLVMStyleWithColumns(20));
  // In a chain of << with three operands, the second can be broken with no line
  // break before it and the third must be broken with a line break before it.
  verifyFormat("abcd << \"line line \"\n"
               "        \"line\"\n"
               "     << \"line line \"\n"
               "        \"line\";",
               "abcd << \"line line line\" << \"line line line\";",
               getLLVMStyleWithColumns(20));
  // In a chain of binary operators with two operands, the second can be broken
  // with no line break before it.
  verifyFormat("abcd + \"line line \"\n"
               "       \"line line\";",
               "abcd + \"line line line line\";", getLLVMStyleWithColumns(20));
  // In a chain of binary operators with three operands, the second must be
  // broken with a line break before it.
  verifyFormat("abcd +\n"
               "    \"line line \"\n"
               "    \"line line\" +\n"
               "    e;",
               "abcd + \"line line line line\" + e;",
               getLLVMStyleWithColumns(20));
  // In a function call with two operands, with AlignAfterOpenBracket enabled,
  // the first must be broken with a line break before it.
  FormatStyle Style = getLLVMStyleWithColumns(25);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  verifyFormat("someFunction(\n"
               "    \"long long long \"\n"
               "    \"long\",\n"
               "    a);",
               "someFunction(\"long long long long\", a);", Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;
  verifyFormat("someFunction(\n"
               "    \"long long long \"\n"
               "    \"long\",\n"
               "    a\n"
               ");",
               Style);
}

TEST_F(FormatTest, DontSplitStringLiteralsWithEscapedNewlines) {
  verifyFormat("aaaaaaaaaaa = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";",
               "aaaaaaaaaaa  =  \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";");
}

TEST_F(FormatTest, CountsCharactersInMultilineRawStringLiterals) {
  verifyFormat("f(g(R\"x(raw literal)x\", a), b);",
               "f(g(R\"x(raw literal)x\",   a), b);", getGoogleStyle());
  verifyFormat("fffffffffff(g(R\"x(\n"
               "multiline raw string literal xxxxxxxxxxxxxx\n"
               ")x\",\n"
               "              a),\n"
               "            b);",
               "fffffffffff(g(R\"x(\n"
               "multiline raw string literal xxxxxxxxxxxxxx\n"
               ")x\", a), b);",
               getGoogleStyleWithColumns(20));
  verifyFormat("fffffffffff(\n"
               "    g(R\"x(qqq\n"
               "multiline raw string literal xxxxxxxxxxxxxx\n"
               ")x\",\n"
               "      a),\n"
               "    b);",
               "fffffffffff(g(R\"x(qqq\n"
               "multiline raw string literal xxxxxxxxxxxxxx\n"
               ")x\", a), b);",
               getGoogleStyleWithColumns(20));

  verifyNoChange("fffffffffff(R\"x(\n"
                 "multiline raw string literal xxxxxxxxxxxxxx\n"
                 ")x\");",
                 getGoogleStyleWithColumns(20));
  verifyFormat("fffffffffff(R\"x(\n"
               "multiline raw string literal xxxxxxxxxxxxxx\n"
               ")x\" + bbbbbb);",
               "fffffffffff(R\"x(\n"
               "multiline raw string literal xxxxxxxxxxxxxx\n"
               ")x\" +   bbbbbb);",
               getGoogleStyleWithColumns(20));
  verifyFormat("fffffffffff(\n"
               "    R\"x(\n"
               "multiline raw string literal xxxxxxxxxxxxxx\n"
               ")x\" +\n"
               "    bbbbbb);",
               "fffffffffff(\n"
               " R\"x(\n"
               "multiline raw string literal xxxxxxxxxxxxxx\n"
               ")x\" + bbbbbb);",
               getGoogleStyleWithColumns(20));
  verifyFormat("fffffffffff(R\"(single line raw string)\" + bbbbbb);",
               "fffffffffff(\n"
               " R\"(single line raw string)\" + bbbbbb);");
}

TEST_F(FormatTest, SkipsUnknownStringLiterals) {
  verifyFormat("string a = \"unterminated;");
  verifyFormat("function(\"unterminated,\n"
               "         OtherParameter);",
               "function(  \"unterminated,\n"
               "    OtherParameter);");
}

TEST_F(FormatTest, DoesNotTryToParseUDLiteralsInPreCpp11Code) {
  FormatStyle Style = getLLVMStyle();
  Style.Standard = FormatStyle::LS_Cpp03;
  verifyFormat("#define x(_a) printf(\"foo\" _a);",
               "#define x(_a) printf(\"foo\"_a);", Style);
}

TEST_F(FormatTest, CppLexVersion) {
  FormatStyle Style = getLLVMStyle();
  // Formatting of x * y differs if x is a type.
  verifyFormat("void foo() { MACRO(a * b); }", Style);
  verifyFormat("void foo() { MACRO(int *b); }", Style);

  // LLVM style uses latest lexer.
  verifyFormat("void foo() { MACRO(char8_t *b); }", Style);
  Style.Standard = FormatStyle::LS_Cpp17;
  // But in c++17, char8_t isn't a keyword.
  verifyFormat("void foo() { MACRO(char8_t * b); }", Style);
}

TEST_F(FormatTest, UnderstandsCpp1y) { verifyFormat("int bi{1'000'000};"); }

TEST_F(FormatTest, BreakStringLiteralsBeforeUnbreakableTokenSequence) {
  verifyFormat("someFunction(\"aaabbbcccd\"\n"
               "             \"ddeeefff\");",
               "someFunction(\"aaabbbcccdddeeefff\");",
               getLLVMStyleWithColumns(25));
  verifyFormat("someFunction1234567890(\n"
               "    \"aaabbbcccdddeeefff\");",
               "someFunction1234567890(\"aaabbbcccdddeeefff\");",
               getLLVMStyleWithColumns(26));
  verifyFormat("someFunction1234567890(\n"
               "    \"aaabbbcccdddeeeff\"\n"
               "    \"f\");",
               "someFunction1234567890(\"aaabbbcccdddeeefff\");",
               getLLVMStyleWithColumns(25));
  verifyFormat("someFunction1234567890(\n"
               "    \"aaabbbcccdddeeeff\"\n"
               "    \"f\");",
               "someFunction1234567890(\"aaabbbcccdddeeefff\");",
               getLLVMStyleWithColumns(24));
  verifyFormat("someFunction(\n"
               "    \"aaabbbcc ddde \"\n"
               "    \"efff\");",
               "someFunction(\"aaabbbcc ddde efff\");",
               getLLVMStyleWithColumns(25));
  verifyFormat("someFunction(\"aaabbbccc \"\n"
               "             \"ddeeefff\");",
               "someFunction(\"aaabbbccc ddeeefff\");",
               getLLVMStyleWithColumns(25));
  verifyFormat("someFunction1234567890(\n"
               "    \"aaabb \"\n"
               "    \"cccdddeeefff\");",
               "someFunction1234567890(\"aaabb cccdddeeefff\");",
               getLLVMStyleWithColumns(25));
  verifyFormat("#define A          \\\n"
               "  string s =       \\\n"
               "      \"123456789\"  \\\n"
               "      \"0\";         \\\n"
               "  int i;",
               "#define A string s = \"1234567890\"; int i;",
               getLLVMStyleWithColumns(20));
  verifyFormat("someFunction(\n"
               "    \"aaabbbcc \"\n"
               "    \"dddeeefff\");",
               "someFunction(\"aaabbbcc dddeeefff\");",
               getLLVMStyleWithColumns(25));
}

TEST_F(FormatTest, DoNotBreakStringLiteralsInEscapeSequence) {
  verifyFormat("\"\\a\"", getLLVMStyleWithColumns(3));
  verifyFormat("\"\\\"", getLLVMStyleWithColumns(2));
  // FIXME: unstable test case
  EXPECT_EQ("\"test\"\n"
            "\"\\n\"",
            format("\"test\\n\"", getLLVMStyleWithColumns(7)));
  // FIXME: unstable test case
  EXPECT_EQ("\"tes\\\\\"\n"
            "\"n\"",
            format("\"tes\\\\n\"", getLLVMStyleWithColumns(7)));
  // FIXME: unstable test case
  EXPECT_EQ("\"\\\\\\\\\"\n"
            "\"\\n\"",
            format("\"\\\\\\\\\\n\"", getLLVMStyleWithColumns(7)));
  verifyFormat("\"\\uff01\"", getLLVMStyleWithColumns(7));
  // FIXME: unstable test case
  EXPECT_EQ("\"\\uff01\"\n"
            "\"test\"",
            format("\"\\uff01test\"", getLLVMStyleWithColumns(8)));
  verifyFormat("\"\\Uff01ff02\"", getLLVMStyleWithColumns(11));
  // FIXME: unstable test case
  EXPECT_EQ("\"\\x000000000001\"\n"
            "\"next\"",
            format("\"\\x000000000001next\"", getLLVMStyleWithColumns(16)));
  verifyFormat("\"\\x000000000001next\"", getLLVMStyleWithColumns(15));
  verifyFormat("\"\\x000000000001\"", getLLVMStyleWithColumns(7));
  // FIXME: unstable test case
  EXPECT_EQ("\"test\"\n"
            "\"\\000000\"\n"
            "\"000001\"",
            format("\"test\\000000000001\"", getLLVMStyleWithColumns(9)));
  // FIXME: unstable test case
  EXPECT_EQ("\"test\\000\"\n"
            "\"00000000\"\n"
            "\"1\"",
            format("\"test\\000000000001\"", getLLVMStyleWithColumns(10)));
}

TEST_F(FormatTest, DoNotCreateUnreasonableUnwrappedLines) {
  verifyFormat("void f() {\n"
               "  return g() {}\n"
               "  void h() {}");
  verifyFormat("int a[] = {void forgot_closing_brace(){f();\n"
               "g();\n"
               "}");
}

TEST_F(FormatTest, DoNotPrematurelyEndUnwrappedLineForReturnStatements) {
  verifyFormat(
      "void f() { return C{param1, param2}.SomeCall(param1, param2); }");
}

TEST_F(FormatTest, FormatsClosingBracesInEmptyNestedBlocks) {
  verifyFormat("class X {\n"
               "  void f() {\n"
               "  }\n"
               "};",
               getLLVMStyleWithColumns(12));
}

TEST_F(FormatTest, ConfigurableIndentWidth) {
  FormatStyle EightIndent = getLLVMStyleWithColumns(18);
  EightIndent.IndentWidth = 8;
  EightIndent.ContinuationIndentWidth = 8;
  verifyFormat("void f() {\n"
               "        someFunction();\n"
               "        if (true) {\n"
               "                f();\n"
               "        }\n"
               "}",
               EightIndent);
  verifyFormat("class X {\n"
               "        void f() {\n"
               "        }\n"
               "};",
               EightIndent);
  verifyFormat("int x[] = {\n"
               "        call(),\n"
               "        call()};",
               EightIndent);
}

TEST_F(FormatTest, ConfigurableFunctionDeclarationIndentAfterType) {
  verifyFormat("double\n"
               "f();",
               getLLVMStyleWithColumns(8));
}

TEST_F(FormatTest, ConfigurableUseOfTab) {
  FormatStyle Tab = getLLVMStyleWithColumns(42);
  Tab.IndentWidth = 8;
  Tab.UseTab = FormatStyle::UT_Always;
  Tab.AlignEscapedNewlines = FormatStyle::ENAS_Left;

  verifyFormat("if (aaaaaaaa && // q\n"
               "    bb)\t\t// w\n"
               "\t;",
               "if (aaaaaaaa &&// q\n"
               "bb)// w\n"
               ";",
               Tab);
  verifyFormat("if (aaa && bbb) // w\n"
               "\t;",
               "if(aaa&&bbb)// w\n"
               ";",
               Tab);

  verifyFormat("class X {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t\t     parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  verifyFormat("#define A                        \\\n"
               "\tvoid f() {               \\\n"
               "\t\tsomeFunction(    \\\n"
               "\t\t    parameter1,  \\\n"
               "\t\t    parameter2); \\\n"
               "\t}",
               Tab);
  verifyFormat("int a;\t      // x\n"
               "int bbbbbbbb; // x",
               Tab);

  FormatStyle TabAlignment = Tab;
  TabAlignment.AlignConsecutiveDeclarations.Enabled = true;
  TabAlignment.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("unsigned long long big;\n"
               "char*\t\t   ptr;",
               TabAlignment);
  TabAlignment.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("unsigned long long big;\n"
               "char *\t\t   ptr;",
               TabAlignment);
  TabAlignment.PointerAlignment = FormatStyle::PAS_Right;
  verifyFormat("unsigned long long big;\n"
               "char\t\t  *ptr;",
               TabAlignment);

  Tab.TabWidth = 4;
  Tab.IndentWidth = 8;
  verifyFormat("class TabWidth4Indent8 {\n"
               "\t\tvoid f() {\n"
               "\t\t\t\tsomeFunction(parameter1,\n"
               "\t\t\t\t\t\t\t parameter2);\n"
               "\t\t}\n"
               "};",
               Tab);

  Tab.TabWidth = 4;
  Tab.IndentWidth = 4;
  verifyFormat("class TabWidth4Indent4 {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t\t\t\t parameter2);\n"
               "\t}\n"
               "};",
               Tab);

  Tab.TabWidth = 8;
  Tab.IndentWidth = 4;
  verifyFormat("class TabWidth8Indent4 {\n"
               "    void f() {\n"
               "\tsomeFunction(parameter1,\n"
               "\t\t     parameter2);\n"
               "    }\n"
               "};",
               Tab);

  Tab.TabWidth = 8;
  Tab.IndentWidth = 8;
  verifyFormat("/*\n"
               "\t      a\t\tcomment\n"
               "\t      in multiple lines\n"
               "       */",
               "   /*\t \t \n"
               " \t \t a\t\tcomment\t \t\n"
               " \t \t in multiple lines\t\n"
               " \t  */",
               Tab);

  TabAlignment.UseTab = FormatStyle::UT_ForIndentation;
  TabAlignment.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("void f() {\n"
               "\tunsigned long long big;\n"
               "\tchar*              ptr;\n"
               "}",
               TabAlignment);
  TabAlignment.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("void f() {\n"
               "\tunsigned long long big;\n"
               "\tchar *             ptr;\n"
               "}",
               TabAlignment);
  TabAlignment.PointerAlignment = FormatStyle::PAS_Right;
  verifyFormat("void f() {\n"
               "\tunsigned long long big;\n"
               "\tchar              *ptr;\n"
               "}",
               TabAlignment);

  Tab.UseTab = FormatStyle::UT_ForIndentation;
  verifyFormat("{\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "};",
               Tab);
  verifyFormat("enum AA {\n"
               "\ta1, // Force multiple lines\n"
               "\ta2,\n"
               "\ta3\n"
               "};",
               Tab);
  verifyFormat("if (aaaaaaaa && // q\n"
               "    bb)         // w\n"
               "\t;",
               "if (aaaaaaaa &&// q\n"
               "bb)// w\n"
               ";",
               Tab);
  verifyFormat("class X {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t             parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  verifyFormat("{\n"
               "\tQ(\n"
               "\t    {\n"
               "\t\t    int a;\n"
               "\t\t    someFunction(aaaaaaaa,\n"
               "\t\t                 bbbbbbb);\n"
               "\t    },\n"
               "\t    p);\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/* aaaa\n"
               "\t   bbbb */\n"
               "}",
               "{\n"
               "/* aaaa\n"
               "   bbbb */\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t  bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               "{\n"
               "/*\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "*/\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t// aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t// bbbbbbbbbbbbb\n"
               "}",
               "{\n"
               "\t// aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t  bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               "{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               Tab);
  verifyNoChange("{\n"
                 "\t/*\n"
                 "\n"
                 "\t*/\n"
                 "}",
                 Tab);
  verifyNoChange("{\n"
                 "\t/*\n"
                 " asdf\n"
                 "\t*/\n"
                 "}",
                 Tab);

  verifyFormat("void f() {\n"
               "\treturn true ? aaaaaaaaaaaaaaaaaa\n"
               "\t            : bbbbbbbbbbbbbbbbbb\n"
               "}",
               Tab);
  FormatStyle TabNoBreak = Tab;
  TabNoBreak.BreakBeforeTernaryOperators = false;
  verifyFormat("void f() {\n"
               "\treturn true ? aaaaaaaaaaaaaaaaaa :\n"
               "\t              bbbbbbbbbbbbbbbbbb\n"
               "}",
               TabNoBreak);
  verifyFormat("void f() {\n"
               "\treturn true ?\n"
               "\t           aaaaaaaaaaaaaaaaaaaa :\n"
               "\t           bbbbbbbbbbbbbbbbbbbb\n"
               "}",
               TabNoBreak);

  Tab.UseTab = FormatStyle::UT_Never;
  verifyFormat("/*\n"
               "              a\t\tcomment\n"
               "              in multiple lines\n"
               "       */",
               "   /*\t \t \n"
               " \t \t a\t\tcomment\t \t\n"
               " \t \t in multiple lines\t\n"
               " \t  */",
               Tab);
  verifyFormat("/* some\n"
               "   comment */",
               " \t \t /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("int a; /* some\n"
               "   comment */",
               " \t \t int a; /* some\n"
               " \t \t    comment */",
               Tab);

  verifyFormat("int a; /* some\n"
               "comment */",
               " \t \t int\ta; /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("f(\"\t\t\"); /* some\n"
               "    comment */",
               " \t \t f(\"\t\t\"); /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("{\n"
               "        /*\n"
               "         * Comment\n"
               "         */\n"
               "        int i;\n"
               "}",
               "{\n"
               "\t/*\n"
               "\t * Comment\n"
               "\t */\n"
               "\t int i;\n"
               "}",
               Tab);

  Tab.UseTab = FormatStyle::UT_ForContinuationAndIndentation;
  Tab.TabWidth = 8;
  Tab.IndentWidth = 8;
  verifyFormat("if (aaaaaaaa && // q\n"
               "    bb)         // w\n"
               "\t;",
               "if (aaaaaaaa &&// q\n"
               "bb)// w\n"
               ";",
               Tab);
  verifyFormat("if (aaa && bbb) // w\n"
               "\t;",
               "if(aaa&&bbb)// w\n"
               ";",
               Tab);
  verifyFormat("class X {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t\t     parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  verifyFormat("#define A                        \\\n"
               "\tvoid f() {               \\\n"
               "\t\tsomeFunction(    \\\n"
               "\t\t    parameter1,  \\\n"
               "\t\t    parameter2); \\\n"
               "\t}",
               Tab);
  Tab.TabWidth = 4;
  Tab.IndentWidth = 8;
  verifyFormat("class TabWidth4Indent8 {\n"
               "\t\tvoid f() {\n"
               "\t\t\t\tsomeFunction(parameter1,\n"
               "\t\t\t\t\t\t\t parameter2);\n"
               "\t\t}\n"
               "};",
               Tab);
  Tab.TabWidth = 4;
  Tab.IndentWidth = 4;
  verifyFormat("class TabWidth4Indent4 {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t\t\t\t parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  Tab.TabWidth = 8;
  Tab.IndentWidth = 4;
  verifyFormat("class TabWidth8Indent4 {\n"
               "    void f() {\n"
               "\tsomeFunction(parameter1,\n"
               "\t\t     parameter2);\n"
               "    }\n"
               "};",
               Tab);
  Tab.TabWidth = 8;
  Tab.IndentWidth = 8;
  verifyFormat("/*\n"
               "\t      a\t\tcomment\n"
               "\t      in multiple lines\n"
               "       */",
               "   /*\t \t \n"
               " \t \t a\t\tcomment\t \t\n"
               " \t \t in multiple lines\t\n"
               " \t  */",
               Tab);
  verifyFormat("{\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "};",
               Tab);
  verifyFormat("enum AA {\n"
               "\ta1, // Force multiple lines\n"
               "\ta2,\n"
               "\ta3\n"
               "};",
               Tab);
  verifyFormat("if (aaaaaaaa && // q\n"
               "    bb)         // w\n"
               "\t;",
               "if (aaaaaaaa &&// q\n"
               "bb)// w\n"
               ";",
               Tab);
  verifyFormat("class X {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t\t     parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  verifyFormat("{\n"
               "\tQ(\n"
               "\t    {\n"
               "\t\t    int a;\n"
               "\t\t    someFunction(aaaaaaaa,\n"
               "\t\t\t\t bbbbbbb);\n"
               "\t    },\n"
               "\t    p);\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/* aaaa\n"
               "\t   bbbb */\n"
               "}",
               "{\n"
               "/* aaaa\n"
               "   bbbb */\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t  bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               "{\n"
               "/*\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "*/\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t// aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t// bbbbbbbbbbbbb\n"
               "}",
               "{\n"
               "\t// aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t  bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               "{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               Tab);
  verifyNoChange("{\n"
                 "\t/*\n"
                 "\n"
                 "\t*/\n"
                 "}",
                 Tab);
  verifyNoChange("{\n"
                 "\t/*\n"
                 " asdf\n"
                 "\t*/\n"
                 "}",
                 Tab);
  verifyFormat("/* some\n"
               "   comment */",
               " \t \t /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("int a; /* some\n"
               "   comment */",
               " \t \t int a; /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("int a; /* some\n"
               "comment */",
               " \t \t int\ta; /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("f(\"\t\t\"); /* some\n"
               "    comment */",
               " \t \t f(\"\t\t\"); /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t * Comment\n"
               "\t */\n"
               "\tint i;\n"
               "}",
               "{\n"
               "\t/*\n"
               "\t * Comment\n"
               "\t */\n"
               "\t int i;\n"
               "}",
               Tab);
  Tab.TabWidth = 2;
  Tab.IndentWidth = 2;
  verifyFormat("{\n"
               "\t/* aaaa\n"
               "\t\t bbbb */\n"
               "}",
               "{\n"
               "/* aaaa\n"
               "\t bbbb */\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t\taaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t\tbbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               "{\n"
               "/*\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "*/\n"
               "}",
               Tab);
  Tab.AlignConsecutiveAssignments.Enabled = true;
  Tab.AlignConsecutiveDeclarations.Enabled = true;
  Tab.TabWidth = 4;
  Tab.IndentWidth = 4;
  verifyFormat("class Assign {\n"
               "\tvoid f() {\n"
               "\t\tint         x      = 123;\n"
               "\t\tint         random = 4;\n"
               "\t\tstd::string alphabet =\n"
               "\t\t\t\"abcdefghijklmnopqrstuvwxyz\";\n"
               "\t}\n"
               "};",
               Tab);

  Tab.UseTab = FormatStyle::UT_AlignWithSpaces;
  Tab.TabWidth = 8;
  Tab.IndentWidth = 8;
  verifyFormat("if (aaaaaaaa && // q\n"
               "    bb)         // w\n"
               "\t;",
               "if (aaaaaaaa &&// q\n"
               "bb)// w\n"
               ";",
               Tab);
  verifyFormat("if (aaa && bbb) // w\n"
               "\t;",
               "if(aaa&&bbb)// w\n"
               ";",
               Tab);
  verifyFormat("class X {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t             parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  verifyFormat("#define A                        \\\n"
               "\tvoid f() {               \\\n"
               "\t\tsomeFunction(    \\\n"
               "\t\t    parameter1,  \\\n"
               "\t\t    parameter2); \\\n"
               "\t}",
               Tab);
  Tab.TabWidth = 4;
  Tab.IndentWidth = 8;
  verifyFormat("class TabWidth4Indent8 {\n"
               "\t\tvoid f() {\n"
               "\t\t\t\tsomeFunction(parameter1,\n"
               "\t\t\t\t             parameter2);\n"
               "\t\t}\n"
               "};",
               Tab);
  Tab.TabWidth = 4;
  Tab.IndentWidth = 4;
  verifyFormat("class TabWidth4Indent4 {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t             parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  Tab.TabWidth = 8;
  Tab.IndentWidth = 4;
  verifyFormat("class TabWidth8Indent4 {\n"
               "    void f() {\n"
               "\tsomeFunction(parameter1,\n"
               "\t             parameter2);\n"
               "    }\n"
               "};",
               Tab);
  Tab.TabWidth = 8;
  Tab.IndentWidth = 8;
  verifyFormat("/*\n"
               "              a\t\tcomment\n"
               "              in multiple lines\n"
               "       */",
               "   /*\t \t \n"
               " \t \t a\t\tcomment\t \t\n"
               " \t \t in multiple lines\t\n"
               " \t  */",
               Tab);
  verifyFormat("{\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "};",
               Tab);
  verifyFormat("enum AA {\n"
               "\ta1, // Force multiple lines\n"
               "\ta2,\n"
               "\ta3\n"
               "};",
               Tab);
  verifyFormat("if (aaaaaaaa && // q\n"
               "    bb)         // w\n"
               "\t;",
               "if (aaaaaaaa &&// q\n"
               "bb)// w\n"
               ";",
               Tab);
  verifyFormat("class X {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t             parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  verifyFormat("{\n"
               "\tQ(\n"
               "\t    {\n"
               "\t\t    int a;\n"
               "\t\t    someFunction(aaaaaaaa,\n"
               "\t\t                 bbbbbbb);\n"
               "\t    },\n"
               "\t    p);\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/* aaaa\n"
               "\t   bbbb */\n"
               "}",
               "{\n"
               "/* aaaa\n"
               "   bbbb */\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t  bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               "{\n"
               "/*\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "*/\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t// aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t// bbbbbbbbbbbbb\n"
               "}",
               "{\n"
               "\t// aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t  bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               "{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               Tab);
  verifyNoChange("{\n"
                 "\t/*\n"
                 "\n"
                 "\t*/\n"
                 "}",
                 Tab);
  verifyNoChange("{\n"
                 "\t/*\n"
                 " asdf\n"
                 "\t*/\n"
                 "}",
                 Tab);
  verifyFormat("/* some\n"
               "   comment */",
               " \t \t /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("int a; /* some\n"
               "   comment */",
               " \t \t int a; /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("int a; /* some\n"
               "comment */",
               " \t \t int\ta; /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("f(\"\t\t\"); /* some\n"
               "    comment */",
               " \t \t f(\"\t\t\"); /* some\n"
               " \t \t    comment */",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t * Comment\n"
               "\t */\n"
               "\tint i;\n"
               "}",
               "{\n"
               "\t/*\n"
               "\t * Comment\n"
               "\t */\n"
               "\t int i;\n"
               "}",
               Tab);
  Tab.TabWidth = 2;
  Tab.IndentWidth = 2;
  verifyFormat("{\n"
               "\t/* aaaa\n"
               "\t   bbbb */\n"
               "}",
               "{\n"
               "/* aaaa\n"
               "   bbbb */\n"
               "}",
               Tab);
  verifyFormat("{\n"
               "\t/*\n"
               "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "\t  bbbbbbbbbbbbb\n"
               "\t*/\n"
               "}",
               "{\n"
               "/*\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
               "*/\n"
               "}",
               Tab);
  Tab.AlignConsecutiveAssignments.Enabled = true;
  Tab.AlignConsecutiveDeclarations.Enabled = true;
  Tab.TabWidth = 4;
  Tab.IndentWidth = 4;
  verifyFormat("class Assign {\n"
               "\tvoid f() {\n"
               "\t\tint         x      = 123;\n"
               "\t\tint         random = 4;\n"
               "\t\tstd::string alphabet =\n"
               "\t\t\t\"abcdefghijklmnopqrstuvwxyz\";\n"
               "\t}\n"
               "};",
               Tab);
  Tab.AlignOperands = FormatStyle::OAS_Align;
  verifyFormat("int aaaaaaaaaa = bbbbbbbbbbbbbbbbbbbb +\n"
               "                 cccccccccccccccccccc;",
               Tab);
  // no alignment
  verifyFormat("int aaaaaaaaaa =\n"
               "\tbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               Tab);
  verifyFormat("return aaaaaaaaaaaaaaaa ? 111111111111111\n"
               "       : bbbbbbbbbbbbbb ? 222222222222222\n"
               "                        : 333333333333333;",
               Tab);
  Tab.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  Tab.AlignOperands = FormatStyle::OAS_AlignAfterOperator;
  verifyFormat("int aaaaaaaaaa = bbbbbbbbbbbbbbbbbbbb\n"
               "               + cccccccccccccccccccc;",
               Tab);
}

TEST_F(FormatTest, ZeroTabWidth) {
  FormatStyle Tab = getLLVMStyleWithColumns(42);
  Tab.IndentWidth = 8;
  Tab.UseTab = FormatStyle::UT_Never;
  Tab.TabWidth = 0;
  verifyFormat("void a(){\n"
               "    // line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t// line starts with '\t'\n"
               "};",
               Tab);

  verifyFormat("void a(){\n"
               "    // line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t\t// line starts with '\t'\n"
               "};",
               Tab);

  Tab.UseTab = FormatStyle::UT_ForIndentation;
  verifyFormat("void a(){\n"
               "    // line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t// line starts with '\t'\n"
               "};",
               Tab);

  verifyFormat("void a(){\n"
               "    // line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t\t// line starts with '\t'\n"
               "};",
               Tab);

  Tab.UseTab = FormatStyle::UT_ForContinuationAndIndentation;
  verifyFormat("void a(){\n"
               "    // line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t// line starts with '\t'\n"
               "};",
               Tab);

  verifyFormat("void a(){\n"
               "    // line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t\t// line starts with '\t'\n"
               "};",
               Tab);

  Tab.UseTab = FormatStyle::UT_AlignWithSpaces;
  verifyFormat("void a(){\n"
               "    // line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t// line starts with '\t'\n"
               "};",
               Tab);

  verifyFormat("void a(){\n"
               "    // line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t\t// line starts with '\t'\n"
               "};",
               Tab);

  Tab.UseTab = FormatStyle::UT_Always;
  verifyFormat("void a(){\n"
               "// line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t// line starts with '\t'\n"
               "};",
               Tab);

  verifyFormat("void a(){\n"
               "// line starts with '\t'\n"
               "};",
               "void a(){\n"
               "\t\t// line starts with '\t'\n"
               "};",
               Tab);
}

TEST_F(FormatTest, CalculatesOriginalColumn) {
  verifyFormat("\"qqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
               "q\"; /* some\n"
               "       comment */",
               "  \"qqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
               "q\"; /* some\n"
               "       comment */");
  verifyFormat("// qqqqqqqqqqqqqqqqqqqqqqqqqq\n"
               "/* some\n"
               "   comment */",
               "// qqqqqqqqqqqqqqqqqqqqqqqqqq\n"
               " /* some\n"
               "    comment */");
  verifyFormat("// qqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
               "qqq\n"
               "/* some\n"
               "   comment */",
               "// qqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
               "qqq\n"
               " /* some\n"
               "    comment */");
  verifyFormat("inttt qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
               "wwww; /* some\n"
               "         comment */",
               "  inttt qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
               "wwww; /* some\n"
               "         comment */");
}

TEST_F(FormatTest, ConfigurableSpaceBeforeParens) {
  FormatStyle NoSpace = getLLVMStyle();
  NoSpace.SpaceBeforeParens = FormatStyle::SBPO_Never;

  verifyFormat("while(true)\n"
               "  continue;",
               NoSpace);
  verifyFormat("for(;;)\n"
               "  continue;",
               NoSpace);
  verifyFormat("if(true)\n"
               "  f();\n"
               "else if(true)\n"
               "  f();",
               NoSpace);
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while(something());",
               NoSpace);
  verifyFormat("switch(x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               NoSpace);
  verifyFormat("auto i = std::make_unique<int>(5);", NoSpace);
  verifyFormat("size_t x = sizeof(x);", NoSpace);
  verifyFormat("auto f(int x) -> decltype(x);", NoSpace);
  verifyFormat("auto f(int x) -> typeof(x);", NoSpace);
  verifyFormat("auto f(int x) -> _Atomic(x);", NoSpace);
  verifyFormat("auto f(int x) -> __underlying_type(x);", NoSpace);
  verifyFormat("int f(T x) noexcept(x.create());", NoSpace);
  verifyFormat("alignas(128) char a[128];", NoSpace);
  verifyFormat("size_t x = alignof(MyType);", NoSpace);
  verifyFormat("static_assert(sizeof(char) == 1, \"Impossible!\");", NoSpace);
  verifyFormat("int f() throw(Deprecated);", NoSpace);
  verifyFormat("typedef void (*cb)(int);", NoSpace);
  verifyFormat("T A::operator()();", NoSpace);
  verifyFormat("X A::operator++(T);", NoSpace);
  verifyFormat("auto lambda = []() { return 0; };", NoSpace);

  FormatStyle Space = getLLVMStyle();
  Space.SpaceBeforeParens = FormatStyle::SBPO_Always;

  verifyFormat("int f ();", Space);
  verifyFormat("bool operator< ();", Space);
  verifyFormat("bool operator> ();", Space);
  verifyFormat("void f (int a, T b) {\n"
               "  while (true)\n"
               "    continue;\n"
               "}",
               Space);
  verifyFormat("if (true)\n"
               "  f ();\n"
               "else if (true)\n"
               "  f ();",
               Space);
  verifyFormat("do {\n"
               "  do_something ();\n"
               "} while (something ());",
               Space);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Space);
  verifyFormat("A::A () : a (1) {}", Space);
  verifyFormat("void f () __attribute__ ((asdf));", Space);
  verifyFormat("*(&a + 1);\n"
               "&((&a)[1]);\n"
               "a[(b + c) * d];\n"
               "(((a + 1) * 2) + 3) * 4;",
               Space);
  verifyFormat("#define A(x) x", Space);
  verifyFormat("#define A (x) x", Space);
  verifyFormat("#if defined(x)\n"
               "#endif",
               Space);
  verifyFormat("auto i = std::make_unique<int> (5);", Space);
  verifyFormat("size_t x = sizeof (x);", Space);
  verifyFormat("auto f (int x) -> decltype (x);", Space);
  verifyFormat("auto f (int x) -> typeof (x);", Space);
  verifyFormat("auto f (int x) -> _Atomic (x);", Space);
  verifyFormat("auto f (int x) -> __underlying_type (x);", Space);
  verifyFormat("int f (T x) noexcept (x.create ());", Space);
  verifyFormat("alignas (128) char a[128];", Space);
  verifyFormat("size_t x = alignof (MyType);", Space);
  verifyFormat("static_assert (sizeof (char) == 1, \"Impossible!\");", Space);
  verifyFormat("int f () throw (Deprecated);", Space);
  verifyFormat("typedef void (*cb) (int);", Space);
  // FIXME these tests regressed behaviour.
  // verifyFormat("T A::operator() ();", Space);
  // verifyFormat("X A::operator++ (T);", Space);
  verifyFormat("auto lambda = [] () { return 0; };", Space);
  verifyFormat("int x = int (y);", Space);
  verifyFormat("#define F(...) __VA_OPT__ (__VA_ARGS__)", Space);
  verifyFormat("__builtin_LINE ()", Space);
  verifyFormat("__builtin_UNKNOWN ()", Space);

  FormatStyle SomeSpace = getLLVMStyle();
  SomeSpace.SpaceBeforeParens = FormatStyle::SBPO_NonEmptyParentheses;

  verifyFormat("[]() -> float {}", SomeSpace);
  verifyFormat("[] (auto foo) {}", SomeSpace);
  verifyFormat("[foo]() -> int {}", SomeSpace);
  verifyFormat("int f();", SomeSpace);
  verifyFormat("void f (int a, T b) {\n"
               "  while (true)\n"
               "    continue;\n"
               "}",
               SomeSpace);
  verifyFormat("if (true)\n"
               "  f();\n"
               "else if (true)\n"
               "  f();",
               SomeSpace);
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while (something());",
               SomeSpace);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               SomeSpace);
  verifyFormat("A::A() : a (1) {}", SomeSpace);
  verifyFormat("void f() __attribute__ ((asdf));", SomeSpace);
  verifyFormat("*(&a + 1);\n"
               "&((&a)[1]);\n"
               "a[(b + c) * d];\n"
               "(((a + 1) * 2) + 3) * 4;",
               SomeSpace);
  verifyFormat("#define A(x) x", SomeSpace);
  verifyFormat("#define A (x) x", SomeSpace);
  verifyFormat("#if defined(x)\n"
               "#endif",
               SomeSpace);
  verifyFormat("auto i = std::make_unique<int> (5);", SomeSpace);
  verifyFormat("size_t x = sizeof (x);", SomeSpace);
  verifyFormat("auto f (int x) -> decltype (x);", SomeSpace);
  verifyFormat("auto f (int x) -> typeof (x);", SomeSpace);
  verifyFormat("auto f (int x) -> _Atomic (x);", SomeSpace);
  verifyFormat("auto f (int x) -> __underlying_type (x);", SomeSpace);
  verifyFormat("int f (T x) noexcept (x.create());", SomeSpace);
  verifyFormat("alignas (128) char a[128];", SomeSpace);
  verifyFormat("size_t x = alignof (MyType);", SomeSpace);
  verifyFormat("static_assert (sizeof (char) == 1, \"Impossible!\");",
               SomeSpace);
  verifyFormat("int f() throw (Deprecated);", SomeSpace);
  verifyFormat("typedef void (*cb) (int);", SomeSpace);
  verifyFormat("T A::operator()();", SomeSpace);
  // FIXME these tests regressed behaviour.
  // verifyFormat("X A::operator++ (T);", SomeSpace);
  verifyFormat("int x = int (y);", SomeSpace);
  verifyFormat("auto lambda = []() { return 0; };", SomeSpace);

  FormatStyle SpaceControlStatements = getLLVMStyle();
  SpaceControlStatements.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  SpaceControlStatements.SpaceBeforeParensOptions.AfterControlStatements = true;

  verifyFormat("while (true)\n"
               "  continue;",
               SpaceControlStatements);
  verifyFormat("if (true)\n"
               "  f();\n"
               "else if (true)\n"
               "  f();",
               SpaceControlStatements);
  verifyFormat("for (;;) {\n"
               "  do_something();\n"
               "}",
               SpaceControlStatements);
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while (something());",
               SpaceControlStatements);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               SpaceControlStatements);

  FormatStyle SpaceFuncDecl = getLLVMStyle();
  SpaceFuncDecl.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  SpaceFuncDecl.SpaceBeforeParensOptions.AfterFunctionDeclarationName = true;

  verifyFormat("int f ();", SpaceFuncDecl);
  verifyFormat("void f(int a, T b) {}", SpaceFuncDecl);
  verifyFormat("A::A() : a(1) {}", SpaceFuncDecl);
  verifyFormat("void f () __attribute__((asdf));", SpaceFuncDecl);
  verifyFormat("#define A(x) x", SpaceFuncDecl);
  verifyFormat("#define A (x) x", SpaceFuncDecl);
  verifyFormat("#if defined(x)\n"
               "#endif",
               SpaceFuncDecl);
  verifyFormat("auto i = std::make_unique<int>(5);", SpaceFuncDecl);
  verifyFormat("size_t x = sizeof(x);", SpaceFuncDecl);
  verifyFormat("auto f (int x) -> decltype(x);", SpaceFuncDecl);
  verifyFormat("auto f (int x) -> typeof(x);", SpaceFuncDecl);
  verifyFormat("auto f (int x) -> _Atomic(x);", SpaceFuncDecl);
  verifyFormat("auto f (int x) -> __underlying_type(x);", SpaceFuncDecl);
  verifyFormat("int f (T x) noexcept(x.create());", SpaceFuncDecl);
  verifyFormat("alignas(128) char a[128];", SpaceFuncDecl);
  verifyFormat("size_t x = alignof(MyType);", SpaceFuncDecl);
  verifyFormat("static_assert(sizeof(char) == 1, \"Impossible!\");",
               SpaceFuncDecl);
  verifyFormat("int f () throw(Deprecated);", SpaceFuncDecl);
  verifyFormat("typedef void (*cb)(int);", SpaceFuncDecl);
  // FIXME these tests regressed behaviour.
  // verifyFormat("T A::operator() ();", SpaceFuncDecl);
  // verifyFormat("X A::operator++ (T);", SpaceFuncDecl);
  verifyFormat("T A::operator()() {}", SpaceFuncDecl);
  verifyFormat("auto lambda = []() { return 0; };", SpaceFuncDecl);
  verifyFormat("int x = int(y);", SpaceFuncDecl);
  verifyFormat("M(std::size_t R, std::size_t C) : C(C), data(R) {}",
               SpaceFuncDecl);

  FormatStyle SpaceFuncDef = getLLVMStyle();
  SpaceFuncDef.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  SpaceFuncDef.SpaceBeforeParensOptions.AfterFunctionDefinitionName = true;

  verifyFormat("int f();", SpaceFuncDef);
  verifyFormat("void f (int a, T b) {}", SpaceFuncDef);
  verifyFormat("A::A () : a(1) {}", SpaceFuncDef);
  verifyFormat("void f() __attribute__((asdf));", SpaceFuncDef);
  verifyFormat("#define A(x) x", SpaceFuncDef);
  verifyFormat("#define A (x) x", SpaceFuncDef);
  verifyFormat("#if defined(x)\n"
               "#endif",
               SpaceFuncDef);
  verifyFormat("auto i = std::make_unique<int>(5);", SpaceFuncDef);
  verifyFormat("size_t x = sizeof(x);", SpaceFuncDef);
  verifyFormat("auto f(int x) -> decltype(x);", SpaceFuncDef);
  verifyFormat("auto f(int x) -> typeof(x);", SpaceFuncDef);
  verifyFormat("auto f(int x) -> _Atomic(x);", SpaceFuncDef);
  verifyFormat("auto f(int x) -> __underlying_type(x);", SpaceFuncDef);
  verifyFormat("int f(T x) noexcept(x.create());", SpaceFuncDef);
  verifyFormat("alignas(128) char a[128];", SpaceFuncDef);
  verifyFormat("size_t x = alignof(MyType);", SpaceFuncDef);
  verifyFormat("static_assert(sizeof(char) == 1, \"Impossible!\");",
               SpaceFuncDef);
  verifyFormat("int f() throw(Deprecated);", SpaceFuncDef);
  verifyFormat("typedef void (*cb)(int);", SpaceFuncDef);
  verifyFormat("T A::operator()();", SpaceFuncDef);
  verifyFormat("X A::operator++(T);", SpaceFuncDef);
  // verifyFormat("T A::operator() () {}", SpaceFuncDef);
  verifyFormat("auto lambda = [] () { return 0; };", SpaceFuncDef);
  verifyFormat("int x = int(y);", SpaceFuncDef);
  verifyFormat("M (std::size_t R, std::size_t C) : C(C), data(R) {}",
               SpaceFuncDef);

  FormatStyle SpaceIfMacros = getLLVMStyle();
  SpaceIfMacros.IfMacros.clear();
  SpaceIfMacros.IfMacros.push_back("MYIF");
  SpaceIfMacros.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  SpaceIfMacros.SpaceBeforeParensOptions.AfterIfMacros = true;
  verifyFormat("MYIF (a)\n  return;", SpaceIfMacros);
  verifyFormat("MYIF (a)\n  return;\nelse MYIF (b)\n  return;", SpaceIfMacros);
  verifyFormat("MYIF (a)\n  return;\nelse\n  return;", SpaceIfMacros);

  FormatStyle SpaceForeachMacros = getLLVMStyle();
  EXPECT_EQ(SpaceForeachMacros.AllowShortBlocksOnASingleLine,
            FormatStyle::SBS_Never);
  EXPECT_EQ(SpaceForeachMacros.AllowShortLoopsOnASingleLine, false);
  SpaceForeachMacros.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  SpaceForeachMacros.SpaceBeforeParensOptions.AfterForeachMacros = true;
  verifyFormat("for (;;) {\n"
               "}",
               SpaceForeachMacros);
  verifyFormat("foreach (Item *item, itemlist) {\n"
               "}",
               SpaceForeachMacros);
  verifyFormat("Q_FOREACH (Item *item, itemlist) {\n"
               "}",
               SpaceForeachMacros);
  verifyFormat("BOOST_FOREACH (Item *item, itemlist) {\n"
               "}",
               SpaceForeachMacros);
  verifyFormat("UNKNOWN_FOREACH(Item *item, itemlist) {}", SpaceForeachMacros);

  FormatStyle SomeSpace2 = getLLVMStyle();
  SomeSpace2.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  SomeSpace2.SpaceBeforeParensOptions.BeforeNonEmptyParentheses = true;
  verifyFormat("[]() -> float {}", SomeSpace2);
  verifyFormat("[] (auto foo) {}", SomeSpace2);
  verifyFormat("[foo]() -> int {}", SomeSpace2);
  verifyFormat("int f();", SomeSpace2);
  verifyFormat("void f (int a, T b) {\n"
               "  while (true)\n"
               "    continue;\n"
               "}",
               SomeSpace2);
  verifyFormat("if (true)\n"
               "  f();\n"
               "else if (true)\n"
               "  f();",
               SomeSpace2);
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while (something());",
               SomeSpace2);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               SomeSpace2);
  verifyFormat("A::A() : a (1) {}", SomeSpace2);
  verifyFormat("void f() __attribute__ ((asdf));", SomeSpace2);
  verifyFormat("*(&a + 1);\n"
               "&((&a)[1]);\n"
               "a[(b + c) * d];\n"
               "(((a + 1) * 2) + 3) * 4;",
               SomeSpace2);
  verifyFormat("#define A(x) x", SomeSpace2);
  verifyFormat("#define A (x) x", SomeSpace2);
  verifyFormat("#if defined(x)\n"
               "#endif",
               SomeSpace2);
  verifyFormat("auto i = std::make_unique<int> (5);", SomeSpace2);
  verifyFormat("size_t x = sizeof (x);", SomeSpace2);
  verifyFormat("auto f (int x) -> decltype (x);", SomeSpace2);
  verifyFormat("auto f (int x) -> typeof (x);", SomeSpace2);
  verifyFormat("auto f (int x) -> _Atomic (x);", SomeSpace2);
  verifyFormat("auto f (int x) -> __underlying_type (x);", SomeSpace2);
  verifyFormat("int f (T x) noexcept (x.create());", SomeSpace2);
  verifyFormat("alignas (128) char a[128];", SomeSpace2);
  verifyFormat("size_t x = alignof (MyType);", SomeSpace2);
  verifyFormat("static_assert (sizeof (char) == 1, \"Impossible!\");",
               SomeSpace2);
  verifyFormat("int f() throw (Deprecated);", SomeSpace2);
  verifyFormat("typedef void (*cb) (int);", SomeSpace2);
  verifyFormat("T A::operator()();", SomeSpace2);
  // verifyFormat("X A::operator++ (T);", SomeSpace2);
  verifyFormat("int x = int (y);", SomeSpace2);
  verifyFormat("auto lambda = []() { return 0; };", SomeSpace2);

  FormatStyle SpaceAfterOverloadedOperator = getLLVMStyle();
  SpaceAfterOverloadedOperator.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  SpaceAfterOverloadedOperator.SpaceBeforeParensOptions
      .AfterOverloadedOperator = true;

  verifyFormat("auto operator++ () -> int;", SpaceAfterOverloadedOperator);
  verifyFormat("X A::operator++ ();", SpaceAfterOverloadedOperator);
  verifyFormat("some_object.operator++ ();", SpaceAfterOverloadedOperator);
  verifyFormat("auto func() -> int;", SpaceAfterOverloadedOperator);

  SpaceAfterOverloadedOperator.SpaceBeforeParensOptions
      .AfterOverloadedOperator = false;

  verifyFormat("auto operator++() -> int;", SpaceAfterOverloadedOperator);
  verifyFormat("X A::operator++();", SpaceAfterOverloadedOperator);
  verifyFormat("some_object.operator++();", SpaceAfterOverloadedOperator);
  verifyFormat("auto func() -> int;", SpaceAfterOverloadedOperator);

  auto SpaceAfterRequires = getLLVMStyle();
  SpaceAfterRequires.SpaceBeforeParens = FormatStyle::SBPO_Custom;
  EXPECT_FALSE(
      SpaceAfterRequires.SpaceBeforeParensOptions.AfterRequiresInClause);
  EXPECT_FALSE(
      SpaceAfterRequires.SpaceBeforeParensOptions.AfterRequiresInExpression);
  verifyFormat("void f(auto x)\n"
               "  requires requires(int i) { x + i; }\n"
               "{}",
               SpaceAfterRequires);
  verifyFormat("void f(auto x)\n"
               "  requires(requires(int i) { x + i; })\n"
               "{}",
               SpaceAfterRequires);
  verifyFormat("if (requires(int i) { x + i; })\n"
               "  return;",
               SpaceAfterRequires);
  verifyFormat("bool b = requires(int i) { x + i; };", SpaceAfterRequires);
  verifyFormat("template <typename T>\n"
               "  requires(Foo<T>)\n"
               "class Bar;",
               SpaceAfterRequires);

  SpaceAfterRequires.SpaceBeforeParensOptions.AfterRequiresInClause = true;
  verifyFormat("void f(auto x)\n"
               "  requires requires(int i) { x + i; }\n"
               "{}",
               SpaceAfterRequires);
  verifyFormat("void f(auto x)\n"
               "  requires (requires(int i) { x + i; })\n"
               "{}",
               SpaceAfterRequires);
  verifyFormat("if (requires(int i) { x + i; })\n"
               "  return;",
               SpaceAfterRequires);
  verifyFormat("bool b = requires(int i) { x + i; };", SpaceAfterRequires);
  verifyFormat("template <typename T>\n"
               "  requires (Foo<T>)\n"
               "class Bar;",
               SpaceAfterRequires);

  SpaceAfterRequires.SpaceBeforeParensOptions.AfterRequiresInClause = false;
  SpaceAfterRequires.SpaceBeforeParensOptions.AfterRequiresInExpression = true;
  verifyFormat("void f(auto x)\n"
               "  requires requires (int i) { x + i; }\n"
               "{}",
               SpaceAfterRequires);
  verifyFormat("void f(auto x)\n"
               "  requires(requires (int i) { x + i; })\n"
               "{}",
               SpaceAfterRequires);
  verifyFormat("if (requires (int i) { x + i; })\n"
               "  return;",
               SpaceAfterRequires);
  verifyFormat("bool b = requires (int i) { x + i; };", SpaceAfterRequires);
  verifyFormat("template <typename T>\n"
               "  requires(Foo<T>)\n"
               "class Bar;",
               SpaceAfterRequires);

  SpaceAfterRequires.SpaceBeforeParensOptions.AfterRequiresInClause = true;
  verifyFormat("void f(auto x)\n"
               "  requires requires (int i) { x + i; }\n"
               "{}",
               SpaceAfterRequires);
  verifyFormat("void f(auto x)\n"
               "  requires (requires (int i) { x + i; })\n"
               "{}",
               SpaceAfterRequires);
  verifyFormat("if (requires (int i) { x + i; })\n"
               "  return;",
               SpaceAfterRequires);
  verifyFormat("bool b = requires (int i) { x + i; };", SpaceAfterRequires);
  verifyFormat("template <typename T>\n"
               "  requires (Foo<T>)\n"
               "class Bar;",
               SpaceAfterRequires);
}

TEST_F(FormatTest, SpaceAfterLogicalNot) {
  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpaceAfterLogicalNot = true;

  verifyFormat("bool x = ! y", Spaces);
  verifyFormat("if (! isFailure())", Spaces);
  verifyFormat("if (! (a && b))", Spaces);
  verifyFormat("\"Error!\"", Spaces);
  verifyFormat("! ! x", Spaces);
}

TEST_F(FormatTest, ConfigurableSpacesInParens) {
  FormatStyle Spaces = getLLVMStyle();

  verifyFormat("do_something(::globalVar);", Spaces);
  verifyFormat("call(x, y, z);", Spaces);
  verifyFormat("call();", Spaces);
  verifyFormat("std::function<void(int, int)> callback;", Spaces);
  verifyFormat("void inFunction() { std::function<void(int, int)> fct; }",
               Spaces);
  verifyFormat("while ((bool)1)\n"
               "  continue;",
               Spaces);
  verifyFormat("for (;;)\n"
               "  continue;",
               Spaces);
  verifyFormat("if (true)\n"
               "  f();\n"
               "else if (true)\n"
               "  f();",
               Spaces);
  verifyFormat("do {\n"
               "  do_something((int)i);\n"
               "} while (something());",
               Spaces);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Spaces);
  verifyFormat("SomeType *__attribute__((attr)) *a = NULL;", Spaces);
  verifyFormat("void __attribute__((naked)) foo(int bar)", Spaces);
  verifyFormat("void f() __attribute__((asdf));", Spaces);

  Spaces.SpacesInParens = FormatStyle::SIPO_Custom;
  Spaces.SpacesInParensOptions = {};
  Spaces.SpacesInParensOptions.Other = true;
  Spaces.SpacesInParensOptions.InConditionalStatements = true;
  verifyFormat("do_something( ::globalVar );", Spaces);
  verifyFormat("call( x, y, z );", Spaces);
  verifyFormat("call();", Spaces);
  verifyFormat("std::function<void( int, int )> callback;", Spaces);
  verifyFormat("void inFunction() { std::function<void( int, int )> fct; }",
               Spaces);
  verifyFormat("while ( (bool)1 )\n"
               "  continue;",
               Spaces);
  verifyFormat("for ( ;; )\n"
               "  continue;",
               Spaces);
  verifyFormat("if ( true )\n"
               "  f();\n"
               "else if ( true )\n"
               "  f();",
               Spaces);
  verifyFormat("do {\n"
               "  do_something( (int)i );\n"
               "} while ( something() );",
               Spaces);
  verifyFormat("switch ( x ) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Spaces);
  verifyFormat("SomeType *__attribute__( ( attr ) ) *a = NULL;", Spaces);
  verifyFormat("void __attribute__( ( naked ) ) foo( int bar )", Spaces);
  verifyFormat("void f() __attribute__( ( asdf ) );", Spaces);

  Spaces.SpacesInParens = FormatStyle::SIPO_Custom;
  Spaces.SpacesInParensOptions = {};
  Spaces.SpacesInParensOptions.InCStyleCasts = true;
  verifyFormat("Type *A = ( Type * )P;", Spaces);
  verifyFormat("Type *A = ( vector<Type *, int *> )P;", Spaces);
  verifyFormat("x = ( int32 )y;", Spaces);
  verifyFormat("throw ( int32 )x;", Spaces);
  verifyFormat("int a = ( int )(2.0f);", Spaces);
  verifyFormat("#define AA(X) sizeof((( X * )NULL)->a)", Spaces);
  verifyFormat("my_int a = ( my_int )sizeof(int);", Spaces);
  verifyFormat("#define x (( int )-1)", Spaces);

  // Run the first set of tests again with:
  Spaces.SpacesInParens = FormatStyle::SIPO_Custom;
  Spaces.SpacesInParensOptions = {};
  Spaces.SpacesInParensOptions.InEmptyParentheses = true;
  Spaces.SpacesInParensOptions.InCStyleCasts = true;
  verifyFormat("call(x, y, z);", Spaces);
  verifyFormat("call( );", Spaces);
  verifyFormat("std::function<void(int, int)> callback;", Spaces);
  verifyFormat("while (( bool )1)\n"
               "  continue;",
               Spaces);
  verifyFormat("for (;;)\n"
               "  continue;",
               Spaces);
  verifyFormat("if (true)\n"
               "  f( );\n"
               "else if (true)\n"
               "  f( );",
               Spaces);
  verifyFormat("do {\n"
               "  do_something(( int )i);\n"
               "} while (something( ));",
               Spaces);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Spaces);
  verifyFormat("SomeType *__attribute__((attr)) *a = NULL;", Spaces);
  verifyFormat("void __attribute__((naked)) foo(int bar)", Spaces);
  verifyFormat("void f( ) __attribute__((asdf));", Spaces);

  // Run the first set of tests again with:
  Spaces.SpaceAfterCStyleCast = true;
  verifyFormat("call(x, y, z);", Spaces);
  verifyFormat("call( );", Spaces);
  verifyFormat("std::function<void(int, int)> callback;", Spaces);
  verifyFormat("while (( bool ) 1)\n"
               "  continue;",
               Spaces);
  verifyFormat("for (;;)\n"
               "  continue;",
               Spaces);
  verifyFormat("if (true)\n"
               "  f( );\n"
               "else if (true)\n"
               "  f( );",
               Spaces);
  verifyFormat("do {\n"
               "  do_something(( int ) i);\n"
               "} while (something( ));",
               Spaces);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Spaces);
  verifyFormat("#define CONF_BOOL(x) ( bool * ) ( void * ) (x)", Spaces);
  verifyFormat("#define CONF_BOOL(x) ( bool * ) (x)", Spaces);
  verifyFormat("#define CONF_BOOL(x) ( bool ) (x)", Spaces);
  verifyFormat("bool *y = ( bool * ) ( void * ) (x);", Spaces);
  verifyFormat("bool *y = ( bool * ) (x);", Spaces);
  verifyFormat("throw ( int32 ) x;", Spaces);
  verifyFormat("SomeType *__attribute__((attr)) *a = NULL;", Spaces);
  verifyFormat("void __attribute__((naked)) foo(int bar)", Spaces);
  verifyFormat("void f( ) __attribute__((asdf));", Spaces);

  // Run subset of tests again with:
  Spaces.SpacesInParensOptions.InCStyleCasts = false;
  Spaces.SpaceAfterCStyleCast = true;
  verifyFormat("while ((bool) 1)\n"
               "  continue;",
               Spaces);
  verifyFormat("do {\n"
               "  do_something((int) i);\n"
               "} while (something( ));",
               Spaces);

  verifyFormat("size_t idx = (size_t) (ptr - ((char *) file));", Spaces);
  verifyFormat("size_t idx = (size_t) a;", Spaces);
  verifyFormat("size_t idx = (size_t) (a - 1);", Spaces);
  verifyFormat("size_t idx = (a->*foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (a->foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (*foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (*(foo))(a - 1);", Spaces);
  verifyFormat("#define CONF_BOOL(x) (bool *) (void *) (x)", Spaces);
  verifyFormat("#define CONF_BOOL(x) (bool *) (void *) (int) (x)", Spaces);
  verifyFormat("bool *y = (bool *) (void *) (x);", Spaces);
  verifyFormat("bool *y = (bool *) (void *) (int) (x);", Spaces);
  verifyFormat("bool *y = (bool *) (void *) (int) foo(x);", Spaces);
  verifyFormat("throw (int32) x;", Spaces);
  verifyFormat("SomeType *__attribute__((attr)) *a = NULL;", Spaces);
  verifyFormat("void __attribute__((naked)) foo(int bar)", Spaces);
  verifyFormat("void f( ) __attribute__((asdf));", Spaces);

  Spaces.ColumnLimit = 80;
  Spaces.IndentWidth = 4;
  Spaces.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  verifyFormat("void foo( ) {\n"
               "    size_t foo = (*(function))(\n"
               "        Foooo, Barrrrr, Foooo, Barrrr, FoooooooooLooooong, "
               "BarrrrrrrrrrrrLong,\n"
               "        FoooooooooLooooong);\n"
               "}",
               Spaces);
  Spaces.SpaceAfterCStyleCast = false;
  verifyFormat("size_t idx = (size_t)(ptr - ((char *)file));", Spaces);
  verifyFormat("size_t idx = (size_t)a;", Spaces);
  verifyFormat("size_t idx = (size_t)(a - 1);", Spaces);
  verifyFormat("size_t idx = (a->*foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (a->foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (*foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (*(foo))(a - 1);", Spaces);

  verifyFormat("void foo( ) {\n"
               "    size_t foo = (*(function))(\n"
               "        Foooo, Barrrrr, Foooo, Barrrr, FoooooooooLooooong, "
               "BarrrrrrrrrrrrLong,\n"
               "        FoooooooooLooooong);\n"
               "}",
               Spaces);

  Spaces.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;
  verifyFormat("void foo( ) {\n"
               "    size_t foo = (*(function))(\n"
               "        Foooo, Barrrrr, Foooo, Barrrr, FoooooooooLooooong, "
               "BarrrrrrrrrrrrLong,\n"
               "        FoooooooooLooooong\n"
               "    );\n"
               "}",
               Spaces);
  verifyFormat("size_t idx = (size_t)(ptr - ((char *)file));", Spaces);
  verifyFormat("size_t idx = (size_t)a;", Spaces);
  verifyFormat("size_t idx = (size_t)(a - 1);", Spaces);
  verifyFormat("size_t idx = (a->*foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (a->foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (*foo)(a - 1);", Spaces);
  verifyFormat("size_t idx = (*(foo))(a - 1);", Spaces);
}

TEST_F(FormatTest, ConfigurableSpacesInSquareBrackets) {
  verifyFormat("int a[5];");
  verifyFormat("a[3] += 42;");

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInSquareBrackets = true;
  // Not lambdas.
  verifyFormat("int a[ 5 ];", Spaces);
  verifyFormat("a[ 3 ] += 42;", Spaces);
  verifyFormat("constexpr char hello[]{\"hello\"};", Spaces);
  verifyFormat("double &operator[](int i) { return 0; }\n"
               "int i;",
               Spaces);
  verifyFormat("std::unique_ptr<int[]> foo() {}", Spaces);
  verifyFormat("int i = a[ a ][ a ]->f();", Spaces);
  verifyFormat("int i = (*b)[ a ]->f();", Spaces);
  // Lambdas.
  verifyFormat("int c = []() -> int { return 2; }();", Spaces);
  verifyFormat("return [ i, args... ] {};", Spaces);
  verifyFormat("int foo = [ &bar ]() {};", Spaces);
  verifyFormat("int foo = [ = ]() {};", Spaces);
  verifyFormat("int foo = [ & ]() {};", Spaces);
  verifyFormat("int foo = [ =, &bar ]() {};", Spaces);
  verifyFormat("int foo = [ &bar, = ]() {};", Spaces);
}

TEST_F(FormatTest, ConfigurableSpaceBeforeBrackets) {
  FormatStyle NoSpaceStyle = getLLVMStyle();
  verifyFormat("int a[5];", NoSpaceStyle);
  verifyFormat("a[3] += 42;", NoSpaceStyle);

  verifyFormat("int a[1];", NoSpaceStyle);
  verifyFormat("int 1 [a];", NoSpaceStyle);
  verifyFormat("int a[1][2];", NoSpaceStyle);
  verifyFormat("a[7] = 5;", NoSpaceStyle);
  verifyFormat("int a = (f())[23];", NoSpaceStyle);
  verifyFormat("f([] {})", NoSpaceStyle);

  FormatStyle Space = getLLVMStyle();
  Space.SpaceBeforeSquareBrackets = true;
  verifyFormat("int c = []() -> int { return 2; }();", Space);
  verifyFormat("return [i, args...] {};", Space);

  verifyFormat("int a [5];", Space);
  verifyFormat("a [3] += 42;", Space);
  verifyFormat("constexpr char hello []{\"hello\"};", Space);
  verifyFormat("double &operator[](int i) { return 0; }\n"
               "int i;",
               Space);
  verifyFormat("std::unique_ptr<int []> foo() {}", Space);
  verifyFormat("int i = a [a][a]->f();", Space);
  verifyFormat("int i = (*b) [a]->f();", Space);

  verifyFormat("int a [1];", Space);
  verifyFormat("int 1 [a];", Space);
  verifyFormat("int a [1][2];", Space);
  verifyFormat("a [7] = 5;", Space);
  verifyFormat("int a = (f()) [23];", Space);
  verifyFormat("f([] {})", Space);
}

TEST_F(FormatTest, ConfigurableSpaceBeforeAssignmentOperators) {
  verifyFormat("int a = 5;");
  verifyFormat("a += 42;");
  verifyFormat("a or_eq 8;");

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpaceBeforeAssignmentOperators = false;
  verifyFormat("int a= 5;", Spaces);
  verifyFormat("a+= 42;", Spaces);
  verifyFormat("a or_eq 8;", Spaces);
}

TEST_F(FormatTest, ConfigurableSpaceBeforeColon) {
  verifyFormat("class Foo : public Bar {};");
  verifyFormat("Foo::Foo() : foo(1) {}");
  verifyFormat("for (auto a : b) {\n}");
  verifyFormat("int x = a ? b : c;");
  verifyFormat("{\n"
               "label0:\n"
               "  int x = 0;\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "default:\n"
               "}");
  verifyFormat("switch (allBraces) {\n"
               "case 1: {\n"
               "  break;\n"
               "}\n"
               "case 2: {\n"
               "  [[fallthrough]];\n"
               "}\n"
               "default: {\n"
               "  break;\n"
               "}\n"
               "}");

  FormatStyle CtorInitializerStyle = getLLVMStyleWithColumns(30);
  CtorInitializerStyle.SpaceBeforeCtorInitializerColon = false;
  verifyFormat("class Foo : public Bar {};", CtorInitializerStyle);
  verifyFormat("Foo::Foo(): foo(1) {}", CtorInitializerStyle);
  verifyFormat("for (auto a : b) {\n}", CtorInitializerStyle);
  verifyFormat("int x = a ? b : c;", CtorInitializerStyle);
  verifyFormat("{\n"
               "label1:\n"
               "  int x = 0;\n"
               "}",
               CtorInitializerStyle);
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "default:\n"
               "}",
               CtorInitializerStyle);
  verifyFormat("switch (allBraces) {\n"
               "case 1: {\n"
               "  break;\n"
               "}\n"
               "case 2: {\n"
               "  [[fallthrough]];\n"
               "}\n"
               "default: {\n"
               "  break;\n"
               "}\n"
               "}",
               CtorInitializerStyle);
  CtorInitializerStyle.BreakConstructorInitializers =
      FormatStyle::BCIS_AfterColon;
  verifyFormat("Fooooooooooo::Fooooooooooo():\n"
               "    aaaaaaaaaaaaaaaa(1),\n"
               "    bbbbbbbbbbbbbbbb(2) {}",
               CtorInitializerStyle);
  CtorInitializerStyle.BreakConstructorInitializers =
      FormatStyle::BCIS_BeforeComma;
  verifyFormat("Fooooooooooo::Fooooooooooo()\n"
               "    : aaaaaaaaaaaaaaaa(1)\n"
               "    , bbbbbbbbbbbbbbbb(2) {}",
               CtorInitializerStyle);
  CtorInitializerStyle.BreakConstructorInitializers =
      FormatStyle::BCIS_BeforeColon;
  verifyFormat("Fooooooooooo::Fooooooooooo()\n"
               "    : aaaaaaaaaaaaaaaa(1),\n"
               "      bbbbbbbbbbbbbbbb(2) {}",
               CtorInitializerStyle);
  CtorInitializerStyle.ConstructorInitializerIndentWidth = 0;
  verifyFormat("Fooooooooooo::Fooooooooooo()\n"
               ": aaaaaaaaaaaaaaaa(1),\n"
               "  bbbbbbbbbbbbbbbb(2) {}",
               CtorInitializerStyle);

  FormatStyle InheritanceStyle = getLLVMStyleWithColumns(30);
  InheritanceStyle.SpaceBeforeInheritanceColon = false;
  verifyFormat("class Foo: public Bar {};", InheritanceStyle);
  verifyFormat("Foo::Foo() : foo(1) {}", InheritanceStyle);
  verifyFormat("for (auto a : b) {\n}", InheritanceStyle);
  verifyFormat("int x = a ? b : c;", InheritanceStyle);
  verifyFormat("{\n"
               "label2:\n"
               "  int x = 0;\n"
               "}",
               InheritanceStyle);
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "default:\n"
               "}",
               InheritanceStyle);
  verifyFormat("switch (allBraces) {\n"
               "case 1: {\n"
               "  break;\n"
               "}\n"
               "case 2: {\n"
               "  [[fallthrough]];\n"
               "}\n"
               "default: {\n"
               "  break;\n"
               "}\n"
               "}",
               InheritanceStyle);
  InheritanceStyle.BreakInheritanceList = FormatStyle::BILS_AfterComma;
  verifyFormat("class Foooooooooooooooooooooo\n"
               "    : public aaaaaaaaaaaaaaaaaa,\n"
               "      public bbbbbbbbbbbbbbbbbb {\n"
               "}",
               InheritanceStyle);
  InheritanceStyle.BreakInheritanceList = FormatStyle::BILS_AfterColon;
  verifyFormat("class Foooooooooooooooooooooo:\n"
               "    public aaaaaaaaaaaaaaaaaa,\n"
               "    public bbbbbbbbbbbbbbbbbb {\n"
               "}",
               InheritanceStyle);
  InheritanceStyle.BreakInheritanceList = FormatStyle::BILS_BeforeComma;
  verifyFormat("class Foooooooooooooooooooooo\n"
               "    : public aaaaaaaaaaaaaaaaaa\n"
               "    , public bbbbbbbbbbbbbbbbbb {\n"
               "}",
               InheritanceStyle);
  InheritanceStyle.BreakInheritanceList = FormatStyle::BILS_BeforeColon;
  verifyFormat("class Foooooooooooooooooooooo\n"
               "    : public aaaaaaaaaaaaaaaaaa,\n"
               "      public bbbbbbbbbbbbbbbbbb {\n"
               "}",
               InheritanceStyle);
  InheritanceStyle.ConstructorInitializerIndentWidth = 0;
  verifyFormat("class Foooooooooooooooooooooo\n"
               ": public aaaaaaaaaaaaaaaaaa,\n"
               "  public bbbbbbbbbbbbbbbbbb {}",
               InheritanceStyle);

  FormatStyle ForLoopStyle = getLLVMStyle();
  ForLoopStyle.SpaceBeforeRangeBasedForLoopColon = false;
  verifyFormat("class Foo : public Bar {};", ForLoopStyle);
  verifyFormat("Foo::Foo() : foo(1) {}", ForLoopStyle);
  verifyFormat("for (auto a: b) {\n}", ForLoopStyle);
  verifyFormat("int x = a ? b : c;", ForLoopStyle);
  verifyFormat("{\n"
               "label2:\n"
               "  int x = 0;\n"
               "}",
               ForLoopStyle);
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "default:\n"
               "}",
               ForLoopStyle);
  verifyFormat("switch (allBraces) {\n"
               "case 1: {\n"
               "  break;\n"
               "}\n"
               "case 2: {\n"
               "  [[fallthrough]];\n"
               "}\n"
               "default: {\n"
               "  break;\n"
               "}\n"
               "}",
               ForLoopStyle);

  FormatStyle CaseStyle = getLLVMStyle();
  CaseStyle.SpaceBeforeCaseColon = true;
  verifyFormat("class Foo : public Bar {};", CaseStyle);
  verifyFormat("Foo::Foo() : foo(1) {}", CaseStyle);
  verifyFormat("for (auto a : b) {\n}", CaseStyle);
  verifyFormat("int x = a ? b : c;", CaseStyle);
  verifyFormat("switch (x) {\n"
               "case 1 :\n"
               "default :\n"
               "}",
               CaseStyle);
  verifyFormat("switch (allBraces) {\n"
               "case 1 : {\n"
               "  break;\n"
               "}\n"
               "case 2 : {\n"
               "  [[fallthrough]];\n"
               "}\n"
               "default : {\n"
               "  break;\n"
               "}\n"
               "}",
               CaseStyle);
  // Goto labels should not be affected.
  verifyFormat("switch (x) {\n"
               "goto_label:\n"
               "default :\n"
               "}",
               CaseStyle);
  verifyFormat("switch (x) {\n"
               "goto_label: { break; }\n"
               "default : {\n"
               "  break;\n"
               "}\n"
               "}",
               CaseStyle);

  FormatStyle NoSpaceStyle = getLLVMStyle();
  EXPECT_EQ(NoSpaceStyle.SpaceBeforeCaseColon, false);
  NoSpaceStyle.SpaceBeforeCtorInitializerColon = false;
  NoSpaceStyle.SpaceBeforeInheritanceColon = false;
  NoSpaceStyle.SpaceBeforeRangeBasedForLoopColon = false;
  verifyFormat("class Foo: public Bar {};", NoSpaceStyle);
  verifyFormat("Foo::Foo(): foo(1) {}", NoSpaceStyle);
  verifyFormat("for (auto a: b) {\n}", NoSpaceStyle);
  verifyFormat("int x = a ? b : c;", NoSpaceStyle);
  verifyFormat("{\n"
               "label3:\n"
               "  int x = 0;\n"
               "}",
               NoSpaceStyle);
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "default:\n"
               "}",
               NoSpaceStyle);
  verifyFormat("switch (allBraces) {\n"
               "case 1: {\n"
               "  break;\n"
               "}\n"
               "case 2: {\n"
               "  [[fallthrough]];\n"
               "}\n"
               "default: {\n"
               "  break;\n"
               "}\n"
               "}",
               NoSpaceStyle);

  FormatStyle InvertedSpaceStyle = getLLVMStyle();
  InvertedSpaceStyle.SpaceBeforeCaseColon = true;
  InvertedSpaceStyle.SpaceBeforeCtorInitializerColon = false;
  InvertedSpaceStyle.SpaceBeforeInheritanceColon = false;
  InvertedSpaceStyle.SpaceBeforeRangeBasedForLoopColon = false;
  verifyFormat("class Foo: public Bar {};", InvertedSpaceStyle);
  verifyFormat("Foo::Foo(): foo(1) {}", InvertedSpaceStyle);
  verifyFormat("for (auto a: b) {\n}", InvertedSpaceStyle);
  verifyFormat("int x = a ? b : c;", InvertedSpaceStyle);
  verifyFormat("{\n"
               "label3:\n"
               "  int x = 0;\n"
               "}",
               InvertedSpaceStyle);
  verifyFormat("switch (x) {\n"
               "case 1 :\n"
               "case 2 : {\n"
               "  break;\n"
               "}\n"
               "default :\n"
               "  break;\n"
               "}",
               InvertedSpaceStyle);
  verifyFormat("switch (allBraces) {\n"
               "case 1 : {\n"
               "  break;\n"
               "}\n"
               "case 2 : {\n"
               "  [[fallthrough]];\n"
               "}\n"
               "default : {\n"
               "  break;\n"
               "}\n"
               "}",
               InvertedSpaceStyle);
}

TEST_F(FormatTest, ConfigurableSpaceAroundPointerQualifiers) {
  FormatStyle Style = getLLVMStyle();

  Style.PointerAlignment = FormatStyle::PAS_Left;
  Style.SpaceAroundPointerQualifiers = FormatStyle::SAPQ_Default;
  verifyFormat("void* const* x = NULL;", Style);

#define verifyQualifierSpaces(Code, Pointers, Qualifiers)                      \
  do {                                                                         \
    Style.PointerAlignment = FormatStyle::Pointers;                            \
    Style.SpaceAroundPointerQualifiers = FormatStyle::Qualifiers;              \
    verifyFormat(Code, Style);                                                 \
  } while (false)

  verifyQualifierSpaces("void* const* x = NULL;", PAS_Left, SAPQ_Default);
  verifyQualifierSpaces("void *const *x = NULL;", PAS_Right, SAPQ_Default);
  verifyQualifierSpaces("void * const * x = NULL;", PAS_Middle, SAPQ_Default);

  verifyQualifierSpaces("void* const* x = NULL;", PAS_Left, SAPQ_Before);
  verifyQualifierSpaces("void * const *x = NULL;", PAS_Right, SAPQ_Before);
  verifyQualifierSpaces("void * const * x = NULL;", PAS_Middle, SAPQ_Before);

  verifyQualifierSpaces("void* const * x = NULL;", PAS_Left, SAPQ_After);
  verifyQualifierSpaces("void *const *x = NULL;", PAS_Right, SAPQ_After);
  verifyQualifierSpaces("void * const * x = NULL;", PAS_Middle, SAPQ_After);

  verifyQualifierSpaces("void* const * x = NULL;", PAS_Left, SAPQ_Both);
  verifyQualifierSpaces("void * const *x = NULL;", PAS_Right, SAPQ_Both);
  verifyQualifierSpaces("void * const * x = NULL;", PAS_Middle, SAPQ_Both);

  verifyQualifierSpaces("Foo::operator void const*();", PAS_Left, SAPQ_Default);
  verifyQualifierSpaces("Foo::operator void const *();", PAS_Right,
                        SAPQ_Default);
  verifyQualifierSpaces("Foo::operator void const *();", PAS_Middle,
                        SAPQ_Default);

  verifyQualifierSpaces("Foo::operator void const*();", PAS_Left, SAPQ_Before);
  verifyQualifierSpaces("Foo::operator void const *();", PAS_Right,
                        SAPQ_Before);
  verifyQualifierSpaces("Foo::operator void const *();", PAS_Middle,
                        SAPQ_Before);

  verifyQualifierSpaces("Foo::operator void const *();", PAS_Left, SAPQ_After);
  verifyQualifierSpaces("Foo::operator void const *();", PAS_Right, SAPQ_After);
  verifyQualifierSpaces("Foo::operator void const *();", PAS_Middle,
                        SAPQ_After);

  verifyQualifierSpaces("Foo::operator void const *();", PAS_Left, SAPQ_Both);
  verifyQualifierSpaces("Foo::operator void const *();", PAS_Right, SAPQ_Both);
  verifyQualifierSpaces("Foo::operator void const *();", PAS_Middle, SAPQ_Both);

#undef verifyQualifierSpaces

  FormatStyle Spaces = getLLVMStyle();
  Spaces.AttributeMacros.push_back("qualified");
  Spaces.PointerAlignment = FormatStyle::PAS_Right;
  Spaces.SpaceAroundPointerQualifiers = FormatStyle::SAPQ_Default;
  verifyFormat("SomeType *volatile *a = NULL;", Spaces);
  verifyFormat("SomeType *__attribute__((attr)) *a = NULL;", Spaces);
  verifyFormat("std::vector<SomeType *const *> x;", Spaces);
  verifyFormat("std::vector<SomeType *qualified *> x;", Spaces);
  verifyFormat("std::vector<SomeVar * NotAQualifier> x;", Spaces);
  Spaces.SpaceAroundPointerQualifiers = FormatStyle::SAPQ_Before;
  verifyFormat("SomeType * volatile *a = NULL;", Spaces);
  verifyFormat("SomeType * __attribute__((attr)) *a = NULL;", Spaces);
  verifyFormat("std::vector<SomeType * const *> x;", Spaces);
  verifyFormat("std::vector<SomeType * qualified *> x;", Spaces);
  verifyFormat("std::vector<SomeVar * NotAQualifier> x;", Spaces);

  // Check that SAPQ_Before doesn't result in extra spaces for PAS_Left.
  Spaces.PointerAlignment = FormatStyle::PAS_Left;
  Spaces.SpaceAroundPointerQualifiers = FormatStyle::SAPQ_Before;
  verifyFormat("SomeType* volatile* a = NULL;", Spaces);
  verifyFormat("SomeType* __attribute__((attr))* a = NULL;", Spaces);
  verifyFormat("std::vector<SomeType* const*> x;", Spaces);
  verifyFormat("std::vector<SomeType* qualified*> x;", Spaces);
  verifyFormat("std::vector<SomeVar * NotAQualifier> x;", Spaces);
  // However, setting it to SAPQ_After should add spaces after __attribute, etc.
  Spaces.SpaceAroundPointerQualifiers = FormatStyle::SAPQ_After;
  verifyFormat("SomeType* volatile * a = NULL;", Spaces);
  verifyFormat("SomeType* __attribute__((attr)) * a = NULL;", Spaces);
  verifyFormat("std::vector<SomeType* const *> x;", Spaces);
  verifyFormat("std::vector<SomeType* qualified *> x;", Spaces);
  verifyFormat("std::vector<SomeVar * NotAQualifier> x;", Spaces);

  // PAS_Middle should not have any noticeable changes even for SAPQ_Both
  Spaces.PointerAlignment = FormatStyle::PAS_Middle;
  Spaces.SpaceAroundPointerQualifiers = FormatStyle::SAPQ_After;
  verifyFormat("SomeType * volatile * a = NULL;", Spaces);
  verifyFormat("SomeType * __attribute__((attr)) * a = NULL;", Spaces);
  verifyFormat("std::vector<SomeType * const *> x;", Spaces);
  verifyFormat("std::vector<SomeType * qualified *> x;", Spaces);
  verifyFormat("std::vector<SomeVar * NotAQualifier> x;", Spaces);
}

TEST_F(FormatTest, AlignConsecutiveMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.AlignConsecutiveDeclarations.Enabled = true;

  verifyFormat("#define a 3\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  verifyFormat("#define f(x) (x * x)\n"
               "#define fff(x, y, z) (x * y + z)\n"
               "#define ffff(x, y) (x - y)",
               Style);

  verifyFormat("#define foo(x, y) (x + y)\n"
               "#define bar (5, 6)(2 + 2)",
               Style);

  verifyFormat("#define a 3\n"
               "#define bbbb 4\n"
               "#define ccc (5)\n"
               "#define f(x) (x * x)\n"
               "#define fff(x, y, z) (x * y + z)\n"
               "#define ffff(x, y) (x - y)",
               Style);

  Style.AlignConsecutiveMacros.Enabled = true;
  verifyFormat("#define a    3\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               Style);

  verifyFormat("#define true  1\n"
               "#define false 0",
               Style);

  verifyFormat("#define f(x)         (x * x)\n"
               "#define fff(x, y, z) (x * y + z)\n"
               "#define ffff(x, y)   (x - y)",
               Style);

  verifyFormat("#define foo(x, y) (x + y)\n"
               "#define bar       (5, 6)(2 + 2)",
               Style);

  verifyFormat("#define a            3\n"
               "#define bbbb         4\n"
               "#define ccc          (5)\n"
               "#define f(x)         (x * x)\n"
               "#define fff(x, y, z) (x * y + z)\n"
               "#define ffff(x, y)   (x - y)",
               Style);

  verifyFormat("#define a         5\n"
               "#define foo(x, y) (x + y)\n"
               "#define CCC       (6)\n"
               "auto lambda = []() {\n"
               "  auto  ii = 0;\n"
               "  float j  = 0;\n"
               "  return 0;\n"
               "};\n"
               "int   i  = 0;\n"
               "float i2 = 0;\n"
               "auto  v  = type{\n"
               "    i = 1,   //\n"
               "    (i = 2), //\n"
               "    i = 3    //\n"
               "};",
               Style);

  Style.AlignConsecutiveMacros.Enabled = false;
  Style.ColumnLimit = 20;

  verifyFormat("#define a          \\\n"
               "  \"aabbbbbbbbbbbb\"\n"
               "#define D          \\\n"
               "  \"aabbbbbbbbbbbb\" \\\n"
               "  \"ccddeeeeeeeee\"\n"
               "#define B          \\\n"
               "  \"QQQQQQQQQQQQQ\"  \\\n"
               "  \"FFFFFFFFFFFFF\"  \\\n"
               "  \"LLLLLLLL\"",
               Style);

  Style.AlignConsecutiveMacros.Enabled = true;
  verifyFormat("#define a          \\\n"
               "  \"aabbbbbbbbbbbb\"\n"
               "#define D          \\\n"
               "  \"aabbbbbbbbbbbb\" \\\n"
               "  \"ccddeeeeeeeee\"\n"
               "#define B          \\\n"
               "  \"QQQQQQQQQQQQQ\"  \\\n"
               "  \"FFFFFFFFFFFFF\"  \\\n"
               "  \"LLLLLLLL\"",
               Style);

  // Test across comments
  Style.MaxEmptyLinesToKeep = 10;
  Style.ReflowComments = false;
  Style.AlignConsecutiveMacros.AcrossComments = true;
  verifyFormat("#define a    3\n"
               "// line comment\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a 3\n"
               "// line comment\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  verifyFormat("#define a    3\n"
               "/* block comment */\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a  3\n"
               "/* block comment */\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  verifyFormat("#define a    3\n"
               "/* multi-line *\n"
               " * block comment */\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a 3\n"
               "/* multi-line *\n"
               " * block comment */\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  verifyFormat("#define a    3\n"
               "// multi-line line comment\n"
               "//\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a  3\n"
               "// multi-line line comment\n"
               "//\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  verifyFormat("#define a 3\n"
               "// empty lines still break.\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a     3\n"
               "// empty lines still break.\n"
               "\n"
               "#define bbbb     4\n"
               "#define ccc  (5)",
               Style);

  // Test across empty lines
  Style.AlignConsecutiveMacros.AcrossComments = false;
  Style.AlignConsecutiveMacros.AcrossEmptyLines = true;
  verifyFormat("#define a    3\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a 3\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  verifyFormat("#define a    3\n"
               "\n"
               "\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a        3\n"
               "\n"
               "\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  verifyFormat("#define a 3\n"
               "// comments should break alignment\n"
               "//\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a        3\n"
               "// comments should break alignment\n"
               "//\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  // Test across empty lines and comments
  Style.AlignConsecutiveMacros.AcrossComments = true;
  verifyFormat("#define a    3\n"
               "\n"
               "// line comment\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               Style);

  verifyFormat("#define a    3\n"
               "\n"
               "\n"
               "/* multi-line *\n"
               " * block comment */\n"
               "\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a 3\n"
               "\n"
               "\n"
               "/* multi-line *\n"
               " * block comment */\n"
               "\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc (5)",
               Style);

  verifyFormat("#define a    3\n"
               "\n"
               "\n"
               "/* multi-line *\n"
               " * block comment */\n"
               "\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc  (5)",
               "#define a 3\n"
               "\n"
               "\n"
               "/* multi-line *\n"
               " * block comment */\n"
               "\n"
               "\n"
               "#define bbbb 4\n"
               "#define ccc       (5)",
               Style);
}

TEST_F(FormatTest, AlignConsecutiveAssignmentsAcrossEmptyLines) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveMacros.Enabled = true;
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  Alignment.AlignConsecutiveAssignments.AcrossEmptyLines = true;

  Alignment.MaxEmptyLinesToKeep = 10;
  /* Test alignment across empty lines */
  verifyFormat("int a           = 5;\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a       = 5;\n"
               "\n"
               "int oneTwoThree= 123;",
               Alignment);
  verifyFormat("int a           = 5;\n"
               "int one         = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a           = 5;\n"
               "int one         = 1;\n"
               "\n"
               "int oneTwoThree = 123;\n"
               "int oneTwo      = 12;",
               "int a = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;\n"
               "int oneTwo = 12;",
               Alignment);

  /* Test across comments */
  verifyFormat("int a = 5;\n"
               "/* block comment */\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "/* block comment */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a = 5;\n"
               "// line comment\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "// line comment\n"
               "int oneTwoThree=123;",
               Alignment);

  /* Test across comments and newlines */
  verifyFormat("int a = 5;\n"
               "\n"
               "/* block comment */\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "/* block comment */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a = 5;\n"
               "\n"
               "// line comment\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "// line comment\n"
               "int oneTwoThree=123;",
               Alignment);
}

TEST_F(FormatTest, AlignConsecutiveDeclarationsAcrossEmptyLinesAndComments) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveDeclarations.Enabled = true;
  Alignment.AlignConsecutiveDeclarations.AcrossEmptyLines = true;
  Alignment.AlignConsecutiveDeclarations.AcrossComments = true;

  Alignment.MaxEmptyLinesToKeep = 10;
  /* Test alignment across empty lines */
  verifyFormat("int         a = 5;\n"
               "\n"
               "float const oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "float const oneTwoThree = 123;",
               Alignment);
  verifyFormat("int         a = 5;\n"
               "float const one = 1;\n"
               "\n"
               "int         oneTwoThree = 123;",
               "int a = 5;\n"
               "float const one = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               Alignment);

  /* Test across comments */
  verifyFormat("float const a = 5;\n"
               "/* block comment */\n"
               "int         oneTwoThree = 123;",
               "float const a = 5;\n"
               "/* block comment */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("float const a = 5;\n"
               "// line comment\n"
               "int         oneTwoThree = 123;",
               "float const a = 5;\n"
               "// line comment\n"
               "int oneTwoThree=123;",
               Alignment);

  /* Test across comments and newlines */
  verifyFormat("float const a = 5;\n"
               "\n"
               "/* block comment */\n"
               "int         oneTwoThree = 123;",
               "float const a = 5;\n"
               "\n"
               "/* block comment */\n"
               "int         oneTwoThree=123;",
               Alignment);

  verifyFormat("float const a = 5;\n"
               "\n"
               "// line comment\n"
               "int         oneTwoThree = 123;",
               "float const a = 5;\n"
               "\n"
               "// line comment\n"
               "int oneTwoThree=123;",
               Alignment);
}

TEST_F(FormatTest, AlignConsecutiveBitFieldsAcrossEmptyLinesAndComments) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveBitFields.Enabled = true;
  Alignment.AlignConsecutiveBitFields.AcrossEmptyLines = true;
  Alignment.AlignConsecutiveBitFields.AcrossComments = true;

  Alignment.MaxEmptyLinesToKeep = 10;
  /* Test alignment across empty lines */
  verifyFormat("int a            : 5;\n"
               "\n"
               "int longbitfield : 6;",
               "int a : 5;\n"
               "\n"
               "int longbitfield : 6;",
               Alignment);
  verifyFormat("int a            : 5;\n"
               "int one          : 1;\n"
               "\n"
               "int longbitfield : 6;",
               "int a : 5;\n"
               "int one : 1;\n"
               "\n"
               "int longbitfield : 6;",
               Alignment);

  /* Test across comments */
  verifyFormat("int a            : 5;\n"
               "/* block comment */\n"
               "int longbitfield : 6;",
               "int a : 5;\n"
               "/* block comment */\n"
               "int longbitfield : 6;",
               Alignment);
  verifyFormat("int a            : 5;\n"
               "int one          : 1;\n"
               "// line comment\n"
               "int longbitfield : 6;",
               "int a : 5;\n"
               "int one : 1;\n"
               "// line comment\n"
               "int longbitfield : 6;",
               Alignment);

  /* Test across comments and newlines */
  verifyFormat("int a            : 5;\n"
               "/* block comment */\n"
               "\n"
               "int longbitfield : 6;",
               "int a : 5;\n"
               "/* block comment */\n"
               "\n"
               "int longbitfield : 6;",
               Alignment);
  verifyFormat("int a            : 5;\n"
               "int one          : 1;\n"
               "\n"
               "// line comment\n"
               "\n"
               "int longbitfield : 6;",
               "int a : 5;\n"
               "int one : 1;\n"
               "\n"
               "// line comment \n"
               "\n"
               "int longbitfield : 6;",
               Alignment);
}

TEST_F(FormatTest, AlignConsecutiveAssignmentsAcrossComments) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveMacros.Enabled = true;
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  Alignment.AlignConsecutiveAssignments.AcrossComments = true;

  Alignment.MaxEmptyLinesToKeep = 10;
  /* Test alignment across empty lines */
  verifyFormat("int a = 5;\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a       = 5;\n"
               "\n"
               "int oneTwoThree= 123;",
               Alignment);
  verifyFormat("int a   = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               Alignment);

  /* Test across comments */
  verifyFormat("int a           = 5;\n"
               "/* block comment */\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "/* block comment */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "// line comment\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "// line comment\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "/*\n"
               " * multi-line block comment\n"
               " */\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "/*\n"
               " * multi-line block comment\n"
               " */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "//\n"
               "// multi-line line comment\n"
               "//\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "//\n"
               "// multi-line line comment\n"
               "//\n"
               "int oneTwoThree=123;",
               Alignment);

  /* Test across comments and newlines */
  verifyFormat("int a = 5;\n"
               "\n"
               "/* block comment */\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "/* block comment */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a = 5;\n"
               "\n"
               "// line comment\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "// line comment\n"
               "int oneTwoThree=123;",
               Alignment);
}

TEST_F(FormatTest, AlignConsecutiveAssignmentsAcrossEmptyLinesAndComments) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveMacros.Enabled = true;
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  Alignment.AlignConsecutiveAssignments.AcrossEmptyLines = true;
  Alignment.AlignConsecutiveAssignments.AcrossComments = true;
  verifyFormat("int a           = 5;\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a           = method();\n"
               "int oneTwoThree = 133;",
               Alignment);
  verifyFormat("a &= 5;\n"
               "bcd *= 5;\n"
               "ghtyf += 5;\n"
               "dvfvdb -= 5;\n"
               "a /= 5;\n"
               "vdsvsv %= 5;\n"
               "sfdbddfbdfbb ^= 5;\n"
               "dvsdsv |= 5;\n"
               "int dsvvdvsdvvv = 123;",
               Alignment);
  verifyFormat("int i = 1, j = 10;\n"
               "something = 2000;",
               Alignment);
  verifyFormat("something = 2000;\n"
               "int i = 1, j = 10;",
               Alignment);
  verifyFormat("something = 2000;\n"
               "another   = 911;\n"
               "int i = 1, j = 10;\n"
               "oneMore = 1;\n"
               "i       = 2;",
               Alignment);
  verifyFormat("int a   = 5;\n"
               "int one = 1;\n"
               "method();\n"
               "int oneTwoThree = 123;\n"
               "int oneTwo      = 12;",
               Alignment);
  verifyFormat("int oneTwoThree = 123;\n"
               "int oneTwo      = 12;\n"
               "method();",
               Alignment);
  verifyFormat("int oneTwoThree = 123; // comment\n"
               "int oneTwo      = 12;  // comment",
               Alignment);

  // Bug 25167
  /* Uncomment when fixed
    verifyFormat("#if A\n"
                 "#else\n"
                 "int aaaaaaaa = 12;\n"
                 "#endif\n"
                 "#if B\n"
                 "#else\n"
                 "int a = 12;\n"
                 "#endif",
                 Alignment);
    verifyFormat("enum foo {\n"
                 "#if A\n"
                 "#else\n"
                 "  aaaaaaaa = 12;\n"
                 "#endif\n"
                 "#if B\n"
                 "#else\n"
                 "  a = 12;\n"
                 "#endif\n"
                 "};",
                 Alignment);
  */

  Alignment.MaxEmptyLinesToKeep = 10;
  /* Test alignment across empty lines */
  verifyFormat("int a           = 5;\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a       = 5;\n"
               "\n"
               "int oneTwoThree= 123;",
               Alignment);
  verifyFormat("int a           = 5;\n"
               "int one         = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a           = 5;\n"
               "int one         = 1;\n"
               "\n"
               "int oneTwoThree = 123;\n"
               "int oneTwo      = 12;",
               "int a = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;\n"
               "int oneTwo = 12;",
               Alignment);

  /* Test across comments */
  verifyFormat("int a           = 5;\n"
               "/* block comment */\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "/* block comment */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "// line comment\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "// line comment\n"
               "int oneTwoThree=123;",
               Alignment);

  /* Test across comments and newlines */
  verifyFormat("int a           = 5;\n"
               "\n"
               "/* block comment */\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "/* block comment */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "\n"
               "// line comment\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "// line comment\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "//\n"
               "// multi-line line comment\n"
               "//\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "//\n"
               "// multi-line line comment\n"
               "//\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "/*\n"
               " *  multi-line block comment\n"
               " */\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "/*\n"
               " *  multi-line block comment\n"
               " */\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "\n"
               "/* block comment */\n"
               "\n"
               "\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "/* block comment */\n"
               "\n"
               "\n"
               "\n"
               "int oneTwoThree=123;",
               Alignment);

  verifyFormat("int a           = 5;\n"
               "\n"
               "// line comment\n"
               "\n"
               "\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "\n"
               "// line comment\n"
               "\n"
               "\n"
               "\n"
               "int oneTwoThree=123;",
               Alignment);

  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_DontAlign;
  verifyFormat("#define A \\\n"
               "  int aaaa       = 12; \\\n"
               "  int b          = 23; \\\n"
               "  int ccc        = 234; \\\n"
               "  int dddddddddd = 2345;",
               Alignment);
  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  verifyFormat("#define A               \\\n"
               "  int aaaa       = 12;  \\\n"
               "  int b          = 23;  \\\n"
               "  int ccc        = 234; \\\n"
               "  int dddddddddd = 2345;",
               Alignment);
  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_Right;
  verifyFormat("#define A                                                      "
               "                \\\n"
               "  int aaaa       = 12;                                         "
               "                \\\n"
               "  int b          = 23;                                         "
               "                \\\n"
               "  int ccc        = 234;                                        "
               "                \\\n"
               "  int dddddddddd = 2345;",
               Alignment);
  verifyFormat("void SomeFunction(int parameter = 1, int i = 2, int j = 3, int "
               "k = 4, int l = 5,\n"
               "                  int m = 6) {\n"
               "  int j      = 10;\n"
               "  otherThing = 1;\n"
               "}",
               Alignment);
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int i   = 1;\n"
               "  int j   = 2;\n"
               "  int big = 10000;\n"
               "}",
               Alignment);
  verifyFormat("class C {\n"
               "public:\n"
               "  int i            = 1;\n"
               "  virtual void f() = 0;\n"
               "};",
               Alignment);
  verifyFormat("int i = 1;\n"
               "if (SomeType t = getSomething()) {\n"
               "}\n"
               "int j   = 2;\n"
               "int big = 10000;",
               Alignment);
  verifyFormat("int j = 7;\n"
               "for (int k = 0; k < N; ++k) {\n"
               "}\n"
               "int j   = 2;\n"
               "int big = 10000;\n"
               "}",
               Alignment);
  Alignment.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat("int i = 1;\n"
               "LooooooooooongType loooooooooooooooooooooongVariable\n"
               "    = someLooooooooooooooooongFunction();\n"
               "int j = 2;",
               Alignment);
  Alignment.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  verifyFormat("int i = 1;\n"
               "LooooooooooongType loooooooooooooooooooooongVariable =\n"
               "    someLooooooooooooooooongFunction();\n"
               "int j = 2;",
               Alignment);

  verifyFormat("auto lambda = []() {\n"
               "  auto i = 0;\n"
               "  return 0;\n"
               "};\n"
               "int i  = 0;\n"
               "auto v = type{\n"
               "    i = 1,   //\n"
               "    (i = 2), //\n"
               "    i = 3    //\n"
               "};",
               Alignment);

  verifyFormat(
      "int i      = 1;\n"
      "SomeType a = SomeFunction(looooooooooooooooooooooongParameterA,\n"
      "                          loooooooooooooooooooooongParameterB);\n"
      "int j      = 2;",
      Alignment);

  verifyFormat("template <typename T, typename T_0 = very_long_type_name_0,\n"
               "          typename B   = very_long_type_name_1,\n"
               "          typename T_2 = very_long_type_name_2>\n"
               "auto foo() {}",
               Alignment);
  verifyFormat("int a, b = 1;\n"
               "int c  = 2;\n"
               "int dd = 3;",
               Alignment);
  verifyFormat("int aa       = ((1 > 2) ? 3 : 4);\n"
               "float b[1][] = {{3.f}};",
               Alignment);
  verifyFormat("for (int i = 0; i < 1; i++)\n"
               "  int x = 1;",
               Alignment);
  verifyFormat("for (i = 0; i < 1; i++)\n"
               "  x = 1;\n"
               "y = 1;",
               Alignment);

  Alignment.ReflowComments = true;
  Alignment.ColumnLimit = 50;
  verifyFormat("int x   = 0;\n"
               "int yy  = 1; /// specificlennospace\n"
               "int zzz = 2;",
               "int x   = 0;\n"
               "int yy  = 1; ///specificlennospace\n"
               "int zzz = 2;",
               Alignment);
}

TEST_F(FormatTest, AlignCompoundAssignments) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  Alignment.AlignConsecutiveAssignments.AlignCompound = true;
  Alignment.AlignConsecutiveAssignments.PadOperators = false;
  verifyFormat("sfdbddfbdfbb    = 5;\n"
               "dvsdsv          = 5;\n"
               "int dsvvdvsdvvv = 123;",
               Alignment);
  verifyFormat("sfdbddfbdfbb   ^= 5;\n"
               "dvsdsv         |= 5;\n"
               "int dsvvdvsdvvv = 123;",
               Alignment);
  verifyFormat("sfdbddfbdfbb   ^= 5;\n"
               "dvsdsv        <<= 5;\n"
               "int dsvvdvsdvvv = 123;",
               Alignment);
  verifyFormat("int xxx = 5;\n"
               "xxx     = 5;\n"
               "{\n"
               "  int yyy = 6;\n"
               "  yyy     = 6;\n"
               "}",
               Alignment);
  verifyFormat("int xxx = 5;\n"
               "xxx    += 5;\n"
               "{\n"
               "  int yyy = 6;\n"
               "  yyy    += 6;\n"
               "}",
               Alignment);
  // Test that `<=` is not treated as a compound assignment.
  verifyFormat("aa &= 5;\n"
               "b <= 10;\n"
               "c = 15;",
               Alignment);
  Alignment.AlignConsecutiveAssignments.PadOperators = true;
  verifyFormat("sfdbddfbdfbb    = 5;\n"
               "dvsdsv          = 5;\n"
               "int dsvvdvsdvvv = 123;",
               Alignment);
  verifyFormat("sfdbddfbdfbb    ^= 5;\n"
               "dvsdsv          |= 5;\n"
               "int dsvvdvsdvvv  = 123;",
               Alignment);
  verifyFormat("sfdbddfbdfbb     ^= 5;\n"
               "dvsdsv          <<= 5;\n"
               "int dsvvdvsdvvv   = 123;",
               Alignment);
  verifyFormat("a   += 5;\n"
               "one  = 1;\n"
               "\n"
               "oneTwoThree = 123;",
               "a += 5;\n"
               "one = 1;\n"
               "\n"
               "oneTwoThree = 123;",
               Alignment);
  verifyFormat("a   += 5;\n"
               "one  = 1;\n"
               "//\n"
               "oneTwoThree = 123;",
               "a += 5;\n"
               "one = 1;\n"
               "//\n"
               "oneTwoThree = 123;",
               Alignment);
  Alignment.AlignConsecutiveAssignments.AcrossEmptyLines = true;
  verifyFormat("a           += 5;\n"
               "one          = 1;\n"
               "\n"
               "oneTwoThree  = 123;",
               "a += 5;\n"
               "one = 1;\n"
               "\n"
               "oneTwoThree = 123;",
               Alignment);
  verifyFormat("a   += 5;\n"
               "one  = 1;\n"
               "//\n"
               "oneTwoThree = 123;",
               "a += 5;\n"
               "one = 1;\n"
               "//\n"
               "oneTwoThree = 123;",
               Alignment);
  Alignment.AlignConsecutiveAssignments.AcrossEmptyLines = false;
  Alignment.AlignConsecutiveAssignments.AcrossComments = true;
  verifyFormat("a   += 5;\n"
               "one  = 1;\n"
               "\n"
               "oneTwoThree = 123;",
               "a += 5;\n"
               "one = 1;\n"
               "\n"
               "oneTwoThree = 123;",
               Alignment);
  verifyFormat("a           += 5;\n"
               "one          = 1;\n"
               "//\n"
               "oneTwoThree  = 123;",
               "a += 5;\n"
               "one = 1;\n"
               "//\n"
               "oneTwoThree = 123;",
               Alignment);
  Alignment.AlignConsecutiveAssignments.AcrossEmptyLines = true;
  verifyFormat("a            += 5;\n"
               "one         >>= 1;\n"
               "\n"
               "oneTwoThree   = 123;",
               "a += 5;\n"
               "one >>= 1;\n"
               "\n"
               "oneTwoThree = 123;",
               Alignment);
  verifyFormat("a            += 5;\n"
               "one           = 1;\n"
               "//\n"
               "oneTwoThree <<= 123;",
               "a += 5;\n"
               "one = 1;\n"
               "//\n"
               "oneTwoThree <<= 123;",
               Alignment);
}

TEST_F(FormatTest, AlignConsecutiveAssignments) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveMacros.Enabled = true;
  verifyFormat("int a = 5;\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a = 5;\n"
               "int oneTwoThree = 123;",
               Alignment);

  Alignment.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("int a           = 5;\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a           = method();\n"
               "int oneTwoThree = 133;",
               Alignment);
  verifyFormat("aa <= 5;\n"
               "a &= 5;\n"
               "bcd *= 5;\n"
               "ghtyf += 5;\n"
               "dvfvdb -= 5;\n"
               "a /= 5;\n"
               "vdsvsv %= 5;\n"
               "sfdbddfbdfbb ^= 5;\n"
               "dvsdsv |= 5;\n"
               "int dsvvdvsdvvv = 123;",
               Alignment);
  verifyFormat("int i = 1, j = 10;\n"
               "something = 2000;",
               Alignment);
  verifyFormat("something = 2000;\n"
               "int i = 1, j = 10;",
               Alignment);
  verifyFormat("something = 2000;\n"
               "another   = 911;\n"
               "int i = 1, j = 10;\n"
               "oneMore = 1;\n"
               "i       = 2;",
               Alignment);
  verifyFormat("int a   = 5;\n"
               "int one = 1;\n"
               "method();\n"
               "int oneTwoThree = 123;\n"
               "int oneTwo      = 12;",
               Alignment);
  verifyFormat("int oneTwoThree = 123;\n"
               "int oneTwo      = 12;\n"
               "method();",
               Alignment);
  verifyFormat("int oneTwoThree = 123; // comment\n"
               "int oneTwo      = 12;  // comment",
               Alignment);
  verifyFormat("int f()         = default;\n"
               "int &operator() = default;\n"
               "int &operator=() {",
               Alignment);
  verifyFormat("int f()         = delete;\n"
               "int &operator() = delete;\n"
               "int &operator=() {",
               Alignment);
  verifyFormat("int f()         = default; // comment\n"
               "int &operator() = default; // comment\n"
               "int &operator=() {",
               Alignment);
  verifyFormat("int f()         = default;\n"
               "int &operator() = default;\n"
               "int &operator==() {",
               Alignment);
  verifyFormat("int f()         = default;\n"
               "int &operator() = default;\n"
               "int &operator<=() {",
               Alignment);
  verifyFormat("int f()         = default;\n"
               "int &operator() = default;\n"
               "int &operator!=() {",
               Alignment);
  verifyFormat("int f()         = default;\n"
               "int &operator() = default;\n"
               "int &operator=();",
               Alignment);
  verifyFormat("int f()         = delete;\n"
               "int &operator() = delete;\n"
               "int &operator=();",
               Alignment);
  verifyFormat("/* long long padding */ int f() = default;\n"
               "int &operator()                 = default;\n"
               "int &operator/**/ =();",
               Alignment);
  // https://llvm.org/PR33697
  FormatStyle AlignmentWithPenalty = getLLVMStyle();
  AlignmentWithPenalty.AlignConsecutiveAssignments.Enabled = true;
  AlignmentWithPenalty.PenaltyReturnTypeOnItsOwnLine = 5000;
  verifyFormat("class SSSSSSSSSSSSSSSSSSSSSSSSSSSS {\n"
               "  void f() = delete;\n"
               "  SSSSSSSSSSSSSSSSSSSSSSSSSSSS &operator=(\n"
               "      const SSSSSSSSSSSSSSSSSSSSSSSSSSSS &other) = delete;\n"
               "};",
               AlignmentWithPenalty);

  // Bug 25167
  /* Uncomment when fixed
    verifyFormat("#if A\n"
                 "#else\n"
                 "int aaaaaaaa = 12;\n"
                 "#endif\n"
                 "#if B\n"
                 "#else\n"
                 "int a = 12;\n"
                 "#endif",
                 Alignment);
    verifyFormat("enum foo {\n"
                 "#if A\n"
                 "#else\n"
                 "  aaaaaaaa = 12;\n"
                 "#endif\n"
                 "#if B\n"
                 "#else\n"
                 "  a = 12;\n"
                 "#endif\n"
                 "};",
                 Alignment);
  */

  verifyFormat("int a = 5;\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a       = 5;\n"
               "\n"
               "int oneTwoThree= 123;",
               Alignment);
  verifyFormat("int a   = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               "int a = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a   = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;\n"
               "int oneTwo      = 12;",
               "int a = 5;\n"
               "int one = 1;\n"
               "\n"
               "int oneTwoThree = 123;\n"
               "int oneTwo = 12;",
               Alignment);
  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_DontAlign;
  verifyFormat("#define A \\\n"
               "  int aaaa       = 12; \\\n"
               "  int b          = 23; \\\n"
               "  int ccc        = 234; \\\n"
               "  int dddddddddd = 2345;",
               Alignment);
  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  verifyFormat("#define A               \\\n"
               "  int aaaa       = 12;  \\\n"
               "  int b          = 23;  \\\n"
               "  int ccc        = 234; \\\n"
               "  int dddddddddd = 2345;",
               Alignment);
  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_Right;
  verifyFormat("#define A                                                      "
               "                \\\n"
               "  int aaaa       = 12;                                         "
               "                \\\n"
               "  int b          = 23;                                         "
               "                \\\n"
               "  int ccc        = 234;                                        "
               "                \\\n"
               "  int dddddddddd = 2345;",
               Alignment);
  verifyFormat("void SomeFunction(int parameter = 1, int i = 2, int j = 3, int "
               "k = 4, int l = 5,\n"
               "                  int m = 6) {\n"
               "  int j      = 10;\n"
               "  otherThing = 1;\n"
               "}",
               Alignment);
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int i   = 1;\n"
               "  int j   = 2;\n"
               "  int big = 10000;\n"
               "}",
               Alignment);
  verifyFormat("class C {\n"
               "public:\n"
               "  int i            = 1;\n"
               "  virtual void f() = 0;\n"
               "};",
               Alignment);
  verifyFormat("int i = 1;\n"
               "if (SomeType t = getSomething()) {\n"
               "}\n"
               "int j   = 2;\n"
               "int big = 10000;",
               Alignment);
  verifyFormat("int j = 7;\n"
               "for (int k = 0; k < N; ++k) {\n"
               "}\n"
               "int j   = 2;\n"
               "int big = 10000;\n"
               "}",
               Alignment);
  Alignment.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat("int i = 1;\n"
               "LooooooooooongType loooooooooooooooooooooongVariable\n"
               "    = someLooooooooooooooooongFunction();\n"
               "int j = 2;",
               Alignment);
  Alignment.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  verifyFormat("int i = 1;\n"
               "LooooooooooongType loooooooooooooooooooooongVariable =\n"
               "    someLooooooooooooooooongFunction();\n"
               "int j = 2;",
               Alignment);

  verifyFormat("auto lambda = []() {\n"
               "  auto i = 0;\n"
               "  return 0;\n"
               "};\n"
               "int i  = 0;\n"
               "auto v = type{\n"
               "    i = 1,   //\n"
               "    (i = 2), //\n"
               "    i = 3    //\n"
               "};",
               Alignment);

  verifyFormat(
      "int i      = 1;\n"
      "SomeType a = SomeFunction(looooooooooooooooooooooongParameterA,\n"
      "                          loooooooooooooooooooooongParameterB);\n"
      "int j      = 2;",
      Alignment);

  verifyFormat("template <typename T, typename T_0 = very_long_type_name_0,\n"
               "          typename B   = very_long_type_name_1,\n"
               "          typename T_2 = very_long_type_name_2>\n"
               "auto foo() {}",
               Alignment);
  verifyFormat("int a, b = 1;\n"
               "int c  = 2;\n"
               "int dd = 3;",
               Alignment);
  verifyFormat("int aa       = ((1 > 2) ? 3 : 4);\n"
               "float b[1][] = {{3.f}};",
               Alignment);
  verifyFormat("for (int i = 0; i < 1; i++)\n"
               "  int x = 1;",
               Alignment);
  verifyFormat("for (i = 0; i < 1; i++)\n"
               "  x = 1;\n"
               "y = 1;",
               Alignment);

  EXPECT_EQ(Alignment.ReflowComments, true);
  Alignment.ColumnLimit = 50;
  verifyFormat("int x   = 0;\n"
               "int yy  = 1; /// specificlennospace\n"
               "int zzz = 2;",
               "int x   = 0;\n"
               "int yy  = 1; ///specificlennospace\n"
               "int zzz = 2;",
               Alignment);

  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};\n"
               "auto b                     = [] {\n"
               "  f();\n"
               "  return;\n"
               "};",
               Alignment);
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};\n"
               "auto b                     = g([] {\n"
               "  f();\n"
               "  return;\n"
               "});",
               Alignment);
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};\n"
               "auto b                     = g(param, [] {\n"
               "  f();\n"
               "  return;\n"
               "});",
               Alignment);
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};\n"
               "auto b                     = [] {\n"
               "  if (condition) {\n"
               "    return;\n"
               "  }\n"
               "};",
               Alignment);

  verifyFormat("auto b = f(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "           ccc ? aaaaa : bbbbb,\n"
               "           dddddddddddddddddddddddddd);",
               Alignment);
  // FIXME: https://llvm.org/PR53497
  // verifyFormat("auto aaaaaaaaaaaa = f();\n"
  //              "auto b            = f(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
  //              "    ccc ? aaaaa : bbbbb,\n"
  //              "    dddddddddddddddddddddddddd);",
  //              Alignment);

  // Confirm proper handling of AlignConsecutiveAssignments with
  // BinPackArguments.
  // See https://llvm.org/PR55360
  Alignment = getLLVMStyleWithColumns(50);
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  Alignment.BinPackArguments = false;
  verifyFormat("int a_long_name = 1;\n"
               "auto b          = B({a_long_name, a_long_name},\n"
               "                    {a_longer_name_for_wrap,\n"
               "                     a_longer_name_for_wrap});",
               Alignment);
  verifyFormat("int a_long_name = 1;\n"
               "auto b          = B{{a_long_name, a_long_name},\n"
               "                    {a_longer_name_for_wrap,\n"
               "                     a_longer_name_for_wrap}};",
               Alignment);

  Alignment.ColumnLimit = 60;
  verifyFormat("using II = typename TI<T, std::tuple<Types...>>::I;\n"
               "using I  = std::conditional_t<II::value >= 0,\n"
               "                              std::ic<int, II::value + 1>,\n"
               "                              std::ic<int, -1>>;",
               Alignment);
}

TEST_F(FormatTest, AlignConsecutiveBitFields) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveBitFields.Enabled = true;
  verifyFormat("int const a     : 5;\n"
               "int oneTwoThree : 23;",
               Alignment);

  // Initializers are allowed starting with c++2a
  verifyFormat("int const a     : 5 = 1;\n"
               "int oneTwoThree : 23 = 0;",
               Alignment);

  Alignment.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("int const a           : 5;\n"
               "int       oneTwoThree : 23;",
               Alignment);

  verifyFormat("int const a           : 5;  // comment\n"
               "int       oneTwoThree : 23; // comment",
               Alignment);

  verifyFormat("int const a           : 5 = 1;\n"
               "int       oneTwoThree : 23 = 0;",
               Alignment);

  Alignment.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("int const a           : 5  = 1;\n"
               "int       oneTwoThree : 23 = 0;",
               Alignment);
  verifyFormat("int const a           : 5  = {1};\n"
               "int       oneTwoThree : 23 = 0;",
               Alignment);

  Alignment.BitFieldColonSpacing = FormatStyle::BFCS_None;
  verifyFormat("int const a          :5;\n"
               "int       oneTwoThree:23;",
               Alignment);

  Alignment.BitFieldColonSpacing = FormatStyle::BFCS_Before;
  verifyFormat("int const a           :5;\n"
               "int       oneTwoThree :23;",
               Alignment);

  Alignment.BitFieldColonSpacing = FormatStyle::BFCS_After;
  verifyFormat("int const a          : 5;\n"
               "int       oneTwoThree: 23;",
               Alignment);

  // Known limitations: ':' is only recognized as a bitfield colon when
  // followed by a number.
  /*
  verifyFormat("int oneTwoThree : SOME_CONSTANT;\n"
               "int a           : 5;",
               Alignment);
  */
}

TEST_F(FormatTest, AlignConsecutiveDeclarations) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveMacros.Enabled = true;
  Alignment.PointerAlignment = FormatStyle::PAS_Right;
  verifyFormat("float const a = 5;\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a = 5;\n"
               "float const oneTwoThree = 123;",
               Alignment);

  Alignment.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("float const a = 5;\n"
               "int         oneTwoThree = 123;",
               Alignment);
  verifyFormat("int         a = method();\n"
               "float const oneTwoThree = 133;",
               Alignment);
  verifyFormat("int i = 1, j = 10;\n"
               "something = 2000;",
               Alignment);
  verifyFormat("something = 2000;\n"
               "int i = 1, j = 10;",
               Alignment);
  verifyFormat("float      something = 2000;\n"
               "double     another = 911;\n"
               "int        i = 1, j = 10;\n"
               "const int *oneMore = 1;\n"
               "unsigned   i = 2;",
               Alignment);
  verifyFormat("float a = 5;\n"
               "int   one = 1;\n"
               "method();\n"
               "const double       oneTwoThree = 123;\n"
               "const unsigned int oneTwo = 12;",
               Alignment);
  verifyFormat("int      oneTwoThree{0}; // comment\n"
               "unsigned oneTwo;         // comment",
               Alignment);
  verifyFormat("unsigned int       *a;\n"
               "int                *b;\n"
               "unsigned int Const *c;\n"
               "unsigned int const *d;\n"
               "unsigned int Const &e;\n"
               "unsigned int const &f;",
               Alignment);
  verifyFormat("Const unsigned int *c;\n"
               "const unsigned int *d;\n"
               "Const unsigned int &e;\n"
               "const unsigned int &f;\n"
               "const unsigned      g;\n"
               "Const unsigned      h;",
               Alignment);
  verifyFormat("float const a = 5;\n"
               "\n"
               "int oneTwoThree = 123;",
               "float const   a = 5;\n"
               "\n"
               "int           oneTwoThree= 123;",
               Alignment);
  verifyFormat("float a = 5;\n"
               "int   one = 1;\n"
               "\n"
               "unsigned oneTwoThree = 123;",
               "float    a = 5;\n"
               "int      one = 1;\n"
               "\n"
               "unsigned oneTwoThree = 123;",
               Alignment);
  verifyFormat("float a = 5;\n"
               "int   one = 1;\n"
               "\n"
               "unsigned oneTwoThree = 123;\n"
               "int      oneTwo = 12;",
               "float    a = 5;\n"
               "int one = 1;\n"
               "\n"
               "unsigned oneTwoThree = 123;\n"
               "int oneTwo = 12;",
               Alignment);
  // Function prototype alignment
  verifyFormat("int    a();\n"
               "double b();",
               Alignment);
  verifyFormat("int    a(int x);\n"
               "double b();",
               Alignment);
  unsigned OldColumnLimit = Alignment.ColumnLimit;
  // We need to set ColumnLimit to zero, in order to stress nested alignments,
  // otherwise the function parameters will be re-flowed onto a single line.
  Alignment.ColumnLimit = 0;
  verifyFormat("int    a(int   x,\n"
               "         float y);\n"
               "double b(int    x,\n"
               "         double y);",
               "int a(int x,\n"
               " float y);\n"
               "double b(int x,\n"
               " double y);",
               Alignment);
  // This ensures that function parameters of function declarations are
  // correctly indented when their owning functions are indented.
  // The failure case here is for 'double y' to not be indented enough.
  verifyFormat("double a(int x);\n"
               "int    b(int    y,\n"
               "         double z);",
               "double a(int x);\n"
               "int b(int y,\n"
               " double z);",
               Alignment);
  // Set ColumnLimit low so that we induce wrapping immediately after
  // the function name and opening paren.
  Alignment.ColumnLimit = 13;
  verifyFormat("int function(\n"
               "    int  x,\n"
               "    bool y);",
               Alignment);
  Alignment.ColumnLimit = OldColumnLimit;
  // Ensure function pointers don't screw up recursive alignment
  verifyFormat("int    a(int x, void (*fp)(int y));\n"
               "double b();",
               Alignment);
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  // Ensure recursive alignment is broken by function braces, so that the
  // "a = 1" does not align with subsequent assignments inside the function
  // body.
  verifyFormat("int func(int a = 1) {\n"
               "  int b  = 2;\n"
               "  int cc = 3;\n"
               "}",
               Alignment);
  verifyFormat("float      something = 2000;\n"
               "double     another   = 911;\n"
               "int        i = 1, j = 10;\n"
               "const int *oneMore = 1;\n"
               "unsigned   i       = 2;",
               Alignment);
  verifyFormat("int      oneTwoThree = {0}; // comment\n"
               "unsigned oneTwo      = 0;   // comment",
               Alignment);
  // Make sure that scope is correctly tracked, in the absence of braces
  verifyFormat("for (int i = 0; i < n; i++)\n"
               "  j = i;\n"
               "double x = 1;",
               Alignment);
  verifyFormat("if (int i = 0)\n"
               "  j = i;\n"
               "double x = 1;",
               Alignment);
  // Ensure operator[] and operator() are comprehended
  verifyFormat("struct test {\n"
               "  long long int foo();\n"
               "  int           operator[](int a);\n"
               "  double        bar();\n"
               "};",
               Alignment);
  verifyFormat("struct test {\n"
               "  long long int foo();\n"
               "  int           operator()(int a);\n"
               "  double        bar();\n"
               "};",
               Alignment);
  // http://llvm.org/PR52914
  verifyFormat("char *a[]     = {\"a\", // comment\n"
               "                 \"bb\"};\n"
               "int   bbbbbbb = 0;",
               Alignment);

  // PAS_Right
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int const i   = 1;\n"
               "  int      *j   = 2;\n"
               "  int       big = 10000;\n"
               "\n"
               "  unsigned oneTwoThree = 123;\n"
               "  int      oneTwo      = 12;\n"
               "  method();\n"
               "  float k  = 2;\n"
               "  int   ll = 10000;\n"
               "}",
               "void SomeFunction(int parameter= 0) {\n"
               " int const  i= 1;\n"
               "  int *j=2;\n"
               " int big  =  10000;\n"
               "\n"
               "unsigned oneTwoThree  =123;\n"
               "int oneTwo = 12;\n"
               "  method();\n"
               "float k= 2;\n"
               "int ll=10000;\n"
               "}",
               Alignment);
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int const i   = 1;\n"
               "  int     **j   = 2, ***k;\n"
               "  int      &k   = i;\n"
               "  int     &&l   = i + j;\n"
               "  int       big = 10000;\n"
               "\n"
               "  unsigned oneTwoThree = 123;\n"
               "  int      oneTwo      = 12;\n"
               "  method();\n"
               "  float k  = 2;\n"
               "  int   ll = 10000;\n"
               "}",
               "void SomeFunction(int parameter= 0) {\n"
               " int const  i= 1;\n"
               "  int **j=2,***k;\n"
               "int &k=i;\n"
               "int &&l=i+j;\n"
               " int big  =  10000;\n"
               "\n"
               "unsigned oneTwoThree  =123;\n"
               "int oneTwo = 12;\n"
               "  method();\n"
               "float k= 2;\n"
               "int ll=10000;\n"
               "}",
               Alignment);
  // variables are aligned at their name, pointers are at the right most
  // position
  verifyFormat("int   *a;\n"
               "int  **b;\n"
               "int ***c;\n"
               "int    foobar;",
               Alignment);

  // PAS_Left
  FormatStyle AlignmentLeft = Alignment;
  AlignmentLeft.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int const i   = 1;\n"
               "  int*      j   = 2;\n"
               "  int       big = 10000;\n"
               "\n"
               "  unsigned oneTwoThree = 123;\n"
               "  int      oneTwo      = 12;\n"
               "  method();\n"
               "  float k  = 2;\n"
               "  int   ll = 10000;\n"
               "}",
               "void SomeFunction(int parameter= 0) {\n"
               " int const  i= 1;\n"
               "  int *j=2;\n"
               " int big  =  10000;\n"
               "\n"
               "unsigned oneTwoThree  =123;\n"
               "int oneTwo = 12;\n"
               "  method();\n"
               "float k= 2;\n"
               "int ll=10000;\n"
               "}",
               AlignmentLeft);
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int const i   = 1;\n"
               "  int**     j   = 2;\n"
               "  int&      k   = i;\n"
               "  int&&     l   = i + j;\n"
               "  int       big = 10000;\n"
               "\n"
               "  unsigned oneTwoThree = 123;\n"
               "  int      oneTwo      = 12;\n"
               "  method();\n"
               "  float k  = 2;\n"
               "  int   ll = 10000;\n"
               "}",
               "void SomeFunction(int parameter= 0) {\n"
               " int const  i= 1;\n"
               "  int **j=2;\n"
               "int &k=i;\n"
               "int &&l=i+j;\n"
               " int big  =  10000;\n"
               "\n"
               "unsigned oneTwoThree  =123;\n"
               "int oneTwo = 12;\n"
               "  method();\n"
               "float k= 2;\n"
               "int ll=10000;\n"
               "}",
               AlignmentLeft);
  // variables are aligned at their name, pointers are at the left most position
  verifyFormat("int*   a;\n"
               "int**  b;\n"
               "int*** c;\n"
               "int    foobar;",
               AlignmentLeft);

  // PAS_Middle
  FormatStyle AlignmentMiddle = Alignment;
  AlignmentMiddle.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int const i   = 1;\n"
               "  int *     j   = 2;\n"
               "  int       big = 10000;\n"
               "\n"
               "  unsigned oneTwoThree = 123;\n"
               "  int      oneTwo      = 12;\n"
               "  method();\n"
               "  float k  = 2;\n"
               "  int   ll = 10000;\n"
               "}",
               "void SomeFunction(int parameter= 0) {\n"
               " int const  i= 1;\n"
               "  int *j=2;\n"
               " int big  =  10000;\n"
               "\n"
               "unsigned oneTwoThree  =123;\n"
               "int oneTwo = 12;\n"
               "  method();\n"
               "float k= 2;\n"
               "int ll=10000;\n"
               "}",
               AlignmentMiddle);
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int const i   = 1;\n"
               "  int **    j   = 2, ***k;\n"
               "  int &     k   = i;\n"
               "  int &&    l   = i + j;\n"
               "  int       big = 10000;\n"
               "\n"
               "  unsigned oneTwoThree = 123;\n"
               "  int      oneTwo      = 12;\n"
               "  method();\n"
               "  float k  = 2;\n"
               "  int   ll = 10000;\n"
               "}",
               "void SomeFunction(int parameter= 0) {\n"
               " int const  i= 1;\n"
               "  int **j=2,***k;\n"
               "int &k=i;\n"
               "int &&l=i+j;\n"
               " int big  =  10000;\n"
               "\n"
               "unsigned oneTwoThree  =123;\n"
               "int oneTwo = 12;\n"
               "  method();\n"
               "float k= 2;\n"
               "int ll=10000;\n"
               "}",
               AlignmentMiddle);
  // variables are aligned at their name, pointers are in the middle
  verifyFormat("int *   a;\n"
               "int *   b;\n"
               "int *** c;\n"
               "int     foobar;",
               AlignmentMiddle);

  Alignment.AlignConsecutiveAssignments.Enabled = false;
  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_DontAlign;
  verifyFormat("#define A \\\n"
               "  int       aaaa = 12; \\\n"
               "  float     b = 23; \\\n"
               "  const int ccc = 234; \\\n"
               "  unsigned  dddddddddd = 2345;",
               Alignment);
  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  verifyFormat("#define A              \\\n"
               "  int       aaaa = 12; \\\n"
               "  float     b = 23;    \\\n"
               "  const int ccc = 234; \\\n"
               "  unsigned  dddddddddd = 2345;",
               Alignment);
  Alignment.AlignEscapedNewlines = FormatStyle::ENAS_Right;
  Alignment.ColumnLimit = 30;
  verifyFormat("#define A                    \\\n"
               "  int       aaaa = 12;       \\\n"
               "  float     b = 23;          \\\n"
               "  const int ccc = 234;       \\\n"
               "  int       dddddddddd = 2345;",
               Alignment);
  Alignment.ColumnLimit = 80;
  verifyFormat("void SomeFunction(int parameter = 1, int i = 2, int j = 3, int "
               "k = 4, int l = 5,\n"
               "                  int m = 6) {\n"
               "  const int j = 10;\n"
               "  otherThing = 1;\n"
               "}",
               Alignment);
  verifyFormat("void SomeFunction(int parameter = 0) {\n"
               "  int const i = 1;\n"
               "  int      *j = 2;\n"
               "  int       big = 10000;\n"
               "}",
               Alignment);
  verifyFormat("class C {\n"
               "public:\n"
               "  int          i = 1;\n"
               "  virtual void f() = 0;\n"
               "};",
               Alignment);
  verifyFormat("float i = 1;\n"
               "if (SomeType t = getSomething()) {\n"
               "}\n"
               "const unsigned j = 2;\n"
               "int            big = 10000;",
               Alignment);
  verifyFormat("float j = 7;\n"
               "for (int k = 0; k < N; ++k) {\n"
               "}\n"
               "unsigned j = 2;\n"
               "int      big = 10000;\n"
               "}",
               Alignment);
  Alignment.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat("float              i = 1;\n"
               "LooooooooooongType loooooooooooooooooooooongVariable\n"
               "    = someLooooooooooooooooongFunction();\n"
               "int j = 2;",
               Alignment);
  Alignment.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  verifyFormat("int                i = 1;\n"
               "LooooooooooongType loooooooooooooooooooooongVariable =\n"
               "    someLooooooooooooooooongFunction();\n"
               "int j = 2;",
               Alignment);

  Alignment.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("auto lambda = []() {\n"
               "  auto  ii = 0;\n"
               "  float j  = 0;\n"
               "  return 0;\n"
               "};\n"
               "int   i  = 0;\n"
               "float i2 = 0;\n"
               "auto  v  = type{\n"
               "    i = 1,   //\n"
               "    (i = 2), //\n"
               "    i = 3    //\n"
               "};",
               Alignment);
  Alignment.AlignConsecutiveAssignments.Enabled = false;

  verifyFormat(
      "int      i = 1;\n"
      "SomeType a = SomeFunction(looooooooooooooooooooooongParameterA,\n"
      "                          loooooooooooooooooooooongParameterB);\n"
      "int      j = 2;",
      Alignment);

  // Test interactions with ColumnLimit and AlignConsecutiveAssignments:
  // We expect declarations and assignments to align, as long as it doesn't
  // exceed the column limit, starting a new alignment sequence whenever it
  // happens.
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  Alignment.ColumnLimit = 30;
  verifyFormat("float    ii              = 1;\n"
               "unsigned j               = 2;\n"
               "int someVerylongVariable = 1;\n"
               "AnotherLongType  ll = 123456;\n"
               "VeryVeryLongType k  = 2;\n"
               "int              myvar = 1;",
               Alignment);
  Alignment.ColumnLimit = 80;
  Alignment.AlignConsecutiveAssignments.Enabled = false;

  verifyFormat(
      "template <typename LongTemplate, typename VeryLongTemplateTypeName,\n"
      "          typename LongType, typename B>\n"
      "auto foo() {}",
      Alignment);
  verifyFormat("float a, b = 1;\n"
               "int   c = 2;\n"
               "int   dd = 3;",
               Alignment);
  verifyFormat("int   aa = ((1 > 2) ? 3 : 4);\n"
               "float b[1][] = {{3.f}};",
               Alignment);
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("float a, b = 1;\n"
               "int   c  = 2;\n"
               "int   dd = 3;",
               Alignment);
  verifyFormat("int   aa     = ((1 > 2) ? 3 : 4);\n"
               "float b[1][] = {{3.f}};",
               Alignment);
  Alignment.AlignConsecutiveAssignments.Enabled = false;

  Alignment.ColumnLimit = 30;
  Alignment.BinPackParameters = false;
  verifyFormat("void foo(float     a,\n"
               "         float     b,\n"
               "         int       c,\n"
               "         uint32_t *d) {\n"
               "  int   *e = 0;\n"
               "  float  f = 0;\n"
               "  double g = 0;\n"
               "}\n"
               "void bar(ino_t     a,\n"
               "         int       b,\n"
               "         uint32_t *c,\n"
               "         bool      d) {}",
               Alignment);
  Alignment.BinPackParameters = true;
  Alignment.ColumnLimit = 80;

  // Bug 33507
  Alignment.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat(
      "auto found = range::find_if(vsProducts, [&](auto * aProduct) {\n"
      "  static const Version verVs2017;\n"
      "  return true;\n"
      "});",
      Alignment);
  Alignment.PointerAlignment = FormatStyle::PAS_Right;

  // See llvm.org/PR35641
  Alignment.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("int func() { //\n"
               "  int      b;\n"
               "  unsigned c;\n"
               "}",
               Alignment);

  // See PR37175
  FormatStyle Style = getMozillaStyle();
  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("DECOR1 /**/ int8_t /**/ DECOR2 /**/\n"
               "foo(int a);",
               "DECOR1 /**/ int8_t /**/ DECOR2 /**/ foo (int a);", Style);

  Alignment.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("unsigned int*       a;\n"
               "int*                b;\n"
               "unsigned int Const* c;\n"
               "unsigned int const* d;\n"
               "unsigned int Const& e;\n"
               "unsigned int const& f;",
               Alignment);
  verifyFormat("Const unsigned int* c;\n"
               "const unsigned int* d;\n"
               "Const unsigned int& e;\n"
               "const unsigned int& f;\n"
               "const unsigned      g;\n"
               "Const unsigned      h;",
               Alignment);

  Alignment.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("unsigned int *       a;\n"
               "int *                b;\n"
               "unsigned int Const * c;\n"
               "unsigned int const * d;\n"
               "unsigned int Const & e;\n"
               "unsigned int const & f;",
               Alignment);
  verifyFormat("Const unsigned int * c;\n"
               "const unsigned int * d;\n"
               "Const unsigned int & e;\n"
               "const unsigned int & f;\n"
               "const unsigned       g;\n"
               "Const unsigned       h;",
               Alignment);

  // See PR46529
  FormatStyle BracedAlign = getLLVMStyle();
  BracedAlign.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("const auto result{[]() {\n"
               "  const auto something = 1;\n"
               "  return 2;\n"
               "}};",
               BracedAlign);
  verifyFormat("int foo{[]() {\n"
               "  int bar{0};\n"
               "  return 0;\n"
               "}()};",
               BracedAlign);
  BracedAlign.Cpp11BracedListStyle = false;
  verifyFormat("const auto result{ []() {\n"
               "  const auto something = 1;\n"
               "  return 2;\n"
               "} };",
               BracedAlign);
  verifyFormat("int foo{ []() {\n"
               "  int bar{ 0 };\n"
               "  return 0;\n"
               "}() };",
               BracedAlign);
}

TEST_F(FormatTest, AlignConsecutiveShortCaseStatements) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AllowShortCaseLabelsOnASingleLine = true;
  Alignment.AlignConsecutiveShortCaseStatements.Enabled = true;

  verifyFormat("switch (level) {\n"
               "case log::info:    return \"info\";\n"
               "case log::warning: return \"warning\";\n"
               "default:           return \"default\";\n"
               "}",
               Alignment);

  verifyFormat("switch (level) {\n"
               "case log::info:    return \"info\";\n"
               "case log::warning: return \"warning\";\n"
               "}",
               "switch (level) {\n"
               "case log::info: return \"info\";\n"
               "case log::warning:\n"
               "  return \"warning\";\n"
               "}",
               Alignment);

  // Empty case statements push out the alignment, but non-short case labels
  // don't.
  verifyFormat("switch (level) {\n"
               "case log::info:     return \"info\";\n"
               "case log::critical:\n"
               "case log::warning:\n"
               "case log::severe:   return \"severe\";\n"
               "case log::extra_severe:\n"
               "  // comment\n"
               "  return \"extra_severe\";\n"
               "}",
               Alignment);

  // Verify comments and empty lines break the alignment.
  verifyNoChange("switch (level) {\n"
                 "case log::info:    return \"info\";\n"
                 "case log::warning: return \"warning\";\n"
                 "// comment\n"
                 "case log::critical: return \"critical\";\n"
                 "default:            return \"default\";\n"
                 "\n"
                 "case log::severe: return \"severe\";\n"
                 "}",
                 Alignment);

  // Empty case statements don't break the alignment, and potentially push it
  // out.
  verifyFormat("switch (level) {\n"
               "case log::info:     return \"info\";\n"
               "case log::warning:\n"
               "case log::critical:\n"
               "default:            return \"default\";\n"
               "}",
               Alignment);

  // Implicit fallthrough cases can be aligned with either a comment or
  // [[fallthrough]]
  verifyFormat("switch (level) {\n"
               "case log::info:     return \"info\";\n"
               "case log::warning:  // fallthrough\n"
               "case log::error:    return \"error\";\n"
               "case log::critical: /*fallthrough*/\n"
               "case log::severe:   return \"severe\";\n"
               "case log::diag:     [[fallthrough]];\n"
               "default:            return \"default\";\n"
               "}",
               Alignment);

  // Verify trailing comment that needs a reflow also gets aligned properly.
  verifyFormat("switch (level) {\n"
               "case log::info:    return \"info\";\n"
               "case log::warning: // fallthrough\n"
               "case log::error:   return \"error\";\n"
               "}",
               "switch (level) {\n"
               "case log::info:    return \"info\";\n"
               "case log::warning: //fallthrough\n"
               "case log::error:   return \"error\";\n"
               "}",
               Alignment);

  // Verify adjacent non-short case statements don't change the alignment, and
  // properly break the set of consecutive statements.
  verifyFormat("switch (level) {\n"
               "case log::critical:\n"
               "  // comment\n"
               "  return \"critical\";\n"
               "case log::info:    return \"info\";\n"
               "case log::warning: return \"warning\";\n"
               "default:\n"
               "  // comment\n"
               "  return \"\";\n"
               "case log::error:  return \"error\";\n"
               "case log::severe: return \"severe\";\n"
               "case log::extra_critical:\n"
               "  // comment\n"
               "  return \"extra critical\";\n"
               "}",
               Alignment);

  Alignment.SpaceBeforeCaseColon = true;
  verifyFormat("switch (level) {\n"
               "case log::info :    return \"info\";\n"
               "case log::warning : return \"warning\";\n"
               "default :           return \"default\";\n"
               "}",
               Alignment);
  Alignment.SpaceBeforeCaseColon = false;

  // Make sure we don't incorrectly align correctly across nested switch cases.
  verifyFormat("switch (level) {\n"
               "case log::info:    return \"info\";\n"
               "case log::warning: return \"warning\";\n"
               "case log::other:\n"
               "  switch (sublevel) {\n"
               "  case log::info:    return \"info\";\n"
               "  case log::warning: return \"warning\";\n"
               "  }\n"
               "  break;\n"
               "case log::error: return \"error\";\n"
               "default:         return \"default\";\n"
               "}",
               "switch (level) {\n"
               "case log::info:    return \"info\";\n"
               "case log::warning: return \"warning\";\n"
               "case log::other: switch (sublevel) {\n"
               "  case log::info:    return \"info\";\n"
               "  case log::warning: return \"warning\";\n"
               "}\n"
               "break;\n"
               "case log::error: return \"error\";\n"
               "default:         return \"default\";\n"
               "}",
               Alignment);

  Alignment.AlignConsecutiveShortCaseStatements.AcrossEmptyLines = true;

  verifyFormat("switch (level) {\n"
               "case log::info:    return \"info\";\n"
               "\n"
               "case log::warning: return \"warning\";\n"
               "}",
               "switch (level) {\n"
               "case log::info: return \"info\";\n"
               "\n"
               "case log::warning: return \"warning\";\n"
               "}",
               Alignment);

  Alignment.AlignConsecutiveShortCaseStatements.AcrossComments = true;

  verifyNoChange("switch (level) {\n"
                 "case log::info:    return \"info\";\n"
                 "\n"
                 "/* block comment */\n"
                 "\n"
                 "// line comment\n"
                 "case log::warning: return \"warning\";\n"
                 "}",
                 Alignment);

  Alignment.AlignConsecutiveShortCaseStatements.AcrossEmptyLines = false;

  verifyFormat("switch (level) {\n"
               "case log::info:    return \"info\";\n"
               "//\n"
               "case log::warning: return \"warning\";\n"
               "}",
               Alignment);

  Alignment.AlignConsecutiveShortCaseStatements.AlignCaseColons = true;

  verifyFormat("switch (level) {\n"
               "case log::info   : return \"info\";\n"
               "case log::warning: return \"warning\";\n"
               "default          : return \"default\";\n"
               "}",
               Alignment);

  // With AlignCaseColons, empty case statements don't break alignment of
  // consecutive case statements (and are aligned).
  verifyFormat("switch (level) {\n"
               "case log::info    : return \"info\";\n"
               "case log::warning :\n"
               "case log::critical:\n"
               "default           : return \"default\";\n"
               "}",
               Alignment);

  // Final non-short case labels shouldn't have their colon aligned
  verifyFormat("switch (level) {\n"
               "case log::info    : return \"info\";\n"
               "case log::warning :\n"
               "case log::critical:\n"
               "case log::severe  : return \"severe\";\n"
               "default:\n"
               "  // comment\n"
               "  return \"default\";\n"
               "}",
               Alignment);

  // Verify adjacent non-short case statements break the set of consecutive
  // alignments and aren't aligned with adjacent non-short case statements if
  // AlignCaseColons is set.
  verifyFormat("switch (level) {\n"
               "case log::critical:\n"
               "  // comment\n"
               "  return \"critical\";\n"
               "case log::info   : return \"info\";\n"
               "case log::warning: return \"warning\";\n"
               "default:\n"
               "  // comment\n"
               "  return \"\";\n"
               "case log::error : return \"error\";\n"
               "case log::severe: return \"severe\";\n"
               "case log::extra_critical:\n"
               "  // comment\n"
               "  return \"extra critical\";\n"
               "}",
               Alignment);

  Alignment.SpaceBeforeCaseColon = true;
  verifyFormat("switch (level) {\n"
               "case log::info    : return \"info\";\n"
               "case log::warning : return \"warning\";\n"
               "case log::error   :\n"
               "default           : return \"default\";\n"
               "}",
               Alignment);
}

TEST_F(FormatTest, AlignWithLineBreaks) {
  auto Style = getLLVMStyleWithColumns(120);

  EXPECT_EQ(Style.AlignConsecutiveAssignments,
            FormatStyle::AlignConsecutiveStyle(
                {/*Enabled=*/false, /*AcrossEmptyLines=*/false,
                 /*AcrossComments=*/false, /*AlignCompound=*/false,
                 /*PadOperators=*/true}));
  EXPECT_EQ(Style.AlignConsecutiveDeclarations,
            FormatStyle::AlignConsecutiveStyle({}));
  verifyFormat("void foo() {\n"
               "  int myVar = 5;\n"
               "  double x = 3.14;\n"
               "  auto str = \"Hello \"\n"
               "             \"World\";\n"
               "  auto s = \"Hello \"\n"
               "           \"Again\";\n"
               "}",
               Style);

  // clang-format off
  verifyFormat("void foo() {\n"
               "  const int capacityBefore = Entries.capacity();\n"
               "  const auto newEntry = Entries.emplaceHint(std::piecewise_construct, std::forward_as_tuple(uniqueId),\n"
               "                                            std::forward_as_tuple(id, uniqueId, name, threadCreation));\n"
               "  const X newEntry2 = Entries.emplaceHint(std::piecewise_construct, std::forward_as_tuple(uniqueId),\n"
               "                                          std::forward_as_tuple(id, uniqueId, name, threadCreation));\n"
               "}",
               Style);
  // clang-format on

  Style.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("void foo() {\n"
               "  int myVar = 5;\n"
               "  double x  = 3.14;\n"
               "  auto str  = \"Hello \"\n"
               "              \"World\";\n"
               "  auto s    = \"Hello \"\n"
               "              \"Again\";\n"
               "}",
               Style);

  // clang-format off
  verifyFormat("void foo() {\n"
               "  const int capacityBefore = Entries.capacity();\n"
               "  const auto newEntry      = Entries.emplaceHint(std::piecewise_construct, std::forward_as_tuple(uniqueId),\n"
               "                                                 std::forward_as_tuple(id, uniqueId, name, threadCreation));\n"
               "  const X newEntry2        = Entries.emplaceHint(std::piecewise_construct, std::forward_as_tuple(uniqueId),\n"
               "                                                 std::forward_as_tuple(id, uniqueId, name, threadCreation));\n"
               "}",
               Style);
  // clang-format on

  Style.AlignConsecutiveAssignments.Enabled = false;
  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("void foo() {\n"
               "  int    myVar = 5;\n"
               "  double x = 3.14;\n"
               "  auto   str = \"Hello \"\n"
               "               \"World\";\n"
               "  auto   s = \"Hello \"\n"
               "             \"Again\";\n"
               "}",
               Style);

  // clang-format off
  verifyFormat("void foo() {\n"
               "  const int  capacityBefore = Entries.capacity();\n"
               "  const auto newEntry = Entries.emplaceHint(std::piecewise_construct, std::forward_as_tuple(uniqueId),\n"
               "                                            std::forward_as_tuple(id, uniqueId, name, threadCreation));\n"
               "  const X    newEntry2 = Entries.emplaceHint(std::piecewise_construct, std::forward_as_tuple(uniqueId),\n"
               "                                             std::forward_as_tuple(id, uniqueId, name, threadCreation));\n"
               "}",
               Style);
  // clang-format on

  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.AlignConsecutiveDeclarations.Enabled = true;

  verifyFormat("void foo() {\n"
               "  int    myVar = 5;\n"
               "  double x     = 3.14;\n"
               "  auto   str   = \"Hello \"\n"
               "                 \"World\";\n"
               "  auto   s     = \"Hello \"\n"
               "                 \"Again\";\n"
               "}",
               Style);

  // clang-format off
  verifyFormat("void foo() {\n"
               "  const int  capacityBefore = Entries.capacity();\n"
               "  const auto newEntry       = Entries.emplaceHint(std::piecewise_construct, std::forward_as_tuple(uniqueId),\n"
               "                                                  std::forward_as_tuple(id, uniqueId, name, threadCreation));\n"
               "  const X    newEntry2      = Entries.emplaceHint(std::piecewise_construct, std::forward_as_tuple(uniqueId),\n"
               "                                                  std::forward_as_tuple(id, uniqueId, name, threadCreation));\n"
               "}",
               Style);
  // clang-format on

  Style = getLLVMStyleWithColumns(20);
  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.IndentWidth = 4;

  verifyFormat("void foo() {\n"
               "    int i1 = 1;\n"
               "    int j  = 0;\n"
               "    int k  = bar(\n"
               "        argument1,\n"
               "        argument2);\n"
               "}",
               Style);

  verifyFormat("unsigned i = 0;\n"
               "int a[]    = {\n"
               "    1234567890,\n"
               "    -1234567890};",
               Style);

  Style.ColumnLimit = 120;

  // clang-format off
  verifyFormat("void SomeFunc() {\n"
               "    newWatcher.maxAgeUsec = ToLegacyTimestamp(GetMaxAge(FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec),\n"
               "                                                        seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));\n"
               "    newWatcher.maxAge     = ToLegacyTimestamp(GetMaxAge(FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec),\n"
               "                                                        seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));\n"
               "    newWatcher.max        = ToLegacyTimestamp(GetMaxAge(FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec),\n"
               "                                                        seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));\n"
               "}",
               Style);
  // clang-format on

  Style.BinPackArguments = false;

  // clang-format off
  verifyFormat("void SomeFunc() {\n"
               "    newWatcher.maxAgeUsec = ToLegacyTimestamp(GetMaxAge(\n"
               "        FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec), seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));\n"
               "    newWatcher.maxAge     = ToLegacyTimestamp(GetMaxAge(\n"
               "        FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec), seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));\n"
               "    newWatcher.max        = ToLegacyTimestamp(GetMaxAge(\n"
               "        FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec), seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));\n"
               "}",
               Style);
  // clang-format on
}

TEST_F(FormatTest, AlignWithInitializerPeriods) {
  auto Style = getLLVMStyleWithColumns(60);

  verifyFormat("void foo1(void) {\n"
               "  BYTE p[1] = 1;\n"
               "  A B = {.one_foooooooooooooooo = 2,\n"
               "         .two_fooooooooooooo = 3,\n"
               "         .three_fooooooooooooo = 4};\n"
               "  BYTE payload = 2;\n"
               "}",
               Style);

  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.AlignConsecutiveDeclarations.Enabled = false;
  verifyFormat("void foo2(void) {\n"
               "  BYTE p[1]    = 1;\n"
               "  A B          = {.one_foooooooooooooooo = 2,\n"
               "                  .two_fooooooooooooo    = 3,\n"
               "                  .three_fooooooooooooo  = 4};\n"
               "  BYTE payload = 2;\n"
               "}",
               Style);

  Style.AlignConsecutiveAssignments.Enabled = false;
  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("void foo3(void) {\n"
               "  BYTE p[1] = 1;\n"
               "  A    B = {.one_foooooooooooooooo = 2,\n"
               "            .two_fooooooooooooo = 3,\n"
               "            .three_fooooooooooooo = 4};\n"
               "  BYTE payload = 2;\n"
               "}",
               Style);

  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("void foo4(void) {\n"
               "  BYTE p[1]    = 1;\n"
               "  A    B       = {.one_foooooooooooooooo = 2,\n"
               "                  .two_fooooooooooooo    = 3,\n"
               "                  .three_fooooooooooooo  = 4};\n"
               "  BYTE payload = 2;\n"
               "}",
               Style);
}

TEST_F(FormatTest, LinuxBraceBreaking) {
  FormatStyle LinuxBraceStyle = getLLVMStyle();
  LinuxBraceStyle.BreakBeforeBraces = FormatStyle::BS_Linux;
  verifyFormat("namespace a\n"
               "{\n"
               "class A\n"
               "{\n"
               "  void f()\n"
               "  {\n"
               "    if (true) {\n"
               "      a();\n"
               "      b();\n"
               "    } else {\n"
               "      a();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "struct B {\n"
               "  int x;\n"
               "};\n"
               "} // namespace a",
               LinuxBraceStyle);
  verifyFormat("enum X {\n"
               "  Y = 0,\n"
               "}",
               LinuxBraceStyle);
  verifyFormat("struct S {\n"
               "  int Type;\n"
               "  union {\n"
               "    int x;\n"
               "    double y;\n"
               "  } Value;\n"
               "  class C\n"
               "  {\n"
               "    MyFavoriteType Value;\n"
               "  } Class;\n"
               "}",
               LinuxBraceStyle);
}

TEST_F(FormatTest, MozillaBraceBreaking) {
  FormatStyle MozillaBraceStyle = getLLVMStyle();
  MozillaBraceStyle.BreakBeforeBraces = FormatStyle::BS_Mozilla;
  MozillaBraceStyle.FixNamespaceComments = false;
  verifyFormat("namespace a {\n"
               "class A\n"
               "{\n"
               "  void f()\n"
               "  {\n"
               "    if (true) {\n"
               "      a();\n"
               "      b();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "enum E\n"
               "{\n"
               "  A,\n"
               "  // foo\n"
               "  B,\n"
               "  C\n"
               "};\n"
               "struct B\n"
               "{\n"
               "  int x;\n"
               "};\n"
               "}",
               MozillaBraceStyle);
  verifyFormat("struct S\n"
               "{\n"
               "  int Type;\n"
               "  union\n"
               "  {\n"
               "    int x;\n"
               "    double y;\n"
               "  } Value;\n"
               "  class C\n"
               "  {\n"
               "    MyFavoriteType Value;\n"
               "  } Class;\n"
               "}",
               MozillaBraceStyle);
}

TEST_F(FormatTest, StroustrupBraceBreaking) {
  FormatStyle StroustrupBraceStyle = getLLVMStyle();
  StroustrupBraceStyle.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  verifyFormat("namespace a {\n"
               "class A {\n"
               "  void f()\n"
               "  {\n"
               "    if (true) {\n"
               "      a();\n"
               "      b();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "struct B {\n"
               "  int x;\n"
               "};\n"
               "} // namespace a",
               StroustrupBraceStyle);

  verifyFormat("void foo()\n"
               "{\n"
               "  if (a) {\n"
               "    a();\n"
               "  }\n"
               "  else {\n"
               "    b();\n"
               "  }\n"
               "}",
               StroustrupBraceStyle);

  verifyFormat("#ifdef _DEBUG\n"
               "int foo(int i = 0)\n"
               "#else\n"
               "int foo(int i = 5)\n"
               "#endif\n"
               "{\n"
               "  return i;\n"
               "}",
               StroustrupBraceStyle);

  verifyFormat("void foo() {}\n"
               "void bar()\n"
               "#ifdef _DEBUG\n"
               "{\n"
               "  foo();\n"
               "}\n"
               "#else\n"
               "{\n"
               "}\n"
               "#endif",
               StroustrupBraceStyle);

  verifyFormat("void foobar() { int i = 5; }\n"
               "#ifdef _DEBUG\n"
               "void bar() {}\n"
               "#else\n"
               "void bar() { foobar(); }\n"
               "#endif",
               StroustrupBraceStyle);
}

TEST_F(FormatTest, AllmanBraceBreaking) {
  FormatStyle AllmanBraceStyle = getLLVMStyle();
  AllmanBraceStyle.BreakBeforeBraces = FormatStyle::BS_Allman;

  verifyFormat("namespace a\n"
               "{\n"
               "void f();\n"
               "void g();\n"
               "} // namespace a",
               "namespace a\n"
               "{\n"
               "void f();\n"
               "void g();\n"
               "}",
               AllmanBraceStyle);

  verifyFormat("namespace a\n"
               "{\n"
               "class A\n"
               "{\n"
               "  void f()\n"
               "  {\n"
               "    if (true)\n"
               "    {\n"
               "      a();\n"
               "      b();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "struct B\n"
               "{\n"
               "  int x;\n"
               "};\n"
               "union C\n"
               "{\n"
               "};\n"
               "} // namespace a",
               AllmanBraceStyle);

  verifyFormat("void f()\n"
               "{\n"
               "  if (true)\n"
               "  {\n"
               "    a();\n"
               "  }\n"
               "  else if (false)\n"
               "  {\n"
               "    b();\n"
               "  }\n"
               "  else\n"
               "  {\n"
               "    c();\n"
               "  }\n"
               "}",
               AllmanBraceStyle);

  verifyFormat("void f()\n"
               "{\n"
               "  for (int i = 0; i < 10; ++i)\n"
               "  {\n"
               "    a();\n"
               "  }\n"
               "  while (false)\n"
               "  {\n"
               "    b();\n"
               "  }\n"
               "  do\n"
               "  {\n"
               "    c();\n"
               "  } while (false)\n"
               "}",
               AllmanBraceStyle);

  verifyFormat("void f(int a)\n"
               "{\n"
               "  switch (a)\n"
               "  {\n"
               "  case 0:\n"
               "    break;\n"
               "  case 1:\n"
               "  {\n"
               "    break;\n"
               "  }\n"
               "  case 2:\n"
               "  {\n"
               "  }\n"
               "  break;\n"
               "  default:\n"
               "    break;\n"
               "  }\n"
               "}",
               AllmanBraceStyle);

  verifyFormat("enum X\n"
               "{\n"
               "  Y = 0,\n"
               "}",
               AllmanBraceStyle);
  verifyFormat("enum X\n"
               "{\n"
               "  Y = 0\n"
               "}",
               AllmanBraceStyle);

  verifyFormat("@interface BSApplicationController ()\n"
               "{\n"
               "@private\n"
               "  id _extraIvar;\n"
               "}\n"
               "@end",
               AllmanBraceStyle);

  verifyFormat("#ifdef _DEBUG\n"
               "int foo(int i = 0)\n"
               "#else\n"
               "int foo(int i = 5)\n"
               "#endif\n"
               "{\n"
               "  return i;\n"
               "}",
               AllmanBraceStyle);

  verifyFormat("void foo() {}\n"
               "void bar()\n"
               "#ifdef _DEBUG\n"
               "{\n"
               "  foo();\n"
               "}\n"
               "#else\n"
               "{\n"
               "}\n"
               "#endif",
               AllmanBraceStyle);

  verifyFormat("void foobar() { int i = 5; }\n"
               "#ifdef _DEBUG\n"
               "void bar() {}\n"
               "#else\n"
               "void bar() { foobar(); }\n"
               "#endif",
               AllmanBraceStyle);

  EXPECT_EQ(AllmanBraceStyle.AllowShortLambdasOnASingleLine,
            FormatStyle::SLS_All);

  verifyFormat("[](int i) { return i + 2; };\n"
               "[](int i, int j)\n"
               "{\n"
               "  auto x = i + j;\n"
               "  auto y = i * j;\n"
               "  return x ^ y;\n"
               "};\n"
               "void foo()\n"
               "{\n"
               "  auto shortLambda = [](int i) { return i + 2; };\n"
               "  auto longLambda = [](int i, int j)\n"
               "  {\n"
               "    auto x = i + j;\n"
               "    auto y = i * j;\n"
               "    return x ^ y;\n"
               "  };\n"
               "}",
               AllmanBraceStyle);

  AllmanBraceStyle.AllowShortLambdasOnASingleLine = FormatStyle::SLS_None;

  verifyFormat("[](int i)\n"
               "{\n"
               "  return i + 2;\n"
               "};\n"
               "[](int i, int j)\n"
               "{\n"
               "  auto x = i + j;\n"
               "  auto y = i * j;\n"
               "  return x ^ y;\n"
               "};\n"
               "void foo()\n"
               "{\n"
               "  auto shortLambda = [](int i)\n"
               "  {\n"
               "    return i + 2;\n"
               "  };\n"
               "  auto longLambda = [](int i, int j)\n"
               "  {\n"
               "    auto x = i + j;\n"
               "    auto y = i * j;\n"
               "    return x ^ y;\n"
               "  };\n"
               "}",
               AllmanBraceStyle);

  // Reset
  AllmanBraceStyle.AllowShortLambdasOnASingleLine = FormatStyle::SLS_All;

  // This shouldn't affect ObjC blocks..
  verifyFormat("[self doSomeThingWithACompletionHandler:^{\n"
               "  // ...\n"
               "  int i;\n"
               "}];",
               AllmanBraceStyle);
  verifyFormat("void (^block)(void) = ^{\n"
               "  // ...\n"
               "  int i;\n"
               "};",
               AllmanBraceStyle);
  // .. or dict literals.
  verifyFormat("void f()\n"
               "{\n"
               "  // ...\n"
               "  [object someMethod:@{@\"a\" : @\"b\"}];\n"
               "}",
               AllmanBraceStyle);
  verifyFormat("void f()\n"
               "{\n"
               "  // ...\n"
               "  [object someMethod:@{a : @\"b\"}];\n"
               "}",
               AllmanBraceStyle);
  verifyFormat("int f()\n"
               "{ // comment\n"
               "  return 42;\n"
               "}",
               AllmanBraceStyle);

  AllmanBraceStyle.ColumnLimit = 19;
  verifyFormat("void f() { int i; }", AllmanBraceStyle);
  AllmanBraceStyle.ColumnLimit = 18;
  verifyFormat("void f()\n"
               "{\n"
               "  int i;\n"
               "}",
               AllmanBraceStyle);
  AllmanBraceStyle.ColumnLimit = 80;

  FormatStyle BreakBeforeBraceShortIfs = AllmanBraceStyle;
  BreakBeforeBraceShortIfs.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_WithoutElse;
  BreakBeforeBraceShortIfs.AllowShortLoopsOnASingleLine = true;
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  if (b)\n"
               "  {\n"
               "    return;\n"
               "  }\n"
               "}",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  if constexpr (b)\n"
               "  {\n"
               "    return;\n"
               "  }\n"
               "}",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  if CONSTEXPR (b)\n"
               "  {\n"
               "    return;\n"
               "  }\n"
               "}",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  if (b) return;\n"
               "}",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  if constexpr (b) return;\n"
               "}",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  if CONSTEXPR (b) return;\n"
               "}",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  while (b)\n"
               "  {\n"
               "    return;\n"
               "  }\n"
               "}",
               BreakBeforeBraceShortIfs);
}

TEST_F(FormatTest, WhitesmithsBraceBreaking) {
  FormatStyle WhitesmithsBraceStyle = getLLVMStyleWithColumns(0);
  WhitesmithsBraceStyle.BreakBeforeBraces = FormatStyle::BS_Whitesmiths;

  // Make a few changes to the style for testing purposes
  WhitesmithsBraceStyle.AllowShortFunctionsOnASingleLine =
      FormatStyle::SFS_Empty;
  WhitesmithsBraceStyle.AllowShortLambdasOnASingleLine = FormatStyle::SLS_None;

  // FIXME: this test case can't decide whether there should be a blank line
  // after the ~D() line or not. It adds one if one doesn't exist in the test
  // and it removes the line if one exists.
  /*
  verifyFormat("class A;\n"
               "namespace B\n"
               "  {\n"
               "class C;\n"
               "// Comment\n"
               "class D\n"
               "  {\n"
               "public:\n"
               "  D();\n"
               "  ~D() {}\n"
               "private:\n"
               "  enum E\n"
               "    {\n"
               "    F\n"
               "    }\n"
               "  };\n"
               "  } // namespace B",
               WhitesmithsBraceStyle);
  */

  WhitesmithsBraceStyle.NamespaceIndentation = FormatStyle::NI_None;
  verifyFormat("namespace a\n"
               "  {\n"
               "class A\n"
               "  {\n"
               "  void f()\n"
               "    {\n"
               "    if (true)\n"
               "      {\n"
               "      a();\n"
               "      b();\n"
               "      }\n"
               "    }\n"
               "  void g()\n"
               "    {\n"
               "    return;\n"
               "    }\n"
               "  };\n"
               "struct B\n"
               "  {\n"
               "  int x;\n"
               "  };\n"
               "  } // namespace a",
               WhitesmithsBraceStyle);

  verifyFormat("namespace a\n"
               "  {\n"
               "namespace b\n"
               "  {\n"
               "class A\n"
               "  {\n"
               "  void f()\n"
               "    {\n"
               "    if (true)\n"
               "      {\n"
               "      a();\n"
               "      b();\n"
               "      }\n"
               "    }\n"
               "  void g()\n"
               "    {\n"
               "    return;\n"
               "    }\n"
               "  };\n"
               "struct B\n"
               "  {\n"
               "  int x;\n"
               "  };\n"
               "  } // namespace b\n"
               "  } // namespace a",
               WhitesmithsBraceStyle);

  WhitesmithsBraceStyle.NamespaceIndentation = FormatStyle::NI_Inner;
  verifyFormat("namespace a\n"
               "  {\n"
               "namespace b\n"
               "  {\n"
               "  class A\n"
               "    {\n"
               "    void f()\n"
               "      {\n"
               "      if (true)\n"
               "        {\n"
               "        a();\n"
               "        b();\n"
               "        }\n"
               "      }\n"
               "    void g()\n"
               "      {\n"
               "      return;\n"
               "      }\n"
               "    };\n"
               "  struct B\n"
               "    {\n"
               "    int x;\n"
               "    };\n"
               "  } // namespace b\n"
               "  } // namespace a",
               WhitesmithsBraceStyle);

  WhitesmithsBraceStyle.NamespaceIndentation = FormatStyle::NI_All;
  verifyFormat("namespace a\n"
               "  {\n"
               "  namespace b\n"
               "    {\n"
               "    class A\n"
               "      {\n"
               "      void f()\n"
               "        {\n"
               "        if (true)\n"
               "          {\n"
               "          a();\n"
               "          b();\n"
               "          }\n"
               "        }\n"
               "      void g()\n"
               "        {\n"
               "        return;\n"
               "        }\n"
               "      };\n"
               "    struct B\n"
               "      {\n"
               "      int x;\n"
               "      };\n"
               "    } // namespace b\n"
               "  } // namespace a",
               WhitesmithsBraceStyle);

  verifyFormat("void f()\n"
               "  {\n"
               "  if (true)\n"
               "    {\n"
               "    a();\n"
               "    }\n"
               "  else if (false)\n"
               "    {\n"
               "    b();\n"
               "    }\n"
               "  else\n"
               "    {\n"
               "    c();\n"
               "    }\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("void f()\n"
               "  {\n"
               "  for (int i = 0; i < 10; ++i)\n"
               "    {\n"
               "    a();\n"
               "    }\n"
               "  while (false)\n"
               "    {\n"
               "    b();\n"
               "    }\n"
               "  do\n"
               "    {\n"
               "    c();\n"
               "    } while (false)\n"
               "  }",
               WhitesmithsBraceStyle);

  WhitesmithsBraceStyle.IndentCaseLabels = true;
  verifyFormat("void switchTest1(int a)\n"
               "  {\n"
               "  switch (a)\n"
               "    {\n"
               "    case 2:\n"
               "      {\n"
               "      }\n"
               "      break;\n"
               "    }\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("void switchTest2(int a)\n"
               "  {\n"
               "  switch (a)\n"
               "    {\n"
               "    case 0:\n"
               "      break;\n"
               "    case 1:\n"
               "      {\n"
               "      break;\n"
               "      }\n"
               "    case 2:\n"
               "      {\n"
               "      }\n"
               "      break;\n"
               "    default:\n"
               "      break;\n"
               "    }\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("void switchTest3(int a)\n"
               "  {\n"
               "  switch (a)\n"
               "    {\n"
               "    case 0:\n"
               "      {\n"
               "      foo(x);\n"
               "      }\n"
               "      break;\n"
               "    default:\n"
               "      {\n"
               "      foo(1);\n"
               "      }\n"
               "      break;\n"
               "    }\n"
               "  }",
               WhitesmithsBraceStyle);

  WhitesmithsBraceStyle.IndentCaseLabels = false;

  verifyFormat("void switchTest4(int a)\n"
               "  {\n"
               "  switch (a)\n"
               "    {\n"
               "  case 2:\n"
               "    {\n"
               "    }\n"
               "    break;\n"
               "    }\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("void switchTest5(int a)\n"
               "  {\n"
               "  switch (a)\n"
               "    {\n"
               "  case 0:\n"
               "    break;\n"
               "  case 1:\n"
               "    {\n"
               "    foo();\n"
               "    break;\n"
               "    }\n"
               "  case 2:\n"
               "    {\n"
               "    }\n"
               "    break;\n"
               "  default:\n"
               "    break;\n"
               "    }\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("void switchTest6(int a)\n"
               "  {\n"
               "  switch (a)\n"
               "    {\n"
               "  case 0:\n"
               "    {\n"
               "    foo(x);\n"
               "    }\n"
               "    break;\n"
               "  default:\n"
               "    {\n"
               "    foo(1);\n"
               "    }\n"
               "    break;\n"
               "    }\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("enum X\n"
               "  {\n"
               "  Y = 0, // testing\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("enum X\n"
               "  {\n"
               "  Y = 0\n"
               "  }",
               WhitesmithsBraceStyle);
  verifyFormat("enum X\n"
               "  {\n"
               "  Y = 0,\n"
               "  Z = 1\n"
               "  };",
               WhitesmithsBraceStyle);

  verifyFormat("@interface BSApplicationController ()\n"
               "  {\n"
               "@private\n"
               "  id _extraIvar;\n"
               "  }\n"
               "@end",
               WhitesmithsBraceStyle);

  verifyFormat("#ifdef _DEBUG\n"
               "int foo(int i = 0)\n"
               "#else\n"
               "int foo(int i = 5)\n"
               "#endif\n"
               "  {\n"
               "  return i;\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("void foo() {}\n"
               "void bar()\n"
               "#ifdef _DEBUG\n"
               "  {\n"
               "  foo();\n"
               "  }\n"
               "#else\n"
               "  {\n"
               "  }\n"
               "#endif",
               WhitesmithsBraceStyle);

  verifyFormat("void foobar()\n"
               "  {\n"
               "  int i = 5;\n"
               "  }\n"
               "#ifdef _DEBUG\n"
               "void bar() {}\n"
               "#else\n"
               "void bar()\n"
               "  {\n"
               "  foobar();\n"
               "  }\n"
               "#endif",
               WhitesmithsBraceStyle);

  // This shouldn't affect ObjC blocks..
  verifyFormat("[self doSomeThingWithACompletionHandler:^{\n"
               "  // ...\n"
               "  int i;\n"
               "}];",
               WhitesmithsBraceStyle);
  verifyFormat("void (^block)(void) = ^{\n"
               "  // ...\n"
               "  int i;\n"
               "};",
               WhitesmithsBraceStyle);
  // .. or dict literals.
  verifyFormat("void f()\n"
               "  {\n"
               "  [object someMethod:@{@\"a\" : @\"b\"}];\n"
               "  }",
               WhitesmithsBraceStyle);

  verifyFormat("int f()\n"
               "  { // comment\n"
               "  return 42;\n"
               "  }",
               WhitesmithsBraceStyle);

  FormatStyle BreakBeforeBraceShortIfs = WhitesmithsBraceStyle;
  BreakBeforeBraceShortIfs.AllowShortIfStatementsOnASingleLine =
      FormatStyle::SIS_OnlyFirstIf;
  BreakBeforeBraceShortIfs.AllowShortLoopsOnASingleLine = true;
  verifyFormat("void f(bool b)\n"
               "  {\n"
               "  if (b)\n"
               "    {\n"
               "    return;\n"
               "    }\n"
               "  }",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "  {\n"
               "  if (b) return;\n"
               "  }",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "  {\n"
               "  while (b)\n"
               "    {\n"
               "    return;\n"
               "    }\n"
               "  }",
               BreakBeforeBraceShortIfs);
}

TEST_F(FormatTest, GNUBraceBreaking) {
  FormatStyle GNUBraceStyle = getLLVMStyle();
  GNUBraceStyle.BreakBeforeBraces = FormatStyle::BS_GNU;
  verifyFormat("namespace a\n"
               "{\n"
               "class A\n"
               "{\n"
               "  void f()\n"
               "  {\n"
               "    int a;\n"
               "    {\n"
               "      int b;\n"
               "    }\n"
               "    if (true)\n"
               "      {\n"
               "        a();\n"
               "        b();\n"
               "      }\n"
               "  }\n"
               "  void g() { return; }\n"
               "}\n"
               "} // namespace a",
               GNUBraceStyle);

  verifyFormat("void f()\n"
               "{\n"
               "  if (true)\n"
               "    {\n"
               "      a();\n"
               "    }\n"
               "  else if (false)\n"
               "    {\n"
               "      b();\n"
               "    }\n"
               "  else\n"
               "    {\n"
               "      c();\n"
               "    }\n"
               "}",
               GNUBraceStyle);

  verifyFormat("void f()\n"
               "{\n"
               "  for (int i = 0; i < 10; ++i)\n"
               "    {\n"
               "      a();\n"
               "    }\n"
               "  while (false)\n"
               "    {\n"
               "      b();\n"
               "    }\n"
               "  do\n"
               "    {\n"
               "      c();\n"
               "    }\n"
               "  while (false);\n"
               "}",
               GNUBraceStyle);

  verifyFormat("void f(int a)\n"
               "{\n"
               "  switch (a)\n"
               "    {\n"
               "    case 0:\n"
               "      break;\n"
               "    case 1:\n"
               "      {\n"
               "        break;\n"
               "      }\n"
               "    case 2:\n"
               "      {\n"
               "      }\n"
               "      break;\n"
               "    default:\n"
               "      break;\n"
               "    }\n"
               "}",
               GNUBraceStyle);

  verifyFormat("enum X\n"
               "{\n"
               "  Y = 0,\n"
               "}",
               GNUBraceStyle);

  verifyFormat("@interface BSApplicationController ()\n"
               "{\n"
               "@private\n"
               "  id _extraIvar;\n"
               "}\n"
               "@end",
               GNUBraceStyle);

  verifyFormat("#ifdef _DEBUG\n"
               "int foo(int i = 0)\n"
               "#else\n"
               "int foo(int i = 5)\n"
               "#endif\n"
               "{\n"
               "  return i;\n"
               "}",
               GNUBraceStyle);

  verifyFormat("void foo() {}\n"
               "void bar()\n"
               "#ifdef _DEBUG\n"
               "{\n"
               "  foo();\n"
               "}\n"
               "#else\n"
               "{\n"
               "}\n"
               "#endif",
               GNUBraceStyle);

  verifyFormat("void foobar() { int i = 5; }\n"
               "#ifdef _DEBUG\n"
               "void bar() {}\n"
               "#else\n"
               "void bar() { foobar(); }\n"
               "#endif",
               GNUBraceStyle);
}

TEST_F(FormatTest, WebKitBraceBreaking) {
  FormatStyle WebKitBraceStyle = getLLVMStyle();
  WebKitBraceStyle.BreakBeforeBraces = FormatStyle::BS_WebKit;
  WebKitBraceStyle.FixNamespaceComments = false;
  verifyFormat("namespace a {\n"
               "class A {\n"
               "  void f()\n"
               "  {\n"
               "    if (true) {\n"
               "      a();\n"
               "      b();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "enum E {\n"
               "  A,\n"
               "  // foo\n"
               "  B,\n"
               "  C\n"
               "};\n"
               "struct B {\n"
               "  int x;\n"
               "};\n"
               "}",
               WebKitBraceStyle);
  verifyFormat("struct S {\n"
               "  int Type;\n"
               "  union {\n"
               "    int x;\n"
               "    double y;\n"
               "  } Value;\n"
               "  class C {\n"
               "    MyFavoriteType Value;\n"
               "  } Class;\n"
               "};",
               WebKitBraceStyle);
}

TEST_F(FormatTest, CatchExceptionReferenceBinding) {
  verifyFormat("void f() {\n"
               "  try {\n"
               "  } catch (const Exception &e) {\n"
               "  }\n"
               "}");
}

TEST_F(FormatTest, CatchAlignArrayOfStructuresRightAlignment) {
  auto Style = getLLVMStyle();
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Right;
  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("struct test demo[] = {\n"
               "    {56,    23, \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    { 7,     5,    \"!!\"}\n"
               "};",
               Style);

  verifyFormat("struct test demo[] = {\n"
               "    {56,    23, \"hello\"}, // first line\n"
               "    {-1, 93463, \"world\"}, // second line\n"
               "    { 7,     5,    \"!!\"}  // third line\n"
               "};",
               Style);

  verifyFormat("struct test demo[4] = {\n"
               "    { 56,    23, 21,       \"oh\"}, // first line\n"
               "    { -1, 93463, 22,       \"my\"}, // second line\n"
               "    {  7,     5,  1, \"goodness\"}  // third line\n"
               "    {234,     5,  1, \"gracious\"}  // fourth line\n"
               "};",
               Style);

  verifyFormat("struct test demo[3] = {\n"
               "    {56,    23, \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    { 7,     5,    \"!!\"}\n"
               "};",
               Style);

  verifyFormat("struct test demo[3] = {\n"
               "    {int{56},    23, \"hello\"},\n"
               "    {int{-1}, 93463, \"world\"},\n"
               "    { int{7},     5,    \"!!\"}\n"
               "};",
               Style);

  verifyFormat("struct test demo[] = {\n"
               "    {56,    23, \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    { 7,     5,    \"!!\"},\n"
               "};",
               Style);

  verifyFormat("test demo[] = {\n"
               "    {56,    23, \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    { 7,     5,    \"!!\"},\n"
               "};",
               Style);

  verifyFormat("demo = std::array<struct test, 3>{\n"
               "    test{56,    23, \"hello\"},\n"
               "    test{-1, 93463, \"world\"},\n"
               "    test{ 7,     5,    \"!!\"},\n"
               "};",
               Style);

  verifyFormat("test demo[] = {\n"
               "    {56,    23, \"hello\"},\n"
               "#if X\n"
               "    {-1, 93463, \"world\"},\n"
               "#endif\n"
               "    { 7,     5,    \"!!\"}\n"
               "};",
               Style);

  verifyFormat(
      "test demo[] = {\n"
      "    { 7,    23,\n"
      "     \"hello world i am a very long line that really, in any\"\n"
      "     \"just world, ought to be split over multiple lines\"},\n"
      "    {-1, 93463,                                  \"world\"},\n"
      "    {56,     5,                                     \"!!\"}\n"
      "};",
      Style);

  verifyFormat("return GradForUnaryCwise(g, {\n"
               "                                {{\"sign\"}, \"Sign\",  "
               "  {\"x\", \"dy\"}},\n"
               "                                {  {\"dx\"},  \"Mul\", {\"dy\""
               ", \"sign\"}},\n"
               "});",
               Style);

  Style.Cpp11BracedListStyle = false;
  verifyFormat("struct test demo[] = {\n"
               "  { 56,    23, \"hello\" },\n"
               "  { -1, 93463, \"world\" },\n"
               "  {  7,     5,    \"!!\" }\n"
               "};",
               Style);
  Style.Cpp11BracedListStyle = true;

  Style.ColumnLimit = 0;
  verifyFormat(
      "test demo[] = {\n"
      "    {56,    23, \"hello world i am a very long line that really, "
      "in any just world, ought to be split over multiple lines\"},\n"
      "    {-1, 93463,                                                  "
      "                                                 \"world\"},\n"
      "    { 7,     5,                                                  "
      "                                                    \"!!\"},\n"
      "};",
      "test demo[] = {{56, 23, \"hello world i am a very long line "
      "that really, in any just world, ought to be split over multiple "
      "lines\"},{-1, 93463, \"world\"},{7, 5, \"!!\"},};",
      Style);

  Style.ColumnLimit = 80;
  verifyFormat("test demo[] = {\n"
               "    {56,    23, /* a comment */ \"hello\"},\n"
               "    {-1, 93463,                 \"world\"},\n"
               "    { 7,     5,                    \"!!\"}\n"
               "};",
               Style);

  verifyFormat("test demo[] = {\n"
               "    {56,    23,                    \"hello\"},\n"
               "    {-1, 93463, \"world\" /* comment here */},\n"
               "    { 7,     5,                       \"!!\"}\n"
               "};",
               Style);

  verifyFormat("test demo[] = {\n"
               "    {56, /* a comment */ 23, \"hello\"},\n"
               "    {-1,              93463, \"world\"},\n"
               "    { 7,                  5,    \"!!\"}\n"
               "};",
               Style);

  Style.ColumnLimit = 20;
  verifyFormat("demo = std::array<\n"
               "    struct test, 3>{\n"
               "    test{\n"
               "         56,    23,\n"
               "         \"hello \"\n"
               "         \"world i \"\n"
               "         \"am a very \"\n"
               "         \"long line \"\n"
               "         \"that \"\n"
               "         \"really, \"\n"
               "         \"in any \"\n"
               "         \"just \"\n"
               "         \"world, \"\n"
               "         \"ought to \"\n"
               "         \"be split \"\n"
               "         \"over \"\n"
               "         \"multiple \"\n"
               "         \"lines\"},\n"
               "    test{-1, 93463,\n"
               "         \"world\"},\n"
               "    test{ 7,     5,\n"
               "         \"!!\"   },\n"
               "};",
               "demo = std::array<struct test, 3>{test{56, 23, \"hello world "
               "i am a very long line that really, in any just world, ought "
               "to be split over multiple lines\"},test{-1, 93463, \"world\"},"
               "test{7, 5, \"!!\"},};",
               Style);
  // This caused a core dump by enabling Alignment in the LLVMStyle globally
  Style = getLLVMStyleWithColumns(50);
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Right;
  verifyFormat("static A x = {\n"
               "    {{init1, init2, init3, init4},\n"
               "     {init1, init2, init3, init4}}\n"
               "};",
               Style);
  // TODO: Fix the indentations below when this option is fully functional.
  verifyFormat("int a[][] = {\n"
               "    {\n"
               "     {0, 2}, //\n"
               " {1, 2}  //\n"
               "    }\n"
               "};",
               Style);
  Style.ColumnLimit = 100;
  verifyFormat(
      "test demo[] = {\n"
      "    {56,    23,\n"
      "     \"hello world i am a very long line that really, in any just world"
      ", ought to be split over \"\n"
      "     \"multiple lines\"  },\n"
      "    {-1, 93463, \"world\"},\n"
      "    { 7,     5,    \"!!\"},\n"
      "};",
      "test demo[] = {{56, 23, \"hello world i am a very long line "
      "that really, in any just world, ought to be split over multiple "
      "lines\"},{-1, 93463, \"world\"},{7, 5, \"!!\"},};",
      Style);

  Style = getLLVMStyleWithColumns(50);
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Right;
  verifyFormat("struct test demo[] = {\n"
               "    {56,    23, \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    { 7,     5,    \"!!\"}\n"
               "};\n"
               "static A x = {\n"
               "    {{init1, init2, init3, init4},\n"
               "     {init1, init2, init3, init4}}\n"
               "};",
               Style);
  Style.ColumnLimit = 100;
  Style.AlignConsecutiveAssignments.AcrossComments = true;
  Style.AlignConsecutiveDeclarations.AcrossComments = true;
  verifyFormat("struct test demo[] = {\n"
               "    {56,    23, \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    { 7,     5,    \"!!\"}\n"
               "};\n"
               "struct test demo[4] = {\n"
               "    { 56,    23, 21,       \"oh\"}, // first line\n"
               "    { -1, 93463, 22,       \"my\"}, // second line\n"
               "    {  7,     5,  1, \"goodness\"}  // third line\n"
               "    {234,     5,  1, \"gracious\"}  // fourth line\n"
               "};",
               Style);
  verifyFormat(
      "test demo[] = {\n"
      "    {56,\n"
      "     \"hello world i am a very long line that really, in any just world"
      ", ought to be split over \"\n"
      "     \"multiple lines\",    23},\n"
      "    {-1,      \"world\", 93463},\n"
      "    { 7,         \"!!\",     5},\n"
      "};",
      "test demo[] = {{56, \"hello world i am a very long line "
      "that really, in any just world, ought to be split over multiple "
      "lines\", 23},{-1, \"world\", 93463},{7, \"!!\", 5},};",
      Style);
}

TEST_F(FormatTest, CatchAlignArrayOfStructuresLeftAlignment) {
  auto Style = getLLVMStyle();
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Left;
  /* FIXME: This case gets misformatted.
  verifyFormat("auto foo = Items{\n"
               "    Section{0, bar(), },\n"
               "    Section{1, boo()  }\n"
               "};",
               Style);
  */
  verifyFormat("auto foo = Items{\n"
               "    Section{\n"
               "            0, bar(),\n"
               "            }\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {56, 23,    \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    {7,  5,     \"!!\"   }\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {56, 23,    \"hello\"}, // first line\n"
               "    {-1, 93463, \"world\"}, // second line\n"
               "    {7,  5,     \"!!\"   }  // third line\n"
               "};",
               Style);
  verifyFormat("struct test demo[4] = {\n"
               "    {56,  23,    21, \"oh\"      }, // first line\n"
               "    {-1,  93463, 22, \"my\"      }, // second line\n"
               "    {7,   5,     1,  \"goodness\"}  // third line\n"
               "    {234, 5,     1,  \"gracious\"}  // fourth line\n"
               "};",
               Style);
  verifyFormat("struct test demo[3] = {\n"
               "    {56, 23,    \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    {7,  5,     \"!!\"   }\n"
               "};",
               Style);

  verifyFormat("struct test demo[3] = {\n"
               "    {int{56}, 23,    \"hello\"},\n"
               "    {int{-1}, 93463, \"world\"},\n"
               "    {int{7},  5,     \"!!\"   }\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {56, 23,    \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    {7,  5,     \"!!\"   },\n"
               "};",
               Style);
  verifyFormat("test demo[] = {\n"
               "    {56, 23,    \"hello\"},\n"
               "    {-1, 93463, \"world\"},\n"
               "    {7,  5,     \"!!\"   },\n"
               "};",
               Style);
  verifyFormat("demo = std::array<struct test, 3>{\n"
               "    test{56, 23,    \"hello\"},\n"
               "    test{-1, 93463, \"world\"},\n"
               "    test{7,  5,     \"!!\"   },\n"
               "};",
               Style);
  verifyFormat("test demo[] = {\n"
               "    {56, 23,    \"hello\"},\n"
               "#if X\n"
               "    {-1, 93463, \"world\"},\n"
               "#endif\n"
               "    {7,  5,     \"!!\"   }\n"
               "};",
               Style);
  verifyFormat(
      "test demo[] = {\n"
      "    {7,  23,\n"
      "     \"hello world i am a very long line that really, in any\"\n"
      "     \"just world, ought to be split over multiple lines\"},\n"
      "    {-1, 93463, \"world\"                                 },\n"
      "    {56, 5,     \"!!\"                                    }\n"
      "};",
      Style);

  verifyFormat("return GradForUnaryCwise(g, {\n"
               "                                {{\"sign\"}, \"Sign\", {\"x\", "
               "\"dy\"}   },\n"
               "                                {{\"dx\"},   \"Mul\",  "
               "{\"dy\", \"sign\"}},\n"
               "});",
               Style);

  Style.Cpp11BracedListStyle = false;
  verifyFormat("struct test demo[] = {\n"
               "  { 56, 23,    \"hello\" },\n"
               "  { -1, 93463, \"world\" },\n"
               "  { 7,  5,     \"!!\"    }\n"
               "};",
               Style);
  Style.Cpp11BracedListStyle = true;

  Style.ColumnLimit = 0;
  verifyFormat(
      "test demo[] = {\n"
      "    {56, 23,    \"hello world i am a very long line that really, in any "
      "just world, ought to be split over multiple lines\"},\n"
      "    {-1, 93463, \"world\"                                               "
      "                                                   },\n"
      "    {7,  5,     \"!!\"                                                  "
      "                                                   },\n"
      "};",
      "test demo[] = {{56, 23, \"hello world i am a very long line "
      "that really, in any just world, ought to be split over multiple "
      "lines\"},{-1, 93463, \"world\"},{7, 5, \"!!\"},};",
      Style);

  Style.ColumnLimit = 80;
  verifyFormat("test demo[] = {\n"
               "    {56, 23,    /* a comment */ \"hello\"},\n"
               "    {-1, 93463, \"world\"                },\n"
               "    {7,  5,     \"!!\"                   }\n"
               "};",
               Style);

  verifyFormat("test demo[] = {\n"
               "    {56, 23,    \"hello\"                   },\n"
               "    {-1, 93463, \"world\" /* comment here */},\n"
               "    {7,  5,     \"!!\"                      }\n"
               "};",
               Style);

  verifyFormat("test demo[] = {\n"
               "    {56, /* a comment */ 23, \"hello\"},\n"
               "    {-1, 93463,              \"world\"},\n"
               "    {7,  5,                  \"!!\"   }\n"
               "};",
               Style);

  Style.ColumnLimit = 20;
  // FIXME: unstable test case
  EXPECT_EQ(
      "demo = std::array<\n"
      "    struct test, 3>{\n"
      "    test{\n"
      "         56, 23,\n"
      "         \"hello \"\n"
      "         \"world i \"\n"
      "         \"am a very \"\n"
      "         \"long line \"\n"
      "         \"that \"\n"
      "         \"really, \"\n"
      "         \"in any \"\n"
      "         \"just \"\n"
      "         \"world, \"\n"
      "         \"ought to \"\n"
      "         \"be split \"\n"
      "         \"over \"\n"
      "         \"multiple \"\n"
      "         \"lines\"},\n"
      "    test{-1, 93463,\n"
      "         \"world\"},\n"
      "    test{7,  5,\n"
      "         \"!!\"   },\n"
      "};",
      format("demo = std::array<struct test, 3>{test{56, 23, \"hello world "
             "i am a very long line that really, in any just world, ought "
             "to be split over multiple lines\"},test{-1, 93463, \"world\"},"
             "test{7, 5, \"!!\"},};",
             Style));

  // This caused a core dump by enabling Alignment in the LLVMStyle globally
  Style = getLLVMStyleWithColumns(50);
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Left;
  verifyFormat("static A x = {\n"
               "    {{init1, init2, init3, init4},\n"
               "     {init1, init2, init3, init4}}\n"
               "};",
               Style);
  Style.ColumnLimit = 100;
  verifyFormat(
      "test demo[] = {\n"
      "    {56, 23,\n"
      "     \"hello world i am a very long line that really, in any just world"
      ", ought to be split over \"\n"
      "     \"multiple lines\"  },\n"
      "    {-1, 93463, \"world\"},\n"
      "    {7,  5,     \"!!\"   },\n"
      "};",
      "test demo[] = {{56, 23, \"hello world i am a very long line "
      "that really, in any just world, ought to be split over multiple "
      "lines\"},{-1, 93463, \"world\"},{7, 5, \"!!\"},};",
      Style);
}

TEST_F(FormatTest, UnderstandsPragmas) {
  verifyFormat("#pragma omp reduction(| : var)");
  verifyFormat("#pragma omp reduction(+ : var)");

  verifyFormat("#pragma mark Any non-hyphenated or hyphenated string "
               "(including parentheses).",
               "#pragma    mark   Any non-hyphenated or hyphenated string "
               "(including parentheses).");

  verifyFormat("#pragma mark Any non-hyphenated or hyphenated string "
               "(including parentheses).",
               "#pragma    mark   Any non-hyphenated or hyphenated string "
               "(including parentheses).");

  verifyFormat("#pragma comment(linker,    \\\n"
               "                \"argument\" \\\n"
               "                \"argument\"",
               "#pragma comment(linker,      \\\n"
               "                 \"argument\" \\\n"
               "                 \"argument\"",
               getStyleWithColumns(
                   getChromiumStyle(FormatStyle::LanguageKind::LK_Cpp), 32));
}

TEST_F(FormatTest, UnderstandsPragmaOmpTarget) {
  verifyFormat("#pragma omp target map(to : var)");
  verifyFormat("#pragma omp target map(to : var[ : N])");
  verifyFormat("#pragma omp target map(to : var[0 : N])");
  verifyFormat("#pragma omp target map(always, to : var[0 : N])");

  verifyFormat(
      "#pragma omp target       \\\n"
      "    reduction(+ : var)   \\\n"
      "    map(to : A[0 : N])   \\\n"
      "    map(to : B[0 : N])   \\\n"
      "    map(from : C[0 : N]) \\\n"
      "    firstprivate(i)      \\\n"
      "    firstprivate(j)      \\\n"
      "    firstprivate(k)",
      "#pragma omp target reduction(+:var) map(to:A[0:N]) map(to:B[0:N]) "
      "map(from:C[0:N]) firstprivate(i) firstprivate(j) firstprivate(k)",
      getLLVMStyleWithColumns(26));
}

TEST_F(FormatTest, UnderstandPragmaOption) {
  verifyFormat("#pragma option -C -A");

  verifyFormat("#pragma option -C -A", "#pragma    option   -C   -A");
}

TEST_F(FormatTest, UnderstandPragmaRegion) {
  auto Style = getLLVMStyleWithColumns(0);
  verifyFormat("#pragma region TEST(FOO : BAR)", Style);
  verifyFormat("#pragma region TEST(FOO: NOSPACE)", Style);
}

TEST_F(FormatTest, OptimizeBreakPenaltyVsExcess) {
  FormatStyle Style = getLLVMStyleWithColumns(20);

  // See PR41213
  verifyFormat("/*\n"
               " *\t9012345\n"
               " * /8901\n"
               " */",
               "/*\n"
               " *\t9012345 /8901\n"
               " */",
               Style);
  verifyFormat("/*\n"
               " *345678\n"
               " *\t/8901\n"
               " */",
               "/*\n"
               " *345678\t/8901\n"
               " */",
               Style);

  verifyFormat("int a; // the\n"
               "       // comment",
               Style);
  verifyNoChange("int a; /* first line\n"
                 "        * second\n"
                 "        * line third\n"
                 "        * line\n"
                 "        */",
                 Style);
  verifyFormat("int a; // first line\n"
               "       // second\n"
               "       // line third\n"
               "       // line",
               "int a; // first line\n"
               "       // second line\n"
               "       // third line",
               Style);

  Style.PenaltyExcessCharacter = 90;
  verifyFormat("int a; // the comment", Style);
  verifyFormat("int a; // the comment\n"
               "       // aaa",
               "int a; // the comment aaa", Style);
  verifyNoChange("int a; /* first line\n"
                 "        * second line\n"
                 "        * third line\n"
                 "        */",
                 Style);
  verifyFormat("int a; // first line\n"
               "       // second line\n"
               "       // third line",
               Style);
  // FIXME: Investigate why this is not getting the same layout as the test
  // above.
  verifyFormat("int a; /* first line\n"
               "        * second line\n"
               "        * third line\n"
               "        */",
               "int a; /* first line second line third line"
               "\n*/",
               Style);

  verifyFormat("// foo bar baz bazfoo\n"
               "// foo bar foo bar",
               "// foo bar baz bazfoo\n"
               "// foo bar foo           bar",
               Style);
  verifyFormat("// foo bar baz bazfoo\n"
               "// foo bar foo bar",
               "// foo bar baz      bazfoo\n"
               "// foo            bar foo bar",
               Style);

  // FIXME: Optimally, we'd keep bazfoo on the first line and reflow bar to the
  // next one.
  verifyFormat("// foo bar baz bazfoo\n"
               "// bar foo bar",
               "// foo bar baz      bazfoo bar\n"
               "// foo            bar",
               Style);

  // FIXME: unstable test case
  EXPECT_EQ("// foo bar baz bazfoo\n"
            "// foo bar baz bazfoo\n"
            "// bar foo bar",
            format("// foo bar baz      bazfoo\n"
                   "// foo bar baz      bazfoo bar\n"
                   "// foo bar",
                   Style));

  // FIXME: unstable test case
  EXPECT_EQ("// foo bar baz bazfoo\n"
            "// foo bar baz bazfoo\n"
            "// bar foo bar",
            format("// foo bar baz      bazfoo\n"
                   "// foo bar baz      bazfoo bar\n"
                   "// foo           bar",
                   Style));

  // Make sure we do not keep protruding characters if strict mode reflow is
  // cheaper than keeping protruding characters.
  Style.ColumnLimit = 21;
  verifyFormat("// foo foo foo foo\n"
               "// foo foo foo foo\n"
               "// foo foo foo foo",
               "// foo foo foo foo foo foo foo foo foo foo foo foo", Style);

  verifyFormat("int a = /* long block\n"
               "           comment */\n"
               "    42;",
               "int a = /* long block comment */ 42;", Style);
}

TEST_F(FormatTest, BreakPenaltyAfterLParen) {
  FormatStyle Style = getLLVMStyle();
  Style.ColumnLimit = 8;
  Style.PenaltyExcessCharacter = 15;
  verifyFormat("int foo(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  Style.PenaltyBreakOpenParenthesis = 200;
  verifyFormat("int foo(int aaaaaaaaaaaaaaaaaaaaaaaa);",
               "int foo(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
}

TEST_F(FormatTest, BreakPenaltyAfterCastLParen) {
  FormatStyle Style = getLLVMStyle();
  Style.ColumnLimit = 5;
  Style.PenaltyExcessCharacter = 150;
  verifyFormat("foo((\n"
               "    int)aaaaaaaaaaaaaaaaaaaaaaaa);",

               Style);
  Style.PenaltyBreakOpenParenthesis = 100000;
  verifyFormat("foo((int)\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaa);",
               "foo((\n"
               "int)aaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
}

TEST_F(FormatTest, BreakPenaltyAfterForLoopLParen) {
  FormatStyle Style = getLLVMStyle();
  Style.ColumnLimit = 4;
  Style.PenaltyExcessCharacter = 100;
  verifyFormat("for (\n"
               "    int iiiiiiiiiiiiiiiii =\n"
               "        0;\n"
               "    iiiiiiiiiiiiiiiii <\n"
               "    2;\n"
               "    iiiiiiiiiiiiiiiii++) {\n"
               "}",

               Style);
  Style.PenaltyBreakOpenParenthesis = 1250;
  verifyFormat("for (int iiiiiiiiiiiiiiiii =\n"
               "         0;\n"
               "     iiiiiiiiiiiiiiiii <\n"
               "     2;\n"
               "     iiiiiiiiiiiiiiiii++) {\n"
               "}",
               "for (\n"
               "    int iiiiiiiiiiiiiiiii =\n"
               "        0;\n"
               "    iiiiiiiiiiiiiiiii <\n"
               "    2;\n"
               "    iiiiiiiiiiiiiiiii++) {\n"
               "}",
               Style);
}

TEST_F(FormatTest, WorksFor8bitEncodings) {
  // FIXME: unstable test case
  EXPECT_EQ("\"\xce\xe4\xed\xe0\xe6\xe4\xfb \xe2 \"\n"
            "\"\xf1\xf2\xf3\xe4\xb8\xed\xf3\xfe \"\n"
            "\"\xe7\xe8\xec\xed\xfe\xfe \"\n"
            "\"\xef\xee\xf0\xf3...\"",
            format("\"\xce\xe4\xed\xe0\xe6\xe4\xfb \xe2 "
                   "\xf1\xf2\xf3\xe4\xb8\xed\xf3\xfe \xe7\xe8\xec\xed\xfe\xfe "
                   "\xef\xee\xf0\xf3...\"",
                   getLLVMStyleWithColumns(12)));
}

TEST_F(FormatTest, HandlesUTF8BOM) {
  verifyFormat("\xef\xbb\xbf");
  verifyFormat("\xef\xbb\xbf#include <iostream>");
  verifyFormat("\xef\xbb\xbf\n#include <iostream>");
}

// FIXME: Encode Cyrillic and CJK characters below to appease MS compilers.
#if !defined(_MSC_VER)

TEST_F(FormatTest, CountsUTF8CharactersProperly) {
  verifyFormat("\"Однажды в студёную зимнюю пору...\"",
               getLLVMStyleWithColumns(35));
  verifyFormat("\"一 二 三 四 五 六 七 八 九 十\"",
               getLLVMStyleWithColumns(31));
  verifyFormat("// Однажды в студёную зимнюю пору...",
               getLLVMStyleWithColumns(36));
  verifyFormat("// 一 二 三 四 五 六 七 八 九 十", getLLVMStyleWithColumns(32));
  verifyFormat("/* Однажды в студёную зимнюю пору... */",
               getLLVMStyleWithColumns(39));
  verifyFormat("/* 一 二 三 四 五 六 七 八 九 十 */",
               getLLVMStyleWithColumns(35));
}

TEST_F(FormatTest, SplitsUTF8Strings) {
  // Non-printable characters' width is currently considered to be the length in
  // bytes in UTF8. The characters can be displayed in very different manner
  // (zero-width, single width with a substitution glyph, expanded to their code
  // (e.g. "<8d>"), so there's no single correct way to handle them.
  // FIXME: unstable test case
  EXPECT_EQ("\"aaaaÄ\"\n"
            "\"\xc2\x8d\";",
            format("\"aaaaÄ\xc2\x8d\";", getLLVMStyleWithColumns(10)));
  // FIXME: unstable test case
  EXPECT_EQ("\"aaaaaaaÄ\"\n"
            "\"\xc2\x8d\";",
            format("\"aaaaaaaÄ\xc2\x8d\";", getLLVMStyleWithColumns(10)));
  // FIXME: unstable test case
  EXPECT_EQ("\"Однажды, в \"\n"
            "\"студёную \"\n"
            "\"зимнюю \"\n"
            "\"пору,\"",
            format("\"Однажды, в студёную зимнюю пору,\"",
                   getLLVMStyleWithColumns(13)));
  // FIXME: unstable test case
  EXPECT_EQ(
      "\"一 二 三 \"\n"
      "\"四 五六 \"\n"
      "\"七 八 九 \"\n"
      "\"十\"",
      format("\"一 二 三 四 五六 七 八 九 十\"", getLLVMStyleWithColumns(11)));
  // FIXME: unstable test case
  EXPECT_EQ("\"一\t\"\n"
            "\"二 \t\"\n"
            "\"三 四 \"\n"
            "\"五\t\"\n"
            "\"六 \t\"\n"
            "\"七 \"\n"
            "\"八九十\tqq\"",
            format("\"一\t二 \t三 四 五\t六 \t七 八九十\tqq\"",
                   getLLVMStyleWithColumns(11)));

  // UTF8 character in an escape sequence.
  // FIXME: unstable test case
  EXPECT_EQ("\"aaaaaa\"\n"
            "\"\\\xC2\x8D\"",
            format("\"aaaaaa\\\xC2\x8D\"", getLLVMStyleWithColumns(10)));
}

TEST_F(FormatTest, HandlesDoubleWidthCharsInMultiLineStrings) {
  verifyFormat("const char *sssss =\n"
               "    \"一二三四五六七八\\\n"
               " 九 十\";",
               "const char *sssss = \"一二三四五六七八\\\n"
               " 九 十\";",
               getLLVMStyleWithColumns(30));
}

TEST_F(FormatTest, SplitsUTF8LineComments) {
  verifyFormat("// aaaaÄ\xc2\x8d", getLLVMStyleWithColumns(10));
  verifyFormat("// Я из лесу\n"
               "// вышел; был\n"
               "// сильный\n"
               "// мороз.",
               "// Я из лесу вышел; был сильный мороз.",
               getLLVMStyleWithColumns(13));
  verifyFormat("// 一二三\n"
               "// 四五六七\n"
               "// 八  九\n"
               "// 十",
               "// 一二三 四五六七 八  九 十", getLLVMStyleWithColumns(9));
}

TEST_F(FormatTest, SplitsUTF8BlockComments) {
  verifyFormat("/* Гляжу,\n"
               " * поднимается\n"
               " * медленно в\n"
               " * гору\n"
               " * Лошадка,\n"
               " * везущая\n"
               " * хворосту\n"
               " * воз. */",
               "/* Гляжу, поднимается медленно в гору\n"
               " * Лошадка, везущая хворосту воз. */",
               getLLVMStyleWithColumns(13));
  verifyFormat("/* 一二三\n"
               " * 四五六七\n"
               " * 八  九\n"
               " * 十  */",
               "/* 一二三 四五六七 八  九 十  */", getLLVMStyleWithColumns(9));
  verifyFormat("/* 𝓣𝓮𝓼𝓽 𝔣𝔬𝔲𝔯\n"
               " * 𝕓𝕪𝕥𝕖\n"
               " * 𝖀𝕿𝕱-𝟠 */",
               "/* 𝓣𝓮𝓼𝓽 𝔣𝔬𝔲𝔯 𝕓𝕪𝕥𝕖 𝖀𝕿𝕱-𝟠 */", getLLVMStyleWithColumns(12));
}

#endif // _MSC_VER

TEST_F(FormatTest, ConstructorInitializerIndentWidth) {
  FormatStyle Style = getLLVMStyle();

  Style.ConstructorInitializerIndentWidth = 4;
  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
      Style);

  Style.ConstructorInitializerIndentWidth = 2;
  verifyFormat(
      "SomeClass::Constructor()\n"
      "  : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
      Style);

  Style.ConstructorInitializerIndentWidth = 0;
  verifyFormat(
      "SomeClass::Constructor()\n"
      ": aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "  aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
      Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  verifyFormat(
      "SomeLongTemplateVariableName<\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>",
      Style);
  verifyFormat("bool smaller = 1 < "
               "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(\n"
               "                       "
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);

  Style.BreakConstructorInitializers = FormatStyle::BCIS_AfterColon;
  verifyFormat("SomeClass::Constructor() :\n"
               "aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa),\n"
               "aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa) {}",
               Style);
}

TEST_F(FormatTest, BreakConstructorInitializersBeforeComma) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeComma;
  Style.ConstructorInitializerIndentWidth = 4;
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "    , b(b)\n"
               "    , c(c) {}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a) {}",
               Style);

  Style.ColumnLimit = 0;
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a) {}",
               Style);
  verifyFormat("SomeClass::Constructor() noexcept\n"
               "    : a(a) {}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "    , b(b)\n"
               "    , c(c) {}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a) {\n"
               "  foo();\n"
               "  bar();\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "    , b(b)\n"
               "    , c(c) {\n}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a) {\n}",
               Style);

  Style.ColumnLimit = 80;
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  Style.ConstructorInitializerIndentWidth = 2;
  verifyFormat("SomeClass::Constructor()\n"
               "  : a(a)\n"
               "  , b(b)\n"
               "  , c(c) {}",
               Style);

  Style.ConstructorInitializerIndentWidth = 0;
  verifyFormat("SomeClass::Constructor()\n"
               ": a(a)\n"
               ", b(b)\n"
               ", c(c) {}",
               Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  Style.ConstructorInitializerIndentWidth = 4;
  verifyFormat("SomeClass::Constructor() : aaaaaaaa(aaaaaaaa) {}", Style);
  verifyFormat(
      "SomeClass::Constructor() : aaaaa(aaaaa), aaaaa(aaaaa), aaaaa(aaaaa)",
      Style);
  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaa(aaaaaaaa), aaaaaaaa(aaaaaaaa), aaaaaaaa(aaaaaaaa) {}",
      Style);
  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaaaaa(aaaaaaaa) {}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaa(aaaaa), aaaaa(aaaaa), aaaaa(aaaaa)",
               Style);
  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaa(aaaaaaaa), aaaaaaaa(aaaaaaaa), aaaaaaaa(aaaaaaaa) {}",
      Style);

  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLine;
  Style.ConstructorInitializerIndentWidth = 4;
  Style.ColumnLimit = 60;
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaaaaa(aaaaaaaa)\n"
               "    , aaaaaaaa(aaaaaaaa)\n"
               "    , aaaaaaaa(aaaaaaaa) {}",
               Style);
  Style.PackConstructorInitializers = FormatStyle::PCIS_NextLineOnly;
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaaaaa(aaaaaaaa)\n"
               "    , aaaaaaaa(aaaaaaaa)\n"
               "    , aaaaaaaa(aaaaaaaa) {}",
               Style);
}

TEST_F(FormatTest, ConstructorInitializersWithPreprocessorDirective) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeComma;
  Style.ConstructorInitializerIndentWidth = 4;
  verifyFormat("SomeClass::Constructor()\n"
               "    : a{a}\n"
               "    , b{b} {}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a{a}\n"
               "#if CONDITION\n"
               "    , b{b}\n"
               "#endif\n"
               "{\n}",
               Style);
  Style.ConstructorInitializerIndentWidth = 2;
  verifyFormat("SomeClass::Constructor()\n"
               "#if CONDITION\n"
               "  : a{a}\n"
               "#endif\n"
               "  , b{b}\n"
               "  , c{c} {\n}",
               Style);
  Style.ConstructorInitializerIndentWidth = 0;
  verifyFormat("SomeClass::Constructor()\n"
               ": a{a}\n"
               "#ifdef CONDITION\n"
               ", b{b}\n"
               "#else\n"
               ", c{c}\n"
               "#endif\n"
               ", d{d} {\n}",
               Style);
  Style.ConstructorInitializerIndentWidth = 4;
  verifyFormat("SomeClass::Constructor()\n"
               "    : a{a}\n"
               "#if WINDOWS\n"
               "#if DEBUG\n"
               "    , b{0}\n"
               "#else\n"
               "    , b{1}\n"
               "#endif\n"
               "#else\n"
               "#if DEBUG\n"
               "    , b{2}\n"
               "#else\n"
               "    , b{3}\n"
               "#endif\n"
               "#endif\n"
               "{\n}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a{a}\n"
               "#if WINDOWS\n"
               "    , b{0}\n"
               "#if DEBUG\n"
               "    , c{0}\n"
               "#else\n"
               "    , c{1}\n"
               "#endif\n"
               "#else\n"
               "#if DEBUG\n"
               "    , c{2}\n"
               "#else\n"
               "    , c{3}\n"
               "#endif\n"
               "    , b{1}\n"
               "#endif\n"
               "{\n}",
               Style);
}

TEST_F(FormatTest, Destructors) {
  verifyFormat("void F(int &i) { i.~int(); }");
  verifyFormat("void F(int &i) { i->~int(); }");
}

TEST_F(FormatTest, FormatsWithWebKitStyle) {
  FormatStyle Style = getWebKitStyle();

  // Don't indent in outer namespaces.
  verifyFormat("namespace outer {\n"
               "int i;\n"
               "namespace inner {\n"
               "    int i;\n"
               "} // namespace inner\n"
               "} // namespace outer\n"
               "namespace other_outer {\n"
               "int i;\n"
               "}",
               Style);

  // Don't indent case labels.
  verifyFormat("switch (variable) {\n"
               "case 1:\n"
               "case 2:\n"
               "    doSomething();\n"
               "    break;\n"
               "default:\n"
               "    ++variable;\n"
               "}",
               Style);

  // Wrap before binary operators.
  verifyFormat(
      "void f()\n"
      "{\n"
      "    if (aaaaaaaaaaaaaaaa\n"
      "        && bbbbbbbbbbbbbbbbbbbbbbbb\n"
      "        && (cccccccccccccccccccccccccc || dddddddddddddddddddd))\n"
      "        return;\n"
      "}",
      "void f() {\n"
      "if (aaaaaaaaaaaaaaaa\n"
      "&& bbbbbbbbbbbbbbbbbbbbbbbb\n"
      "&& (cccccccccccccccccccccccccc || dddddddddddddddddddd))\n"
      "return;\n"
      "}",
      Style);

  // Allow functions on a single line.
  verifyFormat("void f() { return; }", Style);

  // Allow empty blocks on a single line and insert a space in empty blocks.
  verifyFormat("void f() { }", "void f() {}", Style);
  verifyFormat("while (true) { }", "while (true) {}", Style);
  // However, don't merge non-empty short loops.
  verifyFormat("while (true) {\n"
               "    continue;\n"
               "}",
               "while (true) { continue; }", Style);

  // Constructor initializers are formatted one per line with the "," on the
  // new line.
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    , aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaa, // break\n"
               "          aaaaaaaaaaaaaa)\n"
               "    , aaaaaaaaaaaaaaaaaaaaaaa()\n"
               "{\n"
               "}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "{\n"
               "}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "{\n"
               "}",
               "SomeClass::Constructor():a(a){}", Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "    , b(b)\n"
               "    , c(c)\n"
               "{\n"
               "}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "{\n"
               "    foo();\n"
               "    bar();\n"
               "}",
               Style);

  // Access specifiers should be aligned left.
  verifyFormat("class C {\n"
               "public:\n"
               "    int i;\n"
               "};",
               Style);

  // Do not align comments.
  verifyFormat("int a; // Do not\n"
               "double b; // align comments.",
               Style);

  // Do not align operands.
  verifyFormat("ASSERT(aaaa\n"
               "    || bbbb);",
               "ASSERT ( aaaa\n||bbbb);", Style);

  // Accept input's line breaks.
  verifyFormat("if (aaaaaaaaaaaaaaa\n"
               "    || bbbbbbbbbbbbbbb) {\n"
               "    i++;\n"
               "}",
               "if (aaaaaaaaaaaaaaa\n"
               "|| bbbbbbbbbbbbbbb) { i++; }",
               Style);
  verifyFormat("if (aaaaaaaaaaaaaaa || bbbbbbbbbbbbbbb) {\n"
               "    i++;\n"
               "}",
               "if (aaaaaaaaaaaaaaa || bbbbbbbbbbbbbbb) { i++; }", Style);

  // Don't automatically break all macro definitions (llvm.org/PR17842).
  verifyFormat("#define aNumber 10", Style);
  // However, generally keep the line breaks that the user authored.
  verifyFormat("#define aNumber \\\n"
               "    10",
               "#define aNumber \\\n"
               " 10",
               Style);

  // Keep empty and one-element array literals on a single line.
  verifyFormat("NSArray* a = [[NSArray alloc] initWithArray:@[]\n"
               "                                  copyItems:YES];",
               "NSArray*a=[[NSArray alloc] initWithArray:@[]\n"
               "copyItems:YES];",
               Style);
  verifyFormat("NSArray* a = [[NSArray alloc] initWithArray:@[ @\"a\" ]\n"
               "                                  copyItems:YES];",
               "NSArray*a=[[NSArray alloc]initWithArray:@[ @\"a\" ]\n"
               "             copyItems:YES];",
               Style);
  // FIXME: This does not seem right, there should be more indentation before
  // the array literal's entries. Nested blocks have the same problem.
  verifyFormat("NSArray* a = [[NSArray alloc] initWithArray:@[\n"
               "    @\"a\",\n"
               "    @\"a\"\n"
               "]\n"
               "                                  copyItems:YES];",
               "NSArray* a = [[NSArray alloc] initWithArray:@[\n"
               "     @\"a\",\n"
               "     @\"a\"\n"
               "     ]\n"
               "       copyItems:YES];",
               Style);
  verifyFormat(
      "NSArray* a = [[NSArray alloc] initWithArray:@[ @\"a\", @\"a\" ]\n"
      "                                  copyItems:YES];",
      "NSArray* a = [[NSArray alloc] initWithArray:@[ @\"a\", @\"a\" ]\n"
      "   copyItems:YES];",
      Style);

  verifyFormat("[self.a b:c c:d];", Style);
  verifyFormat("[self.a b:c\n"
               "        c:d];",
               "[self.a b:c\n"
               "c:d];",
               Style);
}

TEST_F(FormatTest, FormatsLambdas) {
  verifyFormat("int c = [b]() mutable { return [&b] { return b++; }(); }();");
  verifyFormat(
      "int c = [b]() mutable noexcept { return [&b] { return b++; }(); }();");
  verifyFormat("int c = [&] { [=] { return b++; }(); }();");
  verifyFormat("int c = [&, &a, a] { [=, c, &d] { return b++; }(); }();");
  verifyFormat("int c = [&a, &a, a] { [=, a, b, &c] { return b++; }(); }();");
  verifyFormat("auto c = {[&a, &a, a] { [=, a, b, &c] { return b++; }(); }}");
  verifyFormat("auto c = {[&a, &a, a] { [=, a, b, &c] {}(); }}");
  verifyFormat("auto c = [a = [b = 42] {}] {};");
  verifyFormat("auto c = [a = &i + 10, b = [] {}] {};");
  verifyFormat("int x = f(*+[] {});");
  verifyFormat("void f() {\n"
               "  other(x.begin(), x.end(), [&](int, int) { return 1; });\n"
               "}");
  verifyFormat("void f() {\n"
               "  other(x.begin(), //\n"
               "        x.end(),   //\n"
               "        [&](int, int) { return 1; });\n"
               "}");
  verifyFormat("void f() {\n"
               "  other.other.other.other.other(\n"
               "      x.begin(), x.end(),\n"
               "      [something, rather](int, int, int, int, int, int, int) { "
               "return 1; });\n"
               "}");
  verifyFormat(
      "void f() {\n"
      "  other.other.other.other.other(\n"
      "      x.begin(), x.end(),\n"
      "      [something, rather](int, int, int, int, int, int, int) {\n"
      "        //\n"
      "      });\n"
      "}");
  verifyFormat("SomeFunction([]() { // A cool function...\n"
               "  return 43;\n"
               "});");
  verifyFormat("SomeFunction([]() {\n"
               "#define A a\n"
               "  return 43;\n"
               "});",
               "SomeFunction([](){\n"
               "#define A a\n"
               "return 43;\n"
               "});");
  verifyFormat("void f() {\n"
               "  SomeFunction([](decltype(x), A *a) {});\n"
               "  SomeFunction([](typeof(x), A *a) {});\n"
               "  SomeFunction([](_Atomic(x), A *a) {});\n"
               "  SomeFunction([](__underlying_type(x), A *a) {});\n"
               "}");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    [](const aaaaaaaaaa &a) { return a; });");
  verifyFormat("string abc = SomeFunction(aaaaaaaaaaaaa, aaaaa, []() {\n"
               "  SomeOtherFunctioooooooooooooooooooooooooon();\n"
               "});");
  verifyFormat("Constructor()\n"
               "    : Field([] { // comment\n"
               "        int i;\n"
               "      }) {}");
  verifyFormat("auto my_lambda = [](const string &some_parameter) {\n"
               "  return some_parameter.size();\n"
               "};");
  verifyFormat("std::function<std::string(const std::string &)> my_lambda =\n"
               "    [](const string &s) { return s; };");
  verifyFormat("int i = aaaaaa ? 1 //\n"
               "               : [] {\n"
               "                   return 2; //\n"
               "                 }();");
  verifyFormat("llvm::errs() << \"number of twos is \"\n"
               "             << std::count_if(v.begin(), v.end(), [](int x) {\n"
               "                  return x == 2; // force break\n"
               "                });");
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    [=](int iiiiiiiiiiii) {\n"
               "      return aaaaaaaaaaaaaaaaaaaaaaa !=\n"
               "             aaaaaaaaaaaaaaaaaaaaaaa;\n"
               "    });",
               getLLVMStyleWithColumns(60));

  verifyFormat("SomeFunction({[&] {\n"
               "                // comment\n"
               "              },\n"
               "              [&] {\n"
               "                // comment\n"
               "              }});");
  verifyFormat("SomeFunction({[&] {\n"
               "  // comment\n"
               "}});");
  verifyFormat(
      "virtual aaaaaaaaaaaaaaaa(\n"
      "    std::function<bool()> bbbbbbbbbbbb = [&]() { return true; },\n"
      "    aaaaa aaaaaaaaa);");

  // Lambdas with return types.
  verifyFormat("int c = []() -> int { return 2; }();");
  verifyFormat("int c = []() -> int * { return 2; }();");
  verifyFormat("int c = []() -> vector<int> { return {2}; }();");
  verifyFormat("Foo([]() -> std::vector<int> { return {2}; }());");
  verifyFormat("foo([]() noexcept -> int {});");
  verifyGoogleFormat("auto a = [&b, c](D* d) -> D* {};");
  verifyGoogleFormat("auto a = [&b, c](D* d) -> pair<D*, D*> {};");
  verifyGoogleFormat("auto a = [&b, c](D* d) -> D& {};");
  verifyGoogleFormat("auto a = [&b, c](D* d) -> const D* {};");
  verifyFormat("[a, a]() -> a<1> {};");
  verifyFormat("[]() -> foo<5 + 2> { return {}; };");
  verifyFormat("[]() -> foo<5 - 2> { return {}; };");
  verifyFormat("[]() -> foo<5 / 2> { return {}; };");
  verifyFormat("[]() -> foo<5 * 2> { return {}; };");
  verifyFormat("[]() -> foo<5 % 2> { return {}; };");
  verifyFormat("[]() -> foo<5 << 2> { return {}; };");
  verifyFormat("[]() -> foo<!5> { return {}; };");
  verifyFormat("[]() -> foo<~5> { return {}; };");
  verifyFormat("[]() -> foo<5 | 2> { return {}; };");
  verifyFormat("[]() -> foo<5 || 2> { return {}; };");
  verifyFormat("[]() -> foo<5 & 2> { return {}; };");
  verifyFormat("[]() -> foo<5 && 2> { return {}; };");
  verifyFormat("[]() -> foo<5 == 2> { return {}; };");
  verifyFormat("[]() -> foo<5 != 2> { return {}; };");
  verifyFormat("[]() -> foo<5 >= 2> { return {}; };");
  verifyFormat("[]() -> foo<5 <= 2> { return {}; };");
  verifyFormat("[]() -> foo<5 < 2> { return {}; };");
  verifyFormat("[]() -> foo<2 ? 1 : 0> { return {}; };");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 + 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 - 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 / 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 * 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 % 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 << 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<!5> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<~5> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 | 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 || 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 & 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 && 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 == 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 != 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 >= 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 <= 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<5 < 2> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("namespace bar {\n"
               "// broken:\n"
               "auto foo{[]() -> foo<2 ? 1 : 0> { return {}; }};\n"
               "} // namespace bar");
  verifyFormat("[]() -> a<1> {};");
  verifyFormat("[]() -> a<1> { ; };");
  verifyFormat("[]() -> a<1> { ; }();");
  verifyFormat("[a, a]() -> a<true> {};");
  verifyFormat("[]() -> a<true> {};");
  verifyFormat("[]() -> a<true> { ; };");
  verifyFormat("[]() -> a<true> { ; }();");
  verifyFormat("[a, a]() -> a<false> {};");
  verifyFormat("[]() -> a<false> {};");
  verifyFormat("[]() -> a<false> { ; };");
  verifyFormat("[]() -> a<false> { ; }();");
  verifyFormat("auto foo{[]() -> foo<false> { ; }};");
  verifyFormat("namespace bar {\n"
               "auto foo{[]() -> foo<false> { ; }};\n"
               "} // namespace bar");
  verifyFormat("auto aaaaaaaa = [](int i, // break for some reason\n"
               "                   int j) -> int {\n"
               "  return ffffffffffffffffffffffffffffffffffffffffffff(i * j);\n"
               "};");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaa(\n"
      "    [](aaaaaaaaaaaaaaaaaaaaaaaaaaa &aaa) -> aaaaaaaaaaaaaaaa {\n"
      "      return aaaaaaaaaaaaaaaaa;\n"
      "    });",
      getLLVMStyleWithColumns(70));
  verifyFormat("[]() //\n"
               "    -> int {\n"
               "  return 1; //\n"
               "};");
  verifyFormat("[]() -> Void<T...> {};");
  verifyFormat("[a, b]() -> Tuple<T...> { return {}; };");
  verifyFormat("SomeFunction({[]() -> int[] { return {}; }});");
  verifyFormat("SomeFunction({[]() -> int *[] { return {}; }});");
  verifyFormat("SomeFunction({[]() -> int (*)[] { return {}; }});");
  verifyFormat("SomeFunction({[]() -> ns::type<int (*)[]> { return {}; }});");
  verifyFormat("return int{[x = x]() { return x; }()};");

  // Lambdas with explicit template argument lists.
  verifyFormat(
      "auto L = []<template <typename> class T, class U>(T<U> &&a) {};");
  verifyFormat("auto L = []<class T>(T) {\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "};");
  verifyFormat("auto L = []<class... T>(T...) {\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "};");
  verifyFormat("auto L = []<typename... T>(T...) {\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "};");
  verifyFormat("auto L = []<template <typename...> class T>(T...) {\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "};");
  verifyFormat("auto L = []</*comment*/ class... T>(T...) {\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "};");
  verifyFormat("auto L = []<int... T>(T...) {\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "};");
  verifyFormat("auto L = []<Foo... T>(T...) {\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "};");

  // Lambdas that fit on a single line within an argument list are not forced
  // onto new lines.
  verifyFormat("SomeFunction([] {});");
  verifyFormat("SomeFunction(0, [] {});");
  verifyFormat("SomeFunction([] {}, 0);");
  verifyFormat("SomeFunction(0, [] {}, 0);");
  verifyFormat("SomeFunction([] { return 0; }, 0);");
  verifyFormat("SomeFunction(a, [] { return 0; }, b);");
  verifyFormat("SomeFunction([] { return 0; }, [] { return 0; });");
  verifyFormat("SomeFunction([] { return 0; }, [] { return 0; }, b);");
  verifyFormat("auto loooooooooooooooooooooooooooong =\n"
               "    SomeFunction([] { return 0; }, [] { return 0; }, b);");
  // Exceeded column limit. We need to break.
  verifyFormat("auto loooooooooooooooooooooooooooongName = SomeFunction(\n"
               "    [] { return anotherLooooooooooonoooooooongName; }, [] { "
               "return 0; }, b);");

  // Multiple multi-line lambdas in the same parentheses change indentation
  // rules. These lambdas are always forced to start on new lines.
  verifyFormat("SomeFunction(\n"
               "    []() {\n"
               "      //\n"
               "    },\n"
               "    []() {\n"
               "      //\n"
               "    });");

  // A multi-line lambda passed as arg0 is always pushed to the next line.
  verifyFormat("SomeFunction(\n"
               "    [this] {\n"
               "      //\n"
               "    },\n"
               "    1);");

  // A multi-line lambda passed as arg1 forces arg0 to be pushed out, just like
  // the arg0 case above.
  auto Style = getGoogleStyle();
  Style.BinPackArguments = false;
  verifyFormat("SomeFunction(\n"
               "    a,\n"
               "    [this] {\n"
               "      //\n"
               "    },\n"
               "    b);",
               Style);
  verifyFormat("SomeFunction(\n"
               "    a,\n"
               "    [this] {\n"
               "      //\n"
               "    },\n"
               "    b);");

  // A lambda with a very long line forces arg0 to be pushed out irrespective of
  // the BinPackArguments value (as long as the code is wide enough).
  verifyFormat(
      "something->SomeFunction(\n"
      "    a,\n"
      "    [this] {\n"
      "      "
      "D0000000000000000000000000000000000000000000000000000000000001();\n"
      "    },\n"
      "    b);");

  // A multi-line lambda is pulled up as long as the introducer fits on the
  // previous line and there are no further args.
  verifyFormat("function(1, [this, that] {\n"
               "  //\n"
               "});");
  verifyFormat("function([this, that] {\n"
               "  //\n"
               "});");
  // FIXME: this format is not ideal and we should consider forcing the first
  // arg onto its own line.
  verifyFormat("function(a, b, c, //\n"
               "         d, [this, that] {\n"
               "           //\n"
               "         });");

  // Multiple lambdas are treated correctly even when there is a short arg0.
  verifyFormat("SomeFunction(\n"
               "    1,\n"
               "    [this] {\n"
               "      //\n"
               "    },\n"
               "    [this] {\n"
               "      //\n"
               "    },\n"
               "    1);");

  // More complex introducers.
  verifyFormat("return [i, args...] {};");

  // Not lambdas.
  verifyFormat("constexpr char hello[]{\"hello\"};");
  verifyFormat("double &operator[](int i) { return 0; }\n"
               "int i;");
  verifyFormat("std::unique_ptr<int[]> foo() {}");
  verifyFormat("int i = a[a][a]->f();");
  verifyFormat("int i = (*b)[a]->f();");

  // Other corner cases.
  verifyFormat("void f() {\n"
               "  bar([]() {} // Did not respect SpacesBeforeTrailingComments\n"
               "  );\n"
               "}");
  verifyFormat("auto k = *[](int *j) { return j; }(&i);");

  // Lambdas created through weird macros.
  verifyFormat("void f() {\n"
               "  MACRO((const AA &a) { return 1; });\n"
               "  MACRO((AA &a) { return 1; });\n"
               "}");

  verifyFormat("if (blah_blah(whatever, whatever, [] {\n"
               "      doo_dah();\n"
               "      doo_dah();\n"
               "    })) {\n"
               "}");
  verifyFormat("if constexpr (blah_blah(whatever, whatever, [] {\n"
               "                doo_dah();\n"
               "                doo_dah();\n"
               "              })) {\n"
               "}");
  verifyFormat("if CONSTEXPR (blah_blah(whatever, whatever, [] {\n"
               "                doo_dah();\n"
               "                doo_dah();\n"
               "              })) {\n"
               "}");
  verifyFormat("auto lambda = []() {\n"
               "  int a = 2\n"
               "#if A\n"
               "          + 2\n"
               "#endif\n"
               "      ;\n"
               "};");

  // Lambdas with complex multiline introducers.
  verifyFormat(
      "aaaaaaaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]()\n"
      "        -> ::std::unordered_set<\n"
      "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa> {\n"
      "      //\n"
      "    });");

  FormatStyle DoNotMerge = getLLVMStyle();
  DoNotMerge.AllowShortLambdasOnASingleLine = FormatStyle::SLS_None;
  verifyFormat("auto c = []() {\n"
               "  return b;\n"
               "};",
               "auto c = []() { return b; };", DoNotMerge);
  verifyFormat("auto c = []() {\n"
               "};",
               " auto c = []() {};", DoNotMerge);

  FormatStyle MergeEmptyOnly = getLLVMStyle();
  MergeEmptyOnly.AllowShortLambdasOnASingleLine = FormatStyle::SLS_Empty;
  verifyFormat("auto c = []() {\n"
               "  return b;\n"
               "};",
               "auto c = []() {\n"
               "  return b;\n"
               " };",
               MergeEmptyOnly);
  verifyFormat("auto c = []() {};",
               "auto c = []() {\n"
               "};",
               MergeEmptyOnly);

  FormatStyle MergeInline = getLLVMStyle();
  MergeInline.AllowShortLambdasOnASingleLine = FormatStyle::SLS_Inline;
  verifyFormat("auto c = []() {\n"
               "  return b;\n"
               "};",
               "auto c = []() { return b; };", MergeInline);
  verifyFormat("function([]() { return b; })", MergeInline);
  verifyFormat("function([]() { return b; }, a)", MergeInline);
  verifyFormat("function(a, []() { return b; })", MergeInline);

  // Check option "BraceWrapping.BeforeLambdaBody" and different state of
  // AllowShortLambdasOnASingleLine
  FormatStyle LLVMWithBeforeLambdaBody = getLLVMStyle();
  LLVMWithBeforeLambdaBody.BreakBeforeBraces = FormatStyle::BS_Custom;
  LLVMWithBeforeLambdaBody.BraceWrapping.BeforeLambdaBody = true;
  LLVMWithBeforeLambdaBody.AllowShortLambdasOnASingleLine =
      FormatStyle::ShortLambdaStyle::SLS_None;
  verifyFormat("FctWithOneNestedLambdaInline_SLS_None(\n"
               "    []()\n"
               "    {\n"
               "      return 17;\n"
               "    });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithOneNestedLambdaEmpty_SLS_None(\n"
               "    []()\n"
               "    {\n"
               "    });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("auto fct_SLS_None = []()\n"
               "{\n"
               "  return 17;\n"
               "};",
               LLVMWithBeforeLambdaBody);
  verifyFormat("TwoNestedLambdas_SLS_None(\n"
               "    []()\n"
               "    {\n"
               "      return Call(\n"
               "          []()\n"
               "          {\n"
               "            return 17;\n"
               "          });\n"
               "    });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("void Fct() {\n"
               "  return {[]()\n"
               "          {\n"
               "            return 17;\n"
               "          }};\n"
               "}",
               LLVMWithBeforeLambdaBody);

  LLVMWithBeforeLambdaBody.AllowShortLambdasOnASingleLine =
      FormatStyle::ShortLambdaStyle::SLS_Empty;
  verifyFormat("FctWithOneNestedLambdaInline_SLS_Empty(\n"
               "    []()\n"
               "    {\n"
               "      return 17;\n"
               "    });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithOneNestedLambdaEmpty_SLS_Empty([]() {});",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithOneNestedLambdaEmptyInsideAVeryVeryVeryVeryVeryVeryVeryL"
               "ongFunctionName_SLS_Empty(\n"
               "    []() {});",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithMultipleParams_SLS_Empty(A, B,\n"
               "                                []()\n"
               "                                {\n"
               "                                  return 17;\n"
               "                                });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("auto fct_SLS_Empty = []()\n"
               "{\n"
               "  return 17;\n"
               "};",
               LLVMWithBeforeLambdaBody);
  verifyFormat("TwoNestedLambdas_SLS_Empty(\n"
               "    []()\n"
               "    {\n"
               "      return Call([]() {});\n"
               "    });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("TwoNestedLambdas_SLS_Empty(A,\n"
               "                           []()\n"
               "                           {\n"
               "                             return Call([]() {});\n"
               "                           });",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "FctWithLongLineInLambda_SLS_Empty(\n"
      "    []()\n"
      "    {\n"
      "      return HereAVeryLongLine(ThatWillBeFormatted, OnMultipleLine,\n"
      "                               AndShouldNotBeConsiderAsInline,\n"
      "                               LambdaBodyMustBeBreak);\n"
      "    });",
      LLVMWithBeforeLambdaBody);

  LLVMWithBeforeLambdaBody.AllowShortLambdasOnASingleLine =
      FormatStyle::ShortLambdaStyle::SLS_Inline;
  verifyFormat("FctWithOneNestedLambdaInline_SLS_Inline([]() { return 17; });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithOneNestedLambdaEmpty_SLS_Inline([]() {});",
               LLVMWithBeforeLambdaBody);
  verifyFormat("auto fct_SLS_Inline = []()\n"
               "{\n"
               "  return 17;\n"
               "};",
               LLVMWithBeforeLambdaBody);
  verifyFormat("TwoNestedLambdas_SLS_Inline([]() { return Call([]() { return "
               "17; }); });",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "FctWithLongLineInLambda_SLS_Inline(\n"
      "    []()\n"
      "    {\n"
      "      return HereAVeryLongLine(ThatWillBeFormatted, OnMultipleLine,\n"
      "                               AndShouldNotBeConsiderAsInline,\n"
      "                               LambdaBodyMustBeBreak);\n"
      "    });",
      LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithMultipleParams_SLS_Inline("
               "VeryLongParameterThatShouldAskToBeOnMultiLine,\n"
               "                                 []() { return 17; });",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "FctWithMultipleParams_SLS_Inline(FirstParam, []() { return 17; });",
      LLVMWithBeforeLambdaBody);

  LLVMWithBeforeLambdaBody.AllowShortLambdasOnASingleLine =
      FormatStyle::ShortLambdaStyle::SLS_All;
  verifyFormat("FctWithOneNestedLambdaInline_SLS_All([]() { return 17; });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithOneNestedLambdaEmpty_SLS_All([]() {});",
               LLVMWithBeforeLambdaBody);
  verifyFormat("auto fct_SLS_All = []() { return 17; };",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithOneParam_SLS_All(\n"
               "    []()\n"
               "    {\n"
               "      // A cool function...\n"
               "      return 43;\n"
               "    });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithMultipleParams_SLS_All("
               "VeryLongParameterThatShouldAskToBeOnMultiLine,\n"
               "                              []() { return 17; });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithMultipleParams_SLS_All(A, []() { return 17; });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithMultipleParams_SLS_All(A, B, []() { return 17; });",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "FctWithLongLineInLambda_SLS_All(\n"
      "    []()\n"
      "    {\n"
      "      return HereAVeryLongLine(ThatWillBeFormatted, OnMultipleLine,\n"
      "                               AndShouldNotBeConsiderAsInline,\n"
      "                               LambdaBodyMustBeBreak);\n"
      "    });",
      LLVMWithBeforeLambdaBody);
  verifyFormat(
      "auto fct_SLS_All = []()\n"
      "{\n"
      "  return HereAVeryLongLine(ThatWillBeFormatted, OnMultipleLine,\n"
      "                           AndShouldNotBeConsiderAsInline,\n"
      "                           LambdaBodyMustBeBreak);\n"
      "};",
      LLVMWithBeforeLambdaBody);
  LLVMWithBeforeLambdaBody.BinPackParameters = false;
  verifyFormat("FctAllOnSameLine_SLS_All([]() { return S; }, Fst, Second);",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "FctWithLongLineInLambda_SLS_All([]() { return SomeValueNotSoLong; },\n"
      "                                FirstParam,\n"
      "                                SecondParam,\n"
      "                                ThirdParam,\n"
      "                                FourthParam);",
      LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithLongLineInLambda_SLS_All(\n"
               "    []() { return "
               "SomeValueVeryVeryVeryVeryVeryVeryVeryVeryVeryLong; },\n"
               "    FirstParam,\n"
               "    SecondParam,\n"
               "    ThirdParam,\n"
               "    FourthParam);",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "FctWithLongLineInLambda_SLS_All(FirstParam,\n"
      "                                SecondParam,\n"
      "                                ThirdParam,\n"
      "                                FourthParam,\n"
      "                                []() { return SomeValueNotSoLong; });",
      LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithLongLineInLambda_SLS_All(\n"
               "    []()\n"
               "    {\n"
               "      return "
               "HereAVeryLongLineThatWillBeFormattedOnMultipleLineAndShouldNotB"
               "eConsiderAsInline;\n"
               "    });",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "FctWithLongLineInLambda_SLS_All(\n"
      "    []()\n"
      "    {\n"
      "      return HereAVeryLongLine(ThatWillBeFormatted, OnMultipleLine,\n"
      "                               AndShouldNotBeConsiderAsInline,\n"
      "                               LambdaBodyMustBeBreak);\n"
      "    });",
      LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithTwoParams_SLS_All(\n"
               "    []()\n"
               "    {\n"
               "      // A cool function...\n"
               "      return 43;\n"
               "    },\n"
               "    87);",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithTwoParams_SLS_All([]() { return 43; }, 87);",
               LLVMWithBeforeLambdaBody);
  verifyFormat("FctWithOneNestedLambdas_SLS_All([]() { return 17; });",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "TwoNestedLambdas_SLS_All([]() { return Call([]() { return 17; }); });",
      LLVMWithBeforeLambdaBody);
  verifyFormat("TwoNestedLambdas_SLS_All([]() { return Call([]() { return 17; "
               "}); }, x);",
               LLVMWithBeforeLambdaBody);
  verifyFormat("TwoNestedLambdas_SLS_All(\n"
               "    []()\n"
               "    {\n"
               "      // A cool function...\n"
               "      return Call([]() { return 17; });\n"
               "    });",
               LLVMWithBeforeLambdaBody);
  verifyFormat("TwoNestedLambdas_SLS_All(\n"
               "    []()\n"
               "    {\n"
               "      return Call(\n"
               "          []()\n"
               "          {\n"
               "            // A cool function...\n"
               "            return 17;\n"
               "          });\n"
               "    });",
               LLVMWithBeforeLambdaBody);

  LLVMWithBeforeLambdaBody.AllowShortLambdasOnASingleLine =
      FormatStyle::ShortLambdaStyle::SLS_None;

  verifyFormat("auto select = [this]() -> const Library::Object *\n"
               "{\n"
               "  return MyAssignment::SelectFromList(this);\n"
               "};",
               LLVMWithBeforeLambdaBody);

  verifyFormat("auto select = [this]() -> const Library::Object &\n"
               "{\n"
               "  return MyAssignment::SelectFromList(this);\n"
               "};",
               LLVMWithBeforeLambdaBody);

  verifyFormat("auto select = [this]() -> std::unique_ptr<Object>\n"
               "{\n"
               "  return MyAssignment::SelectFromList(this);\n"
               "};",
               LLVMWithBeforeLambdaBody);

  verifyFormat("namespace test {\n"
               "class Test {\n"
               "public:\n"
               "  Test() = default;\n"
               "};\n"
               "} // namespace test",
               LLVMWithBeforeLambdaBody);

  // Lambdas with different indentation styles.
  Style = getLLVMStyleWithColumns(60);
  verifyFormat("Result doSomething(Promise promise) {\n"
               "  return promise.then(\n"
               "      [this, obj = std::move(s)](int bar) mutable {\n"
               "        return someObject.startAsyncAction().then(\n"
               "            [this, &obj](Result result) mutable {\n"
               "              result.processMore();\n"
               "            });\n"
               "      });\n"
               "}",
               Style);
  Style.LambdaBodyIndentation = FormatStyle::LBI_OuterScope;
  verifyFormat("Result doSomething(Promise promise) {\n"
               "  return promise.then(\n"
               "      [this, obj = std::move(s)](int bar) mutable {\n"
               "    return obj.startAsyncAction().then(\n"
               "        [this, &obj](Result result) mutable {\n"
               "      result.processMore();\n"
               "    });\n"
               "  });\n"
               "}",
               Style);
  verifyFormat("Result doSomething(Promise promise) {\n"
               "  return promise.then([this, obj = std::move(s)] {\n"
               "    return obj.startAsyncAction().then(\n"
               "        [this, &obj](Result result) mutable {\n"
               "      result.processMore();\n"
               "    });\n"
               "  });\n"
               "}",
               Style);
  verifyFormat("void test() {\n"
               "  ([]() -> auto {\n"
               "    int b = 32;\n"
               "    return 3;\n"
               "  }).foo();\n"
               "}",
               Style);
  verifyFormat("void test() {\n"
               "  []() -> auto {\n"
               "    int b = 32;\n"
               "    return 3;\n"
               "  }\n"
               "}",
               Style);
  verifyFormat("std::sort(v.begin(), v.end(),\n"
               "          [](const auto &foo, const auto &bar) {\n"
               "  return foo.baz < bar.baz;\n"
               "});",
               Style);
  verifyFormat("void test() {\n"
               "  (\n"
               "      []() -> auto {\n"
               "    int b = 32;\n"
               "    return 3;\n"
               "  }, foo, bar)\n"
               "      .foo();\n"
               "}",
               Style);
  verifyFormat("void test() {\n"
               "  ([]() -> auto {\n"
               "    int b = 32;\n"
               "    return 3;\n"
               "  })\n"
               "      .foo()\n"
               "      .bar();\n"
               "}",
               Style);
  verifyFormat("#define A                                                  \\\n"
               "  [] {                                                     \\\n"
               "    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(                   \\\n"
               "        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx);            \\\n"
               "  }",
               Style);
  verifyFormat("void foo() {\n"
               "  aFunction(1, b(c(foo, bar, baz, [](d) {\n"
               "    auto f = e(d);\n"
               "    return f;\n"
               "  })));\n"
               "}",
               Style);
  verifyFormat("void foo() {\n"
               "  aFunction(1, b(c(foo, Bar{}, baz, [](d) -> Foo {\n"
               "    auto f = e(foo, [&] {\n"
               "      auto g = h();\n"
               "      return g;\n"
               "    }, qux, [&] -> Bar {\n"
               "      auto i = j();\n"
               "      return i;\n"
               "    });\n"
               "    return f;\n"
               "  })));\n"
               "}",
               Style);
  verifyFormat("Namespace::Foo::Foo(LongClassName bar,\n"
               "                    AnotherLongClassName baz)\n"
               "    : baz{baz}, func{[&] {\n"
               "  auto qux = bar;\n"
               "  return aFunkyFunctionCall(qux);\n"
               "}} {}",
               Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  // FIXME: The following test should pass, but fails at the time of writing.
#if 0
  // As long as all the non-lambda arguments fit on a single line, AlwaysBreak
  // doesn't force an initial line break, even if lambdas span multiple lines.
  verifyFormat("void foo() {\n"
               "  aFunction(\n"
               "      [](d) -> Foo {\n"
               "    auto f = e(d);\n"
               "    return f;\n"
               "  }, foo, Bar{}, [] {\n"
               "    auto g = h();\n"
               "    return g;\n"
               "  }, baz);\n"
               "}",
               Style);
#endif
  // A long non-lambda argument forces arguments to span multiple lines and thus
  // forces an initial line break when using AlwaysBreak.
  verifyFormat("void foo() {\n"
               "  aFunction(\n"
               "      1,\n"
               "      [](d) -> Foo {\n"
               "    auto f = e(d);\n"
               "    return f;\n"
               "  }, foo, Bar{},\n"
               "      [] {\n"
               "    auto g = h();\n"
               "    return g;\n"
               "  }, bazzzzz,\n"
               "      quuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuux);\n"
               "}",
               Style);
  Style.BinPackArguments = false;
  verifyFormat("void foo() {\n"
               "  aFunction(\n"
               "      1,\n"
               "      [](d) -> Foo {\n"
               "    auto f = e(d);\n"
               "    return f;\n"
               "  },\n"
               "      foo,\n"
               "      Bar{},\n"
               "      [] {\n"
               "    auto g = h();\n"
               "    return g;\n"
               "  },\n"
               "      bazzzzz,\n"
               "      quuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuux);\n"
               "}",
               Style);
  Style.BinPackArguments = true;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.BeforeLambdaBody = true;
  verifyFormat("void foo() {\n"
               "  aFunction(\n"
               "      1, b(c(foo, Bar{}, baz, [](d) -> Foo\n"
               "  {\n"
               "    auto f = e(\n"
               "        [&]\n"
               "    {\n"
               "      auto g = h();\n"
               "      return g;\n"
               "    }, qux, [&] -> Bar\n"
               "    {\n"
               "      auto i = j();\n"
               "      return i;\n"
               "    });\n"
               "    return f;\n"
               "  })));\n"
               "}",
               Style);
}

TEST_F(FormatTest, LambdaWithLineComments) {
  FormatStyle LLVMWithBeforeLambdaBody = getLLVMStyle();
  LLVMWithBeforeLambdaBody.BreakBeforeBraces = FormatStyle::BS_Custom;
  LLVMWithBeforeLambdaBody.BraceWrapping.BeforeLambdaBody = true;
  LLVMWithBeforeLambdaBody.AllowShortLambdasOnASingleLine =
      FormatStyle::ShortLambdaStyle::SLS_All;

  verifyFormat("auto k = []() { return; }", LLVMWithBeforeLambdaBody);
  verifyFormat("auto k = []() // comment\n"
               "{ return; }",
               LLVMWithBeforeLambdaBody);
  verifyFormat("auto k = []() /* comment */ { return; }",
               LLVMWithBeforeLambdaBody);
  verifyFormat("auto k = []() /* comment */ /* comment */ { return; }",
               LLVMWithBeforeLambdaBody);
  verifyFormat("auto k = []() // X\n"
               "{ return; }",
               LLVMWithBeforeLambdaBody);
  verifyFormat(
      "auto k = []() // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
      "{ return; }",
      LLVMWithBeforeLambdaBody);

  LLVMWithBeforeLambdaBody.ColumnLimit = 0;

  verifyFormat("foo([]()\n"
               "    {\n"
               "      bar();    //\n"
               "      return 1; // comment\n"
               "    }());",
               "foo([]() {\n"
               "  bar(); //\n"
               "  return 1; // comment\n"
               "}());",
               LLVMWithBeforeLambdaBody);
  verifyFormat("foo(\n"
               "    1, MACRO {\n"
               "      baz();\n"
               "      bar(); // comment\n"
               "    },\n"
               "    []() {});",
               "foo(\n"
               "  1, MACRO { baz(); bar(); // comment\n"
               "  }, []() {}\n"
               ");",
               LLVMWithBeforeLambdaBody);
}

TEST_F(FormatTest, EmptyLinesInLambdas) {
  verifyFormat("auto lambda = []() {\n"
               "  x(); //\n"
               "};",
               "auto lambda = []() {\n"
               "\n"
               "  x(); //\n"
               "\n"
               "};");
}

TEST_F(FormatTest, FormatsBlocks) {
  FormatStyle ShortBlocks = getLLVMStyle();
  ShortBlocks.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Always;
  verifyFormat("int (^Block)(int, int);", ShortBlocks);
  verifyFormat("int (^Block1)(int, int) = ^(int i, int j)", ShortBlocks);
  verifyFormat("void (^block)(int) = ^(id test) { int i; };", ShortBlocks);
  verifyFormat("void (^block)(int) = ^(int test) { int i; };", ShortBlocks);
  verifyFormat("void (^block)(int) = ^id(int test) { int i; };", ShortBlocks);
  verifyFormat("void (^block)(int) = ^int(int test) { int i; };", ShortBlocks);

  verifyFormat("foo(^{ bar(); });", ShortBlocks);
  verifyFormat("foo(a, ^{ bar(); });", ShortBlocks);
  verifyFormat("{ void (^block)(Object *x); }", ShortBlocks);

  verifyFormat("[operation setCompletionBlock:^{\n"
               "  [self onOperationDone];\n"
               "}];");
  verifyFormat("int i = {[operation setCompletionBlock:^{\n"
               "  [self onOperationDone];\n"
               "}]};");
  verifyFormat("[operation setCompletionBlock:^(int *i) {\n"
               "  f();\n"
               "}];");
  verifyFormat("int a = [operation block:^int(int *i) {\n"
               "  return 1;\n"
               "}];");
  verifyFormat("[myObject doSomethingWith:arg1\n"
               "                      aaa:^int(int *a) {\n"
               "                        return 1;\n"
               "                      }\n"
               "                      bbb:f(a * bbbbbbbb)];");

  verifyFormat("[operation setCompletionBlock:^{\n"
               "  [self.delegate newDataAvailable];\n"
               "}];",
               getLLVMStyleWithColumns(60));
  verifyFormat("dispatch_async(_fileIOQueue, ^{\n"
               "  NSString *path = [self sessionFilePath];\n"
               "  if (path) {\n"
               "    // ...\n"
               "  }\n"
               "});");
  verifyFormat("[[SessionService sharedService]\n"
               "    loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
               "      if (window) {\n"
               "        [self windowDidLoad:window];\n"
               "      } else {\n"
               "        [self errorLoadingWindow];\n"
               "      }\n"
               "    }];");
  verifyFormat("void (^largeBlock)(void) = ^{\n"
               "  // ...\n"
               "};",
               getLLVMStyleWithColumns(40));
  verifyFormat("[[SessionService sharedService]\n"
               "    loadWindowWithCompletionBlock: //\n"
               "        ^(SessionWindow *window) {\n"
               "          if (window) {\n"
               "            [self windowDidLoad:window];\n"
               "          } else {\n"
               "            [self errorLoadingWindow];\n"
               "          }\n"
               "        }];",
               getLLVMStyleWithColumns(60));
  verifyFormat("[myObject doSomethingWith:arg1\n"
               "    firstBlock:^(Foo *a) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }\n"
               "    secondBlock:^(Bar *b) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }\n"
               "    thirdBlock:^Foo(Bar *b) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }];");
  verifyFormat("[myObject doSomethingWith:arg1\n"
               "               firstBlock:-1\n"
               "              secondBlock:^(Bar *b) {\n"
               "                // ...\n"
               "                int i;\n"
               "              }];");

  verifyFormat("f(^{\n"
               "  @autoreleasepool {\n"
               "    if (a) {\n"
               "      g();\n"
               "    }\n"
               "  }\n"
               "});");
  verifyFormat("Block b = ^int *(A *a, B *b) {\n"
               "};");
  verifyFormat("BOOL (^aaa)(void) = ^BOOL {\n"
               "};");

  FormatStyle FourIndent = getLLVMStyle();
  FourIndent.ObjCBlockIndentWidth = 4;
  verifyFormat("[operation setCompletionBlock:^{\n"
               "    [self onOperationDone];\n"
               "}];",
               FourIndent);
}

TEST_F(FormatTest, FormatsBlocksWithZeroColumnWidth) {
  FormatStyle ZeroColumn = getLLVMStyleWithColumns(0);

  verifyFormat("[[SessionService sharedService] "
               "loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
               "  if (window) {\n"
               "    [self windowDidLoad:window];\n"
               "  } else {\n"
               "    [self errorLoadingWindow];\n"
               "  }\n"
               "}];",
               ZeroColumn);
  verifyFormat("[[SessionService sharedService]\n"
               "    loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
               "      if (window) {\n"
               "        [self windowDidLoad:window];\n"
               "      } else {\n"
               "        [self errorLoadingWindow];\n"
               "      }\n"
               "    }];",
               "[[SessionService sharedService]\n"
               "loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
               "                if (window) {\n"
               "    [self windowDidLoad:window];\n"
               "  } else {\n"
               "    [self errorLoadingWindow];\n"
               "  }\n"
               "}];",
               ZeroColumn);
  verifyFormat("[myObject doSomethingWith:arg1\n"
               "    firstBlock:^(Foo *a) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }\n"
               "    secondBlock:^(Bar *b) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }\n"
               "    thirdBlock:^Foo(Bar *b) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }];",
               ZeroColumn);
  verifyFormat("f(^{\n"
               "  @autoreleasepool {\n"
               "    if (a) {\n"
               "      g();\n"
               "    }\n"
               "  }\n"
               "});",
               ZeroColumn);
  verifyFormat("void (^largeBlock)(void) = ^{\n"
               "  // ...\n"
               "};",
               ZeroColumn);

  ZeroColumn.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Always;
  verifyFormat("void (^largeBlock)(void) = ^{ int i; };",
               "void   (^largeBlock)(void) = ^{ int   i; };", ZeroColumn);
  ZeroColumn.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Never;
  verifyFormat("void (^largeBlock)(void) = ^{\n"
               "  int i;\n"
               "};",
               "void   (^largeBlock)(void) = ^{ int   i; };", ZeroColumn);
}

TEST_F(FormatTest, SupportsCRLF) {
  verifyFormat("int a;\r\n"
               "int b;\r\n"
               "int c;",
               "int a;\r\n"
               "  int b;\r\n"
               "    int c;");
  verifyFormat("int a;\r\n"
               "int b;\r\n"
               "int c;\r\n",
               "int a;\r\n"
               "  int b;\n"
               "    int c;\r\n");
  verifyFormat("int a;\n"
               "int b;\n"
               "int c;",
               "int a;\r\n"
               "  int b;\n"
               "    int c;");
  // FIXME: unstable test case
  EXPECT_EQ("\"aaaaaaa \"\r\n"
            "\"bbbbbbb\";\r\n",
            format("\"aaaaaaa bbbbbbb\";\r\n", getLLVMStyleWithColumns(10)));
  verifyFormat("#define A \\\r\n"
               "  b;      \\\r\n"
               "  c;      \\\r\n"
               "  d;",
               "#define A \\\r\n"
               "  b; \\\r\n"
               "  c; d; ",
               getGoogleStyle());

  verifyNoChange("/*\r\n"
                 "multi line block comments\r\n"
                 "should not introduce\r\n"
                 "an extra carriage return\r\n"
                 "*/");
  verifyFormat("/*\r\n"
               "\r\n"
               "*/",
               "/*\r\n"
               "    \r\r\r\n"
               "*/");

  FormatStyle style = getLLVMStyle();

  EXPECT_EQ(style.LineEnding, FormatStyle::LE_DeriveLF);
  verifyFormat("union FooBarBazQux {\n"
               "  int foo;\n"
               "  int bar;\n"
               "  int baz;\n"
               "};",
               "union FooBarBazQux {\r\n"
               "  int foo;\n"
               "  int bar;\r\n"
               "  int baz;\n"
               "};",
               style);
  style.LineEnding = FormatStyle::LE_DeriveCRLF;
  verifyFormat("union FooBarBazQux {\r\n"
               "  int foo;\r\n"
               "  int bar;\r\n"
               "  int baz;\r\n"
               "};",
               "union FooBarBazQux {\r\n"
               "  int foo;\n"
               "  int bar;\r\n"
               "  int baz;\n"
               "};",
               style);

  style.LineEnding = FormatStyle::LE_LF;
  verifyFormat("union FooBarBazQux {\n"
               "  int foo;\n"
               "  int bar;\n"
               "  int baz;\n"
               "  int qux;\n"
               "};",
               "union FooBarBazQux {\r\n"
               "  int foo;\n"
               "  int bar;\r\n"
               "  int baz;\n"
               "  int qux;\r\n"
               "};",
               style);
  style.LineEnding = FormatStyle::LE_CRLF;
  verifyFormat("union FooBarBazQux {\r\n"
               "  int foo;\r\n"
               "  int bar;\r\n"
               "  int baz;\r\n"
               "  int qux;\r\n"
               "};",
               "union FooBarBazQux {\r\n"
               "  int foo;\n"
               "  int bar;\r\n"
               "  int baz;\n"
               "  int qux;\n"
               "};",
               style);

  style.LineEnding = FormatStyle::LE_DeriveLF;
  verifyFormat("union FooBarBazQux {\r\n"
               "  int foo;\r\n"
               "  int bar;\r\n"
               "  int baz;\r\n"
               "  int qux;\r\n"
               "};",
               "union FooBarBazQux {\r\n"
               "  int foo;\n"
               "  int bar;\r\n"
               "  int baz;\n"
               "  int qux;\r\n"
               "};",
               style);
  style.LineEnding = FormatStyle::LE_DeriveCRLF;
  verifyFormat("union FooBarBazQux {\n"
               "  int foo;\n"
               "  int bar;\n"
               "  int baz;\n"
               "  int qux;\n"
               "};",
               "union FooBarBazQux {\r\n"
               "  int foo;\n"
               "  int bar;\r\n"
               "  int baz;\n"
               "  int qux;\n"
               "};",
               style);
}

TEST_F(FormatTest, MunchSemicolonAfterBlocks) {
  verifyFormat("MY_CLASS(C) {\n"
               "  int i;\n"
               "  int j;\n"
               "};");
}

TEST_F(FormatTest, ConfigurableContinuationIndentWidth) {
  FormatStyle TwoIndent = getLLVMStyleWithColumns(15);
  TwoIndent.ContinuationIndentWidth = 2;

  verifyFormat("int i =\n"
               "  longFunction(\n"
               "    arg);",
               "int i = longFunction(arg);", TwoIndent);

  FormatStyle SixIndent = getLLVMStyleWithColumns(20);
  SixIndent.ContinuationIndentWidth = 6;

  verifyFormat("int i =\n"
               "      longFunction(\n"
               "            arg);",
               "int i = longFunction(arg);", SixIndent);
}

TEST_F(FormatTest, WrappedClosingParenthesisIndent) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("int Foo::getter(\n"
               "    //\n"
               ") const {\n"
               "  return foo;\n"
               "}",
               Style);
  verifyFormat("void Foo::setter(\n"
               "    //\n"
               ") {\n"
               "  foo = 1;\n"
               "}",
               Style);
}

TEST_F(FormatTest, SpacesInAngles) {
  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInAngles = FormatStyle::SIAS_Always;

  verifyFormat("vector< ::std::string > x1;", Spaces);
  verifyFormat("Foo< int, Bar > x2;", Spaces);
  verifyFormat("Foo< ::int, ::Bar > x3;", Spaces);

  verifyFormat("static_cast< int >(arg);", Spaces);
  verifyFormat("template < typename T0, typename T1 > void f() {}", Spaces);
  verifyFormat("f< int, float >();", Spaces);
  verifyFormat("template <> g() {}", Spaces);
  verifyFormat("template < std::vector< int > > f() {}", Spaces);
  verifyFormat("std::function< void(int, int) > fct;", Spaces);
  verifyFormat("void inFunction() { std::function< void(int, int) > fct; }",
               Spaces);

  Spaces.Standard = FormatStyle::LS_Cpp03;
  Spaces.SpacesInAngles = FormatStyle::SIAS_Always;
  verifyFormat("A< A< int > >();", Spaces);

  Spaces.SpacesInAngles = FormatStyle::SIAS_Never;
  verifyFormat("A<A<int> >();", Spaces);

  Spaces.SpacesInAngles = FormatStyle::SIAS_Leave;
  verifyFormat("vector< ::std::string> x4;", "vector<::std::string> x4;",
               Spaces);
  verifyFormat("vector< ::std::string > x4;", "vector<::std::string > x4;",
               Spaces);

  verifyFormat("A<A<int> >();", Spaces);
  verifyFormat("A<A<int> >();", "A<A<int>>();", Spaces);
  verifyFormat("A< A< int > >();", Spaces);

  Spaces.Standard = FormatStyle::LS_Cpp11;
  Spaces.SpacesInAngles = FormatStyle::SIAS_Always;
  verifyFormat("A< A< int > >();", Spaces);

  Spaces.SpacesInAngles = FormatStyle::SIAS_Never;
  verifyFormat("vector<::std::string> x4;", Spaces);
  verifyFormat("vector<int> x5;", Spaces);
  verifyFormat("Foo<int, Bar> x6;", Spaces);
  verifyFormat("Foo<::int, ::Bar> x7;", Spaces);

  verifyFormat("A<A<int>>();", Spaces);

  Spaces.SpacesInAngles = FormatStyle::SIAS_Leave;
  verifyFormat("vector<::std::string> x4;", Spaces);
  verifyFormat("vector< ::std::string > x4;", Spaces);
  verifyFormat("vector<int> x5;", Spaces);
  verifyFormat("vector< int > x5;", Spaces);
  verifyFormat("Foo<int, Bar> x6;", Spaces);
  verifyFormat("Foo< int, Bar > x6;", Spaces);
  verifyFormat("Foo<::int, ::Bar> x7;", Spaces);
  verifyFormat("Foo< ::int, ::Bar > x7;", Spaces);

  verifyFormat("A<A<int>>();", Spaces);
  verifyFormat("A< A< int > >();", Spaces);
  verifyFormat("A<A<int > >();", Spaces);
  verifyFormat("A< A< int>>();", Spaces);

  Spaces.SpacesInAngles = FormatStyle::SIAS_Always;
  verifyFormat("// clang-format off\n"
               "foo<<<1, 1>>>();\n"
               "// clang-format on",
               Spaces);
  verifyFormat("// clang-format off\n"
               "foo< < <1, 1> > >();\n"
               "// clang-format on",
               Spaces);
}

TEST_F(FormatTest, SpaceAfterTemplateKeyword) {
  FormatStyle Style = getLLVMStyle();
  Style.SpaceAfterTemplateKeyword = false;
  verifyFormat("template<int> void foo();", Style);
}

TEST_F(FormatTest, TripleAngleBrackets) {
  verifyFormat("f<<<1, 1>>>();");
  verifyFormat("f<<<1, 1, 1, s>>>();");
  verifyFormat("f<<<a, b, c, d>>>();");
  verifyFormat("f<<<1, 1>>>();", "f <<< 1, 1 >>> ();");
  verifyFormat("f<param><<<1, 1>>>();");
  verifyFormat("f<1><<<1, 1>>>();");
  verifyFormat("f<param><<<1, 1>>>();", "f< param > <<< 1, 1 >>> ();");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaaaaaaaaaa<<<\n    1, 1>>>();");
  verifyFormat("aaaaaaaaaaaaaaa<aaaaaaaaa, aaaaaaaaaa, aaaaaaaaaaaaaa>\n"
               "    <<<aaaaaaaaa, aaaaaaaaaa, aaaaaaaaaaaaaaaaaa>>>();");
}

TEST_F(FormatTest, MergeLessLessAtEnd) {
  verifyFormat("<<");
  verifyFormat("< < <", "\\\n<<<");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaallvm::outs() <<");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaaallvm::outs()\n    <<");
}

TEST_F(FormatTest, HandleUnbalancedImplicitBracesAcrossPPBranches) {
  std::string code = "#if A\n"
                     "#if B\n"
                     "a.\n"
                     "#endif\n"
                     "    a = 1;\n"
                     "#else\n"
                     "#endif\n"
                     "#if C\n"
                     "#else\n"
                     "#endif\n";
  verifyFormat(code);
}

TEST_F(FormatTest, HandleConflictMarkers) {
  // Git/SVN conflict markers.
  verifyFormat("int a;\n"
               "void f() {\n"
               "  callme(some(parameter1,\n"
               "<<<<<<< text by the vcs\n"
               "              parameter2),\n"
               "||||||| text by the vcs\n"
               "              parameter2),\n"
               "         parameter3,\n"
               "======= text by the vcs\n"
               "              parameter2, parameter3),\n"
               ">>>>>>> text by the vcs\n"
               "         otherparameter);",
               "int a;\n"
               "void f() {\n"
               "  callme(some(parameter1,\n"
               "<<<<<<< text by the vcs\n"
               "  parameter2),\n"
               "||||||| text by the vcs\n"
               "  parameter2),\n"
               "  parameter3,\n"
               "======= text by the vcs\n"
               "  parameter2,\n"
               "  parameter3),\n"
               ">>>>>>> text by the vcs\n"
               "  otherparameter);");

  // Perforce markers.
  verifyFormat("void f() {\n"
               "  function(\n"
               ">>>> text by the vcs\n"
               "      parameter,\n"
               "==== text by the vcs\n"
               "      parameter,\n"
               "==== text by the vcs\n"
               "      parameter,\n"
               "<<<< text by the vcs\n"
               "      parameter);",
               "void f() {\n"
               "  function(\n"
               ">>>> text by the vcs\n"
               "  parameter,\n"
               "==== text by the vcs\n"
               "  parameter,\n"
               "==== text by the vcs\n"
               "  parameter,\n"
               "<<<< text by the vcs\n"
               "  parameter);");

  verifyNoChange("<<<<<<<\n"
                 "|||||||\n"
                 "=======\n"
                 ">>>>>>>");

  verifyNoChange("<<<<<<<\n"
                 "|||||||\n"
                 "int i;\n"
                 "=======\n"
                 ">>>>>>>");

  // FIXME: Handle parsing of macros around conflict markers correctly:
  verifyFormat("#define Macro \\\n"
               "<<<<<<<\n"
               "Something \\\n"
               "|||||||\n"
               "Else \\\n"
               "=======\n"
               "Other \\\n"
               ">>>>>>>\n"
               "    End int i;",
               "#define Macro \\\n"
               "<<<<<<<\n"
               "  Something \\\n"
               "|||||||\n"
               "  Else \\\n"
               "=======\n"
               "  Other \\\n"
               ">>>>>>>\n"
               "  End\n"
               "int i;");

  verifyFormat(R"(====
#ifdef A
a
#else
b
#endif
)");
}

TEST_F(FormatTest, DisableRegions) {
  verifyFormat("int i;\n"
               "// clang-format off\n"
               "  int j;\n"
               "// clang-format on\n"
               "int k;",
               " int  i;\n"
               "   // clang-format off\n"
               "  int j;\n"
               " // clang-format on\n"
               "   int   k;");
  verifyFormat("int i;\n"
               "/* clang-format off */\n"
               "  int j;\n"
               "/* clang-format on */\n"
               "int k;",
               " int  i;\n"
               "   /* clang-format off */\n"
               "  int j;\n"
               " /* clang-format on */\n"
               "   int   k;");

  // Don't reflow comments within disabled regions.
  verifyFormat("// clang-format off\n"
               "// long long long long long long line\n"
               "/* clang-format on */\n"
               "/* long long long\n"
               " * long long long\n"
               " * line */\n"
               "int i;\n"
               "/* clang-format off */\n"
               "/* long long long long long long line */",
               "// clang-format off\n"
               "// long long long long long long line\n"
               "/* clang-format on */\n"
               "/* long long long long long long line */\n"
               "int i;\n"
               "/* clang-format off */\n"
               "/* long long long long long long line */",
               getLLVMStyleWithColumns(20));

  verifyFormat("int *i;\n"
               "// clang-format off:\n"
               "int* j;\n"
               "// clang-format on: 1\n"
               "int *k;",
               "int* i;\n"
               "// clang-format off:\n"
               "int* j;\n"
               "// clang-format on: 1\n"
               "int* k;");

  verifyFormat("int *i;\n"
               "// clang-format off:0\n"
               "int* j;\n"
               "// clang-format only\n"
               "int* k;",
               "int* i;\n"
               "// clang-format off:0\n"
               "int* j;\n"
               "// clang-format only\n"
               "int* k;");

  verifyNoChange("// clang-format off\n"
                 "#if 0\n"
                 "        #if SHOULD_STAY_INDENTED\n"
                 " #endif\n"
                 "#endif\n"
                 "// clang-format on");
}

TEST_F(FormatTest, DoNotCrashOnInvalidInput) {
  format("? ) =");
  verifyNoCrash("#define a\\\n /**/}");
}

TEST_F(FormatTest, FormatsTableGenCode) {
  FormatStyle Style = getLLVMStyle();
  Style.Language = FormatStyle::LK_TableGen;
  verifyFormat("include \"a.td\"\ninclude \"b.td\"", Style);
}

TEST_F(FormatTest, ArrayOfTemplates) {
  verifyFormat("auto a = new unique_ptr<int>[10];",
               "auto a = new unique_ptr<int > [ 10];");

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInSquareBrackets = true;
  verifyFormat("auto a = new unique_ptr<int>[ 10 ];",
               "auto a = new unique_ptr<int > [10];", Spaces);
}

TEST_F(FormatTest, ArrayAsTemplateType) {
  verifyFormat("auto a = unique_ptr<Foo<Bar>[10]>;",
               "auto a = unique_ptr < Foo < Bar>[ 10]> ;");

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInSquareBrackets = true;
  verifyFormat("auto a = unique_ptr<Foo<Bar>[ 10 ]>;",
               "auto a = unique_ptr < Foo < Bar>[10]> ;", Spaces);
}

TEST_F(FormatTest, NoSpaceAfterSuper) { verifyFormat("__super::FooBar();"); }

TEST_F(FormatTest, FormatSortsUsingDeclarations) {
  verifyFormat("using std::cin;\n"
               "using std::cout;",
               "using std::cout;\n"
               "using std::cin;",
               getGoogleStyle());
}

TEST_F(FormatTest, UTF8CharacterLiteralCpp03) {
  FormatStyle Style = getLLVMStyle();
  Style.Standard = FormatStyle::LS_Cpp03;
  // cpp03 recognize this string as identifier u8 and literal character 'a'
  verifyFormat("auto c = u8 'a';", "auto c = u8'a';", Style);
}

TEST_F(FormatTest, UTF8CharacterLiteralCpp11) {
  // u8'a' is a C++17 feature, utf8 literal character, LS_Cpp11 covers
  // all modes, including C++11, C++14 and C++17
  verifyFormat("auto c = u8'a';");
}

TEST_F(FormatTest, DoNotFormatLikelyXml) {
  verifyGoogleFormat("<!-- ;> -->");
  verifyNoChange(" <!-- >; -->", getGoogleStyle());
}

TEST_F(FormatTest, StructuredBindings) {
  // Structured bindings is a C++17 feature.
  // all modes, including C++11, C++14 and C++17
  verifyFormat("auto [a, b] = f();");
  verifyFormat("auto [a, b] = f();", "auto[a, b] = f();");
  verifyFormat("const auto [a, b] = f();", "const   auto[a, b] = f();");
  verifyFormat("auto const [a, b] = f();", "auto  const[a, b] = f();");
  verifyFormat("auto const volatile [a, b] = f();",
               "auto  const   volatile[a, b] = f();");
  verifyFormat("auto [a, b, c] = f();", "auto   [  a  ,  b,c   ] = f();");
  verifyFormat("auto &[a, b, c] = f();", "auto   &[  a  ,  b,c   ] = f();");
  verifyFormat("auto &&[a, b, c] = f();", "auto   &&[  a  ,  b,c   ] = f();");
  verifyFormat("auto const &[a, b] = f();", "auto  const&[a, b] = f();");
  verifyFormat("auto const volatile &&[a, b] = f();",
               "auto  const  volatile  &&[a, b] = f();");
  verifyFormat("auto const &&[a, b] = f();", "auto  const   &&  [a, b] = f();");
  verifyFormat("const auto &[a, b] = f();", "const  auto  &  [a, b] = f();");
  verifyFormat("const auto volatile &&[a, b] = f();",
               "const  auto   volatile  &&[a, b] = f();");
  verifyFormat("volatile const auto &&[a, b] = f();",
               "volatile  const  auto   &&[a, b] = f();");
  verifyFormat("const auto &&[a, b] = f();", "const  auto  &&  [a, b] = f();");

  // Make sure we don't mistake structured bindings for lambdas.
  FormatStyle PointerMiddle = getLLVMStyle();
  PointerMiddle.PointerAlignment = FormatStyle::PAS_Middle;
  verifyGoogleFormat("auto [a1, b]{A * i};");
  verifyFormat("auto [a2, b]{A * i};");
  verifyFormat("auto [a3, b]{A * i};", PointerMiddle);
  verifyGoogleFormat("auto const [a1, b]{A * i};");
  verifyFormat("auto const [a2, b]{A * i};");
  verifyFormat("auto const [a3, b]{A * i};", PointerMiddle);
  verifyGoogleFormat("auto const& [a1, b]{A * i};");
  verifyFormat("auto const &[a2, b]{A * i};");
  verifyFormat("auto const & [a3, b]{A * i};", PointerMiddle);
  verifyGoogleFormat("auto const&& [a1, b]{A * i};");
  verifyFormat("auto const &&[a2, b]{A * i};");
  verifyFormat("auto const && [a3, b]{A * i};", PointerMiddle);

  verifyFormat("for (const auto &&[a, b] : some_range) {\n}",
               "for (const auto   &&   [a, b] : some_range) {\n}");
  verifyFormat("for (const auto &[a, b] : some_range) {\n}",
               "for (const auto   &   [a, b] : some_range) {\n}");
  verifyFormat("for (const auto [a, b] : some_range) {\n}",
               "for (const auto[a, b] : some_range) {\n}");
  verifyFormat("auto [x, y](expr);", "auto[x,y]  (expr);");
  verifyFormat("auto &[x, y](expr);", "auto  &  [x,y]  (expr);");
  verifyFormat("auto &&[x, y](expr);", "auto  &&  [x,y]  (expr);");
  verifyFormat("auto const &[x, y](expr);", "auto  const  &  [x,y]  (expr);");
  verifyFormat("auto const &&[x, y](expr);", "auto  const  &&  [x,y]  (expr);");
  verifyFormat("auto [x, y]{expr};", "auto[x,y]     {expr};");
  verifyFormat("auto const &[x, y]{expr};", "auto  const  &  [x,y]  {expr};");
  verifyFormat("auto const &&[x, y]{expr};", "auto  const  &&  [x,y]  {expr};");

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInSquareBrackets = true;
  verifyFormat("auto [ a, b ] = f();", Spaces);
  verifyFormat("auto &&[ a, b ] = f();", Spaces);
  verifyFormat("auto &[ a, b ] = f();", Spaces);
  verifyFormat("auto const &&[ a, b ] = f();", Spaces);
  verifyFormat("auto const &[ a, b ] = f();", Spaces);
}

TEST_F(FormatTest, FileAndCode) {
  EXPECT_EQ(FormatStyle::LK_Cpp, guessLanguage("foo.cc", ""));
  EXPECT_EQ(FormatStyle::LK_ObjC, guessLanguage("foo.m", ""));
  EXPECT_EQ(FormatStyle::LK_ObjC, guessLanguage("foo.mm", ""));
  EXPECT_EQ(FormatStyle::LK_Cpp, guessLanguage("foo.h", ""));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "@interface Foo\n@end"));
  EXPECT_EQ(
      FormatStyle::LK_ObjC,
      guessLanguage("foo.h", "#define TRY(x, y) @try { x; } @finally { y; }"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "#define AVAIL(x) @available(x, *))"));
  EXPECT_EQ(FormatStyle::LK_ObjC, guessLanguage("foo.h", "@class Foo;"));
  EXPECT_EQ(FormatStyle::LK_Cpp, guessLanguage("foo", ""));
  EXPECT_EQ(FormatStyle::LK_ObjC, guessLanguage("foo", "@interface Foo\n@end"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "int DoStuff(CGRect rect);"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage(
                "foo.h", "#define MY_POINT_MAKE(x, y) CGPointMake((x), (y));"));
  EXPECT_EQ(
      FormatStyle::LK_Cpp,
      guessLanguage("foo.h", "#define FOO(...) auto bar = [] __VA_ARGS__;"));
  // Only one of the two preprocessor regions has ObjC-like code.
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "#if A\n"
                                   "#define B() C\n"
                                   "#else\n"
                                   "#define B() [NSString a:@\"\"]\n"
                                   "#endif"));
}

TEST_F(FormatTest, GuessLanguageWithCpp11AttributeSpecifiers) {
  EXPECT_EQ(FormatStyle::LK_Cpp, guessLanguage("foo.h", "[[noreturn]];"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "array[[calculator getIndex]];"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "[[noreturn, deprecated(\"so sorry\")]];"));
  EXPECT_EQ(
      FormatStyle::LK_Cpp,
      guessLanguage("foo.h", "[[noreturn, deprecated(\"gone, sorry\")]];"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "[[noreturn foo] bar];"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "[[clang::fallthrough]];"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "[[clang:fallthrough] foo];"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "[[gsl::suppress(\"type\")]];"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "[[using clang: fallthrough]];"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "[[abusing clang:fallthrough] bar];"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "[[using gsl: suppress(\"type\")]];"));
  EXPECT_EQ(
      FormatStyle::LK_Cpp,
      guessLanguage("foo.h", "for (auto &&[endpoint, stream] : streams_)"));
  EXPECT_EQ(
      FormatStyle::LK_Cpp,
      guessLanguage("foo.h",
                    "[[clang::callable_when(\"unconsumed\", \"unknown\")]]"));
  EXPECT_EQ(FormatStyle::LK_Cpp, guessLanguage("foo.h", "[[foo::bar, ...]]"));
}

TEST_F(FormatTest, GuessLanguageWithCaret) {
  EXPECT_EQ(FormatStyle::LK_Cpp, guessLanguage("foo.h", "FOO(^);"));
  EXPECT_EQ(FormatStyle::LK_Cpp, guessLanguage("foo.h", "FOO(^, Bar);"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "int(^)(char, float);"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "int(^foo)(char, float);"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "int(^foo[10])(char, float);"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "int(^foo[kNumEntries])(char, float);"));
  EXPECT_EQ(
      FormatStyle::LK_ObjC,
      guessLanguage("foo.h", "int(^foo[(kNumEntries + 10)])(char, float);"));
}

TEST_F(FormatTest, GuessLanguageWithPragmas) {
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "__pragma(warning(disable:))"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "#pragma(warning(disable:))"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "_Pragma(warning(disable:))"));
}

TEST_F(FormatTest, FormatsInlineAsmSymbolicNames) {
  // ASM symbolic names are identifiers that must be surrounded by [] without
  // space in between:
  // https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#InputOperands

  // Example from https://bugs.llvm.org/show_bug.cgi?id=45108.
  verifyFormat(R"(//
asm volatile("mrs %x[result], FPCR" : [result] "=r"(result));
)");

  // A list of several ASM symbolic names.
  verifyFormat(R"(asm("mov %[e], %[d]" : [d] "=rm"(d), [e] "rm"(*e));)");

  // ASM symbolic names in inline ASM with inputs and outputs.
  verifyFormat(R"(//
asm("cmoveq %1, %2, %[result]"
    : [result] "=r"(result)
    : "r"(test), "r"(new), "[result]"(old));
)");

  // ASM symbolic names in inline ASM with no outputs.
  verifyFormat(R"(asm("mov %[e], %[d]" : : [d] "=rm"(d), [e] "rm"(*e));)");
}

TEST_F(FormatTest, GuessedLanguageWithInlineAsmClobbers) {
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "void f() {\n"
                                   "  asm (\"mov %[e], %[d]\"\n"
                                   "     : [d] \"=rm\" (d)\n"
                                   "       [e] \"rm\" (*e));\n"
                                   "}"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "void f() {\n"
                                   "  _asm (\"mov %[e], %[d]\"\n"
                                   "     : [d] \"=rm\" (d)\n"
                                   "       [e] \"rm\" (*e));\n"
                                   "}"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "void f() {\n"
                                   "  __asm (\"mov %[e], %[d]\"\n"
                                   "     : [d] \"=rm\" (d)\n"
                                   "       [e] \"rm\" (*e));\n"
                                   "}"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "void f() {\n"
                                   "  __asm__ (\"mov %[e], %[d]\"\n"
                                   "     : [d] \"=rm\" (d)\n"
                                   "       [e] \"rm\" (*e));\n"
                                   "}"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "void f() {\n"
                                   "  asm (\"mov %[e], %[d]\"\n"
                                   "     : [d] \"=rm\" (d),\n"
                                   "       [e] \"rm\" (*e));\n"
                                   "}"));
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "void f() {\n"
                                   "  asm volatile (\"mov %[e], %[d]\"\n"
                                   "     : [d] \"=rm\" (d)\n"
                                   "       [e] \"rm\" (*e));\n"
                                   "}"));
}

TEST_F(FormatTest, GuessLanguageWithChildLines) {
  EXPECT_EQ(FormatStyle::LK_Cpp,
            guessLanguage("foo.h", "#define FOO ({ std::string s; })"));
  EXPECT_EQ(FormatStyle::LK_ObjC,
            guessLanguage("foo.h", "#define FOO ({ NSString *s; })"));
  EXPECT_EQ(
      FormatStyle::LK_Cpp,
      guessLanguage("foo.h", "#define FOO ({ foo(); ({ std::string s; }) })"));
  EXPECT_EQ(
      FormatStyle::LK_ObjC,
      guessLanguage("foo.h", "#define FOO ({ foo(); ({ NSString *s; }) })"));
}

TEST_F(FormatTest, TypenameMacros) {
  std::vector<std::string> TypenameMacros = {"STACK_OF", "LIST", "TAILQ_ENTRY"};

  // Test case reported in https://bugs.llvm.org/show_bug.cgi?id=30353
  FormatStyle Google = getGoogleStyleWithColumns(0);
  Google.TypenameMacros = TypenameMacros;
  verifyFormat("struct foo {\n"
               "  int bar;\n"
               "  TAILQ_ENTRY(a) bleh;\n"
               "};",
               Google);

  FormatStyle Macros = getLLVMStyle();
  Macros.TypenameMacros = TypenameMacros;

  verifyFormat("STACK_OF(int) a;", Macros);
  verifyFormat("STACK_OF(int) *a;", Macros);
  verifyFormat("STACK_OF(int const *) *a;", Macros);
  verifyFormat("STACK_OF(int *const) *a;", Macros);
  verifyFormat("STACK_OF(int, string) a;", Macros);
  verifyFormat("STACK_OF(LIST(int)) a;", Macros);
  verifyFormat("STACK_OF(LIST(int)) a, b;", Macros);
  verifyFormat("for (LIST(int) *a = NULL; a;) {\n}", Macros);
  verifyFormat("STACK_OF(int) f(LIST(int) *arg);", Macros);
  verifyFormat("vector<LIST(uint64_t) *attr> x;", Macros);
  verifyFormat("vector<LIST(uint64_t) *const> f(LIST(uint64_t) *arg);", Macros);

  Macros.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("STACK_OF(int)* a;", Macros);
  verifyFormat("STACK_OF(int*)* a;", Macros);
  verifyFormat("x = (STACK_OF(uint64_t))*a;", Macros);
  verifyFormat("x = (STACK_OF(uint64_t))&a;", Macros);
  verifyFormat("vector<STACK_OF(uint64_t)* attr> x;", Macros);
}

TEST_F(FormatTest, AtomicQualifier) {
  // Check that we treate _Atomic as a type and not a function call
  FormatStyle Google = getGoogleStyleWithColumns(0);
  verifyFormat("struct foo {\n"
               "  int a1;\n"
               "  _Atomic(a) a2;\n"
               "  _Atomic(_Atomic(int) *const) a3;\n"
               "};",
               Google);
  verifyFormat("_Atomic(uint64_t) a;");
  verifyFormat("_Atomic(uint64_t) *a;");
  verifyFormat("_Atomic(uint64_t const *) *a;");
  verifyFormat("_Atomic(uint64_t *const) *a;");
  verifyFormat("_Atomic(const uint64_t *) *a;");
  verifyFormat("_Atomic(uint64_t) a;");
  verifyFormat("_Atomic(_Atomic(uint64_t)) a;");
  verifyFormat("_Atomic(_Atomic(uint64_t)) a, b;");
  verifyFormat("for (_Atomic(uint64_t) *a = NULL; a;) {\n}");
  verifyFormat("_Atomic(uint64_t) f(_Atomic(uint64_t) *arg);");

  verifyFormat("_Atomic(uint64_t) *s(InitValue);");
  verifyFormat("_Atomic(uint64_t) *s{InitValue};");
  FormatStyle Style = getLLVMStyle();
  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("_Atomic(uint64_t)* s(InitValue);", Style);
  verifyFormat("_Atomic(uint64_t)* s{InitValue};", Style);
  verifyFormat("_Atomic(int)* a;", Style);
  verifyFormat("_Atomic(int*)* a;", Style);
  verifyFormat("vector<_Atomic(uint64_t)* attr> x;", Style);

  Style.SpacesInParens = FormatStyle::SIPO_Custom;
  Style.SpacesInParensOptions.InCStyleCasts = true;
  verifyFormat("x = ( _Atomic(uint64_t) )*a;", Style);
  Style.SpacesInParensOptions.InCStyleCasts = false;
  Style.SpacesInParensOptions.Other = true;
  verifyFormat("x = (_Atomic( uint64_t ))*a;", Style);
  verifyFormat("x = (_Atomic( uint64_t ))&a;", Style);
}

TEST_F(FormatTest, C11Generic) {
  verifyFormat("_Generic(x, int: 1, default: 0)");
  verifyFormat("#define cbrt(X) _Generic((X), float: cbrtf, default: cbrt)(X)");
  verifyFormat("_Generic(x, const char *: 1, char *const: 16, int: 8);");
  verifyFormat("_Generic(x, int: f1, const int: f2)();");
  verifyFormat("_Generic(x, struct A: 1, void (*)(void): 2);");

  verifyFormat("_Generic(x,\n"
               "    float: f,\n"
               "    default: d,\n"
               "    long double: ld,\n"
               "    float _Complex: fc,\n"
               "    double _Complex: dc,\n"
               "    long double _Complex: ldc)");

  FormatStyle Style = getLLVMStyle();
  Style.ColumnLimit = 40;
  verifyFormat("#define LIMIT_MAX(T)                   \\\n"
               "  _Generic(((T)0),                     \\\n"
               "      unsigned int: UINT_MAX,          \\\n"
               "      unsigned long: ULONG_MAX,        \\\n"
               "      unsigned long long: ULLONG_MAX)",
               Style);
  verifyFormat("_Generic(x,\n"
               "    struct A: 1,\n"
               "    void (*)(void): 2);",
               Style);

  Style.ContinuationIndentWidth = 2;
  verifyFormat("_Generic(x,\n"
               "  struct A: 1,\n"
               "  void (*)(void): 2);",
               Style);
}

TEST_F(FormatTest, AmbersandInLamda) {
  // Test case reported in https://bugs.llvm.org/show_bug.cgi?id=41899
  FormatStyle AlignStyle = getLLVMStyle();
  AlignStyle.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("auto lambda = [&a = a]() { a = 2; };", AlignStyle);
  AlignStyle.PointerAlignment = FormatStyle::PAS_Right;
  verifyFormat("auto lambda = [&a = a]() { a = 2; };", AlignStyle);
}

TEST_F(FormatTest, TrailingReturnTypeAuto) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("[]() -> auto { return Val; }", Style);
  verifyFormat("[]() -> auto * { return Val; }", Style);
  verifyFormat("[]() -> auto & { return Val; }", Style);
  verifyFormat("auto foo() -> auto { return Val; }", Style);
  verifyFormat("auto foo() -> auto * { return Val; }", Style);
  verifyFormat("auto foo() -> auto & { return Val; }", Style);
}

TEST_F(FormatTest, SpacesInConditionalStatement) {
  FormatStyle Spaces = getLLVMStyle();
  Spaces.IfMacros.clear();
  Spaces.IfMacros.push_back("MYIF");
  Spaces.SpacesInParens = FormatStyle::SIPO_Custom;
  Spaces.SpacesInParensOptions.InConditionalStatements = true;
  verifyFormat("for ( int i = 0; i; i++ )\n  continue;", Spaces);
  verifyFormat("if ( !a )\n  return;", Spaces);
  verifyFormat("if ( a )\n  return;", Spaces);
  verifyFormat("if constexpr ( a )\n  return;", Spaces);
  verifyFormat("MYIF ( a )\n  return;", Spaces);
  verifyFormat("MYIF ( a )\n  return;\nelse MYIF ( b )\n  return;", Spaces);
  verifyFormat("MYIF ( a )\n  return;\nelse\n  return;", Spaces);
  verifyFormat("switch ( a )\ncase 1:\n  return;", Spaces);
  verifyFormat("while ( a )\n  return;", Spaces);
  verifyFormat("while ( (a && b) )\n  return;", Spaces);
  verifyFormat("do {\n} while ( 1 != 0 );", Spaces);
  verifyFormat("try {\n} catch ( const std::exception & ) {\n}", Spaces);
  // Check that space on the left of "::" is inserted as expected at beginning
  // of condition.
  verifyFormat("while ( ::func() )\n  return;", Spaces);

  // Check impact of ControlStatementsExceptControlMacros is honored.
  Spaces.SpaceBeforeParens =
      FormatStyle::SBPO_ControlStatementsExceptControlMacros;
  verifyFormat("MYIF( a )\n  return;", Spaces);
  verifyFormat("MYIF( a )\n  return;\nelse MYIF( b )\n  return;", Spaces);
  verifyFormat("MYIF( a )\n  return;\nelse\n  return;", Spaces);
}

TEST_F(FormatTest, AlternativeOperators) {
  // Test case for ensuring alternate operators are not
  // combined with their right most neighbour.
  verifyFormat("int a and b;");
  verifyFormat("int a and_eq b;");
  verifyFormat("int a bitand b;");
  verifyFormat("int a bitor b;");
  verifyFormat("int a compl b;");
  verifyFormat("int a not b;");
  verifyFormat("int a not_eq b;");
  verifyFormat("int a or b;");
  verifyFormat("int a xor b;");
  verifyFormat("int a xor_eq b;");
  verifyFormat("return this not_eq bitand other;");
  verifyFormat("bool operator not_eq(const X bitand other)");

  verifyFormat("int a and 5;");
  verifyFormat("int a and_eq 5;");
  verifyFormat("int a bitand 5;");
  verifyFormat("int a bitor 5;");
  verifyFormat("int a compl 5;");
  verifyFormat("int a not 5;");
  verifyFormat("int a not_eq 5;");
  verifyFormat("int a or 5;");
  verifyFormat("int a xor 5;");
  verifyFormat("int a xor_eq 5;");

  verifyFormat("int a compl(5);");
  verifyFormat("int a not(5);");

  /* FIXME handle alternate tokens
   * https://en.cppreference.com/w/cpp/language/operator_alternative
  // alternative tokens
  verifyFormat("compl foo();");     //  ~foo();
  verifyFormat("foo() <%%>;");      // foo();
  verifyFormat("void foo() <%%>;"); // void foo(){}
  verifyFormat("int a <:1:>;");     // int a[1];[
  verifyFormat("%:define ABC abc"); // #define ABC abc
  verifyFormat("%:%:");             // ##
  */
}

TEST_F(FormatTest, STLWhileNotDefineChed) {
  verifyFormat("#if defined(while)\n"
               "#define while EMIT WARNING C4005\n"
               "#endif // while");
}

TEST_F(FormatTest, OperatorSpacing) {
  FormatStyle Style = getLLVMStyle();
  Style.PointerAlignment = FormatStyle::PAS_Right;
  verifyFormat("Foo::operator*();", Style);
  verifyFormat("Foo::operator void *();", Style);
  verifyFormat("Foo::operator void **();", Style);
  verifyFormat("Foo::operator void *&();", Style);
  verifyFormat("Foo::operator void *&&();", Style);
  verifyFormat("Foo::operator void const *();", Style);
  verifyFormat("Foo::operator void const **();", Style);
  verifyFormat("Foo::operator void const *&();", Style);
  verifyFormat("Foo::operator void const *&&();", Style);
  verifyFormat("Foo::operator()(void *);", Style);
  verifyFormat("Foo::operator*(void *);", Style);
  verifyFormat("Foo::operator*();", Style);
  verifyFormat("Foo::operator**();", Style);
  verifyFormat("Foo::operator&();", Style);
  verifyFormat("Foo::operator<int> *();", Style);
  verifyFormat("Foo::operator<Foo> *();", Style);
  verifyFormat("Foo::operator<int> **();", Style);
  verifyFormat("Foo::operator<Foo> **();", Style);
  verifyFormat("Foo::operator<int> &();", Style);
  verifyFormat("Foo::operator<Foo> &();", Style);
  verifyFormat("Foo::operator<int> &&();", Style);
  verifyFormat("Foo::operator<Foo> &&();", Style);
  verifyFormat("Foo::operator<int> *&();", Style);
  verifyFormat("Foo::operator<Foo> *&();", Style);
  verifyFormat("Foo::operator<int> *&&();", Style);
  verifyFormat("Foo::operator<Foo> *&&();", Style);
  verifyFormat("operator*(int (*)(), class Foo);", Style);

  verifyFormat("Foo::operator&();", Style);
  verifyFormat("Foo::operator void &();", Style);
  verifyFormat("Foo::operator void const &();", Style);
  verifyFormat("Foo::operator()(void &);", Style);
  verifyFormat("Foo::operator&(void &);", Style);
  verifyFormat("Foo::operator&();", Style);
  verifyFormat("operator&(int (&)(), class Foo);", Style);
  verifyFormat("operator&&(int (&)(), class Foo);", Style);

  verifyFormat("Foo::operator&&();", Style);
  verifyFormat("Foo::operator**();", Style);
  verifyFormat("Foo::operator void &&();", Style);
  verifyFormat("Foo::operator void const &&();", Style);
  verifyFormat("Foo::operator()(void &&);", Style);
  verifyFormat("Foo::operator&&(void &&);", Style);
  verifyFormat("Foo::operator&&();", Style);
  verifyFormat("operator&&(int (&&)(), class Foo);", Style);
  verifyFormat("operator const nsTArrayRight<E> &()", Style);
  verifyFormat("[[nodiscard]] operator const nsTArrayRight<E, Allocator> &()",
               Style);
  verifyFormat("operator void **()", Style);
  verifyFormat("operator const FooRight<Object> &()", Style);
  verifyFormat("operator const FooRight<Object> *()", Style);
  verifyFormat("operator const FooRight<Object> **()", Style);
  verifyFormat("operator const FooRight<Object> *&()", Style);
  verifyFormat("operator const FooRight<Object> *&&()", Style);

  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("Foo::operator*();", Style);
  verifyFormat("Foo::operator**();", Style);
  verifyFormat("Foo::operator void*();", Style);
  verifyFormat("Foo::operator void**();", Style);
  verifyFormat("Foo::operator void*&();", Style);
  verifyFormat("Foo::operator void*&&();", Style);
  verifyFormat("Foo::operator void const*();", Style);
  verifyFormat("Foo::operator void const**();", Style);
  verifyFormat("Foo::operator void const*&();", Style);
  verifyFormat("Foo::operator void const*&&();", Style);
  verifyFormat("Foo::operator/*comment*/ void*();", Style);
  verifyFormat("Foo::operator/*a*/ const /*b*/ void*();", Style);
  verifyFormat("Foo::operator/*a*/ volatile /*b*/ void*();", Style);
  verifyFormat("Foo::operator()(void*);", Style);
  verifyFormat("Foo::operator*(void*);", Style);
  verifyFormat("Foo::operator*();", Style);
  verifyFormat("Foo::operator<int>*();", Style);
  verifyFormat("Foo::operator<Foo>*();", Style);
  verifyFormat("Foo::operator<int>**();", Style);
  verifyFormat("Foo::operator<Foo>**();", Style);
  verifyFormat("Foo::operator<Foo>*&();", Style);
  verifyFormat("Foo::operator<int>&();", Style);
  verifyFormat("Foo::operator<Foo>&();", Style);
  verifyFormat("Foo::operator<int>&&();", Style);
  verifyFormat("Foo::operator<Foo>&&();", Style);
  verifyFormat("Foo::operator<int>*&();", Style);
  verifyFormat("Foo::operator<Foo>*&();", Style);
  verifyFormat("operator*(int (*)(), class Foo);", Style);

  verifyFormat("Foo::operator&();", Style);
  verifyFormat("Foo::operator void&();", Style);
  verifyFormat("Foo::operator void const&();", Style);
  verifyFormat("Foo::operator/*comment*/ void&();", Style);
  verifyFormat("Foo::operator/*a*/ const /*b*/ void&();", Style);
  verifyFormat("Foo::operator/*a*/ volatile /*b*/ void&();", Style);
  verifyFormat("Foo::operator()(void&);", Style);
  verifyFormat("Foo::operator&(void&);", Style);
  verifyFormat("Foo::operator&();", Style);
  verifyFormat("operator&(int (&)(), class Foo);", Style);
  verifyFormat("operator&(int (&&)(), class Foo);", Style);
  verifyFormat("operator&&(int (&&)(), class Foo);", Style);

  verifyFormat("Foo::operator&&();", Style);
  verifyFormat("Foo::operator void&&();", Style);
  verifyFormat("Foo::operator void const&&();", Style);
  verifyFormat("Foo::operator/*comment*/ void&&();", Style);
  verifyFormat("Foo::operator/*a*/ const /*b*/ void&&();", Style);
  verifyFormat("Foo::operator/*a*/ volatile /*b*/ void&&();", Style);
  verifyFormat("Foo::operator()(void&&);", Style);
  verifyFormat("Foo::operator&&(void&&);", Style);
  verifyFormat("Foo::operator&&();", Style);
  verifyFormat("operator&&(int (&&)(), class Foo);", Style);
  verifyFormat("operator const nsTArrayLeft<E>&()", Style);
  verifyFormat("[[nodiscard]] operator const nsTArrayLeft<E, Allocator>&()",
               Style);
  verifyFormat("operator void**()", Style);
  verifyFormat("operator const FooLeft<Object>&()", Style);
  verifyFormat("operator const FooLeft<Object>*()", Style);
  verifyFormat("operator const FooLeft<Object>**()", Style);
  verifyFormat("operator const FooLeft<Object>*&()", Style);
  verifyFormat("operator const FooLeft<Object>*&&()", Style);

  // PR45107
  verifyFormat("operator Vector<String>&();", Style);
  verifyFormat("operator const Vector<String>&();", Style);
  verifyFormat("operator foo::Bar*();", Style);
  verifyFormat("operator const Foo<X>::Bar<Y>*();", Style);
  verifyFormat("operator/*a*/ const /*b*/ Foo /*c*/<X> /*d*/ ::Bar<Y>*();",
               Style);

  Style.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("Foo::operator*();", Style);
  verifyFormat("Foo::operator void *();", Style);
  verifyFormat("Foo::operator()(void *);", Style);
  verifyFormat("Foo::operator*(void *);", Style);
  verifyFormat("Foo::operator*();", Style);
  verifyFormat("operator*(int (*)(), class Foo);", Style);

  verifyFormat("Foo::operator&();", Style);
  verifyFormat("Foo::operator void &();", Style);
  verifyFormat("Foo::operator void const &();", Style);
  verifyFormat("Foo::operator()(void &);", Style);
  verifyFormat("Foo::operator&(void &);", Style);
  verifyFormat("Foo::operator&();", Style);
  verifyFormat("operator&(int (&)(), class Foo);", Style);

  verifyFormat("Foo::operator&&();", Style);
  verifyFormat("Foo::operator void &&();", Style);
  verifyFormat("Foo::operator void const &&();", Style);
  verifyFormat("Foo::operator()(void &&);", Style);
  verifyFormat("Foo::operator&&(void &&);", Style);
  verifyFormat("Foo::operator&&();", Style);
  verifyFormat("operator&&(int (&&)(), class Foo);", Style);
}

TEST_F(FormatTest, OperatorPassedAsAFunctionPtr) {
  FormatStyle Style = getLLVMStyle();
  // PR46157
  verifyFormat("foo(operator+, -42);", Style);
  verifyFormat("foo(operator++, -42);", Style);
  verifyFormat("foo(operator--, -42);", Style);
  verifyFormat("foo(-42, operator--);", Style);
  verifyFormat("foo(-42, operator, );", Style);
  verifyFormat("foo(operator, , -42);", Style);
}

TEST_F(FormatTest, WhitespaceSensitiveMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.WhitespaceSensitiveMacros.push_back("FOO");

  // Newlines are important here.
  verifyNoChange("FOO(1+2 )\n", Style);
  verifyNoChange("FOO(a:b:c)\n", Style);

  // Don't use the helpers here, since 'mess up' will change the whitespace
  // and these are all whitespace sensitive by definition
  verifyNoChange("FOO(String-ized&Messy+But(: :Still)=Intentional);", Style);
  verifyNoChange("FOO(String-ized&Messy+But\\(: :Still)=Intentional);", Style);
  verifyNoChange("FOO(String-ized&Messy+But,: :Still=Intentional);", Style);
  verifyNoChange("FOO(String-ized&Messy+But,: :\n"
                 "       Still=Intentional);",
                 Style);
  Style.AlignConsecutiveAssignments.Enabled = true;
  verifyNoChange("FOO(String-ized=&Messy+But,: :\n"
                 "       Still=Intentional);",
                 Style);

  Style.ColumnLimit = 21;
  verifyNoChange("FOO(String-ized&Messy+But: :Still=Intentional);", Style);
}

TEST_F(FormatTest, VeryLongNamespaceCommentSplit) {
  // These tests are not in NamespaceEndCommentsFixerTest because that doesn't
  // test its interaction with line wrapping
  FormatStyle Style = getLLVMStyleWithColumns(80);
  verifyFormat("namespace {\n"
               "int i;\n"
               "int j;\n"
               "} // namespace",
               Style);

  verifyFormat("namespace AAA {\n"
               "int i;\n"
               "int j;\n"
               "} // namespace AAA",
               Style);

  verifyFormat("namespace Averyveryveryverylongnamespace {\n"
               "int i;\n"
               "int j;\n"
               "} // namespace Averyveryveryverylongnamespace",
               "namespace Averyveryveryverylongnamespace {\n"
               "int i;\n"
               "int j;\n"
               "}",
               Style);

  verifyFormat(
      "namespace "
      "would::it::save::you::a::lot::of::time::if_::i::just::gave::up::and_::\n"
      "    went::mad::now {\n"
      "int i;\n"
      "int j;\n"
      "} // namespace\n"
      "  // "
      "would::it::save::you::a::lot::of::time::if_::i::just::gave::up::and_::"
      "went::mad::now",
      "namespace "
      "would::it::save::you::a::lot::of::time::if_::i::"
      "just::gave::up::and_::went::mad::now {\n"
      "int i;\n"
      "int j;\n"
      "}",
      Style);

  // This used to duplicate the comment again and again on subsequent runs
  verifyFormat(
      "namespace "
      "would::it::save::you::a::lot::of::time::if_::i::just::gave::up::and_::\n"
      "    went::mad::now {\n"
      "int i;\n"
      "int j;\n"
      "} // namespace\n"
      "  // "
      "would::it::save::you::a::lot::of::time::if_::i::just::gave::up::and_::"
      "went::mad::now",
      "namespace "
      "would::it::save::you::a::lot::of::time::if_::i::"
      "just::gave::up::and_::went::mad::now {\n"
      "int i;\n"
      "int j;\n"
      "} // namespace\n"
      "  // "
      "would::it::save::you::a::lot::of::time::if_::i::just::gave::up::"
      "and_::went::mad::now",
      Style);
}

TEST_F(FormatTest, LikelyUnlikely) {
  FormatStyle Style = getLLVMStyle();

  verifyFormat("if (argc > 5) [[unlikely]] {\n"
               "  return 29;\n"
               "}",
               Style);

  verifyFormat("if (argc > 5) [[likely]] {\n"
               "  return 29;\n"
               "}",
               Style);

  verifyFormat("if (argc > 5) [[unlikely]] {\n"
               "  return 29;\n"
               "} else [[likely]] {\n"
               "  return 42;\n"
               "}",
               Style);

  verifyFormat("if (argc > 5) [[unlikely]] {\n"
               "  return 29;\n"
               "} else if (argc > 10) [[likely]] {\n"
               "  return 99;\n"
               "} else {\n"
               "  return 42;\n"
               "}",
               Style);

  verifyFormat("if (argc > 5) [[gnu::unused]] {\n"
               "  return 29;\n"
               "}",
               Style);

  verifyFormat("if (argc > 5) [[unlikely]]\n"
               "  return 29;",
               Style);
  verifyFormat("if (argc > 5) [[likely]]\n"
               "  return 29;",
               Style);

  verifyFormat("while (limit > 0) [[unlikely]] {\n"
               "  --limit;\n"
               "}",
               Style);
  verifyFormat("for (auto &limit : limits) [[likely]] {\n"
               "  --limit;\n"
               "}",
               Style);

  verifyFormat("for (auto &limit : limits) [[unlikely]]\n"
               "  --limit;",
               Style);
  verifyFormat("while (limit > 0) [[likely]]\n"
               "  --limit;",
               Style);

  Style.AttributeMacros.push_back("UNLIKELY");
  Style.AttributeMacros.push_back("LIKELY");
  verifyFormat("if (argc > 5) UNLIKELY\n"
               "  return 29;",
               Style);

  verifyFormat("if (argc > 5) UNLIKELY {\n"
               "  return 29;\n"
               "}",
               Style);
  verifyFormat("if (argc > 5) UNLIKELY {\n"
               "  return 29;\n"
               "} else [[likely]] {\n"
               "  return 42;\n"
               "}",
               Style);
  verifyFormat("if (argc > 5) UNLIKELY {\n"
               "  return 29;\n"
               "} else LIKELY {\n"
               "  return 42;\n"
               "}",
               Style);
  verifyFormat("if (argc > 5) [[unlikely]] {\n"
               "  return 29;\n"
               "} else LIKELY {\n"
               "  return 42;\n"
               "}",
               Style);

  verifyFormat("for (auto &limit : limits) UNLIKELY {\n"
               "  --limit;\n"
               "}",
               Style);
  verifyFormat("while (limit > 0) LIKELY {\n"
               "  --limit;\n"
               "}",
               Style);

  verifyFormat("while (limit > 0) UNLIKELY\n"
               "  --limit;",
               Style);
  verifyFormat("for (auto &limit : limits) LIKELY\n"
               "  --limit;",
               Style);
}

TEST_F(FormatTest, PenaltyIndentedWhitespace) {
  verifyFormat("Constructor()\n"
               "    : aaaaaa(aaaaaa), aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                          aaaa(aaaaaaaaaaaaaaaaaa, "
               "aaaaaaaaaaaaaaaaaat))");
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaa(aaaaaa), "
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaa)");

  FormatStyle StyleWithWhitespacePenalty = getLLVMStyle();
  StyleWithWhitespacePenalty.PenaltyIndentedWhitespace = 5;
  verifyFormat("Constructor()\n"
               "    : aaaaaa(aaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaa(aaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaat))",
               StyleWithWhitespacePenalty);
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaa(aaaaaa), "
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaa)",
               StyleWithWhitespacePenalty);
}

TEST_F(FormatTest, LLVMDefaultStyle) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("extern \"C\" {\n"
               "int foo();\n"
               "}",
               Style);
}
TEST_F(FormatTest, GNUDefaultStyle) {
  FormatStyle Style = getGNUStyle();
  verifyFormat("extern \"C\"\n"
               "{\n"
               "  int foo ();\n"
               "}",
               Style);
}
TEST_F(FormatTest, MozillaDefaultStyle) {
  FormatStyle Style = getMozillaStyle();
  verifyFormat("extern \"C\"\n"
               "{\n"
               "  int foo();\n"
               "}",
               Style);
}
TEST_F(FormatTest, GoogleDefaultStyle) {
  FormatStyle Style = getGoogleStyle();
  verifyFormat("extern \"C\" {\n"
               "int foo();\n"
               "}",
               Style);
}
TEST_F(FormatTest, ChromiumDefaultStyle) {
  FormatStyle Style = getChromiumStyle(FormatStyle::LanguageKind::LK_Cpp);
  verifyFormat("extern \"C\" {\n"
               "int foo();\n"
               "}",
               Style);
}
TEST_F(FormatTest, MicrosoftDefaultStyle) {
  FormatStyle Style = getMicrosoftStyle(FormatStyle::LanguageKind::LK_Cpp);
  verifyFormat("extern \"C\"\n"
               "{\n"
               "    int foo();\n"
               "}",
               Style);
}
TEST_F(FormatTest, WebKitDefaultStyle) {
  FormatStyle Style = getWebKitStyle();
  verifyFormat("extern \"C\" {\n"
               "int foo();\n"
               "}",
               Style);
}

TEST_F(FormatTest, Concepts) {
  EXPECT_EQ(getLLVMStyle().BreakBeforeConceptDeclarations,
            FormatStyle::BBCDS_Always);

  // The default in LLVM style is REI_OuterScope, but these tests were written
  // when the default was REI_Keyword.
  FormatStyle Style = getLLVMStyle();
  Style.RequiresExpressionIndentation = FormatStyle::REI_Keyword;

  verifyFormat("template <typename T>\n"
               "concept True = true;");

  verifyFormat("template <typename T>\n"
               "concept C = ((false || foo()) && C2<T>) ||\n"
               "            (std::trait<T>::value && Baz) || sizeof(T) >= 6;",
               getLLVMStyleWithColumns(60));

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = true && requires(T t) { t.bar(); } && "
               "sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = true && requires(T t) {\n"
               "                                 t.bar();\n"
               "                                 t.baz();\n"
               "                               } && sizeof(T) <= 8;",
               Style);

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = true && requires(T t) { // Comment\n"
               "                                 t.bar();\n"
               "                                 t.baz();\n"
               "                               } && sizeof(T) <= 8;",
               Style);

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = false || requires(T t) { t.bar(); } && "
               "sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = Unit<T> && !DerivedUnit<T>;");

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = Unit<T> && !(DerivedUnit<T>);");

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = Unit<T> && !!DerivedUnit<T>;");

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = !!false || requires(T t) { t.bar(); } "
               "&& sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck =\n"
               "    static_cast<bool>(0) || requires(T t) { t.bar(); } && "
               "sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = bool(0) || requires(T t) { t.bar(); } "
               "&& sizeof(T) <= 8;");

  verifyFormat(
      "template <typename T>\n"
      "concept DelayedCheck =\n"
      "    (bool)(0) || requires(T t) { t.bar(); } && sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept DelayedCheck = (bool)0 || requires(T t) { t.bar(); } "
               "&& sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept Size = sizeof(T) >= 5 && requires(T t) { t.bar(); } && "
               "sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept Size = 2 < 5 && 2 <= 5 && 8 >= 5 && 8 > 5 &&\n"
               "               requires(T t) {\n"
               "                 t.bar();\n"
               "                 t.baz();\n"
               "               } && sizeof(T) <= 8 && !(4 < 3);",
               getLLVMStyleWithColumns(60));

  verifyFormat("template <typename T>\n"
               "concept TrueOrNot = IsAlwaysTrue || IsNeverTrue;");

  verifyFormat("template <typename T>\n"
               "concept C = foo();");

  verifyFormat("template <typename T>\n"
               "concept C = foo(T());");

  verifyFormat("template <typename T>\n"
               "concept C = foo(T{});");

  verifyFormat("template <typename T>\n"
               "concept Size = V<sizeof(T)>::Value > 5;");

  verifyFormat("template <typename T>\n"
               "concept True = S<T>::Value;");

  verifyFormat("template <S T>\n"
               "concept True = T.field;");

  verifyFormat(
      "template <typename T>\n"
      "concept C = []() { return true; }() && requires(T t) { t.bar(); } &&\n"
      "            sizeof(T) <= 8;");

  // FIXME: This is misformatted because the fake l paren starts at bool, not at
  // the lambda l square.
  verifyFormat("template <typename T>\n"
               "concept C = [] -> bool { return true; }() && requires(T t) { "
               "t.bar(); } &&\n"
               "                      sizeof(T) <= 8;");

  verifyFormat(
      "template <typename T>\n"
      "concept C = decltype([]() { return std::true_type{}; }())::value &&\n"
      "            requires(T t) { t.bar(); } && sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept C = decltype([]() { return std::true_type{}; "
               "}())::value && requires(T t) { t.bar(); } && sizeof(T) <= 8;",
               getLLVMStyleWithColumns(120));

  verifyFormat("template <typename T>\n"
               "concept C = decltype([]() -> std::true_type { return {}; "
               "}())::value &&\n"
               "            requires(T t) { t.bar(); } && sizeof(T) <= 8;");

  verifyFormat("template <typename T>\n"
               "concept C = true;\n"
               "Foo Bar;");

  verifyFormat("template <typename T>\n"
               "concept Hashable = requires(T a) {\n"
               "                     { std::hash<T>{}(a) } -> "
               "std::convertible_to<std::size_t>;\n"
               "                   };",
               Style);

  verifyFormat(
      "template <typename T>\n"
      "concept EqualityComparable = requires(T a, T b) {\n"
      "                               { a == b } -> std::same_as<bool>;\n"
      "                             };",
      Style);

  verifyFormat(
      "template <typename T>\n"
      "concept EqualityComparable = requires(T a, T b) {\n"
      "                               { a == b } -> std::same_as<bool>;\n"
      "                               { a != b } -> std::same_as<bool>;\n"
      "                             };",
      Style);

  verifyFormat("template <typename T>\n"
               "concept WeakEqualityComparable = requires(T a, T b) {\n"
               "                                   { a == b };\n"
               "                                   { a != b };\n"
               "                                 };",
               Style);

  verifyFormat("template <typename T>\n"
               "concept HasSizeT = requires { typename T::size_t; };");

  verifyFormat("template <typename T>\n"
               "concept Semiregular =\n"
               "    DefaultConstructible<T> && CopyConstructible<T> && "
               "CopyAssignable<T> &&\n"
               "    requires(T a, std::size_t n) {\n"
               "      requires Same<T *, decltype(&a)>;\n"
               "      { a.~T() } noexcept;\n"
               "      requires Same<T *, decltype(new T)>;\n"
               "      requires Same<T *, decltype(new T[n])>;\n"
               "      { delete new T; };\n"
               "      { delete new T[n]; };\n"
               "    };",
               Style);

  verifyFormat("template <typename T>\n"
               "concept Semiregular =\n"
               "    requires(T a, std::size_t n) {\n"
               "      requires Same<T *, decltype(&a)>;\n"
               "      { a.~T() } noexcept;\n"
               "      requires Same<T *, decltype(new T)>;\n"
               "      requires Same<T *, decltype(new T[n])>;\n"
               "      { delete new T; };\n"
               "      { delete new T[n]; };\n"
               "      { new T } -> std::same_as<T *>;\n"
               "    } && DefaultConstructible<T> && CopyConstructible<T> && "
               "CopyAssignable<T>;",
               Style);

  verifyFormat(
      "template <typename T>\n"
      "concept Semiregular =\n"
      "    DefaultConstructible<T> && requires(T a, std::size_t n) {\n"
      "                                 requires Same<T *, decltype(&a)>;\n"
      "                                 { a.~T() } noexcept;\n"
      "                                 requires Same<T *, decltype(new T)>;\n"
      "                                 requires Same<T *, decltype(new "
      "T[n])>;\n"
      "                                 { delete new T; };\n"
      "                                 { delete new T[n]; };\n"
      "                               } && CopyConstructible<T> && "
      "CopyAssignable<T>;",
      Style);

  verifyFormat("template <typename T>\n"
               "concept Two = requires(T t) {\n"
               "                { t.foo() } -> std::same_as<Bar>;\n"
               "              } && requires(T &&t) {\n"
               "                     { t.foo() } -> std::same_as<Bar &&>;\n"
               "                   };",
               Style);

  verifyFormat(
      "template <typename T>\n"
      "concept C = requires(T x) {\n"
      "              { *x } -> std::convertible_to<typename T::inner>;\n"
      "              { x + 1 } noexcept -> std::same_as<int>;\n"
      "              { x * 1 } -> std::convertible_to<T>;\n"
      "            };",
      Style);

  verifyFormat("template <typename T>\n"
               "concept C = requires(T x) {\n"
               "              {\n"
               "                long_long_long_function_call(1, 2, 3, 4, 5)\n"
               "              } -> long_long_concept_name<T>;\n"
               "              {\n"
               "                long_long_long_function_call(1, 2, 3, 4, 5)\n"
               "              } noexcept -> long_long_concept_name<T>;\n"
               "            };",
               Style);

  verifyFormat(
      "template <typename T, typename U = T>\n"
      "concept Swappable = requires(T &&t, U &&u) {\n"
      "                      swap(std::forward<T>(t), std::forward<U>(u));\n"
      "                      swap(std::forward<U>(u), std::forward<T>(t));\n"
      "                    };",
      Style);

  verifyFormat("template <typename T, typename U>\n"
               "concept Common = requires(T &&t, U &&u) {\n"
               "                   typename CommonType<T, U>;\n"
               "                   { CommonType<T, U>(std::forward<T>(t)) };\n"
               "                 };",
               Style);

  verifyFormat("template <typename T, typename U>\n"
               "concept Common = requires(T &&t, U &&u) {\n"
               "                   typename CommonType<T, U>;\n"
               "                   { CommonType<T, U>{std::forward<T>(t)} };\n"
               "                 };",
               Style);

  verifyFormat(
      "template <typename T>\n"
      "concept C = requires(T t) {\n"
      "              requires Bar<T> && Foo<T>;\n"
      "              requires((trait<T> && Baz) || (T2<T> && Foo<T>));\n"
      "            };",
      Style);

  verifyFormat("template <typename T>\n"
               "concept HasFoo = requires(T t) {\n"
               "                   { t.foo() };\n"
               "                   t.foo();\n"
               "                 };\n"
               "template <typename T>\n"
               "concept HasBar = requires(T t) {\n"
               "                   { t.bar() };\n"
               "                   t.bar();\n"
               "                 };",
               Style);

  verifyFormat("template <typename T>\n"
               "concept Large = sizeof(T) > 10;");

  verifyFormat("template <typename T, typename U>\n"
               "concept FooableWith = requires(T t, U u) {\n"
               "                        typename T::foo_type;\n"
               "                        { t.foo(u) } -> typename T::foo_type;\n"
               "                        t++;\n"
               "                      };\n"
               "void doFoo(FooableWith<int> auto t) { t.foo(3); }",
               Style);

  verifyFormat("template <typename T>\n"
               "concept Context = is_specialization_of_v<context, T>;");

  verifyFormat("template <typename T>\n"
               "concept Node = std::is_object_v<T>;");

  verifyFormat("template <class T>\n"
               "concept integral = __is_integral(T);");

  verifyFormat("template <class T>\n"
               "concept is2D = __array_extent(T, 1) == 2;");

  verifyFormat("template <class T>\n"
               "concept isRhs = __is_rvalue_expr(std::declval<T>() + 2)");

  verifyFormat("template <class T, class T2>\n"
               "concept Same = __is_same_as<T, T2>;");

  verifyFormat(
      "template <class _InIt, class _OutIt>\n"
      "concept _Can_reread_dest =\n"
      "    std::forward_iterator<_OutIt> &&\n"
      "    std::same_as<std::iter_value_t<_InIt>, std::iter_value_t<_OutIt>>;");

  Style.BreakBeforeConceptDeclarations = FormatStyle::BBCDS_Allowed;

  verifyFormat(
      "template <typename T>\n"
      "concept C = requires(T t) {\n"
      "              requires Bar<T> && Foo<T>;\n"
      "              requires((trait<T> && Baz) || (T2<T> && Foo<T>));\n"
      "            };",
      Style);

  verifyFormat("template <typename T>\n"
               "concept HasFoo = requires(T t) {\n"
               "                   { t.foo() };\n"
               "                   t.foo();\n"
               "                 };\n"
               "template <typename T>\n"
               "concept HasBar = requires(T t) {\n"
               "                   { t.bar() };\n"
               "                   t.bar();\n"
               "                 };",
               Style);

  verifyFormat("template <typename T> concept True = true;", Style);

  verifyFormat("template <typename T>\n"
               "concept C = decltype([]() -> std::true_type { return {}; "
               "}())::value &&\n"
               "            requires(T t) { t.bar(); } && sizeof(T) <= 8;",
               Style);

  verifyFormat("template <typename T>\n"
               "concept Semiregular =\n"
               "    DefaultConstructible<T> && CopyConstructible<T> && "
               "CopyAssignable<T> &&\n"
               "    requires(T a, std::size_t n) {\n"
               "      requires Same<T *, decltype(&a)>;\n"
               "      { a.~T() } noexcept;\n"
               "      requires Same<T *, decltype(new T)>;\n"
               "      requires Same<T *, decltype(new T[n])>;\n"
               "      { delete new T; };\n"
               "      { delete new T[n]; };\n"
               "    };",
               Style);

  Style.BreakBeforeConceptDeclarations = FormatStyle::BBCDS_Never;

  verifyFormat("template <typename T> concept C =\n"
               "    requires(T t) {\n"
               "      requires Bar<T> && Foo<T>;\n"
               "      requires((trait<T> && Baz) || (T2<T> && Foo<T>));\n"
               "    };",
               Style);

  verifyFormat("template <typename T> concept HasFoo = requires(T t) {\n"
               "                                         { t.foo() };\n"
               "                                         t.foo();\n"
               "                                       };\n"
               "template <typename T> concept HasBar = requires(T t) {\n"
               "                                         { t.bar() };\n"
               "                                         t.bar();\n"
               "                                       };",
               Style);

  verifyFormat("template <typename T> concept True = true;", Style);

  verifyFormat(
      "template <typename T> concept C =\n"
      "    decltype([]() -> std::true_type { return {}; }())::value &&\n"
      "    requires(T t) { t.bar(); } && sizeof(T) <= 8;",
      Style);

  verifyFormat("template <typename T> concept Semiregular =\n"
               "    DefaultConstructible<T> && CopyConstructible<T> && "
               "CopyAssignable<T> &&\n"
               "    requires(T a, std::size_t n) {\n"
               "      requires Same<T *, decltype(&a)>;\n"
               "      { a.~T() } noexcept;\n"
               "      requires Same<T *, decltype(new T)>;\n"
               "      requires Same<T *, decltype(new T[n])>;\n"
               "      { delete new T; };\n"
               "      { delete new T[n]; };\n"
               "    };",
               Style);

  // The following tests are invalid C++, we just want to make sure we don't
  // assert.
  verifyNoCrash("template <typename T>\n"
                "concept C = requires C2<T>;");

  verifyNoCrash("template <typename T>\n"
                "concept C = 5 + 4;");

  verifyNoCrash("template <typename T>\n"
                "concept C = class X;");

  verifyNoCrash("template <typename T>\n"
                "concept C = [] && true;");

  verifyNoCrash("template <typename T>\n"
                "concept C = [] && requires(T t) { typename T::size_type; };");
}

TEST_F(FormatTest, RequiresClausesPositions) {
  auto Style = getLLVMStyle();
  EXPECT_EQ(Style.RequiresClausePosition, FormatStyle::RCPS_OwnLine);
  EXPECT_EQ(Style.IndentRequiresClause, true);

  // The default in LLVM style is REI_OuterScope, but these tests were written
  // when the default was REI_Keyword.
  Style.RequiresExpressionIndentation = FormatStyle::REI_Keyword;

  verifyFormat("template <typename T>\n"
               "  requires(Foo<T> && std::trait<T>)\n"
               "struct Bar;",
               Style);

  verifyFormat("template <typename T>\n"
               "  requires(Foo<T> && std::trait<T>)\n"
               "class Bar {\n"
               "public:\n"
               "  Bar(T t);\n"
               "  bool baz();\n"
               "};",
               Style);

  verifyFormat(
      "template <typename T>\n"
      "  requires requires(T &&t) {\n"
      "             typename T::I;\n"
      "             requires(F<typename T::I> && std::trait<typename T::I>);\n"
      "           }\n"
      "Bar(T) -> Bar<typename T::I>;",
      Style);

  verifyFormat("template <typename T>\n"
               "  requires(Foo<T> && std::trait<T>)\n"
               "constexpr T MyGlobal;",
               Style);

  verifyFormat("template <typename T>\n"
               "  requires Foo<T> && requires(T t) {\n"
               "                       { t.baz() } -> std::same_as<bool>;\n"
               "                       requires std::same_as<T::Factor, int>;\n"
               "                     }\n"
               "inline int bar(T t) {\n"
               "  return t.baz() ? T::Factor : 5;\n"
               "}",
               Style);

  verifyFormat("template <typename T>\n"
               "inline int bar(T t)\n"
               "  requires Foo<T> && requires(T t) {\n"
               "                       { t.baz() } -> std::same_as<bool>;\n"
               "                       requires std::same_as<T::Factor, int>;\n"
               "                     }\n"
               "{\n"
               "  return t.baz() ? T::Factor : 5;\n"
               "}",
               Style);

  verifyFormat("template <typename T>\n"
               "  requires F<T>\n"
               "int bar(T t) {\n"
               "  return 5;\n"
               "}",
               Style);

  verifyFormat("template <typename T>\n"
               "int bar(T t)\n"
               "  requires F<T>\n"
               "{\n"
               "  return 5;\n"
               "}",
               Style);

  verifyFormat("template <typename T>\n"
               "int S::bar(T t) &&\n"
               "  requires F<T>\n"
               "{\n"
               "  return 5;\n"
               "}",
               Style);

  verifyFormat("template <typename T>\n"
               "int bar(T t)\n"
               "  requires F<T>;",
               Style);

  Style.IndentRequiresClause = false;
  verifyFormat("template <typename T>\n"
               "requires F<T>\n"
               "int bar(T t) {\n"
               "  return 5;\n"
               "}",
               Style);

  verifyFormat("template <typename T>\n"
               "int S::bar(T t) &&\n"
               "requires F<T>\n"
               "{\n"
               "  return 5;\n"
               "}",
               Style);

  verifyFormat("template <typename T>\n"
               "int bar(T t)\n"
               "requires F<T>\n"
               "{\n"
               "  return 5;\n"
               "}",
               Style);

  Style.RequiresClausePosition = FormatStyle::RCPS_SingleLine;
  verifyFormat("template <typename T> requires Foo<T> struct Bar {};\n"
               "template <typename T> requires Foo<T> void bar() {}\n"
               "template <typename T> void bar() requires Foo<T> {}\n"
               "template <typename T> void bar() requires Foo<T>;\n"
               "template <typename T> void S::bar() && requires Foo<T> {}\n"
               "template <typename T> requires Foo<T> Bar(T) -> Bar<T>;",
               Style);

  auto ColumnStyle = Style;
  ColumnStyle.ColumnLimit = 40;
  verifyFormat("template <typename AAAAAAA>\n"
               "requires Foo<T> struct Bar {};\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<T> void bar() {}\n"
               "template <typename AAAAAAA>\n"
               "void bar() requires Foo<T> {}\n"
               "template <typename T>\n"
               "void S::bar() && requires Foo<T> {}\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<T> Baz(T) -> Baz<T>;",
               ColumnStyle);

  verifyFormat("template <typename T>\n"
               "requires Foo<AAAAAAA> struct Bar {};\n"
               "template <typename T>\n"
               "requires Foo<AAAAAAA> void bar() {}\n"
               "template <typename T>\n"
               "void bar() requires Foo<AAAAAAA> {}\n"
               "template <typename T>\n"
               "requires Foo<AAAAAAA> Bar(T) -> Bar<T>;",
               ColumnStyle);

  verifyFormat("template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "struct Bar {};\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "void bar() {}\n"
               "template <typename AAAAAAA>\n"
               "void bar()\n"
               "    requires Foo<AAAAAAAAAAAAAAAA> {}\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAA> Bar(T) -> Bar<T>;\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "Bar(T) -> Bar<T>;",
               ColumnStyle);

  Style.RequiresClausePosition = FormatStyle::RCPS_WithFollowing;
  ColumnStyle.RequiresClausePosition = FormatStyle::RCPS_WithFollowing;

  verifyFormat("template <typename T>\n"
               "requires Foo<T> struct Bar {};\n"
               "template <typename T>\n"
               "requires Foo<T> void bar() {}\n"
               "template <typename T>\n"
               "void bar()\n"
               "requires Foo<T> {}\n"
               "template <typename T>\n"
               "void bar()\n"
               "requires Foo<T>;\n"
               "template <typename T>\n"
               "void S::bar() &&\n"
               "requires Foo<T> {}\n"
               "template <typename T>\n"
               "requires Foo<T> Bar(T) -> Bar<T>;",
               Style);

  verifyFormat("template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "struct Bar {};\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "void bar() {}\n"
               "template <typename AAAAAAA>\n"
               "void bar()\n"
               "requires Foo<AAAAAAAAAAAAAAAA> {}\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAA> Bar(T) -> Bar<T>;\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "Bar(T) -> Bar<T>;",
               ColumnStyle);

  Style.IndentRequiresClause = true;
  ColumnStyle.IndentRequiresClause = true;

  verifyFormat("template <typename T>\n"
               "  requires Foo<T> struct Bar {};\n"
               "template <typename T>\n"
               "  requires Foo<T> void bar() {}\n"
               "template <typename T>\n"
               "void bar()\n"
               "  requires Foo<T> {}\n"
               "template <typename T>\n"
               "void S::bar() &&\n"
               "  requires Foo<T> {}\n"
               "template <typename T>\n"
               "  requires Foo<T> Bar(T) -> Bar<T>;",
               Style);

  verifyFormat("template <typename AAAAAAA>\n"
               "  requires Foo<AAAAAAAAAAAAAAAA>\n"
               "struct Bar {};\n"
               "template <typename AAAAAAA>\n"
               "  requires Foo<AAAAAAAAAAAAAAAA>\n"
               "void bar() {}\n"
               "template <typename AAAAAAA>\n"
               "void bar()\n"
               "  requires Foo<AAAAAAAAAAAAAAAA> {}\n"
               "template <typename AAAAAAA>\n"
               "  requires Foo<AAAAAA> Bar(T) -> Bar<T>;\n"
               "template <typename AAAAAAA>\n"
               "  requires Foo<AAAAAAAAAAAAAAAA>\n"
               "Bar(T) -> Bar<T>;",
               ColumnStyle);

  Style.RequiresClausePosition = FormatStyle::RCPS_WithPreceding;
  ColumnStyle.RequiresClausePosition = FormatStyle::RCPS_WithPreceding;

  verifyFormat("template <typename T> requires Foo<T>\n"
               "struct Bar {};\n"
               "template <typename T> requires Foo<T>\n"
               "void bar() {}\n"
               "template <typename T>\n"
               "void bar() requires Foo<T>\n"
               "{}\n"
               "template <typename T> void bar() requires Foo<T>;\n"
               "template <typename T>\n"
               "void S::bar() && requires Foo<T>\n"
               "{}\n"
               "template <typename T> requires Foo<T>\n"
               "Bar(T) -> Bar<T>;",
               Style);

  verifyFormat("template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "struct Bar {};\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "void bar() {}\n"
               "template <typename AAAAAAA>\n"
               "void bar()\n"
               "    requires Foo<AAAAAAAAAAAAAAAA>\n"
               "{}\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAA>\n"
               "Bar(T) -> Bar<T>;\n"
               "template <typename AAAAAAA>\n"
               "requires Foo<AAAAAAAAAAAAAAAA>\n"
               "Bar(T) -> Bar<T>;",
               ColumnStyle);
}

TEST_F(FormatTest, RequiresClauses) {
  verifyFormat("struct [[nodiscard]] zero_t {\n"
               "  template <class T>\n"
               "    requires requires { number_zero_v<T>; }\n"
               "  [[nodiscard]] constexpr operator T() const {\n"
               "    return number_zero_v<T>;\n"
               "  }\n"
               "};");

  verifyFormat("template <class T>\n"
               "  requires(std::same_as<int, T>)\n"
               "decltype(auto) fun() {}");

  auto Style = getLLVMStyle();

  verifyFormat(
      "template <typename T>\n"
      "  requires is_default_constructible_v<hash<T>> and\n"
      "           is_copy_constructible_v<hash<T>> and\n"
      "           is_move_constructible_v<hash<T>> and\n"
      "           is_copy_assignable_v<hash<T>> and "
      "is_move_assignable_v<hash<T>> and\n"
      "           is_destructible_v<hash<T>> and is_swappable_v<hash<T>> and\n"
      "           is_callable_v<hash<T>(T)> and\n"
      "           is_same_v<size_t, decltype(hash<T>(declval<T>()))> and\n"
      "           is_same_v<size_t, decltype(hash<T>(declval<T &>()))> and\n"
      "           is_same_v<size_t, decltype(hash<T>(declval<const T &>()))>\n"
      "struct S {};",
      Style);

  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat(
      "template <typename T>\n"
      "  requires is_default_constructible_v<hash<T>>\n"
      "           and is_copy_constructible_v<hash<T>>\n"
      "           and is_move_constructible_v<hash<T>>\n"
      "           and is_copy_assignable_v<hash<T>> and "
      "is_move_assignable_v<hash<T>>\n"
      "           and is_destructible_v<hash<T>> and is_swappable_v<hash<T>>\n"
      "           and is_callable_v<hash<T>(T)>\n"
      "           and is_same_v<size_t, decltype(hash<T>(declval<T>()))>\n"
      "           and is_same_v<size_t, decltype(hash<T>(declval<T &>()))>\n"
      "           and is_same_v<size_t, decltype(hash<T>(declval<const T "
      "&>()))>\n"
      "struct S {};",
      Style);

  Style = getLLVMStyle();
  Style.ConstructorInitializerIndentWidth = 4;
  Style.BreakConstructorInitializers = FormatStyle::BCIS_BeforeColon;
  Style.PackConstructorInitializers = FormatStyle::PCIS_Never;
  verifyFormat("constexpr Foo(Foo const &other)\n"
               "  requires std::is_copy_constructible<T>\n"
               "    : value{other.value} {\n"
               "  do_magic();\n"
               "  do_more_magic();\n"
               "}",
               Style);

  // Not a clause, but we once hit an assert.
  verifyFormat("#if 0\n"
               "#else\n"
               "foo();\n"
               "#endif\n"
               "bar(requires);");
}

TEST_F(FormatTest, RequiresExpressionIndentation) {
  auto Style = getLLVMStyle();
  EXPECT_EQ(Style.RequiresExpressionIndentation, FormatStyle::REI_OuterScope);

  verifyFormat("template <typename T>\n"
               "concept C = requires(T t) {\n"
               "  typename T::value;\n"
               "  requires requires(typename T::value v) {\n"
               "    { t == v } -> std::same_as<bool>;\n"
               "  };\n"
               "};",
               Style);

  verifyFormat("template <typename T>\n"
               "void bar(T)\n"
               "  requires Foo<T> && requires(T t) {\n"
               "    { t.foo() } -> std::same_as<int>;\n"
               "  } && requires(T t) {\n"
               "    { t.bar() } -> std::same_as<bool>;\n"
               "    --t;\n"
               "  };",
               Style);

  verifyFormat("template <typename T>\n"
               "  requires Foo<T> &&\n"
               "           requires(T t) {\n"
               "             { t.foo() } -> std::same_as<int>;\n"
               "           } && requires(T t) {\n"
               "             { t.bar() } -> std::same_as<bool>;\n"
               "             --t;\n"
               "           }\n"
               "void bar(T);",
               Style);

  verifyFormat("template <typename T> void f() {\n"
               "  if constexpr (requires(T t) {\n"
               "                  { t.bar() } -> std::same_as<bool>;\n"
               "                }) {\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("template <typename T> void f() {\n"
               "  if constexpr (condition && requires(T t) {\n"
               "                  { t.bar() } -> std::same_as<bool>;\n"
               "                }) {\n"
               "  }\n"
               "}",
               Style);

  verifyFormat("template <typename T> struct C {\n"
               "  void f()\n"
               "    requires requires(T t) {\n"
               "      { t.bar() } -> std::same_as<bool>;\n"
               "    };\n"
               "};",
               Style);

  Style.RequiresExpressionIndentation = FormatStyle::REI_Keyword;

  verifyFormat("template <typename T>\n"
               "concept C = requires(T t) {\n"
               "              typename T::value;\n"
               "              requires requires(typename T::value v) {\n"
               "                         { t == v } -> std::same_as<bool>;\n"
               "                       };\n"
               "            };",
               Style);

  verifyFormat(
      "template <typename T>\n"
      "void bar(T)\n"
      "  requires Foo<T> && requires(T t) {\n"
      "                       { t.foo() } -> std::same_as<int>;\n"
      "                     } && requires(T t) {\n"
      "                            { t.bar() } -> std::same_as<bool>;\n"
      "                            --t;\n"
      "                          };",
      Style);

  verifyFormat("template <typename T>\n"
               "  requires Foo<T> &&\n"
               "           requires(T t) {\n"
               "             { t.foo() } -> std::same_as<int>;\n"
               "           } && requires(T t) {\n"
               "                  { t.bar() } -> std::same_as<bool>;\n"
               "                  --t;\n"
               "                }\n"
               "void bar(T);",
               Style);

  verifyFormat("template <typename T> void f() {\n"
               "  if constexpr (requires(T t) {\n"
               "                  { t.bar() } -> std::same_as<bool>;\n"
               "                }) {\n"
               "  }\n"
               "}",
               Style);

  verifyFormat(
      "template <typename T> void f() {\n"
      "  if constexpr (condition && requires(T t) {\n"
      "                               { t.bar() } -> std::same_as<bool>;\n"
      "                             }) {\n"
      "  }\n"
      "}",
      Style);

  verifyFormat("template <typename T> struct C {\n"
               "  void f()\n"
               "    requires requires(T t) {\n"
               "               { t.bar() } -> std::same_as<bool>;\n"
               "             };\n"
               "};",
               Style);
}

TEST_F(FormatTest, StatementAttributeLikeMacros) {
  FormatStyle Style = getLLVMStyle();
  StringRef Source = "void Foo::slot() {\n"
                     "  unsigned char MyChar = 'x';\n"
                     "  emit signal(MyChar);\n"
                     "  Q_EMIT signal(MyChar);\n"
                     "}";

  verifyFormat(Source, Style);

  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("void Foo::slot() {\n"
               "  unsigned char MyChar = 'x';\n"
               "  emit          signal(MyChar);\n"
               "  Q_EMIT signal(MyChar);\n"
               "}",
               Source, Style);

  Style.StatementAttributeLikeMacros.push_back("emit");
  verifyFormat(Source, Style);

  Style.StatementAttributeLikeMacros = {};
  verifyFormat("void Foo::slot() {\n"
               "  unsigned char MyChar = 'x';\n"
               "  emit          signal(MyChar);\n"
               "  Q_EMIT        signal(MyChar);\n"
               "}",
               Source, Style);
}

TEST_F(FormatTest, IndentAccessModifiers) {
  FormatStyle Style = getLLVMStyle();
  Style.IndentAccessModifiers = true;
  // Members are *two* levels below the record;
  // Style.IndentWidth == 2, thus yielding a 4 spaces wide indentation.
  verifyFormat("class C {\n"
               "    int i;\n"
               "};",
               Style);
  verifyFormat("union C {\n"
               "    int i;\n"
               "    unsigned u;\n"
               "};",
               Style);
  // Access modifiers should be indented one level below the record.
  verifyFormat("class C {\n"
               "  public:\n"
               "    int i;\n"
               "};",
               Style);
  verifyFormat("class C {\n"
               "  public /* comment */:\n"
               "    int i;\n"
               "};",
               Style);
  verifyFormat("struct S {\n"
               "  private:\n"
               "    class C {\n"
               "        int j;\n"
               "\n"
               "      public:\n"
               "        C();\n"
               "    };\n"
               "\n"
               "  public:\n"
               "    int i;\n"
               "};",
               Style);
  // Enumerations are not records and should be unaffected.
  Style.AllowShortEnumsOnASingleLine = false;
  verifyFormat("enum class E {\n"
               "  A,\n"
               "  B\n"
               "};",
               Style);
  // Test with a different indentation width;
  // also proves that the result is Style.AccessModifierOffset agnostic.
  Style.IndentWidth = 3;
  verifyFormat("class C {\n"
               "   public:\n"
               "      int i;\n"
               "};",
               Style);
  verifyFormat("class C {\n"
               "   public /**/:\n"
               "      int i;\n"
               "};",
               Style);
}

TEST_F(FormatTest, LimitlessStringsAndComments) {
  auto Style = getLLVMStyleWithColumns(0);
  constexpr StringRef Code =
      "/**\n"
      " * This is a multiline comment with quite some long lines, at least for "
      "the LLVM Style.\n"
      " * We will redo this with strings and line comments. Just to  check if "
      "everything is working.\n"
      " */\n"
      "bool foo() {\n"
      "  /* Single line multi line comment. */\n"
      "  const std::string String = \"This is a multiline string with quite "
      "some long lines, at least for the LLVM Style.\"\n"
      "                             \"We already did it with multi line "
      "comments, and we will do it with line comments. Just to check if "
      "everything is working.\";\n"
      "  // This is a line comment (block) with quite some long lines, at "
      "least for the LLVM Style.\n"
      "  // We already did this with multi line comments and strings. Just to "
      "check if everything is working.\n"
      "  const std::string SmallString = \"Hello World\";\n"
      "  // Small line comment\n"
      "  return String.size() > SmallString.size();\n"
      "}";
  verifyNoChange(Code, Style);
}

TEST_F(FormatTest, FormatDecayCopy) {
  // error cases from unit tests
  verifyFormat("foo(auto())");
  verifyFormat("foo(auto{})");
  verifyFormat("foo(auto({}))");
  verifyFormat("foo(auto{{}})");

  verifyFormat("foo(auto(1))");
  verifyFormat("foo(auto{1})");
  verifyFormat("foo(new auto(1))");
  verifyFormat("foo(new auto{1})");
  verifyFormat("decltype(auto(1)) x;");
  verifyFormat("decltype(auto{1}) x;");
  verifyFormat("auto(x);");
  verifyFormat("auto{x};");
  verifyFormat("new auto{x};");
  verifyFormat("auto{x} = y;");
  verifyFormat("auto(x) = y;"); // actually a declaration, but this is clearly
                                // the user's own fault
  verifyFormat("integral auto(x) = y;"); // actually a declaration, but this is
                                         // clearly the user's own fault
  verifyFormat("auto (*p)() = f;");
}

TEST_F(FormatTest, Cpp20ModulesSupport) {
  FormatStyle Style = getLLVMStyle();
  Style.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Never;
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;

  verifyFormat("export import foo;", Style);
  verifyFormat("export import foo:bar;", Style);
  verifyFormat("export import foo.bar;", Style);
  verifyFormat("export import foo.bar:baz;", Style);
  verifyFormat("export import :bar;", Style);
  verifyFormat("export module foo:bar;", Style);
  verifyFormat("export module foo;", Style);
  verifyFormat("export module foo.bar;", Style);
  verifyFormat("export module foo.bar:baz;", Style);
  verifyFormat("export import <string_view>;", Style);
  verifyFormat("export import <Foo/Bar>;", Style);

  verifyFormat("export type_name var;", Style);
  verifyFormat("template <class T> export using A = B<T>;", Style);
  verifyFormat("export using A = B;", Style);
  verifyFormat("export int func() {\n"
               "  foo();\n"
               "}",
               Style);
  verifyFormat("export struct {\n"
               "  int foo;\n"
               "};",
               Style);
  verifyFormat("export {\n"
               "  int foo;\n"
               "};",
               Style);
  verifyFormat("export export char const *hello() { return \"hello\"; }");

  verifyFormat("import bar;", Style);
  verifyFormat("import foo.bar;", Style);
  verifyFormat("import foo:bar;", Style);
  verifyFormat("import :bar;", Style);
  verifyFormat("import /* module partition */ :bar;", Style);
  verifyFormat("import <ctime>;", Style);
  verifyFormat("import \"header\";", Style);

  verifyFormat("module foo;", Style);
  verifyFormat("module foo:bar;", Style);
  verifyFormat("module foo.bar;", Style);
  verifyFormat("module;", Style);

  verifyFormat("export namespace hi {\n"
               "const char *sayhi();\n"
               "}",
               Style);

  verifyFormat("module :private;", Style);
  verifyFormat("import <foo/bar.h>;", Style);
  verifyFormat("import foo...bar;", Style);
  verifyFormat("import ..........;", Style);
  verifyFormat("module foo:private;", Style);
  verifyFormat("import a", Style);
  verifyFormat("module a", Style);
  verifyFormat("export import a", Style);
  verifyFormat("export module a", Style);

  verifyFormat("import", Style);
  verifyFormat("module", Style);
  verifyFormat("export", Style);

  verifyFormat("import /* not keyword */ = val ? 2 : 1;");
}

TEST_F(FormatTest, CoroutineForCoawait) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("for co_await (auto x : range())\n  ;");
  verifyFormat("for (auto i : arr) {\n"
               "}",
               Style);
  verifyFormat("for co_await (auto i : arr) {\n"
               "}",
               Style);
  verifyFormat("for co_await (auto i : foo(T{})) {\n"
               "}",
               Style);
}

TEST_F(FormatTest, CoroutineCoAwait) {
  verifyFormat("int x = co_await foo();");
  verifyFormat("int x = (co_await foo());");
  verifyFormat("co_await (42);");
  verifyFormat("void operator co_await(int);");
  verifyFormat("void operator co_await(a);");
  verifyFormat("co_await a;");
  verifyFormat("co_await missing_await_resume{};");
  verifyFormat("co_await a; // comment");
  verifyFormat("void test0() { co_await a; }");
  verifyFormat("co_await co_await co_await foo();");
  verifyFormat("co_await foo().bar();");
  verifyFormat("co_await [this]() -> Task { co_return x; }");
  verifyFormat("co_await [this](int a, int b) -> Task { co_return co_await "
               "foo(); }(x, y);");

  FormatStyle Style = getLLVMStyleWithColumns(40);
  verifyFormat("co_await [this](int a, int b) -> Task {\n"
               "  co_return co_await foo();\n"
               "}(x, y);",
               Style);
  verifyFormat("co_await;");
}

TEST_F(FormatTest, CoroutineCoYield) {
  verifyFormat("int x = co_yield foo();");
  verifyFormat("int x = (co_yield foo());");
  verifyFormat("co_yield (42);");
  verifyFormat("co_yield {42};");
  verifyFormat("co_yield 42;");
  verifyFormat("co_yield n++;");
  verifyFormat("co_yield ++n;");
  verifyFormat("co_yield;");
}

TEST_F(FormatTest, CoroutineCoReturn) {
  verifyFormat("co_return (42);");
  verifyFormat("co_return;");
  verifyFormat("co_return {};");
  verifyFormat("co_return x;");
  verifyFormat("co_return co_await foo();");
  verifyFormat("co_return co_yield foo();");
}

TEST_F(FormatTest, EmptyShortBlock) {
  auto Style = getLLVMStyle();
  Style.AllowShortBlocksOnASingleLine = FormatStyle::SBS_Empty;

  verifyFormat("try {\n"
               "  doA();\n"
               "} catch (Exception &e) {\n"
               "  e.printStackTrace();\n"
               "}",
               Style);

  verifyFormat("try {\n"
               "  doA();\n"
               "} catch (Exception &e) {}",
               Style);
}

TEST_F(FormatTest, ShortTemplatedArgumentLists) {
  auto Style = getLLVMStyle();

  verifyFormat("template <> struct S : Template<int (*)[]> {};", Style);
  verifyFormat("template <> struct S : Template<int (*)[10]> {};", Style);
  verifyFormat("struct Y : X<[] { return 0; }> {};", Style);
  verifyFormat("struct Y<[] { return 0; }> {};", Style);

  verifyFormat("struct Z : X<decltype([] { return 0; }){}> {};", Style);
  verifyFormat("template <int N> struct Foo<char[N]> {};", Style);
}

TEST_F(FormatTest, MultilineLambdaInConditional) {
  auto Style = getLLVMStyleWithColumns(70);
  verifyFormat("auto aLengthyIdentifier = oneExpressionSoThatWeBreak ? []() {\n"
               "  ;\n"
               "  return 5;\n"
               "}()\n"
               "                                                     : 2;",
               Style);
  verifyFormat(
      "auto aLengthyIdentifier = oneExpressionSoThatWeBreak ? 2 : []() {\n"
      "  ;\n"
      "  return 5;\n"
      "}();",
      Style);

  Style = getLLVMStyleWithColumns(60);
  verifyFormat("auto aLengthyIdentifier = oneExpressionSoThatWeBreak\n"
               "                              ? []() {\n"
               "                                  ;\n"
               "                                  return 5;\n"
               "                                }()\n"
               "                              : 2;",
               Style);
  verifyFormat("auto aLengthyIdentifier =\n"
               "    oneExpressionSoThatWeBreak ? 2 : []() {\n"
               "      ;\n"
               "      return 5;\n"
               "    }();",
               Style);

  Style = getLLVMStyleWithColumns(40);
  verifyFormat("auto aLengthyIdentifier =\n"
               "    oneExpressionSoThatWeBreak ? []() {\n"
               "      ;\n"
               "      return 5;\n"
               "    }()\n"
               "                               : 2;",
               Style);
  verifyFormat("auto aLengthyIdentifier =\n"
               "    oneExpressionSoThatWeBreak\n"
               "        ? 2\n"
               "        : []() {\n"
               "            ;\n"
               "            return 5;\n"
               "          };",
               Style);
}

TEST_F(FormatTest, AlignAfterOpenBracketBlockIndent) {
  auto Style = getLLVMStyle();

  StringRef Short = "functionCall(paramA, paramB, paramC);\n"
                    "void functionDecl(int a, int b, int c);";

  StringRef Medium = "functionCall(paramA, paramB, paramC, paramD, paramE, "
                     "paramF, paramG, paramH, paramI);\n"
                     "void functionDecl(int argumentA, int argumentB, int "
                     "argumentC, int argumentD, int argumentE);";

  verifyFormat(Short, Style);

  StringRef NoBreak = "functionCall(paramA, paramB, paramC, paramD, paramE, "
                      "paramF, paramG, paramH,\n"
                      "             paramI);\n"
                      "void functionDecl(int argumentA, int argumentB, int "
                      "argumentC, int argumentD,\n"
                      "                  int argumentE);";

  verifyFormat(NoBreak, Medium, Style);
  verifyFormat(NoBreak,
               "functionCall(\n"
               "    paramA,\n"
               "    paramB,\n"
               "    paramC,\n"
               "    paramD,\n"
               "    paramE,\n"
               "    paramF,\n"
               "    paramG,\n"
               "    paramH,\n"
               "    paramI\n"
               ");\n"
               "void functionDecl(\n"
               "    int argumentA,\n"
               "    int argumentB,\n"
               "    int argumentC,\n"
               "    int argumentD,\n"
               "    int argumentE\n"
               ");",
               Style);

  verifyFormat("outerFunctionCall(nestedFunctionCall(argument1),\n"
               "                  nestedLongFunctionCall(argument1, "
               "argument2, argument3,\n"
               "                                         argument4, "
               "argument5));",
               Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;

  verifyFormat(Short, Style);
  verifyFormat(
      "functionCall(\n"
      "    paramA, paramB, paramC, paramD, paramE, paramF, paramG, paramH, "
      "paramI\n"
      ");\n"
      "void functionDecl(\n"
      "    int argumentA, int argumentB, int argumentC, int argumentD, int "
      "argumentE\n"
      ");",
      Medium, Style);

  Style.AllowAllArgumentsOnNextLine = false;
  Style.AllowAllParametersOfDeclarationOnNextLine = false;

  verifyFormat(Short, Style);
  verifyFormat(
      "functionCall(\n"
      "    paramA, paramB, paramC, paramD, paramE, paramF, paramG, paramH, "
      "paramI\n"
      ");\n"
      "void functionDecl(\n"
      "    int argumentA, int argumentB, int argumentC, int argumentD, int "
      "argumentE\n"
      ");",
      Medium, Style);

  Style.BinPackArguments = false;
  Style.BinPackParameters = false;

  verifyFormat(Short, Style);

  verifyFormat("functionCall(\n"
               "    paramA,\n"
               "    paramB,\n"
               "    paramC,\n"
               "    paramD,\n"
               "    paramE,\n"
               "    paramF,\n"
               "    paramG,\n"
               "    paramH,\n"
               "    paramI\n"
               ");\n"
               "void functionDecl(\n"
               "    int argumentA,\n"
               "    int argumentB,\n"
               "    int argumentC,\n"
               "    int argumentD,\n"
               "    int argumentE\n"
               ");",
               Medium, Style);

  verifyFormat("outerFunctionCall(\n"
               "    nestedFunctionCall(argument1),\n"
               "    nestedLongFunctionCall(\n"
               "        argument1,\n"
               "        argument2,\n"
               "        argument3,\n"
               "        argument4,\n"
               "        argument5\n"
               "    )\n"
               ");",
               Style);

  verifyFormat("int a = (int)b;", Style);
  verifyFormat("int a = (int)b;",
               "int a = (\n"
               "    int\n"
               ") b;",
               Style);

  verifyFormat("return (true);", Style);
  verifyFormat("return (true);",
               "return (\n"
               "    true\n"
               ");",
               Style);

  verifyFormat("void foo();", Style);
  verifyFormat("void foo();",
               "void foo(\n"
               ");",
               Style);

  verifyFormat("void foo() {}", Style);
  verifyFormat("void foo() {}",
               "void foo(\n"
               ") {\n"
               "}",
               Style);

  verifyFormat("auto string = std::string();", Style);
  verifyFormat("auto string = std::string();",
               "auto string = std::string(\n"
               ");",
               Style);

  verifyFormat("void (*functionPointer)() = nullptr;", Style);
  verifyFormat("void (*functionPointer)() = nullptr;",
               "void (\n"
               "    *functionPointer\n"
               ")\n"
               "(\n"
               ") = nullptr;",
               Style);
}

TEST_F(FormatTest, AlignAfterOpenBracketBlockIndentIfStatement) {
  auto Style = getLLVMStyle();

  verifyFormat("if (foo()) {\n"
               "  return;\n"
               "}",
               Style);

  verifyFormat("if (quitelongarg !=\n"
               "    (alsolongarg - 1)) { // ABC is a very longgggggggggggg "
               "comment\n"
               "  return;\n"
               "}",
               Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;

  verifyFormat("if (foo()) {\n"
               "  return;\n"
               "}",
               Style);

  verifyFormat("if (quitelongarg !=\n"
               "    (alsolongarg - 1)) { // ABC is a very longgggggggggggg "
               "comment\n"
               "  return;\n"
               "}",
               Style);
}

TEST_F(FormatTest, AlignAfterOpenBracketBlockIndentForStatement) {
  auto Style = getLLVMStyle();

  verifyFormat("for (int i = 0; i < 5; ++i) {\n"
               "  doSomething();\n"
               "}",
               Style);

  verifyFormat("for (int myReallyLongCountVariable = 0; "
               "myReallyLongCountVariable < count;\n"
               "     myReallyLongCountVariable++) {\n"
               "  doSomething();\n"
               "}",
               Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;

  verifyFormat("for (int i = 0; i < 5; ++i) {\n"
               "  doSomething();\n"
               "}",
               Style);

  verifyFormat("for (int myReallyLongCountVariable = 0; "
               "myReallyLongCountVariable < count;\n"
               "     myReallyLongCountVariable++) {\n"
               "  doSomething();\n"
               "}",
               Style);
}

TEST_F(FormatTest, AlignAfterOpenBracketBlockIndentInitializers) {
  auto Style = getLLVMStyleWithColumns(60);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;
  // Aggregate initialization.
  verifyFormat("int LooooooooooooooooooooooooongVariable[2] = {\n"
               "    10000000, 20000000\n"
               "};",
               Style);
  verifyFormat("SomeStruct s{\n"
               "    \"xxxxxxxxxxxxxxxx\", \"yyyyyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzzzzz\"\n"
               "};",
               Style);
  // Designated initializers.
  verifyFormat("int LooooooooooooooooooooooooongVariable[2] = {\n"
               "    [0] = 10000000, [1] = 20000000\n"
               "};",
               Style);
  verifyFormat("SomeStruct s{\n"
               "    .foo = \"xxxxxxxxxxxxx\",\n"
               "    .bar = \"yyyyyyyyyyyyy\",\n"
               "    .baz = \"zzzzzzzzzzzzz\"\n"
               "};",
               Style);
  // List initialization.
  verifyFormat("SomeStruct s{\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  verifyFormat("SomeStruct{\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  verifyFormat("new SomeStruct{\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Member initializer.
  verifyFormat("class SomeClass {\n"
               "  SomeStruct s{\n"
               "      \"xxxxxxxxxxxxx\",\n"
               "      \"yyyyyyyyyyyyy\",\n"
               "      \"zzzzzzzzzzzzz\",\n"
               "  };\n"
               "};",
               Style);
  // Constructor member initializer.
  verifyFormat("SomeClass::SomeClass : strct{\n"
               "                           \"xxxxxxxxxxxxx\",\n"
               "                           \"yyyyyyyyyyyyy\",\n"
               "                           \"zzzzzzzzzzzzz\",\n"
               "                       } {}",
               Style);
  // Copy initialization.
  verifyFormat("SomeStruct s = SomeStruct{\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Copy list initialization.
  verifyFormat("SomeStruct s = {\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Assignment operand initialization.
  verifyFormat("s = {\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Returned object initialization.
  verifyFormat("return {\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Initializer list.
  verifyFormat("auto initializerList = {\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "};",
               Style);
  // Function parameter initialization.
  verifyFormat("func({\n"
               "    \"xxxxxxxxxxxxx\",\n"
               "    \"yyyyyyyyyyyyy\",\n"
               "    \"zzzzzzzzzzzzz\",\n"
               "});",
               Style);
  // Nested init lists.
  verifyFormat("SomeStruct s = {\n"
               "    {{init1, init2, init3, init4, init5},\n"
               "     {init1, init2, init3, init4, init5}}\n"
               "};",
               Style);
  verifyFormat("SomeStruct s = {\n"
               "    {{\n"
               "         .init1 = 1,\n"
               "         .init2 = 2,\n"
               "         .init3 = 3,\n"
               "         .init4 = 4,\n"
               "         .init5 = 5,\n"
               "     },\n"
               "     {init1, init2, init3, init4, init5}}\n"
               "};",
               Style);
  verifyFormat("SomeArrayT a[3] = {\n"
               "    {\n"
               "        foo,\n"
               "        bar,\n"
               "    },\n"
               "    {\n"
               "        foo,\n"
               "        bar,\n"
               "    },\n"
               "    SomeArrayT{},\n"
               "};",
               Style);
  verifyFormat("SomeArrayT a[3] = {\n"
               "    {foo},\n"
               "    {\n"
               "        {\n"
               "            init1,\n"
               "            init2,\n"
               "            init3,\n"
               "        },\n"
               "        {\n"
               "            init1,\n"
               "            init2,\n"
               "            init3,\n"
               "        },\n"
               "    },\n"
               "    {baz},\n"
               "};",
               Style);
}

TEST_F(FormatTest, UnderstandsDigraphs) {
  verifyFormat("int arr<:5:> = {};");
  verifyFormat("int arr[5] = <%%>;");
  verifyFormat("int arr<:::qualified_variable:> = {};");
  verifyFormat("int arr[::qualified_variable] = <%%>;");
  verifyFormat("%:include <header>");
  verifyFormat("%:define A x##y");
  verifyFormat("#define A x%:%:y");
}

TEST_F(FormatTest, AlignArrayOfStructuresLeftAlignmentNonSquare) {
  auto Style = getLLVMStyle();
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Left;
  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.AlignConsecutiveDeclarations.Enabled = true;

  // The AlignArray code is incorrect for non square Arrays and can cause
  // crashes, these tests assert that the array is not changed but will
  // also act as regression tests for when it is properly fixed
  verifyFormat("struct test demo[] = {\n"
               "    {1, 2},\n"
               "    {3, 4, 5},\n"
               "    {6, 7, 8}\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {1, 2, 3, 4, 5},\n"
               "    {3, 4, 5},\n"
               "    {6, 7, 8}\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {1, 2, 3, 4, 5},\n"
               "    {3, 4, 5},\n"
               "    {6, 7, 8, 9, 10, 11, 12}\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {1, 2, 3},\n"
               "    {3, 4, 5},\n"
               "    {6, 7, 8, 9, 10, 11, 12}\n"
               "};",
               Style);

  verifyFormat("S{\n"
               "    {},\n"
               "    {},\n"
               "    {a, b}\n"
               "};",
               Style);
  verifyFormat("S{\n"
               "    {},\n"
               "    {},\n"
               "    {a, b},\n"
               "};",
               Style);
  verifyFormat("void foo() {\n"
               "  auto thing = test{\n"
               "      {\n"
               "       {13}, {something}, // A\n"
               "      }\n"
               "  };\n"
               "}",
               "void foo() {\n"
               "  auto thing = test{\n"
               "      {\n"
               "       {13},\n"
               "       {something}, // A\n"
               "      }\n"
               "  };\n"
               "}",
               Style);
}

TEST_F(FormatTest, AlignArrayOfStructuresRightAlignmentNonSquare) {
  auto Style = getLLVMStyle();
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Right;
  Style.AlignConsecutiveAssignments.Enabled = true;
  Style.AlignConsecutiveDeclarations.Enabled = true;

  // The AlignArray code is incorrect for non square Arrays and can cause
  // crashes, these tests assert that the array is not changed but will
  // also act as regression tests for when it is properly fixed
  verifyFormat("struct test demo[] = {\n"
               "    {1, 2},\n"
               "    {3, 4, 5},\n"
               "    {6, 7, 8}\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {1, 2, 3, 4, 5},\n"
               "    {3, 4, 5},\n"
               "    {6, 7, 8}\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {1, 2, 3, 4, 5},\n"
               "    {3, 4, 5},\n"
               "    {6, 7, 8, 9, 10, 11, 12}\n"
               "};",
               Style);
  verifyFormat("struct test demo[] = {\n"
               "    {1, 2, 3},\n"
               "    {3, 4, 5},\n"
               "    {6, 7, 8, 9, 10, 11, 12}\n"
               "};",
               Style);

  verifyFormat("S{\n"
               "    {},\n"
               "    {},\n"
               "    {a, b}\n"
               "};",
               Style);
  verifyFormat("S{\n"
               "    {},\n"
               "    {},\n"
               "    {a, b},\n"
               "};",
               Style);
  verifyFormat("void foo() {\n"
               "  auto thing = test{\n"
               "      {\n"
               "       {13}, {something}, // A\n"
               "      }\n"
               "  };\n"
               "}",
               "void foo() {\n"
               "  auto thing = test{\n"
               "      {\n"
               "       {13},\n"
               "       {something}, // A\n"
               "      }\n"
               "  };\n"
               "}",
               Style);
}

TEST_F(FormatTest, FormatsVariableTemplates) {
  verifyFormat("inline bool var = is_integral_v<int> && is_signed_v<int>;");
  verifyFormat("template <typename T> "
               "inline bool var = is_integral_v<T> && is_signed_v<T>;");
}

TEST_F(FormatTest, RemoveSemicolon) {
  FormatStyle Style = getLLVMStyle();
  Style.RemoveSemicolon = true;

  verifyFormat("int max(int a, int b) { return a > b ? a : b; }",
               "int max(int a, int b) { return a > b ? a : b; };", Style);

  verifyFormat("int max(int a, int b) { return a > b ? a : b; }",
               "int max(int a, int b) { return a > b ? a : b; };;", Style);

  verifyFormat("class Foo {\n"
               "  int getSomething() const { return something; }\n"
               "};",
               "class Foo {\n"
               "  int getSomething() const { return something; };\n"
               "};",
               Style);

  verifyFormat("class Foo {\n"
               "  int getSomething() const { return something; }\n"
               "};",
               "class Foo {\n"
               "  int getSomething() const { return something; };;\n"
               "};",
               Style);

  verifyFormat("for (;;) {\n"
               "}",
               Style);

  verifyFormat("class [[deprecated(\"\")]] C {\n"
               "  int i;\n"
               "};",
               Style);

  verifyFormat("struct EXPORT_MACRO [[nodiscard]] C {\n"
               "  int i;\n"
               "};",
               Style);

  verifyIncompleteFormat("class C final [[deprecated(l]] {});", Style);

  // These tests are here to show a problem that may not be easily
  // solved, our implementation to remove semicolons is only as good
  // as our FunctionLBrace detection and this fails for empty braces
  // because we can't distringuish this from a bracelist.
  // We will enable when that is resolved.
#if 0
  verifyFormat("void main() {}", "void main() {};", Style);
  verifyFormat("void foo() {} //\n"
               "int bar;",
               "void foo() {}; //\n"
               "; int bar;",
               Style);
#endif
}

TEST_F(FormatTest, BreakAfterAttributes) {
  FormatStyle Style = getLLVMStyle();
  EXPECT_EQ(Style.BreakAfterAttributes, FormatStyle::ABS_Never);

  constexpr StringRef Code("[[nodiscard]] inline int f(int &i);\n"
                           "[[foo([[]])]] [[nodiscard]]\n"
                           "int g(int &i);\n"
                           "[[nodiscard]]\n"
                           "inline int f(int &i) {\n"
                           "  i = 1;\n"
                           "  return 0;\n"
                           "}\n"
                           "[[foo([[]])]] [[nodiscard]] int g(int &i) {\n"
                           "  i = 0;\n"
                           "  return 1;\n"
                           "}");

  verifyFormat("[[nodiscard]] inline int f(int &i);\n"
               "[[foo([[]])]] [[nodiscard]] int g(int &i);\n"
               "[[nodiscard]] inline int f(int &i) {\n"
               "  i = 1;\n"
               "  return 0;\n"
               "}\n"
               "[[foo([[]])]] [[nodiscard]] int g(int &i) {\n"
               "  i = 0;\n"
               "  return 1;\n"
               "}",
               Code, Style);

  Style.BreakAfterAttributes = FormatStyle::ABS_Leave;
  verifyNoChange(Code, Style);

  Style.BreakAfterAttributes = FormatStyle::ABS_Always;
  verifyFormat("[[nodiscard]]\n"
               "inline int f(int &i);\n"
               "[[foo([[]])]] [[nodiscard]]\n"
               "int g(int &i);\n"
               "[[nodiscard]]\n"
               "inline int f(int &i) {\n"
               "  i = 1;\n"
               "  return 0;\n"
               "}\n"
               "[[foo([[]])]] [[nodiscard]]\n"
               "int g(int &i) {\n"
               "  i = 0;\n"
               "  return 1;\n"
               "}",
               Code, Style);

  constexpr StringRef CtorDtorCode("struct Foo {\n"
                                   "  [[deprecated]] Foo();\n"
                                   "  [[deprecated]] Foo() {}\n"
                                   "  [[deprecated]] ~Foo();\n"
                                   "  [[deprecated]] ~Foo() {}\n"
                                   "  [[deprecated]] void f();\n"
                                   "  [[deprecated]] void f() {}\n"
                                   "};\n"
                                   "[[deprecated]] Bar::Bar() {}\n"
                                   "[[deprecated]] Bar::~Bar() {}\n"
                                   "[[deprecated]] void g() {}");
  verifyFormat("struct Foo {\n"
               "  [[deprecated]]\n"
               "  Foo();\n"
               "  [[deprecated]]\n"
               "  Foo() {}\n"
               "  [[deprecated]]\n"
               "  ~Foo();\n"
               "  [[deprecated]]\n"
               "  ~Foo() {}\n"
               "  [[deprecated]]\n"
               "  void f();\n"
               "  [[deprecated]]\n"
               "  void f() {}\n"
               "};\n"
               "[[deprecated]]\n"
               "Bar::Bar() {}\n"
               "[[deprecated]]\n"
               "Bar::~Bar() {}\n"
               "[[deprecated]]\n"
               "void g() {}",
               CtorDtorCode, Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Linux;
  verifyFormat("struct Foo {\n"
               "  [[deprecated]]\n"
               "  Foo();\n"
               "  [[deprecated]]\n"
               "  Foo()\n"
               "  {\n"
               "  }\n"
               "  [[deprecated]]\n"
               "  ~Foo();\n"
               "  [[deprecated]]\n"
               "  ~Foo()\n"
               "  {\n"
               "  }\n"
               "  [[deprecated]]\n"
               "  void f();\n"
               "  [[deprecated]]\n"
               "  void f()\n"
               "  {\n"
               "  }\n"
               "};\n"
               "[[deprecated]]\n"
               "Bar::Bar()\n"
               "{\n"
               "}\n"
               "[[deprecated]]\n"
               "Bar::~Bar()\n"
               "{\n"
               "}\n"
               "[[deprecated]]\n"
               "void g()\n"
               "{\n"
               "}",
               CtorDtorCode, Style);
}

TEST_F(FormatTest, InsertNewlineAtEOF) {
  FormatStyle Style = getLLVMStyle();
  Style.InsertNewlineAtEOF = true;

  verifyNoChange("int i;\n", Style);
  verifyFormat("int i;\n", "int i;", Style);
}

TEST_F(FormatTest, KeepEmptyLinesAtEOF) {
  FormatStyle Style = getLLVMStyle();
  Style.KeepEmptyLinesAtEOF = true;

  const StringRef Code{"int i;\n\n"};
  verifyNoChange(Code, Style);
  verifyFormat(Code, "int i;\n\n\n", Style);
}

TEST_F(FormatTest, SpaceAfterUDL) {
  verifyFormat("auto c = (4s).count();");
  verifyFormat("auto x = 5s .count() == 5;");
}

TEST_F(FormatTest, InterfaceAsClassMemberName) {
  verifyFormat("class Foo {\n"
               "  int interface;\n"
               "  Foo::Foo(int iface) : interface{iface} {}\n"
               "}");
}

TEST_F(FormatTest, PreprocessorOverlappingRegions) {
  verifyFormat("#ifdef\n\n"
               "#else\n"
               "#endif",
               "#ifdef \n"
               "    \n"
               "\n"
               "#else \n"
               "#endif ",
               getGoogleStyle());
}

TEST_F(FormatTest, RemoveParentheses) {
  FormatStyle Style = getLLVMStyle();
  EXPECT_EQ(Style.RemoveParentheses, FormatStyle::RPS_Leave);

  Style.RemoveParentheses = FormatStyle::RPS_MultipleParentheses;
  verifyFormat("int x __attribute__((aligned(16))) = 0;", Style);
  verifyFormat("decltype((foo->bar)) baz;", Style);
  verifyFormat("class __declspec(dllimport) X {};",
               "class __declspec((dllimport)) X {};", Style);
  verifyFormat("int x = (({ 0; }));", "int x = ((({ 0; })));", Style);
  verifyFormat("while (a)\n"
               "  b;",
               "while (((a)))\n"
               "  b;",
               Style);
  verifyFormat("while ((a = b))\n"
               "  c;",
               "while (((a = b)))\n"
               "  c;",
               Style);
  verifyFormat("if (a)\n"
               "  b;",
               "if (((a)))\n"
               "  b;",
               Style);
  verifyFormat("if constexpr ((a = b))\n"
               "  c;",
               "if constexpr (((a = b)))\n"
               "  c;",
               Style);
  verifyFormat("if (({ a; }))\n"
               "  b;",
               "if ((({ a; })))\n"
               "  b;",
               Style);
  verifyFormat("return (0);", "return (((0)));", Style);
  verifyFormat("return (({ 0; }));", "return ((({ 0; })));", Style);

  Style.RemoveParentheses = FormatStyle::RPS_ReturnStatement;
  verifyFormat("return 0;", "return (0);", Style);
  verifyFormat("co_return 0;", "co_return ((0));", Style);
  verifyFormat("return 0;", "return (((0)));", Style);
  verifyFormat("return ({ 0; });", "return ((({ 0; })));", Style);

  Style.ColumnLimit = 25;
  verifyFormat("return (a + b) - (c + d);",
               "return (((a + b)) -\n"
               "        ((c + d)));",
               Style);
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang
