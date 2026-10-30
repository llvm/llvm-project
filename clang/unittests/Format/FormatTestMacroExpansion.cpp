//===- unittest/Format/FormatMacroExpansion.cpp - Formatting unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "format-test-macro-expansion"

namespace clang {
namespace format {
namespace test {
namespace {

class FormatTestMacroExpansion : public FormatTestBase {};

TEST_F(FormatTestMacroExpansion, UnexpandConfiguredMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("CLASS=class C {");
  Style.Macros.push_back("SEMI=;");
  Style.Macros.push_back("STMT=f();");
  Style.Macros.push_back("ID(x)=x");
  Style.Macros.push_back("ID3(x, y, z)=x y z");
  Style.Macros.push_back("CALL(x)=f([] { x })");
  Style.Macros.push_back("ASSIGN_OR_RETURN(a, b)=a = (b)");
  Style.Macros.push_back("ASSIGN_OR_RETURN(a, b, c)=a = (b); if (x) return c");
  Style.Macros.push_back("MOCK_METHOD(r, n, a, s)=r n a s");

  verifyFormat("ID(nested(a(b, c), d))", Style);
  verifyFormat("CLASS\n"
               "  a *b;\n"
               "};",
               Style);
  verifyFormat("SEMI\n"
               "SEMI\n"
               "SEMI",
               Style);
  verifyFormat("STMT\n"
               "STMT\n"
               "STMT",
               Style);
  verifyFormat("void f() { ID(a *b); }", Style);
  verifyFormat("ID(\n"
               "    {\n"
               "      ID(a *b);\n"
               "    });",
               Style);
  verifyIncompleteFormat("ID3({, ID(a *b),\n"
                         "    ;\n"
                         "  });",
                         Style);

  verifyFormat("ID(CALL(CALL(return a * b;)));", Style);

  verifyFormat("ASSIGN_OR_RETURN(MySomewhatLongType *variable,\n"
               "                 MySomewhatLongFunction(SomethingElse()));",
               Style);
  verifyFormat(
      "ASSIGN_OR_RETURN(MySomewhatLongType *variable,\n"
      "                 MySomewhatLongFunction(SomethingElse()), RetMe());",
      Style);

  verifyFormat(
      "void f() {\n"
      "  ASSIGN_OR_RETURN(MySomewhatLongType* variable,\n"
      "                   MySomewhatLongFunction(SomethingElse()));\n"
      "  ASSIGN_OR_RETURN(MySomewhatLongType* variable,\n"
      "                   MySomewhatLongFunction(SomethingElse()), RetMe());",
      getGoogleStyle());

  verifyFormat(R"(
#define MACRO(a, b) ID(a + b)
)",
               Style);
  EXPECT_EQ(R"(
int a;
int b;
int c;
int d;
int e;
int f;
ID(
    namespace foo {
    int a;
    }) // namespace k
)",
            format(R"(
int a;
int b;
int c;
int d;
int e;
int f;
ID(namespace foo { int a; })  // namespace k
)",
                   Style));
  verifyFormat(R"(ID(
    //
    ({ ; }))
)",
               Style);

  Style.ColumnLimit = 35;
  // FIXME: Arbitrary formatting of macros where the end of the logical
  // line is in the middle of a macro call are not working yet.
  verifyFormat(R"(ID(
    void f();
    void)
ID(g) ID(()) ID(
    ;
    void g();)
)",
               Style);

  Style.ColumnLimit = 10;
  verifyFormat("STMT\n"
               "STMT\n"
               "STMT",
               Style);

  EXPECT_EQ(R"(
ID(CALL(CALL(
    a *b)));
)",
            format(R"(
ID(CALL(CALL(a * b)));
)",
                   Style));

  // FIXME: If we want to support unbalanced braces or parens from macro
  // expansions we need to re-think how we propagate errors in
  // TokenAnnotator::parseLine; for investigation, switching the inner loop of
  // TokenAnnotator::parseLine to return LT_Other instead of LT_Invalid in case
  // of !consumeToken() changes the formatting of the test below and makes it
  // believe it has a fully correct formatting.
  EXPECT_EQ(R"(
ID3(
    {
      CLASS
        a *b;
      };
    },
    ID(x *y);
    ,
    STMT
    STMT
    STMT)
void f();
)",
            format(R"(
ID3({CLASS a*b; };}, ID(x*y);, STMT STMT STMT)
void f();
)",
                   Style));

  verifyFormat("ID(a(\n"
               "#ifdef A\n"
               "    b, c\n"
               "#else\n"
               "    d(e)\n"
               "#endif\n"
               "    ))",
               Style);
  Style.ColumnLimit = 80;
  verifyFormat(R"(ASSIGN_OR_RETURN(
    // Comment
    a b, c);
)",
               Style);
  Style.ColumnLimit = 30;
  verifyFormat(R"(ASSIGN_OR_RETURN(
    // Comment
    //
    a b,
    xxxxxxxxxxxx(
        yyyyyyyyyyyyyyyyy,
        zzzzzzzzzzzzzzzzzz),
    f([]() {
      a();
      b();
    }));
)",
               Style);
  verifyFormat(R"(int a = []() {
  ID(
      x;
      y;
      z;)
  ;
}();
)",
               Style);
  EXPECT_EQ(
      R"(ASSIGN_OR_RETURN((
====
#))
})",
      format(R"(ASSIGN_OR_RETURN((
====
#))
})",
             Style, SC_ExpectIncomplete));
  EXPECT_EQ(R"(ASSIGN_OR_RETURN(
}
(
====
#),
    a))",
            format(R"(ASSIGN_OR_RETURN(
}
(
====
#),
a))",
                   Style, SC_ExpectIncomplete));
  EXPECT_EQ(R"(ASSIGN_OR_RETURN(a
//
====
#
                 <))",
            format(R"(ASSIGN_OR_RETURN(a
//
====
#
                 <))",
                   Style));
  verifyFormat("class C {\n"
               "  MOCK_METHOD(R, f,\n"
               "              (a *b, c *d),\n"
               "              (override));\n"
               "};",
               Style);
}

TEST_F(FormatTestMacroExpansion, KeepParensWhenExpandingObjectLikeMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("FN=class C { int f");
  Style.Macros.push_back("FN_ARG(x)=class C { int f(x)");
  verifyFormat("void f() {\n"
               "  FN(a *b);\n"
               "  };\n"
               "}",
               Style);
  verifyFormat("void f() {\n"
               "  FN_ARG(a *b);\n"
               "  };\n"
               "}",
               Style);
}

TEST_F(FormatTestMacroExpansion, ShortFunctionsAndBlocksWithMacroExpansion) {
  FormatStyle Style = getLLVMStyleWithColumns(35);
  Style.AllowShortIfStatementsOnASingleLine = FormatStyle::SIS_WithoutElse;
  Style.Macros = {
      "ID(x)=x",
      "VERY_LONG_MACRO_NAME(x)=x",
      "EXPANDS_LONG(x)=very_long_function_call(x)",
      "IF_ERROR_RETURN(x)=if (x) return x",
      "ASSIGN_OR_RETURN(a, b, c)=a = (b); if (x) return c",
      "STMTS=f(); g();",
      "STMTS_NO_SEMI=f(); g()",
  };

  verifyFormat("void f() { ID(a *b); }", Style);
  verifyFormat("void f() { int x = ID(a * b); }", Style);
  verifyFormat("void f() {\n"
               "  VERY_LONG_MACRO_NAME(a *b);\n"
               "}",
               Style);
  verifyFormat("void f() {\n"
               "  int x = VERY_LONG_MACRO_NAME(x);\n"
               "}",
               Style);
  verifyFormat("void f() {\n"
               "  EXPANDS_LONG(a * b) +\n"
               "      c;\n"
               "}",
               Style);
  verifyFormat("void f() { IF_ERROR_RETURN(x); }", Style);
  verifyFormat("void f() {\n"
               "  IF_ERROR_RETURN(a * b);\n"
               "}",
               Style);
  verifyFormat("void f() {\n"
               "  ASSIGN_OR_RETURN(v, F(), R());\n"
               "}",
               Style);
  FormatStyle Style50 = Style;
  Style50.ColumnLimit = 50;
  verifyFormat("void f() { ASSIGN_OR_RETURN(v, F(), R()); }", Style50);
  verifyFormat("void f() {\n"
               "  STMTS\n"
               "}",
               Style);
  verifyFormat("void f() {\n"
               "  STMTS_NO_SEMI;\n"
               "}",
               Style);

  verifyFormat("void f() {\n"
               "  if (a) ID(b *c);\n"
               "  if (a)\n"
               "    VERY_LONG_MACRO_NAME(b *c);\n"
               "  if (a) IF_ERROR_RETURN(b * c);\n"
               "}",
               Style);

  verifyFormat("void f(int very_long_param_1,\n"
               "       int very_long_param_2) {}\n"
               "class SomeVeryLongClassName\n"
               "    : public BaseClassName {};\n"
               "void g() { ID(a *b); }",
               "void f(int very_long_param_1,\n"
               "       int very_long_param_2) {\n"
               "}\n"
               "class SomeVeryLongClassName\n"
               "    : public BaseClassName {\n"
               "};\n"
               "void g() {\n"
               "  ID(a *b);\n"
               "}",
               Style);
}

TEST_F(FormatTestMacroExpansion, DoesNotExpandFunctionLikeMacrosWithoutParens) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("CLASS()=class C {");
  verifyFormat("CLASS void f();\n"
               "}\n"
               ";",
               Style);
}

TEST_F(FormatTestMacroExpansion,
       ContinueFormattingAfterUnclosedParensAfterObjectLikeMacro) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("O=class {");
  verifyIncompleteFormat("O(auto x = [](){\n"
                         "    f();}",
                         Style);
}

TEST_F(FormatTestMacroExpansion, CommaAsOperator) {
  FormatStyle Style = getGoogleStyleWithColumns(42);
  Style.Macros.push_back("MACRO(a, b, c)=a=(b); if(x) c");
  verifyFormat("MACRO(auto a,\n"
               "      looooongfunction(first, second,\n"
               "                       third),\n"
               "      fourth);",
               Style);
}

TEST_F(FormatTestMacroExpansion, ForcedBreakDiffers) {
  FormatStyle Style = getGoogleStyleWithColumns(40);
  Style.Macros.push_back("MACRO(a, b)=a=(b)");
  verifyFormat("//\n"
               "MACRO(const type variable,\n"
               "      functtioncall(\n"
               "          first, longsecondarg, third));",
               Style);
}

TEST_F(FormatTestMacroExpansion,
       PreferNotBreakingBetweenReturnTypeAndFunction) {
  FormatStyle Style = getGoogleStyleWithColumns(22);
  Style.Macros.push_back("MOCK_METHOD(r, n, a)=r n a");
  // In the expanded code, we parse a full function signature, and afterwards
  // know that we prefer not to break before the function name.
  verifyFormat("MOCK_METHOD(\n"
               "    type, variable,\n"
               "    (type));",
               Style);
}

TEST_F(FormatTestMacroExpansion, IndentChildrenWithinMacroCall) {
  FormatStyle Style = getGoogleStyleWithColumns(22);
  Style.Macros.push_back("MACRO(a, b)=a=(b)");
  verifyFormat("void f() {\n"
               "  MACRO(a b, call([] {\n"
               "          if (expr) {\n"
               "            indent();\n"
               "          }\n"
               "        }));\n"
               "}",
               Style);
}

TEST_F(FormatTestMacroExpansion, PPDirectiveInDiscardedMacroArgs) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros = {"A=a", "ID(x)=x", "PAIR(x, y)=x y", "STMT=f();", "EMPTY="};

  verifyIncompleteFormat("A(\n"
                         "#endif",
                         Style);
  verifyIncompleteFormat("ID(\n"
                         "#endif",
                         Style);
  verifyIncompleteFormat("ID(\n"
                         "#define X 1",
                         Style);
  verifyFormat("A(\n"
               "#if X\n"
               "    b;\n"
               "#endif\n"
               ")",
               Style);
  verifyFormat("ID(a,\n"
               "#if X\n"
               "   b\n"
               "#endif\n"
               ");",
               Style);
  verifyFormat("PAIR(\n"
               "#define X ,\n"
               "    a)",
               Style);
  verifyFormat("ID(\n"
               "#if 0\n"
               ",\n"
               "#endif\n"
               "    if (a) {\n"
               "      f();\n"
               "    })",
               Style);
  verifyFormat("STMT\n"
               "#define F(x) g(x)\n"
               "b;",
               "STMT\n"
               "#define F(x) g( x )\n"
               "b;",
               Style);
  verifyFormat("EMPTY(\n"
               "#define F(x) g(x)\n"
               "1)",
               "EMPTY(\n"
               "#define F(x)  g(x)\n"
               "1)",
               Style);
  EXPECT_EQ("A(\n"
            "ID(\n"
            "#define F(x) g(x)\n"
            "))",
            format("A(ID(\n"
                   "#define F(x)  g(x)\n"
                   "))",
                   Style, SC_ExpectIncomplete));

  Style.IndentPPDirectives = FormatStyle::PPDIS_BeforeHash;
  verifyFormat("#if OUTER\n"
               "EMPTY(\n"
               "  #define X 1\n"
               ")\n"
               "#endif",
               Style);
  verifyFormat("void f() {\n"
               "  if (x) {\n"
               "    ID(a,\n"
               "#if Y\n"
               "  #define Z 1\n"
               "#endif\n"
               "    );\n"
               "  }\n"
               "}",
               Style);
}

TEST_F(FormatTestMacroExpansion, PPDirectiveBeforeEmptyExpansion) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("ID(x)=x");
  verifyFormat("#define X 1\n"
               "ID()",
               "#define X   1\n"
               "ID()",
               Style);
  verifyFormat("#define X 1\n"
               "ID()",
               "#define X   1\n"
               "ID()",
               Style, {tooling::Range(0, 13)}); // line 1
}

TEST_F(FormatTestMacroExpansion, ObjectLikeMacroCalledWithArgsDoesNotHang) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("CASE=case");
  verifyNoCrash("const char *fct(int wki) {\n"
                "  switch (wki) {\n"
                "    CASE(1, \"1\");\n"
                "    CASE(2, \"2\");\n"
                "    default:\n"
                "      return \"123\";\n"
                "  }\n"
                "}",
                Style);
  verifyNoCrash("CASE(1, \"1\");", Style);
}

TEST_F(FormatTestMacroExpansion, ExpandsAdjacentMacroCallsInOrder) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("ID(x)=x");

  verifyFormat("ID(a;)\n"
               "ID(\n"
               "    // c\n"
               "    b;)",
               Style);
}

TEST_F(FormatTestMacroExpansion, TokensAfterMacroCallAreNotPartOfCall) {
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("ID(x)=x");

  verifyFormat("ID(a;) // c\n"
               "ID(b;)",
               "ID(\n"
               "    a;) // c\n"
               "ID(b;)",
               Style);
  verifyFormat("int x = ID(1) // c\n"
               "        + 2;",
               Style);
  verifyFormat("ID(a;)\n"
               "#if X\n"
               "int b;\n"
               "#endif",
               Style);
}

TEST_F(FormatTestMacroExpansion,
       SeparateDefinitionBlocksWithMacroExpandedStruct) {
  // Reproduces https://llvm.org/PR78263
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterStruct = true;
  Style.SeparateDefinitionBlocks = FormatStyle::SDS_Always;
  Style.Macros.push_back("PACK(x)=x");

  verifyFormat("static bool foo;\n"
               "\n"
               "//! \\brief Blah\n"
               "PACK(\n"
               "    struct Foo\n"
               "    {\n"
               "      uint16_t a;\n"
               "    });\n"
               "\n"
               "PACK(\n"
               "    struct Bar\n"
               "    {\n"
               "      uint32_t a;\n"
               "    });\n"
               "\n"
               "PACK(\n"
               "    struct Baz\n"
               "    {\n"
               "      uint16_t a;\n"
               "    });",
               "static bool foo;\n"
               "\n"
               "//! \\brief Blah\n"
               "PACK(struct Foo\n"
               "{\n"
               "  uint16_t a;\n"
               "});\n"
               "\n"
               "PACK(struct Bar\n"
               "{\n"
               "  uint32_t a;\n"
               "});\n"
               "\n"
               "PACK(struct Baz\n"
               "{\n"
               "  uint16_t a;\n"
               "});",
               Style);
}

TEST_F(FormatTestMacroExpansion, MacroExpansionWithIfElseBranches) {
  // Reproduces https://llvm.org/PR233300
  FormatStyle Style = getLLVMStyle();
  Style.Macros.push_back("M(x)=x");

  verifyFormat("#if A\n"
               "auto l = []\n"
               "#else\n"
               "if\n"
               "#endif\n"
               "    M((int a) {});",
               Style);

  Style.Macros = {"M(a,b,c)=a b(c)"};
  verifyNoCrash("({\n"
                "#if A\n"
                "{[]\n"
                "#else\n"
                "if\n"
                "#endif\n"
                "M(, (int a){int b, ))",
                Style);
}

TEST_F(FormatTestMacroExpansion, RawStringLiteralInMacroArg) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_Cpp);
  Style.Macros.push_back("ASSIGN_OR_RETURN(a, b)=a = (b)");

  StringRef Code1 =
      "void f() {\n"
      "  ASSIGN_OR_RETURN(FlowConfig flow_config,\n"
      "                   ParseTextProto<FlowConfig>(Substitute(\n"
      "                       R\"pb(\n"
      "                         key: value\n"
      "                       )pb\",\n"
      "                       arg1, arg2, arg3, arg4)));\n"
      "}";
  verifyFormat(Code1, Code1, Style);

  StringRef Code2 =
      "void f() {\n"
      "  ASSIGN_OR_RETURN(\n"
      "      FlowConfig flow_config,\n"
      "      "
      "proto2::contrib::parse_proto::ParseTextProto<FlowConfig>(absl::"
      "Substitute(\n"
      "          R\"pb(\n"
      "            key: value\n"
      "          )pb\",\n"
      "          kPreprocessingFlowConfig, kAudioConfig, kPostConfig)));\n"
      "}";
  verifyFormat(Code2, Code2, Style);

  StringRef Code3 =
      "void f() {\n"
      "  ASSIGN_OR_RETURN(\n"
      "      // comment\n"
      "      TestCase test_case, ParseTextProto<TestCase>(R\"pb(\n"
      "        name: \"foo\"\n"
      "      )pb\"));\n"
      "}";
  verifyFormat(Code3, Code3, Style);
}

TEST_F(FormatTestMacroExpansion, NestedMacrosInLambdas) {
  // These are tests for regressions reported in
  // https://github.com/llvm/llvm-project/pull/169037#issuecomment-4056423543
  // clang-format off
  verifyFormat(
      "void f() {\n"
      "  RETURN_IF_ERROR(Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "      a,\n"
      "      []() {\n"
      "        if (z()) {\n"
      "          ASSIGN_OR_RETURN(w, Wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww());\n"
      "        }\n"
      "      }));\n"
      "}",
      getGoogleStyle());

  verifyFormat(
      "void g() {\n"
      "  RETURN_IF_ERROR(q(\n"
      "      w,\n"
      "      []() {\n"
      "        if (z()) {\n"
      "          ASSIGN_OR_RETURN(w, Wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww());\n"
      "        }\n"
      "      }));\n"
      "}",
      getGoogleStyle());

  verifyFormat(
      "void f() {\n"
      "  ABSL_RETURN_IF_ERROR(\n"
      "      Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "          a,\n"
      "          []() {\n"
      "            if (z()) {\n"
      "              ABSL_ASSIGN_OR_RETURN(w, Wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww());\n"
      "            }\n"
      "          }));\n"
      "}",
      getGoogleStyle());

  verifyFormat(
      "void g() {\n"
      "  ABSL_RETURN_IF_ERROR(q(\n"
      "      w,\n"
      "      []() {\n"
      "        if (z()) {\n"
      "          ABSL_ASSIGN_OR_RETURN(w, Wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww());\n"
      "        }\n"
      "      }));\n"
      "}",
      getGoogleStyle());
  // clang-format on
}

// Verify that the predefined Google macros cause `*` to be treated as a
// pointer declaration rather than multiplication.
TEST_F(FormatTestMacroExpansion, GoogleMacrosFormatsAsPtrDecl) {
  verifyFormat("void f() { ASSIGN_OR_RETURN(Type* x, F()); }",
               getGoogleStyle());
  verifyFormat("void f() { ABSL_ASSIGN_OR_RETURN(Type* x, F()); }",
               getGoogleStyle());
  verifyFormat("void f() { ASSERT_OK_AND_ASSIGN(Type* x, F()); }",
               getGoogleStyle());
  verifyFormat("void f() { ABSL_ASSERT_OK_AND_ASSIGN(Type* x, F()); }",
               getGoogleStyle());
}

// Macros whose expansion contains multiple statements or control flow (e.g.,
// `if`) produce multiple unwrapped lines, which prevents short function
// merging.
//
// FIXME: These should format onto a single line (e.g.,
// `void f() { RETURN_IF_ERROR(F()); }`).
TEST_F(FormatTestMacroExpansion, GoogleMacrosShortFunctionFormattingIsBroken) {
  // RETURN_IF_ERROR expands to `if (x) return expr`.
  verifyFormat("void f() {\n"
               "  RETURN_IF_ERROR(F());\n"
               "}",
               getGoogleStyle());
  verifyFormat("void f() {\n"
               "  ABSL_RETURN_IF_ERROR(F());\n"
               "}",
               getGoogleStyle());
  // 3-arg ASSIGN_OR_RETURN expands to `a = (b); if (x) return c`.
  verifyFormat("void f() {\n"
               "  ASSIGN_OR_RETURN(auto x, F(), R());\n"
               "}",
               getGoogleStyle());
  verifyFormat("void f() {\n"
               "  ABSL_ASSIGN_OR_RETURN(auto x, F(), R());\n"
               "}",
               getGoogleStyle());
}
} // namespace
} // namespace test
} // namespace format
} // namespace clang
