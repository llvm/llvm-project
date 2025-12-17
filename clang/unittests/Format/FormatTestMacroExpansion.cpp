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
  verifyFormat("ASSIGN_OR_RETURN(MySomewhatLongType *variable,\n"
               "                 MySomewhatLongFunction(SomethingElse()), "
               "ReturnMe());",
               Style);

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
    }
) // namespace k
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
  verifyFormat("void f() {\n"
               "  FN(a *b);\n"
               "  };\n"
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

} // namespace
} // namespace test
} // namespace format
} // namespace clang
