//===- unittest/Format/AlignBracketsTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "align-brackets-test"

namespace clang {
namespace format {
namespace test {
namespace {

class AlignBracketsTest : public FormatTestBase {};

TEST_F(AlignBracketsTest, AlignsAfterOpenBracket) {
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

  Style.ColumnLimit = 30;
  verifyFormat("for (int foo = 0; foo < FOO;\n"
               "    ++foo) {\n"
               "  bar(foo);\n"
               "}",
               Style);
  Style.ColumnLimit = 80;

  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  Style.BinPackArguments = false;
  Style.BinPackParameters = FormatStyle::BPPS_OnePerLine;
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
  verifyFormat(
      "fooooooooooo(new BARRRRRRRRR(\n"
      "    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXZZZZZZZZZZZZZZZZZZZZZZZZZ()));",
      Style);
  verifyFormat(
      "fooooooooooo(::new BARRRRRRRRR(\n"
      "    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXZZZZZZZZZZZZZZZZZZZZZZZZZ()));",
      Style);
  verifyFormat(
      "fooooooooooo(new FOO::BARRRR(\n"
      "    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXZZZZZZZZZZZZZZZZZZZZZZZZZ()));",
      Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;
  Style.BinPackArguments = false;
  Style.BinPackParameters = FormatStyle::BPPS_OnePerLine;
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
  verifyFormat("void foo(\n"
               "    void (*foobarpntr)(\n"
               "        aaaaaaaaaaaaaaaaaa *,\n"
               "        bbbbbbbbbbbbbb *,\n"
               "        cccccccccccccccccccc *,\n"
               "        dddddddddddddddddd *\n"
               "    )\n"
               ");",
               Style);
  verifyFormat("aaaaaaa<bbbbbbbb> const aaaaaaaaaa{\n"
               "    aaaaaaaaaaaaa(aaaaaaaaaaa, aaaaaaaaaaaaaaaa)\n"
               "};",
               Style);

  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    const bool &aaaaaaaaa, const void *aaaaaaaaaa\n"
               ") const {\n"
               "  return true;\n"
               "}",
               Style);
  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    const bool &aaaaaaaaaa, const void *aaaaaaaaaa\n"
               ") const;",
               Style);
  verifyFormat("void aaaaaaaaa(\n"
               "    int aaaaaa, int bbbbbb, int cccccc, int dddddddddd\n"
               ") const noexcept -> std::vector<of_very_long_type>;",
               Style);
  verifyFormat(
      "x = aaaaaaaaaaaaaaa(\n"
      "    \"a aaaaaaa aaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaa aaaaaaaaaaaaa\"\n"
      ");",
      Style);
  Style.ColumnLimit = 60;
  verifyFormat("auto lambda =\n"
               "    [&b](\n"
               "        auto aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    ) {};",
               Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    &bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               ");",
               Style);
}

TEST_F(AlignBracketsTest, AlignAfterOpenBracketBlockIndent) {
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
  Style.BinPackParameters = FormatStyle::BPPS_OnePerLine;

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

TEST_F(AlignBracketsTest, AlignAfterOpenBracketBlockIndentIfStatement) {
  auto Style = getLLVMStyle();

  verifyFormat("if (foo()) {\n"
               "  return;\n"
               "}",
               Style);

  verifyFormat("if (quiteLongArg !=\n"
               "    (alsoLongArg - 1)) { // ABC is a very longgggggggggggg "
               "comment\n"
               "  return;\n"
               "}",
               Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;

  verifyFormat("if (foo()) {\n"
               "  return;\n"
               "}",
               Style);

  verifyFormat("if (quiteLongArg !=\n"
               "    (alsoLongArg - 1)) { // ABC is a very longgggggggggggg "
               "comment\n"
               "  return;\n"
               "}",
               Style);

  verifyFormat("void foo() {\n"
               "  if (camelCaseName < alsoLongName ||\n"
               "      anotherEvenLongerName <=\n"
               "          thisReallyReallyReallyReallyReallyReallyLongerName ||"
               "\n"
               "      otherName < thisLastName) {\n"
               "    return;\n"
               "  } else if (quiteLongName < alsoLongName ||\n"
               "             anotherEvenLongerName <=\n"
               "                 thisReallyReallyReallyReallyReallyReallyLonger"
               "Name ||\n"
               "             otherName < thisLastName) {\n"
               "    return;\n"
               "  }\n"
               "}",
               Style);

  Style.ContinuationIndentWidth = 2;
  verifyFormat("void foo() {\n"
               "  if (ThisIsRatherALongIfClause && thatIExpectToBeBroken ||\n"
               "      ontoMultipleLines && whenFormattedCorrectly) {\n"
               "    if (false) {\n"
               "      return;\n"
               "    } else if (thisIsRatherALongIfClause && "
               "thatIExpectToBeBroken ||\n"
               "               ontoMultipleLines && whenFormattedCorrectly) {\n"
               "      return;\n"
               "    }\n"
               "  }\n"
               "}",
               Style);
}

TEST_F(AlignBracketsTest, AlignAfterOpenBracketBlockIndentForStatement) {
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

TEST_F(AlignBracketsTest, AlignAfterOpenBracketBlockIndentInitializers) {
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

TEST_F(AlignBracketsTest, AllowAllArgumentsOnNextLineDontAlign) {
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

TEST_F(AlignBracketsTest, FormatsDeclarationBreakAlways) {
  FormatStyle BreakAlways = getGoogleStyle();
  BreakAlways.BinPackParameters = FormatStyle::BPPS_AlwaysOnePerLine;
  verifyFormat("void f(int a,\n"
               "       int b);",
               BreakAlways);
  verifyFormat("void f(int aaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       int bbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "       int cccccccccccccccccccccccc);",
               BreakAlways);

  // Ensure AlignAfterOpenBracket interacts correctly with BinPackParameters set
  // to BPPS_AlwaysOnePerLine.
  BreakAlways.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  verifyFormat(
      "void someLongFunctionName(\n"
      "    int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    int b);",
      BreakAlways);
  BreakAlways.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;
  verifyFormat(
      "void someLongFunctionName(\n"
      "    int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    int b\n"
      ");",
      BreakAlways);
}

TEST_F(AlignBracketsTest, FormatsDefinitionBreakAlways) {
  FormatStyle BreakAlways = getGoogleStyle();
  BreakAlways.BinPackParameters = FormatStyle::BPPS_AlwaysOnePerLine;
  verifyFormat("void f(int a,\n"
               "       int b) {\n"
               "  f(a, b);\n"
               "}",
               BreakAlways);

  // Ensure BinPackArguments interact correctly when BinPackParameters is set to
  // BPPS_AlwaysOnePerLine.
  verifyFormat("void f(int aaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       int bbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "       int cccccccccccccccccccccccc) {\n"
               "  f(aaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "    cccccccccccccccccccccccc);\n"
               "}",
               BreakAlways);
  BreakAlways.BinPackArguments = false;
  verifyFormat("void f(int aaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       int bbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "       int cccccccccccccccccccccccc) {\n"
               "  f(aaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "    cccccccccccccccccccccccc);\n"
               "}",
               BreakAlways);

  // Ensure BreakFunctionDefinitionParameters interacts correctly when
  // BinPackParameters is set to BPPS_AlwaysOnePerLine.
  BreakAlways.BreakFunctionDefinitionParameters = true;
  verifyFormat("void f(\n"
               "    int a,\n"
               "    int b) {\n"
               "  f(a, b);\n"
               "}",
               BreakAlways);
  BreakAlways.BreakFunctionDefinitionParameters = false;

  // Ensure AlignAfterOpenBracket interacts correctly with BinPackParameters set
  // to BPPS_AlwaysOnePerLine.
  BreakAlways.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  verifyFormat(
      "void someLongFunctionName(\n"
      "    int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    int b) {\n"
      "  someLongFunctionName(\n"
      "      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, b);\n"
      "}",
      BreakAlways);
  BreakAlways.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;
  verifyFormat(
      "void someLongFunctionName(\n"
      "    int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    int b\n"
      ") {\n"
      "  someLongFunctionName(\n"
      "      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, b\n"
      "  );\n"
      "}",
      BreakAlways);
}

TEST_F(AlignBracketsTest, ParenthesesAndOperandAlignment) {
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

TEST_F(AlignBracketsTest, BlockIndentAndNamespace) {
  auto Style = getLLVMStyleWithColumns(120);
  Style.AllowShortNamespacesOnASingleLine = true;
  Style.AlignAfterOpenBracket = FormatStyle::BAS_BlockIndent;

  verifyNoCrash(
      "namespace {\n"
      "void xxxxxxxxxxxxxxxxxxxxx(nnnnn::TTTTTTTTTTTTT const *mmmm,\n"
      "                           YYYYYYYYYYYYYYYYY &yyyyyyyyyyyyyy);\n"
      "} //",
      Style);
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang
