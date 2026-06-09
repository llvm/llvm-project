//===- unittest/Format/AlignmentTest.cpp - Aligning unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "alignment-test"

namespace clang {
namespace format {
namespace test {
namespace {

class AlignmentTest : public test::FormatTestBase {};

TEST_F(AlignmentTest, ConsecutiveMacros) {
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
  Style.ReflowComments = FormatStyle::RCS_Never;
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

  Style.ColumnLimit = 30;
  verifyFormat("#define MY_FUNC(x) callMe(X)\n"
               "#define MY_LONG_CONSTANT 17",
               Style);
}

TEST_F(AlignmentTest, ConsecutiveAssignments) {
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

  verifyFormat("int abcdefghijk = 111;\n"
               "auto lambda     = [] {\n"
               "  int c = call(1, //\n"
               "               2, //\n"
               "               3, //\n"
               "               4);\n"
               "};",
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

  EXPECT_EQ(Alignment.ReflowComments, FormatStyle::RCS_Always);
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
               "  return \"Hello \"\n"
               "         \"World\";\n"
               "});",
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

  // Aligning lines should not mess up the comments. However, feel free to
  // change the test if it turns out that comments inside the closure should not
  // be aligned with those outside it.
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};  //\n"
               "auto b                     = [] { //\n"
               "  return;                         //\n"
               "};",
               Alignment);
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};  //\n"
               "auto b                     = [] { //\n"
               "  return aaaaaaaaaaaaaaaaaaaaa;   //\n"
               "};",
               Alignment);
  verifyFormat("auto aaaaaaaaaaaaaaa = {};      //\n"
               "auto b               = [] {     //\n"
               "  return aaaaaaaaaaaaaaaaaaaaa; //\n"
               "};",
               Alignment);

  verifyFormat("auto b = f(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "           ccc ? aaaaa : bbbbb,\n"
               "           dddddddddddddddddddddddddd);",
               Alignment);
  verifyFormat("auto aaaaaaaaaaaa = f();\n"
               "auto b            = f(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                      ccc ? aaaaa : bbbbb,\n"
               "                      dddddddddddddddddddddddddd);",
               Alignment);

  // Confirm proper handling of AlignConsecutiveAssignments with
  // BinPackArguments.
  // See https://llvm.org/PR55360
  Alignment = getLLVMStyleWithColumns(50);
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  Alignment.PackArguments.BinPack = FormatStyle::BPAS_OnePerLine;
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

  Alignment = getLLVMStyleWithColumns(60);
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("using II = typename TI<T, std::tuple<Types...>>::I;\n"
               "using I  = std::conditional_t<II::value >= 0,\n"
               "                              std::ic<int, II::value + 1>,\n"
               "                              std::ic<int, -1>>;",
               Alignment);
  verifyFormat("SomeName = Foo;\n"
               "X        = func<Type, Type>(looooooooooooooooooooooooong,\n"
               "                            arrrrrrrrrrg);",
               Alignment);

  Alignment.ColumnLimit = 80;
  Alignment.SpacesInAngles = FormatStyle::SIAS_Always;
  verifyFormat("void **ptr = reinterpret_cast< void ** >(unkn);\n"
               "ptr        = reinterpret_cast< void ** >(ptr[0]);",
               Alignment);
  verifyFormat("quint32 *dstimg  = reinterpret_cast< quint32 * >(out(i));\n"
               "quint32 *dstmask = reinterpret_cast< quint32 * >(outmask(i));",
               Alignment);

  Alignment.SpacesInParens = FormatStyle::SIPO_Custom;
  Alignment.SpacesInParensOptions.InCStyleCasts = true;
  verifyFormat("void **ptr = ( void ** )unkn;\n"
               "ptr        = ( void ** )ptr[0];",
               Alignment);
  verifyFormat("quint32 *dstimg  = ( quint32 * )out.scanLine(i);\n"
               "quint32 *dstmask = ( quint32 * )outmask.scanLine(i);",
               Alignment);
}

TEST_F(AlignmentTest, ConsecutiveAssignmentsAcrossEmptyLines) {
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

TEST_F(AlignmentTest, ConsecutiveAssignmentsAcrossComments) {
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

TEST_F(AlignmentTest, ConsecutiveAssignmentsAcrossEmptyLinesAndComments) {
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

  Alignment.ReflowComments = FormatStyle::RCS_Always;
  Alignment.ColumnLimit = 50;
  verifyFormat("int x   = 0;\n"
               "int yy  = 1; /// specificlennospace\n"
               "int zzz = 2;",
               "int x   = 0;\n"
               "int yy  = 1; ///specificlennospace\n"
               "int zzz = 2;",
               Alignment);
}

TEST_F(AlignmentTest, ConsecutiveEnumAssignments) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveAssignments.EnumAssignments = true;
  verifyFormat("enum ValueKind {\n"
               "  VK_Argument   = 1,\n"
               "  VK_BasicBlock = 2,\n"
               "  VK_Segment    = 8,\n"
               "};",
               Alignment);
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  Alignment.AlignConsecutiveAssignments.EnumAssignments = false;
  verifyFormat("enum ValueKind {\n"
               "  VK_Argument   = 1,\n"
               "  VK_BasicBlock = 2,\n"
               "  VK_Segment    = 8,\n"
               "};",
               Alignment);
}

TEST_F(AlignmentTest, ConsecutiveCompoundAssignments) {
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

TEST_F(AlignmentTest, ConsecutiveDeclarations) {
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
  verifyFormat(
      "foo          fooNode(ConvertStdStringToUString(fieldNames[chIdx]),\n"
      "                     // asdf\n"
      "                     // foo1 foo2 foo12345\n"
      "                     SomeFunctionAB(a123456789012345));\n"
      "const size_t v1234567890123456789012345678901234;",
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
  verifyFormat("int    a(const Test & = Test());\n"
               "int    a1(int &foo, const Test & = Test());\n"
               "int    a2(int &foo, const Test &name = Test());\n"
               "double b();",
               Alignment);
  verifyFormat("struct Test {\n"
               "  Test(const Test &) = default;\n"
               "  ~Test() = default;\n"
               "  Test &operator=(const Test &) = default;\n"
               "};",
               Alignment);

  // The comment to the right should still align right.
  verifyFormat("void foo(int   name, // name\n"
               "         float name, // name\n"
               "         int   name) // name\n"
               "{}",
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
  // Set ColumnLimit low so that we break the argument list in multiple lines.
  Alignment.ColumnLimit = 35;
  verifyFormat("int    a3(SomeTypeName1 &x,\n"
               "          SomeTypeName2 &y,\n"
               "          const Test & = Test());\n"
               "double b();",
               Alignment);
  Alignment.ColumnLimit = OldColumnLimit;
  // Ensure function pointers don't screw up recursive alignment
  verifyFormat("int    a(int x, void (*fp)(int y));\n"
               "double b();",
               Alignment);
  Alignment.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("struct Test {\n"
               "  Test(const Test &)            = default;\n"
               "  ~Test()                       = default;\n"
               "  Test &operator=(const Test &) = default;\n"
               "};",
               Alignment);
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
  // http://llvm.org/PR68079
  verifyFormat("using Fn   = int (A::*)();\n"
               "using RFn  = int (A::*)() &;\n"
               "using RRFn = int (A::*)() &&;",
               Alignment);
  verifyFormat("using Fn   = int (A::*)();\n"
               "using RFn  = int *(A::*)() &;\n"
               "using RRFn = double (A::*)() &&;",
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

  verifyFormat("int    a(SomeType& foo, const Test& = Test());\n"
               "double b();",
               AlignmentLeft);

  auto Style = AlignmentLeft;
  Style.AlignConsecutiveDeclarations.AlignFunctionPointers = true;
  Style.PackParameters.BinPack = FormatStyle::BPPS_OnePerLine;
  verifyFormat("int function_name(const wchar_t*  title,\n"
               "                  int             x          = 0,\n"
               "                  long            extraStyle = 0,\n"
               "                  bool            readOnly   = false,\n"
               "                  FancyClassType* module     = nullptr);",
               Style);

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

  verifyFormat("int    a(SomeType & foo, const Test & = Test());\n"
               "double b();",
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
  // When assignments are nested, each level should be aligned.
  verifyFormat("float i2 = 0;\n"
               "auto  v  = type{i2 = 1, //\n"
               "                i  = 3};",
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
  Alignment.PackParameters.BinPack = FormatStyle::BPPS_OnePerLine;
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
  Alignment.PackParameters.BinPack = FormatStyle::BPPS_BinPack;
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
  Style = getMozillaStyle();
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
  BracedAlign.Cpp11BracedListStyle = FormatStyle::BLS_Block;
  verifyFormat("const auto result{ []() {\n"
               "  const auto something = 1;\n"
               "  return 2;\n"
               "} };",
               BracedAlign);
  verifyFormat("const volatile auto result{ []() {\n"
               "  const auto something = 1;\n"
               "  return 2;\n"
               "} };",
               BracedAlign);
  verifyFormat("int foo{ []() {\n"
               "  int bar{ 0 };\n"
               "  return 0;\n"
               "}() };",
               BracedAlign);

  Alignment.AlignConsecutiveDeclarations.AlignFunctionDeclarations = false;
  verifyFormat("unsigned int f1(void);\n"
               "void f2(void);\n"
               "size_t f3(void);",
               Alignment);
}

TEST_F(AlignmentTest, ConsecutiveDeclarationsAcrossEmptyLinesAndComments) {
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

TEST_F(AlignmentTest, ConsecutiveBitFields) {
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

TEST_F(AlignmentTest, ConsecutiveBitFieldsAcrossEmptyLinesAndComments) {
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

TEST_F(AlignmentTest, ConsecutiveAlignConsecutiveShortCaseStatements) {
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

  Alignment.ColumnLimit = 40;
  verifyFormat("switch (level) {\n"
               "default: return \"a bit longer string\";\n"
               "case log::warning: return \"foo\";\n"
               "}",
               Alignment);
  Alignment.ColumnLimit = 80;

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

  verifyFormat("switch (level) {\n"
               "case log::error   :\n"
               "default           : return \"default\";\n"
               "case log::info    : return \"info\";\n"
               "case log::warning : return \"warning\";\n"
               "}",
               Alignment);
}

TEST_F(AlignmentTest, AlignWithLineBreaks) {
  auto Style = getLLVMStyleWithColumns(120);

  EXPECT_EQ(Style.AlignConsecutiveAssignments,
            FormatStyle::AlignConsecutiveStyle(
                {/*Enabled=*/false, /*AcrossEmptyLines=*/false,
                 /*AcrossComments=*/false, /*AlignCompound=*/false,
                 /*AlignFunctionDeclarations=*/false,
                 /*AlignFunctionPointers=*/false,
                 /*EnumAssignments=*/false, /*PadOperators=*/true}));
  EXPECT_EQ(Style.AlignConsecutiveDeclarations,
            FormatStyle::AlignConsecutiveStyle(
                {/*Enabled=*/false, /*AcrossEmptyLines=*/false,
                 /*AcrossComments=*/false, /*AlignCompound=*/false,
                 /*AlignFunctionDeclarations=*/true,
                 /*AlignFunctionPointers=*/false,
                 /*EnumAssignments=*/false, /*PadOperators=*/false}));
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

  verifyFormat("auto someLongName = 3;\n"
               "auto x            = someLongExpression //\n"
               "                    | ranges::views::values;",
               Style);
  verifyFormat(
      "veryverylongvariablename = somethingelse;\n"
      "shortervariablename      = anotherverylonglonglongvariablename + //\n"
      "                           somevariablethatwastoolongtofitonthesamerow;",
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

  verifyFormat("void foo() {\n"
               "  int    myVar = 5;\n"
               "  double x     = 3.14;\n"
               "  auto   str   = (\"Hello \"\n"
               "                  \"World\");\n"
               "  auto   s     = (\"Hello \"\n"
               "                  \"Again\");\n"
               "}",
               Style);

  verifyFormat("A    B       = {\"Hello \"\n"
               "                \"World\"};\n"
               "BYTE payload = 2;",
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

  // The start of the closure is indented from the start of the line. It should
  // not move with the equal sign.
  Style.ContinuationIndentWidth = 6;
  Style.IndentWidth = 8;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.BeforeLambdaBody = true;
  Style.BraceWrapping.IndentBraces = true;
  Style.ColumnLimit = 32;
  verifyFormat("auto aaaaaaaaaaa = {};\n"
               "b = []() constexpr\n"
               "      -> aaaaaaaaaaaaaaaaaaaaaaa\n"
               "        {\n"
               "                return {}; //\n"
               "        };",
               Style);
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};\n"
               "b = []()\n"
               "        {\n"
               "                return; //\n"
               "        };",
               Style);
  Style.ColumnLimit = 33;
  verifyFormat("auto aaaaaaaaaaa = {};\n"
               "b                = []() constexpr\n"
               "      -> aaaaaaaaaaaaaaaaaaaaaaa\n"
               "        {\n"
               "                return {}; //\n"
               "        };",
               Style);
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};\n"
               "b                          = []()\n"
               "        {\n"
               "                return; //\n"
               "        };",
               Style);

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

  Style.PackArguments.BinPack = FormatStyle::BPAS_OnePerLine;

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

  Style = getLLVMStyle();
  Style.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("param->fault_depth = grid->jfault * grid->dz; //\n"
               "grid->corner       = grid->jlid + 1;          //\n"
               "param->peclet      = param->V                 //\n"
               "                     * param->L * 1000.0      //\n"
               "                     / param->kappa;          //",
               Style);
  Style.AlignOperands = FormatStyle::OAS_AlignAfterOperator;
  verifyFormat("param->fault_depth = grid->jfault * grid->dz; //\n"
               "grid->corner       = grid->jlid + 1;          //\n"
               "param->peclet      = param->V                 //\n"
               "                   * param->L * 1000.0        //\n"
               "                   / param->kappa;            //",
               Style);

  Style = getLLVMStyleWithColumns(70);
  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat(
      "ReturnType\n"
      "MyFancyIntefaceFunction(Context       *context,\n"
      "                        ALongTypeName *response) noexcept override;\n"
      "ReturnType func();",
      Style);

  verifyFormat(
      "ReturnType\n"
      "MyFancyIntefaceFunction(B<int>          *context,\n"
      "                        decltype(AFunc) *response) noexcept override;\n"
      "ReturnType func();",
      Style);

  Style.AlignConsecutiveAssignments.Enabled = true;
  verifyFormat("float i2 = 0;\n"
               "auto  v  = false ? type{}\n"
               "                 : type{\n"
               "                       1,\n"
               "                   };",
               Style);

  verifyFormat("const char *const MatHtool[] = {};\n"
               "const char        Htool[]    = \"@article{m,\\n\"\n"
               "                               \" \"\n"
               "                               \" \"\n"
               "                               \" \"\n"
               "                               \"}\\n\";",
               Style);

  Style.ColumnLimit = 15;
  verifyFormat("int i1 = 1;\n"
               "k      = bar(\n"
               "    argument1,\n"
               "    argument2);",
               Style);

  Style.ColumnLimit = 45;
  verifyFormat("auto xxxxxxxx = foo;\n"
               "auto x = whatever ? some / long -\n"
               "                        computition / stuff\n"
               "                  : random;",
               Style);
}

TEST_F(AlignmentTest, AlignWithInitializerPeriods) {
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

  // The lines inside the braces are supposed to be indented by
  // BracedInitializerIndentWidth from the start of the line. They should not
  // move with the opening brace.
  verifyFormat("void foo2(void) {\n"
               "  BYTE p[1] = 1;\n"
               "  A B       = {\n"
               "      .one_foooooooooooooooo = 2,\n"
               "      .two_fooooooooooooo    = 3,\n"
               "      .three_fooooooooooooo  = 4,\n"
               "  };\n"
               "  BYTE payload = 2;\n"
               "}",
               Style);

  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};\n"
               "auto b                     = g([] {\n"
               "  x = {.one_foooooooooooooooo = 2, //\n"
               "       .two_fooooooooooooo    = 3, //\n"
               "       .three_fooooooooooooo  = 4};\n"
               "});",
               Style);

  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaa = {};\n"
               "auto b                     = {.a = {\n"
               "                                  .a = 0,\n"
               "                              }};",
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

TEST_F(AlignmentTest, CatchAlignArrayOfStructuresRightAlignment) {
  auto Style = getLLVMStyle();
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Right;
  verifyNoCrash("f({\n"
                "table({}, table({{\"\", false}}, {}))\n"
                "});",
                Style);

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

  verifyNoCrash("Foo f[] = {\n"
                "    [0] = { 1, },\n"
                "    [i] { 1, },\n"
                "};",
                Style);
  verifyNoCrash("Foo foo[] = {\n"
                "    [0] = {1, 1},\n"
                "    [1] { 1, 1, },\n"
                "    [2] { 1, 1, },\n"
                "};",
                Style);
  verifyNoCrash("test arr[] = {\n"
                "#define FOO(i) {i, i},\n"
                "SOME_GENERATOR(FOO)\n"
                "{2, 2}\n"
                "};",
                Style);

  verifyFormat("return GradForUnaryCwise(g, {\n"
               "                                {{\"sign\"}, \"Sign\",  "
               "  {\"x\", \"dy\"}},\n"
               "                                {  {\"dx\"},  \"Mul\", {\"dy\""
               ", \"sign\"}},\n"
               "});",
               Style);

  Style.Cpp11BracedListStyle = FormatStyle::BLS_Block;
  verifyFormat("struct test demo[] = {\n"
               "  { 56,    23, \"hello\" },\n"
               "  { -1, 93463, \"world\" },\n"
               "  {  7,     5,    \"!!\" }\n"
               "};",
               Style);
  Style.Cpp11BracedListStyle = FormatStyle::BLS_AlignFirstComment;

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
#if 0
  verifyFormat("int a[][] = {\n"
               "    {\n"
               "     {0, 2}, //\n"
               "     {1, 2}  //\n"
               "    }\n"
               "};",
               Style);
#endif
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

TEST_F(AlignmentTest, CatchAlignArrayOfStructuresLeftAlignment) {
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

  verifyNoCrash("Foo f[] = {\n"
                "    [0] = { 1, },\n"
                "    [i] { 1, },\n"
                "};",
                Style);
  verifyNoCrash("Foo foo[] = {\n"
                "    [0] = {1, 1},\n"
                "    [1] { 1, 1, },\n"
                "    [2] { 1, 1, },\n"
                "};",
                Style);
  verifyNoCrash("test arr[] = {\n"
                "#define FOO(i) {i, i},\n"
                "SOME_GENERATOR(FOO)\n"
                "{2, 2}\n"
                "};",
                Style);

  verifyFormat("return GradForUnaryCwise(g, {\n"
               "                                {{\"sign\"}, \"Sign\", {\"x\", "
               "\"dy\"}   },\n"
               "                                {{\"dx\"},   \"Mul\",  "
               "{\"dy\", \"sign\"}},\n"
               "});",
               Style);

  verifyNoCrash(
      "PANEL_Ic PANEL_ic[PANEL_IC_NUMBER] =\n"
      "    {\n"
      "        {PIC(0),   PIC(0),   PIC(99),  PIC(81),  0}, // Backbox\n"
      "        {PIC(1),   PIC(83),  PIC(191), PIC(137), 0}, // AK47\n"
      "\n"
      "#define PICALL1(a, b, c, d) \\\n"
      "    { PIC(a), PIC(b), PIC(c), PIC(d), 1 }\n"
      "\n"
      "        PICALL1(1, 1, 75, 50),\n"
      "};",
      Style);

  Style.AlignEscapedNewlines = FormatStyle::ENAS_DontAlign;
  verifyFormat("#define FOO \\\n"
               "  int foo[][2] = { \\\n"
               "      {0, 1} \\\n"
               "  };",
               Style);

  Style.Cpp11BracedListStyle = FormatStyle::BLS_Block;
  verifyFormat("struct test demo[] = {\n"
               "  { 56, 23,    \"hello\" },\n"
               "  { -1, 93463, \"world\" },\n"
               "  { 7,  5,     \"!!\"    }\n"
               "};",
               Style);
  Style.Cpp11BracedListStyle = FormatStyle::BLS_AlignFirstComment;

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
  verifyFormat("Foo foo = {\n"
               "    // comment\n"
               "    {1, 2}\n"
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

  Style.ColumnLimit = 25;
  verifyNoCrash("Type foo{\n"
                "    {\n"
                "        1,  // A\n"
                "        2,  // B\n"
                "        3,  // C\n"
                "    },\n"
                "    \"hello\",\n"
                "};",
                Style);
  verifyNoCrash("Type object[X][Y] = {\n"
                "    {{val}, {val}, {val}},\n"
                "    {{val}, {val}, // some comment\n"
                "                   {val}}\n"
                "};",
                Style);

  Style.ColumnLimit = 120;
  verifyNoCrash(
      "T v[] {\n"
      "    { AAAAAAAAAAAAAAAAAAAAAAAAA::aaaaaaaaaaaaaaaaaaa, "
      "AAAAAAAAAAAAAAAAAAAAAAAAA::aaaaaaaaaaaaaaaaaaaaaaaa, 1, 0.000000000f, "
      "\"00000000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000\" },\n"
      "};",
      Style);

  Style.SpacesInParens = FormatStyle::SIPO_Custom;
  Style.SpacesInParensOptions.Other = true;
  verifyFormat("Foo foo[] = {\n"
               "    {1, 1},\n"
               "    {1, 1},\n"
               "};",
               Style);
}

TEST_F(AlignmentTest, AlignArrayOfStructuresGithubIssues) {
  // https://github.com/llvm/llvm-project/issues/138151
  // Summary: Aligning arrays of structures with UseTab: AlignWithSpaces does
  // not use spaces to align columns
  FormatStyle Style = getGoogleStyle();
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Left;
  Style.UseTab = FormatStyle::UT_AlignWithSpaces;
  Style.IndentWidth = 4;
  Style.TabWidth = 4;

  verifyFormat(
      "std::vector<Foo> foos = {\n"
      "\t{LONG_NAME,                0,                        i | j},\n"
      "\t{LONG_NAME,                0,                        i | j},\n"
      "\t{LONGER_NAME,              0,                        i | j},\n"
      "\t{LONGER_NAME,              0,                        i    },\n"
      "\t{THIS_IS_A_VERY_LONG_NAME, 0,                        j    },\n"
      "\t{LONGER_NAME,              THIS_IS_A_VERY_LONG_NAME, i    },\n"
      "\t{LONG_NAME,                THIS_IS_A_VERY_LONG_NAME, j    }\n"
      "};\n",
      Style);

  // https://github.com/llvm/llvm-project/issues/85937
  // Summary: Macro escaped newlines are not aligned properly when both
  // AlignEscapedNewLines and AlignArrayOfStructures are used
  Style = getLLVMStyleWithColumns(80);
  Style.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Left;

  verifyFormat(R"(
#define DEFINE_COMMAND_PROCESS_TABLE(Enum)      \
  const STExample TCommand::EXPL_MAIN[] = {     \
      {Enum::GetName(),      " shows help "  }, \
      {Enum::GetAttribute(), " do something "}, \
      {Enum::GetState(),     " do whatever " }, \
  };
)",
               Style);

  // https://github.com/llvm/llvm-project/issues/53442
  // Summary: alignment of columns does not use spaces when UseTab:
  // AlignWithSpaces
  Style = getLLVMStyle();
  Style.AlignArrayOfStructures = FormatStyle::AIAS_Left;
  Style.IndentWidth = 4;
  Style.TabWidth = 4;
  Style.UseTab = FormatStyle::UT_AlignWithSpaces;
  Style.BreakBeforeBraces = FormatStyle::BS_Allman;

  verifyFormat(
      "const map<string, int64_t> CoreReport::GetGameCountersRolloverInfo()\n"
      "{\n"
      "\tstatic map<string, int64_t> counterRolloverInfo{\n"
      "\t\t{\"CashIn\",                   4000000000},\n"
      "\t\t{\"CoinIn\",                   4000000000},\n"
      "\t\t{\"QuantityMultiProgressive\", 65535     },\n"
      "\t};\n"
      "\treturn counterRolloverInfo;\n"
      "}\n",
      Style);
}

TEST_F(AlignmentTest, AlignArrayOfStructuresLeftAlignmentNonSquare) {
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
               "       {13},\n"
               "       {something}, // A\n"
               "      }\n"
               "  };\n"
               "}",
               Style);
}

TEST_F(AlignmentTest, AlignArrayOfStructuresRightAlignmentNonSquare) {
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
               "       {13},\n"
               "       {something}, // A\n"
               "      }\n"
               "  };\n"
               "}",
               Style);
}

TEST_F(AlignmentTest, AlignInsidePreprocessorElseBlock) {
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
  verifyFormat("auto foo = [] { return; };\n"
               "#if FOO\n"
               "#else\n"
               "count = bar;\n"
               "mbid  = bid;\n"
               "#endif",
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

TEST_F(AlignmentTest, ContinuedAligned) {
  FormatStyle Style = getLLVMStyleWithColumns(60);
  Style.UseTab = FormatStyle::UT_AlignWithSpaces;
  Style.TabWidth = Style.IndentWidth = Style.ContinuationIndentWidth = 4;

  verifyFormat("for (;;) {\n"
               "\tif (bar(aaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbb,\n"
               "\t        cccccccccccccccccccccccccccccc)) {\n"
               "\t\treturn {};\n"
               "\t}\n"
               "}",
               Style);
  verifyFormat("bar([]() {\n"
               "\tconst AAAAAA aaaaa =\n"
               "\t\tAAAAAAAAAA(foo(bbbbbbbbbbbbbbbbbbbbbbbbbb),\n"
               "\t\t           foo(cccccccccccccccccccccccccc),\n"
               "\t\t           foo(ddddddddddddddddddddddddd)) +\n"
               "\t\teeeeeeeee;\n"
               "});",
               Style);

  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Always;
  verifyFormat("bar(\n"
               "\t[]() {\n"
               "\t\tif constexpr (std::is_same_v<aaaaaa,\n"
               "\t\t                             T>) // comment line 1\n"
               "\t\t                                 // comment line 2\n"
               "\t\t{\n"
               "\t\t}\n"
               "\t},\n"
               "\tvariant);",
               Style);

  Style.ColumnLimit = 40;
  Style.IndentWidth = Style.TabWidth = Style.ContinuationIndentWidth = 8;

  verifyFormat("void f() {\n"
               "\tint aaaaaaaaaaaaaaaaaaaa =\n"
               "\t\t000000000000000001 ? 2\n"
               "\t\t                   : 3;\n"
               "}",
               Style);
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang
