//===- unittests/Lex/DependencyDirectivesSourceMinimizer.cpp -  -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/DependencyDirectivesSourceMinimizer.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::minimize_source_to_dependency_directives;

namespace clang {

bool minimizeSourceToDependencyDirectives(StringRef Input,
                                          SmallVectorImpl<char> &Out) {
  SmallVector<minimize_source_to_dependency_directives::Token, 32> Tokens;
  return minimizeSourceToDependencyDirectives(Input, Out, Tokens);
}

} // end namespace clang

namespace {

TEST(MinimizeSourceToDependencyDirectivesTest, Empty) {
  SmallVector<char, 128> Out;
  SmallVector<Token, 4> Tokens;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("", Out, Tokens));
  EXPECT_TRUE(Out.empty());
  ASSERT_EQ(1u, Tokens.size());
  ASSERT_EQ(pp_eof, Tokens.back().K);

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("abc def\nxyz", Out, Tokens));
  EXPECT_TRUE(Out.empty());
  ASSERT_EQ(1u, Tokens.size());
  ASSERT_EQ(pp_eof, Tokens.back().K);
}

TEST(MinimizeSourceToDependencyDirectivesTest, AllTokens) {
  SmallVector<char, 128> Out;
  SmallVector<Token, 4> Tokens;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define A\n"
                                           "#undef A\n"
                                           "#endif\n"
                                           "#if A\n"
                                           "#ifdef A\n"
                                           "#ifndef A\n"
                                           "#elif A\n"
                                           "#else\n"
                                           "#include <A>\n"
                                           "#include_next <A>\n"
                                           "#__include_macros <A>\n"
                                           "#import <A>\n"
                                           "@import A;\n"
                                           "#pragma clang module import A\n",
                                           Out, Tokens));
  EXPECT_EQ(pp_define, Tokens[0].K);
  EXPECT_EQ(pp_undef, Tokens[1].K);
  EXPECT_EQ(pp_endif, Tokens[2].K);
  EXPECT_EQ(pp_if, Tokens[3].K);
  EXPECT_EQ(pp_ifdef, Tokens[4].K);
  EXPECT_EQ(pp_ifndef, Tokens[5].K);
  EXPECT_EQ(pp_elif, Tokens[6].K);
  EXPECT_EQ(pp_else, Tokens[7].K);
  EXPECT_EQ(pp_include, Tokens[8].K);
  EXPECT_EQ(pp_include_next, Tokens[9].K);
  EXPECT_EQ(pp___include_macros, Tokens[10].K);
  EXPECT_EQ(pp_import, Tokens[11].K);
  EXPECT_EQ(decl_at_import, Tokens[12].K);
  EXPECT_EQ(pp_pragma_import, Tokens[13].K);
  EXPECT_EQ(pp_eof, Tokens[14].K);
}

TEST(MinimizeSourceToDependencyDirectivesTest, Define) {
  SmallVector<char, 128> Out;
  SmallVector<Token, 4> Tokens;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO", Out, Tokens));
  EXPECT_STREQ("#define MACRO\n", Out.data());
  ASSERT_EQ(2u, Tokens.size());
  ASSERT_EQ(pp_define, Tokens.front().K);
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineSpacing) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO\n\n\n", Out));
  EXPECT_STREQ("#define MACRO\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO \n\n\n", Out));
  EXPECT_STREQ("#define MACRO\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO a \n\n\n", Out));
  EXPECT_STREQ("#define MACRO a\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define   MACRO\n\n\n", Out));
  EXPECT_STREQ("#define MACRO\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineMacroArguments) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define MACRO()", Out));
  EXPECT_STREQ("#define MACRO()\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO(a, b...)", Out));
  EXPECT_STREQ("#define MACRO(a,b...)\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO content", Out));
  EXPECT_STREQ("#define MACRO content\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO   con  tent   ", Out));
  EXPECT_STREQ("#define MACRO con  tent\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO()   con  tent   ", Out));
  EXPECT_STREQ("#define MACRO() con  tent\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineInvalidMacroArguments) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define MACRO((a))", Out));
  EXPECT_STREQ("#define MACRO(/* invalid */\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define MACRO(", Out));
  EXPECT_STREQ("#define MACRO(/* invalid */\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO(a * b)", Out));
  EXPECT_STREQ("#define MACRO(/* invalid */\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineHorizontalWhitespace) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO(\t)\tcon \t tent\t", Out));
  EXPECT_STREQ("#define MACRO() con \t tent\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO(\f)\fcon \f tent\f", Out));
  EXPECT_STREQ("#define MACRO() con \f tent\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO(\v)\vcon \v tent\v", Out));
  EXPECT_STREQ("#define MACRO() con \v tent\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO \t\v\f\v\t con\f\t\vtent\v\f \v", Out));
  EXPECT_STREQ("#define MACRO con\f\t\vtent\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineMultilineArgs) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO(a        \\\n"
                                           "              )",
                                           Out));
  EXPECT_STREQ("#define MACRO(a)\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO(a,       \\\n"
                                           "              b)       \\\n"
                                           "        call((a),      \\\n"
                                           "             (b))",
                                           Out));
  EXPECT_STREQ("#define MACRO(a,b) call((a),(b))\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest,
     DefineMultilineArgsCarriageReturn) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO(a,       \\\r"
                                           "              b)       \\\r"
                                           "        call((a),      \\\r"
                                           "             (b))",
                                           Out));
  EXPECT_STREQ("#define MACRO(a,b) call((a),(b))\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest,
     DefineMultilineArgsCarriageReturnNewline) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO(a,       \\\r\n"
                                           "              b)       \\\r\n"
                                           "        call((a),      \\\r\n"
                                           "             (b))",
                                           Out));
  EXPECT_STREQ("#define MACRO(a,b) call((a),(b))\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest,
     DefineMultilineArgsNewlineCarriageReturn) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO(a,       \\\n\r"
                                           "              b)       \\\n\r"
                                           "        call((a),      \\\n\r"
                                           "             (b))",
                                           Out));
  EXPECT_STREQ("#define MACRO(a,b) call((a),(b))\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineNumber) {
  SmallVector<char, 128> Out;

  ASSERT_TRUE(minimizeSourceToDependencyDirectives("#define 0\n", Out));
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineNoName) {
  SmallVector<char, 128> Out;

  ASSERT_TRUE(minimizeSourceToDependencyDirectives("#define &\n", Out));
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineNoWhitespace) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define AND&\n", Out));
  EXPECT_STREQ("#define AND &\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define AND\\\n"
                                                    "&\n",
                                                    Out));
  EXPECT_STREQ("#define AND &\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, MultilineComment) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO a/*\n"
                                           "  /*\n"
                                           "#define MISSING abc\n"
                                           "  /*\n"
                                           "  /* something */ \n"
                                           "#include  /* \"def\" */ <abc> \n",
                                           Out));
  EXPECT_STREQ("#define MACRO a\n"
               "#include <abc>\n",
               Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, MultilineCommentInStrings) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define MACRO1 \"/*\"\n"
                                                    "#define MACRO2 \"*/\"\n",
                                                    Out));
  EXPECT_STREQ("#define MACRO1 \"/*\"\n"
               "#define MACRO2 \"*/\"\n",
               Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, Ifdef) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#ifdef A\n"
                                                    "#define B\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("#ifdef A\n"
               "#define B\n"
               "#endif\n",
               Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#ifdef A\n"
                                                    "#define B\n"
                                                    "#elif B\n"
                                                    "#define C\n"
                                                    "#elif C\n"
                                                    "#define D\n"
                                                    "#else\n"
                                                    "#define E\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("#ifdef A\n"
               "#define B\n"
               "#elif B\n"
               "#define C\n"
               "#elif C\n"
               "#define D\n"
               "#else\n"
               "#define E\n"
               "#endif\n",
               Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, EmptyIfdef) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#ifdef A\n"
                                                    "#elif B\n"
                                                    "#elif C\n"
                                                    "#else D\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, Pragma) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#pragma A\n", Out));
  EXPECT_STREQ("", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#pragma clang\n", Out));
  EXPECT_STREQ("", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#pragma clang module\n", Out));
  EXPECT_STREQ("", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#pragma clang module impor\n", Out));
  EXPECT_STREQ("", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#pragma clang module import\n", Out));
  EXPECT_STREQ("#pragma clang module import\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, Include) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#include \"A\"\n", Out));
  EXPECT_STREQ("#include \"A\"\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#include <A>\n", Out));
  EXPECT_STREQ("#include <A>\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#include_next <A>\n", Out));
  EXPECT_STREQ("#include_next <A>\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#import <A>\n", Out));
  EXPECT_STREQ("#import <A>\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#__include_macros <A>\n", Out));
  EXPECT_STREQ("#__include_macros <A>\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, AtImport) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("@import A;\n", Out));
  EXPECT_STREQ("@import A;\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(" @ import  A;\n", Out));
  EXPECT_STREQ("@import A;\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("@import A\n;", Out));
  EXPECT_STREQ("@import A;\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("@import A.B;\n", Out));
  EXPECT_STREQ("@import A.B;\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "@import /*x*/ A /*x*/ . /*x*/ B /*x*/ \n /*x*/ ; /*x*/", Out));
  EXPECT_STREQ("@import A.B;\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, AtImportFailures) {
  SmallVector<char, 128> Out;

  ASSERT_TRUE(minimizeSourceToDependencyDirectives("@import A\n", Out));
  ASSERT_TRUE(minimizeSourceToDependencyDirectives("@import MACRO(A);\n", Out));
  ASSERT_TRUE(minimizeSourceToDependencyDirectives("@import \" \";\n", Out));
}

TEST(MinimizeSourceToDependencyDirectivesTest, RawStringLiteral) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#ifndef GUARD\n"
                                                    "#define GUARD\n"
                                                    "R\"()\"\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("#ifndef GUARD\n"
               "#define GUARD\n"
               "#endif\n",
               Out.data());

  bool RawStringLiteralResult = minimizeSourceToDependencyDirectives(
      "#ifndef GUARD\n"
      "#define GUARD\n"
      R"raw(static constexpr char bytes[] = R"(-?:\,[]{}#&*!|>'"%@`)";)raw"
      "\n"
      "#endif\n",
      Out);
  ASSERT_FALSE(RawStringLiteralResult);
  EXPECT_STREQ("#ifndef GUARD\n"
               "#define GUARD\n"
               "#endif\n",
               Out.data());

  bool RawStringLiteralResult2 = minimizeSourceToDependencyDirectives(
      "#ifndef GUARD\n"
      "#define GUARD\n"
      R"raw(static constexpr char bytes[] = R"abc(-?:\,[]{}#&*!|>'"%@`)abc";)raw"
      "\n"
      "#endif\n",
      Out);
  ASSERT_FALSE(RawStringLiteralResult2);
  EXPECT_STREQ("#ifndef GUARD\n"
               "#define GUARD\n"
               "#endif\n",
               Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, SplitIdentifier) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#if\\\n"
                                                    "ndef GUARD\n"
                                                    "#define GUARD\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("#ifndef GUARD\n"
               "#define GUARD\n"
               "#endif\n",
               Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define GUA\\\n"
                                                    "RD\n",
                                                    Out));
  EXPECT_STREQ("#define GUARD\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define GUA\\\r"
                                                    "RD\n",
                                                    Out));
  EXPECT_STREQ("#define GUARD\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define GUA\\\n"
                                                    "           RD\n",
                                                    Out));
  EXPECT_STREQ("#define GUA RD\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, PoundWarningAndError) {
  SmallVector<char, 128> Out;

  for (auto Source : {
           "#warning '\n#include <t.h>\n",
           "#warning \"\n#include <t.h>\n",
           "#warning /*\n#include <t.h>\n",
           "#warning \\\n#include <t.h>\n#include <t.h>\n",
           "#error '\n#include <t.h>\n",
           "#error \"\n#include <t.h>\n",
           "#error /*\n#include <t.h>\n",
           "#error \\\n#include <t.h>\n#include <t.h>\n",
       }) {
    ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
    EXPECT_STREQ("#include <t.h>\n", Out.data());
  }

  for (auto Source : {
           "#warning \\\n#include <t.h>\n",
           "#error \\\n#include <t.h>\n",
           "#if MACRO\n#warning '\n#endif\n",
           "#if MACRO\n#warning \"\n#endif\n",
           "#if MACRO\n#warning /*\n#endif\n",
           "#if MACRO\n#error '\n#endif\n",
           "#if MACRO\n#error \"\n#endif\n",
           "#if MACRO\n#error /*\n#endif\n",
       }) {
    ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
    EXPECT_STREQ("", Out.data());
  }
}

TEST(MinimizeSourceToDependencyDirectivesTest, CharacterLiteral) {
  SmallVector<char, 128> Out;

  StringRef Source = R"(
#include <bob>
int a = 0'1;
int b = 0xfa'af'fa;
int c = 12 ' ';
#include <foo>
)";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#include <bob>\n#include <foo>\n", Out.data());
}

} // end anonymous namespace
