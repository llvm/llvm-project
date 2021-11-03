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
                                           "#elifdef A\n"
                                           "#elifndef A\n"
                                           "#elif A\n"
                                           "#else\n"
                                           "#include <A>\n"
                                           "#include_next <A>\n"
                                           "#__include_macros <A>\n"
                                           "#import <A>\n"
                                           "@import A;\n"
                                           "#pragma clang module import A\n"
                                           "#pragma push_macro(A)\n"
                                           "#pragma pop_macro(A)\n"
                                           "#pragma include_alias(<A>, <B>)\n"
                                           "export module m;\n"
                                           "import m;\n",
                                           Out, Tokens));
  EXPECT_EQ(pp_define, Tokens[0].K);
  EXPECT_EQ(pp_undef, Tokens[1].K);
  EXPECT_EQ(pp_endif, Tokens[2].K);
  EXPECT_EQ(pp_if, Tokens[3].K);
  EXPECT_EQ(pp_ifdef, Tokens[4].K);
  EXPECT_EQ(pp_ifndef, Tokens[5].K);
  EXPECT_EQ(pp_elifdef, Tokens[6].K);
  EXPECT_EQ(pp_elifndef, Tokens[7].K);
  EXPECT_EQ(pp_elif, Tokens[8].K);
  EXPECT_EQ(pp_else, Tokens[9].K);
  EXPECT_EQ(pp_include, Tokens[10].K);
  EXPECT_EQ(pp_include_next, Tokens[11].K);
  EXPECT_EQ(pp___include_macros, Tokens[12].K);
  EXPECT_EQ(pp_import, Tokens[13].K);
  EXPECT_EQ(decl_at_import, Tokens[14].K);
  EXPECT_EQ(pp_pragma_import, Tokens[15].K);
  EXPECT_EQ(pp_pragma_push_macro, Tokens[16].K);
  EXPECT_EQ(pp_pragma_pop_macro, Tokens[17].K);
  EXPECT_EQ(pp_pragma_include_alias, Tokens[18].K);
  EXPECT_EQ(cxx_export_decl, Tokens[19].K);
  EXPECT_EQ(cxx_module_decl, Tokens[20].K);
  EXPECT_EQ(cxx_import_decl, Tokens[21].K);
  EXPECT_EQ(pp_eof, Tokens[22].K);
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
  EXPECT_STREQ("#define MACRO(a,b) call((a), (b))\n", Out.data());
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
  EXPECT_STREQ("#define MACRO(a,b) call((a), (b))\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineMultilineArgsStringize) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define MACRO(a,b) \\\n"
                                                    "                #a \\\n"
                                                    "                #b",
                                                    Out));
  EXPECT_STREQ("#define MACRO(a,b) #a #b\n", Out.data());
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
  EXPECT_STREQ("#define MACRO(a,b) call((a), (b))\n", Out.data());
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
  EXPECT_STREQ("#define MACRO(a,b) call((a), (b))\n", Out.data());
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

TEST(MinimizeSourceToDependencyDirectivesTest, Elifdef) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#ifdef A\n"
                                                    "#define B\n"
                                                    "#elifdef C\n"
                                                    "#define D\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("#ifdef A\n"
               "#define B\n"
               "#elifdef C\n"
               "#define D\n"
               "#endif\n",
               Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#ifdef A\n"
                                                    "#define B\n"
                                                    "#elifdef B\n"
                                                    "#define C\n"
                                                    "#elifndef C\n"
                                                    "#define D\n"
                                                    "#else\n"
                                                    "#define E\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("#ifdef A\n"
               "#define B\n"
               "#elifdef B\n"
               "#define C\n"
               "#elifndef C\n"
               "#define D\n"
               "#else\n"
               "#define E\n"
               "#endif\n",
               Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, EmptyIfdef) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#ifdef A\n"
                                                    "void skip();\n"
                                                    "#elif B\n"
                                                    "#elif C\n"
                                                    "#else D\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("#ifdef A\n"
               "#elif B\n"
               "#elif C\n"
               "#endif\n",
               Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, EmptyElifdef) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#ifdef A\n"
                                                    "void skip();\n"
                                                    "#elifdef B\n"
                                                    "#elifndef C\n"
                                                    "#else D\n"
                                                    "#endif\n",
                                                    Out));
  EXPECT_STREQ("#ifdef A\n"
               "#elifdef B\n"
               "#elifndef C\n"
               "#endif\n",
               Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, Pragma) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#pragma A\n", Out));
  EXPECT_STREQ("", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#pragma push_macro(\"MACRO\")\n", Out));
  EXPECT_STREQ("#pragma push_macro(\"MACRO\")\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#pragma pop_macro(\"MACRO\")\n", Out));
  EXPECT_STREQ("#pragma pop_macro(\"MACRO\")\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#pragma include_alias(\"A\", \"B\")\n", Out));
  EXPECT_STREQ("#pragma include_alias(\"A\", \"B\")\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#pragma include_alias(<A>, <B>)\n", Out));
  EXPECT_STREQ("#pragma include_alias(<A>, <B>)\n", Out.data());

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

TEST(MinimizeSourceToDependencyDirectivesTest,
     WhitespaceAfterLineContinuationSlash) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define A 1 + \\  \n"
                                                    "2 + \\\t\n"
                                                    "3\n",
                                                    Out));
  EXPECT_STREQ("#define A 1 + 2 + 3\n", Out.data());
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
       }) {
    ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
    EXPECT_STREQ("", Out.data());
  }

  for (auto Source : {
           "#if MACRO\n#warning '\n#endif\n",
           "#if MACRO\n#warning \"\n#endif\n",
           "#if MACRO\n#warning /*\n#endif\n",
           "#if MACRO\n#error '\n#endif\n",
           "#if MACRO\n#error \"\n#endif\n",
           "#if MACRO\n#error /*\n#endif\n",
       }) {
    ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
    EXPECT_STREQ("#if MACRO\n#endif\n", Out.data());
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

TEST(MinimizeSourceToDependencyDirectivesTest, CharacterLiteralPrefixL) {
  SmallVector<char, 128> Out;

  StringRef Source = R"(L'P'
#if DEBUG
// '
#endif
#include <test.h>
)";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#if DEBUG\n#endif\n#include <test.h>\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, CharacterLiteralPrefixU) {
  SmallVector<char, 128> Out;

  StringRef Source = R"(int x = U'P';
#include <test.h>
// '
)";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#include <test.h>\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, CharacterLiteralPrefixu) {
  SmallVector<char, 128> Out;

  StringRef Source = R"(int x = u'b';
int y = u8'a';
int z = 128'78;
#include <test.h>
// '
)";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#include <test.h>\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, PragmaOnce) {
  SmallVector<char, 128> Out;
  SmallVector<Token, 4> Tokens;

  StringRef Source = R"(// comment
#pragma once
// another comment
#include <test.h>
)";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out, Tokens));
  EXPECT_STREQ("#pragma once\n#include <test.h>\n", Out.data());
  ASSERT_EQ(Tokens.size(), 3u);
  EXPECT_EQ(Tokens[0].K,
            minimize_source_to_dependency_directives::pp_pragma_once);

  Source = R"(// comment
    #pragma once extra tokens
    // another comment
    #include <test.h>
    )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#pragma once\n#include <test.h>\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest,
     SkipLineStringCharLiteralsUntilNewline) {
  SmallVector<char, 128> Out;

  StringRef Source = R"(#if NEVER_ENABLED
    #define why(fmt, ...) #error don't try me
    #endif

    void foo();
)";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ(
      "#if NEVER_ENABLED\n#define why(fmt,...) #error don't try me\n#endif\n",
      Out.data());

  Source = R"(#if NEVER_ENABLED
      #define why(fmt, ...) "quote dropped
      #endif

      void foo();
  )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ(
      "#if NEVER_ENABLED\n#define why(fmt,...) \"quote dropped\n#endif\n",
      Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest,
     SupportWhitespaceBeforeLineContinuationInStringSkipping) {
  SmallVector<char, 128> Out;

  StringRef Source = "#define X '\\ \t\nx'\nvoid foo() {}";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#define X '\\ \t\nx'\n", Out.data());

  Source = "#define X \"\\ \r\nx\"\nvoid foo() {}";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#define X \"\\ \r\nx\"\n", Out.data());

  Source = "#define X \"\\ \r\nx\n#include <x>\n";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#define X \"\\ \r\nx\n#include <x>\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, CxxModules) {
  SmallVector<char, 128> Out;
  SmallVector<Token, 4> Tokens;

  StringRef Source = R"(
    module;
    #include "textual-header.h"

    export module m;
    exp\
ort \
      import \
      :l [[rename]];

    export void f();

    void h() {
      import.a = 3;
      import = 3;
      import <<= 3;
      import->a = 3;
      import();
      import . a();

      import a b d e d e f e;
      import foo [[no_unique_address]];
      import foo();
      import f(:sefse);
      import f(->a = 3);
    }
    )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out, Tokens));
  EXPECT_STREQ("#include \"textual-header.h\"\nexport module m;\n"
               "export import :l [[rename]];\n"
               "import <<= 3;\nimport a b d e d e f e;\n"
               "import foo [[no_unique_address]];\nimport foo();\n"
               "import f(:sefse);\nimport f(->a = 3);\n", Out.data());
  ASSERT_EQ(Tokens.size(), 12u);
  EXPECT_EQ(Tokens[0].K,
            minimize_source_to_dependency_directives::pp_include);
  EXPECT_EQ(Tokens[2].K,
            minimize_source_to_dependency_directives::cxx_module_decl);
}

TEST(MinimizeSourceToDependencyDirectivesTest, SkippedPPRangesBasic) {
  SmallString<128> Out;
  SmallVector<Token, 32> Toks;
  StringRef Source = "#ifndef GUARD\n"
                     "#define GUARD\n"
                     "void foo();\n"
                     "#endif\n";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out, Toks));
  SmallVector<SkippedRange, 4> Ranges;
  ASSERT_FALSE(computeSkippedRanges(Toks, Ranges));
  EXPECT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0].Offset, 0);
  EXPECT_EQ(Ranges[0].Length, (int)Out.find("#endif"));
}

TEST(MinimizeSourceToDependencyDirectivesTest, SkippedPPRangesBasicElifdef) {
  SmallString<128> Out;
  SmallVector<Token, 32> Toks;
  StringRef Source = "#ifdef BLAH\n"
                     "void skip();\n"
                     "#elifdef BLAM\n"
                     "void skip();\n"
                     "#elifndef GUARD\n"
                     "#define GUARD\n"
                     "void foo();\n"
                     "#endif\n";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out, Toks));
  SmallVector<SkippedRange, 4> Ranges;
  ASSERT_FALSE(computeSkippedRanges(Toks, Ranges));
  EXPECT_EQ(Ranges.size(), 3u);
  EXPECT_EQ(Ranges[0].Offset, 0);
  EXPECT_EQ(Ranges[0].Length, (int)Out.find("#elifdef"));
  EXPECT_EQ(Ranges[1].Offset, (int)Out.find("#elifdef"));
  EXPECT_EQ(Ranges[1].Offset + Ranges[1].Length, (int)Out.find("#elifndef"));
  EXPECT_EQ(Ranges[2].Offset, (int)Out.find("#elifndef"));
  EXPECT_EQ(Ranges[2].Offset + Ranges[2].Length, (int)Out.rfind("#endif"));
}

TEST(MinimizeSourceToDependencyDirectivesTest, SkippedPPRangesNested) {
  SmallString<128> Out;
  SmallVector<Token, 32> Toks;
  StringRef Source = "#ifndef GUARD\n"
                     "#define GUARD\n"
                     "#if FOO\n"
                     "#include hello\n"
                     "#elif BAR\n"
                     "#include bye\n"
                     "#endif\n"
                     "#else\n"
                     "#include nothing\n"
                     "#endif\n";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out, Toks));
  SmallVector<SkippedRange, 4> Ranges;
  ASSERT_FALSE(computeSkippedRanges(Toks, Ranges));
  EXPECT_EQ(Ranges.size(), 4u);
  EXPECT_EQ(Ranges[0].Offset, (int)Out.find("#if FOO"));
  EXPECT_EQ(Ranges[0].Offset + Ranges[0].Length, (int)Out.find("#elif"));
  EXPECT_EQ(Ranges[1].Offset, (int)Out.find("#elif BAR"));
  EXPECT_EQ(Ranges[1].Offset + Ranges[1].Length, (int)Out.find("#endif"));
  EXPECT_EQ(Ranges[2].Offset, 0);
  EXPECT_EQ(Ranges[2].Length, (int)Out.find("#else"));
  EXPECT_EQ(Ranges[3].Offset, (int)Out.find("#else"));
  EXPECT_EQ(Ranges[3].Offset + Ranges[3].Length, (int)Out.rfind("#endif"));
}

} // end anonymous namespace
