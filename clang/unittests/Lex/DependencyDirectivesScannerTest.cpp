//===- unittests/Lex/DependencyDirectivesScannerTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/DependencyDirectivesScanner.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::dependency_directives_scan;

static bool minimizeSourceToDependencyDirectives(
    StringRef Input, SmallVectorImpl<char> &Out,
    SmallVectorImpl<dependency_directives_scan::Token> &Tokens,
    SmallVectorImpl<Directive> &Directives) {
  Out.clear();
  Tokens.clear();
  Directives.clear();
  if (scanSourceForDependencyDirectives(Input, Tokens, Directives))
    return true;

  raw_svector_ostream OS(Out);
  printDependencyDirectivesAsSource(Input, Directives, OS);
  if (!Out.empty() && Out.back() != '\n')
    Out.push_back('\n');
  Out.push_back('\0');
  Out.pop_back();

  return false;
}

static bool minimizeSourceToDependencyDirectives(StringRef Input,
                                                 SmallVectorImpl<char> &Out) {
  SmallVector<dependency_directives_scan::Token, 16> Tokens;
  SmallVector<Directive, 32> Directives;
  return minimizeSourceToDependencyDirectives(Input, Out, Tokens, Directives);
}

namespace {

TEST(MinimizeSourceToDependencyDirectivesTest, Empty) {
  SmallVector<char, 128> Out;
  SmallVector<dependency_directives_scan::Token, 4> Tokens;
  SmallVector<Directive, 4> Directives;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("", Out, Tokens, Directives));
  EXPECT_TRUE(Out.empty());
  EXPECT_TRUE(Tokens.empty());
  ASSERT_EQ(1u, Directives.size());
  ASSERT_EQ(pp_eof, Directives.back().Kind);

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("abc def\nxyz", Out, Tokens,
                                                    Directives));
  EXPECT_STREQ("<TokBeforeEOF>\n", Out.data());
  EXPECT_TRUE(Tokens.empty());
  ASSERT_EQ(2u, Directives.size());
  EXPECT_EQ(tokens_present_before_eof, Directives[0].Kind);
  EXPECT_EQ(pp_eof, Directives[1].Kind);
}

TEST(MinimizeSourceToDependencyDirectivesTest, AllTokens) {
  SmallVector<char, 128> Out;
  SmallVector<dependency_directives_scan::Token, 4> Tokens;
  SmallVector<Directive, 4> Directives;

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
                                           Out, Tokens, Directives));
  EXPECT_EQ(pp_define, Directives[0].Kind);
  EXPECT_EQ(pp_undef, Directives[1].Kind);
  EXPECT_EQ(pp_endif, Directives[2].Kind);
  EXPECT_EQ(pp_if, Directives[3].Kind);
  EXPECT_EQ(pp_ifdef, Directives[4].Kind);
  EXPECT_EQ(pp_ifndef, Directives[5].Kind);
  EXPECT_EQ(pp_elifdef, Directives[6].Kind);
  EXPECT_EQ(pp_elifndef, Directives[7].Kind);
  EXPECT_EQ(pp_elif, Directives[8].Kind);
  EXPECT_EQ(pp_else, Directives[9].Kind);
  EXPECT_EQ(pp_include, Directives[10].Kind);
  EXPECT_EQ(pp_include_next, Directives[11].Kind);
  EXPECT_EQ(pp___include_macros, Directives[12].Kind);
  EXPECT_EQ(pp_import, Directives[13].Kind);
  EXPECT_EQ(decl_at_import, Directives[14].Kind);
  EXPECT_EQ(pp_pragma_import, Directives[15].Kind);
  EXPECT_EQ(pp_pragma_push_macro, Directives[16].Kind);
  EXPECT_EQ(pp_pragma_pop_macro, Directives[17].Kind);
  EXPECT_EQ(pp_pragma_include_alias, Directives[18].Kind);
  EXPECT_EQ(cxx_export_module_decl, Directives[19].Kind);
  EXPECT_EQ(cxx_import_decl, Directives[20].Kind);
  EXPECT_EQ(pp_eof, Directives[21].Kind);
}

TEST(MinimizeSourceToDependencyDirectivesTest, EmptyHash) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#\n#define MACRO a\n", Out));
  EXPECT_STREQ("#define MACRO a\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, Define) {
  SmallVector<char, 128> Out;
  SmallVector<dependency_directives_scan::Token, 4> Tokens;
  SmallVector<Directive, 4> Directives;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define MACRO", Out,
                                                    Tokens, Directives));
  EXPECT_STREQ("#define MACRO\n", Out.data());
  ASSERT_EQ(4u, Tokens.size());
  ASSERT_EQ(2u, Directives.size());
  ASSERT_EQ(pp_define, Directives.front().Kind);
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
  EXPECT_STREQ("#define MACRO con tent\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO()   con  tent   ", Out));
  EXPECT_STREQ("#define MACRO() con tent\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineInvalidMacroArguments) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define MACRO((a))", Out));
  EXPECT_STREQ("#define MACRO((a))\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define MACRO(", Out));
  EXPECT_STREQ("#define MACRO(\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#define MACRO(a * b)", Out));
  EXPECT_STREQ("#define MACRO(a*b)\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineHorizontalWhitespace) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO(\t)\tcon \t tent\t", Out));
  EXPECT_STREQ("#define MACRO() con tent\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO(\f)\fcon \f tent\f", Out));
  EXPECT_STREQ("#define MACRO() con tent\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO(\v)\vcon \v tent\v", Out));
  EXPECT_STREQ("#define MACRO() con tent\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO \t\v\f\v\t con\f\t\vtent\v\f \v", Out));
  EXPECT_STREQ("#define MACRO con tent\n", Out.data());
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

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define 0\n", Out));
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineNoName) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define &\n", Out));
}

TEST(MinimizeSourceToDependencyDirectivesTest, DefineNoWhitespace) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define AND&\n", Out));
  EXPECT_STREQ("#define AND&\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define AND\\\n"
                                                    "&\n",
                                                    Out));
  EXPECT_STREQ("#define AND\\\n"
               "&\n",
               Out.data());
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

TEST(MinimizeSourceToDependencyDirectivesTest, CommentSlashSlashStar) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#define MACRO 1 //* blah */\n", Out));
  EXPECT_STREQ("#define MACRO 1\n", Out.data());
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
  EXPECT_STREQ("<TokBeforeEOF>\n", Out.data());

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
  EXPECT_STREQ("<TokBeforeEOF>\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#pragma clang module\n", Out));
  EXPECT_STREQ("<TokBeforeEOF>\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives(
      "#pragma clang module impor\n", Out));
  EXPECT_STREQ("<TokBeforeEOF>\n", Out.data());

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
      minimizeSourceToDependencyDirectives("#include <A//A.h>\n", Out));
  EXPECT_STREQ("#include <A//A.h>\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#include \"A//A.h\"\n", Out));
  EXPECT_STREQ("#include \"A//A.h\"\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#include_next <A>\n", Out));
  EXPECT_STREQ("#include_next <A>\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#import <A>\n", Out));
  EXPECT_STREQ("#import <A>\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#import <A//A.h>\n", Out));
  EXPECT_STREQ("#import <A//A.h>\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#import \"A//A.h\"\n", Out));
  EXPECT_STREQ("#import \"A//A.h\"\n", Out.data());

  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("#__include_macros <A>\n", Out));
  EXPECT_STREQ("#__include_macros <A>\n", Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#include MACRO\n", Out));
  EXPECT_STREQ("#include MACRO\n", Out.data());
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
  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives("@import MACRO(A);\n", Out));
  ASSERT_FALSE(minimizeSourceToDependencyDirectives("@import \" \";\n", Out));
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
  EXPECT_STREQ("#if\\\n"
               "ndef GUARD\n"
               "#define GUARD\n"
               "#endif\n",
               Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define GUA\\\n"
                                                    "RD\n",
                                                    Out));
  EXPECT_STREQ("#define GUA\\\n"
               "RD\n",
               Out.data());

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define GUA\\\r"
                                                    "RD\n",
                                                    Out));
  EXPECT_STREQ("#define GUA\\\r"
               "RD\n",
               Out.data());

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
  EXPECT_STREQ("#define A 1+\\  \n"
               "2+\\\t\n"
               "3\n",
               Out.data());
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
    EXPECT_STREQ("<TokBeforeEOF>\n", Out.data());
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
  SmallVector<dependency_directives_scan::Token, 4> Tokens;
  SmallVector<Directive, 4> Directives;

  StringRef Source = R"(// comment
#pragma once
// another comment
#include <test.h>
)";
  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives(Source, Out, Tokens, Directives));
  EXPECT_STREQ("#pragma once\n#include <test.h>\n", Out.data());
  ASSERT_EQ(Directives.size(), 3u);
  EXPECT_EQ(Directives[0].Kind, dependency_directives_scan::pp_pragma_once);

  Source = R"(// comment
    #pragma once extra tokens
    // another comment
    #include <test.h>
    )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#pragma once extra tokens\n#include <test.h>\n", Out.data());
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
      "#if NEVER_ENABLED\n#define why(fmt,...) #error don't try me\n#endif\n"
      "<TokBeforeEOF>\n",
      Out.data());

  Source = R"(#if NEVER_ENABLED
      #define why(fmt, ...) "quote dropped
      #endif

      void foo();
  )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ(
      "#if NEVER_ENABLED\n#define why(fmt,...) \"quote dropped\n#endif\n"
      "<TokBeforeEOF>\n",
      Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest,
     SupportWhitespaceBeforeLineContinuation) {
  SmallVector<char, 128> Out;

  ASSERT_FALSE(minimizeSourceToDependencyDirectives("#define FOO(BAR) \\\n"
                                                    "  #BAR\\\n"
                                                    "  baz\n",
                                                    Out));
  EXPECT_STREQ("#define FOO(BAR) #BAR baz\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest,
     SupportWhitespaceBeforeLineContinuationInStringSkipping) {
  SmallVector<char, 128> Out;

  StringRef Source = "#define X '\\ \t\nx'\nvoid foo() {}";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#define X '\\ \t\nx'\n<TokBeforeEOF>\n", Out.data());

  Source = "#define X \"\\ \r\nx\"\nvoid foo() {}";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#define X \"\\ \r\nx\"\n<TokBeforeEOF>\n", Out.data());

  Source = "#define X \"\\ \r\nx\n#include <x>\n";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#define X\"\\ \r\nx\n#include <x>\n", Out.data());
}

TEST(MinimizeSourceToDependencyDirectivesTest, CxxModules) {
  SmallVector<char, 128> Out;
  SmallVector<dependency_directives_scan::Token, 4> Tokens;
  SmallVector<Directive, 4> Directives;

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
  ASSERT_FALSE(
      minimizeSourceToDependencyDirectives(Source, Out, Tokens, Directives));
  EXPECT_STREQ("#include \"textual-header.h\"\nexport module m;"
               "exp\\\nort import:l[[rename]];"
               "import<<=3;import a b d e d e f e;"
               "import foo[[no_unique_address]];import foo();"
               "import f(:sefse);import f(->a=3);"
               "<TokBeforeEOF>\n",
               Out.data());
  ASSERT_EQ(Directives.size(), 11u);
  EXPECT_EQ(Directives[0].Kind, pp_include);
  EXPECT_EQ(Directives[1].Kind, cxx_export_module_decl);
}

TEST(MinimizeSourceToDependencyDirectivesTest, TokensBeforeEOF) {
  SmallString<128> Out;

  StringRef Source = R"(
    #define A
    #ifdef B
    int x;
    #endif
    )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#define A\n<TokBeforeEOF>\n", Out.data());

  Source = R"(
    #ifndef A
    #define A
    #endif // some comment

    // other comment
    )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#ifndef A\n#define A\n#endif\n", Out.data());

  Source = R"(
    #ifndef A
    #define A
    #endif /* some comment

    */
    )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#ifndef A\n#define A\n#endif\n", Out.data());

  Source = R"(
    #ifndef A
    #define A
    #endif /* some comment

    */
    int x;
    )";
  ASSERT_FALSE(minimizeSourceToDependencyDirectives(Source, Out));
  EXPECT_STREQ("#ifndef A\n#define A\n#endif\n<TokBeforeEOF>\n", Out.data());
}

} // end anonymous namespace
