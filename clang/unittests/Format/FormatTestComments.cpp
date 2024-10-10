//===- unittest/Format/FormatTestComments.cpp - Formatting unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "format-test-comments"

namespace clang {
namespace format {
namespace test {
namespace {

FormatStyle getGoogleStyle() { return getGoogleStyle(FormatStyle::LK_Cpp); }

class FormatTestComments : public FormatTestBase {};

//===----------------------------------------------------------------------===//
// Tests for comments.
//===----------------------------------------------------------------------===//

TEST_F(FormatTestComments, UnderstandsSingleLineComments) {
  verifyFormat("//* */");
  verifyFormat("// line 1\n"
               "// line 2\n"
               "void f() {}");

  EXPECT_EQ("// comment", format("//comment"));
  EXPECT_EQ("// #comment", format("//#comment"));

  EXPECT_EQ("// comment\n"
            "// clang-format on",
            format("//comment\n"
                   "// clang-format on"));

  verifyFormat("void f() {\n"
               "  // Doesn't do anything\n"
               "}");
  verifyFormat("SomeObject\n"
               "    // Calling someFunction on SomeObject\n"
               "    .someFunction();");
  verifyFormat("auto result = SomeObject\n"
               "                  // Calling someFunction on SomeObject\n"
               "                  .someFunction();");
  verifyFormat("void f(int i,  // some comment (probably for i)\n"
               "       int j,  // some comment (probably for j)\n"
               "       int k); // some comment (probably for k)");
  verifyFormat("void f(int i,\n"
               "       // some comment (probably for j)\n"
               "       int j,\n"
               "       // some comment (probably for k)\n"
               "       int k);");

  verifyFormat("int i    // This is a fancy variable\n"
               "    = 5; // with nicely aligned comment.");

  verifyFormat("// Leading comment.\n"
               "int a; // Trailing comment.");
  verifyFormat("int a; // Trailing comment\n"
               "       // on 2\n"
               "       // or 3 lines.\n"
               "int b;");
  verifyFormat("int a; // Trailing comment\n"
               "\n"
               "// Leading comment.\n"
               "int b;");
  verifyFormat("int a;    // Comment.\n"
               "          // More details.\n"
               "int bbbb; // Another comment.");
  verifyFormat(
      "int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; // comment\n"
      "int bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;   // comment\n"
      "int cccccccccccccccccccccccccccccc;       // comment\n"
      "int ddd;                     // looooooooooooooooooooooooong comment\n"
      "int aaaaaaaaaaaaaaaaaaaaaaa; // comment\n"
      "int bbbbbbbbbbbbbbbbbbbbb;   // comment\n"
      "int ccccccccccccccccccc;     // comment");

  verifyFormat("#include \"a\"     // comment\n"
               "#include \"a/b/c\" // comment");
  verifyFormat("#include <a>     // comment\n"
               "#include <a/b/c> // comment");
  EXPECT_EQ("#include \"a\"     // comment\n"
            "#include \"a/b/c\" // comment",
            format("#include \\\n"
                   "  \"a\" // comment\n"
                   "#include \"a/b/c\" // comment"));

  verifyFormat("enum E {\n"
               "  // comment\n"
               "  VAL_A, // comment\n"
               "  VAL_B\n"
               "};");

  EXPECT_EQ("enum A {\n"
            "  // line a\n"
            "  a,\n"
            "  b, // line b\n"
            "\n"
            "  // line c\n"
            "  c\n"
            "};",
            format("enum A {\n"
                   "  // line a\n"
                   "  a,\n"
                   "  b, // line b\n"
                   "\n"
                   "  // line c\n"
                   "  c\n"
                   "};",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("enum A {\n"
            "  a, // line 1\n"
            "  // line 2\n"
            "};",
            format("enum A {\n"
                   "  a, // line 1\n"
                   "  // line 2\n"
                   "};",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("enum A {\n"
            "  a, // line 1\n"
            "     // line 2\n"
            "};",
            format("enum A {\n"
                   "  a, // line 1\n"
                   "   // line 2\n"
                   "};",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("enum A {\n"
            "  a, // line 1\n"
            "  // line 2\n"
            "  b\n"
            "};",
            format("enum A {\n"
                   "  a, // line 1\n"
                   "  // line 2\n"
                   "  b\n"
                   "};",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("enum A {\n"
            "  a, // line 1\n"
            "     // line 2\n"
            "  b\n"
            "};",
            format("enum A {\n"
                   "  a, // line 1\n"
                   "   // line 2\n"
                   "  b\n"
                   "};",
                   getLLVMStyleWithColumns(20)));
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; // Trailing comment");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    // Comment inside a statement.\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;");
  verifyFormat("SomeFunction(a,\n"
               "             // comment\n"
               "             b + x);");
  verifyFormat("SomeFunction(a, a,\n"
               "             // comment\n"
               "             b + x);");
  verifyFormat(
      "bool aaaaaaaaaaaaa = // comment\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");

  verifyFormat("int aaaa; // aaaaa\n"
               "int aa;   // aaaaaaa",
               getLLVMStyleWithColumns(20));

  EXPECT_EQ("void f() { // This does something ..\n"
            "}\n"
            "int a; // This is unrelated",
            format("void f()    {     // This does something ..\n"
                   "  }\n"
                   "int   a;     // This is unrelated"));
  EXPECT_EQ("class C {\n"
            "  void f() { // This does something ..\n"
            "  } // awesome..\n"
            "\n"
            "  int a; // This is unrelated\n"
            "};",
            format("class C{void f()    { // This does something ..\n"
                   "      } // awesome..\n"
                   " \n"
                   "int a;    // This is unrelated\n"
                   "};"));

  EXPECT_EQ("int i; // single line trailing comment",
            format("int i;\\\n// single line trailing comment"));

  verifyGoogleFormat("int a;  // Trailing comment.");

  verifyFormat("someFunction(anotherFunction( // Force break.\n"
               "    parameter));");

  verifyGoogleFormat("#endif  // HEADER_GUARD");

  verifyFormat("const char *test[] = {\n"
               "    // A\n"
               "    \"aaaa\",\n"
               "    // B\n"
               "    \"aaaaa\"};");
  verifyGoogleFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaa);  // 81_cols_with_this_comment");
  EXPECT_EQ("D(a, {\n"
            "  // test\n"
            "  int a;\n"
            "});",
            format("D(a, {\n"
                   "// test\n"
                   "int a;\n"
                   "});"));

  EXPECT_EQ("lineWith(); // comment\n"
            "// at start\n"
            "otherLine();",
            format("lineWith();   // comment\n"
                   "// at start\n"
                   "otherLine();"));
  EXPECT_EQ("lineWith(); // comment\n"
            "/*\n"
            " * at start */\n"
            "otherLine();",
            format("lineWith();   // comment\n"
                   "/*\n"
                   " * at start */\n"
                   "otherLine();"));
  EXPECT_EQ("lineWith(); // comment\n"
            "            // at start\n"
            "otherLine();",
            format("lineWith();   // comment\n"
                   " // at start\n"
                   "otherLine();"));

  EXPECT_EQ("lineWith(); // comment\n"
            "// at start\n"
            "otherLine(); // comment",
            format("lineWith();   // comment\n"
                   "// at start\n"
                   "otherLine();   // comment"));
  EXPECT_EQ("lineWith();\n"
            "// at start\n"
            "otherLine(); // comment",
            format("lineWith();\n"
                   " // at start\n"
                   "otherLine();   // comment"));
  EXPECT_EQ("// first\n"
            "// at start\n"
            "otherLine(); // comment",
            format("// first\n"
                   " // at start\n"
                   "otherLine();   // comment"));
  EXPECT_EQ("f();\n"
            "// first\n"
            "// at start\n"
            "otherLine(); // comment",
            format("f();\n"
                   "// first\n"
                   " // at start\n"
                   "otherLine();   // comment"));
  verifyFormat("f(); // comment\n"
               "// first\n"
               "// at start\n"
               "otherLine();");
  EXPECT_EQ("f(); // comment\n"
            "// first\n"
            "// at start\n"
            "otherLine();",
            format("f();   // comment\n"
                   "// first\n"
                   " // at start\n"
                   "otherLine();"));
  EXPECT_EQ("f(); // comment\n"
            "     // first\n"
            "// at start\n"
            "otherLine();",
            format("f();   // comment\n"
                   " // first\n"
                   "// at start\n"
                   "otherLine();"));
  EXPECT_EQ("void f() {\n"
            "  lineWith(); // comment\n"
            "  // at start\n"
            "}",
            format("void              f() {\n"
                   "  lineWith(); // comment\n"
                   "  // at start\n"
                   "}"));
  EXPECT_EQ("int xy; // a\n"
            "int z;  // b",
            format("int xy;    // a\n"
                   "int z;    //b"));
  EXPECT_EQ("int xy; // a\n"
            "int z; // bb",
            format("int xy;    // a\n"
                   "int z;    //bb",
                   getLLVMStyleWithColumns(12)));

  verifyFormat("#define A                                                  \\\n"
               "  int i; /* iiiiiiiiiiiiiiiiiiiii */                       \\\n"
               "  int jjjjjjjjjjjjjjjjjjjjjjjj; /* */",
               getLLVMStyleWithColumns(60));
  verifyFormat(
      "#define A                                                   \\\n"
      "  int i;                        /* iiiiiiiiiiiiiiiiiiiii */ \\\n"
      "  int jjjjjjjjjjjjjjjjjjjjjjjj; /* */",
      getLLVMStyleWithColumns(61));

  verifyFormat("if ( // This is some comment\n"
               "    x + 3) {\n"
               "}");
  EXPECT_EQ("if ( // This is some comment\n"
            "     // spanning two lines\n"
            "    x + 3) {\n"
            "}",
            format("if( // This is some comment\n"
                   "     // spanning two lines\n"
                   " x + 3) {\n"
                   "}"));

  verifyNoCrash("/\\\n/");
  verifyNoCrash("/\\\n* */");
  // The 0-character somehow makes the lexer return a proper comment.
  verifyNoCrash(StringRef("/*\\\0\n/", 6));
}

TEST_F(FormatTestComments, KeepsParameterWithTrailingCommentsOnTheirOwnLine) {
  EXPECT_EQ("SomeFunction(a,\n"
            "             b, // comment\n"
            "             c);",
            format("SomeFunction(a,\n"
                   "          b, // comment\n"
                   "      c);"));
  EXPECT_EQ("SomeFunction(a, b,\n"
            "             // comment\n"
            "             c);",
            format("SomeFunction(a,\n"
                   "          b,\n"
                   "  // comment\n"
                   "      c);"));
  EXPECT_EQ("SomeFunction(a, b, // comment (unclear relation)\n"
            "             c);",
            format("SomeFunction(a, b, // comment (unclear relation)\n"
                   "      c);"));
  EXPECT_EQ("SomeFunction(a, // comment\n"
            "             b,\n"
            "             c); // comment",
            format("SomeFunction(a,     // comment\n"
                   "          b,\n"
                   "      c); // comment"));
  EXPECT_EQ("aaaaaaaaaa(aaaa(aaaa,\n"
            "                aaaa), //\n"
            "           aaaa, bbbbb);",
            format("aaaaaaaaaa(aaaa(aaaa,\n"
                   "aaaa), //\n"
                   "aaaa, bbbbb);"));

  FormatStyle BreakAlways = getLLVMStyle();
  BreakAlways.BinPackParameters = FormatStyle::BPPS_AlwaysOnePerLine;
  verifyFormat("int SomeFunction(a,\n"
               "                 b, // comment\n"
               "                 c,\n"
               "                 d);",
               BreakAlways);
  verifyFormat("int SomeFunction(a,\n"
               "                 b,\n"
               "                 // comment\n"
               "                 c);",
               BreakAlways);
}

TEST_F(FormatTestComments, RemovesTrailingWhitespaceOfComments) {
  EXPECT_EQ("// comment", format("// comment  "));
  EXPECT_EQ("int aaaaaaa, bbbbbbb; // comment",
            format("int aaaaaaa, bbbbbbb; // comment                   ",
                   getLLVMStyleWithColumns(33)));
  EXPECT_EQ("// comment\\\n", format("// comment\\\n  \t \v   \f   "));
  EXPECT_EQ("// comment    \\\n", format("// comment    \\\n  \t \v   \f   "));
}

TEST_F(FormatTestComments, UnderstandsBlockComments) {
  verifyFormat("f(/*noSpaceAfterParameterNamingComment=*/true);");
  verifyFormat("void f() { g(/*aaa=*/x, /*bbb=*/!y, /*c=*/::c); }");
  verifyFormat("fooooooooooooooooooooooooooooo(\n"
               "    /*qq_=*/move(q), [this, b](bar<void(uint32_t)> b) {},\n"
               "    c);",
               getLLVMStyleWithColumns(60));
  EXPECT_EQ("f(aaaaaaaaaaaaaaaaaaaaaaaaa, /* Trailing comment for aa... */\n"
            "  bbbbbbbbbbbbbbbbbbbbbbbbb);",
            format("f(aaaaaaaaaaaaaaaaaaaaaaaaa ,   \\\n"
                   "/* Trailing comment for aa... */\n"
                   "  bbbbbbbbbbbbbbbbbbbbbbbbb);"));
  EXPECT_EQ(
      "f(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "  /* Leading comment for bb... */ bbbbbbbbbbbbbbbbbbbbbbbbb);",
      format("f(aaaaaaaaaaaaaaaaaaaaaaaaa    ,   \n"
             "/* Leading comment for bb... */   bbbbbbbbbbbbbbbbbbbbbbbbb);"));

  verifyFormat(
      "void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaaa) { /*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*/ }",
      "void      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "                      aaaaaaaaaaaaaaaaaa  ,\n"
      "    aaaaaaaaaaaaaaaaaa) {   /*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*/\n"
      "}");

  verifyFormat("f(/* aaaaaaaaaaaaaaaaaa = */\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat(
      "int aaaaaaaaaaaaa(/* 1st */ int bbbbbbbbbb, /* 2nd */ int ccccccccccc,\n"
      "                  /* 3rd */ int dddddddddddd);");

  auto Style = getLLVMStyle();
  Style.BinPackParameters = FormatStyle::BPPS_OnePerLine;
  verifyFormat("aaaaaaaa(/* parameter 1 */ aaaaaa,\n"
               "         /* parameter 2 */ aaaaaa,\n"
               "         /* parameter 3 */ aaaaaa,\n"
               "         /* parameter 4 */ aaaaaa);",
               Style);
  verifyFormat("int a(/* 1st */ int b, /* 2nd */ int c);", Style);
  verifyFormat("int aaaaaaaaaaaaa(/* 1st */ int bbbbbbbbbb,\n"
               "                  /* 2nd */ int ccccccccccc,\n"
               "                  /* 3rd */ int dddddddddddd);",
               Style);

  Style.BinPackParameters = FormatStyle::BPPS_AlwaysOnePerLine;
  verifyFormat("int a(/* 1st */ int b,\n"
               "      /* 2nd */ int c);",
               Style);

  // Aligning block comments in macros.
  verifyGoogleFormat("#define A        \\\n"
                     "  int i;   /*a*/ \\\n"
                     "  int jjj; /*b*/");
}

TEST_F(FormatTestComments, AlignsBlockComments) {
  EXPECT_EQ("/*\n"
            " * Really multi-line\n"
            " * comment.\n"
            " */\n"
            "void f() {}",
            format("  /*\n"
                   "   * Really multi-line\n"
                   "   * comment.\n"
                   "   */\n"
                   "  void f() {}"));
  EXPECT_EQ("class C {\n"
            "  /*\n"
            "   * Another multi-line\n"
            "   * comment.\n"
            "   */\n"
            "  void f() {}\n"
            "};",
            format("class C {\n"
                   "/*\n"
                   " * Another multi-line\n"
                   " * comment.\n"
                   " */\n"
                   "void f() {}\n"
                   "};"));
  EXPECT_EQ("/*\n"
            "  1. This is a comment with non-trivial formatting.\n"
            "     1.1. We have to indent/outdent all lines equally\n"
            "         1.1.1. to keep the formatting.\n"
            " */",
            format("  /*\n"
                   "    1. This is a comment with non-trivial formatting.\n"
                   "       1.1. We have to indent/outdent all lines equally\n"
                   "           1.1.1. to keep the formatting.\n"
                   "   */"));
  EXPECT_EQ("/*\n"
            "Don't try to outdent if there's not enough indentation.\n"
            "*/",
            format("  /*\n"
                   " Don't try to outdent if there's not enough indentation.\n"
                   " */"));

  EXPECT_EQ("int i; /* Comment with empty...\n"
            "        *\n"
            "        * line. */",
            format("int i; /* Comment with empty...\n"
                   "        *\n"
                   "        * line. */"));
  EXPECT_EQ("int foobar = 0; /* comment */\n"
            "int bar = 0;    /* multiline\n"
            "                   comment 1 */\n"
            "int baz = 0;    /* multiline\n"
            "                   comment 2 */\n"
            "int bzz = 0;    /* multiline\n"
            "                   comment 3 */",
            format("int foobar = 0; /* comment */\n"
                   "int bar = 0;    /* multiline\n"
                   "                   comment 1 */\n"
                   "int baz = 0; /* multiline\n"
                   "                comment 2 */\n"
                   "int bzz = 0;         /* multiline\n"
                   "                        comment 3 */"));
  EXPECT_EQ("int foobar = 0; /* comment */\n"
            "int bar = 0;    /* multiline\n"
            "   comment */\n"
            "int baz = 0;    /* multiline\n"
            "comment */",
            format("int foobar = 0; /* comment */\n"
                   "int bar = 0; /* multiline\n"
                   "comment */\n"
                   "int baz = 0;        /* multiline\n"
                   "comment */"));
}

TEST_F(FormatTestComments, CommentReflowingCanBeTurnedOff) {
  FormatStyle Style = getLLVMStyleWithColumns(20);
  Style.ReflowComments = FormatStyle::RCS_Never;
  verifyFormat("// aaaaaaaaa aaaaaaaaaa aaaaaaaaaa", Style);
  verifyFormat("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa */", Style);
  verifyNoChange("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa\n"
                 "aaaaaaaaa*/",
                 Style);
  verifyNoChange("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa\n"
                 "    aaaaaaaaa*/",
                 Style);
  verifyNoChange("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa\n"
                 " *    aaaaaaaaa*/",
                 Style);
}

TEST_F(FormatTestComments, CommentReflowingCanApplyOnlyToIndents) {
  FormatStyle Style = getLLVMStyleWithColumns(20);
  Style.ReflowComments = FormatStyle::RCS_IndentOnly;
  verifyFormat("// aaaaaaaaa aaaaaaaaaa aaaaaaaaaa", Style);
  verifyFormat("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa */", Style);
  verifyNoChange("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa\n"
                 "aaaaaaaaa*/",
                 Style);
  verifyNoChange("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa\n"
                 "    aaaaaaaaa*/",
                 Style);
  verifyFormat("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa\n"
               " * aaaaaaaaa*/",
               "/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa\n"
               "      * aaaaaaaaa*/",
               Style);
}

TEST_F(FormatTestComments, CorrectlyHandlesLengthOfBlockComments) {
  EXPECT_EQ("double *x; /* aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
            "              aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa */",
            format("double *x; /* aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
                   "              aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa */"));
  EXPECT_EQ(
      "void ffffffffffff(\n"
      "    int aaaaaaaa, int bbbbbbbb,\n"
      "    int cccccccccccc) { /*\n"
      "                           aaaaaaaaaa\n"
      "                           aaaaaaaaaaaaa\n"
      "                           bbbbbbbbbbbbbb\n"
      "                           bbbbbbbbbb\n"
      "                         */\n"
      "}",
      format("void ffffffffffff(int aaaaaaaa, int bbbbbbbb, int cccccccccccc)\n"
             "{ /*\n"
             "     aaaaaaaaaa aaaaaaaaaaaaa\n"
             "     bbbbbbbbbbbbbb bbbbbbbbbb\n"
             "   */\n"
             "}",
             getLLVMStyleWithColumns(40)));
}

TEST_F(FormatTestComments, DontBreakNonTrailingBlockComments) {
  EXPECT_EQ("void ffffffffff(\n"
            "    int aaaaa /* test */);",
            format("void ffffffffff(int aaaaa /* test */);",
                   getLLVMStyleWithColumns(35)));
}

TEST_F(FormatTestComments, SplitsLongCxxComments) {
  EXPECT_EQ("// A comment that\n"
            "// doesn't fit on\n"
            "// one line",
            format("// A comment that doesn't fit on one line",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/// A comment that\n"
            "/// doesn't fit on\n"
            "/// one line",
            format("/// A comment that doesn't fit on one line",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//! A comment that\n"
            "//! doesn't fit on\n"
            "//! one line",
            format("//! A comment that doesn't fit on one line",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// a b c d\n"
            "// e f  g\n"
            "// h i j k",
            format("// a b c d e f  g h i j k", getLLVMStyleWithColumns(10)));
  EXPECT_EQ(
      "// a b c d\n"
      "// e f  g\n"
      "// h i j k",
      format("\\\n// a b c d e f  g h i j k", getLLVMStyleWithColumns(10)));
  EXPECT_EQ("if (true) // A comment that\n"
            "          // doesn't fit on\n"
            "          // one line",
            format("if (true) // A comment that doesn't fit on one line   ",
                   getLLVMStyleWithColumns(30)));
  verifyNoChange("//    Don't_touch_leading_whitespace",
                 getLLVMStyleWithColumns(20));
  EXPECT_EQ("// Add leading\n"
            "// whitespace",
            format("//Add leading whitespace", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/// Add leading\n"
            "/// whitespace",
            format("///Add leading whitespace", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//! Add leading\n"
            "//! whitespace",
            format("//!Add leading whitespace", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// whitespace", format("//whitespace"));
  EXPECT_EQ("// Even if it makes the line exceed the column\n"
            "// limit",
            format("//Even if it makes the line exceed the column limit",
                   getLLVMStyleWithColumns(51)));
  verifyFormat("//--But not here");
  EXPECT_EQ("/// line 1\n"
            "// add leading whitespace",
            format("/// line 1\n"
                   "//add leading whitespace",
                   getLLVMStyleWithColumns(30)));
  EXPECT_EQ("/// line 1\n"
            "/// line 2\n"
            "//! line 3\n"
            "//! line 4\n"
            "//! line 5\n"
            "// line 6\n"
            "// line 7",
            format("///line 1\n"
                   "///line 2\n"
                   "//! line 3\n"
                   "//!line 4\n"
                   "//!line 5\n"
                   "// line 6\n"
                   "//line 7",
                   getLLVMStyleWithColumns(20)));

  EXPECT_EQ("// aa bb cc dd",
            format("// aa bb             cc dd                   ",
                   getLLVMStyleWithColumns(15)));

  EXPECT_EQ("// A comment before\n"
            "// a macro\n"
            "// definition\n"
            "#define a b",
            format("// A comment before a macro definition\n"
                   "#define a b",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("void ffffff(\n"
            "    int aaaaaaaaa,  // wwww\n"
            "    int bbbbbbbbbb, // xxxxxxx\n"
            "                    // yyyyyyyyyy\n"
            "    int c, int d, int e) {}",
            format("void ffffff(\n"
                   "    int aaaaaaaaa, // wwww\n"
                   "    int bbbbbbbbbb, // xxxxxxx yyyyyyyyyy\n"
                   "    int c, int d, int e) {}",
                   getLLVMStyleWithColumns(40)));
  verifyFormat("//\t aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
               getLLVMStyleWithColumns(20));
  EXPECT_EQ(
      "#define XXX // a b c d\n"
      "            // e f g h",
      format("#define XXX // a b c d e f g h", getLLVMStyleWithColumns(22)));
  EXPECT_EQ(
      "#define XXX // q w e r\n"
      "            // t y u i",
      format("#define XXX //q w e r t y u i", getLLVMStyleWithColumns(22)));
  EXPECT_EQ("{\n"
            "  //\n"
            "  //\\\n"
            "  // long 1 2 3 4 5\n"
            "}",
            format("{\n"
                   "  //\n"
                   "  //\\\n"
                   "  // long 1 2 3 4 5\n"
                   "}",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("{\n"
            "  //\n"
            "  //\\\n"
            "  // long 1 2 3 4 5\n"
            "  // 6\n"
            "}",
            format("{\n"
                   "  //\n"
                   "  //\\\n"
                   "  // long 1 2 3 4 5 6\n"
                   "}",
                   getLLVMStyleWithColumns(20)));

  EXPECT_EQ("//: A comment that\n"
            "//: doesn't fit on\n"
            "//: one line",
            format("//: A comment that doesn't fit on one line",
                   getLLVMStyleWithColumns(20)));

  verifyFormat(
      "//\t\t\t\tofMap(message.velocity, 0, 127, 0, ofGetWidth()\n"
      "//* 0.2)",
      "//\t\t\t\tofMap(message.velocity, 0, 127, 0, ofGetWidth() * 0.2)");
}

TEST_F(FormatTestComments, PreservesHangingIndentInCxxComments) {
  EXPECT_EQ("//     A comment\n"
            "//     that doesn't\n"
            "//     fit on one\n"
            "//     line",
            format("//     A comment that doesn't fit on one line",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("///     A comment\n"
            "///     that doesn't\n"
            "///     fit on one\n"
            "///     line",
            format("///     A comment that doesn't fit on one line",
                   getLLVMStyleWithColumns(20)));
}

TEST_F(FormatTestComments, DontSplitLineCommentsWithEscapedNewlines) {
  EXPECT_EQ("// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
            "// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
            "// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            format("// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
                   "// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
                   "// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
  EXPECT_EQ("int a; // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
            "       // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
            "       // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            format("int a; // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
                   "       // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
                   "       // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                   getLLVMStyleWithColumns(50)));
  // FIXME: One day we might want to implement adjustment of leading whitespace
  // of the consecutive lines in this kind of comment:
  EXPECT_EQ("double\n"
            "    a; // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
            "          // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
            "          // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            format("double a; // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
                   "          // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
                   "          // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                   getLLVMStyleWithColumns(49)));
}

TEST_F(FormatTestComments, DontIntroduceMultilineComments) {
  // Avoid introducing a multiline comment by breaking after `\`.
  for (int ColumnLimit = 15; ColumnLimit <= 17; ++ColumnLimit) {
    EXPECT_EQ(
        "// aaaaaaaaaa\n"
        "// \\ bb",
        format("// aaaaaaaaaa \\ bb", getLLVMStyleWithColumns(ColumnLimit)));
    EXPECT_EQ(
        "// aaaaaaaaa\n"
        "// \\  bb",
        format("// aaaaaaaaa \\  bb", getLLVMStyleWithColumns(ColumnLimit)));
    EXPECT_EQ(
        "// aaaaaaaaa\n"
        "// \\  \\ bb",
        format("// aaaaaaaaa \\  \\ bb", getLLVMStyleWithColumns(ColumnLimit)));
  }
}

TEST_F(FormatTestComments, DontSplitLineCommentsWithPragmas) {
  FormatStyle Pragmas = getLLVMStyleWithColumns(30);
  Pragmas.CommentPragmas = "^ IWYU pragma:";
  EXPECT_EQ(
      "// IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb",
      format("// IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb", Pragmas));
  EXPECT_EQ(
      "/* IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb */",
      format("/* IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb */", Pragmas));
}

TEST_F(FormatTestComments, PriorityOfCommentBreaking) {
  EXPECT_EQ("if (xxx ==\n"
            "        yyy && // aaaaaaaaaaaa bbbbbbbbb\n"
            "    zzz)\n"
            "  q();",
            format("if (xxx == yyy && // aaaaaaaaaaaa bbbbbbbbb\n"
                   "    zzz) q();",
                   getLLVMStyleWithColumns(40)));
  EXPECT_EQ("if (xxxxxxxxxx ==\n"
            "        yyy && // aaaaaa bbbbbbbb cccc\n"
            "    zzz)\n"
            "  q();",
            format("if (xxxxxxxxxx == yyy && // aaaaaa bbbbbbbb cccc\n"
                   "    zzz) q();",
                   getLLVMStyleWithColumns(40)));
  EXPECT_EQ("if (xxxxxxxxxx &&\n"
            "        yyy || // aaaaaa bbbbbbbb cccc\n"
            "    zzz)\n"
            "  q();",
            format("if (xxxxxxxxxx && yyy || // aaaaaa bbbbbbbb cccc\n"
                   "    zzz) q();",
                   getLLVMStyleWithColumns(40)));
  EXPECT_EQ("fffffffff(\n"
            "    &xxx, // aaaaaaaaaaaa bbbbbbbbbbb\n"
            "    zzz);",
            format("fffffffff(&xxx, // aaaaaaaaaaaa bbbbbbbbbbb\n"
                   " zzz);",
                   getLLVMStyleWithColumns(40)));
}

TEST_F(FormatTestComments, MultiLineCommentsInDefines) {
  EXPECT_EQ("#define A(x) /* \\\n"
            "  a comment     \\\n"
            "  inside */     \\\n"
            "  f();",
            format("#define A(x) /* \\\n"
                   "  a comment     \\\n"
                   "  inside */     \\\n"
                   "  f();",
                   getLLVMStyleWithColumns(17)));
  EXPECT_EQ("#define A(      \\\n"
            "    x) /*       \\\n"
            "  a comment     \\\n"
            "  inside */     \\\n"
            "  f();",
            format("#define A(      \\\n"
                   "    x) /*       \\\n"
                   "  a comment     \\\n"
                   "  inside */     \\\n"
                   "  f();",
                   getLLVMStyleWithColumns(17)));
}

TEST_F(FormatTestComments, ParsesCommentsAdjacentToPPDirectives) {
  EXPECT_EQ("namespace {}\n// Test\n#define A",
            format("namespace {}\n   // Test\n#define A"));
  EXPECT_EQ("namespace {}\n/* Test */\n#define A",
            format("namespace {}\n   /* Test */\n#define A"));
  EXPECT_EQ("namespace {}\n/* Test */ #define A",
            format("namespace {}\n   /* Test */    #define A"));
}

TEST_F(FormatTestComments, KeepsLevelOfCommentBeforePPDirective) {
  // Keep the current level if the comment was originally not aligned with
  // the preprocessor directive.
  EXPECT_EQ("void f() {\n"
            "  int i;\n"
            "  /* comment */\n"
            "#ifdef A\n"
            "  int j;\n"
            "}",
            format("void f() {\n"
                   "  int i;\n"
                   "  /* comment */\n"
                   "#ifdef A\n"
                   "  int j;\n"
                   "}"));

  EXPECT_EQ("void f() {\n"
            "  int i;\n"
            "  /* comment */\n"
            "\n"
            "#ifdef A\n"
            "  int j;\n"
            "}",
            format("void f() {\n"
                   "  int i;\n"
                   "  /* comment */\n"
                   "\n"
                   "#ifdef A\n"
                   "  int j;\n"
                   "}"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    ++i;\n"
            "  }\n"
            "  // comment\n"
            "#ifdef A\n"
            "  int j;\n"
            "#endif\n"
            "}",
            format("int f(int i) {\n"
                   "  if (true) {\n"
                   "    ++i;\n"
                   "  }\n"
                   "  // comment\n"
                   "#ifdef A\n"
                   "int j;\n"
                   "#endif\n"
                   "}"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    i++;\n"
            "  } else {\n"
            "    // comment in else\n"
            "#ifdef A\n"
            "    j++;\n"
            "#endif\n"
            "  }\n"
            "}",
            format("int f(int i) {\n"
                   "  if (true) {\n"
                   "    i++;\n"
                   "  } else {\n"
                   "  // comment in else\n"
                   "#ifdef A\n"
                   "    j++;\n"
                   "#endif\n"
                   "  }\n"
                   "}"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    i++;\n"
            "  } else {\n"
            "    /* comment in else */\n"
            "#ifdef A\n"
            "    j++;\n"
            "#endif\n"
            "  }\n"
            "}",
            format("int f(int i) {\n"
                   "  if (true) {\n"
                   "    i++;\n"
                   "  } else {\n"
                   "  /* comment in else */\n"
                   "#ifdef A\n"
                   "    j++;\n"
                   "#endif\n"
                   "  }\n"
                   "}"));

  // Keep the current level if there is an empty line between the comment and
  // the preprocessor directive.
  EXPECT_EQ("void f() {\n"
            "  int i;\n"
            "  /* comment */\n"
            "\n"
            "#ifdef A\n"
            "  int j;\n"
            "}",
            format("void f() {\n"
                   "  int i;\n"
                   "/* comment */\n"
                   "\n"
                   "#ifdef A\n"
                   "  int j;\n"
                   "}"));

  EXPECT_EQ("void f() {\n"
            "  int i;\n"
            "  return i;\n"
            "}\n"
            "// comment\n"
            "\n"
            "#ifdef A\n"
            "int i;\n"
            "#endif // A",
            format("void f() {\n"
                   "   int i;\n"
                   "  return i;\n"
                   "}\n"
                   "// comment\n"
                   "\n"
                   "#ifdef A\n"
                   "int i;\n"
                   "#endif // A"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    ++i;\n"
            "  }\n"
            "  // comment\n"
            "\n"
            "#ifdef A\n"
            "  int j;\n"
            "#endif\n"
            "}",
            format("int f(int i) {\n"
                   "   if (true) {\n"
                   "    ++i;\n"
                   "  }\n"
                   "  // comment\n"
                   "\n"
                   "#ifdef A\n"
                   "  int j;\n"
                   "#endif\n"
                   "}"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    i++;\n"
            "  } else {\n"
            "    // comment in else\n"
            "\n"
            "#ifdef A\n"
            "    j++;\n"
            "#endif\n"
            "  }\n"
            "}",
            format("int f(int i) {\n"
                   "  if (true) {\n"
                   "    i++;\n"
                   "  } else {\n"
                   "// comment in else\n"
                   "\n"
                   "#ifdef A\n"
                   "    j++;\n"
                   "#endif\n"
                   "  }\n"
                   "}"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    i++;\n"
            "  } else {\n"
            "    /* comment in else */\n"
            "\n"
            "#ifdef A\n"
            "    j++;\n"
            "#endif\n"
            "  }\n"
            "}",
            format("int f(int i) {\n"
                   "  if (true) {\n"
                   "    i++;\n"
                   "  } else {\n"
                   "/* comment in else */\n"
                   "\n"
                   "#ifdef A\n"
                   "    j++;\n"
                   "#endif\n"
                   "  }\n"
                   "}"));

  // Align with the preprocessor directive if the comment was originally aligned
  // with the preprocessor directive and there is no newline between the comment
  // and the preprocessor directive.
  EXPECT_EQ("void f() {\n"
            "  int i;\n"
            "/* comment */\n"
            "#ifdef A\n"
            "  int j;\n"
            "}",
            format("void f() {\n"
                   "  int i;\n"
                   "/* comment */\n"
                   "#ifdef A\n"
                   "  int j;\n"
                   "}"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    ++i;\n"
            "  }\n"
            "// comment\n"
            "#ifdef A\n"
            "  int j;\n"
            "#endif\n"
            "}",
            format("int f(int i) {\n"
                   "   if (true) {\n"
                   "    ++i;\n"
                   "  }\n"
                   "// comment\n"
                   "#ifdef A\n"
                   "  int j;\n"
                   "#endif\n"
                   "}"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    i++;\n"
            "  } else {\n"
            "// comment in else\n"
            "#ifdef A\n"
            "    j++;\n"
            "#endif\n"
            "  }\n"
            "}",
            format("int f(int i) {\n"
                   "  if (true) {\n"
                   "    i++;\n"
                   "  } else {\n"
                   " // comment in else\n"
                   " #ifdef A\n"
                   "    j++;\n"
                   "#endif\n"
                   "  }\n"
                   "}"));

  EXPECT_EQ("int f(int i) {\n"
            "  if (true) {\n"
            "    i++;\n"
            "  } else {\n"
            "/* comment in else */\n"
            "#ifdef A\n"
            "    j++;\n"
            "#endif\n"
            "  }\n"
            "}",
            format("int f(int i) {\n"
                   "  if (true) {\n"
                   "    i++;\n"
                   "  } else {\n"
                   " /* comment in else */\n"
                   " #ifdef A\n"
                   "    j++;\n"
                   "#endif\n"
                   "  }\n"
                   "}"));

  const StringRef Code("void func() {\n"
                       "  // clang-format off\n"
                       "  #define KV(value) #value, value\n"
                       "  // clang-format on\n"
                       "}");
  verifyNoChange(Code);

  auto Style = getLLVMStyle();
  Style.IndentPPDirectives = FormatStyle::PPDIS_BeforeHash;
  verifyFormat("#ifdef FOO\n"
               "  // Foo\n"
               "  #define Foo foo\n"
               "#else\n"
               "  // Bar\n"
               "  #define Bar bar\n"
               "#endif",
               Style);
}

TEST_F(FormatTestComments, CommentsBetweenUnbracedBodyAndPPDirective) {
  verifyFormat("{\n"
               "  if (a)\n"
               "    f(); // comment\n"
               "#define A\n"
               "}");

  verifyFormat("{\n"
               "  while (a)\n"
               "    f();\n"
               "// comment\n"
               "#define A\n"
               "}");

  verifyNoChange("{\n"
                 "  if (a)\n"
                 "    f();\n"
                 "  // comment\n"
                 "#define A\n"
                 "}");

  verifyNoChange("{\n"
                 "  while (a)\n"
                 "    if (b)\n"
                 "      f();\n"
                 "  // comment\n"
                 "#define A\n"
                 "}");
}

TEST_F(FormatTestComments, SplitsLongLinesInComments) {
  // FIXME: Do we need to fix up the "  */" at the end?
  // It doesn't look like any of our current logic triggers this.
  EXPECT_EQ("/* This is a long\n"
            " * comment that\n"
            " * doesn't fit on\n"
            " * one line.  */",
            format("/* "
                   "This is a long                                         "
                   "comment that "
                   "doesn't                                    "
                   "fit on one line.  */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ(
      "/* a b c d\n"
      " * e f  g\n"
      " * h i j k\n"
      " */",
      format("/* a b c d e f  g h i j k */", getLLVMStyleWithColumns(10)));
  EXPECT_EQ(
      "/* a b c d\n"
      " * e f  g\n"
      " * h i j k\n"
      " */",
      format("\\\n/* a b c d e f  g h i j k */", getLLVMStyleWithColumns(10)));
  EXPECT_EQ("/*\n"
            "This is a long\n"
            "comment that doesn't\n"
            "fit on one line.\n"
            "*/",
            format("/*\n"
                   "This is a long                                         "
                   "comment that doesn't                                    "
                   "fit on one line.                                      \n"
                   "*/",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/*\n"
            " * This is a long\n"
            " * comment that\n"
            " * doesn't fit on\n"
            " * one line.\n"
            " */",
            format("/*      \n"
                   " * This is a long "
                   "   comment that     "
                   "   doesn't fit on   "
                   "   one line.                                            \n"
                   " */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/*\n"
            " * This_is_a_comment_with_words_that_dont_fit_on_one_line\n"
            " * so_it_should_be_broken\n"
            " * wherever_a_space_occurs\n"
            " */",
            format("/*\n"
                   " * This_is_a_comment_with_words_that_dont_fit_on_one_line "
                   "   so_it_should_be_broken "
                   "   wherever_a_space_occurs                             \n"
                   " */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/*\n"
            " *    This_comment_can_not_be_broken_into_lines\n"
            " */",
            format("/*\n"
                   " *    This_comment_can_not_be_broken_into_lines\n"
                   " */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("{\n"
            "  /*\n"
            "  This is another\n"
            "  long comment that\n"
            "  doesn't fit on one\n"
            "  line    1234567890\n"
            "  */\n"
            "}",
            format("{\n"
                   "/*\n"
                   "This is another     "
                   "  long comment that "
                   "  doesn't fit on one"
                   "  line    1234567890\n"
                   "*/\n"
                   "}",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("{\n"
            "  /*\n"
            "   * This        i s\n"
            "   * another comment\n"
            "   * t hat  doesn' t\n"
            "   * fit on one l i\n"
            "   * n e\n"
            "   */\n"
            "}",
            format("{\n"
                   "/*\n"
                   " * This        i s"
                   "   another comment"
                   "   t hat  doesn' t"
                   "   fit on one l i"
                   "   n e\n"
                   " */\n"
                   "}",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/*\n"
            " * This is a long\n"
            " * comment that\n"
            " * doesn't fit on\n"
            " * one line\n"
            " */",
            format("   /*\n"
                   "    * This is a long comment that doesn't fit on one line\n"
                   "    */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("{\n"
            "  if (something) /* This is a\n"
            "                    long\n"
            "                    comment */\n"
            "    ;\n"
            "}",
            format("{\n"
                   "  if (something) /* This is a long comment */\n"
                   "    ;\n"
                   "}",
                   getLLVMStyleWithColumns(30)));

  EXPECT_EQ("/* A comment before\n"
            " * a macro\n"
            " * definition */\n"
            "#define a b",
            format("/* A comment before a macro definition */\n"
                   "#define a b",
                   getLLVMStyleWithColumns(20)));

  EXPECT_EQ("/* some comment\n"
            " *   a comment that\n"
            " * we break another\n"
            " * comment we have\n"
            " * to break a left\n"
            " * comment\n"
            " */",
            format("  /* some comment\n"
                   "       *   a comment that we break\n"
                   "   * another comment we have to break\n"
                   "* a left comment\n"
                   "   */",
                   getLLVMStyleWithColumns(20)));

  EXPECT_EQ("/**\n"
            " * multiline block\n"
            " * comment\n"
            " *\n"
            " */",
            format("/**\n"
                   " * multiline block comment\n"
                   " *\n"
                   " */",
                   getLLVMStyleWithColumns(20)));

  // This reproduces a crashing bug where both adaptStartOfLine and
  // getCommentSplit were trying to wrap after the "/**".
  verifyFormat("/** multilineblockcommentwithnowrapopportunity */",
               getLLVMStyleWithColumns(20));

  EXPECT_EQ("/*\n"
            "\n"
            "\n"
            "    */",
            format("  /*       \n"
                   "      \n"
                   "               \n"
                   "      */"));

  EXPECT_EQ("/* a a */",
            format("/* a a            */", getLLVMStyleWithColumns(15)));
  EXPECT_EQ("/* a a bc  */",
            format("/* a a            bc  */", getLLVMStyleWithColumns(15)));
  EXPECT_EQ("/* aaa aaa\n"
            " * aaaaa */",
            format("/* aaa aaa aaaaa       */", getLLVMStyleWithColumns(15)));
  EXPECT_EQ("/* aaa aaa\n"
            " * aaaaa     */",
            format("/* aaa aaa aaaaa     */", getLLVMStyleWithColumns(15)));
}

TEST_F(FormatTestComments, SplitsLongLinesInCommentsInPreprocessor) {
  EXPECT_EQ("#define X          \\\n"
            "  /*               \\\n"
            "   Test            \\\n"
            "   Macro comment   \\\n"
            "   with a long     \\\n"
            "   line            \\\n"
            "   */              \\\n"
            "  A + B",
            format("#define X \\\n"
                   "  /*\n"
                   "   Test\n"
                   "   Macro comment with a long  line\n"
                   "   */ \\\n"
                   "  A + B",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("#define X          \\\n"
            "  /* Macro comment \\\n"
            "     with a long   \\\n"
            "     line */       \\\n"
            "  A + B",
            format("#define X \\\n"
                   "  /* Macro comment with a long\n"
                   "     line */ \\\n"
                   "  A + B",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("#define X          \\\n"
            "  /* Macro comment \\\n"
            "   * with a long   \\\n"
            "   * line */       \\\n"
            "  A + B",
            format("#define X \\\n"
                   "  /* Macro comment with a long  line */ \\\n"
                   "  A + B",
                   getLLVMStyleWithColumns(20)));
}

TEST_F(FormatTestComments, KeepsTrailingPPCommentsAndSectionCommentsSeparate) {
  verifyFormat("#ifdef A // line about A\n"
               "// section comment\n"
               "#endif",
               getLLVMStyleWithColumns(80));
  verifyFormat("#ifdef A // line 1 about A\n"
               "         // line 2 about A\n"
               "// section comment\n"
               "#endif",
               getLLVMStyleWithColumns(80));
  EXPECT_EQ("#ifdef A // line 1 about A\n"
            "         // line 2 about A\n"
            "// section comment\n"
            "#endif",
            format("#ifdef A // line 1 about A\n"
                   "          // line 2 about A\n"
                   "// section comment\n"
                   "#endif",
                   getLLVMStyleWithColumns(80)));
  verifyFormat("int f() {\n"
               "  int i;\n"
               "#ifdef A // comment about A\n"
               "  // section comment 1\n"
               "  // section comment 2\n"
               "  i = 2;\n"
               "#else // comment about #else\n"
               "  // section comment 3\n"
               "  i = 4;\n"
               "#endif\n"
               "}",
               getLLVMStyleWithColumns(80));
}

TEST_F(FormatTestComments, AlignsPPElseEndifComments) {
  verifyFormat("#if A\n"
               "#else  // A\n"
               "int iiii;\n"
               "#endif // B",
               getLLVMStyleWithColumns(20));
  verifyFormat("#if A\n"
               "#else  // A\n"
               "int iiii; // CC\n"
               "#endif // B",
               getLLVMStyleWithColumns(20));
  EXPECT_EQ("#if A\n"
            "#else  // A1\n"
            "       // A2\n"
            "int ii;\n"
            "#endif // B",
            format("#if A\n"
                   "#else  // A1\n"
                   "       // A2\n"
                   "int ii;\n"
                   "#endif // B",
                   getLLVMStyleWithColumns(20)));
}

TEST_F(FormatTestComments, CommentsInStaticInitializers) {
  EXPECT_EQ(
      "static SomeType type = {aaaaaaaaaaaaaaaaaaaa, /* comment */\n"
      "                        aaaaaaaaaaaaaaaaaaaa /* comment */,\n"
      "                        /* comment */ aaaaaaaaaaaaaaaaaaaa,\n"
      "                        aaaaaaaaaaaaaaaaaaaa, // comment\n"
      "                        aaaaaaaaaaaaaaaaaaaa};",
      format("static SomeType type = { aaaaaaaaaaaaaaaaaaaa  ,  /* comment */\n"
             "                   aaaaaaaaaaaaaaaaaaaa   /* comment */ ,\n"
             "                     /* comment */   aaaaaaaaaaaaaaaaaaaa ,\n"
             "              aaaaaaaaaaaaaaaaaaaa ,   // comment\n"
             "                  aaaaaaaaaaaaaaaaaaaa };"));
  verifyFormat("static SomeType type = {aaaaaaaaaaa, // comment for aa...\n"
               "                        bbbbbbbbbbb, ccccccccccc};");
  verifyFormat("static SomeType type = {aaaaaaaaaaa,\n"
               "                        // comment for bb....\n"
               "                        bbbbbbbbbbb, ccccccccccc};");
  verifyGoogleFormat(
      "static SomeType type = {aaaaaaaaaaa,  // comment for aa...\n"
      "                        bbbbbbbbbbb, ccccccccccc};");
  verifyGoogleFormat("static SomeType type = {aaaaaaaaaaa,\n"
                     "                        // comment for bb....\n"
                     "                        bbbbbbbbbbb, ccccccccccc};");

  verifyFormat("S s = {{a, b, c},  // Group #1\n"
               "       {d, e, f},  // Group #2\n"
               "       {g, h, i}}; // Group #3");
  verifyFormat("S s = {{// Group #1\n"
               "        a, b, c},\n"
               "       {// Group #2\n"
               "        d, e, f},\n"
               "       {// Group #3\n"
               "        g, h, i}};");

  EXPECT_EQ("S s = {\n"
            "    // Some comment\n"
            "    a,\n"
            "\n"
            "    // Comment after empty line\n"
            "    b}",
            format("S s =    {\n"
                   "      // Some comment\n"
                   "  a,\n"
                   "  \n"
                   "     // Comment after empty line\n"
                   "      b\n"
                   "}"));
  EXPECT_EQ("S s = {\n"
            "    /* Some comment */\n"
            "    a,\n"
            "\n"
            "    /* Comment after empty line */\n"
            "    b}",
            format("S s =    {\n"
                   "      /* Some comment */\n"
                   "  a,\n"
                   "  \n"
                   "     /* Comment after empty line */\n"
                   "      b\n"
                   "}"));
  verifyFormat("const uint8_t aaaaaaaaaaaaaaaaaaaaaa[0] = {\n"
               "    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // comment\n"
               "    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // comment\n"
               "    0x00, 0x00, 0x00, 0x00};            // comment");
}

TEST_F(FormatTestComments, LineCommentsAfterRightBrace) {
  EXPECT_EQ("if (true) { // comment about branch\n"
            "  // comment about f\n"
            "  f();\n"
            "}",
            format("if (true) { // comment about branch\n"
                   "  // comment about f\n"
                   "  f();\n"
                   "}",
                   getLLVMStyleWithColumns(80)));
  EXPECT_EQ("if (1) { // if line 1\n"
            "         // if line 2\n"
            "         // if line 3\n"
            "  // f line 1\n"
            "  // f line 2\n"
            "  f();\n"
            "} else { // else line 1\n"
            "         // else line 2\n"
            "         // else line 3\n"
            "  // g line 1\n"
            "  g();\n"
            "}",
            format("if (1) { // if line 1\n"
                   "          // if line 2\n"
                   "        // if line 3\n"
                   "  // f line 1\n"
                   "    // f line 2\n"
                   "  f();\n"
                   "} else { // else line 1\n"
                   "        // else line 2\n"
                   "         // else line 3\n"
                   "  // g line 1\n"
                   "  g();\n"
                   "}"));
  EXPECT_EQ("do { // line 1\n"
            "     // line 2\n"
            "     // line 3\n"
            "  f();\n"
            "} while (true);",
            format("do { // line 1\n"
                   "     // line 2\n"
                   "   // line 3\n"
                   "  f();\n"
                   "} while (true);",
                   getLLVMStyleWithColumns(80)));
  EXPECT_EQ("while (a < b) { // line 1\n"
            "  // line 2\n"
            "  // line 3\n"
            "  f();\n"
            "}",
            format("while (a < b) {// line 1\n"
                   "  // line 2\n"
                   "  // line 3\n"
                   "  f();\n"
                   "}",
                   getLLVMStyleWithColumns(80)));
}

TEST_F(FormatTestComments, ReflowsComments) {
  // Break a long line and reflow with the full next line.
  EXPECT_EQ("// long long long\n"
            "// long long",
            format("// long long long long\n"
                   "// long",
                   getLLVMStyleWithColumns(20)));

  // Keep the trailing newline while reflowing.
  EXPECT_EQ("// long long long\n"
            "// long long",
            format("// long long long long\n"
                   "// long",
                   getLLVMStyleWithColumns(20)));

  // Break a long line and reflow with a part of the next line.
  EXPECT_EQ("// long long long\n"
            "// long long\n"
            "// long_long",
            format("// long long long long\n"
                   "// long long_long",
                   getLLVMStyleWithColumns(20)));

  // Break but do not reflow if the first word from the next line is too long.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// long_long_long",
            format("// long long long long\n"
                   "// long_long_long",
                   getLLVMStyleWithColumns(20)));

  // Don't break or reflow short lines.
  verifyFormat("// long\n"
               "// long long long lo\n"
               "// long long long lo\n"
               "// long",
               getLLVMStyleWithColumns(20));

  // Keep prefixes and decorations while reflowing.
  EXPECT_EQ("/// long long long\n"
            "/// long long",
            format("/// long long long long\n"
                   "/// long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//! long long long\n"
            "//! long long",
            format("//! long long long long\n"
                   "//! long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* long long long\n"
            " * long long */",
            format("/* long long long long\n"
                   " * long */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("///< long long long\n"
            "///< long long",
            format("///< long long long long\n"
                   "///< long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//!< long long long\n"
            "//!< long long",
            format("//!< long long long long\n"
                   "//!< long",
                   getLLVMStyleWithColumns(20)));

  // Don't bring leading whitespace up while reflowing.
  EXPECT_EQ("/*  long long long\n"
            " * long long long\n"
            " */",
            format("/*  long long long long\n"
                   " *  long long\n"
                   " */",
                   getLLVMStyleWithColumns(20)));

  // Reflow the last line of a block comment with its trailing '*/'.
  EXPECT_EQ("/* long long long\n"
            "   long long */",
            format("/* long long long long\n"
                   "   long */",
                   getLLVMStyleWithColumns(20)));

  // Reflow two short lines; keep the postfix of the last one.
  EXPECT_EQ("/* long long long\n"
            " * long long long */",
            format("/* long long long long\n"
                   " * long\n"
                   " * long */",
                   getLLVMStyleWithColumns(20)));

  // Put the postfix of the last short reflow line on a newline if it doesn't
  // fit.
  EXPECT_EQ("/* long long long\n"
            " * long long longg\n"
            " */",
            format("/* long long long long\n"
                   " * long\n"
                   " * longg */",
                   getLLVMStyleWithColumns(20)));

  // Reflow lines with leading whitespace.
  EXPECT_EQ("{\n"
            "  /*\n"
            "   * long long long\n"
            "   * long long long\n"
            "   * long long long\n"
            "   */\n"
            "}",
            format("{\n"
                   "/*\n"
                   " * long long long long\n"
                   " *   long\n"
                   " * long long long long\n"
                   " */\n"
                   "}",
                   getLLVMStyleWithColumns(20)));

  // Break single line block comments that are first in the line with ' *'
  // decoration.
  EXPECT_EQ("/* long long long\n"
            " * long */",
            format("/* long long long long */", getLLVMStyleWithColumns(20)));

  // Break single line block comment that are not first in the line with '  '
  // decoration.
  EXPECT_EQ("int i; /* long long\n"
            "          long */",
            format("int i; /* long long long */", getLLVMStyleWithColumns(20)));

  // Reflow a line that goes just over the column limit.
  EXPECT_EQ("// long long long\n"
            "// lon long",
            format("// long long long lon\n"
                   "// long",
                   getLLVMStyleWithColumns(20)));

  // Stop reflowing if the next line has a different indentation than the
  // previous line.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "//  long long\n"
            "//  long",
            format("// long long long long\n"
                   "//  long long\n"
                   "//  long",
                   getLLVMStyleWithColumns(20)));

  // Reflow into the last part of a really long line that has been broken into
  // multiple lines.
  EXPECT_EQ("// long long long\n"
            "// long long long\n"
            "// long long long",
            format("// long long long long long long long long\n"
                   "// long",
                   getLLVMStyleWithColumns(20)));

  // Break the first line, then reflow the beginning of the second and third
  // line up.
  EXPECT_EQ("// long long long\n"
            "// lon1 lon2 lon2\n"
            "// lon2 lon3 lon3",
            format("// long long long lon1\n"
                   "// lon2 lon2 lon2\n"
                   "// lon3 lon3",
                   getLLVMStyleWithColumns(20)));

  // Reflow the beginning of the second line, then break the rest.
  EXPECT_EQ("// long long long\n"
            "// lon1 lon2 lon2\n"
            "// lon2 lon2 lon2\n"
            "// lon3",
            format("// long long long lon1\n"
                   "// lon2 lon2 lon2 lon2 lon2 lon3",
                   getLLVMStyleWithColumns(20)));

  // Shrink the first line, then reflow the second line up.
  EXPECT_EQ("// long long long", format("// long              long\n"
                                        "// long",
                                        getLLVMStyleWithColumns(20)));

  // Don't shrink leading whitespace.
  verifyNoChange("int i; ///           a", getLLVMStyleWithColumns(20));

  // Shrink trailing whitespace if there is no postfix and reflow.
  EXPECT_EQ("// long long long\n"
            "// long long",
            format("// long long long long    \n"
                   "// long",
                   getLLVMStyleWithColumns(20)));

  // Shrink trailing whitespace to a single one if there is postfix.
  EXPECT_EQ("/* long long long */",
            format("/* long long long     */", getLLVMStyleWithColumns(20)));

  // Break a block comment postfix if exceeding the line limit.
  EXPECT_EQ("/*               long\n"
            " */",
            format("/*               long */", getLLVMStyleWithColumns(20)));

  // Reflow indented comments.
  EXPECT_EQ("{\n"
            "  // long long long\n"
            "  // long long\n"
            "  int i; /* long lon\n"
            "            g long\n"
            "          */\n"
            "}",
            format("{\n"
                   "  // long long long long\n"
                   "  // long\n"
                   "  int i; /* long lon g\n"
                   "            long */\n"
                   "}",
                   getLLVMStyleWithColumns(20)));

  // Don't realign trailing comments after reflow has happened.
  EXPECT_EQ("// long long long\n"
            "// long long\n"
            "long i; // long",
            format("// long long long long\n"
                   "// long\n"
                   "long i; // long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// long long long\n"
            "// longng long long\n"
            "// long lo",
            format("// long long long longng\n"
                   "// long long long\n"
                   "// lo",
                   getLLVMStyleWithColumns(20)));

  // Reflow lines after a broken line.
  EXPECT_EQ("int a; // Trailing\n"
            "       // comment on\n"
            "       // 2 or 3\n"
            "       // lines.",
            format("int a; // Trailing comment\n"
                   "       // on 2\n"
                   "       // or 3\n"
                   "       // lines.",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/// This long line\n"
            "/// gets reflown.",
            format("/// This long line gets\n"
                   "/// reflown.",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//! This long line\n"
            "//! gets reflown.",
            format(" //! This long line gets\n"
                   " //! reflown.",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* This long line\n"
            " * gets reflown.\n"
            " */",
            format("/* This long line gets\n"
                   " * reflown.\n"
                   " */",
                   getLLVMStyleWithColumns(20)));

  // Reflow after indentation makes a line too long.
  EXPECT_EQ("{\n"
            "  // long long long\n"
            "  // lo long\n"
            "}",
            format("{\n"
                   "// long long long lo\n"
                   "// long\n"
                   "}",
                   getLLVMStyleWithColumns(20)));

  // Break and reflow multiple lines.
  EXPECT_EQ("/*\n"
            " * Reflow the end of\n"
            " * line by 11 22 33\n"
            " * 4.\n"
            " */",
            format("/*\n"
                   " * Reflow the end of line\n"
                   " * by\n"
                   " * 11\n"
                   " * 22\n"
                   " * 33\n"
                   " * 4.\n"
                   " */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/// First line gets\n"
            "/// broken. Second\n"
            "/// line gets\n"
            "/// reflown and\n"
            "/// broken. Third\n"
            "/// gets reflown.",
            format("/// First line gets broken.\n"
                   "/// Second line gets reflown and broken.\n"
                   "/// Third gets reflown.",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("int i; // first long\n"
            "       // long snd\n"
            "       // long.",
            format("int i; // first long long\n"
                   "       // snd long.",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("{\n"
            "  // first long line\n"
            "  // line second\n"
            "  // long line line\n"
            "  // third long line\n"
            "  // line\n"
            "}",
            format("{\n"
                   "  // first long line line\n"
                   "  // second long line line\n"
                   "  // third long line line\n"
                   "}",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("int i; /* first line\n"
            "        * second\n"
            "        * line third\n"
            "        * line\n"
            "        */",
            format("int i; /* first line\n"
                   "        * second line\n"
                   "        * third line\n"
                   "        */",
                   getLLVMStyleWithColumns(20)));

  // Reflow the last two lines of a section that starts with a line having
  // different indentation.
  EXPECT_EQ("//     long\n"
            "// long long long\n"
            "// long long",
            format("//     long\n"
                   "// long long long long\n"
                   "// long",
                   getLLVMStyleWithColumns(20)));

  // Keep the block comment endling '*/' while reflowing.
  EXPECT_EQ("/* Long long long\n"
            " * line short */",
            format("/* Long long long line\n"
                   " * short */",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow between separate blocks of comments.
  EXPECT_EQ("/* First comment\n"
            " * block will */\n"
            "/* Snd\n"
            " */",
            format("/* First comment block\n"
                   " * will */\n"
                   "/* Snd\n"
                   " */",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow across blank comment lines.
  EXPECT_EQ("int i; // This long\n"
            "       // line gets\n"
            "       // broken.\n"
            "       //\n"
            "       // keep.",
            format("int i; // This long line gets broken.\n"
                   "       //  \n"
                   "       // keep.",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("{\n"
            "  /// long long long\n"
            "  /// long long\n"
            "  ///\n"
            "  /// long\n"
            "}",
            format("{\n"
                   "  /// long long long long\n"
                   "  /// long\n"
                   "  ///\n"
                   "  /// long\n"
                   "}",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//! long long long\n"
            "//! long\n"
            "\n"
            "//! long",
            format("//! long long long long\n"
                   "\n"
                   "//! long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* long long long\n"
            "   long\n"
            "\n"
            "   long */",
            format("/* long long long long\n"
                   "\n"
                   "   long */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* long long long\n"
            " * long\n"
            " *\n"
            " * long */",
            format("/* long long long long\n"
                   " *\n"
                   " * long */",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow lines having content that is a single character.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// l",
            format("// long long long long\n"
                   "// l",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow lines starting with two punctuation characters.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// ... --- ...",
            format("// long long long long\n"
                   "// ... --- ...",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow lines starting with '@'.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// @param arg",
            format("// long long long long\n"
                   "// @param arg",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow lines starting with '\'.
  verifyFormat("// long long long\n"
               "// long\n"
               "// \\param arg",
               "// long long long long\n"
               "// \\param arg",
               getLLVMStyleWithColumns(20));

  // Don't reflow lines starting with 'TODO'.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// TODO: long",
            format("// long long long long\n"
                   "// TODO: long",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow lines starting with 'FIXME'.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// FIXME: long",
            format("// long long long long\n"
                   "// FIXME: long",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow lines starting with 'XXX'.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// XXX: long",
            format("// long long long long\n"
                   "// XXX: long",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow comment pragmas.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// IWYU pragma:",
            format("// long long long long\n"
                   "// IWYU pragma:",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* long long long\n"
            " * long\n"
            " * IWYU pragma:\n"
            " */",
            format("/* long long long long\n"
                   " * IWYU pragma:\n"
                   " */",
                   getLLVMStyleWithColumns(20)));

  // Reflow lines that have a non-punctuation character among their first 2
  // characters.
  EXPECT_EQ("// long long long\n"
            "// long 'long'",
            format("// long long long long\n"
                   "// 'long'",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow between separate blocks of comments.
  EXPECT_EQ("/* First comment\n"
            " * block will */\n"
            "/* Snd\n"
            " */",
            format("/* First comment block\n"
                   " * will */\n"
                   "/* Snd\n"
                   " */",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow lines having different indentation.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "//  long",
            format("// long long long long\n"
                   "//  long",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow separate bullets in list
  EXPECT_EQ("// - long long long\n"
            "// long\n"
            "// - long",
            format("// - long long long long\n"
                   "// - long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// * long long long\n"
            "// long\n"
            "// * long",
            format("// * long long long long\n"
                   "// * long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// + long long long\n"
            "// long\n"
            "// + long",
            format("// + long long long long\n"
                   "// + long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// 1. long long long\n"
            "// long\n"
            "// 2. long",
            format("// 1. long long long long\n"
                   "// 2. long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// -# long long long\n"
            "// long\n"
            "// -# long",
            format("// -# long long long long\n"
                   "// -# long",
                   getLLVMStyleWithColumns(20)));

  EXPECT_EQ("// - long long long\n"
            "// long long long\n"
            "// - long",
            format("// - long long long long\n"
                   "// long long\n"
                   "// - long",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// - long long long\n"
            "// long long long\n"
            "// long\n"
            "// - long",
            format("// - long long long long\n"
                   "// long long long\n"
                   "// - long",
                   getLLVMStyleWithColumns(20)));

  // Large number (>2 digits) are not list items
  EXPECT_EQ("// long long long\n"
            "// long 1024. long.",
            format("// long long long long\n"
                   "// 1024. long.",
                   getLLVMStyleWithColumns(20)));

  // Do not break before number, to avoid introducing a non-reflowable doxygen
  // list item.
  EXPECT_EQ("// long long\n"
            "// long 10. long.",
            format("// long long long 10.\n"
                   "// long.",
                   getLLVMStyleWithColumns(20)));

  // Don't break or reflow after implicit string literals.
  verifyFormat("#include <t> // l l l\n"
               "             // l",
               getLLVMStyleWithColumns(20));

  // Don't break or reflow comments on import lines.
  EXPECT_EQ("#include \"t\" /* l l l\n"
            "                * l */",
            format("#include \"t\" /* l l l\n"
                   "                * l */",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow between different trailing comment sections.
  EXPECT_EQ("int i; // long long\n"
            "       // long\n"
            "int j; // long long\n"
            "       // long",
            format("int i; // long long long\n"
                   "int j; // long long long",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow if the first word on the next line is longer than the
  // available space at current line.
  EXPECT_EQ("int i; // trigger\n"
            "       // reflow\n"
            "       // longsec",
            format("int i; // trigger reflow\n"
                   "       // longsec",
                   getLLVMStyleWithColumns(20)));

  // Simple case that correctly handles reflow in parameter lists.
  EXPECT_EQ("a = f(/* looooooooong\n"
            "       * long long\n"
            "       */\n"
            "      a);",
            format("a = f(/* looooooooong long\n* long\n*/ a);",
                   getLLVMStyleWithColumns(22)));
  // Tricky case that has fewer lines if we reflow the comment, ending up with
  // fewer lines.
  EXPECT_EQ("a = f(/* loooooong\n"
            "       * long long\n"
            "       */\n"
            "      a);",
            format("a = f(/* loooooong long\n* long\n*/ a);",
                   getLLVMStyleWithColumns(22)));

  // Keep empty comment lines.
  EXPECT_EQ("/**/", format(" /**/", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* */", format(" /* */", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/*  */", format(" /*  */", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//", format(" //  ", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("///", format(" ///  ", getLLVMStyleWithColumns(20)));
}

TEST_F(FormatTestComments, ReflowsCommentsPrecise) {
  // FIXME: This assumes we do not continue compressing whitespace once we are
  // in reflow mode. Consider compressing whitespace.

  // Test that we stop reflowing precisely at the column limit.
  // After reflowing, "// reflows into   foo" does not fit the column limit,
  // so we compress the whitespace.
  EXPECT_EQ("// some text that\n"
            "// reflows into foo",
            format("// some text that reflows\n"
                   "// into   foo",
                   getLLVMStyleWithColumns(20)));
  // Given one more column, "// reflows into   foo" does fit the limit, so we
  // do not compress the whitespace.
  EXPECT_EQ("// some text that\n"
            "// reflows into   foo",
            format("// some text that reflows\n"
                   "// into   foo",
                   getLLVMStyleWithColumns(21)));

  // Make sure that we correctly account for the space added in the reflow case
  // when making the reflowing decision.
  // First, when the next line ends precisely one column over the limit, do not
  // reflow.
  EXPECT_EQ("// some text that\n"
            "// reflows\n"
            "// into1234567",
            format("// some text that reflows\n"
                   "// into1234567",
                   getLLVMStyleWithColumns(21)));
  // Secondly, when the next line ends later, but the first word in that line
  // is precisely one column over the limit, do not reflow.
  EXPECT_EQ("// some text that\n"
            "// reflows\n"
            "// into1234567 f",
            format("// some text that reflows\n"
                   "// into1234567 f",
                   getLLVMStyleWithColumns(21)));
}

TEST_F(FormatTestComments, ReflowsCommentsWithExtraWhitespace) {
  // Baseline.
  EXPECT_EQ("// some text\n"
            "// that re flows",
            format("// some text that\n"
                   "// re flows",
                   getLLVMStyleWithColumns(16)));
  EXPECT_EQ("// some text\n"
            "// that re flows",
            format("// some text that\n"
                   "// re    flows",
                   getLLVMStyleWithColumns(16)));
  EXPECT_EQ("/* some text\n"
            " * that re flows\n"
            " */",
            format("/* some text that\n"
                   "*      re       flows\n"
                   "*/",
                   getLLVMStyleWithColumns(16)));
  // FIXME: We do not reflow if the indent of two subsequent lines differs;
  // given that this is different behavior from block comments, do we want
  // to keep this?
  EXPECT_EQ("// some text\n"
            "// that\n"
            "//     re flows",
            format("// some text that\n"
                   "//     re       flows",
                   getLLVMStyleWithColumns(16)));
  // Space within parts of a line that fit.
  // FIXME: Use the earliest possible split while reflowing to compress the
  // whitespace within the line.
  EXPECT_EQ("// some text that\n"
            "// does re   flow\n"
            "// more  here",
            format("// some text that does\n"
                   "// re   flow  more  here",
                   getLLVMStyleWithColumns(21)));
}

TEST_F(FormatTestComments, IgnoresIf0Contents) {
  EXPECT_EQ("#if 0\n"
            "}{)(&*(^%%#%@! fsadj f;ldjs ,:;| <<<>>>][)(][\n"
            "#endif\n"
            "void f() {}",
            format("#if 0\n"
                   "}{)(&*(^%%#%@! fsadj f;ldjs ,:;| <<<>>>][)(][\n"
                   "#endif\n"
                   "void f(  ) {  }"));
  EXPECT_EQ("#if false\n"
            "void f(  ) {  }\n"
            "#endif\n"
            "void g() {}",
            format("#if false\n"
                   "void f(  ) {  }\n"
                   "#endif\n"
                   "void g(  ) {  }"));
  EXPECT_EQ("enum E {\n"
            "  One,\n"
            "  Two,\n"
            "#if 0\n"
            "Three,\n"
            "      Four,\n"
            "#endif\n"
            "  Five\n"
            "};",
            format("enum E {\n"
                   "  One,Two,\n"
                   "#if 0\n"
                   "Three,\n"
                   "      Four,\n"
                   "#endif\n"
                   "  Five};"));
  EXPECT_EQ("enum F {\n"
            "  One,\n"
            "#if 1\n"
            "  Two,\n"
            "#if 0\n"
            "Three,\n"
            "      Four,\n"
            "#endif\n"
            "  Five\n"
            "#endif\n"
            "};",
            format("enum F {\n"
                   "One,\n"
                   "#if 1\n"
                   "Two,\n"
                   "#if 0\n"
                   "Three,\n"
                   "      Four,\n"
                   "#endif\n"
                   "Five\n"
                   "#endif\n"
                   "};"));
  EXPECT_EQ("enum G {\n"
            "  One,\n"
            "#if 0\n"
            "Two,\n"
            "#else\n"
            "  Three,\n"
            "#endif\n"
            "  Four\n"
            "};",
            format("enum G {\n"
                   "One,\n"
                   "#if 0\n"
                   "Two,\n"
                   "#else\n"
                   "Three,\n"
                   "#endif\n"
                   "Four\n"
                   "};"));
  EXPECT_EQ("enum H {\n"
            "  One,\n"
            "#if 0\n"
            "#ifdef Q\n"
            "Two,\n"
            "#else\n"
            "Three,\n"
            "#endif\n"
            "#endif\n"
            "  Four\n"
            "};",
            format("enum H {\n"
                   "One,\n"
                   "#if 0\n"
                   "#ifdef Q\n"
                   "Two,\n"
                   "#else\n"
                   "Three,\n"
                   "#endif\n"
                   "#endif\n"
                   "Four\n"
                   "};"));
  EXPECT_EQ("enum I {\n"
            "  One,\n"
            "#if /* test */ 0 || 1\n"
            "Two,\n"
            "Three,\n"
            "#endif\n"
            "  Four\n"
            "};",
            format("enum I {\n"
                   "One,\n"
                   "#if /* test */ 0 || 1\n"
                   "Two,\n"
                   "Three,\n"
                   "#endif\n"
                   "Four\n"
                   "};"));
  EXPECT_EQ("enum J {\n"
            "  One,\n"
            "#if 0\n"
            "#if 0\n"
            "Two,\n"
            "#else\n"
            "Three,\n"
            "#endif\n"
            "Four,\n"
            "#endif\n"
            "  Five\n"
            "};",
            format("enum J {\n"
                   "One,\n"
                   "#if 0\n"
                   "#if 0\n"
                   "Two,\n"
                   "#else\n"
                   "Three,\n"
                   "#endif\n"
                   "Four,\n"
                   "#endif\n"
                   "Five\n"
                   "};"));

  // Ignore stuff in SWIG-blocks.
  EXPECT_EQ("#ifdef SWIG\n"
            "}{)(&*(^%%#%@! fsadj f;ldjs ,:;| <<<>>>][)(][\n"
            "#endif\n"
            "void f() {}",
            format("#ifdef SWIG\n"
                   "}{)(&*(^%%#%@! fsadj f;ldjs ,:;| <<<>>>][)(][\n"
                   "#endif\n"
                   "void f(  ) {  }"));
  EXPECT_EQ("#ifndef SWIG\n"
            "void f() {}\n"
            "#endif",
            format("#ifndef SWIG\n"
                   "void f(      ) {       }\n"
                   "#endif"));
}

TEST_F(FormatTestComments, DontCrashOnBlockComments) {
  EXPECT_EQ(
      "int xxxxxxxxx; /* "
      "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\n"
      "zzzzzz\n"
      "0*/",
      format("int xxxxxxxxx;                          /* "
             "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy zzzzzz\n"
             "0*/"));
}

TEST_F(FormatTestComments, BlockCommentsInControlLoops) {
  verifyFormat("if (0) /* a comment in a strange place */ {\n"
               "  f();\n"
               "}");
  verifyFormat("if (0) /* a comment in a strange place */ {\n"
               "  f();\n"
               "} /* another comment */ else /* comment #3 */ {\n"
               "  g();\n"
               "}");
  verifyFormat("while (0) /* a comment in a strange place */ {\n"
               "  f();\n"
               "}");
  verifyFormat("for (;;) /* a comment in a strange place */ {\n"
               "  f();\n"
               "}");
  verifyFormat("do /* a comment in a strange place */ {\n"
               "  f();\n"
               "} /* another comment */ while (0);");
}

TEST_F(FormatTestComments, BlockComments) {
  EXPECT_EQ("/* */ /* */ /* */\n/* */ /* */ /* */",
            format("/* *//* */  /* */\n/* *//* */  /* */"));
  EXPECT_EQ("/* */ a /* */ b;", format("  /* */  a/* */  b;"));
  EXPECT_EQ("#define A /*123*/ \\\n"
            "  b\n"
            "/* */\n"
            "someCall(\n"
            "    parameter);",
            format("#define A /*123*/ b\n"
                   "/* */\n"
                   "someCall(parameter);",
                   getLLVMStyleWithColumns(15)));

  EXPECT_EQ("#define A\n"
            "/* */ someCall(\n"
            "    parameter);",
            format("#define A\n"
                   "/* */someCall(parameter);",
                   getLLVMStyleWithColumns(15)));
  verifyNoChange("/*\n**\n*/");
  EXPECT_EQ("/*\n"
            " *\n"
            " * aaaaaa\n"
            " * aaaaaa\n"
            " */",
            format("/*\n"
                   "*\n"
                   " * aaaaaa aaaaaa\n"
                   "*/",
                   getLLVMStyleWithColumns(10)));
  EXPECT_EQ("/*\n"
            "**\n"
            "* aaaaaa\n"
            "*aaaaaa\n"
            "*/",
            format("/*\n"
                   "**\n"
                   "* aaaaaa aaaaaa\n"
                   "*/",
                   getLLVMStyleWithColumns(10)));
  EXPECT_EQ("int aaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
            "    /* line 1\n"
            "       bbbbbbbbbbbb */\n"
            "    bbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
            format("int aaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
                   "    /* line 1\n"
                   "       bbbbbbbbbbbb */ bbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
                   getLLVMStyleWithColumns(50)));

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = FormatStyle::BPPS_OnePerLine;
  EXPECT_EQ("someFunction(1, /* comment 1 */\n"
            "             2, /* comment 2 */\n"
            "             3, /* comment 3 */\n"
            "             aaaa,\n"
            "             bbbb);",
            format("someFunction (1,   /* comment 1 */\n"
                   "                2, /* comment 2 */  \n"
                   "               3,   /* comment 3 */\n"
                   "aaaa, bbbb );",
                   NoBinPacking));
  verifyFormat(
      "bool aaaaaaaaaaaaa = /* comment: */ aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  EXPECT_EQ(
      "bool aaaaaaaaaaaaa = /* trailing comment */\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaaa;",
      format(
          "bool       aaaaaaaaaaaaa =       /* trailing comment */\n"
          "    aaaaaaaaaaaaaaaaaaaaaaaaaaa||aaaaaaaaaaaaaaaaaaaaaaaaa    ||\n"
          "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa   || aaaaaaaaaaaaaaaaaaaaaaaaaa;"));
  EXPECT_EQ(
      "int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; /* comment */\n"
      "int bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;   /* comment */\n"
      "int cccccccccccccccccccccccccccccc;       /* comment */",
      format("int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; /* comment */\n"
             "int      bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; /* comment */\n"
             "int    cccccccccccccccccccccccccccccc;  /* comment */"));

  verifyFormat("void f(int * /* unused */) {}");

  EXPECT_EQ("/*\n"
            " **\n"
            " */",
            format("/*\n"
                   " **\n"
                   " */"));
  EXPECT_EQ("/*\n"
            " *q\n"
            " */",
            format("/*\n"
                   " *q\n"
                   " */"));
  EXPECT_EQ("/*\n"
            " * q\n"
            " */",
            format("/*\n"
                   " * q\n"
                   " */"));
  EXPECT_EQ("/*\n"
            " **/",
            format("/*\n"
                   " **/"));
  EXPECT_EQ("/*\n"
            " ***/",
            format("/*\n"
                   " ***/"));
}

TEST_F(FormatTestComments, BlockCommentsInMacros) {
  EXPECT_EQ("#define A          \\\n"
            "  {                \\\n"
            "    /* one line */ \\\n"
            "    someCall();",
            format("#define A {        \\\n"
                   "  /* one line */   \\\n"
                   "  someCall();",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("#define A          \\\n"
            "  {                \\\n"
            "    /* previous */ \\\n"
            "    /* one line */ \\\n"
            "    someCall();",
            format("#define A {        \\\n"
                   "  /* previous */   \\\n"
                   "  /* one line */   \\\n"
                   "  someCall();",
                   getLLVMStyleWithColumns(20)));
}

TEST_F(FormatTestComments, BlockCommentsAtEndOfLine) {
  EXPECT_EQ("a = {\n"
            "    1111 /*    */\n"
            "};",
            format("a = {1111 /*    */\n"
                   "};",
                   getLLVMStyleWithColumns(15)));
  EXPECT_EQ("a = {\n"
            "    1111 /*      */\n"
            "};",
            format("a = {1111 /*      */\n"
                   "};",
                   getLLVMStyleWithColumns(15)));
  EXPECT_EQ("a = {\n"
            "    1111 /*      a\n"
            "          */\n"
            "};",
            format("a = {1111 /*      a */\n"
                   "};",
                   getLLVMStyleWithColumns(15)));
}

TEST_F(FormatTestComments, BreaksAfterMultilineBlockCommentsInParamLists) {
  EXPECT_EQ("a = f(/* long\n"
            "         long */\n"
            "      a);",
            format("a = f(/* long long */ a);", getLLVMStyleWithColumns(16)));
  EXPECT_EQ("a = f(\n"
            "    /* long\n"
            "       long */\n"
            "    a);",
            format("a = f(/* long long */ a);", getLLVMStyleWithColumns(15)));

  EXPECT_EQ("a = f(/* long\n"
            "         long\n"
            "       */\n"
            "      a);",
            format("a = f(/* long\n"
                   "         long\n"
                   "       */a);",
                   getLLVMStyleWithColumns(16)));

  EXPECT_EQ("a = f(/* long\n"
            "         long\n"
            "       */\n"
            "      a);",
            format("a = f(/* long\n"
                   "         long\n"
                   "       */ a);",
                   getLLVMStyleWithColumns(16)));

  EXPECT_EQ("a = f(/* long\n"
            "         long\n"
            "       */\n"
            "      (1 + 1));",
            format("a = f(/* long\n"
                   "         long\n"
                   "       */ (1 + 1));",
                   getLLVMStyleWithColumns(16)));

  EXPECT_EQ(
      "a = f(a,\n"
      "      /* long\n"
      "         long */\n"
      "      b);",
      format("a = f(a, /* long long */ b);", getLLVMStyleWithColumns(16)));

  EXPECT_EQ(
      "a = f(\n"
      "    a,\n"
      "    /* long\n"
      "       long */\n"
      "    b);",
      format("a = f(a, /* long long */ b);", getLLVMStyleWithColumns(15)));

  EXPECT_EQ("a = f(a,\n"
            "      /* long\n"
            "         long */\n"
            "      (1 + 1));",
            format("a = f(a, /* long long */ (1 + 1));",
                   getLLVMStyleWithColumns(16)));
  EXPECT_EQ("a = f(\n"
            "    a,\n"
            "    /* long\n"
            "       long */\n"
            "    (1 + 1));",
            format("a = f(a, /* long long */ (1 + 1));",
                   getLLVMStyleWithColumns(15)));
}

TEST_F(FormatTestComments, IndentLineCommentsInStartOfBlockAtEndOfFile) {
  verifyFormat("{\n"
               "  // a\n"
               "  // b");
}

TEST_F(FormatTestComments, AlignTrailingComments) {
  EXPECT_EQ("#define MACRO(V)                       \\\n"
            "  V(Rt2) /* one more char */           \\\n"
            "  V(Rs)  /* than here  */              \\\n"
            "/* comment 3 */\n",
            format("#define MACRO(V)\\\n"
                   "V(Rt2)  /* one more char */ \\\n"
                   "V(Rs) /* than here  */    \\\n"
                   "/* comment 3 */\n",
                   getLLVMStyleWithColumns(40)));
  EXPECT_EQ("int i = f(abc, // line 1\n"
            "          d,   // line 2\n"
            "               // line 3\n"
            "          b);",
            format("int i = f(abc, // line 1\n"
                   "          d, // line 2\n"
                   "             // line 3\n"
                   "          b);",
                   getLLVMStyleWithColumns(40)));

  // Align newly broken trailing comments.
  EXPECT_EQ("int ab; // line\n"
            "int a;  // long\n"
            "        // long",
            format("int ab; // line\n"
                   "int a; // long long",
                   getLLVMStyleWithColumns(15)));
  EXPECT_EQ("int ab; // line\n"
            "int a;  // long\n"
            "        // long\n"
            "        // long",
            format("int ab; // line\n"
                   "int a; // long long\n"
                   "       // long",
                   getLLVMStyleWithColumns(15)));
  EXPECT_EQ("int ab; // line\n"
            "int a;  // long\n"
            "        // long\n"
            "pt c;   // long",
            format("int ab; // line\n"
                   "int a; // long long\n"
                   "pt c; // long",
                   getLLVMStyleWithColumns(15)));
  EXPECT_EQ("int ab; // line\n"
            "int a;  // long\n"
            "        // long\n"
            "\n"
            "// long",
            format("int ab; // line\n"
                   "int a; // long long\n"
                   "\n"
                   "// long",
                   getLLVMStyleWithColumns(15)));

  // Don't align newly broken trailing comments if that would put them over the
  // column limit.
  EXPECT_EQ("int i, j; // line 1\n"
            "int k; // line longg\n"
            "       // long",
            format("int i, j; // line 1\n"
                   "int k; // line longg long",
                   getLLVMStyleWithColumns(20)));

  // Always align if ColumnLimit = 0
  EXPECT_EQ("int i, j; // line 1\n"
            "int k;    // line longg long",
            format("int i, j; // line 1\n"
                   "int k; // line longg long",
                   getLLVMStyleWithColumns(0)));

  // Align comment line sections aligned with the next token with the next
  // token.
  EXPECT_EQ("class A {\n"
            "public: // public comment\n"
            "  // comment about a\n"
            "  int a;\n"
            "};",
            format("class A {\n"
                   "public: // public comment\n"
                   "  // comment about a\n"
                   "  int a;\n"
                   "};",
                   getLLVMStyleWithColumns(40)));
  EXPECT_EQ("class A {\n"
            "public: // public comment 1\n"
            "        // public comment 2\n"
            "  // comment 1 about a\n"
            "  // comment 2 about a\n"
            "  int a;\n"
            "};",
            format("class A {\n"
                   "public: // public comment 1\n"
                   "   // public comment 2\n"
                   "  // comment 1 about a\n"
                   "  // comment 2 about a\n"
                   "  int a;\n"
                   "};",
                   getLLVMStyleWithColumns(40)));
  EXPECT_EQ("int f(int n) { // comment line 1 on f\n"
            "               // comment line 2 on f\n"
            "  // comment line 1 before return\n"
            "  // comment line 2 before return\n"
            "  return n; // comment line 1 on return\n"
            "            // comment line 2 on return\n"
            "  // comment line 1 after return\n"
            "}",
            format("int f(int n) { // comment line 1 on f\n"
                   "   // comment line 2 on f\n"
                   "  // comment line 1 before return\n"
                   "  // comment line 2 before return\n"
                   "  return n; // comment line 1 on return\n"
                   "   // comment line 2 on return\n"
                   "  // comment line 1 after return\n"
                   "}",
                   getLLVMStyleWithColumns(40)));
  EXPECT_EQ("int f(int n) {\n"
            "  switch (n) { // comment line 1 on switch\n"
            "               // comment line 2 on switch\n"
            "  // comment line 1 before case 1\n"
            "  // comment line 2 before case 1\n"
            "  case 1: // comment line 1 on case 1\n"
            "          // comment line 2 on case 1\n"
            "    // comment line 1 before return 1\n"
            "    // comment line 2 before return 1\n"
            "    return 1; // comment line 1 on return 1\n"
            "              // comment line 2 on return 1\n"
            "  // comment line 1 before default\n"
            "  // comment line 2 before default\n"
            "  default: // comment line 1 on default\n"
            "           // comment line 2 on default\n"
            "    // comment line 1 before return 2\n"
            "    return 2 * f(n - 1); // comment line 1 on return 2\n"
            "                         // comment line 2 on return 2\n"
            "    // comment line 1 after return\n"
            "    // comment line 2 after return\n"
            "  }\n"
            "}",
            format("int f(int n) {\n"
                   "  switch (n) { // comment line 1 on switch\n"
                   "              // comment line 2 on switch\n"
                   "    // comment line 1 before case 1\n"
                   "    // comment line 2 before case 1\n"
                   "    case 1: // comment line 1 on case 1\n"
                   "              // comment line 2 on case 1\n"
                   "    // comment line 1 before return 1\n"
                   "    // comment line 2 before return 1\n"
                   "    return 1;  // comment line 1 on return 1\n"
                   "             // comment line 2 on return 1\n"
                   "    // comment line 1 before default\n"
                   "    // comment line 2 before default\n"
                   "    default:   // comment line 1 on default\n"
                   "                // comment line 2 on default\n"
                   "    // comment line 1 before return 2\n"
                   "    return 2 * f(n - 1); // comment line 1 on return 2\n"
                   "                        // comment line 2 on return 2\n"
                   "    // comment line 1 after return\n"
                   "     // comment line 2 after return\n"
                   "  }\n"
                   "}",
                   getLLVMStyleWithColumns(80)));

  // If all the lines in a sequence of line comments are aligned with the next
  // token, the first line belongs to the previous token and the other lines
  // belong to the next token.
  EXPECT_EQ("int a; // line about a\n"
            "long b;",
            format("int a; // line about a\n"
                   "       long b;",
                   getLLVMStyleWithColumns(80)));
  EXPECT_EQ("int a; // line about a\n"
            "// line about b\n"
            "long b;",
            format("int a; // line about a\n"
                   "       // line about b\n"
                   "       long b;",
                   getLLVMStyleWithColumns(80)));
  EXPECT_EQ("int a; // line about a\n"
            "// line 1 about b\n"
            "// line 2 about b\n"
            "long b;",
            format("int a; // line about a\n"
                   "       // line 1 about b\n"
                   "       // line 2 about b\n"
                   "       long b;",
                   getLLVMStyleWithColumns(80)));

  // Checks an edge case in preprocessor handling.
  // These comments should *not* be aligned
  EXPECT_EQ(
      "#if FOO\n"
      "#else\n"
      "long a; // Line about a\n"
      "#endif\n"
      "#if BAR\n"
      "#else\n"
      "long b_long_name; // Line about b\n"
      "#endif",
      format("#if FOO\n"
             "#else\n"
             "long a;           // Line about a\n" // Previous (bad) behavior
             "#endif\n"
             "#if BAR\n"
             "#else\n"
             "long b_long_name; // Line about b\n"
             "#endif",
             getLLVMStyleWithColumns(80)));

  // bug 47589
  EXPECT_EQ(
      "namespace m {\n\n"
      "#define FOO_GLOBAL 0      // Global scope.\n"
      "#define FOO_LINKLOCAL 1   // Link-local scope.\n"
      "#define FOO_SITELOCAL 2   // Site-local scope (deprecated).\n"
      "#define FOO_UNIQUELOCAL 3 // Unique local\n"
      "#define FOO_NODELOCAL 4   // Loopback\n\n"
      "} // namespace m",
      format("namespace m {\n\n"
             "#define FOO_GLOBAL 0   // Global scope.\n"
             "#define FOO_LINKLOCAL 1  // Link-local scope.\n"
             "#define FOO_SITELOCAL 2  // Site-local scope (deprecated).\n"
             "#define FOO_UNIQUELOCAL 3 // Unique local\n"
             "#define FOO_NODELOCAL 4  // Loopback\n\n"
             "} // namespace m",
             getLLVMStyleWithColumns(80)));

  // https://llvm.org/PR53441
  verifyFormat("/* */  //\n"
               "int a; //");
  verifyFormat("/**/   //\n"
               "int a; //");
}

TEST_F(FormatTestComments, AlignTrailingCommentsAcrossEmptyLines) {
  FormatStyle Style = getLLVMStyle();
  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Always;
  Style.AlignTrailingComments.OverEmptyLines = 1;
  verifyFormat("#include \"a.h\"  // simple\n"
               "\n"
               "#include \"aa.h\" // example case",
               Style);

  verifyFormat("#include \"a.h\"   // align across\n"
               "\n"
               "#include \"aa.h\"  // two empty lines\n"
               "\n"
               "#include \"aaa.h\" // in a row",
               Style);

  verifyFormat("#include \"a.h\"      // align\n"
               "#include \"aa.h\"     // comment\n"
               "#include \"aaa.h\"    // blocks\n"
               "\n"
               "#include \"aaaa.h\"   // across\n"
               "#include \"aaaaa.h\"  // one\n"
               "#include \"aaaaaa.h\" // empty line",
               Style);

  verifyFormat("#include \"a.h\"  // align trailing comments\n"
               "#include \"a.h\"\n"
               "#include \"aa.h\" // across a line without comment",
               Style);

  verifyFormat("#include \"a.h\"   // align across\n"
               "#include \"a.h\"\n"
               "#include \"aa.h\"  // two lines without comment\n"
               "#include \"a.h\"\n"
               "#include \"aaa.h\" // in a row",
               Style);

  verifyFormat("#include \"a.h\"      // align\n"
               "#include \"aa.h\"     // comment\n"
               "#include \"aaa.h\"    // blocks\n"
               "#include \"a.h\"\n"
               "#include \"aaaa.h\"   // across\n"
               "#include \"aaaaa.h\"  // a line without\n"
               "#include \"aaaaaa.h\" // comment",
               Style);

  // Start of testing OverEmptyLines
  Style.MaxEmptyLinesToKeep = 3;
  Style.AlignTrailingComments.OverEmptyLines = 2;
  // Cannot use verifyFormat here
  // test::messUp removes all new lines which changes the logic
  EXPECT_EQ("#include \"a.h\" // comment\n"
            "\n"
            "\n"
            "\n"
            "#include \"ab.h\"      // comment\n"
            "\n"
            "\n"
            "#include \"abcdefg.h\" // comment",
            format("#include \"a.h\" // comment\n"
                   "\n"
                   "\n"
                   "\n"
                   "#include \"ab.h\" // comment\n"
                   "\n"
                   "\n"
                   "#include \"abcdefg.h\" // comment",
                   Style));

  Style.MaxEmptyLinesToKeep = 1;
  Style.AlignTrailingComments.OverEmptyLines = 1;
  // End of testing OverEmptyLines

  Style.ColumnLimit = 15;
  EXPECT_EQ("int ab; // line\n"
            "int a;  // long\n"
            "        // long\n"
            "\n"
            "        // long",
            format("int ab; // line\n"
                   "int a; // long long\n"
                   "\n"
                   "// long",
                   Style));

  Style.ColumnLimit = 15;
  EXPECT_EQ("int ab; // line\n"
            "\n"
            "int a;  // long\n"
            "        // long",
            format("int ab; // line\n"
                   "\n"
                   "int a; // long long",
                   Style));

  Style.ColumnLimit = 30;
  EXPECT_EQ("int foo = 12345; // comment\n"
            "int bar =\n"
            "    1234;  // This is a very\n"
            "           // long comment\n"
            "           // which is wrapped\n"
            "           // arround.\n"
            "\n"
            "int x = 2; // Is this still\n"
            "           // aligned?",
            format("int foo = 12345; // comment\n"
                   "int bar = 1234; // This is a very long comment\n"
                   "                // which is wrapped arround.\n"
                   "\n"
                   "int x = 2; // Is this still aligned?",
                   Style));

  Style.ColumnLimit = 35;
  EXPECT_EQ("int foo = 12345; // comment\n"
            "int bar =\n"
            "    1234; // This is a very long\n"
            "          // comment which is\n"
            "          // wrapped arround.\n"
            "\n"
            "int x =\n"
            "    2; // Is this still aligned?",
            format("int foo = 12345; // comment\n"
                   "int bar = 1234; // This is a very long comment\n"
                   "                // which is wrapped arround.\n"
                   "\n"
                   "int x = 2; // Is this still aligned?",
                   Style));

  Style.ColumnLimit = 40;
  EXPECT_EQ("int foo = 12345; // comment\n"
            "int bar =\n"
            "    1234; // This is a very long comment\n"
            "          // which is wrapped arround.\n"
            "\n"
            "int x = 2; // Is this still aligned?",
            format("int foo = 12345; // comment\n"
                   "int bar = 1234; // This is a very long comment\n"
                   "                // which is wrapped arround.\n"
                   "\n"
                   "int x = 2; // Is this still aligned?",
                   Style));

  Style.ColumnLimit = 45;
  EXPECT_EQ("int foo = 12345; // comment\n"
            "int bar =\n"
            "    1234;  // This is a very long comment\n"
            "           // which is wrapped arround.\n"
            "\n"
            "int x = 2; // Is this still aligned?",
            format("int foo = 12345; // comment\n"
                   "int bar = 1234; // This is a very long comment\n"
                   "                // which is wrapped arround.\n"
                   "\n"
                   "int x = 2; // Is this still aligned?",
                   Style));

  Style.ColumnLimit = 80;
  EXPECT_EQ("int a; // line about a\n"
            "\n"
            "// line about b\n"
            "long b;",
            format("int a; // line about a\n"
                   "\n"
                   "       // line about b\n"
                   "       long b;",
                   Style));

  Style.ColumnLimit = 80;
  EXPECT_EQ("int a; // line about a\n"
            "\n"
            "// line 1 about b\n"
            "// line 2 about b\n"
            "long b;",
            format("int a; // line about a\n"
                   "\n"
                   "       // line 1 about b\n"
                   "       // line 2 about b\n"
                   "       long b;",
                   Style));
}

TEST_F(FormatTestComments, AlignTrailingCommentsLeave) {
  FormatStyle Style = getLLVMStyle();
  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Leave;

  EXPECT_EQ("int a;// do not touch\n"
            "int b; // any comments\n"
            "int c;  // comment\n"
            "int d;   // comment",
            format("int a;// do not touch\n"
                   "int b; // any comments\n"
                   "int c;  // comment\n"
                   "int d;   // comment",
                   Style));

  EXPECT_EQ("int a;   // do not touch\n"
            "int b;  // any comments\n"
            "int c; // comment\n"
            "int d;// comment",
            format("int a;   // do not touch\n"
                   "int b;  // any comments\n"
                   "int c; // comment\n"
                   "int d;// comment",
                   Style));

  EXPECT_EQ("// do not touch\n"
            "int a;  // any comments\n"
            "\n"
            "   // comment\n"
            "// comment\n"
            "\n"
            "// comment",
            format("// do not touch\n"
                   "int a;  // any comments\n"
                   "\n"
                   "   // comment\n"
                   "// comment\n"
                   "\n"
                   "// comment",
                   Style));

  EXPECT_EQ("// do not touch\n"
            "int a;  // any comments\n"
            "\n"
            "   // comment\n"
            "// comment\n"
            "\n"
            "// comment",
            format("// do not touch\n"
                   "int a;  // any comments\n"
                   "\n"
                   "\n"
                   "   // comment\n"
                   "// comment\n"
                   "\n"
                   "\n"
                   "// comment",
                   Style));

  verifyFormat("namespace ns {\n"
               "int i;\n"
               "int j;\n"
               "} // namespace ns",
               "namespace ns {\n"
               "int i;\n"
               "int j;\n"
               "}",
               Style);

  Style.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  verifyNoChange("#define FOO    \\\n"
                 "  /* foo(); */ \\\n"
                 "  bar();",
                 Style);

  // Allow to keep 2 empty lines
  Style.MaxEmptyLinesToKeep = 2;
  EXPECT_EQ("// do not touch\n"
            "int a;  // any comments\n"
            "\n"
            "\n"
            "   // comment\n"
            "// comment\n"
            "\n"
            "// comment",
            format("// do not touch\n"
                   "int a;  // any comments\n"
                   "\n"
                   "\n"
                   "   // comment\n"
                   "// comment\n"
                   "\n"
                   "// comment",
                   Style));
  Style.MaxEmptyLinesToKeep = 1;

  // Just format comments normally when leaving exceeds the column limit
  Style.ColumnLimit = 35;
  EXPECT_EQ("int foo = 12345; // comment\n"
            "int bar =\n"
            "    1234; // This is a very long\n"
            "          // comment which is\n"
            "          // wrapped arround.",
            format("int foo = 12345; // comment\n"
                   "int bar = 1234;       // This is a very long comment\n"
                   "          // which is wrapped arround.",
                   Style));

  Style = getLLVMStyle();
  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Leave;
  Style.TabWidth = 2;
  Style.UseTab = FormatStyle::UT_ForIndentation;
  verifyNoChange("{\n"
                 "\t// f\n"
                 "\tf();\n"
                 "\n"
                 "\t// g\n"
                 "\tg();\n"
                 "\t{\n"
                 "\t\t// h();  // h\n"
                 "\t\tfoo();  // foo\n"
                 "\t}\n"
                 "}",
                 Style);
}

TEST_F(FormatTestComments, DontAlignNamespaceComments) {
  FormatStyle Style = getLLVMStyle();
  Style.NamespaceIndentation = FormatStyle::NI_All;
  Style.NamespaceMacros.push_back("TESTSUITE");
  Style.ShortNamespaceLines = 0;

  StringRef Input = "namespace A {\n"
                    "  TESTSUITE(B) {\n"
                    "    namespace C {\n"
                    "      namespace D { //\n"
                    "      } // namespace D\n"
                    "      std::string Foo = Bar; // Comment\n"
                    "      std::string BazString = Baz;   // C2\n"
                    "    }          // namespace C\n"
                    "  }\n"
                    "} // NaMeSpAcE A";

  EXPECT_TRUE(Style.FixNamespaceComments);
  EXPECT_EQ(Style.AlignTrailingComments.Kind, FormatStyle::TCAS_Always);
  verifyFormat("namespace A {\n"
               "  TESTSUITE(B) {\n"
               "    namespace C {\n"
               "      namespace D { //\n"
               "      } // namespace D\n"
               "      std::string Foo = Bar;       // Comment\n"
               "      std::string BazString = Baz; // C2\n"
               "    } // namespace C\n"
               "  } // TESTSUITE(B)\n"
               "} // NaMeSpAcE A",
               Input, Style);

  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Never;
  verifyFormat("namespace A {\n"
               "  TESTSUITE(B) {\n"
               "    namespace C {\n"
               "      namespace D { //\n"
               "      } // namespace D\n"
               "      std::string Foo = Bar; // Comment\n"
               "      std::string BazString = Baz; // C2\n"
               "    } // namespace C\n"
               "  } // TESTSUITE(B)\n"
               "} // NaMeSpAcE A",
               Input, Style);

  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Leave;
  verifyFormat("namespace A {\n"
               "  TESTSUITE(B) {\n"
               "    namespace C {\n"
               "      namespace D { //\n"
               "      } // namespace D\n"
               "      std::string Foo = Bar; // Comment\n"
               "      std::string BazString = Baz;   // C2\n"
               "    }          // namespace C\n"
               "  } // TESTSUITE(B)\n"
               "} // NaMeSpAcE A",
               Input, Style);

  Style.FixNamespaceComments = false;
  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Always;
  verifyFormat("namespace A {\n"
               "  TESTSUITE(B) {\n"
               "    namespace C {\n"
               "      namespace D { //\n"
               "      } // namespace D\n"
               "      std::string Foo = Bar;       // Comment\n"
               "      std::string BazString = Baz; // C2\n"
               "    } // namespace C\n"
               "  }\n"
               "} // NaMeSpAcE A",
               Input, Style);

  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Never;
  verifyFormat("namespace A {\n"
               "  TESTSUITE(B) {\n"
               "    namespace C {\n"
               "      namespace D { //\n"
               "      } // namespace D\n"
               "      std::string Foo = Bar; // Comment\n"
               "      std::string BazString = Baz; // C2\n"
               "    } // namespace C\n"
               "  }\n"
               "} // NaMeSpAcE A",
               Input, Style);

  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Leave;
  verifyFormat("namespace A {\n"
               "  TESTSUITE(B) {\n"
               "    namespace C {\n"
               "      namespace D { //\n"
               "      } // namespace D\n"
               "      std::string Foo = Bar; // Comment\n"
               "      std::string BazString = Baz;   // C2\n"
               "    }          // namespace C\n"
               "  }\n"
               "} // NaMeSpAcE A",
               Input, Style);

  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Always;
  Style.FixNamespaceComments = true;
  Input = "namespace A {\n"
          "  int Foo;\n"
          "  int Bar;\n"
          "}\n"
          "// Comment";

  verifyFormat("namespace A {\n"
               "  int Foo;\n"
               "  int Bar;\n"
               "} // namespace A\n"
               "// Comment",
               Input, Style);

  Style.FixNamespaceComments = false;
  verifyFormat(Input, Style);
}

TEST_F(FormatTestComments, DontAlignOverScope) {
  verifyFormat("if (foo) {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("if (foo) {\n"
               "  // something\n"
               "} else {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("if (foo) {\n"
               "  // something\n"
               "} else if (foo) {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("while (foo) {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("for (;;) {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("do {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} while (foo); // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("do\n"
               "  int aLongVariable; // with comment\n"
               "while (foo); // not aigned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("do\n"
               "  int aLongVariable; // with comment\n"
               "/**/ while (foo); // not aigned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("switch (foo) {\n"
               "case 7: {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} // case not aligned\n"
               "} // switch also not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("switch (foo) {\n"
               "default: {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} // case not aligned\n"
               "} // switch also not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("class C {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "}; // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("struct S {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "}; // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("union U {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "}; // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("enum E {\n"
               "  aLongVariable, // with comment\n"
               "  f              // aligned\n"
               "}; // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");

  verifyFormat("void foo() {\n"
               "  {\n"
               "    int aLongVariable; // with comment\n"
               "    int f;             // aligned\n"
               "  } // not aligned\n"
               "  int bar;    // new align\n"
               "  int foobar; // group\n"
               "}");

  verifyFormat("auto longLambda = [] { // comment\n"
               "  int aLongVariable;   // with comment\n"
               "  int f;               // aligned\n"
               "}; // not aligned\n"
               "int bar;                             // new align\n"
               "int foobar;                          // group\n"
               "auto shortLambda = [] { return 5; }; // aligned");

  verifyFormat("auto longLambdaResult = [] { // comment\n"
               "  int aLongVariable;         // with comment\n"
               "  int f;                     // aligned\n"
               "}(); // not aligned\n"
               "int bar;                               // new align\n"
               "int foobar;                            // group\n"
               "auto shortLambda = [] { return 5; }(); // aligned");

  verifyFormat(
      "auto longLambdaResult = [](auto I, auto J) { // comment\n"
      "  int aLongVariable;                         // with comment\n"
      "  int f;                                     // aligned\n"
      "}(\"Input\", 5); // not aligned\n"
      "int bar;                                                 // new align\n"
      "int foobar;                                              // group\n"
      "auto shortL = [](auto I, auto J) { return 5; }(\"In\", 5); // aligned");

  verifyFormat("enum E1 { V1, V2 };                    // Aligned\n"
               "enum E2 { LongerNames, InThis, Enum }; // Comments");

  verifyFormat("class C {\n"
               "  int aLongVariable; // with comment\n"
               "  int f;             // aligned\n"
               "} /* middle comment */; // not aligned\n"
               "int bar;    // new align\n"
               "int foobar; // group");
}

TEST_F(FormatTestComments, AlignsBlockCommentDecorations) {
  EXPECT_EQ("/*\n"
            " */",
            format("/*\n"
                   "*/"));
  EXPECT_EQ("/*\n"
            " */",
            format("/*\n"
                   " */"));
  EXPECT_EQ("/*\n"
            " */",
            format("/*\n"
                   "  */"));

  // Align a single line.
  EXPECT_EQ("/*\n"
            " * line */",
            format("/*\n"
                   "* line */"));
  EXPECT_EQ("/*\n"
            " * line */",
            format("/*\n"
                   " * line */"));
  EXPECT_EQ("/*\n"
            " * line */",
            format("/*\n"
                   "  * line */"));
  EXPECT_EQ("/*\n"
            " * line */",
            format("/*\n"
                   "   * line */"));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   "* line */"));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   " * line */"));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   "  * line */"));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   "   * line */"));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   "    * line */"));

  // Align the end '*/' after a line.
  EXPECT_EQ("/*\n"
            " * line\n"
            " */",
            format("/*\n"
                   "* line\n"
                   "*/"));
  EXPECT_EQ("/*\n"
            " * line\n"
            " */",
            format("/*\n"
                   "   * line\n"
                   "  */"));
  EXPECT_EQ("/*\n"
            " * line\n"
            " */",
            format("/*\n"
                   "  * line\n"
                   "  */"));

  // Align two lines.
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   " * line 2 */"));
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   "* line 2 */"));
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   "  * line 2 */"));
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   "   * line 2 */"));
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   "    * line 2 */"));
  EXPECT_EQ("int i; /* line 1\n"
            "        * line 2 */",
            format("int i; /* line 1\n"
                   "* line 2 */"));
  EXPECT_EQ("int i; /* line 1\n"
            "        * line 2 */",
            format("int i; /* line 1\n"
                   "        * line 2 */"));
  EXPECT_EQ("int i; /* line 1\n"
            "        * line 2 */",
            format("int i; /* line 1\n"
                   "             * line 2 */"));

  // Align several lines.
  EXPECT_EQ("/* line 1\n"
            " * line 2\n"
            " * line 3 */",
            format("/* line 1\n"
                   " * line 2\n"
                   "* line 3 */"));
  EXPECT_EQ("/* line 1\n"
            " * line 2\n"
            " * line 3 */",
            format("/* line 1\n"
                   "  * line 2\n"
                   "* line 3 */"));
  EXPECT_EQ("/*\n"
            "** line 1\n"
            "** line 2\n"
            "*/",
            format("/*\n"
                   "** line 1\n"
                   " ** line 2\n"
                   "*/"));

  // Align with different indent after the decorations.
  EXPECT_EQ("/*\n"
            " * line 1\n"
            " *  line 2\n"
            " * line 3\n"
            " *   line 4\n"
            " */",
            format("/*\n"
                   "* line 1\n"
                   "  *  line 2\n"
                   "   * line 3\n"
                   "*   line 4\n"
                   "*/"));

  // Align empty or blank lines.
  EXPECT_EQ("/**\n"
            " *\n"
            " *\n"
            " *\n"
            " */",
            format("/**\n"
                   "*  \n"
                   " * \n"
                   "  *\n"
                   "*/"));

  // Align while breaking and reflowing.
  EXPECT_EQ("/*\n"
            " * long long long\n"
            " * long long\n"
            " *\n"
            " * long */",
            format("/*\n"
                   " * long long long long\n"
                   " * long\n"
                   "  *\n"
                   "* long */",
                   getLLVMStyleWithColumns(20)));
}

TEST_F(FormatTestComments, NoCrash_Bug34236) {
  // This is a test case from a crasher reported in:
  // https://bugs.llvm.org/show_bug.cgi?id=34236
  // Temporarily disable formatting for readability.
  // clang-format off
  EXPECT_EQ(
"/*                                                                */ /*\n"
"                                                                      *       a\n"
"                                                                      * b c d*/",
      format(
"/*                                                                */ /*\n"
" *       a b\n"
" *       c     d*/",
          getLLVMStyleWithColumns(80)));
  // clang-format on
}

TEST_F(FormatTestComments, NonTrailingBlockComments) {
  verifyFormat("const /** comment comment */ A = B;",
               getLLVMStyleWithColumns(40));

  verifyFormat("const /** comment comment comment */ A =\n"
               "    B;",
               getLLVMStyleWithColumns(40));

  EXPECT_EQ("const /** comment comment comment\n"
            "         comment */\n"
            "    A = B;",
            format("const /** comment comment comment comment */\n"
                   "    A = B;",
                   getLLVMStyleWithColumns(40)));
}

TEST_F(FormatTestComments, PythonStyleComments) {
  // Keeps a space after '#'.
  EXPECT_EQ("# comment\n"
            "key: value",
            format("#comment\n"
                   "key:value",
                   getTextProtoStyleWithColumns(20)));
  EXPECT_EQ("# comment\n"
            "key: value",
            format("# comment\n"
                   "key:value",
                   getTextProtoStyleWithColumns(20)));
  // Breaks long comment.
  EXPECT_EQ("# comment comment\n"
            "# comment\n"
            "key: value",
            format("# comment comment comment\n"
                   "key:value",
                   getTextProtoStyleWithColumns(20)));
  // Indents comments.
  EXPECT_EQ("data {\n"
            "  # comment comment\n"
            "  # comment\n"
            "  key: value\n"
            "}",
            format("data {\n"
                   "# comment comment comment\n"
                   "key: value}",
                   getTextProtoStyleWithColumns(20)));
  EXPECT_EQ("data {\n"
            "  # comment comment\n"
            "  # comment\n"
            "  key: value\n"
            "}",
            format("data {# comment comment comment\n"
                   "key: value}",
                   getTextProtoStyleWithColumns(20)));
  // Reflows long comments.
  EXPECT_EQ("# comment comment\n"
            "# comment comment\n"
            "key: value",
            format("# comment comment comment\n"
                   "# comment\n"
                   "key:value",
                   getTextProtoStyleWithColumns(20)));
  // Breaks trailing comments.
  EXPECT_EQ("k: val  # comment\n"
            "        # comment\n"
            "a: 1",
            format("k:val#comment comment\n"
                   "a:1",
                   getTextProtoStyleWithColumns(20)));
  EXPECT_EQ("id {\n"
            "  k: val  # comment\n"
            "          # comment\n"
            "  # line line\n"
            "  a: 1\n"
            "}",
            format("id {k:val#comment comment\n"
                   "# line line\n"
                   "a:1}",
                   getTextProtoStyleWithColumns(20)));
  // Aligns trailing comments.
  EXPECT_EQ("k: val  # commen1\n"
            "        # commen2\n"
            "        # commen3\n"
            "# commen4\n"
            "a: 1  # commen5\n"
            "      # commen6\n"
            "      # commen7",
            format("k:val#commen1 commen2\n"
                   " #commen3\n"
                   "# commen4\n"
                   "a:1#commen5 commen6\n"
                   " #commen7",
                   getTextProtoStyleWithColumns(20)));
}

TEST_F(FormatTestComments, BreaksBeforeTrailingUnbreakableSequence) {
  // The end of /* trail */ is exactly at 80 columns, but the unbreakable
  // trailing sequence ); after it exceeds the column limit. Make sure we
  // correctly break the line in that case.
  verifyFormat("int a =\n"
               "    foo(/* trail */);",
               getLLVMStyleWithColumns(23));
}

TEST_F(FormatTestComments, ReflowBackslashCrash) {
  // clang-format off
  EXPECT_EQ(
"// How to run:\n"
"// bbbbb run \\\n"
"// rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr\n"
"// \\ <log_file> -- --output_directory=\"<output_directory>\"",
  format(
"// How to run:\n"
"// bbbbb run \\\n"
"// rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr \\\n"
"// <log_file> -- --output_directory=\"<output_directory>\""));
  // clang-format on
}

TEST_F(FormatTestComments, IndentsLongJavadocAnnotatedLines) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_Java);
  Style.ColumnLimit = 60;
  FormatStyle Style20 = getGoogleStyle(FormatStyle::LK_Java);
  Style20.ColumnLimit = 20;
  EXPECT_EQ(
      "/**\n"
      " * @param x long long long long long long long long long\n"
      " *     long\n"
      " */",
      format("/**\n"
             " * @param x long long long long long long long long long long\n"
             " */",
             Style));
  EXPECT_EQ("/**\n"
            " * @param x long long long long long long long long long\n"
            " *     long long long long long long long long long long\n"
            " */",
            format("/**\n"
                   " * @param x long long long long long long long long long "
                   "long long long long long long long long long long\n"
                   " */",
                   Style));
  EXPECT_EQ("/**\n"
            " * @param x long long long long long long long long long\n"
            " *     long long long long long long long long long long\n"
            " *     long\n"
            " */",
            format("/**\n"
                   " * @param x long long long long long long long long long "
                   "long long long long long long long long long long long\n"
                   " */",
                   Style));
  EXPECT_EQ("/**\n"
            " * Sentence that\n"
            " * should be broken.\n"
            " * @param short\n"
            " * keep indentation\n"
            " */",
            format("/**\n"
                   " * Sentence that should be broken.\n"
                   " * @param short\n"
                   " * keep indentation\n"
                   " */",
                   Style20));

  EXPECT_EQ("/**\n"
            " * @param l1 long1\n"
            " *     to break\n"
            " * @param l2 long2\n"
            " *     to break\n"
            " */",
            format("/**\n"
                   " * @param l1 long1 to break\n"
                   " * @param l2 long2 to break\n"
                   " */",
                   Style20));

  EXPECT_EQ("/**\n"
            " * @param xx to\n"
            " *     break\n"
            " * no reflow\n"
            " */",
            format("/**\n"
                   " * @param xx to break\n"
                   " * no reflow\n"
                   " */",
                   Style20));

  EXPECT_EQ("/**\n"
            " * @param xx to\n"
            " *     break yes\n"
            " *     reflow\n"
            " */",
            format("/**\n"
                   " * @param xx to break\n"
                   " *     yes reflow\n"
                   " */",
                   Style20));

  FormatStyle JSStyle20 = getGoogleStyle(FormatStyle::LK_JavaScript);
  JSStyle20.ColumnLimit = 20;
  EXPECT_EQ("/**\n"
            " * @param l1 long1\n"
            " *     to break\n"
            " */",
            format("/**\n"
                   " * @param l1 long1 to break\n"
                   " */",
                   JSStyle20));
  EXPECT_EQ("/**\n"
            " * @param {l1 long1\n"
            " *     to break}\n"
            " */",
            format("/**\n"
                   " * @param {l1 long1 to break}\n"
                   " */",
                   JSStyle20));
}

TEST_F(FormatTestComments, SpaceAtLineCommentBegin) {
  FormatStyle Style = getLLVMStyle();
  StringRef NoTextInComment = " //       \n"
                              "\n"
                              "void foo() {// \n"
                              "// \n"
                              "}";

  EXPECT_EQ("//\n"
            "\n"
            "void foo() { //\n"
            "  //\n"
            "}",
            format(NoTextInComment, Style));

  Style.SpacesInLineCommentPrefix.Minimum = 0;
  verifyFormat("//#comment", Style);
  EXPECT_EQ("//\n"
            "\n"
            "void foo() { //\n"
            "  //\n"
            "}",
            format(NoTextInComment, Style));

  Style.SpacesInLineCommentPrefix.Minimum = 5;
  EXPECT_EQ("//     #comment", format("//#comment", Style));
  EXPECT_EQ("//\n"
            "\n"
            "void foo() { //\n"
            "  //\n"
            "}",
            format(NoTextInComment, Style));

  Style = getLLVMStyle();
  StringRef Code =
      "//Free comment without space\n"
      "\n"
      "//   Free comment with 3 spaces\n"
      "\n"
      "///Free Doxygen without space\n"
      "\n"
      "///   Free Doxygen with 3 spaces\n"
      "\n"
      "//🐉 A nice dragon\n"
      "\n"
      "//\t abccba\n"
      "\n"
      "//\\t deffed\n"
      "\n"
      "//   🐉 Another nice dragon\n"
      "\n"
      "//   \t Three leading spaces following tab\n"
      "\n"
      "//   \\t Three leading spaces following backslash\n"
      "\n"
      "/// A Doxygen Comment with a nested list:\n"
      "/// - Foo\n"
      "/// - Bar\n"
      "///   - Baz\n"
      "///   - End\n"
      "///     of the inner list\n"
      "///   .\n"
      "/// .\n"
      "\n"
      "namespace Foo {\n"
      "bool bar(bool b) {\n"
      "  bool ret1 = true; ///<Doxygenstyle without space\n"
      "  bool ret2 = true; ///<   Doxygenstyle with 3 spaces\n"
      "  if (b) {\n"
      "    //Foo\n"
      "\n"
      "    //   In function comment\n"
      "    ret2 = false;\n"
      "  } // End of if\n"
      "\n"
      "//  if (ret1) {\n" // Commented out at the beginning of the line
      "//    return ret2;\n"
      "//  }\n"
      "\n"
      "  //if (ret1) {\n" // Commtented out at the beginning of the content
      "  //  return ret2;\n"
      "  //}\n"
      "\n"
      "  return ret1 && ret2;\n"
      "}\n"
      "}\n"
      "\n"
      "namespace Bar {\n"
      "int foo();\n"
      "} //  namespace Bar\n"
      "//@Nothing added because of the non ascii char\n"
      "\n"
      "//@      Nothing removed because of the non ascii char\n"
      "\n"
      "//  Comment to move to the left\n"
      "//But not this?\n"
      "//  @but this\n"
      "\n"
      "//Comment to move to the right\n"
      "//@ this stays\n"
      "\n"
      "//} will not move\n"
      "\n"
      "//vv will only move\n"
      "//} if the line above does";

  EXPECT_EQ("// Free comment without space\n"
            "\n"
            "//   Free comment with 3 spaces\n"
            "\n"
            "/// Free Doxygen without space\n"
            "\n"
            "///   Free Doxygen with 3 spaces\n"
            "\n"
            "// 🐉 A nice dragon\n"
            "\n"
            "//\t abccba\n"
            "\n"
            "//\\t deffed\n"
            "\n"
            "//   🐉 Another nice dragon\n"
            "\n"
            "//   \t Three leading spaces following tab\n"
            "\n"
            "//   \\t Three leading spaces following backslash\n"
            "\n"
            "/// A Doxygen Comment with a nested list:\n"
            "/// - Foo\n"
            "/// - Bar\n"
            "///   - Baz\n"
            "///   - End\n"
            "///     of the inner list\n"
            "///   .\n"
            "/// .\n"
            "\n"
            "namespace Foo {\n"
            "bool bar(bool b) {\n"
            "  bool ret1 = true; ///< Doxygenstyle without space\n"
            "  bool ret2 = true; ///<   Doxygenstyle with 3 spaces\n"
            "  if (b) {\n"
            "    // Foo\n"
            "\n"
            "    //   In function comment\n"
            "    ret2 = false;\n"
            "  } // End of if\n"
            "\n"
            "  //  if (ret1) {\n"
            "  //    return ret2;\n"
            "  //  }\n"
            "\n"
            "  // if (ret1) {\n"
            "  //   return ret2;\n"
            "  // }\n"
            "\n"
            "  return ret1 && ret2;\n"
            "}\n"
            "} // namespace Foo\n"
            "\n"
            "namespace Bar {\n"
            "int foo();\n"
            "} //  namespace Bar\n"
            "//@Nothing added because of the non ascii char\n"
            "\n"
            "//@      Nothing removed because of the non ascii char\n"
            "\n"
            "//  Comment to move to the left\n"
            "// But not this?\n"
            "//  @but this\n"
            "\n"
            "// Comment to move to the right\n"
            "//@ this stays\n"
            "\n"
            "//} will not move\n"
            "\n"
            "// vv will only move\n"
            "// } if the line above does",
            format(Code, Style));

  Style.SpacesInLineCommentPrefix = {0, 0};
  EXPECT_EQ("//#comment", format("//   #comment", Style));
  EXPECT_EQ("//Free comment without space\n"
            "\n"
            "//Free comment with 3 spaces\n"
            "\n"
            "///Free Doxygen without space\n"
            "\n"
            "///Free Doxygen with 3 spaces\n"
            "\n"
            "//🐉 A nice dragon\n"
            "\n"
            "//\t abccba\n"
            "\n"
            "//\\t deffed\n"
            "\n"
            "//🐉 Another nice dragon\n"
            "\n"
            "//\t Three leading spaces following tab\n"
            "\n"
            "//\\t Three leading spaces following backslash\n"
            "\n"
            "///A Doxygen Comment with a nested list:\n"
            "///- Foo\n"
            "///- Bar\n"
            "///  - Baz\n" // Here we keep the relative indentation
            "///  - End\n"
            "///    of the inner list\n"
            "///  .\n"
            "///.\n"
            "\n"
            "namespace Foo {\n"
            "bool bar(bool b) {\n"
            "  bool ret1 = true; ///<Doxygenstyle without space\n"
            "  bool ret2 = true; ///<Doxygenstyle with 3 spaces\n"
            "  if (b) {\n"
            "    //Foo\n"
            "\n"
            "    //In function comment\n"
            "    ret2 = false;\n"
            "  } //End of if\n"
            "\n"
            "  //if (ret1) {\n"
            "  //  return ret2;\n"
            "  //}\n"
            "\n"
            "  //if (ret1) {\n"
            "  //  return ret2;\n"
            "  //}\n"
            "\n"
            "  return ret1 && ret2;\n"
            "}\n"
            "} //namespace Foo\n"
            "\n"
            "namespace Bar {\n"
            "int foo();\n"
            "} //namespace Bar\n"
            "//@Nothing added because of the non ascii char\n"
            "\n"
            "//@      Nothing removed because of the non ascii char\n"
            "\n"
            "//Comment to move to the left\n"
            "//But not this?\n"
            "//@but this\n"
            "\n"
            "//Comment to move to the right\n"
            "//@ this stays\n"
            "\n"
            "//} will not move\n"
            "\n"
            "//vv will only move\n"
            "//} if the line above does",
            format(Code, Style));

  Style.SpacesInLineCommentPrefix = {2, -1u};
  EXPECT_EQ("//  Free comment without space\n"
            "\n"
            "//   Free comment with 3 spaces\n"
            "\n"
            "///  Free Doxygen without space\n"
            "\n"
            "///   Free Doxygen with 3 spaces\n"
            "\n"
            "//  🐉 A nice dragon\n"
            "\n"
            "//\t abccba\n"
            "\n"
            "//\\t deffed\n"
            "\n"
            "//   🐉 Another nice dragon\n"
            "\n"
            "//   \t Three leading spaces following tab\n"
            "\n"
            "//   \\t Three leading spaces following backslash\n"
            "\n"
            "///  A Doxygen Comment with a nested list:\n"
            "///  - Foo\n"
            "///  - Bar\n"
            "///    - Baz\n"
            "///    - End\n"
            "///      of the inner list\n"
            "///    .\n"
            "///  .\n"
            "\n"
            "namespace Foo {\n"
            "bool bar(bool b) {\n"
            "  bool ret1 = true; ///<  Doxygenstyle without space\n"
            "  bool ret2 = true; ///<   Doxygenstyle with 3 spaces\n"
            "  if (b) {\n"
            "    //  Foo\n"
            "\n"
            "    //   In function comment\n"
            "    ret2 = false;\n"
            "  } //  End of if\n"
            "\n"
            "  //  if (ret1) {\n"
            "  //    return ret2;\n"
            "  //  }\n"
            "\n"
            "  //  if (ret1) {\n"
            "  //    return ret2;\n"
            "  //  }\n"
            "\n"
            "  return ret1 && ret2;\n"
            "}\n"
            "} //  namespace Foo\n"
            "\n"
            "namespace Bar {\n"
            "int foo();\n"
            "} //  namespace Bar\n"
            "//@Nothing added because of the non ascii char\n"
            "\n"
            "//@      Nothing removed because of the non ascii char\n"
            "\n"
            "//  Comment to move to the left\n"
            "//  But not this?\n"
            "//  @but this\n"
            "\n"
            "//  Comment to move to the right\n"
            "//@ this stays\n"
            "\n"
            "//} will not move\n"
            "\n"
            "//  vv will only move\n"
            "//  } if the line above does",
            format(Code, Style));

  Style = getLLVMStyleWithColumns(20);
  StringRef WrapCode = "//Lorem ipsum dolor sit amet\n"
                       "\n"
                       "//  Lorem   ipsum   dolor   sit   amet\n"
                       "\n"
                       "void f() {//Hello World\n"
                       "}";

  EXPECT_EQ("// Lorem ipsum dolor\n"
            "// sit amet\n"
            "\n"
            "//  Lorem   ipsum\n"
            "//  dolor   sit amet\n"
            "\n"
            "void f() { // Hello\n"
            "           // World\n"
            "}",
            format(WrapCode, Style));

  Style.SpacesInLineCommentPrefix = {0, 0};
  EXPECT_EQ("//Lorem ipsum dolor\n"
            "//sit amet\n"
            "\n"
            "//Lorem   ipsum\n"
            "//dolor   sit   amet\n"
            "\n"
            "void f() { //Hello\n"
            "           //World\n"
            "}",
            format(WrapCode, Style));

  Style.SpacesInLineCommentPrefix = {1, 1};
  EXPECT_EQ("// Lorem ipsum dolor\n"
            "// sit amet\n"
            "\n"
            "// Lorem   ipsum\n"
            "// dolor   sit amet\n"
            "\n"
            "void f() { // Hello\n"
            "           // World\n"
            "}",
            format(WrapCode, Style));
  EXPECT_EQ("// x\n"
            "// y",
            format("//   x\n"
                   "// y",
                   Style));
  EXPECT_EQ(
      "// loooooooooooooooooooooooooooooong\n"
      "// commentcomments\n"
      "// normal comments",
      format("//            loooooooooooooooooooooooooooooong commentcomments\n"
             "// normal comments",
             Style));

  Style.SpacesInLineCommentPrefix = {3, 3};
  EXPECT_EQ("//   Lorem ipsum\n"
            "//   dolor sit amet\n"
            "\n"
            "//   Lorem   ipsum\n"
            "//   dolor   sit\n"
            "//   amet\n"
            "\n"
            "void f() { //   Hello\n"
            "           //   World\n"
            "}",
            format(WrapCode, Style));

  Style = getLLVMStyleWithColumns(20);
  StringRef LotsOfSpaces = "//                      This are more spaces "
                           "than the ColumnLimit, what now?\n"
                           "\n"
                           "//   Comment\n"
                           "\n"
                           "// This is a text to split in multiple "
                           "lines, please. Thank you very much!\n"
                           "\n"
                           "// A comment with\n"
                           "//   some indentation that has to be split.\n"
                           "// And now without";
  EXPECT_EQ("//                      This are more spaces "
            "than the ColumnLimit, what now?\n"
            "\n"
            "//   Comment\n"
            "\n"
            "// This is a text to\n"
            "// split in multiple\n"
            "// lines, please.\n"
            "// Thank you very\n"
            "// much!\n"
            "\n"
            "// A comment with\n"
            "//   some\n"
            "//   indentation\n"
            "//   that has to be\n"
            "//   split.\n"
            "// And now without",
            format(LotsOfSpaces, Style));

  Style.SpacesInLineCommentPrefix = {0, 0};
  EXPECT_EQ("//This are more\n"
            "//spaces than the\n"
            "//ColumnLimit, what\n"
            "//now?\n"
            "\n"
            "//Comment\n"
            "\n"
            "//This is a text to\n"
            "//split in multiple\n"
            "//lines, please.\n"
            "//Thank you very\n"
            "//much!\n"
            "\n"
            "//A comment with\n"
            "//  some indentation\n"
            "//  that has to be\n"
            "//  split.\n"
            "//And now without",
            format(LotsOfSpaces, Style));

  Style.SpacesInLineCommentPrefix = {3, 3};
  EXPECT_EQ("//   This are more\n"
            "//   spaces than the\n"
            "//   ColumnLimit,\n"
            "//   what now?\n"
            "\n"
            "//   Comment\n"
            "\n"
            "//   This is a text\n"
            "//   to split in\n"
            "//   multiple lines,\n"
            "//   please. Thank\n"
            "//   you very much!\n"
            "\n"
            "//   A comment with\n"
            "//     some\n"
            "//     indentation\n"
            "//     that has to\n"
            "//     be split.\n"
            "//   And now without",
            format(LotsOfSpaces, Style));

  Style.SpacesInLineCommentPrefix = {30, -1u};
  EXPECT_EQ("//                              This are more spaces than the "
            "ColumnLimit, what now?\n"
            "\n"
            "//                              Comment\n"
            "\n"
            "//                              This is a text to split in "
            "multiple lines, please. Thank you very much!\n"
            "\n"
            "//                              A comment with\n"
            "//                                some indentation that has to be "
            "split.\n"
            "//                              And now without",
            format(LotsOfSpaces, Style));

  Style.SpacesInLineCommentPrefix = {2, 4};
  EXPECT_EQ("//  A Comment to be\n"
            "//  moved\n"
            "//   with indent\n"
            "\n"
            "//  A Comment to be\n"
            "//  moved\n"
            "//   with indent\n"
            "\n"
            "//  A Comment to be\n"
            "//  moved\n"
            "//   with indent\n"
            "\n"
            "//   A Comment to be\n"
            "//   moved\n"
            "//    with indent\n"
            "\n"
            "//    A Comment to\n"
            "//    be moved\n"
            "//     with indent\n"
            "\n"
            "//    A Comment to\n"
            "//    be moved\n"
            "//     with indent\n"
            "\n"
            "//    A Comment to\n"
            "//    be moved\n"
            "//     with indent",
            format("//A Comment to be moved\n"
                   "// with indent\n"
                   "\n"
                   "// A Comment to be moved\n"
                   "//  with indent\n"
                   "\n"
                   "//  A Comment to be moved\n"
                   "//   with indent\n"
                   "\n"
                   "//   A Comment to be moved\n"
                   "//    with indent\n"
                   "\n"
                   "//    A Comment to be moved\n"
                   "//     with indent\n"
                   "\n"
                   "//     A Comment to be moved\n"
                   "//      with indent\n"
                   "\n"
                   "//      A Comment to be moved\n"
                   "//       with indent",
                   Style));

  Style.ColumnLimit = 30;
  EXPECT_EQ("int i; //  A Comment to be\n"
            "       //  moved\n"
            "       //   with indent\n"
            "\n"
            "int i; //  A Comment to be\n"
            "       //  moved\n"
            "       //   with indent\n"
            "\n"
            "int i; //  A Comment to be\n"
            "       //  moved\n"
            "       //   with indent\n"
            "\n"
            "int i; //   A Comment to be\n"
            "       //   moved\n"
            "       //    with indent\n"
            "\n"
            "int i; //    A Comment to be\n"
            "       //    moved\n"
            "       //     with indent\n"
            "\n"
            "int i; //    A Comment to be\n"
            "       //    moved\n"
            "       //     with indent\n"
            "\n"
            "int i; //    A Comment to be\n"
            "       //    moved\n"
            "       //     with indent",
            format("int i;//A Comment to be moved\n"
                   "      // with indent\n"
                   "\n"
                   "int i;// A Comment to be moved\n"
                   "      //  with indent\n"
                   "\n"
                   "int i;//  A Comment to be moved\n"
                   "      //   with indent\n"
                   "\n"
                   "int i;//   A Comment to be moved\n"
                   "      //    with indent\n"
                   "\n"
                   "int i;//    A Comment to be moved\n"
                   "      //     with indent\n"
                   "\n"
                   "int i;//     A Comment to be moved\n"
                   "      //      with indent\n"
                   "\n"
                   "int i;//      A Comment to be moved\n"
                   "      //       with indent",
                   Style));

  Style = getLLVMStyleWithColumns(0);
  EXPECT_EQ("// Free comment without space\n"
            "\n"
            "//   Free comment with 3 spaces\n"
            "\n"
            "/// Free Doxygen without space\n"
            "\n"
            "///   Free Doxygen with 3 spaces\n"
            "\n"
            "// 🐉 A nice dragon\n"
            "\n"
            "//\t abccba\n"
            "\n"
            "//\\t deffed\n"
            "\n"
            "//   🐉 Another nice dragon\n"
            "\n"
            "//   \t Three leading spaces following tab\n"
            "\n"
            "//   \\t Three leading spaces following backslash\n"
            "\n"
            "/// A Doxygen Comment with a nested list:\n"
            "/// - Foo\n"
            "/// - Bar\n"
            "///   - Baz\n"
            "///   - End\n"
            "///     of the inner list\n"
            "///   .\n"
            "/// .\n"
            "\n"
            "namespace Foo {\n"
            "bool bar(bool b) {\n"
            "  bool ret1 = true; ///< Doxygenstyle without space\n"
            "  bool ret2 = true; ///<   Doxygenstyle with 3 spaces\n"
            "  if (b) {\n"
            "    // Foo\n"
            "\n"
            "    //   In function comment\n"
            "    ret2 = false;\n"
            "  } // End of if\n"
            "\n"
            "  //  if (ret1) {\n"
            "  //    return ret2;\n"
            "  //  }\n"
            "\n"
            "  // if (ret1) {\n"
            "  //   return ret2;\n"
            "  // }\n"
            "\n"
            "  return ret1 && ret2;\n"
            "}\n"
            "} // namespace Foo\n"
            "\n"
            "namespace Bar {\n"
            "int foo();\n"
            "} //  namespace Bar\n"
            "//@Nothing added because of the non ascii char\n"
            "\n"
            "//@      Nothing removed because of the non ascii char\n"
            "\n"
            "//  Comment to move to the left\n"
            "// But not this?\n"
            "//  @but this\n"
            "\n"
            "// Comment to move to the right\n"
            "//@ this stays\n"
            "\n"
            "//} will not move\n"
            "\n"
            "// vv will only move\n"
            "// } if the line above does",
            format(Code, Style));

  Style.SpacesInLineCommentPrefix = {0, 0};
  EXPECT_EQ("//Free comment without space\n"
            "\n"
            "//Free comment with 3 spaces\n"
            "\n"
            "///Free Doxygen without space\n"
            "\n"
            "///Free Doxygen with 3 spaces\n"
            "\n"
            "//🐉 A nice dragon\n"
            "\n"
            "//\t abccba\n"
            "\n"
            "//\\t deffed\n"
            "\n"
            "//🐉 Another nice dragon\n"
            "\n"
            "//\t Three leading spaces following tab\n"
            "\n"
            "//\\t Three leading spaces following backslash\n"
            "\n"
            "///A Doxygen Comment with a nested list:\n"
            "///- Foo\n"
            "///- Bar\n"
            "///  - Baz\n" // Here we keep the relative indentation
            "///  - End\n"
            "///    of the inner list\n"
            "///  .\n"
            "///.\n"
            "\n"
            "namespace Foo {\n"
            "bool bar(bool b) {\n"
            "  bool ret1 = true; ///<Doxygenstyle without space\n"
            "  bool ret2 = true; ///<Doxygenstyle with 3 spaces\n"
            "  if (b) {\n"
            "    //Foo\n"
            "\n"
            "    //In function comment\n"
            "    ret2 = false;\n"
            "  } //End of if\n"
            "\n"
            "  //if (ret1) {\n"
            "  //  return ret2;\n"
            "  //}\n"
            "\n"
            "  //if (ret1) {\n"
            "  //  return ret2;\n"
            "  //}\n"
            "\n"
            "  return ret1 && ret2;\n"
            "}\n"
            "} //namespace Foo\n"
            "\n"
            "namespace Bar {\n"
            "int foo();\n"
            "} //namespace Bar\n"
            "//@Nothing added because of the non ascii char\n"
            "\n"
            "//@      Nothing removed because of the non ascii char\n"
            "\n"
            "//Comment to move to the left\n"
            "//But not this?\n"
            "//@but this\n"
            "\n"
            "//Comment to move to the right\n"
            "//@ this stays\n"
            "\n"
            "//} will not move\n"
            "\n"
            "//vv will only move\n"
            "//} if the line above does",
            format(Code, Style));

  Style.SpacesInLineCommentPrefix = {2, -1u};
  EXPECT_EQ("//  Free comment without space\n"
            "\n"
            "//   Free comment with 3 spaces\n"
            "\n"
            "///  Free Doxygen without space\n"
            "\n"
            "///   Free Doxygen with 3 spaces\n"
            "\n"
            "//  🐉 A nice dragon\n"
            "\n"
            "//\t abccba\n"
            "\n"
            "//\\t deffed\n"
            "\n"
            "//   🐉 Another nice dragon\n"
            "\n"
            "//   \t Three leading spaces following tab\n"
            "\n"
            "//   \\t Three leading spaces following backslash\n"
            "\n"
            "///  A Doxygen Comment with a nested list:\n"
            "///  - Foo\n"
            "///  - Bar\n"
            "///    - Baz\n"
            "///    - End\n"
            "///      of the inner list\n"
            "///    .\n"
            "///  .\n"
            "\n"
            "namespace Foo {\n"
            "bool bar(bool b) {\n"
            "  bool ret1 = true; ///<  Doxygenstyle without space\n"
            "  bool ret2 = true; ///<   Doxygenstyle with 3 spaces\n"
            "  if (b) {\n"
            "    //  Foo\n"
            "\n"
            "    //   In function comment\n"
            "    ret2 = false;\n"
            "  } //  End of if\n"
            "\n"
            "  //  if (ret1) {\n"
            "  //    return ret2;\n"
            "  //  }\n"
            "\n"
            "  //  if (ret1) {\n"
            "  //    return ret2;\n"
            "  //  }\n"
            "\n"
            "  return ret1 && ret2;\n"
            "}\n"
            "} //  namespace Foo\n"
            "\n"
            "namespace Bar {\n"
            "int foo();\n"
            "} //  namespace Bar\n"
            "//@Nothing added because of the non ascii char\n"
            "\n"
            "//@      Nothing removed because of the non ascii char\n"
            "\n"
            "//  Comment to move to the left\n"
            "//  But not this?\n"
            "//  @but this\n"
            "\n"
            "//  Comment to move to the right\n"
            "//@ this stays\n"
            "\n"
            "//} will not move\n"
            "\n"
            "//  vv will only move\n"
            "//  } if the line above does",
            format(Code, Style));
}

TEST_F(FormatTestComments, SplitCommentIntroducers) {
  EXPECT_EQ(R"(//
/\
/
)",
            format(R"(//
/\
/ 
  )",
                   getLLVMStyleWithColumns(10)));
}

} // end namespace
} // namespace test
} // end namespace format
} // end namespace clang
