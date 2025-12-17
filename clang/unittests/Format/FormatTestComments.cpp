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

  verifyFormat("// comment", "//comment");
  verifyFormat("// #comment", "//#comment");

  verifyFormat("// comment\n"
               "// clang-format on",
               "//comment\n"
               "// clang-format on");

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
  verifyFormat("#include \"a\"     // comment\n"
               "#include \"a/b/c\" // comment",
               "#include \\\n"
               "  \"a\" // comment\n"
               "#include \"a/b/c\" // comment");

  verifyFormat("enum E {\n"
               "  // comment\n"
               "  VAL_A, // comment\n"
               "  VAL_B\n"
               "};");

  const auto Style20 = getLLVMStyleWithColumns(20);

  verifyFormat("enum A {\n"
               "  // line a\n"
               "  a,\n"
               "  b, // line b\n"
               "\n"
               "  // line c\n"
               "  c\n"
               "};",
               Style20);
  verifyNoChange("enum A {\n"
                 "  a, // line 1\n"
                 "  // line 2\n"
                 "};",
                 Style20);
  verifyFormat("enum A {\n"
               "  a, // line 1\n"
               "     // line 2\n"
               "};",
               "enum A {\n"
               "  a, // line 1\n"
               "   // line 2\n"
               "};",
               Style20);
  verifyNoChange("enum A {\n"
                 "  a, // line 1\n"
                 "  // line 2\n"
                 "  b\n"
                 "};",
                 Style20);
  verifyFormat("enum A {\n"
               "  a, // line 1\n"
               "     // line 2\n"
               "  b\n"
               "};",
               "enum A {\n"
               "  a, // line 1\n"
               "   // line 2\n"
               "  b\n"
               "};",
               Style20);
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
               Style20);

  verifyFormat("void f() { // This does something ..\n"
               "}\n"
               "int a; // This is unrelated",
               "void f()    {     // This does something ..\n"
               "  }\n"
               "int   a;     // This is unrelated");
  verifyFormat("class C {\n"
               "  void f() { // This does something ..\n"
               "  } // awesome..\n"
               "\n"
               "  int a; // This is unrelated\n"
               "};",
               "class C{void f()    { // This does something ..\n"
               "      } // awesome..\n"
               " \n"
               "int a;    // This is unrelated\n"
               "};");

  verifyFormat("int i; // single line trailing comment",
               "int i;\\\n// single line trailing comment");

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
  verifyFormat("D(a, {\n"
               "  // test\n"
               "  int a;\n"
               "});",
               "D(a, {\n"
               "// test\n"
               "int a;\n"
               "});");

  verifyFormat("lineWith(); // comment\n"
               "// at start\n"
               "otherLine();",
               "lineWith();   // comment\n"
               "// at start\n"
               "otherLine();");
  verifyFormat("lineWith(); // comment\n"
               "/*\n"
               " * at start */\n"
               "otherLine();",
               "lineWith();   // comment\n"
               "/*\n"
               " * at start */\n"
               "otherLine();");
  verifyFormat("lineWith(); // comment\n"
               "            // at start\n"
               "otherLine();",
               "lineWith();   // comment\n"
               " // at start\n"
               "otherLine();");

  verifyFormat("lineWith(); // comment\n"
               "// at start\n"
               "otherLine(); // comment",
               "lineWith();   // comment\n"
               "// at start\n"
               "otherLine();   // comment");
  verifyFormat("lineWith();\n"
               "// at start\n"
               "otherLine(); // comment",
               "lineWith();\n"
               " // at start\n"
               "otherLine();   // comment");
  verifyFormat("// first\n"
               "// at start\n"
               "otherLine(); // comment",
               "// first\n"
               " // at start\n"
               "otherLine();   // comment");
  verifyFormat("f();\n"
               "// first\n"
               "// at start\n"
               "otherLine(); // comment",
               "f();\n"
               "// first\n"
               " // at start\n"
               "otherLine();   // comment");
  verifyFormat("f(); // comment\n"
               "// first\n"
               "// at start\n"
               "otherLine();");
  verifyFormat("f(); // comment\n"
               "// first\n"
               "// at start\n"
               "otherLine();",
               "f();   // comment\n"
               "// first\n"
               " // at start\n"
               "otherLine();");
  verifyFormat("f(); // comment\n"
               "     // first\n"
               "// at start\n"
               "otherLine();",
               "f();   // comment\n"
               " // first\n"
               "// at start\n"
               "otherLine();");
  verifyFormat("void f() {\n"
               "  lineWith(); // comment\n"
               "  // at start\n"
               "}",
               "void              f() {\n"
               "  lineWith(); // comment\n"
               "  // at start\n"
               "}");
  verifyFormat("int xy; // a\n"
               "int z;  // b",
               "int xy;    // a\n"
               "int z;    //b");
  verifyFormat("int xy; // a\n"
               "int z; // bb",
               "int xy;    // a\n"
               "int z;    //bb",
               getLLVMStyleWithColumns(12));

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
  verifyFormat("if ( // This is some comment\n"
               "     // spanning two lines\n"
               "    x + 3) {\n"
               "}",
               "if( // This is some comment\n"
               "     // spanning two lines\n"
               " x + 3) {\n"
               "}");

  verifyNoCrash("/\\\n/");
  verifyNoCrash("/\\\n* */");
  // The 0-character somehow makes the lexer return a proper comment.
  verifyNoCrash(StringRef("/*\\\0\n/", 6));
}

TEST_F(FormatTestComments, KeepsParameterWithTrailingCommentsOnTheirOwnLine) {
  verifyFormat("SomeFunction(a,\n"
               "             b, // comment\n"
               "             c);",
               "SomeFunction(a,\n"
               "          b, // comment\n"
               "      c);");
  verifyFormat("SomeFunction(a, b,\n"
               "             // comment\n"
               "             c);",
               "SomeFunction(a,\n"
               "          b,\n"
               "  // comment\n"
               "      c);");
  verifyFormat("SomeFunction(a, b, // comment (unclear relation)\n"
               "             c);",
               "SomeFunction(a, b, // comment (unclear relation)\n"
               "      c);");
  verifyFormat("SomeFunction(a, // comment\n"
               "             b,\n"
               "             c); // comment",
               "SomeFunction(a,     // comment\n"
               "          b,\n"
               "      c); // comment");
  verifyFormat("aaaaaaaaaa(aaaa(aaaa,\n"
               "                aaaa), //\n"
               "           aaaa, bbbbb);",
               "aaaaaaaaaa(aaaa(aaaa,\n"
               "aaaa), //\n"
               "aaaa, bbbbb);");

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
  verifyFormat("// comment", "// comment  ");
  verifyFormat("int aaaaaaa, bbbbbbb; // comment",
               "int aaaaaaa, bbbbbbb; // comment                   ",
               getLLVMStyleWithColumns(33));
  verifyFormat("// comment\\\n", "// comment\\\n  \t \v   \f   ");
  verifyFormat("// comment    \\\n", "// comment    \\\n  \t \v   \f   ");
}

TEST_F(FormatTestComments, UnderstandsBlockComments) {
  verifyFormat("f(/*noSpaceAfterParameterNamingComment=*/true);");
  verifyFormat("void f() { g(/*aaa=*/x, /*bbb=*/!y, /*c=*/::c); }");
  verifyFormat("fooooooooooooooooooooooooooooo(\n"
               "    /*qq_=*/move(q), [this, b](bar<void(uint32_t)> b) {},\n"
               "    c);",
               getLLVMStyleWithColumns(60));
  verifyFormat("f(aaaaaaaaaaaaaaaaaaaaaaaaa, /* Trailing comment for aa... */\n"
               "  bbbbbbbbbbbbbbbbbbbbbbbbb);",
               "f(aaaaaaaaaaaaaaaaaaaaaaaaa ,   \\\n"
               "/* Trailing comment for aa... */\n"
               "  bbbbbbbbbbbbbbbbbbbbbbbbb);");
  verifyFormat("f(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "  /* Leading comment for bb... */ bbbbbbbbbbbbbbbbbbbbbbbbb);",
               "f(aaaaaaaaaaaaaaaaaaaaaaaaa    ,   \n"
               "/* Leading comment for bb... */   bbbbbbbbbbbbbbbbbbbbbbbbb);");

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
  verifyFormat("/*\n"
               " * Really multi-line\n"
               " * comment.\n"
               " */\n"
               "void f() {}",
               "  /*\n"
               "   * Really multi-line\n"
               "   * comment.\n"
               "   */\n"
               "  void f() {}");
  verifyFormat("class C {\n"
               "  /*\n"
               "   * Another multi-line\n"
               "   * comment.\n"
               "   */\n"
               "  void f() {}\n"
               "};",
               "class C {\n"
               "/*\n"
               " * Another multi-line\n"
               " * comment.\n"
               " */\n"
               "void f() {}\n"
               "};");
  verifyFormat("/*\n"
               "  1. This is a comment with non-trivial formatting.\n"
               "     1.1. We have to indent/outdent all lines equally\n"
               "         1.1.1. to keep the formatting.\n"
               " */",
               "  /*\n"
               "    1. This is a comment with non-trivial formatting.\n"
               "       1.1. We have to indent/outdent all lines equally\n"
               "           1.1.1. to keep the formatting.\n"
               "   */");
  verifyFormat("/*\n"
               "Don't try to outdent if there's not enough indentation.\n"
               "*/",
               "  /*\n"
               " Don't try to outdent if there's not enough indentation.\n"
               " */");

  verifyNoChange("int i; /* Comment with empty...\n"
                 "        *\n"
                 "        * line. */");
  verifyFormat("int foobar = 0; /* comment */\n"
               "int bar = 0;    /* multiline\n"
               "                   comment 1 */\n"
               "int baz = 0;    /* multiline\n"
               "                   comment 2 */\n"
               "int bzz = 0;    /* multiline\n"
               "                   comment 3 */",
               "int foobar = 0; /* comment */\n"
               "int bar = 0;    /* multiline\n"
               "                   comment 1 */\n"
               "int baz = 0; /* multiline\n"
               "                comment 2 */\n"
               "int bzz = 0;         /* multiline\n"
               "                        comment 3 */");
  verifyFormat("int foobar = 0; /* comment */\n"
               "int bar = 0;    /* multiline\n"
               "   comment */\n"
               "int baz = 0;    /* multiline\n"
               "comment */",
               "int foobar = 0; /* comment */\n"
               "int bar = 0; /* multiline\n"
               "comment */\n"
               "int baz = 0;        /* multiline\n"
               "comment */");
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
  verifyFormat("double *x; /* aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "              aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa */");
  verifyFormat(
      "void ffffffffffff(\n"
      "    int aaaaaaaa, int bbbbbbbb,\n"
      "    int cccccccccccc) { /*\n"
      "                           aaaaaaaaaa\n"
      "                           aaaaaaaaaaaaa\n"
      "                           bbbbbbbbbbbbbb\n"
      "                           bbbbbbbbbb\n"
      "                         */\n"
      "}",
      "void ffffffffffff(int aaaaaaaa, int bbbbbbbb, int cccccccccccc)\n"
      "{ /*\n"
      "     aaaaaaaaaa aaaaaaaaaaaaa\n"
      "     bbbbbbbbbbbbbb bbbbbbbbbb\n"
      "   */\n"
      "}",
      getLLVMStyleWithColumns(40));
}

TEST_F(FormatTestComments, DontBreakNonTrailingBlockComments) {
  verifyFormat("void ffffffffff(\n"
               "    int aaaaa /* test */);",
               "void ffffffffff(int aaaaa /* test */);",
               getLLVMStyleWithColumns(35));
}

TEST_F(FormatTestComments, SplitsLongCxxComments) {
  const auto Style10 = getLLVMStyleWithColumns(10);
  const auto Style20 = getLLVMStyleWithColumns(20);
  const auto Style22 = getLLVMStyleWithColumns(22);
  const auto Style30 = getLLVMStyleWithColumns(30);

  verifyFormat("// A comment that\n"
               "// doesn't fit on\n"
               "// one line",
               "// A comment that doesn't fit on one line", Style20);
  verifyFormat("/// A comment that\n"
               "/// doesn't fit on\n"
               "/// one line",
               "/// A comment that doesn't fit on one line", Style20);
  verifyFormat("//! A comment that\n"
               "//! doesn't fit on\n"
               "//! one line",
               "//! A comment that doesn't fit on one line", Style20);
  verifyFormat("// a b c d\n"
               "// e f  g\n"
               "// h i j k",
               "// a b c d e f  g h i j k", Style10);
  verifyFormat("// a b c d\n"
               "// e f  g\n"
               "// h i j k",
               "\\\n// a b c d e f  g h i j k", Style10);
  verifyFormat("if (true) // A comment that\n"
               "          // doesn't fit on\n"
               "          // one line",
               "if (true) // A comment that doesn't fit on one line   ",
               Style30);
  verifyNoChange("//    Don't_touch_leading_whitespace", Style20);
  verifyFormat("// Add leading\n"
               "// whitespace",
               "//Add leading whitespace", Style20);
  verifyFormat("/// Add leading\n"
               "/// whitespace",
               "///Add leading whitespace", Style20);
  verifyFormat("//! Add leading\n"
               "//! whitespace",
               "//!Add leading whitespace", Style20);
  verifyFormat("// whitespace", "//whitespace");
  verifyFormat("// Even if it makes the line exceed the column\n"
               "// limit",
               "//Even if it makes the line exceed the column limit",
               getLLVMStyleWithColumns(51));
  verifyFormat("//--But not here");
  verifyFormat("/// line 1\n"
               "// add leading whitespace",
               "/// line 1\n"
               "//add leading whitespace",
               Style30);
  verifyFormat("/// line 1\n"
               "/// line 2\n"
               "//! line 3\n"
               "//! line 4\n"
               "//! line 5\n"
               "// line 6\n"
               "// line 7",
               "///line 1\n"
               "///line 2\n"
               "//! line 3\n"
               "//!line 4\n"
               "//!line 5\n"
               "// line 6\n"
               "//line 7",
               Style20);

  verifyFormat("// aa bb cc dd",
               "// aa bb             cc dd                   ",
               getLLVMStyleWithColumns(15));

  verifyFormat("// A comment before\n"
               "// a macro\n"
               "// definition\n"
               "#define a b",
               "// A comment before a macro definition\n"
               "#define a b",
               Style20);
  verifyFormat("void ffffff(\n"
               "    int aaaaaaaaa,  // wwww\n"
               "    int bbbbbbbbbb, // xxxxxxx\n"
               "                    // yyyyyyyyyy\n"
               "    int c, int d, int e) {}",
               "void ffffff(\n"
               "    int aaaaaaaaa, // wwww\n"
               "    int bbbbbbbbbb, // xxxxxxx yyyyyyyyyy\n"
               "    int c, int d, int e) {}",
               getLLVMStyleWithColumns(40));
  verifyFormat("//\t aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Style20);
  verifyFormat("#define XXX // a b c d\n"
               "            // e f g h",
               "#define XXX // a b c d e f g h", Style22);
  verifyFormat("#define XXX // q w e r\n"
               "            // t y u i",
               "#define XXX //q w e r t y u i", Style22);
  verifyNoChange("{\n"
                 "  //\n"
                 "  //\\\n"
                 "  // long 1 2 3 4 5\n"
                 "}",
                 Style20);
  verifyFormat("{\n"
               "  //\n"
               "  //\\\n"
               "  // long 1 2 3 4 5\n"
               "  // 6\n"
               "}",
               "{\n"
               "  //\n"
               "  //\\\n"
               "  // long 1 2 3 4 5 6\n"
               "}",
               Style20);

  verifyFormat("//: A comment that\n"
               "//: doesn't fit on\n"
               "//: one line",
               "//: A comment that doesn't fit on one line", Style20);

  verifyFormat(
      "//\t\t\t\tofMap(message.velocity, 0, 127, 0, ofGetWidth()\n"
      "//* 0.2)",
      "//\t\t\t\tofMap(message.velocity, 0, 127, 0, ofGetWidth() * 0.2)");
}

TEST_F(FormatTestComments, PreservesHangingIndentInCxxComments) {
  const auto Style20 = getLLVMStyleWithColumns(20);
  verifyFormat("//     A comment\n"
               "//     that doesn't\n"
               "//     fit on one\n"
               "//     line",
               "//     A comment that doesn't fit on one line", Style20);
  verifyFormat("///     A comment\n"
               "///     that doesn't\n"
               "///     fit on one\n"
               "///     line",
               "///     A comment that doesn't fit on one line", Style20);
}

TEST_F(FormatTestComments, DontSplitLineCommentsWithEscapedNewlines) {
  verifyNoChange("// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
                 "// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
                 "// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  verifyNoChange("int a; // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
                 "       // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
                 "       // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                 getLLVMStyleWithColumns(50));
  verifyFormat("double\n"
               "    a; // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
               "       // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
               "       // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
               "double a; // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
               "          // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\\\n"
               "          // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
               getLLVMStyleWithColumns(49));
}

TEST_F(FormatTestComments, DontIntroduceMultilineComments) {
  // Avoid introducing a multiline comment by breaking after `\`.
  auto Style = getLLVMStyle();
  for (int ColumnLimit = 15; ColumnLimit <= 17; ++ColumnLimit) {
    Style.ColumnLimit = ColumnLimit;
    verifyFormat("// aaaaaaaaaa\n"
                 "// \\ bb",
                 "// aaaaaaaaaa \\ bb", Style);
    verifyFormat("// aaaaaaaaa\n"
                 "// \\  bb",
                 "// aaaaaaaaa \\  bb", Style);
    verifyFormat("// aaaaaaaaa\n"
                 "// \\  \\ bb",
                 "// aaaaaaaaa \\  \\ bb", Style);
  }
}

TEST_F(FormatTestComments, DontSplitLineCommentsWithPragmas) {
  FormatStyle Pragmas = getLLVMStyleWithColumns(30);
  Pragmas.CommentPragmas = "^ IWYU pragma:";
  verifyFormat("// IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb", Pragmas);
  verifyFormat("/* IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb */", Pragmas);
}

TEST_F(FormatTestComments, PriorityOfCommentBreaking) {
  const auto Style40 = getLLVMStyleWithColumns(40);
  verifyFormat("if (xxx ==\n"
               "        yyy && // aaaaaaaaaaaa bbbbbbbbb\n"
               "    zzz)\n"
               "  q();",
               "if (xxx == yyy && // aaaaaaaaaaaa bbbbbbbbb\n"
               "    zzz) q();",
               Style40);
  verifyFormat("if (xxxxxxxxxx ==\n"
               "        yyy && // aaaaaa bbbbbbbb cccc\n"
               "    zzz)\n"
               "  q();",
               "if (xxxxxxxxxx == yyy && // aaaaaa bbbbbbbb cccc\n"
               "    zzz) q();",
               Style40);
  verifyFormat("if (xxxxxxxxxx &&\n"
               "        yyy || // aaaaaa bbbbbbbb cccc\n"
               "    zzz)\n"
               "  q();",
               "if (xxxxxxxxxx && yyy || // aaaaaa bbbbbbbb cccc\n"
               "    zzz) q();",
               Style40);
  verifyFormat("fffffffff(\n"
               "    &xxx, // aaaaaaaaaaaa bbbbbbbbbbb\n"
               "    zzz);",
               "fffffffff(&xxx, // aaaaaaaaaaaa bbbbbbbbbbb\n"
               " zzz);",
               Style40);
}

TEST_F(FormatTestComments, MultiLineCommentsInDefines) {
  const auto Style17 = getLLVMStyleWithColumns(17);
  verifyNoChange("#define A(x) /* \\\n"
                 "  a comment     \\\n"
                 "  inside */     \\\n"
                 "  f();",
                 Style17);
  verifyNoChange("#define A(      \\\n"
                 "    x) /*       \\\n"
                 "  a comment     \\\n"
                 "  inside */     \\\n"
                 "  f();",
                 Style17);
}

TEST_F(FormatTestComments, LineCommentsInMacrosDoNotGetEscapedNewlines) {
  FormatStyle Style = getLLVMStyleWithColumns(0);
  Style.ReflowComments = FormatStyle::RCS_Never;
  verifyFormat("#define FOO (1U) // comment\n"
               "                 // comment",
               Style);

  Style.ColumnLimit = 32;
  verifyFormat("#define SOME_MACRO(x) x\n"
               "#define FOO                    \\\n"
               "  SOME_MACRO(1) +              \\\n"
               "      SOME_MACRO(2) // comment\n"
               "                    // comment",
               "#define SOME_MACRO(x) x\n"
               "#define FOO SOME_MACRO(1) + SOME_MACRO(2) // comment\n"
               "                                          // comment",
               Style);
}

TEST_F(FormatTestComments, ParsesCommentsAdjacentToPPDirectives) {
  verifyFormat("namespace {}\n// Test\n#define A",
               "namespace {}\n   // Test\n#define A");
  verifyFormat("namespace {}\n/* Test */\n#define A",
               "namespace {}\n   /* Test */\n#define A");
  verifyFormat("namespace {}\n/* Test */ #define A",
               "namespace {}\n   /* Test */    #define A");
}

TEST_F(FormatTestComments, KeepsLevelOfCommentBeforePPDirective) {
  // Keep the current level if the comment was originally not aligned with
  // the preprocessor directive.
  verifyNoChange("void f() {\n"
                 "  int i;\n"
                 "  /* comment */\n"
                 "#ifdef A\n"
                 "  int j;\n"
                 "}");

  verifyNoChange("void f() {\n"
                 "  int i;\n"
                 "  /* comment */\n"
                 "\n"
                 "#ifdef A\n"
                 "  int j;\n"
                 "}");

  verifyFormat("int f(int i) {\n"
               "  if (true) {\n"
               "    ++i;\n"
               "  }\n"
               "  // comment\n"
               "#ifdef A\n"
               "  int j;\n"
               "#endif\n"
               "}",
               "int f(int i) {\n"
               "  if (true) {\n"
               "    ++i;\n"
               "  }\n"
               "  // comment\n"
               "#ifdef A\n"
               "int j;\n"
               "#endif\n"
               "}");

  verifyFormat("int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               "    // comment in else\n"
               "#ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}",
               "int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               "  // comment in else\n"
               "#ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}");

  verifyFormat("int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               "    /* comment in else */\n"
               "#ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}",
               "int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               "  /* comment in else */\n"
               "#ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}");

  // Keep the current level if there is an empty line between the comment and
  // the preprocessor directive.
  verifyFormat("void f() {\n"
               "  int i;\n"
               "  /* comment */\n"
               "\n"
               "#ifdef A\n"
               "  int j;\n"
               "}",
               "void f() {\n"
               "  int i;\n"
               "/* comment */\n"
               "\n"
               "#ifdef A\n"
               "  int j;\n"
               "}");

  verifyFormat("void f() {\n"
               "  int i;\n"
               "  return i;\n"
               "}\n"
               "// comment\n"
               "\n"
               "#ifdef A\n"
               "int i;\n"
               "#endif // A",
               "void f() {\n"
               "   int i;\n"
               "  return i;\n"
               "}\n"
               "// comment\n"
               "\n"
               "#ifdef A\n"
               "int i;\n"
               "#endif // A");

  verifyFormat("int f(int i) {\n"
               "  if (true) {\n"
               "    ++i;\n"
               "  }\n"
               "  // comment\n"
               "\n"
               "#ifdef A\n"
               "  int j;\n"
               "#endif\n"
               "}",
               "int f(int i) {\n"
               "   if (true) {\n"
               "    ++i;\n"
               "  }\n"
               "  // comment\n"
               "\n"
               "#ifdef A\n"
               "  int j;\n"
               "#endif\n"
               "}");

  verifyFormat("int f(int i) {\n"
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
               "int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               "// comment in else\n"
               "\n"
               "#ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}");

  verifyFormat("int f(int i) {\n"
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
               "int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               "/* comment in else */\n"
               "\n"
               "#ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}");

  // Align with the preprocessor directive if the comment was originally aligned
  // with the preprocessor directive and there is no newline between the comment
  // and the preprocessor directive.
  verifyNoChange("void f() {\n"
                 "  int i;\n"
                 "/* comment */\n"
                 "#ifdef A\n"
                 "  int j;\n"
                 "}");

  verifyFormat("int f(int i) {\n"
               "  if (true) {\n"
               "    ++i;\n"
               "  }\n"
               "// comment\n"
               "#ifdef A\n"
               "  int j;\n"
               "#endif\n"
               "}",
               "int f(int i) {\n"
               "   if (true) {\n"
               "    ++i;\n"
               "  }\n"
               "// comment\n"
               "#ifdef A\n"
               "  int j;\n"
               "#endif\n"
               "}");

  verifyFormat("int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               "// comment in else\n"
               "#ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}",
               "int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               " // comment in else\n"
               " #ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}");

  verifyFormat("int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               "/* comment in else */\n"
               "#ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}",
               "int f(int i) {\n"
               "  if (true) {\n"
               "    i++;\n"
               "  } else {\n"
               " /* comment in else */\n"
               " #ifdef A\n"
               "    j++;\n"
               "#endif\n"
               "  }\n"
               "}");

  constexpr StringRef Code("void func() {\n"
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
  const auto Style10 = getLLVMStyleWithColumns(10);
  const auto Style15 = getLLVMStyleWithColumns(15);
  const auto Style20 = getLLVMStyleWithColumns(20);

  // FIXME: Do we need to fix up the "  */" at the end?
  // It doesn't look like any of our current logic triggers this.
  verifyFormat("/* This is a long\n"
               " * comment that\n"
               " * doesn't fit on\n"
               " * one line.  */",
               "/* "
               "This is a long                                         "
               "comment that "
               "doesn't                                    "
               "fit on one line.  */",
               Style20);
  verifyFormat("/* a b c d\n"
               " * e f  g\n"
               " * h i j k\n"
               " */",
               "/* a b c d e f  g h i j k */", Style10);
  verifyFormat("/* a b c d\n"
               " * e f  g\n"
               " * h i j k\n"
               " */",
               "\\\n/* a b c d e f  g h i j k */", Style10);
  verifyFormat("/*\n"
               "This is a long\n"
               "comment that doesn't\n"
               "fit on one line.\n"
               "*/",
               "/*\n"
               "This is a long                                         "
               "comment that doesn't                                    "
               "fit on one line.                                      \n"
               "*/",
               Style20);
  verifyFormat("/*\n"
               " * This is a long\n"
               " * comment that\n"
               " * doesn't fit on\n"
               " * one line.\n"
               " */",
               "/*      \n"
               " * This is a long "
               "   comment that     "
               "   doesn't fit on   "
               "   one line.                                            \n"
               " */",
               Style20);
  verifyFormat("/*\n"
               " * This_is_a_comment_with_words_that_dont_fit_on_one_line\n"
               " * so_it_should_be_broken\n"
               " * wherever_a_space_occurs\n"
               " */",
               "/*\n"
               " * This_is_a_comment_with_words_that_dont_fit_on_one_line "
               "   so_it_should_be_broken "
               "   wherever_a_space_occurs                             \n"
               " */",
               Style20);
  verifyNoChange("/*\n"
                 " *    This_comment_can_not_be_broken_into_lines\n"
                 " */",
                 Style20);
  verifyFormat("{\n"
               "  /*\n"
               "  This is another\n"
               "  long comment that\n"
               "  doesn't fit on one\n"
               "  line    1234567890\n"
               "  */\n"
               "}",
               "{\n"
               "/*\n"
               "This is another     "
               "  long comment that "
               "  doesn't fit on one"
               "  line    1234567890\n"
               "*/\n"
               "}",
               Style20);
  verifyFormat("{\n"
               "  /*\n"
               "   * This        i s\n"
               "   * another comment\n"
               "   * t hat  doesn' t\n"
               "   * fit on one l i\n"
               "   * n e\n"
               "   */\n"
               "}",
               "{\n"
               "/*\n"
               " * This        i s"
               "   another comment"
               "   t hat  doesn' t"
               "   fit on one l i"
               "   n e\n"
               " */\n"
               "}",
               Style20);
  verifyFormat("/*\n"
               " * This is a long\n"
               " * comment that\n"
               " * doesn't fit on\n"
               " * one line\n"
               " */",
               "   /*\n"
               "    * This is a long comment that doesn't fit on one line\n"
               "    */",
               Style20);
  verifyFormat("{\n"
               "  if (something) /* This is a\n"
               "                    long\n"
               "                    comment */\n"
               "    ;\n"
               "}",
               "{\n"
               "  if (something) /* This is a long comment */\n"
               "    ;\n"
               "}",
               getLLVMStyleWithColumns(30));

  verifyFormat("/* A comment before\n"
               " * a macro\n"
               " * definition */\n"
               "#define a b",
               "/* A comment before a macro definition */\n"
               "#define a b",
               Style20);

  verifyFormat("/* some comment\n"
               " *   a comment that\n"
               " * we break another\n"
               " * comment we have\n"
               " * to break a left\n"
               " * comment\n"
               " */",
               "  /* some comment\n"
               "       *   a comment that we break\n"
               "   * another comment we have to break\n"
               "* a left comment\n"
               "   */",
               Style20);

  verifyFormat("/**\n"
               " * multiline block\n"
               " * comment\n"
               " *\n"
               " */",
               "/**\n"
               " * multiline block comment\n"
               " *\n"
               " */",
               Style20);

  // This reproduces a crashing bug where both adaptStartOfLine and
  // getCommentSplit were trying to wrap after the "/**".
  verifyFormat("/** multilineblockcommentwithnowrapopportunity */", Style20);

  verifyFormat("/*\n"
               "\n"
               "\n"
               "    */",
               "  /*       \n"
               "      \n"
               "               \n"
               "      */");

  verifyFormat("/* a a */", "/* a a            */", Style15);
  verifyFormat("/* a a bc  */", "/* a a            bc  */", Style15);
  verifyFormat("/* aaa aaa\n"
               " * aaaaa */",
               "/* aaa aaa aaaaa       */", Style15);
  verifyFormat("/* aaa aaa\n"
               " * aaaaa     */",
               "/* aaa aaa aaaaa     */", Style15);
}

TEST_F(FormatTestComments, SplitsLongLinesInCommentsInPreprocessor) {
  const auto Style20 = getLLVMStyleWithColumns(20);
  verifyFormat("#define X          \\\n"
               "  /*               \\\n"
               "   Test            \\\n"
               "   Macro comment   \\\n"
               "   with a long     \\\n"
               "   line            \\\n"
               "   */              \\\n"
               "  A + B",
               "#define X \\\n"
               "  /*\n"
               "   Test\n"
               "   Macro comment with a long  line\n"
               "   */ \\\n"
               "  A + B",
               Style20);
  verifyFormat("#define X          \\\n"
               "  /* Macro comment \\\n"
               "     with a long   \\\n"
               "     line */       \\\n"
               "  A + B",
               "#define X \\\n"
               "  /* Macro comment with a long\n"
               "     line */ \\\n"
               "  A + B",
               Style20);
  verifyFormat("#define X          \\\n"
               "  /* Macro comment \\\n"
               "   * with a long   \\\n"
               "   * line */       \\\n"
               "  A + B",
               "#define X \\\n"
               "  /* Macro comment with a long  line */ \\\n"
               "  A + B",
               Style20);
}

TEST_F(FormatTestComments, KeepsTrailingPPCommentsAndSectionCommentsSeparate) {
  verifyFormat("#ifdef A // line about A\n"
               "// section comment\n"
               "#endif");
  verifyFormat("#ifdef A // line 1 about A\n"
               "         // line 2 about A\n"
               "// section comment\n"
               "#endif");
  verifyFormat("#ifdef A // line 1 about A\n"
               "         // line 2 about A\n"
               "// section comment\n"
               "#endif",
               "#ifdef A // line 1 about A\n"
               "          // line 2 about A\n"
               "// section comment\n"
               "#endif");
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
               "}");
}

TEST_F(FormatTestComments, AlignsPPElseEndifComments) {
  const auto Style20 = getLLVMStyleWithColumns(20);
  verifyFormat("#if A\n"
               "#else  // A\n"
               "int iiii;\n"
               "#endif // B",
               Style20);
  verifyFormat("#if A\n"
               "#else  // A\n"
               "int iiii; // CC\n"
               "#endif // B",
               Style20);
  verifyNoChange("#if A\n"
                 "#else  // A1\n"
                 "       // A2\n"
                 "int ii;\n"
                 "#endif // B",
                 Style20);
}

TEST_F(FormatTestComments, CommentsInStaticInitializers) {
  verifyFormat(
      "static SomeType type = {aaaaaaaaaaaaaaaaaaaa, /* comment */\n"
      "                        aaaaaaaaaaaaaaaaaaaa /* comment */,\n"
      "                        /* comment */ aaaaaaaaaaaaaaaaaaaa,\n"
      "                        aaaaaaaaaaaaaaaaaaaa, // comment\n"
      "                        aaaaaaaaaaaaaaaaaaaa};",
      "static SomeType type = { aaaaaaaaaaaaaaaaaaaa  ,  /* comment */\n"
      "                   aaaaaaaaaaaaaaaaaaaa   /* comment */ ,\n"
      "                     /* comment */   aaaaaaaaaaaaaaaaaaaa ,\n"
      "              aaaaaaaaaaaaaaaaaaaa ,   // comment\n"
      "                  aaaaaaaaaaaaaaaaaaaa };");
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

  verifyFormat("S s = {\n"
               "    // Some comment\n"
               "    a,\n"
               "\n"
               "    // Comment after empty line\n"
               "    b}",
               "S s =    {\n"
               "      // Some comment\n"
               "  a,\n"
               "  \n"
               "     // Comment after empty line\n"
               "      b\n"
               "}");
  verifyFormat("S s = {\n"
               "    /* Some comment */\n"
               "    a,\n"
               "\n"
               "    /* Comment after empty line */\n"
               "    b}",
               "S s =    {\n"
               "      /* Some comment */\n"
               "  a,\n"
               "  \n"
               "     /* Comment after empty line */\n"
               "      b\n"
               "}");
  verifyFormat("const uint8_t aaaaaaaaaaaaaaaaaaaaaa[0] = {\n"
               "    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // comment\n"
               "    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // comment\n"
               "    0x00, 0x00, 0x00, 0x00};            // comment");
}

TEST_F(FormatTestComments, LineCommentsAfterRightBrace) {
  verifyFormat("if (true) { // comment about branch\n"
               "  // comment about f\n"
               "  f();\n"
               "}");
  verifyFormat("if (1) { // if line 1\n"
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
               "if (1) { // if line 1\n"
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
               "}");
  verifyFormat("do { // line 1\n"
               "     // line 2\n"
               "     // line 3\n"
               "  f();\n"
               "} while (true);",
               "do { // line 1\n"
               "     // line 2\n"
               "   // line 3\n"
               "  f();\n"
               "} while (true);");
  verifyFormat("while (a < b) { // line 1\n"
               "  // line 2\n"
               "  // line 3\n"
               "  f();\n"
               "}",
               "while (a < b) {// line 1\n"
               "  // line 2\n"
               "  // line 3\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTestComments, ReflowsComments) {
  const auto Style20 = getLLVMStyleWithColumns(20);
  const auto Style22 = getLLVMStyleWithColumns(22);

  // Break a long line and reflow with the full next line.
  verifyFormat("// long long long\n"
               "// long long",
               "// long long long long\n"
               "// long",
               Style20);

  // Keep the trailing newline while reflowing.
  verifyFormat("// long long long\n"
               "// long long",
               "// long long long long\n"
               "// long",
               Style20);

  // Break a long line and reflow with a part of the next line.
  verifyFormat("// long long long\n"
               "// long long\n"
               "// long_long",
               "// long long long long\n"
               "// long long_long",
               Style20);

  // Break but do not reflow if the first word from the next line is too long.
  verifyFormat("// long long long\n"
               "// long\n"
               "// long_long_long",
               "// long long long long\n"
               "// long_long_long",
               Style20);

  // Don't break or reflow short lines.
  verifyFormat("// long\n"
               "// long long long lo\n"
               "// long long long lo\n"
               "// long",
               Style20);

  // Keep prefixes and decorations while reflowing.
  verifyFormat("/// long long long\n"
               "/// long long",
               "/// long long long long\n"
               "/// long",
               Style20);
  verifyFormat("//! long long long\n"
               "//! long long",
               "//! long long long long\n"
               "//! long",
               Style20);
  verifyFormat("/* long long long\n"
               " * long long */",
               "/* long long long long\n"
               " * long */",
               Style20);
  verifyFormat("///< long long long\n"
               "///< long long",
               "///< long long long long\n"
               "///< long",
               Style20);
  verifyFormat("//!< long long long\n"
               "//!< long long",
               "//!< long long long long\n"
               "//!< long",
               Style20);

  // Don't bring leading whitespace up while reflowing.
  verifyFormat("/*  long long long\n"
               " * long long long\n"
               " */",
               "/*  long long long long\n"
               " *  long long\n"
               " */",
               Style20);

  // Reflow the last line of a block comment with its trailing '*/'.
  verifyFormat("/* long long long\n"
               "   long long */",
               "/* long long long long\n"
               "   long */",
               Style20);

  // Reflow two short lines; keep the postfix of the last one.
  verifyFormat("/* long long long\n"
               " * long long long */",
               "/* long long long long\n"
               " * long\n"
               " * long */",
               Style20);

  // Put the postfix of the last short reflow line on a newline if it doesn't
  // fit.
  verifyFormat("/* long long long\n"
               " * long long longg\n"
               " */",
               "/* long long long long\n"
               " * long\n"
               " * longg */",
               Style20);

  // Reflow lines with leading whitespace.
  verifyFormat("{\n"
               "  /*\n"
               "   * long long long\n"
               "   * long long long\n"
               "   * long long long\n"
               "   */\n"
               "}",
               "{\n"
               "/*\n"
               " * long long long long\n"
               " *   long\n"
               " * long long long long\n"
               " */\n"
               "}",
               Style20);

  // Break single line block comments that are first in the line with ' *'
  // decoration.
  verifyFormat("/* long long long\n"
               " * long */",
               "/* long long long long */", Style20);

  // Break single line block comment that are not first in the line with '  '
  // decoration.
  verifyFormat("int i; /* long long\n"
               "          long */",
               "int i; /* long long long */", Style20);

  // Reflow a line that goes just over the column limit.
  verifyFormat("// long long long\n"
               "// lon long",
               "// long long long lon\n"
               "// long",
               Style20);

  // Stop reflowing if the next line has a different indentation than the
  // previous line.
  verifyFormat("// long long long\n"
               "// long\n"
               "//  long long\n"
               "//  long",
               "// long long long long\n"
               "//  long long\n"
               "//  long",
               Style20);

  // Reflow into the last part of a really long line that has been broken into
  // multiple lines.
  verifyFormat("// long long long\n"
               "// long long long\n"
               "// long long long",
               "// long long long long long long long long\n"
               "// long",
               Style20);

  // Break the first line, then reflow the beginning of the second and third
  // line up.
  verifyFormat("// long long long\n"
               "// lon1 lon2 lon2\n"
               "// lon2 lon3 lon3",
               "// long long long lon1\n"
               "// lon2 lon2 lon2\n"
               "// lon3 lon3",
               Style20);

  // Reflow the beginning of the second line, then break the rest.
  verifyFormat("// long long long\n"
               "// lon1 lon2 lon2\n"
               "// lon2 lon2 lon2\n"
               "// lon3",
               "// long long long lon1\n"
               "// lon2 lon2 lon2 lon2 lon2 lon3",
               Style20);

  // Shrink the first line, then reflow the second line up.
  verifyFormat("// long long long",
               "// long              long\n"
               "// long",
               Style20);

  // Don't shrink leading whitespace.
  verifyNoChange("int i; ///           a", Style20);

  // Shrink trailing whitespace if there is no postfix and reflow.
  verifyFormat("// long long long\n"
               "// long long",
               "// long long long long    \n"
               "// long",
               Style20);

  // Shrink trailing whitespace to a single one if there is postfix.
  verifyFormat("/* long long long */", "/* long long long     */", Style20);

  // Break a block comment postfix if exceeding the line limit.
  verifyFormat("/*               long\n"
               " */",
               "/*               long */", Style20);

  // Reflow indented comments.
  verifyFormat("{\n"
               "  // long long long\n"
               "  // long long\n"
               "  int i; /* long lon\n"
               "            g long\n"
               "          */\n"
               "}",
               "{\n"
               "  // long long long long\n"
               "  // long\n"
               "  int i; /* long lon g\n"
               "            long */\n"
               "}",
               Style20);

  // Don't realign trailing comments after reflow has happened.
  verifyFormat("// long long long\n"
               "// long long\n"
               "long i; // long",
               "// long long long long\n"
               "// long\n"
               "long i; // long",
               Style20);
  verifyFormat("// long long long\n"
               "// longng long long\n"
               "// long lo",
               "// long long long longng\n"
               "// long long long\n"
               "// lo",
               Style20);

  // Reflow lines after a broken line.
  verifyFormat("int a; // Trailing\n"
               "       // comment on\n"
               "       // 2 or 3\n"
               "       // lines.",
               "int a; // Trailing comment\n"
               "       // on 2\n"
               "       // or 3\n"
               "       // lines.",
               Style20);
  verifyFormat("/// This long line\n"
               "/// gets reflown.",
               "/// This long line gets\n"
               "/// reflown.",
               Style20);
  verifyFormat("//! This long line\n"
               "//! gets reflown.",
               " //! This long line gets\n"
               " //! reflown.",
               Style20);
  verifyFormat("/* This long line\n"
               " * gets reflown.\n"
               " */",
               "/* This long line gets\n"
               " * reflown.\n"
               " */",
               Style20);

  // Reflow after indentation makes a line too long.
  verifyFormat("{\n"
               "  // long long long\n"
               "  // lo long\n"
               "}",
               "{\n"
               "// long long long lo\n"
               "// long\n"
               "}",
               Style20);

  // Break and reflow multiple lines.
  verifyFormat("/*\n"
               " * Reflow the end of\n"
               " * line by 11 22 33\n"
               " * 4.\n"
               " */",
               "/*\n"
               " * Reflow the end of line\n"
               " * by\n"
               " * 11\n"
               " * 22\n"
               " * 33\n"
               " * 4.\n"
               " */",
               Style20);
  verifyFormat("/// First line gets\n"
               "/// broken. Second\n"
               "/// line gets\n"
               "/// reflown and\n"
               "/// broken. Third\n"
               "/// gets reflown.",
               "/// First line gets broken.\n"
               "/// Second line gets reflown and broken.\n"
               "/// Third gets reflown.",
               Style20);
  verifyFormat("int i; // first long\n"
               "       // long snd\n"
               "       // long.",
               "int i; // first long long\n"
               "       // snd long.",
               Style20);
  verifyFormat("{\n"
               "  // first long line\n"
               "  // line second\n"
               "  // long line line\n"
               "  // third long line\n"
               "  // line\n"
               "}",
               "{\n"
               "  // first long line line\n"
               "  // second long line line\n"
               "  // third long line line\n"
               "}",
               Style20);
  verifyFormat("int i; /* first line\n"
               "        * second\n"
               "        * line third\n"
               "        * line\n"
               "        */",
               "int i; /* first line\n"
               "        * second line\n"
               "        * third line\n"
               "        */",
               Style20);

  // Reflow the last two lines of a section that starts with a line having
  // different indentation.
  verifyFormat("//     long\n"
               "// long long long\n"
               "// long long",
               "//     long\n"
               "// long long long long\n"
               "// long",
               Style20);

  // Keep the block comment endling '*/' while reflowing.
  verifyFormat("/* Long long long\n"
               " * line short */",
               "/* Long long long line\n"
               " * short */",
               Style20);

  // Don't reflow between separate blocks of comments.
  verifyFormat("/* First comment\n"
               " * block will */\n"
               "/* Snd\n"
               " */",
               "/* First comment block\n"
               " * will */\n"
               "/* Snd\n"
               " */",
               Style20);

  // Don't reflow across blank comment lines.
  verifyFormat("int i; // This long\n"
               "       // line gets\n"
               "       // broken.\n"
               "       //\n"
               "       // keep.",
               "int i; // This long line gets broken.\n"
               "       //  \n"
               "       // keep.",
               Style20);
  verifyFormat("{\n"
               "  /// long long long\n"
               "  /// long long\n"
               "  ///\n"
               "  /// long\n"
               "}",
               "{\n"
               "  /// long long long long\n"
               "  /// long\n"
               "  ///\n"
               "  /// long\n"
               "}",
               Style20);
  verifyFormat("//! long long long\n"
               "//! long\n"
               "\n"
               "//! long",
               "//! long long long long\n"
               "\n"
               "//! long",
               Style20);
  verifyFormat("/* long long long\n"
               "   long\n"
               "\n"
               "   long */",
               "/* long long long long\n"
               "\n"
               "   long */",
               Style20);
  verifyFormat("/* long long long\n"
               " * long\n"
               " *\n"
               " * long */",
               "/* long long long long\n"
               " *\n"
               " * long */",
               Style20);

  // Don't reflow lines having content that is a single character.
  verifyFormat("// long long long\n"
               "// long\n"
               "// l",
               "// long long long long\n"
               "// l",
               Style20);

  // Don't reflow lines starting with two punctuation characters.
  verifyFormat("// long long long\n"
               "// long\n"
               "// ... --- ...",
               "// long long long long\n"
               "// ... --- ...",
               Style20);

  // Don't reflow lines starting with '@'.
  verifyFormat("// long long long\n"
               "// long\n"
               "// @param arg",
               "// long long long long\n"
               "// @param arg",
               Style20);

  // Don't reflow lines starting with '\'.
  verifyFormat("// long long long\n"
               "// long\n"
               "// \\param arg",
               "// long long long long\n"
               "// \\param arg",
               Style20);

  // Don't reflow lines starting with 'TODO'.
  verifyFormat("// long long long\n"
               "// long\n"
               "// TODO: long",
               "// long long long long\n"
               "// TODO: long",
               Style20);

  // Don't reflow lines starting with 'FIXME'.
  verifyFormat("// long long long\n"
               "// long\n"
               "// FIXME: long",
               "// long long long long\n"
               "// FIXME: long",
               Style20);

  // Don't reflow lines starting with 'XXX'.
  verifyFormat("// long long long\n"
               "// long\n"
               "// XXX: long",
               "// long long long long\n"
               "// XXX: long",
               Style20);

  // Don't reflow comment pragmas.
  verifyFormat("// long long long\n"
               "// long\n"
               "// IWYU pragma:",
               "// long long long long\n"
               "// IWYU pragma:",
               Style20);
  verifyFormat("/* long long long\n"
               " * long\n"
               " * IWYU pragma:\n"
               " */",
               "/* long long long long\n"
               " * IWYU pragma:\n"
               " */",
               Style20);

  // Reflow lines that have a non-punctuation character among their first 2
  // characters.
  verifyFormat("// long long long\n"
               "// long 'long'",
               "// long long long long\n"
               "// 'long'",
               Style20);

  // Don't reflow between separate blocks of comments.
  verifyFormat("/* First comment\n"
               " * block will */\n"
               "/* Snd\n"
               " */",
               "/* First comment block\n"
               " * will */\n"
               "/* Snd\n"
               " */",
               Style20);

  // Don't reflow lines having different indentation.
  verifyFormat("// long long long\n"
               "// long\n"
               "//  long",
               "// long long long long\n"
               "//  long",
               Style20);

  // Don't reflow separate bullets in list
  verifyFormat("// - long long long\n"
               "// long\n"
               "// - long",
               "// - long long long long\n"
               "// - long",
               Style20);
  verifyFormat("// * long long long\n"
               "// long\n"
               "// * long",
               "// * long long long long\n"
               "// * long",
               Style20);
  verifyFormat("// + long long long\n"
               "// long\n"
               "// + long",
               "// + long long long long\n"
               "// + long",
               Style20);
  verifyFormat("// 1. long long long\n"
               "// long\n"
               "// 2. long",
               "// 1. long long long long\n"
               "// 2. long",
               Style20);
  verifyFormat("// -# long long long\n"
               "// long\n"
               "// -# long",
               "// -# long long long long\n"
               "// -# long",
               Style20);

  verifyFormat("// - long long long\n"
               "// long long long\n"
               "// - long",
               "// - long long long long\n"
               "// long long\n"
               "// - long",
               Style20);
  verifyFormat("// - long long long\n"
               "// long long long\n"
               "// long\n"
               "// - long",
               "// - long long long long\n"
               "// long long long\n"
               "// - long",
               Style20);

  // Large number (>2 digits) are not list items
  verifyFormat("// long long long\n"
               "// long 1024. long.",
               "// long long long long\n"
               "// 1024. long.",
               Style20);

  // Do not break before number, to avoid introducing a non-reflowable doxygen
  // list item.
  verifyFormat("// long long\n"
               "// long 10. long.",
               "// long long long 10.\n"
               "// long.",
               Style20);

  // Don't break or reflow after implicit string literals.
  verifyFormat("#include <t> // l l l\n"
               "             // l",
               Style20);

  // Don't break or reflow comments on import lines.
  verifyNoChange("#include \"t\" /* l l l\n"
                 "                * l */",
                 Style20);

  // Don't reflow between different trailing comment sections.
  verifyFormat("int i; // long long\n"
               "       // long\n"
               "int j; // long long\n"
               "       // long",
               "int i; // long long long\n"
               "int j; // long long long",
               Style20);

  // Don't reflow if the first word on the next line is longer than the
  // available space at current line.
  verifyFormat("int i; // trigger\n"
               "       // reflow\n"
               "       // longsec",
               "int i; // trigger reflow\n"
               "       // longsec",
               Style20);

  // Simple case that correctly handles reflow in parameter lists.
  verifyFormat("a = f(/* looooooooong\n"
               "       * long long\n"
               "       */\n"
               "      a);",
               "a = f(/* looooooooong long\n* long\n*/ a);", Style22);
  // Tricky case that has fewer lines if we reflow the comment, ending up with
  // fewer lines.
  verifyFormat("a = f(/* loooooong\n"
               "       * long long\n"
               "       */\n"
               "      a);",
               "a = f(/* loooooong long\n* long\n*/ a);", Style22);

  // Keep empty comment lines.
  verifyFormat("/**/", " /**/", Style20);
  verifyFormat("/* */", " /* */", Style20);
  verifyFormat("/*  */", " /*  */", Style20);
  verifyFormat("//", " //  ", Style20);
  verifyFormat("///", " ///  ", Style20);
}

TEST_F(FormatTestComments, ReflowsCommentsPrecise) {
  auto Style = getLLVMStyleWithColumns(20);

  // FIXME: This assumes we do not continue compressing whitespace once we are
  // in reflow mode. Consider compressing whitespace.

  // Test that we stop reflowing precisely at the column limit.
  // After reflowing, "// reflows into   foo" does not fit the column limit,
  // so we compress the whitespace.
  verifyFormat("// some text that\n"
               "// reflows into foo",
               "// some text that reflows\n"
               "// into   foo",
               Style);

  Style.ColumnLimit = 21;

  // Given one more column, "// reflows into   foo" does fit the limit, so we
  // do not compress the whitespace.
  verifyFormat("// some text that\n"
               "// reflows into   foo",
               "// some text that reflows\n"
               "// into   foo",
               Style);

  // Make sure that we correctly account for the space added in the reflow case
  // when making the reflowing decision.
  // First, when the next line ends precisely one column over the limit, do not
  // reflow.
  verifyFormat("// some text that\n"
               "// reflows\n"
               "// into1234567",
               "// some text that reflows\n"
               "// into1234567",
               Style);

  // Secondly, when the next line ends later, but the first word in that line
  // is precisely one column over the limit, do not reflow.
  verifyFormat("// some text that\n"
               "// reflows\n"
               "// into1234567 f",
               "// some text that reflows\n"
               "// into1234567 f",
               Style);
}

TEST_F(FormatTestComments, ReflowsCommentsWithExtraWhitespace) {
  const auto Style16 = getLLVMStyleWithColumns(16);

  // Baseline.
  verifyFormat("// some text\n"
               "// that re flows",
               "// some text that\n"
               "// re flows",
               Style16);
  verifyFormat("// some text\n"
               "// that re flows",
               "// some text that\n"
               "// re    flows",
               Style16);
  verifyFormat("/* some text\n"
               " * that re flows\n"
               " */",
               "/* some text that\n"
               "*      re       flows\n"
               "*/",
               Style16);
  // FIXME: We do not reflow if the indent of two subsequent lines differs;
  // given that this is different behavior from block comments, do we want
  // to keep this?
  verifyFormat("// some text\n"
               "// that\n"
               "//     re flows",
               "// some text that\n"
               "//     re       flows",
               Style16);
  // Space within parts of a line that fit.
  // FIXME: Use the earliest possible split while reflowing to compress the
  // whitespace within the line.
  verifyFormat("// some text that\n"
               "// does re   flow\n"
               "// more  here",
               "// some text that does\n"
               "// re   flow  more  here",
               getLLVMStyleWithColumns(21));
}

TEST_F(FormatTestComments, IgnoresIf0Contents) {
  verifyFormat("#if 0\n"
               "}{)(&*(^%%#%@! fsadj f;ldjs ,:;| <<<>>>][)(][\n"
               "#endif\n"
               "void f() {}",
               "#if 0\n"
               "}{)(&*(^%%#%@! fsadj f;ldjs ,:;| <<<>>>][)(][\n"
               "#endif\n"
               "void f(  ) {  }");
  verifyFormat("#if false\n"
               "void f(  ) {  }\n"
               "#endif\n"
               "void g() {}",
               "#if false\n"
               "void f(  ) {  }\n"
               "#endif\n"
               "void g(  ) {  }");
  verifyFormat("enum E {\n"
               "  One,\n"
               "  Two,\n"
               "#if 0\n"
               "Three,\n"
               "      Four,\n"
               "#endif\n"
               "  Five\n"
               "};",
               "enum E {\n"
               "  One,Two,\n"
               "#if 0\n"
               "Three,\n"
               "      Four,\n"
               "#endif\n"
               "  Five};");
  verifyFormat("enum F {\n"
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
               "enum F {\n"
               "One,\n"
               "#if 1\n"
               "Two,\n"
               "#if 0\n"
               "Three,\n"
               "      Four,\n"
               "#endif\n"
               "Five\n"
               "#endif\n"
               "};");
  verifyFormat("enum G {\n"
               "  One,\n"
               "#if 0\n"
               "Two,\n"
               "#else\n"
               "  Three,\n"
               "#endif\n"
               "  Four\n"
               "};",
               "enum G {\n"
               "One,\n"
               "#if 0\n"
               "Two,\n"
               "#else\n"
               "Three,\n"
               "#endif\n"
               "Four\n"
               "};");
  verifyFormat("enum H {\n"
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
               "enum H {\n"
               "One,\n"
               "#if 0\n"
               "#ifdef Q\n"
               "Two,\n"
               "#else\n"
               "Three,\n"
               "#endif\n"
               "#endif\n"
               "Four\n"
               "};");
  verifyFormat("enum I {\n"
               "  One,\n"
               "#if /* test */ 0 || 1\n"
               "Two,\n"
               "Three,\n"
               "#endif\n"
               "  Four\n"
               "};",
               "enum I {\n"
               "One,\n"
               "#if /* test */ 0 || 1\n"
               "Two,\n"
               "Three,\n"
               "#endif\n"
               "Four\n"
               "};");
  verifyFormat("enum J {\n"
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
               "enum J {\n"
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
               "};");

  // Ignore stuff in SWIG-blocks.
  verifyFormat("#ifdef SWIG\n"
               "}{)(&*(^%%#%@! fsadj f;ldjs ,:;| <<<>>>][)(][\n"
               "#endif\n"
               "void f() {}",
               "#ifdef SWIG\n"
               "}{)(&*(^%%#%@! fsadj f;ldjs ,:;| <<<>>>][)(][\n"
               "#endif\n"
               "void f(  ) {  }");
  verifyFormat("#ifndef SWIG\n"
               "void f() {}\n"
               "#endif",
               "#ifndef SWIG\n"
               "void f(      ) {       }\n"
               "#endif");
}

TEST_F(FormatTestComments, DontCrashOnBlockComments) {
  verifyFormat(
      "int xxxxxxxxx; /* "
      "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\n"
      "zzzzzz\n"
      "0*/",
      "int xxxxxxxxx;                          /* "
      "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy zzzzzz\n"
      "0*/");
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
  const auto Style10 = getLLVMStyleWithColumns(10);
  const auto Style15 = getLLVMStyleWithColumns(15);

  verifyFormat("/* */ /* */ /* */\n/* */ /* */ /* */",
               "/* *//* */  /* */\n/* *//* */  /* */");
  verifyFormat("/* */ a /* */ b;", "  /* */  a/* */  b;");
  verifyFormat("#define A /*123*/ \\\n"
               "  b\n"
               "/* */\n"
               "someCall(\n"
               "    parameter);",
               "#define A /*123*/ b\n"
               "/* */\n"
               "someCall(parameter);",
               Style15);

  verifyFormat("#define A\n"
               "/* */ someCall(\n"
               "    parameter);",
               "#define A\n"
               "/* */someCall(parameter);",
               Style15);
  verifyNoChange("/*\n**\n*/");
  verifyFormat("/*\n"
               " *\n"
               " * aaaaaa\n"
               " * aaaaaa\n"
               " */",
               "/*\n"
               "*\n"
               " * aaaaaa aaaaaa\n"
               "*/",
               Style10);
  verifyFormat("/*\n"
               "**\n"
               "* aaaaaa\n"
               "* aaaaaa\n"
               "*/",
               "/*\n"
               "**\n"
               "* aaaaaa aaaaaa\n"
               "*/",
               Style10);
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    /* line 1\n"
               "       bbbbbbbbbbbb */\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               "int aaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    /* line 1\n"
               "       bbbbbbbbbbbb */ bbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               getLLVMStyleWithColumns(50));

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = FormatStyle::BPPS_OnePerLine;
  verifyFormat("someFunction(1, /* comment 1 */\n"
               "             2, /* comment 2 */\n"
               "             3, /* comment 3 */\n"
               "             aaaa,\n"
               "             bbbb);",
               "someFunction (1,   /* comment 1 */\n"
               "                2, /* comment 2 */  \n"
               "               3,   /* comment 3 */\n"
               "aaaa, bbbb );",
               NoBinPacking);
  verifyFormat(
      "bool aaaaaaaaaaaaa = /* comment: */ aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "bool aaaaaaaaaaaaa = /* trailing comment */\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaaa;",
      "bool       aaaaaaaaaaaaa =       /* trailing comment */\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaa||aaaaaaaaaaaaaaaaaaaaaaaaa    ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa   || aaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; /* comment */\n"
               "int bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;   /* comment */\n"
               "int cccccccccccccccccccccccccccccc;       /* comment */",
               "int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; /* comment */\n"
               "int      bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; /* comment */\n"
               "int    cccccccccccccccccccccccccccccc;  /* comment */");

  verifyFormat("void f(int * /* unused */) {}");

  verifyNoChange("/*\n"
                 " **\n"
                 " */");
  verifyNoChange("/*\n"
                 " *q\n"
                 " */");
  verifyNoChange("/*\n"
                 " * q\n"
                 " */");
  verifyNoChange("/*\n"
                 " **/");
  verifyNoChange("/*\n"
                 " ***/");
}

TEST_F(FormatTestComments, BlockCommentsInMacros) {
  const auto Style20 = getLLVMStyleWithColumns(20);
  verifyFormat("#define A          \\\n"
               "  {                \\\n"
               "    /* one line */ \\\n"
               "    someCall();",
               "#define A {        \\\n"
               "  /* one line */   \\\n"
               "  someCall();",
               Style20);
  verifyFormat("#define A          \\\n"
               "  {                \\\n"
               "    /* previous */ \\\n"
               "    /* one line */ \\\n"
               "    someCall();",
               "#define A {        \\\n"
               "  /* previous */   \\\n"
               "  /* one line */   \\\n"
               "  someCall();",
               Style20);
}

TEST_F(FormatTestComments, BlockCommentsAtEndOfLine) {
  const auto Style15 = getLLVMStyleWithColumns(15);
  verifyFormat("a = {\n"
               "    1111 /*    */\n"
               "};",
               "a = {1111 /*    */\n"
               "};",
               Style15);
  verifyFormat("a = {\n"
               "    1111 /*      */\n"
               "};",
               "a = {1111 /*      */\n"
               "};",
               Style15);
  verifyFormat("a = {\n"
               "    1111 /*      a\n"
               "          */\n"
               "};",
               "a = {1111 /*      a */\n"
               "};",
               Style15);
}

TEST_F(FormatTestComments, BreaksAfterMultilineBlockCommentsInParamLists) {
  const auto Style15 = getLLVMStyleWithColumns(15);
  const auto Style16 = getLLVMStyleWithColumns(16);

  verifyFormat("a = f(/* long\n"
               "         long */\n"
               "      a);",
               "a = f(/* long long */ a);", Style16);
  verifyFormat("a = f(\n"
               "    /* long\n"
               "       long */\n"
               "    a);",
               "a = f(/* long long */ a);", Style15);

  verifyFormat("a = f(/* long\n"
               "         long\n"
               "       */\n"
               "      a);",
               "a = f(/* long\n"
               "         long\n"
               "       */a);",
               Style16);

  verifyFormat("a = f(/* long\n"
               "         long\n"
               "       */\n"
               "      a);",
               "a = f(/* long\n"
               "         long\n"
               "       */ a);",
               Style16);

  verifyFormat("a = f(/* long\n"
               "         long\n"
               "       */\n"
               "      (1 + 1));",
               "a = f(/* long\n"
               "         long\n"
               "       */ (1 + 1));",
               Style16);

  verifyFormat("a = f(a,\n"
               "      /* long\n"
               "         long */\n"
               "      b);",
               "a = f(a, /* long long */ b);", Style16);

  verifyFormat("a = f(\n"
               "    a,\n"
               "    /* long\n"
               "       long */\n"
               "    b);",
               "a = f(a, /* long long */ b);", Style15);

  verifyFormat("a = f(a,\n"
               "      /* long\n"
               "         long */\n"
               "      (1 + 1));",
               "a = f(a, /* long long */ (1 + 1));", Style16);
  verifyFormat("a = f(\n"
               "    a,\n"
               "    /* long\n"
               "       long */\n"
               "    (1 + 1));",
               "a = f(a, /* long long */ (1 + 1));", Style15);
}

TEST_F(FormatTestComments, IndentLineCommentsInStartOfBlockAtEndOfFile) {
  verifyFormat("{\n"
               "  // a\n"
               "  // b");
}

TEST_F(FormatTestComments, AlignTrailingComments) {
  const auto Style15 = getLLVMStyleWithColumns(15);
  const auto Style40 = getLLVMStyleWithColumns(40);

  verifyFormat("#define MACRO(V)                       \\\n"
               "  V(Rt2) /* one more char */           \\\n"
               "  V(Rs)  /* than here  */              \\\n"
               "/* comment 3 */\n",
               "#define MACRO(V)\\\n"
               "V(Rt2)  /* one more char */ \\\n"
               "V(Rs) /* than here  */    \\\n"
               "/* comment 3 */\n",
               Style40);
  verifyFormat("int i = f(abc, // line 1\n"
               "          d,   // line 2\n"
               "               // line 3\n"
               "          b);",
               "int i = f(abc, // line 1\n"
               "          d, // line 2\n"
               "             // line 3\n"
               "          b);",
               Style40);

  // Align newly broken trailing comments.
  verifyFormat("int ab; // line\n"
               "int a;  // long\n"
               "        // long",
               "int ab; // line\n"
               "int a; // long long",
               Style15);
  verifyFormat("int ab; // line\n"
               "int a;  // long\n"
               "        // long\n"
               "        // long",
               "int ab; // line\n"
               "int a; // long long\n"
               "       // long",
               Style15);
  verifyFormat("int ab; // line\n"
               "int a;  // long\n"
               "        // long\n"
               "pt c;   // long",
               "int ab; // line\n"
               "int a; // long long\n"
               "pt c; // long",
               Style15);
  verifyFormat("int ab; // line\n"
               "int a;  // long\n"
               "        // long\n"
               "\n"
               "// long",
               "int ab; // line\n"
               "int a; // long long\n"
               "\n"
               "// long",
               Style15);

  // Don't align newly broken trailing comments if that would put them over the
  // column limit.
  verifyFormat("int i, j; // line 1\n"
               "int k; // line longg\n"
               "       // long",
               "int i, j; // line 1\n"
               "int k; // line longg long",
               getLLVMStyleWithColumns(20));

  // Always align if ColumnLimit = 0
  verifyFormat("int i, j; // line 1\n"
               "int k;    // line longg long",
               "int i, j; // line 1\n"
               "int k; // line longg long",
               getLLVMStyleWithColumns(0));

  // Align comment line sections aligned with the next token with the next
  // token.
  verifyFormat("class A {\n"
               "public: // public comment\n"
               "  // comment about a\n"
               "  int a;\n"
               "};",
               Style40);
  verifyFormat("class A {\n"
               "public: // public comment 1\n"
               "        // public comment 2\n"
               "  // comment 1 about a\n"
               "  // comment 2 about a\n"
               "  int a;\n"
               "};",
               "class A {\n"
               "public: // public comment 1\n"
               "   // public comment 2\n"
               "  // comment 1 about a\n"
               "  // comment 2 about a\n"
               "  int a;\n"
               "};",
               Style40);
  verifyFormat("int f(int n) { // comment line 1 on f\n"
               "               // comment line 2 on f\n"
               "  // comment line 1 before return\n"
               "  // comment line 2 before return\n"
               "  return n; // comment line 1 on return\n"
               "            // comment line 2 on return\n"
               "  // comment line 1 after return\n"
               "}",
               "int f(int n) { // comment line 1 on f\n"
               "   // comment line 2 on f\n"
               "  // comment line 1 before return\n"
               "  // comment line 2 before return\n"
               "  return n; // comment line 1 on return\n"
               "   // comment line 2 on return\n"
               "  // comment line 1 after return\n"
               "}",
               Style40);
  verifyFormat("int f(int n) {\n"
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
               "int f(int n) {\n"
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
               "}");

  // If all the lines in a sequence of line comments are aligned with the next
  // token, the first line belongs to the previous token and the other lines
  // belong to the next token.
  verifyFormat("int a; // line about a\n"
               "long b;",
               "int a; // line about a\n"
               "       long b;");
  verifyFormat("int a; // line about a\n"
               "// line about b\n"
               "long b;",
               "int a; // line about a\n"
               "       // line about b\n"
               "       long b;");
  verifyFormat("int a; // line about a\n"
               "// line 1 about b\n"
               "// line 2 about b\n"
               "long b;",
               "int a; // line about a\n"
               "       // line 1 about b\n"
               "       // line 2 about b\n"
               "       long b;");

  // Checks an edge case in preprocessor handling.
  // These comments should *not* be aligned
  verifyFormat("#if FOO\n"
               "#else\n"
               "long a; // Line about a\n"
               "#endif\n"
               "#if BAR\n"
               "#else\n"
               "long b_long_name; // Line about b\n"
               "#endif",
               "#if FOO\n"
               "#else\n"
               "long a;           // Line about a\n" // Previous (bad) behavior
               "#endif\n"
               "#if BAR\n"
               "#else\n"
               "long b_long_name; // Line about b\n"
               "#endif");

  // bug 47589
  verifyFormat("namespace m {\n\n"
               "#define FOO_GLOBAL 0      // Global scope.\n"
               "#define FOO_LINKLOCAL 1   // Link-local scope.\n"
               "#define FOO_SITELOCAL 2   // Site-local scope (deprecated).\n"
               "#define FOO_UNIQUELOCAL 3 // Unique local\n"
               "#define FOO_NODELOCAL 4   // Loopback\n\n"
               "} // namespace m",
               "namespace m {\n\n"
               "#define FOO_GLOBAL 0   // Global scope.\n"
               "#define FOO_LINKLOCAL 1  // Link-local scope.\n"
               "#define FOO_SITELOCAL 2  // Site-local scope (deprecated).\n"
               "#define FOO_UNIQUELOCAL 3 // Unique local\n"
               "#define FOO_NODELOCAL 4  // Loopback\n\n"
               "} // namespace m");

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
  verifyFormat("#include \"a.h\" // comment\n"
               "\n"
               "\n"
               "\n"
               "#include \"ab.h\"      // comment\n"
               "\n"
               "\n"
               "#include \"abcdefg.h\" // comment",
               "#include \"a.h\" // comment\n"
               "\n"
               "\n"
               "\n"
               "#include \"ab.h\" // comment\n"
               "\n"
               "\n"
               "#include \"abcdefg.h\" // comment",
               Style);

  Style.MaxEmptyLinesToKeep = 1;
  Style.AlignTrailingComments.OverEmptyLines = 1;
  // End of testing OverEmptyLines

  Style.ColumnLimit = 15;
  verifyFormat("int ab; // line\n"
               "int a;  // long\n"
               "        // long\n"
               "\n"
               "        // long",
               "int ab; // line\n"
               "int a; // long long\n"
               "\n"
               "// long",
               Style);

  Style.ColumnLimit = 15;
  verifyFormat("int ab; // line\n"
               "\n"
               "int a;  // long\n"
               "        // long",
               "int ab; // line\n"
               "\n"
               "int a; // long long",
               Style);

  Style.ColumnLimit = 30;
  verifyFormat("int foo = 12345; // comment\n"
               "int bar =\n"
               "    1234;  // This is a very\n"
               "           // long comment\n"
               "           // which is wrapped\n"
               "           // arround.\n"
               "\n"
               "int x = 2; // Is this still\n"
               "           // aligned?",
               "int foo = 12345; // comment\n"
               "int bar = 1234; // This is a very long comment\n"
               "                // which is wrapped arround.\n"
               "\n"
               "int x = 2; // Is this still aligned?",
               Style);

  Style.ColumnLimit = 35;
  verifyFormat("int foo = 12345; // comment\n"
               "int bar =\n"
               "    1234; // This is a very long\n"
               "          // comment which is\n"
               "          // wrapped arround.\n"
               "\n"
               "int x =\n"
               "    2; // Is this still aligned?",
               "int foo = 12345; // comment\n"
               "int bar = 1234; // This is a very long comment\n"
               "                // which is wrapped arround.\n"
               "\n"
               "int x = 2; // Is this still aligned?",
               Style);

  Style.ColumnLimit = 40;
  verifyFormat("int foo = 12345; // comment\n"
               "int bar =\n"
               "    1234; // This is a very long comment\n"
               "          // which is wrapped arround.\n"
               "\n"
               "int x = 2; // Is this still aligned?",
               "int foo = 12345; // comment\n"
               "int bar = 1234; // This is a very long comment\n"
               "                // which is wrapped arround.\n"
               "\n"
               "int x = 2; // Is this still aligned?",
               Style);

  Style.ColumnLimit = 45;
  verifyFormat("int foo = 12345; // comment\n"
               "int bar =\n"
               "    1234;  // This is a very long comment\n"
               "           // which is wrapped arround.\n"
               "\n"
               "int x = 2; // Is this still aligned?",
               "int foo = 12345; // comment\n"
               "int bar = 1234; // This is a very long comment\n"
               "                // which is wrapped arround.\n"
               "\n"
               "int x = 2; // Is this still aligned?",
               Style);

  Style.ColumnLimit = 80;
  verifyFormat("int a; // line about a\n"
               "\n"
               "// line about b\n"
               "long b;",
               "int a; // line about a\n"
               "\n"
               "       // line about b\n"
               "       long b;",
               Style);

  Style.ColumnLimit = 80;
  verifyFormat("int a; // line about a\n"
               "\n"
               "// line 1 about b\n"
               "// line 2 about b\n"
               "long b;",
               "int a; // line about a\n"
               "\n"
               "       // line 1 about b\n"
               "       // line 2 about b\n"
               "       long b;",
               Style);
}

TEST_F(FormatTestComments, AlignTrailingCommentsLeave) {
  FormatStyle Style = getLLVMStyle();
  Style.AlignTrailingComments.Kind = FormatStyle::TCAS_Leave;

  verifyNoChange("int a;// do not touch\n"
                 "int b; // any comments\n"
                 "int c;  // comment\n"
                 "int d;   // comment",
                 Style);

  verifyNoChange("int a;   // do not touch\n"
                 "int b;  // any comments\n"
                 "int c; // comment\n"
                 "int d;// comment",
                 Style);

  verifyNoChange("// do not touch\n"
                 "int a;  // any comments\n"
                 "\n"
                 "   // comment\n"
                 "// comment\n"
                 "\n"
                 "// comment",
                 Style);

  verifyFormat("// do not touch\n"
               "int a;  // any comments\n"
               "\n"
               "   // comment\n"
               "// comment\n"
               "\n"
               "// comment",
               "// do not touch\n"
               "int a;  // any comments\n"
               "\n"
               "\n"
               "   // comment\n"
               "// comment\n"
               "\n"
               "\n"
               "// comment",
               Style);

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
  verifyNoChange("// do not touch\n"
                 "int a;  // any comments\n"
                 "\n"
                 "\n"
                 "   // comment\n"
                 "// comment\n"
                 "\n"
                 "// comment",
                 Style);
  Style.MaxEmptyLinesToKeep = 1;

  // Just format comments normally when leaving exceeds the column limit
  Style.ColumnLimit = 35;
  verifyFormat("int foo = 12345; // comment\n"
               "int bar =\n"
               "    1234; // This is a very long\n"
               "          // comment which is\n"
               "          // wrapped arround.",
               "int foo = 12345; // comment\n"
               "int bar = 1234;       // This is a very long comment\n"
               "          // which is wrapped arround.",
               Style);

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

  constexpr StringRef Input("namespace A {\n"
                            "  TESTSUITE(B) {\n"
                            "    namespace C {\n"
                            "      namespace D { //\n"
                            "      } // namespace D\n"
                            "      std::string Foo = Bar; // Comment\n"
                            "      std::string BazString = Baz;   // C2\n"
                            "    }          // namespace C\n"
                            "  }\n"
                            "} // NaMeSpAcE A");

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
  constexpr StringRef Code("namespace A {\n"
                           "  int Foo;\n"
                           "  int Bar;\n"
                           "}\n"
                           "// Comment");

  verifyFormat("namespace A {\n"
               "  int Foo;\n"
               "  int Bar;\n"
               "} // namespace A\n"
               "// Comment",
               Code, Style);

  Style.FixNamespaceComments = false;
  verifyFormat(Code, Style);
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

TEST_F(FormatTestComments, DontAlignOverPPDirective) {
  auto Style = getLLVMStyle();
  Style.AlignTrailingComments.AlignPPAndNotPP = false;

  verifyFormat("int i;    // Aligned\n"
               "int long; // with this\n"
               "#define FOO    // only aligned\n"
               "#define LOOONG // with other pp directives\n"
               "int loooong; // new alignment",
               "int i;//Aligned\n"
               "int long;//with this\n"
               "#define FOO //only aligned\n"
               "#define LOOONG //with other pp directives\n"
               "int loooong; //new alignment",
               Style);

  verifyFormat("#define A  // Comment\n"
               "#define AB // Comment",
               Style);

  Style.ColumnLimit = 30;
  verifyNoChange("#define A // Comment\n"
                 "          // Continued\n"
                 "int i = 0; // New Stuff\n"
                 "           // Continued\n"
                 "#define Func(X)              \\\n"
                 "  X();                       \\\n"
                 "  X(); // Comment\n"
                 "       // Continued\n"
                 "long loong = 1; // Dont align",
                 Style);

  verifyFormat("#define A   // Comment that\n"
               "            // would wrap\n"
               "#define FOO // For the\n"
               "            // alignment\n"
               "#define B   // Also\n"
               "            // aligned",
               "#define A // Comment that would wrap\n"
               "#define FOO // For the alignment\n"
               "#define B // Also\n"
               " // aligned",
               Style);

  Style.AlignTrailingComments.OverEmptyLines = 1;
  verifyNoChange("#define A // Comment\n"
                 "\n"
                 "          // Continued\n"
                 "int i = 0; // New Stuff\n"
                 "\n"
                 "           // Continued\n"
                 "#define Func(X)              \\\n"
                 "  X();                       \\\n"
                 "  X(); // Comment\n"
                 "\n"
                 "       // Continued\n"
                 "long loong = 1; // Dont align",
                 Style);
}

TEST_F(FormatTestComments, AlignsBlockCommentDecorations) {
  verifyFormat("/*\n"
               " */",
               "/*\n"
               "*/");
  verifyNoChange("/*\n"
                 " */");
  verifyFormat("/*\n"
               " */",
               "/*\n"
               "  */");

  // Align a single line.
  verifyFormat("/*\n"
               " * line */",
               "/*\n"
               "* line */");
  verifyNoChange("/*\n"
                 " * line */");
  verifyFormat("/*\n"
               " * line */",
               "/*\n"
               "  * line */");
  verifyFormat("/*\n"
               " * line */",
               "/*\n"
               "   * line */");
  verifyFormat("/**\n"
               " * line */",
               "/**\n"
               "* line */");
  verifyNoChange("/**\n"
                 " * line */");
  verifyFormat("/**\n"
               " * line */",
               "/**\n"
               "  * line */");
  verifyFormat("/**\n"
               " * line */",
               "/**\n"
               "   * line */");
  verifyFormat("/**\n"
               " * line */",
               "/**\n"
               "    * line */");

  // Align the end '*/' after a line.
  verifyFormat("/*\n"
               " * line\n"
               " */",
               "/*\n"
               "* line\n"
               "*/");
  verifyFormat("/*\n"
               " * line\n"
               " */",
               "/*\n"
               "   * line\n"
               "  */");
  verifyFormat("/*\n"
               " * line\n"
               " */",
               "/*\n"
               "  * line\n"
               "  */");

  // Align two lines.
  verifyNoChange("/* line 1\n"
                 " * line 2 */");
  verifyFormat("/* line 1\n"
               " * line 2 */",
               "/* line 1\n"
               "* line 2 */");
  verifyFormat("/* line 1\n"
               " * line 2 */",
               "/* line 1\n"
               "  * line 2 */");
  verifyFormat("/* line 1\n"
               " * line 2 */",
               "/* line 1\n"
               "   * line 2 */");
  verifyFormat("/* line 1\n"
               " * line 2 */",
               "/* line 1\n"
               "    * line 2 */");
  verifyFormat("int i; /* line 1\n"
               "        * line 2 */",
               "int i; /* line 1\n"
               "* line 2 */");
  verifyNoChange("int i; /* line 1\n"
                 "        * line 2 */");
  verifyFormat("int i; /* line 1\n"
               "        * line 2 */",
               "int i; /* line 1\n"
               "             * line 2 */");

  // Align several lines.
  verifyFormat("/* line 1\n"
               " * line 2\n"
               " * line 3 */",
               "/* line 1\n"
               " * line 2\n"
               "* line 3 */");
  verifyFormat("/* line 1\n"
               " * line 2\n"
               " * line 3 */",
               "/* line 1\n"
               "  * line 2\n"
               "* line 3 */");
  verifyFormat("/*\n"
               "** line 1\n"
               "** line 2\n"
               "*/",
               "/*\n"
               "** line 1\n"
               " ** line 2\n"
               "*/");

  // Align with different indent after the decorations.
  verifyFormat("/*\n"
               " * line 1\n"
               " *  line 2\n"
               " * line 3\n"
               " *   line 4\n"
               " */",
               "/*\n"
               "* line 1\n"
               "  *  line 2\n"
               "   * line 3\n"
               "*   line 4\n"
               "*/");

  // Align empty or blank lines.
  verifyFormat("/**\n"
               " *\n"
               " *\n"
               " *\n"
               " */",
               "/**\n"
               "*  \n"
               " * \n"
               "  *\n"
               "*/");

  // Align while breaking and reflowing.
  verifyFormat("/*\n"
               " * long long long\n"
               " * long long\n"
               " *\n"
               " * long */",
               "/*\n"
               " * long long long long\n"
               " * long\n"
               "  *\n"
               "* long */",
               getLLVMStyleWithColumns(20));
}

TEST_F(FormatTestComments, NoCrash_Bug34236) {
  // This is a test case from a crasher reported in:
  // https://bugs.llvm.org/show_bug.cgi?id=34236
  // Temporarily disable formatting for readability.
  // clang-format off
  verifyFormat(
"/*                                                                */ /*\n"
"                                                                      *       a\n"
"                                                                      * b c d*/",
"/*                                                                */ /*\n"
" *       a b\n"
" *       c     d*/");
  // clang-format on
}

TEST_F(FormatTestComments, NonTrailingBlockComments) {
  const auto Style40 = getLLVMStyleWithColumns(40);

  verifyFormat("const /** comment comment */ A = B;", Style40);

  verifyFormat("const /** comment comment comment */ A =\n"
               "    B;",
               Style40);

  verifyFormat("const /** comment comment comment\n"
               "         comment */\n"
               "    A = B;",
               "const /** comment comment comment comment */\n"
               "    A = B;",
               Style40);
}

TEST_F(FormatTestComments, PythonStyleComments) {
  const auto ProtoStyle20 = getTextProtoStyleWithColumns(20);

  // Keeps a space after '#'.
  verifyFormat("# comment\n"
               "key: value",
               "#comment\n"
               "key:value",
               ProtoStyle20);
  verifyFormat("# comment\n"
               "key: value",
               "# comment\n"
               "key:value",
               ProtoStyle20);
  // Breaks long comment.
  verifyFormat("# comment comment\n"
               "# comment\n"
               "key: value",
               "# comment comment comment\n"
               "key:value",
               ProtoStyle20);
  // Indents comments.
  verifyFormat("data {\n"
               "  # comment comment\n"
               "  # comment\n"
               "  key: value\n"
               "}",
               "data {\n"
               "# comment comment comment\n"
               "key: value}",
               ProtoStyle20);
  verifyFormat("data {\n"
               "  # comment comment\n"
               "  # comment\n"
               "  key: value\n"
               "}",
               "data {# comment comment comment\n"
               "key: value}",
               ProtoStyle20);
  // Reflows long comments.
  verifyFormat("# comment comment\n"
               "# comment comment\n"
               "key: value",
               "# comment comment comment\n"
               "# comment\n"
               "key:value",
               ProtoStyle20);
  // Breaks trailing comments.
  verifyFormat("k: val  # comment\n"
               "        # comment\n"
               "a: 1",
               "k:val#comment comment\n"
               "a:1",
               ProtoStyle20);
  verifyFormat("id {\n"
               "  k: val  # comment\n"
               "          # comment\n"
               "  # line line\n"
               "  a: 1\n"
               "}",
               "id {k:val#comment comment\n"
               "# line line\n"
               "a:1}",
               ProtoStyle20);
  // Aligns trailing comments.
  verifyFormat("k: val  # commen1\n"
               "        # commen2\n"
               "        # commen3\n"
               "# commen4\n"
               "a: 1  # commen5\n"
               "      # commen6\n"
               "      # commen7",
               "k:val#commen1 commen2\n"
               " #commen3\n"
               "# commen4\n"
               "a:1#commen5 commen6\n"
               " #commen7",
               ProtoStyle20);
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
  verifyFormat(
"// How to run:\n"
"// bbbbb run \\\n"
"// rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr\n"
"// \\ <log_file> -- --output_directory=\"<output_directory>\"",
"// How to run:\n"
"// bbbbb run \\\n"
"// rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr \\\n"
"// <log_file> -- --output_directory=\"<output_directory>\"");
  // clang-format on
}

TEST_F(FormatTestComments, IndentsLongJavadocAnnotatedLines) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_Java);
  Style.ColumnLimit = 60;
  verifyFormat("/**\n"
               " * @param x long long long long long long long long long\n"
               " *     long\n"
               " */",
               "/**\n"
               " * @param x long long long long long long long long long long\n"
               " */",
               Style);
  verifyFormat("/**\n"
               " * @param x long long long long long long long long long\n"
               " *     long long long long long long long long long long\n"
               " */",
               "/**\n"
               " * @param x long long long long long long long long long "
               "long long long long long long long long long long\n"
               " */",
               Style);
  verifyFormat("/**\n"
               " * @param x long long long long long long long long long\n"
               " *     long long long long long long long long long long\n"
               " *     long\n"
               " */",
               "/**\n"
               " * @param x long long long long long long long long long "
               "long long long long long long long long long long long\n"
               " */",
               Style);

  FormatStyle Style20 = getGoogleStyle(FormatStyle::LK_Java);
  Style20.ColumnLimit = 20;

  verifyFormat("/**\n"
               " * Sentence that\n"
               " * should be broken.\n"
               " * @param short\n"
               " * keep indentation\n"
               " */",
               "/**\n"
               " * Sentence that should be broken.\n"
               " * @param short\n"
               " * keep indentation\n"
               " */",
               Style20);

  verifyFormat("/**\n"
               " * @param l1 long1\n"
               " *     to break\n"
               " * @param l2 long2\n"
               " *     to break\n"
               " */",
               "/**\n"
               " * @param l1 long1 to break\n"
               " * @param l2 long2 to break\n"
               " */",
               Style20);

  verifyFormat("/**\n"
               " * @param xx to\n"
               " *     break\n"
               " * no reflow\n"
               " */",
               "/**\n"
               " * @param xx to break\n"
               " * no reflow\n"
               " */",
               Style20);

  verifyFormat("/**\n"
               " * @param xx to\n"
               " *     break yes\n"
               " *     reflow\n"
               " */",
               "/**\n"
               " * @param xx to break\n"
               " *     yes reflow\n"
               " */",
               Style20);

  FormatStyle JSStyle20 = getGoogleStyle(FormatStyle::LK_JavaScript);
  JSStyle20.ColumnLimit = 20;
  verifyFormat("/**\n"
               " * @param l1 long1\n"
               " *     to break\n"
               " */",
               "/**\n"
               " * @param l1 long1 to break\n"
               " */",
               JSStyle20);
  verifyFormat("/**\n"
               " * @param {l1 long1\n"
               " *     to break}\n"
               " */",
               "/**\n"
               " * @param {l1 long1 to break}\n"
               " */",
               JSStyle20);
}

TEST_F(FormatTestComments, SpaceAtLineCommentBegin) {
  constexpr StringRef NoTextInComment(" //       \n"
                                      "\n"
                                      "void foo() {// \n"
                                      "// \n"
                                      "}");

  verifyFormat("//\n"
               "\n"
               "void foo() { //\n"
               "  //\n"
               "}",
               NoTextInComment);

  auto Style = getLLVMStyle();
  Style.SpacesInLineCommentPrefix.Minimum = 0;
  verifyFormat("//#comment", Style);
  verifyFormat("//\n"
               "\n"
               "void foo() { //\n"
               "  //\n"
               "}",
               NoTextInComment, Style);

  Style.SpacesInLineCommentPrefix.Minimum = 5;
  verifyFormat("//     #comment", "//#comment", Style);
  verifyFormat("//\n"
               "\n"
               "void foo() { //\n"
               "  //\n"
               "}",
               NoTextInComment, Style);

  constexpr StringRef Code(
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
      "//} if the line above does");

  constexpr StringRef Code2(
      "// Free comment without space\n"
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
      "// } if the line above does");

  constexpr StringRef Code3(
      "//Free comment without space\n"
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
      "//} if the line above does");

  constexpr StringRef Code4(
      "//  Free comment without space\n"
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
      "//  } if the line above does");

  verifyFormat(Code2, Code);

  Style = getLLVMStyle();
  Style.SpacesInLineCommentPrefix = {0, 0};
  verifyFormat("//#comment", "//   #comment", Style);
  verifyFormat(Code3, Code, Style);

  Style.SpacesInLineCommentPrefix = {2, -1u};
  verifyFormat(Code4, Code, Style);

  Style = getLLVMStyleWithColumns(20);
  constexpr StringRef WrapCode("//Lorem ipsum dolor sit amet\n"
                               "\n"
                               "//  Lorem   ipsum   dolor   sit   amet\n"
                               "\n"
                               "void f() {//Hello World\n"
                               "}");

  verifyFormat("// Lorem ipsum dolor\n"
               "// sit amet\n"
               "\n"
               "//  Lorem   ipsum\n"
               "//  dolor   sit amet\n"
               "\n"
               "void f() { // Hello\n"
               "           // World\n"
               "}",
               WrapCode, Style);

  Style.SpacesInLineCommentPrefix = {0, 0};
  verifyFormat("//Lorem ipsum dolor\n"
               "//sit amet\n"
               "\n"
               "//Lorem   ipsum\n"
               "//dolor   sit   amet\n"
               "\n"
               "void f() { //Hello\n"
               "           //World\n"
               "}",
               WrapCode, Style);

  Style.SpacesInLineCommentPrefix = {1, 1};
  verifyFormat("// Lorem ipsum dolor\n"
               "// sit amet\n"
               "\n"
               "// Lorem   ipsum\n"
               "// dolor   sit amet\n"
               "\n"
               "void f() { // Hello\n"
               "           // World\n"
               "}",
               WrapCode, Style);
  verifyFormat("// x\n"
               "// y",
               "//   x\n"
               "// y",
               Style);
  verifyFormat(
      "// loooooooooooooooooooooooooooooong\n"
      "// commentcomments\n"
      "// normal comments",
      "//            loooooooooooooooooooooooooooooong commentcomments\n"
      "// normal comments",
      Style);

  Style.SpacesInLineCommentPrefix = {3, 3};
  verifyFormat("//   Lorem ipsum\n"
               "//   dolor sit amet\n"
               "\n"
               "//   Lorem   ipsum\n"
               "//   dolor   sit\n"
               "//   amet\n"
               "\n"
               "void f() { //   Hello\n"
               "           //   World\n"
               "}",
               WrapCode, Style);

  Style = getLLVMStyleWithColumns(20);
  constexpr StringRef LotsOfSpaces(
      "//                      This are more spaces "
      "than the ColumnLimit, what now?\n"
      "\n"
      "//   Comment\n"
      "\n"
      "// This is a text to split in multiple "
      "lines, please. Thank you very much!\n"
      "\n"
      "// A comment with\n"
      "//   some indentation that has to be split.\n"
      "// And now without");
  verifyFormat("//                      This are more spaces "
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
               LotsOfSpaces, Style);

  Style.SpacesInLineCommentPrefix = {0, 0};
  verifyFormat("//This are more\n"
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
               LotsOfSpaces, Style);

  Style.SpacesInLineCommentPrefix = {3, 3};
  verifyFormat("//   This are more\n"
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
               LotsOfSpaces, Style);

  Style.SpacesInLineCommentPrefix = {30, -1u};
  verifyFormat(
      "//                              This are more spaces than the "
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
      LotsOfSpaces, Style);

  Style.SpacesInLineCommentPrefix = {2, 4};
  verifyFormat("//  A Comment to be\n"
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
               "//A Comment to be moved\n"
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
               Style);

  Style.ColumnLimit = 30;
  verifyFormat("int i; //  A Comment to be\n"
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
               "int i;//A Comment to be moved\n"
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
               Style);

  Style = getLLVMStyleWithColumns(0);
  verifyFormat(Code2, Code, Style);

  Style.SpacesInLineCommentPrefix = {0, 0};
  verifyFormat(Code3, Code, Style);

  Style.SpacesInLineCommentPrefix = {2, -1u};
  verifyFormat(Code4, Code, Style);
}

TEST_F(FormatTestComments, SplitCommentIntroducers) {
  verifyFormat("//\n"
               "/\\\n"
               "/\n",
               "//\n"
               "/\\\n"
               "/ \n"
               "  ",
               getLLVMStyleWithColumns(10));
}

TEST_F(FormatTestComments, LineCommentsOnStartOfFunctionCall) {
  verifyFormat("Type name{// Comment\n"
               "          value};");

  auto Style = getLLVMStyle();
  EXPECT_EQ(Style.Cpp11BracedListStyle, FormatStyle::BLS_AlignFirstComment);
  Style.Cpp11BracedListStyle = FormatStyle::BLS_Block;

  verifyFormat("Type name{ // Comment\n"
               "           value\n"
               "};",
               Style);

  Style.Cpp11BracedListStyle = FormatStyle::BLS_FunctionCall;

  verifyFormat("Type name{ // Comment\n"
               "    value};",
               Style);

  verifyFormat("T foo( // Comment\n"
               "    arg);",
               Style);

  verifyFormat("T bar{ // Comment\n"
               "    arg};",
               Style);

  verifyFormat("T baz({ // Comment\n"
               "    arg});",
               Style);

  verifyFormat("T baz{{ // Comment\n"
               "    arg}};",
               Style);

  verifyFormat("T b0z(f( // Comment\n"
               "    arg));",
               Style);

  verifyFormat("T b0z(F{ // Comment\n"
               "    arg});",
               Style);

  verifyFormat("func( // Comment\n"
               "    arg);",
               Style);

  verifyFormat("func({ // Comment\n"
               "    arg});",
               Style);
}

} // end namespace
} // namespace test
} // end namespace format
} // end namespace clang
