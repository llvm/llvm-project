//===- unittest/Format/FormatTestComments.cpp - Formatting unit tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "../Tooling/ReplacementTest.h"
#include "FormatTestUtils.h"

#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

using clang::tooling::ReplacementTest;

namespace clang {
namespace format {
namespace {

FormatStyle getGoogleStyle() { return getGoogleStyle(FormatStyle::LK_Cpp); }

class FormatTestComments : public ::testing::Test {
protected:
  enum StatusCheck {
    SC_ExpectComplete,
    SC_ExpectIncomplete,
    SC_DoNotCheck
  };

  std::string format(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle(),
                     StatusCheck CheckComplete = SC_ExpectComplete) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
    FormattingAttemptStatus Status;
    tooling::Replacements Replaces =
        reformat(Style, Code, Ranges, "<stdin>", &Status);
    if (CheckComplete != SC_DoNotCheck) {
      bool ExpectedCompleteFormat = CheckComplete == SC_ExpectComplete;
      EXPECT_EQ(ExpectedCompleteFormat, Status.FormatComplete)
          << Code << "\n\n";
    }
    ReplacementCount = Replaces.size();
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  FormatStyle getLLVMStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getLLVMStyle();
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  void verifyFormat(llvm::StringRef Code,
                    const FormatStyle &Style = getLLVMStyle()) {
    EXPECT_EQ(Code.str(), format(test::messUp(Code), Style));
  }

  void verifyGoogleFormat(llvm::StringRef Code) {
    verifyFormat(Code, getGoogleStyle());
  }

  /// \brief Verify that clang-format does not crash on the given input.
  void verifyNoCrash(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    format(Code, Style, SC_DoNotCheck);
  }

  int ReplacementCount;
};

//===----------------------------------------------------------------------===//
// Tests for comments.
//===----------------------------------------------------------------------===//

TEST_F(FormatTestComments, UnderstandsSingleLineComments) {
  verifyFormat("//* */");
  verifyFormat("// line 1\n"
               "// line 2\n"
               "void f() {}\n");

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
            "  }          // awesome..\n"
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
  EXPECT_EQ(
      "void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaaa) { /*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*/\n"
      "}",
      format("void      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
             "                      aaaaaaaaaaaaaaaaaa  ,\n"
             "    aaaaaaaaaaaaaaaaaa) {   /*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*/\n"
             "}"));
  verifyFormat("f(/* aaaaaaaaaaaaaaaaaa = */\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = false;
  verifyFormat("aaaaaaaa(/* parameter 1 */ aaaaaa,\n"
               "         /* parameter 2 */ aaaaaa,\n"
               "         /* parameter 3 */ aaaaaa,\n"
               "         /* parameter 4 */ aaaaaa);",
               NoBinPacking);

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
  Style.ReflowComments = false;
  verifyFormat("// aaaaaaaaa aaaaaaaaaa aaaaaaaaaa", Style);
  verifyFormat("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa */", Style);
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
  EXPECT_EQ("//    Don't_touch_leading_whitespace",
            format("//    Don't_touch_leading_whitespace",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// Add leading\n"
            "// whitespace",
            format("//Add leading whitespace", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/// Add leading\n"
            "/// whitespace",
            format("///Add leading whitespace", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//! Add leading\n"
            "//! whitespace",
            format("//!Add leading whitespace", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("// whitespace", format("//whitespace", getLLVMStyle()));
  EXPECT_EQ("// Even if it makes the line exceed the column\n"
            "// limit",
            format("//Even if it makes the line exceed the column limit",
                   getLLVMStyleWithColumns(51)));
  EXPECT_EQ("//--But not here", format("//--But not here", getLLVMStyle()));
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
                   "//line 7", getLLVMStyleWithColumns(20)));

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
  EXPECT_EQ("//\t aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            format("//\t aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                   getLLVMStyleWithColumns(20)));
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
            "  // long 1 2 3 4\n"
            "  // 5\n"
            "}",
            format("{\n"
                   "  //\n"
                   "  //\\\n"
                   "  // long 1 2 3 4 5\n"
                   "}",
                   getLLVMStyleWithColumns(20)));
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

  // Align with the preprocessor directive if the comment was originally aligned
  // with the preprocessor directive.
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
}

TEST_F(FormatTestComments, SplitsLongLinesInComments) {
  EXPECT_EQ("/* This is a long\n"
            " * comment that\n"
            " * doesn't\n"
            " * fit on one line.\n"
            " */",
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

  EXPECT_EQ("/*\n"
            "\n"
            "\n"
            "    */\n",
            format("  /*       \n"
                   "      \n"
                   "               \n"
                   "      */\n"));

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
               "}", getLLVMStyleWithColumns(80));
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
               "    0x00, 0x00, 0x00, 0x00};            // comment\n");
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
            "// long long\n",
            format("// long long long long\n"
                   "// long\n",
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
            "// long_long_long\n",
            format("// long long long long\n"
                   "// long_long_long\n",
                   getLLVMStyleWithColumns(20)));

  // Don't break or reflow short lines.
  verifyFormat("// long\n"
               "// long long long lo\n"
               "// long long long lo\n"
               "// long",
               getLLVMStyleWithColumns(20));

  // Keep prefixes and decorations while reflowing.
  EXPECT_EQ("/// long long long\n"
            "/// long long\n",
            format("/// long long long long\n"
                   "/// long\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//! long long long\n"
            "//! long long\n",
            format("//! long long long long\n"
                   "//! long\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* long long long\n"
            " * long long */",
            format("/* long long long long\n"
                   " * long */",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("///< long long long\n"
            "///< long long\n",
            format("///< long long long long\n"
                   "///< long\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//!< long long long\n"
            "//!< long long\n",
            format("//!< long long long long\n"
                   "//!< long\n",
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
            "// long long long\n",
            format("// long long long long long long long long\n"
                   "// long\n",
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
  EXPECT_EQ("int i; ///           a",
            format("int i; ///           a", getLLVMStyleWithColumns(20)));

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
            "       // lines.\n",
            format("int a; // Trailing comment\n"
                   "       // on 2\n"
                   "       // or 3\n"
                   "       // lines.\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/// This long line\n"
            "/// gets reflown.\n",
            format("/// This long line gets\n"
                   "/// reflown.\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//! This long line\n"
            "//! gets reflown.\n",
            format(" //! This long line gets\n"
                   " //! reflown.\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* This long line\n"
            " * gets reflown.\n"
            " */\n",
            format("/* This long line gets\n"
                   " * reflown.\n"
                   " */\n",
                   getLLVMStyleWithColumns(20)));

  // Reflow after indentation makes a line too long.
  EXPECT_EQ("{\n"
            "  // long long long\n"
            "  // lo long\n"
            "}\n",
            format("{\n"
                   "// long long long lo\n"
                   "// long\n"
                   "}\n",
                   getLLVMStyleWithColumns(20)));

  // Break and reflow multiple lines.
  EXPECT_EQ("/*\n"
            " * Reflow the end of\n"
            " * line by 11 22 33\n"
            " * 4.\n"
            " */\n",
            format("/*\n"
                   " * Reflow the end of line\n"
                   " * by\n"
                   " * 11\n"
                   " * 22\n"
                   " * 33\n"
                   " * 4.\n"
                   " */\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/// First line gets\n"
            "/// broken. Second\n"
            "/// line gets\n"
            "/// reflown and\n"
            "/// broken. Third\n"
            "/// gets reflown.\n",
            format("/// First line gets broken.\n"
                   "/// Second line gets reflown and broken.\n"
                   "/// Third gets reflown.\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("int i; // first long\n"
            "       // long snd\n"
            "       // long.\n",
            format("int i; // first long long\n"
                   "       // snd long.\n",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ("{\n"
            "  // first long line\n"
            "  // line second\n"
            "  // long line line\n"
            "  // third long line\n"
            "  // line\n"
            "}\n",
            format("{\n"
                   "  // first long line line\n"
                   "  // second long line line\n"
                   "  // third long line line\n"
                   "}\n",
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
  EXPECT_EQ(
      "//     long\n"
      "// long long long\n"
      "// long long",
      format("//     long\n"
             "// long long long long\n"
             "// long",
             getLLVMStyleWithColumns(20)));

  // Keep the block comment endling '*/' while reflowing.
  EXPECT_EQ("/* Long long long\n"
            " * line short */\n",
            format("/* Long long long line\n"
                   " * short */\n",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow between separate blocks of comments.
  EXPECT_EQ("/* First comment\n"
            " * block will */\n"
            "/* Snd\n"
            " */\n",
            format("/* First comment block\n"
                   " * will */\n"
                   "/* Snd\n"
                   " */\n",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow across blank comment lines.
  EXPECT_EQ("int i; // This long\n"
            "       // line gets\n"
            "       // broken.\n"
            "       //\n"
            "       // keep.\n",
            format("int i; // This long line gets broken.\n"
                   "       //  \n"
                   "       // keep.\n",
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
            format(
                "// long long long long\n"
                "// ... --- ...",
                getLLVMStyleWithColumns(20)));

  // Don't reflow lines starting with '@'.
  EXPECT_EQ("// long long long\n"
            "// long\n"
            "// @param arg",
            format("// long long long long\n"
                   "// @param arg",
                   getLLVMStyleWithColumns(20)));

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
            format(
                "// long long long long\n"
                "// 'long'",
                getLLVMStyleWithColumns(20)));

  // Don't reflow between separate blocks of comments.
  EXPECT_EQ("/* First comment\n"
            " * block will */\n"
            "/* Snd\n"
            " */\n",
            format("/* First comment block\n"
                   " * will */\n"
                   "/* Snd\n"
                   " */\n",
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
            "       // long\n",
            format("int i; // long long long\n"
                   "int j; // long long long\n",
                   getLLVMStyleWithColumns(20)));

  // Don't reflow if the first word on the next line is longer than the
  // available space at current line.
  EXPECT_EQ("int i; // trigger\n"
            "       // reflow\n"
            "       // longsec\n",
            format("int i; // trigger reflow\n"
                   "       // longsec\n",
                   getLLVMStyleWithColumns(20)));

  // Keep empty comment lines.
  EXPECT_EQ("/**/", format(" /**/", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/* */", format(" /* */", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("/*  */", format(" /*  */", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("//", format(" //  ", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("///", format(" ///  ", getLLVMStyleWithColumns(20)));
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
            "void g() {}\n",
            format("#if false\n"
                   "void f(  ) {  }\n"
                   "#endif\n"
                   "void g(  ) {  }\n"));
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
  EXPECT_EQ("/*\n**\n*/", format("/*\n**\n*/"));
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
  NoBinPacking.BinPackParameters = false;
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
      "int cccccccccccccccccccccccccccccc;       /* comment */\n",
      format("int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; /* comment */\n"
             "int      bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; /* comment */\n"
             "int    cccccccccccccccccccccccccccccc;  /* comment */\n"));

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
            "        // long\n",
            format("int ab; // line\n"
                   "int a; // long long\n",
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
}

TEST_F(FormatTestComments, AlignsBlockCommentDecorations) {
  EXPECT_EQ("/*\n"
            " */",
            format("/*\n"
                   "*/", getLLVMStyle()));
  EXPECT_EQ("/*\n"
            " */",
            format("/*\n"
                   " */", getLLVMStyle()));
  EXPECT_EQ("/*\n"
            " */",
            format("/*\n"
                   "  */", getLLVMStyle()));

  // Align a single line.
  EXPECT_EQ("/*\n"
            " * line */",
            format("/*\n"
                   "* line */",
                   getLLVMStyle()));
  EXPECT_EQ("/*\n"
            " * line */",
            format("/*\n"
                   " * line */",
                   getLLVMStyle()));
  EXPECT_EQ("/*\n"
            " * line */",
            format("/*\n"
                   "  * line */",
                   getLLVMStyle()));
  EXPECT_EQ("/*\n"
            " * line */",
            format("/*\n"
                   "   * line */",
                   getLLVMStyle()));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   "* line */",
                   getLLVMStyle()));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   " * line */",
                   getLLVMStyle()));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   "  * line */",
                   getLLVMStyle()));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   "   * line */",
                   getLLVMStyle()));
  EXPECT_EQ("/**\n"
            " * line */",
            format("/**\n"
                   "    * line */",
                   getLLVMStyle()));

  // Align the end '*/' after a line.
  EXPECT_EQ("/*\n"
            " * line\n"
            " */",
            format("/*\n"
                   "* line\n"
                   "*/", getLLVMStyle()));
  EXPECT_EQ("/*\n"
            " * line\n"
            " */",
            format("/*\n"
                   "   * line\n"
                   "  */", getLLVMStyle()));
  EXPECT_EQ("/*\n"
            " * line\n"
            " */",
            format("/*\n"
                   "  * line\n"
                   "  */", getLLVMStyle()));

  // Align two lines.
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   " * line 2 */",
                   getLLVMStyle()));
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   "* line 2 */",
                   getLLVMStyle()));
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   "  * line 2 */",
                   getLLVMStyle()));
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   "   * line 2 */",
                   getLLVMStyle()));
  EXPECT_EQ("/* line 1\n"
            " * line 2 */",
            format("/* line 1\n"
                   "    * line 2 */",
                   getLLVMStyle()));
  EXPECT_EQ("int i; /* line 1\n"
            "        * line 2 */",
            format("int i; /* line 1\n"
                   "* line 2 */",
                   getLLVMStyle()));
  EXPECT_EQ("int i; /* line 1\n"
            "        * line 2 */",
            format("int i; /* line 1\n"
                   "        * line 2 */",
                   getLLVMStyle()));
  EXPECT_EQ("int i; /* line 1\n"
            "        * line 2 */",
            format("int i; /* line 1\n"
                   "             * line 2 */",
                   getLLVMStyle()));

  // Align several lines.
  EXPECT_EQ("/* line 1\n"
            " * line 2\n"
            " * line 3 */",
            format("/* line 1\n"
                   " * line 2\n"
                   "* line 3 */",
                   getLLVMStyle()));
  EXPECT_EQ("/* line 1\n"
            " * line 2\n"
            " * line 3 */",
            format("/* line 1\n"
                   "  * line 2\n"
                   "* line 3 */",
                   getLLVMStyle()));
  EXPECT_EQ("/*\n"
            "** line 1\n"
            "** line 2\n"
            "*/",
            format("/*\n"
                   "** line 1\n"
                   " ** line 2\n"
                   "*/",
                   getLLVMStyle()));

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
                   "*/", getLLVMStyle()));

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
                   "*/", getLLVMStyle()));

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
} // end namespace
} // end namespace format
} // end namespace clang
