//===- unittest/Format/FormatTest.cpp - Formatting unit tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {
namespace {

FormatStyle getGoogleStyle() { return getGoogleStyle(FormatStyle::LK_Cpp); }

class FormatTest : public ::testing::Test {
protected:
  enum IncompleteCheck {
    IC_ExpectComplete,
    IC_ExpectIncomplete,
    IC_DoNotCheck
  };

  std::string format(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle(),
                     IncompleteCheck CheckIncomplete = IC_ExpectComplete) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
    bool IncompleteFormat = false;
    tooling::Replacements Replaces =
        reformat(Style, Code, Ranges, "<stdin>", &IncompleteFormat);
    if (CheckIncomplete != IC_DoNotCheck) {
      bool ExpectedIncompleteFormat = CheckIncomplete == IC_ExpectIncomplete;
      EXPECT_EQ(ExpectedIncompleteFormat, IncompleteFormat) << Code << "\n\n";
    }
    ReplacementCount = Replaces.size();
    std::string Result = applyAllReplacements(Code, Replaces);
    EXPECT_NE("", Result);
    DEBUG(llvm::errs() << "\n" << Result << "\n\n");
    return Result;
  }

  FormatStyle getLLVMStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getLLVMStyle();
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  FormatStyle getGoogleStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle();
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  void verifyFormat(llvm::StringRef Code,
                    const FormatStyle &Style = getLLVMStyle()) {
    EXPECT_EQ(Code.str(), format(test::messUp(Code), Style));
  }

  void verifyIncompleteFormat(llvm::StringRef Code,
                              const FormatStyle &Style = getLLVMStyle()) {
    EXPECT_EQ(Code.str(),
              format(test::messUp(Code), Style, IC_ExpectIncomplete));
  }

  void verifyGoogleFormat(llvm::StringRef Code) {
    verifyFormat(Code, getGoogleStyle());
  }

  void verifyIndependentOfContext(llvm::StringRef text) {
    verifyFormat(text);
    verifyFormat(llvm::Twine("void f() { " + text + " }").str());
  }

  /// \brief Verify that clang-format does not crash on the given input.
  void verifyNoCrash(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    format(Code, Style, IC_DoNotCheck);
  }

  int ReplacementCount;
};

TEST_F(FormatTest, MessUp) {
  EXPECT_EQ("1 2 3", test::messUp("1 2 3"));
  EXPECT_EQ("1 2 3\n", test::messUp("1\n2\n3\n"));
  EXPECT_EQ("a\n//b\nc", test::messUp("a\n//b\nc"));
  EXPECT_EQ("a\n#b\nc", test::messUp("a\n#b\nc"));
  EXPECT_EQ("a\n#b c d\ne", test::messUp("a\n#b\\\nc\\\nd\ne"));
}

//===----------------------------------------------------------------------===//
// Basic function tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, DoesNotChangeCorrectlyFormattedCode) {
  EXPECT_EQ(";", format(";"));
}

TEST_F(FormatTest, FormatsGlobalStatementsAt0) {
  EXPECT_EQ("int i;", format("  int i;"));
  EXPECT_EQ("\nint i;", format(" \n\t \v \f  int i;"));
  EXPECT_EQ("int i;\nint j;", format("    int i; int j;"));
  EXPECT_EQ("int i;\nint j;", format("    int i;\n  int j;"));
}

TEST_F(FormatTest, FormatsUnwrappedLinesAtFirstFormat) {
  EXPECT_EQ("int i;", format("int\ni;"));
}

TEST_F(FormatTest, FormatsNestedBlockStatements) {
  EXPECT_EQ("{\n  {\n    {}\n  }\n}", format("{{{}}}"));
}

TEST_F(FormatTest, FormatsNestedCall) {
  verifyFormat("Method(f1, f2(f3));");
  verifyFormat("Method(f1(f2, f3()));");
  verifyFormat("Method(f1(f2, (f3())));");
}

TEST_F(FormatTest, NestedNameSpecifiers) {
  verifyFormat("vector<::Type> v;");
  verifyFormat("::ns::SomeFunction(::ns::SomeOtherFunction())");
  verifyFormat("static constexpr bool Bar = decltype(bar())::value;");
  verifyFormat("bool a = 2 < ::SomeFunction();");
}

TEST_F(FormatTest, OnlyGeneratesNecessaryReplacements) {
  EXPECT_EQ("if (a) {\n"
            "  f();\n"
            "}",
            format("if(a){f();}"));
  EXPECT_EQ(4, ReplacementCount);
  EXPECT_EQ("if (a) {\n"
            "  f();\n"
            "}",
            format("if (a) {\n"
                   "  f();\n"
                   "}"));
  EXPECT_EQ(0, ReplacementCount);
  EXPECT_EQ("/*\r\n"
            "\r\n"
            "*/\r\n",
            format("/*\r\n"
                   "\r\n"
                   "*/\r\n"));
  EXPECT_EQ(0, ReplacementCount);
}

TEST_F(FormatTest, RemovesEmptyLines) {
  EXPECT_EQ("class C {\n"
            "  int i;\n"
            "};",
            format("class C {\n"
                   " int i;\n"
                   "\n"
                   "};"));

  // Don't remove empty lines at the start of namespaces or extern "C" blocks.
  EXPECT_EQ("namespace N {\n"
            "\n"
            "int i;\n"
            "}",
            format("namespace N {\n"
                   "\n"
                   "int    i;\n"
                   "}",
                   getGoogleStyle()));
  EXPECT_EQ("extern /**/ \"C\" /**/ {\n"
            "\n"
            "int i;\n"
            "}",
            format("extern /**/ \"C\" /**/ {\n"
                   "\n"
                   "int    i;\n"
                   "}",
                   getGoogleStyle()));

  // ...but do keep inlining and removing empty lines for non-block extern "C"
  // functions.
  verifyFormat("extern \"C\" int f() { return 42; }", getGoogleStyle());
  EXPECT_EQ("extern \"C\" int f() {\n"
            "  int i = 42;\n"
            "  return i;\n"
            "}",
            format("extern \"C\" int f() {\n"
                   "\n"
                   "  int i = 42;\n"
                   "  return i;\n"
                   "}",
                   getGoogleStyle()));

  // Remove empty lines at the beginning and end of blocks.
  EXPECT_EQ("void f() {\n"
            "\n"
            "  if (a) {\n"
            "\n"
            "    f();\n"
            "  }\n"
            "}",
            format("void f() {\n"
                   "\n"
                   "  if (a) {\n"
                   "\n"
                   "    f();\n"
                   "\n"
                   "  }\n"
                   "\n"
                   "}",
                   getLLVMStyle()));
  EXPECT_EQ("void f() {\n"
            "  if (a) {\n"
            "    f();\n"
            "  }\n"
            "}",
            format("void f() {\n"
                   "\n"
                   "  if (a) {\n"
                   "\n"
                   "    f();\n"
                   "\n"
                   "  }\n"
                   "\n"
                   "}",
                   getGoogleStyle()));

  // Don't remove empty lines in more complex control statements.
  EXPECT_EQ("void f() {\n"
            "  if (a) {\n"
            "    f();\n"
            "\n"
            "  } else if (b) {\n"
            "    f();\n"
            "  }\n"
            "}",
            format("void f() {\n"
                   "  if (a) {\n"
                   "    f();\n"
                   "\n"
                   "  } else if (b) {\n"
                   "    f();\n"
                   "\n"
                   "  }\n"
                   "\n"
                   "}"));

  // FIXME: This is slightly inconsistent.
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "}",
            format("namespace {\n"
                   "int i;\n"
                   "\n"
                   "}"));
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "\n"
            "} // namespace",
            format("namespace {\n"
                   "int i;\n"
                   "\n"
                   "}  // namespace"));
}

TEST_F(FormatTest, RecognizesBinaryOperatorKeywords) {
  verifyFormat("x = (a) and (b);");
  verifyFormat("x = (a) or (b);");
  verifyFormat("x = (a) bitand (b);");
  verifyFormat("x = (a) bitor (b);");
  verifyFormat("x = (a) not_eq (b);");
  verifyFormat("x = (a) and_eq (b);");
  verifyFormat("x = (a) or_eq (b);");
  verifyFormat("x = (a) xor (b);");
}

//===----------------------------------------------------------------------===//
// Tests for control statements.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, FormatIfWithoutCompoundStatement) {
  verifyFormat("if (true)\n  f();\ng();");
  verifyFormat("if (a)\n  if (b)\n    if (c)\n      g();\nh();");
  verifyFormat("if (a)\n  if (b) {\n    f();\n  }\ng();");

  FormatStyle AllowsMergedIf = getLLVMStyle();
  AllowsMergedIf.AllowShortIfStatementsOnASingleLine = true;
  verifyFormat("if (a)\n"
               "  // comment\n"
               "  f();",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  ;",
               AllowsMergedIf);
  verifyFormat("if (a)\n"
               "  if (b) return;",
               AllowsMergedIf);

  verifyFormat("if (a) // Can't merge this\n"
               "  f();\n",
               AllowsMergedIf);
  verifyFormat("if (a) /* still don't merge */\n"
               "  f();",
               AllowsMergedIf);
  verifyFormat("if (a) { // Never merge this\n"
               "  f();\n"
               "}",
               AllowsMergedIf);
  verifyFormat("if (a) { /* Never merge this */\n"
               "  f();\n"
               "}",
               AllowsMergedIf);

  AllowsMergedIf.ColumnLimit = 14;
  verifyFormat("if (a) return;", AllowsMergedIf);
  verifyFormat("if (aaaaaaaaa)\n"
               "  return;",
               AllowsMergedIf);

  AllowsMergedIf.ColumnLimit = 13;
  verifyFormat("if (a)\n  return;", AllowsMergedIf);
}

TEST_F(FormatTest, FormatLoopsWithoutCompoundStatement) {
  FormatStyle AllowsMergedLoops = getLLVMStyle();
  AllowsMergedLoops.AllowShortLoopsOnASingleLine = true;
  verifyFormat("while (true) continue;", AllowsMergedLoops);
  verifyFormat("for (;;) continue;", AllowsMergedLoops);
  verifyFormat("for (int &v : vec) v *= 2;", AllowsMergedLoops);
  verifyFormat("while (true)\n"
               "  ;",
               AllowsMergedLoops);
  verifyFormat("for (;;)\n"
               "  ;",
               AllowsMergedLoops);
  verifyFormat("for (;;)\n"
               "  for (;;) continue;",
               AllowsMergedLoops);
  verifyFormat("for (;;) // Can't merge this\n"
               "  continue;",
               AllowsMergedLoops);
  verifyFormat("for (;;) /* still don't merge */\n"
               "  continue;",
               AllowsMergedLoops);
}

TEST_F(FormatTest, FormatShortBracedStatements) {
  FormatStyle AllowSimpleBracedStatements = getLLVMStyle();
  AllowSimpleBracedStatements.AllowShortBlocksOnASingleLine = true;

  AllowSimpleBracedStatements.AllowShortIfStatementsOnASingleLine = true;
  AllowSimpleBracedStatements.AllowShortLoopsOnASingleLine = true;

  verifyFormat("if (true) {}", AllowSimpleBracedStatements);
  verifyFormat("while (true) {}", AllowSimpleBracedStatements);
  verifyFormat("for (;;) {}", AllowSimpleBracedStatements);
  verifyFormat("if (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("while (true) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("for (;;) { f(); }", AllowSimpleBracedStatements);
  verifyFormat("if (true) { //\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  f();\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  f();\n"
               "} else {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);

  verifyFormat("template <int> struct A2 {\n"
               "  struct B {};\n"
               "};",
               AllowSimpleBracedStatements);

  AllowSimpleBracedStatements.AllowShortIfStatementsOnASingleLine = false;
  verifyFormat("if (true) {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("if (true) {\n"
               "  f();\n"
               "} else {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);

  AllowSimpleBracedStatements.AllowShortLoopsOnASingleLine = false;
  verifyFormat("while (true) {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
  verifyFormat("for (;;) {\n"
               "  f();\n"
               "}",
               AllowSimpleBracedStatements);
}

TEST_F(FormatTest, ParseIfElse) {
  verifyFormat("if (true)\n"
               "  if (true)\n"
               "    if (true)\n"
               "      f();\n"
               "    else\n"
               "      g();\n"
               "  else\n"
               "    h();\n"
               "else\n"
               "  i();");
  verifyFormat("if (true)\n"
               "  if (true)\n"
               "    if (true) {\n"
               "      if (true)\n"
               "        f();\n"
               "    } else {\n"
               "      g();\n"
               "    }\n"
               "  else\n"
               "    h();\n"
               "else {\n"
               "  i();\n"
               "}");
  verifyFormat("void f() {\n"
               "  if (a) {\n"
               "  } else {\n"
               "  }\n"
               "}");
}

TEST_F(FormatTest, ElseIf) {
  verifyFormat("if (a) {\n} else if (b) {\n}");
  verifyFormat("if (a)\n"
               "  f();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  h();");
  verifyFormat("if (a) {\n"
               "  f();\n"
               "}\n"
               "// or else ..\n"
               "else {\n"
               "  g()\n"
               "}");

  verifyFormat("if (a) {\n"
               "} else if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaa)) {\n"
               "}");
  verifyFormat("if (a) {\n"
               "} else if (\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
               "}",
               getLLVMStyleWithColumns(62));
}

TEST_F(FormatTest, FormatsForLoop) {
  verifyFormat(
      "for (int VeryVeryLongLoopVariable = 0; VeryVeryLongLoopVariable < 10;\n"
      "     ++VeryVeryLongLoopVariable)\n"
      "  ;");
  verifyFormat("for (;;)\n"
               "  f();");
  verifyFormat("for (;;) {\n}");
  verifyFormat("for (;;) {\n"
               "  f();\n"
               "}");
  verifyFormat("for (int i = 0; (i < 10); ++i) {\n}");

  verifyFormat(
      "for (std::vector<UnwrappedLine>::iterator I = UnwrappedLines.begin(),\n"
      "                                          E = UnwrappedLines.end();\n"
      "     I != E; ++I) {\n}");

  verifyFormat(
      "for (MachineFun::iterator IIII = PrevIt, EEEE = F.end(); IIII != EEEE;\n"
      "     ++IIIII) {\n}");
  verifyFormat("for (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaa =\n"
               "         aaaaaaaaaaaaaaaa.aaaaaaaaaaaaaaa;\n"
               "     aaaaaaaaaaa != aaaaaaaaaaaaaaaaaaa; ++aaaaaaaaaaa) {\n}");
  verifyFormat("for (llvm::ArrayRef<NamedDecl *>::iterator\n"
               "         I = FD->getDeclsInPrototypeScope().begin(),\n"
               "         E = FD->getDeclsInPrototypeScope().end();\n"
               "     I != E; ++I) {\n}");
  verifyFormat("for (SmallVectorImpl<TemplateIdAnnotationn *>::iterator\n"
               "         I = Container.begin(),\n"
               "         E = Container.end();\n"
               "     I != E; ++I) {\n}",
               getLLVMStyleWithColumns(76));

  verifyFormat(
      "for (aaaaaaaaaaaaaaaaa aaaaaaaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;\n"
      "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa !=\n"
      "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
      "     ++aaaaaaaaaaa) {\n}");
  verifyFormat("for (int i = 0; i < aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "                bbbbbbbbbbbbbbbbbbbb < ccccccccccccccc;\n"
               "     ++i) {\n}");
  verifyFormat("for (int aaaaaaaaaaa = 1; aaaaaaaaaaa <= bbbbbbbbbbbbbbb;\n"
               "     aaaaaaaaaaa++, bbbbbbbbbbbbbbbbb++) {\n"
               "}");
  verifyFormat("for (some_namespace::SomeIterator iter( // force break\n"
               "         aaaaaaaaaa);\n"
               "     iter; ++iter) {\n"
               "}");
  verifyFormat("for (auto aaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaa != bbbbbbbbbbbbbbbbbbbbbbb;\n"
               "     ++aaaaaaaaaaaaaaaaaaaaaaaaaaa) {");

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = false;
  verifyFormat("for (int aaaaaaaaaaa = 1;\n"
               "     aaaaaaaaaaa <= aaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaa,\n"
               "                                           aaaaaaaaaaaaaaaa,\n"
               "                                           aaaaaaaaaaaaaaaa,\n"
               "                                           aaaaaaaaaaaaaaaa);\n"
               "     aaaaaaaaaaa++, bbbbbbbbbbbbbbbbb++) {\n"
               "}",
               NoBinPacking);
  verifyFormat(
      "for (std::vector<UnwrappedLine>::iterator I = UnwrappedLines.begin(),\n"
      "                                          E = UnwrappedLines.end();\n"
      "     I != E;\n"
      "     ++I) {\n}",
      NoBinPacking);
}

TEST_F(FormatTest, RangeBasedForLoops) {
  verifyFormat("for (auto aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat("for (auto aaaaaaaaaaaaaaaaaaaaa :\n"
               "     aaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaa, aaaaaaaaaaaaa)) {\n}");
  verifyFormat("for (const aaaaaaaaaaaaaaaaaaaaa &aaaaaaaaa :\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat("for (aaaaaaaaa aaaaaaaaaaaaaaaaaaaaa :\n"
               "     aaaaaaaaaaaa.aaaaaaaaaaaa().aaaaaaaaa().a()) {\n}");
}

TEST_F(FormatTest, ForEachLoops) {
  verifyFormat("void f() {\n"
               "  foreach (Item *item, itemlist) {}\n"
               "  Q_FOREACH (Item *item, itemlist) {}\n"
               "  BOOST_FOREACH (Item *item, itemlist) {}\n"
               "  UNKNOWN_FORACH(Item * item, itemlist) {}\n"
               "}");

  // As function-like macros.
  verifyFormat("#define foreach(x, y)\n"
               "#define Q_FOREACH(x, y)\n"
               "#define BOOST_FOREACH(x, y)\n"
               "#define UNKNOWN_FOREACH(x, y)\n");

  // Not as function-like macros.
  verifyFormat("#define foreach (x, y)\n"
               "#define Q_FOREACH (x, y)\n"
               "#define BOOST_FOREACH (x, y)\n"
               "#define UNKNOWN_FOREACH (x, y)\n");
}

TEST_F(FormatTest, FormatsWhileLoop) {
  verifyFormat("while (true) {\n}");
  verifyFormat("while (true)\n"
               "  f();");
  verifyFormat("while () {\n}");
  verifyFormat("while () {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, FormatsDoWhile) {
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while (something());");
  verifyFormat("do\n"
               "  do_something();\n"
               "while (something());");
}

TEST_F(FormatTest, FormatsSwitchStatement) {
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "  f();\n"
               "  break;\n"
               "case kFoo:\n"
               "case ns::kBar:\n"
               "case kBaz:\n"
               "  break;\n"
               "default:\n"
               "  g();\n"
               "  break;\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  break;\n"
               "}\n"
               "case 2: {\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  {\n"
               "    g();\n"
               "    h();\n"
               "  }\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  if (foo) {\n"
               "    g();\n"
               "    h();\n"
               "  }\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  g();\n"
               "} break;\n"
               "}");
  verifyFormat("switch (test)\n"
               "  ;");
  verifyFormat("switch (x) {\n"
               "default: {\n"
               "  // Do nothing.\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "// comment\n"
               "// if 1, do f()\n"
               "case 1:\n"
               "  f();\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "  // Do amazing stuff\n"
               "  {\n"
               "    f();\n"
               "    g();\n"
               "  }\n"
               "  break;\n"
               "}");
  verifyFormat("#define A          \\\n"
               "  switch (x) {     \\\n"
               "  case a:          \\\n"
               "    foo = b;       \\\n"
               "  }",
               getLLVMStyleWithColumns(20));
  verifyFormat("#define OPERATION_CASE(name)           \\\n"
               "  case OP_name:                        \\\n"
               "    return operations::Operation##name\n",
               getLLVMStyleWithColumns(40));
  verifyFormat("switch (x) {\n"
               "case 1:;\n"
               "default:;\n"
               "  int i;\n"
               "}");

  verifyGoogleFormat("switch (x) {\n"
                     "  case 1:\n"
                     "    f();\n"
                     "    break;\n"
                     "  case kFoo:\n"
                     "  case ns::kBar:\n"
                     "  case kBaz:\n"
                     "    break;\n"
                     "  default:\n"
                     "    g();\n"
                     "    break;\n"
                     "}");
  verifyGoogleFormat("switch (x) {\n"
                     "  case 1: {\n"
                     "    f();\n"
                     "    break;\n"
                     "  }\n"
                     "}");
  verifyGoogleFormat("switch (test)\n"
                     "  ;");

  verifyGoogleFormat("#define OPERATION_CASE(name) \\\n"
                     "  case OP_name:              \\\n"
                     "    return operations::Operation##name\n");
  verifyGoogleFormat("Operation codeToOperation(OperationCode OpCode) {\n"
                     "  // Get the correction operation class.\n"
                     "  switch (OpCode) {\n"
                     "    CASE(Add);\n"
                     "    CASE(Subtract);\n"
                     "    default:\n"
                     "      return operations::Unknown;\n"
                     "  }\n"
                     "#undef OPERATION_CASE\n"
                     "}");
  verifyFormat("DEBUG({\n"
               "  switch (x) {\n"
               "  case A:\n"
               "    f();\n"
               "    break;\n"
               "  // On B:\n"
               "  case B:\n"
               "    g();\n"
               "    break;\n"
               "  }\n"
               "});");
  verifyFormat("switch (a) {\n"
               "case (b):\n"
               "  return;\n"
               "}");

  verifyFormat("switch (a) {\n"
               "case some_namespace::\n"
               "    some_constant:\n"
               "  return;\n"
               "}",
               getLLVMStyleWithColumns(34));
}

TEST_F(FormatTest, CaseRanges) {
  verifyFormat("switch (x) {\n"
               "case 'A' ... 'Z':\n"
               "case 1 ... 5:\n"
               "  break;\n"
               "}");
}

TEST_F(FormatTest, ShortCaseLabels) {
  FormatStyle Style = getLLVMStyle();
  Style.AllowShortCaseLabelsOnASingleLine = true;
  verifyFormat("switch (a) {\n"
               "case 1: x = 1; break;\n"
               "case 2: return;\n"
               "case 3:\n"
               "case 4:\n"
               "case 5: return;\n"
               "case 6: // comment\n"
               "  return;\n"
               "case 7:\n"
               "  // comment\n"
               "  return;\n"
               "case 8:\n"
               "  x = 8; // comment\n"
               "  break;\n"
               "default: y = 1; break;\n"
               "}",
               Style);
  verifyFormat("switch (a) {\n"
               "#if FOO\n"
               "case 0: return 0;\n"
               "#endif\n"
               "}",
               Style);
  verifyFormat("switch (a) {\n"
               "case 1: {\n"
               "}\n"
               "case 2: {\n"
               "  return;\n"
               "}\n"
               "case 3: {\n"
               "  x = 1;\n"
               "  return;\n"
               "}\n"
               "case 4:\n"
               "  if (x)\n"
               "    return;\n"
               "}",
               Style);
  Style.ColumnLimit = 21;
  verifyFormat("switch (a) {\n"
               "case 1: x = 1; break;\n"
               "case 2: return;\n"
               "case 3:\n"
               "case 4:\n"
               "case 5: return;\n"
               "default:\n"
               "  y = 1;\n"
               "  break;\n"
               "}",
               Style);
}

TEST_F(FormatTest, FormatsLabels) {
  verifyFormat("void f() {\n"
               "  some_code();\n"
               "test_label:\n"
               "  some_other_code();\n"
               "  {\n"
               "    some_more_code();\n"
               "  another_label:\n"
               "    some_more_code();\n"
               "  }\n"
               "}");
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label:\n"
               "  some_other_code();\n"
               "}");
  verifyFormat("{\n"
               "  some_code();\n"
               "test_label:;\n"
               "  int i = 0;\n"
               "}");
}

//===----------------------------------------------------------------------===//
// Tests for comments.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, UnderstandsSingleLineComments) {
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

TEST_F(FormatTest, KeepsParameterWithTrailingCommentsOnTheirOwnLine) {
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
}

TEST_F(FormatTest, RemovesTrailingWhitespaceOfComments) {
  EXPECT_EQ("// comment", format("// comment  "));
  EXPECT_EQ("int aaaaaaa, bbbbbbb; // comment",
            format("int aaaaaaa, bbbbbbb; // comment                   ",
                   getLLVMStyleWithColumns(33)));
  EXPECT_EQ("// comment\\\n", format("// comment\\\n  \t \v   \f   "));
  EXPECT_EQ("// comment    \\\n", format("// comment    \\\n  \t \v   \f   "));
}

TEST_F(FormatTest, UnderstandsBlockComments) {
  verifyFormat("f(/*noSpaceAfterParameterNamingComment=*/true);");
  verifyFormat("void f() { g(/*aaa=*/x, /*bbb=*/!y); }");
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

TEST_F(FormatTest, AlignsBlockComments) {
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

TEST_F(FormatTest, CommentReflowingCanBeTurnedOff) {
  FormatStyle Style = getLLVMStyleWithColumns(20);
  Style.ReflowComments = false;
  verifyFormat("// aaaaaaaaa aaaaaaaaaa aaaaaaaaaa", Style);
  verifyFormat("/* aaaaaaaaa aaaaaaaaaa aaaaaaaaaa */", Style);
}

TEST_F(FormatTest, CorrectlyHandlesLengthOfBlockComments) {
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

TEST_F(FormatTest, DontBreakNonTrailingBlockComments) {
  EXPECT_EQ("void ffffffffff(\n"
            "    int aaaaa /* test */);",
            format("void ffffffffff(int aaaaa /* test */);",
                   getLLVMStyleWithColumns(35)));
}

TEST_F(FormatTest, SplitsLongCxxComments) {
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
}

TEST_F(FormatTest, PreservesHangingIndentInCxxComments) {
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

TEST_F(FormatTest, DontSplitLineCommentsWithEscapedNewlines) {
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

TEST_F(FormatTest, DontSplitLineCommentsWithPragmas) {
  FormatStyle Pragmas = getLLVMStyleWithColumns(30);
  Pragmas.CommentPragmas = "^ IWYU pragma:";
  EXPECT_EQ(
      "// IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb",
      format("// IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb", Pragmas));
  EXPECT_EQ(
      "/* IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb */",
      format("/* IWYU pragma: aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb */", Pragmas));
}

TEST_F(FormatTest, PriorityOfCommentBreaking) {
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

TEST_F(FormatTest, MultiLineCommentsInDefines) {
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

TEST_F(FormatTest, ParsesCommentsAdjacentToPPDirectives) {
  EXPECT_EQ("namespace {}\n// Test\n#define A",
            format("namespace {}\n   // Test\n#define A"));
  EXPECT_EQ("namespace {}\n/* Test */\n#define A",
            format("namespace {}\n   /* Test */\n#define A"));
  EXPECT_EQ("namespace {}\n/* Test */ #define A",
            format("namespace {}\n   /* Test */    #define A"));
}

TEST_F(FormatTest, SplitsLongLinesInComments) {
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
            "     *   a comment\n"
            "* that we break\n"
            " * another comment\n"
            "* we have to break\n"
            "* a left comment\n"
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

TEST_F(FormatTest, SplitsLongLinesInCommentsInPreprocessor) {
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

TEST_F(FormatTest, CommentsInStaticInitializers) {
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

TEST_F(FormatTest, IgnoresIf0Contents) {
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
}

//===----------------------------------------------------------------------===//
// Tests for classes, namespaces, etc.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, DoesNotBreakSemiAfterClassDecl) {
  verifyFormat("class A {};");
}

TEST_F(FormatTest, UnderstandsAccessSpecifiers) {
  verifyFormat("class A {\n"
               "public:\n"
               "public: // comment\n"
               "protected:\n"
               "private:\n"
               "  void f() {}\n"
               "};");
  verifyGoogleFormat("class A {\n"
                     " public:\n"
                     " protected:\n"
                     " private:\n"
                     "  void f() {}\n"
                     "};");
  verifyFormat("class A {\n"
               "public slots:\n"
               "  void f1() {}\n"
               "public Q_SLOTS:\n"
               "  void f2() {}\n"
               "protected slots:\n"
               "  void f3() {}\n"
               "protected Q_SLOTS:\n"
               "  void f4() {}\n"
               "private slots:\n"
               "  void f5() {}\n"
               "private Q_SLOTS:\n"
               "  void f6() {}\n"
               "signals:\n"
               "  void g1();\n"
               "Q_SIGNALS:\n"
               "  void g2();\n"
               "};");

  // Don't interpret 'signals' the wrong way.
  verifyFormat("signals.set();");
  verifyFormat("for (Signals signals : f()) {\n}");
  verifyFormat("{\n"
               "  signals.set(); // This needs indentation.\n"
               "}");
}

TEST_F(FormatTest, SeparatesLogicalBlocks) {
  EXPECT_EQ("class A {\n"
            "public:\n"
            "  void f();\n"
            "\n"
            "private:\n"
            "  void g() {}\n"
            "  // test\n"
            "protected:\n"
            "  int h;\n"
            "};",
            format("class A {\n"
                   "public:\n"
                   "void f();\n"
                   "private:\n"
                   "void g() {}\n"
                   "// test\n"
                   "protected:\n"
                   "int h;\n"
                   "};"));
  EXPECT_EQ("class A {\n"
            "protected:\n"
            "public:\n"
            "  void f();\n"
            "};",
            format("class A {\n"
                   "protected:\n"
                   "\n"
                   "public:\n"
                   "\n"
                   "  void f();\n"
                   "};"));

  // Even ensure proper spacing inside macros.
  EXPECT_EQ("#define B     \\\n"
            "  class A {   \\\n"
            "   protected: \\\n"
            "   public:    \\\n"
            "    void f(); \\\n"
            "  };",
            format("#define B     \\\n"
                   "  class A {   \\\n"
                   "   protected: \\\n"
                   "              \\\n"
                   "   public:    \\\n"
                   "              \\\n"
                   "    void f(); \\\n"
                   "  };",
                   getGoogleStyle()));
  // But don't remove empty lines after macros ending in access specifiers.
  EXPECT_EQ("#define A private:\n"
            "\n"
            "int i;",
            format("#define A         private:\n"
                   "\n"
                   "int              i;"));
}

TEST_F(FormatTest, FormatsClasses) {
  verifyFormat("class A : public B {};");
  verifyFormat("class A : public ::B {};");

  verifyFormat(
      "class AAAAAAAAAAAAAAAAAAAA : public BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,\n"
      "                             public CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC {};");
  verifyFormat("class AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
               "    : public BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,\n"
               "      public CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC {};");
  verifyFormat(
      "class A : public B, public C, public D, public E, public F {};");
  verifyFormat("class AAAAAAAAAAAA : public B,\n"
               "                     public C,\n"
               "                     public D,\n"
               "                     public E,\n"
               "                     public F,\n"
               "                     public G {};");

  verifyFormat("class\n"
               "    ReallyReallyLongClassName {\n"
               "  int i;\n"
               "};",
               getLLVMStyleWithColumns(32));
  verifyFormat("struct aaaaaaaaaaaaa : public aaaaaaaaaaaaaaaaaaa< // break\n"
               "                           aaaaaaaaaaaaaaaa> {};");
  verifyFormat("struct aaaaaaaaaaaaaaaaaaaa\n"
               "    : public aaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaaaaaa,\n"
               "                                 aaaaaaaaaaaaaaaaaaaaaa> {};");
  verifyFormat("template <class R, class C>\n"
               "struct Aaaaaaaaaaaaaaaaa<R (C::*)(int) const>\n"
               "    : Aaaaaaaaaaaaaaaaa<R (C::*)(int)> {};");
  verifyFormat("class ::A::B {};");
}

TEST_F(FormatTest, FormatsVariableDeclarationsAfterStructOrClass) {
  verifyFormat("class A {\n} a, b;");
  verifyFormat("struct A {\n} a, b;");
  verifyFormat("union A {\n} a;");
}

TEST_F(FormatTest, FormatsEnum) {
  verifyFormat("enum {\n"
               "  Zero,\n"
               "  One = 1,\n"
               "  Two = One + 1,\n"
               "  Three = (One + Two),\n"
               "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
               "  Five = (One, Two, Three, Four, 5)\n"
               "};");
  verifyGoogleFormat("enum {\n"
                     "  Zero,\n"
                     "  One = 1,\n"
                     "  Two = One + 1,\n"
                     "  Three = (One + Two),\n"
                     "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
                     "  Five = (One, Two, Three, Four, 5)\n"
                     "};");
  verifyFormat("enum Enum {};");
  verifyFormat("enum {};");
  verifyFormat("enum X E {} d;");
  verifyFormat("enum __attribute__((...)) E {} d;");
  verifyFormat("enum __declspec__((...)) E {} d;");
  verifyFormat("enum {\n"
               "  Bar = Foo<int, int>::value\n"
               "};",
               getLLVMStyleWithColumns(30));

  verifyFormat("enum ShortEnum { A, B, C };");
  verifyGoogleFormat("enum ShortEnum { A, B, C };");

  EXPECT_EQ("enum KeepEmptyLines {\n"
            "  ONE,\n"
            "\n"
            "  TWO,\n"
            "\n"
            "  THREE\n"
            "}",
            format("enum KeepEmptyLines {\n"
                   "  ONE,\n"
                   "\n"
                   "  TWO,\n"
                   "\n"
                   "\n"
                   "  THREE\n"
                   "}"));
  verifyFormat("enum E { // comment\n"
               "  ONE,\n"
               "  TWO\n"
               "};\n"
               "int i;");
  // Not enums.
  verifyFormat("enum X f() {\n"
               "  a();\n"
               "  return 42;\n"
               "}");
  verifyFormat("enum X Type::f() {\n"
               "  a();\n"
               "  return 42;\n"
               "}");
  verifyFormat("enum ::X f() {\n"
               "  a();\n"
               "  return 42;\n"
               "}");
  verifyFormat("enum ns::X f() {\n"
               "  a();\n"
               "  return 42;\n"
               "}");
}

TEST_F(FormatTest, FormatsEnumsWithErrors) {
  verifyFormat("enum Type {\n"
               "  One = 0; // These semicolons should be commas.\n"
               "  Two = 1;\n"
               "};");
  verifyFormat("namespace n {\n"
               "enum Type {\n"
               "  One,\n"
               "  Two, // missing };\n"
               "  int i;\n"
               "}\n"
               "void g() {}");
}

TEST_F(FormatTest, FormatsEnumStruct) {
  verifyFormat("enum struct {\n"
               "  Zero,\n"
               "  One = 1,\n"
               "  Two = One + 1,\n"
               "  Three = (One + Two),\n"
               "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
               "  Five = (One, Two, Three, Four, 5)\n"
               "};");
  verifyFormat("enum struct Enum {};");
  verifyFormat("enum struct {};");
  verifyFormat("enum struct X E {} d;");
  verifyFormat("enum struct __attribute__((...)) E {} d;");
  verifyFormat("enum struct __declspec__((...)) E {} d;");
  verifyFormat("enum struct X f() {\n  a();\n  return 42;\n}");
}

TEST_F(FormatTest, FormatsEnumClass) {
  verifyFormat("enum class {\n"
               "  Zero,\n"
               "  One = 1,\n"
               "  Two = One + 1,\n"
               "  Three = (One + Two),\n"
               "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
               "  Five = (One, Two, Three, Four, 5)\n"
               "};");
  verifyFormat("enum class Enum {};");
  verifyFormat("enum class {};");
  verifyFormat("enum class X E {} d;");
  verifyFormat("enum class __attribute__((...)) E {} d;");
  verifyFormat("enum class __declspec__((...)) E {} d;");
  verifyFormat("enum class X f() {\n  a();\n  return 42;\n}");
}

TEST_F(FormatTest, FormatsEnumTypes) {
  verifyFormat("enum X : int {\n"
               "  A, // Force multiple lines.\n"
               "  B\n"
               "};");
  verifyFormat("enum X : int { A, B };");
  verifyFormat("enum X : std::uint32_t { A, B };");
}

TEST_F(FormatTest, FormatsNSEnums) {
  verifyGoogleFormat("typedef NS_ENUM(NSInteger, SomeName) { AAA, BBB }");
  verifyGoogleFormat("typedef NS_ENUM(NSInteger, MyType) {\n"
                     "  // Information about someDecentlyLongValue.\n"
                     "  someDecentlyLongValue,\n"
                     "  // Information about anotherDecentlyLongValue.\n"
                     "  anotherDecentlyLongValue,\n"
                     "  // Information about aThirdDecentlyLongValue.\n"
                     "  aThirdDecentlyLongValue\n"
                     "};");
  verifyGoogleFormat("typedef NS_OPTIONS(NSInteger, MyType) {\n"
                     "  a = 1,\n"
                     "  b = 2,\n"
                     "  c = 3,\n"
                     "};");
  verifyGoogleFormat("typedef CF_ENUM(NSInteger, MyType) {\n"
                     "  a = 1,\n"
                     "  b = 2,\n"
                     "  c = 3,\n"
                     "};");
  verifyGoogleFormat("typedef CF_OPTIONS(NSInteger, MyType) {\n"
                     "  a = 1,\n"
                     "  b = 2,\n"
                     "  c = 3,\n"
                     "};");
}

TEST_F(FormatTest, FormatsBitfields) {
  verifyFormat("struct Bitfields {\n"
               "  unsigned sClass : 8;\n"
               "  unsigned ValueKind : 2;\n"
               "};");
  verifyFormat("struct A {\n"
               "  int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa : 1,\n"
               "      bbbbbbbbbbbbbbbbbbbbbbbbb;\n"
               "};");
  verifyFormat("struct MyStruct {\n"
               "  uchar data;\n"
               "  uchar : 8;\n"
               "  uchar : 8;\n"
               "  uchar other;\n"
               "};");
}

TEST_F(FormatTest, FormatsNamespaces) {
  verifyFormat("namespace some_namespace {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}");
  verifyFormat("namespace {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}");
  verifyFormat("inline namespace X {\n"
               "class A {};\n"
               "void f() { f(); }\n"
               "}");
  verifyFormat("using namespace some_namespace;\n"
               "class A {};\n"
               "void f() { f(); }");

  // This code is more common than we thought; if we
  // layout this correctly the semicolon will go into
  // its own line, which is undesirable.
  verifyFormat("namespace {};");
  verifyFormat("namespace {\n"
               "class A {};\n"
               "};");

  verifyFormat("namespace {\n"
               "int SomeVariable = 0; // comment\n"
               "} // namespace");
  EXPECT_EQ("#ifndef HEADER_GUARD\n"
            "#define HEADER_GUARD\n"
            "namespace my_namespace {\n"
            "int i;\n"
            "} // my_namespace\n"
            "#endif // HEADER_GUARD",
            format("#ifndef HEADER_GUARD\n"
                   " #define HEADER_GUARD\n"
                   "   namespace my_namespace {\n"
                   "int i;\n"
                   "}    // my_namespace\n"
                   "#endif    // HEADER_GUARD"));

  EXPECT_EQ("namespace A::B {\n"
            "class C {};\n"
            "}",
            format("namespace A::B {\n"
                   "class C {};\n"
                   "}"));

  FormatStyle Style = getLLVMStyle();
  Style.NamespaceIndentation = FormatStyle::NI_All;
  EXPECT_EQ("namespace out {\n"
            "  int i;\n"
            "  namespace in {\n"
            "    int i;\n"
            "  } // namespace\n"
            "} // namespace",
            format("namespace out {\n"
                   "int i;\n"
                   "namespace in {\n"
                   "int i;\n"
                   "} // namespace\n"
                   "} // namespace",
                   Style));

  Style.NamespaceIndentation = FormatStyle::NI_Inner;
  EXPECT_EQ("namespace out {\n"
            "int i;\n"
            "namespace in {\n"
            "  int i;\n"
            "} // namespace\n"
            "} // namespace",
            format("namespace out {\n"
                   "int i;\n"
                   "namespace in {\n"
                   "int i;\n"
                   "} // namespace\n"
                   "} // namespace",
                   Style));
}

TEST_F(FormatTest, FormatsExternC) { verifyFormat("extern \"C\" {\nint a;"); }

TEST_F(FormatTest, FormatsInlineASM) {
  verifyFormat("asm(\"xyz\" : \"=a\"(a), \"=d\"(b) : \"a\"(data));");
  verifyFormat("asm(\"nop\" ::: \"memory\");");
  verifyFormat(
      "asm(\"movq\\t%%rbx, %%rsi\\n\\t\"\n"
      "    \"cpuid\\n\\t\"\n"
      "    \"xchgq\\t%%rbx, %%rsi\\n\\t\"\n"
      "    : \"=a\"(*rEAX), \"=S\"(*rEBX), \"=c\"(*rECX), \"=d\"(*rEDX)\n"
      "    : \"a\"(value));");
  EXPECT_EQ(
      "void NS_InvokeByIndex(void *that, unsigned int methodIndex) {\n"
      "  __asm {\n"
      "        mov     edx,[that] // vtable in edx\n"
      "        mov     eax,methodIndex\n"
      "        call    [edx][eax*4] // stdcall\n"
      "  }\n"
      "}",
      format("void NS_InvokeByIndex(void *that,   unsigned int methodIndex) {\n"
             "    __asm {\n"
             "        mov     edx,[that] // vtable in edx\n"
             "        mov     eax,methodIndex\n"
             "        call    [edx][eax*4] // stdcall\n"
             "    }\n"
             "}"));
  EXPECT_EQ("_asm {\n"
            "  xor eax, eax;\n"
            "  cpuid;\n"
            "}",
            format("_asm {\n"
                   "  xor eax, eax;\n"
                   "  cpuid;\n"
                   "}"));
  verifyFormat("void function() {\n"
               "  // comment\n"
               "  asm(\"\");\n"
               "}");
  EXPECT_EQ("__asm {\n"
            "}\n"
            "int i;",
            format("__asm   {\n"
                   "}\n"
                   "int   i;"));
}

TEST_F(FormatTest, FormatTryCatch) {
  verifyFormat("try {\n"
               "  throw a * b;\n"
               "} catch (int a) {\n"
               "  // Do nothing.\n"
               "} catch (...) {\n"
               "  exit(42);\n"
               "}");

  // Function-level try statements.
  verifyFormat("int f() try { return 4; } catch (...) {\n"
               "  return 5;\n"
               "}");
  verifyFormat("class A {\n"
               "  int a;\n"
               "  A() try : a(0) {\n"
               "  } catch (...) {\n"
               "    throw;\n"
               "  }\n"
               "};\n");

  // Incomplete try-catch blocks.
  verifyIncompleteFormat("try {} catch (");
}

TEST_F(FormatTest, FormatSEHTryCatch) {
  verifyFormat("__try {\n"
               "  int a = b * c;\n"
               "} __except (EXCEPTION_EXECUTE_HANDLER) {\n"
               "  // Do nothing.\n"
               "}");

  verifyFormat("__try {\n"
               "  int a = b * c;\n"
               "} __finally {\n"
               "  // Do nothing.\n"
               "}");

  verifyFormat("DEBUG({\n"
               "  __try {\n"
               "  } __finally {\n"
               "  }\n"
               "});\n");
}

TEST_F(FormatTest, IncompleteTryCatchBlocks) {
  verifyFormat("try {\n"
               "  f();\n"
               "} catch {\n"
               "  g();\n"
               "}");
  verifyFormat("try {\n"
               "  f();\n"
               "} catch (A a) MACRO(x) {\n"
               "  g();\n"
               "} catch (B b) MACRO(x) {\n"
               "  g();\n"
               "}");
}

TEST_F(FormatTest, FormatTryCatchBraceStyles) {
  FormatStyle Style = getLLVMStyle();
  for (auto BraceStyle : {FormatStyle::BS_Attach, FormatStyle::BS_Mozilla,
                          FormatStyle::BS_WebKit}) {
    Style.BreakBeforeBraces = BraceStyle;
    verifyFormat("try {\n"
                 "  // something\n"
                 "} catch (...) {\n"
                 "  // something\n"
                 "}",
                 Style);
  }
  Style.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  verifyFormat("try {\n"
               "  // something\n"
               "}\n"
               "catch (...) {\n"
               "  // something\n"
               "}",
               Style);
  verifyFormat("__try {\n"
               "  // something\n"
               "}\n"
               "__finally {\n"
               "  // something\n"
               "}",
               Style);
  verifyFormat("@try {\n"
               "  // something\n"
               "}\n"
               "@finally {\n"
               "  // something\n"
               "}",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  verifyFormat("try\n"
               "{\n"
               "  // something\n"
               "}\n"
               "catch (...)\n"
               "{\n"
               "  // something\n"
               "}",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_GNU;
  verifyFormat("try\n"
               "  {\n"
               "    // something\n"
               "  }\n"
               "catch (...)\n"
               "  {\n"
               "    // something\n"
               "  }",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.BeforeCatch = true;
  verifyFormat("try {\n"
               "  // something\n"
               "}\n"
               "catch (...) {\n"
               "  // something\n"
               "}",
               Style);
}

TEST_F(FormatTest, FormatObjCTryCatch) {
  verifyFormat("@try {\n"
               "  f();\n"
               "} @catch (NSException e) {\n"
               "  @throw;\n"
               "} @finally {\n"
               "  exit(42);\n"
               "}");
  verifyFormat("DEBUG({\n"
               "  @try {\n"
               "  } @finally {\n"
               "  }\n"
               "});\n");
}

TEST_F(FormatTest, FormatObjCAutoreleasepool) {
  FormatStyle Style = getLLVMStyle();
  verifyFormat("@autoreleasepool {\n"
               "  f();\n"
               "}\n"
               "@autoreleasepool {\n"
               "  f();\n"
               "}\n",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  verifyFormat("@autoreleasepool\n"
               "{\n"
               "  f();\n"
               "}\n"
               "@autoreleasepool\n"
               "{\n"
               "  f();\n"
               "}\n",
               Style);
}

TEST_F(FormatTest, StaticInitializers) {
  verifyFormat("static SomeClass SC = {1, 'a'};");

  verifyFormat("static SomeClass WithALoooooooooooooooooooongName = {\n"
               "    100000000, "
               "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"};");

  // Here, everything other than the "}" would fit on a line.
  verifyFormat("static int LooooooooooooooooooooooooongVariable[1] = {\n"
               "    10000000000000000000000000};");
  EXPECT_EQ("S s = {a,\n"
            "\n"
            "       b};",
            format("S s = {\n"
                   "  a,\n"
                   "\n"
                   "  b\n"
                   "};"));

  // FIXME: This would fit into the column limit if we'd fit "{ {" on the first
  // line. However, the formatting looks a bit off and this probably doesn't
  // happen often in practice.
  verifyFormat("static int Variable[1] = {\n"
               "    {1000000000000000000000000000000000000}};",
               getLLVMStyleWithColumns(40));
}

TEST_F(FormatTest, DesignatedInitializers) {
  verifyFormat("const struct A a = {.a = 1, .b = 2};");
  verifyFormat("const struct A a = {.aaaaaaaaaa = 1,\n"
               "                    .bbbbbbbbbb = 2,\n"
               "                    .cccccccccc = 3,\n"
               "                    .dddddddddd = 4,\n"
               "                    .eeeeeeeeee = 5};");
  verifyFormat("const struct Aaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaa = {\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaa = 1,\n"
               "    .bbbbbbbbbbbbbbbbbbbbbbbbbbb = 2,\n"
               "    .ccccccccccccccccccccccccccc = 3,\n"
               "    .ddddddddddddddddddddddddddd = 4,\n"
               "    .eeeeeeeeeeeeeeeeeeeeeeeeeee = 5};");

  verifyGoogleFormat("const struct A a = {.a = 1, .b = 2};");
}

TEST_F(FormatTest, NestedStaticInitializers) {
  verifyFormat("static A x = {{{}}};\n");
  verifyFormat("static A x = {{{init1, init2, init3, init4},\n"
               "               {init1, init2, init3, init4}}};",
               getLLVMStyleWithColumns(50));

  verifyFormat("somes Status::global_reps[3] = {\n"
               "    {kGlobalRef, OK_CODE, NULL, NULL, NULL},\n"
               "    {kGlobalRef, CANCELLED_CODE, NULL, NULL, NULL},\n"
               "    {kGlobalRef, UNKNOWN_CODE, NULL, NULL, NULL}};",
               getLLVMStyleWithColumns(60));
  verifyGoogleFormat("SomeType Status::global_reps[3] = {\n"
                     "    {kGlobalRef, OK_CODE, NULL, NULL, NULL},\n"
                     "    {kGlobalRef, CANCELLED_CODE, NULL, NULL, NULL},\n"
                     "    {kGlobalRef, UNKNOWN_CODE, NULL, NULL, NULL}};");
  verifyFormat("CGRect cg_rect = {{rect.fLeft, rect.fTop},\n"
               "                  {rect.fRight - rect.fLeft, rect.fBottom - "
               "rect.fTop}};");

  verifyFormat(
      "SomeArrayOfSomeType a = {\n"
      "    {{1, 2, 3},\n"
      "     {1, 2, 3},\n"
      "     {111111111111111111111111111111, 222222222222222222222222222222,\n"
      "      333333333333333333333333333333},\n"
      "     {1, 2, 3},\n"
      "     {1, 2, 3}}};");
  verifyFormat(
      "SomeArrayOfSomeType a = {\n"
      "    {{1, 2, 3}},\n"
      "    {{1, 2, 3}},\n"
      "    {{111111111111111111111111111111, 222222222222222222222222222222,\n"
      "      333333333333333333333333333333}},\n"
      "    {{1, 2, 3}},\n"
      "    {{1, 2, 3}}};");

  verifyFormat("struct {\n"
               "  unsigned bit;\n"
               "  const char *const name;\n"
               "} kBitsToOs[] = {{kOsMac, \"Mac\"},\n"
               "                 {kOsWin, \"Windows\"},\n"
               "                 {kOsLinux, \"Linux\"},\n"
               "                 {kOsCrOS, \"Chrome OS\"}};");
  verifyFormat("struct {\n"
               "  unsigned bit;\n"
               "  const char *const name;\n"
               "} kBitsToOs[] = {\n"
               "    {kOsMac, \"Mac\"},\n"
               "    {kOsWin, \"Windows\"},\n"
               "    {kOsLinux, \"Linux\"},\n"
               "    {kOsCrOS, \"Chrome OS\"},\n"
               "};");
}

TEST_F(FormatTest, FormatsSmallMacroDefinitionsInSingleLine) {
  verifyFormat("#define ALooooooooooooooooooooooooooooooooooooooongMacro("
               "                      \\\n"
               "    aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)");
}

TEST_F(FormatTest, DoesNotBreakPureVirtualFunctionDefinition) {
  verifyFormat("virtual void write(ELFWriter *writerrr,\n"
               "                   OwningPtr<FileOutputBuffer> &buffer) = 0;");

  // Do break defaulted and deleted functions.
  verifyFormat("virtual void ~Deeeeeeeestructor() =\n"
               "    default;",
               getLLVMStyleWithColumns(40));
  verifyFormat("virtual void ~Deeeeeeeestructor() =\n"
               "    delete;",
               getLLVMStyleWithColumns(40));
}

TEST_F(FormatTest, BreaksStringLiteralsOnlyInDefine) {
  verifyFormat("# 1111 \"/aaaaaaaaa/aaaaaaaaaaaaaaaaaaa/aaaaaaaa.cpp\" 2 3",
               getLLVMStyleWithColumns(40));
  verifyFormat("#line 11111 \"/aaaaaaaaa/aaaaaaaaaaaaaaaaaaa/aaaaaaaa.cpp\"",
               getLLVMStyleWithColumns(40));
  EXPECT_EQ("#define Q                              \\\n"
            "  \"/aaaaaaaaa/aaaaaaaaaaaaaaaaaaa/\"    \\\n"
            "  \"aaaaaaaa.cpp\"",
            format("#define Q \"/aaaaaaaaa/aaaaaaaaaaaaaaaaaaa/aaaaaaaa.cpp\"",
                   getLLVMStyleWithColumns(40)));
}

TEST_F(FormatTest, UnderstandsLinePPDirective) {
  EXPECT_EQ("# 123 \"A string literal\"",
            format("   #     123    \"A string literal\""));
}

TEST_F(FormatTest, LayoutUnknownPPDirective) {
  EXPECT_EQ("#;", format("#;"));
  verifyFormat("#\n;\n;\n;");
}

TEST_F(FormatTest, UnescapedEndOfLineEndsPPDirective) {
  EXPECT_EQ("#line 42 \"test\"\n",
            format("#  \\\n  line  \\\n  42  \\\n  \"test\"\n"));
  EXPECT_EQ("#define A B\n", format("#  \\\n define  \\\n    A  \\\n       B\n",
                                    getLLVMStyleWithColumns(12)));
}

TEST_F(FormatTest, EndOfFileEndsPPDirective) {
  EXPECT_EQ("#line 42 \"test\"",
            format("#  \\\n  line  \\\n  42  \\\n  \"test\""));
  EXPECT_EQ("#define A B", format("#  \\\n define  \\\n    A  \\\n       B"));
}

TEST_F(FormatTest, DoesntRemoveUnknownTokens) {
  verifyFormat("#define A \\x20");
  verifyFormat("#define A \\ x20");
  EXPECT_EQ("#define A \\ x20", format("#define A \\   x20"));
  verifyFormat("#define A ''");
  verifyFormat("#define A ''qqq");
  verifyFormat("#define A `qqq");
  verifyFormat("f(\"aaaa, bbbb, \"\\\"ccccc\\\"\");");
  EXPECT_EQ("const char *c = STRINGIFY(\n"
            "\\na : b);",
            format("const char * c = STRINGIFY(\n"
                   "\\na : b);"));

  verifyFormat("a\r\\");
  verifyFormat("a\v\\");
  verifyFormat("a\f\\");
}

TEST_F(FormatTest, IndentsPPDirectiveInReducedSpace) {
  verifyFormat("#define A(BB)", getLLVMStyleWithColumns(13));
  verifyFormat("#define A( \\\n    BB)", getLLVMStyleWithColumns(12));
  verifyFormat("#define A( \\\n    A, B)", getLLVMStyleWithColumns(12));
  // FIXME: We never break before the macro name.
  verifyFormat("#define AA( \\\n    B)", getLLVMStyleWithColumns(12));

  verifyFormat("#define A A\n#define A A");
  verifyFormat("#define A(X) A\n#define A A");

  verifyFormat("#define Something Other", getLLVMStyleWithColumns(23));
  verifyFormat("#define Something    \\\n  Other", getLLVMStyleWithColumns(22));
}

TEST_F(FormatTest, HandlePreprocessorDirectiveContext) {
  EXPECT_EQ("// somecomment\n"
            "#include \"a.h\"\n"
            "#define A(  \\\n"
            "    A, B)\n"
            "#include \"b.h\"\n"
            "// somecomment\n",
            format("  // somecomment\n"
                   "  #include \"a.h\"\n"
                   "#define A(A,\\\n"
                   "    B)\n"
                   "    #include \"b.h\"\n"
                   " // somecomment\n",
                   getLLVMStyleWithColumns(13)));
}

TEST_F(FormatTest, LayoutSingleHash) { EXPECT_EQ("#\na;", format("#\na;")); }

TEST_F(FormatTest, LayoutCodeInMacroDefinitions) {
  EXPECT_EQ("#define A    \\\n"
            "  c;         \\\n"
            "  e;\n"
            "f;",
            format("#define A c; e;\n"
                   "f;",
                   getLLVMStyleWithColumns(14)));
}

TEST_F(FormatTest, LayoutRemainingTokens) { EXPECT_EQ("{}", format("{}")); }

TEST_F(FormatTest, MacroDefinitionInsideStatement) {
  EXPECT_EQ("int x,\n"
            "#define A\n"
            "    y;",
            format("int x,\n#define A\ny;"));
}

TEST_F(FormatTest, HashInMacroDefinition) {
  EXPECT_EQ("#define A(c) L#c", format("#define A(c) L#c", getLLVMStyle()));
  verifyFormat("#define A \\\n  b #c;", getLLVMStyleWithColumns(11));
  verifyFormat("#define A  \\\n"
               "  {        \\\n"
               "    f(#c); \\\n"
               "  }",
               getLLVMStyleWithColumns(11));

  verifyFormat("#define A(X)         \\\n"
               "  void function##X()",
               getLLVMStyleWithColumns(22));

  verifyFormat("#define A(a, b, c)   \\\n"
               "  void a##b##c()",
               getLLVMStyleWithColumns(22));

  verifyFormat("#define A void # ## #", getLLVMStyleWithColumns(22));
}

TEST_F(FormatTest, RespectWhitespaceInMacroDefinitions) {
  EXPECT_EQ("#define A (x)", format("#define A (x)"));
  EXPECT_EQ("#define A(x)", format("#define A(x)"));
}

TEST_F(FormatTest, EmptyLinesInMacroDefinitions) {
  EXPECT_EQ("#define A b;", format("#define A \\\n"
                                   "          \\\n"
                                   "  b;",
                                   getLLVMStyleWithColumns(25)));
  EXPECT_EQ("#define A \\\n"
            "          \\\n"
            "  a;      \\\n"
            "  b;",
            format("#define A \\\n"
                   "          \\\n"
                   "  a;      \\\n"
                   "  b;",
                   getLLVMStyleWithColumns(11)));
  EXPECT_EQ("#define A \\\n"
            "  a;      \\\n"
            "          \\\n"
            "  b;",
            format("#define A \\\n"
                   "  a;      \\\n"
                   "          \\\n"
                   "  b;",
                   getLLVMStyleWithColumns(11)));
}

TEST_F(FormatTest, MacroDefinitionsWithIncompleteCode) {
  verifyIncompleteFormat("#define A :");
  verifyFormat("#define SOMECASES  \\\n"
               "  case 1:          \\\n"
               "  case 2\n",
               getLLVMStyleWithColumns(20));
  verifyFormat("#define A template <typename T>");
  verifyIncompleteFormat("#define STR(x) #x\n"
                         "f(STR(this_is_a_string_literal{));");
  verifyFormat("#pragma omp threadprivate( \\\n"
               "    y)), // expected-warning",
               getLLVMStyleWithColumns(28));
  verifyFormat("#d, = };");
  verifyFormat("#if \"a");
  verifyIncompleteFormat("({\n"
                         "#define b     \\\n"
                         "  }           \\\n"
                         "  a\n"
                         "a",
                         getLLVMStyleWithColumns(15));
  verifyFormat("#define A     \\\n"
               "  {           \\\n"
               "    {\n"
               "#define B     \\\n"
               "  }           \\\n"
               "  }",
               getLLVMStyleWithColumns(15));
  verifyNoCrash("#if a\na(\n#else\n#endif\n{a");
  verifyNoCrash("a={0,1\n#if a\n#else\n;\n#endif\n}");
  verifyNoCrash("#if a\na(\n#else\n#endif\n) a {a,b,c,d,f,g};");
  verifyNoCrash("#ifdef A\n a(\n #else\n #endif\n) = []() {      \n)}");
}

TEST_F(FormatTest, MacrosWithoutTrailingSemicolon) {
  verifyFormat("SOME_TYPE_NAME abc;"); // Gated on the newline.
  EXPECT_EQ("class A : public QObject {\n"
            "  Q_OBJECT\n"
            "\n"
            "  A() {}\n"
            "};",
            format("class A  :  public QObject {\n"
                   "     Q_OBJECT\n"
                   "\n"
                   "  A() {\n}\n"
                   "}  ;"));
  EXPECT_EQ("MACRO\n"
            "/*static*/ int i;",
            format("MACRO\n"
                   " /*static*/ int   i;"));
  EXPECT_EQ("SOME_MACRO\n"
            "namespace {\n"
            "void f();\n"
            "}",
            format("SOME_MACRO\n"
                   "  namespace    {\n"
                   "void   f(  );\n"
                   "}"));
  // Only if the identifier contains at least 5 characters.
  EXPECT_EQ("HTTP f();", format("HTTP\nf();"));
  EXPECT_EQ("MACRO\nf();", format("MACRO\nf();"));
  // Only if everything is upper case.
  EXPECT_EQ("class A : public QObject {\n"
            "  Q_Object A() {}\n"
            "};",
            format("class A  :  public QObject {\n"
                   "     Q_Object\n"
                   "  A() {\n}\n"
                   "}  ;"));

  // Only if the next line can actually start an unwrapped line.
  EXPECT_EQ("SOME_WEIRD_LOG_MACRO << SomeThing;",
            format("SOME_WEIRD_LOG_MACRO\n"
                   "<< SomeThing;"));

  verifyFormat("VISIT_GL_CALL(GenBuffers, void, (GLsizei n, GLuint* buffers), "
               "(n, buffers))\n",
               getChromiumStyle(FormatStyle::LK_Cpp));
}

TEST_F(FormatTest, MacroCallsWithoutTrailingSemicolon) {
  EXPECT_EQ("INITIALIZE_PASS_BEGIN(ScopDetection, \"polly-detect\")\n"
            "INITIALIZE_AG_DEPENDENCY(AliasAnalysis)\n"
            "INITIALIZE_PASS_DEPENDENCY(DominatorTree)\n"
            "class X {};\n"
            "INITIALIZE_PASS_END(ScopDetection, \"polly-detect\")\n"
            "int *createScopDetectionPass() { return 0; }",
            format("  INITIALIZE_PASS_BEGIN(ScopDetection, \"polly-detect\")\n"
                   "  INITIALIZE_AG_DEPENDENCY(AliasAnalysis)\n"
                   "  INITIALIZE_PASS_DEPENDENCY(DominatorTree)\n"
                   "  class X {};\n"
                   "  INITIALIZE_PASS_END(ScopDetection, \"polly-detect\")\n"
                   "  int *createScopDetectionPass() { return 0; }"));
  // FIXME: We could probably treat IPC_BEGIN_MESSAGE_MAP/IPC_END_MESSAGE_MAP as
  // braces, so that inner block is indented one level more.
  EXPECT_EQ("int q() {\n"
            "  IPC_BEGIN_MESSAGE_MAP(WebKitTestController, message)\n"
            "  IPC_MESSAGE_HANDLER(xxx, qqq)\n"
            "  IPC_END_MESSAGE_MAP()\n"
            "}",
            format("int q() {\n"
                   "  IPC_BEGIN_MESSAGE_MAP(WebKitTestController, message)\n"
                   "    IPC_MESSAGE_HANDLER(xxx, qqq)\n"
                   "  IPC_END_MESSAGE_MAP()\n"
                   "}"));

  // Same inside macros.
  EXPECT_EQ("#define LIST(L) \\\n"
            "  L(A)          \\\n"
            "  L(B)          \\\n"
            "  L(C)",
            format("#define LIST(L) \\\n"
                   "  L(A) \\\n"
                   "  L(B) \\\n"
                   "  L(C)",
                   getGoogleStyle()));

  // These must not be recognized as macros.
  EXPECT_EQ("int q() {\n"
            "  f(x);\n"
            "  f(x) {}\n"
            "  f(x)->g();\n"
            "  f(x)->*g();\n"
            "  f(x).g();\n"
            "  f(x) = x;\n"
            "  f(x) += x;\n"
            "  f(x) -= x;\n"
            "  f(x) *= x;\n"
            "  f(x) /= x;\n"
            "  f(x) %= x;\n"
            "  f(x) &= x;\n"
            "  f(x) |= x;\n"
            "  f(x) ^= x;\n"
            "  f(x) >>= x;\n"
            "  f(x) <<= x;\n"
            "  f(x)[y].z();\n"
            "  LOG(INFO) << x;\n"
            "  ifstream(x) >> x;\n"
            "}\n",
            format("int q() {\n"
                   "  f(x)\n;\n"
                   "  f(x)\n {}\n"
                   "  f(x)\n->g();\n"
                   "  f(x)\n->*g();\n"
                   "  f(x)\n.g();\n"
                   "  f(x)\n = x;\n"
                   "  f(x)\n += x;\n"
                   "  f(x)\n -= x;\n"
                   "  f(x)\n *= x;\n"
                   "  f(x)\n /= x;\n"
                   "  f(x)\n %= x;\n"
                   "  f(x)\n &= x;\n"
                   "  f(x)\n |= x;\n"
                   "  f(x)\n ^= x;\n"
                   "  f(x)\n >>= x;\n"
                   "  f(x)\n <<= x;\n"
                   "  f(x)\n[y].z();\n"
                   "  LOG(INFO)\n << x;\n"
                   "  ifstream(x)\n >> x;\n"
                   "}\n"));
  EXPECT_EQ("int q() {\n"
            "  F(x)\n"
            "  if (1) {\n"
            "  }\n"
            "  F(x)\n"
            "  while (1) {\n"
            "  }\n"
            "  F(x)\n"
            "  G(x);\n"
            "  F(x)\n"
            "  try {\n"
            "    Q();\n"
            "  } catch (...) {\n"
            "  }\n"
            "}\n",
            format("int q() {\n"
                   "F(x)\n"
                   "if (1) {}\n"
                   "F(x)\n"
                   "while (1) {}\n"
                   "F(x)\n"
                   "G(x);\n"
                   "F(x)\n"
                   "try { Q(); } catch (...) {}\n"
                   "}\n"));
  EXPECT_EQ("class A {\n"
            "  A() : t(0) {}\n"
            "  A(int i) noexcept() : {}\n"
            "  A(X x)\n" // FIXME: function-level try blocks are broken.
            "  try : t(0) {\n"
            "  } catch (...) {\n"
            "  }\n"
            "};",
            format("class A {\n"
                   "  A()\n : t(0) {}\n"
                   "  A(int i)\n noexcept() : {}\n"
                   "  A(X x)\n"
                   "  try : t(0) {} catch (...) {}\n"
                   "};"));
  EXPECT_EQ("class SomeClass {\n"
            "public:\n"
            "  SomeClass() EXCLUSIVE_LOCK_FUNCTION(mu_);\n"
            "};",
            format("class SomeClass {\n"
                   "public:\n"
                   "  SomeClass()\n"
                   "  EXCLUSIVE_LOCK_FUNCTION(mu_);\n"
                   "};"));
  EXPECT_EQ("class SomeClass {\n"
            "public:\n"
            "  SomeClass()\n"
            "      EXCLUSIVE_LOCK_FUNCTION(mu_);\n"
            "};",
            format("class SomeClass {\n"
                   "public:\n"
                   "  SomeClass()\n"
                   "  EXCLUSIVE_LOCK_FUNCTION(mu_);\n"
                   "};",
                   getLLVMStyleWithColumns(40)));

  verifyFormat("MACRO(>)");
}

TEST_F(FormatTest, LayoutMacroDefinitionsStatementsSpanningBlocks) {
  verifyFormat("#define A \\\n"
               "  f({     \\\n"
               "    g();  \\\n"
               "  });",
               getLLVMStyleWithColumns(11));
}

TEST_F(FormatTest, IndentPreprocessorDirectivesAtZero) {
  EXPECT_EQ("{\n  {\n#define A\n  }\n}", format("{{\n#define A\n}}"));
}

TEST_F(FormatTest, FormatHashIfNotAtStartOfLine) {
  verifyFormat("{\n  { a #c; }\n}");
}

TEST_F(FormatTest, FormatUnbalancedStructuralElements) {
  EXPECT_EQ("#define A \\\n  {       \\\n    {\nint i;",
            format("#define A { {\nint i;", getLLVMStyleWithColumns(11)));
  EXPECT_EQ("#define A \\\n  }       \\\n  }\nint i;",
            format("#define A } }\nint i;", getLLVMStyleWithColumns(11)));
}

TEST_F(FormatTest, EscapedNewlines) {
  EXPECT_EQ(
      "#define A \\\n  int i;  \\\n  int j;",
      format("#define A \\\nint i;\\\n  int j;", getLLVMStyleWithColumns(11)));
  EXPECT_EQ("#define A\n\nint i;", format("#define A \\\n\n int i;"));
  EXPECT_EQ("template <class T> f();", format("\\\ntemplate <class T> f();"));
  EXPECT_EQ("/* \\  \\  \\\n*/", format("\\\n/* \\  \\  \\\n*/"));
  EXPECT_EQ("<a\n\\\\\n>", format("<a\n\\\\\n>"));
}

TEST_F(FormatTest, DontCrashOnBlockComments) {
  EXPECT_EQ(
      "int xxxxxxxxx; /* "
      "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\n"
      "zzzzzz\n"
      "0*/",
      format("int xxxxxxxxx;                          /* "
             "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy zzzzzz\n"
             "0*/"));
}

TEST_F(FormatTest, CalculateSpaceOnConsecutiveLinesInMacro) {
  verifyFormat("#define A \\\n"
               "  int v(  \\\n"
               "      a); \\\n"
               "  int i;",
               getLLVMStyleWithColumns(11));
}

TEST_F(FormatTest, MixingPreprocessorDirectivesAndNormalCode) {
  EXPECT_EQ(
      "#define ALooooooooooooooooooooooooooooooooooooooongMacro("
      "                      \\\n"
      "    aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)\n"
      "\n"
      "AlooooooooooooooooooooooooooooooooooooooongCaaaaaaaaaal(\n"
      "    aLooooooooooooooooooooooonPaaaaaaaaaaaaaaaaaaaaarmmmm);\n",
      format("  #define   ALooooooooooooooooooooooooooooooooooooooongMacro("
             "\\\n"
             "aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)\n"
             "  \n"
             "   AlooooooooooooooooooooooooooooooooooooooongCaaaaaaaaaal(\n"
             "  aLooooooooooooooooooooooonPaaaaaaaaaaaaaaaaaaaaarmmmm);\n"));
}

TEST_F(FormatTest, LayoutStatementsAroundPreprocessorDirectives) {
  EXPECT_EQ("int\n"
            "#define A\n"
            "    a;",
            format("int\n#define A\na;"));
  verifyFormat("functionCallTo(\n"
               "    someOtherFunction(\n"
               "        withSomeParameters, whichInSequence,\n"
               "        areLongerThanALine(andAnotherCall,\n"
               "#define A B\n"
               "                           withMoreParamters,\n"
               "                           whichStronglyInfluenceTheLayout),\n"
               "        andMoreParameters),\n"
               "    trailing);",
               getLLVMStyleWithColumns(69));
  verifyFormat("Foo::Foo()\n"
               "#ifdef BAR\n"
               "    : baz(0)\n"
               "#endif\n"
               "{\n"
               "}");
  verifyFormat("void f() {\n"
               "  if (true)\n"
               "#ifdef A\n"
               "    f(42);\n"
               "  x();\n"
               "#else\n"
               "    g();\n"
               "  x();\n"
               "#endif\n"
               "}");
  verifyFormat("void f(param1, param2,\n"
               "       param3,\n"
               "#ifdef A\n"
               "       param4(param5,\n"
               "#ifdef A1\n"
               "              param6,\n"
               "#ifdef A2\n"
               "              param7),\n"
               "#else\n"
               "              param8),\n"
               "       param9,\n"
               "#endif\n"
               "       param10,\n"
               "#endif\n"
               "       param11)\n"
               "#else\n"
               "       param12)\n"
               "#endif\n"
               "{\n"
               "  x();\n"
               "}",
               getLLVMStyleWithColumns(28));
  verifyFormat("#if 1\n"
               "int i;");
  verifyFormat("#if 1\n"
               "#endif\n"
               "#if 1\n"
               "#else\n"
               "#endif\n");
  verifyFormat("DEBUG({\n"
               "  return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;\n"
               "});\n"
               "#if a\n"
               "#else\n"
               "#endif");

  verifyIncompleteFormat("void f(\n"
                         "#if A\n"
                         "    );\n"
                         "#else\n"
                         "#endif");
}

TEST_F(FormatTest, GraciouslyHandleIncorrectPreprocessorConditions) {
  verifyFormat("#endif\n"
               "#if B");
}

TEST_F(FormatTest, FormatsJoinedLinesOnSubsequentRuns) {
  FormatStyle SingleLine = getLLVMStyle();
  SingleLine.AllowShortIfStatementsOnASingleLine = true;
  verifyFormat("#if 0\n"
               "#elif 1\n"
               "#endif\n"
               "void foo() {\n"
               "  if (test) foo2();\n"
               "}",
               SingleLine);
}

TEST_F(FormatTest, LayoutBlockInsideParens) {
  verifyFormat("functionCall({ int i; });");
  verifyFormat("functionCall({\n"
               "  int i;\n"
               "  int j;\n"
               "});");
  verifyFormat("functionCall(\n"
               "    {\n"
               "      int i;\n"
               "      int j;\n"
               "    },\n"
               "    aaaa, bbbb, cccc);");
  verifyFormat("functionA(functionB({\n"
               "            int i;\n"
               "            int j;\n"
               "          }),\n"
               "          aaaa, bbbb, cccc);");
  verifyFormat("functionCall(\n"
               "    {\n"
               "      int i;\n"
               "      int j;\n"
               "    },\n"
               "    aaaa, bbbb, // comment\n"
               "    cccc);");
  verifyFormat("functionA(functionB({\n"
               "            int i;\n"
               "            int j;\n"
               "          }),\n"
               "          aaaa, bbbb, // comment\n"
               "          cccc);");
  verifyFormat("functionCall(aaaa, bbbb, { int i; });");
  verifyFormat("functionCall(aaaa, bbbb, {\n"
               "  int i;\n"
               "  int j;\n"
               "});");
  verifyFormat(
      "Aaa(\n" // FIXME: There shouldn't be a linebreak here.
      "    {\n"
      "      int i; // break\n"
      "    },\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
      "                                     ccccccccccccccccc));");
  verifyFormat("DEBUG({\n"
               "  if (a)\n"
               "    f();\n"
               "});");
}

TEST_F(FormatTest, LayoutBlockInsideStatement) {
  EXPECT_EQ("SOME_MACRO { int i; }\n"
            "int i;",
            format("  SOME_MACRO  {int i;}  int i;"));
}

TEST_F(FormatTest, LayoutNestedBlocks) {
  verifyFormat("void AddOsStrings(unsigned bitmask) {\n"
               "  struct s {\n"
               "    int i;\n"
               "  };\n"
               "  s kBitsToOs[] = {{10}};\n"
               "  for (int i = 0; i < 10; ++i)\n"
               "    return;\n"
               "}");
  verifyFormat("call(parameter, {\n"
               "  something();\n"
               "  // Comment using all columns.\n"
               "  somethingelse();\n"
               "});",
               getLLVMStyleWithColumns(40));
  verifyFormat("DEBUG( //\n"
               "    { f(); }, a);");
  verifyFormat("DEBUG( //\n"
               "    {\n"
               "      f(); //\n"
               "    },\n"
               "    a);");

  EXPECT_EQ("call(parameter, {\n"
            "  something();\n"
            "  // Comment too\n"
            "  // looooooooooong.\n"
            "  somethingElse();\n"
            "});",
            format("call(parameter, {\n"
                   "  something();\n"
                   "  // Comment too looooooooooong.\n"
                   "  somethingElse();\n"
                   "});",
                   getLLVMStyleWithColumns(29)));
  EXPECT_EQ("DEBUG({ int i; });", format("DEBUG({ int   i; });"));
  EXPECT_EQ("DEBUG({ // comment\n"
            "  int i;\n"
            "});",
            format("DEBUG({ // comment\n"
                   "int  i;\n"
                   "});"));
  EXPECT_EQ("DEBUG({\n"
            "  int i;\n"
            "\n"
            "  // comment\n"
            "  int j;\n"
            "});",
            format("DEBUG({\n"
                   "  int  i;\n"
                   "\n"
                   "  // comment\n"
                   "  int  j;\n"
                   "});"));

  verifyFormat("DEBUG({\n"
               "  if (a)\n"
               "    return;\n"
               "});");
  verifyGoogleFormat("DEBUG({\n"
                     "  if (a) return;\n"
                     "});");
  FormatStyle Style = getGoogleStyle();
  Style.ColumnLimit = 45;
  verifyFormat("Debug(aaaaa,\n"
               "      {\n"
               "        if (aaaaaaaaaaaaaaaaaaaaaaaa) return;\n"
               "      },\n"
               "      a);",
               Style);

  verifyFormat("SomeFunction({MACRO({ return output; }), b});");

  verifyNoCrash("^{v^{a}}");
}

TEST_F(FormatTest, FormatNestedBlocksInMacros) {
  EXPECT_EQ("#define MACRO()                     \\\n"
            "  Debug(aaa, /* force line break */ \\\n"
            "        {                           \\\n"
            "          int i;                    \\\n"
            "          int j;                    \\\n"
            "        })",
            format("#define   MACRO()   Debug(aaa,  /* force line break */ \\\n"
                   "          {  int   i;  int  j;   })",
                   getGoogleStyle()));

  EXPECT_EQ("#define A                                       \\\n"
            "  [] {                                          \\\n"
            "    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(        \\\n"
            "        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx); \\\n"
            "  }",
            format("#define A [] { xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx( \\\n"
                   "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx); }",
                   getGoogleStyle()));
}

TEST_F(FormatTest, PutEmptyBlocksIntoOneLine) {
  EXPECT_EQ("{}", format("{}"));
  verifyFormat("enum E {};");
  verifyFormat("enum E {}");
}

TEST_F(FormatTest, FormatBeginBlockEndMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.MacroBlockBegin = "^[A-Z_]+_BEGIN$";
  Style.MacroBlockEnd = "^[A-Z_]+_END$";
  verifyFormat("FOO_BEGIN\n"
               "  FOO_ENTRY\n"
               "FOO_END", Style);
  verifyFormat("FOO_BEGIN\n"
               "  NESTED_FOO_BEGIN\n"
               "    NESTED_FOO_ENTRY\n"
               "  NESTED_FOO_END\n"
               "FOO_END", Style);
  verifyFormat("FOO_BEGIN(Foo, Bar)\n"
               "  int x;\n"
               "  x = 1;\n"
               "FOO_END(Baz)", Style);
}

//===----------------------------------------------------------------------===//
// Line break tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, PreventConfusingIndents) {
  verifyFormat(
      "void f() {\n"
      "  SomeLongMethodName(SomeReallyLongMethod(CallOtherReallyLongMethod(\n"
      "                         parameter, parameter, parameter)),\n"
      "                     SecondLongCall(parameter));\n"
      "}");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    [aaaaaaaaaaaaaaaaaaaaaaaa\n"
      "         [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]\n"
      "         [aaaaaaaaaaaaaaaaaaaaaaaa]];");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaa<\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>,\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaa>;");
  verifyFormat("int a = bbbb && ccc && fffff(\n"
               "#define A Just forcing a new line\n"
               "                           ddd);");
}

TEST_F(FormatTest, LineBreakingInBinaryExpressions) {
  verifyFormat(
      "bool aaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaa).aaaaaaaaaaaaaaaaaaa() ||\n"
      "    bbbbbbbb();");
  verifyFormat(
      "bool aaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaa).aaaaaaaaaaaaaaaaaaa() or\n"
      "    bbbbbbbb();");

  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa != bbbbbbbbbbbbbbbbbb &&\n"
               "    ccccccccc == ddddddddddd;");
  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa != bbbbbbbbbbbbbbbbbb and\n"
               "    ccccccccc == ddddddddddd;");
  verifyFormat(
      "bool aaaaaaaaaaaaaaaaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa not_eq bbbbbbbbbbbbbbbbbb and\n"
      "    ccccccccc == ddddddddddd;");

  verifyFormat("aaaaaa = aaaaaaa(aaaaaaa, // break\n"
               "                 aaaaaa) &&\n"
               "         bbbbbb && cccccc;");
  verifyFormat("aaaaaa = aaaaaaa(aaaaaaa, // break\n"
               "                 aaaaaa) >>\n"
               "         bbbbbb;");
  verifyFormat("aa = Whitespaces.addUntouchableComment(\n"
               "    SourceMgr.getSpellingColumnNumber(\n"
               "        TheLine.Last->FormatTok.Tok.getLocation()) -\n"
               "    1);");

  verifyFormat("if ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "     bbbbbbbbbbbbbbbbbb) && // aaaaaaaaaaaaaaaa\n"
               "    cccccc) {\n}");
  verifyFormat("b = a &&\n"
               "    // Comment\n"
               "    b.c && d;");

  // If the LHS of a comparison is not a binary expression itself, the
  // additional linebreak confuses many people.
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) > 5) {\n"
      "}");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) == 5) {\n"
      "}");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaa.aaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) == 5) {\n"
      "}");
  // Even explicit parentheses stress the precedence enough to make the
  // additional break unnecessary.
  verifyFormat("if ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) == 5) {\n"
               "}");
  // This cases is borderline, but with the indentation it is still readable.
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaa) > aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
      "}",
      getLLVMStyleWithColumns(75));

  // If the LHS is a binary expression, we should still use the additional break
  // as otherwise the formatting hides the operator precedence.
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "    5) {\n"
               "}");

  FormatStyle OnePerLine = getLLVMStyle();
  OnePerLine.BinPackParameters = false;
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}",
      OnePerLine);
}

TEST_F(FormatTest, ExpressionIndentation) {
  verifyFormat("bool value = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "                         bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb +\n"
               "                     bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb &&\n"
               "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa >\n"
               "                 ccccccccccccccccccccccccccccccccccccccccc;");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ==\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *\n"
               "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}");
  verifyFormat("if () {\n"
               "} else if (aaaaa &&\n"
               "           bbbbb > // break\n"
               "               ccccc) {\n"
               "}");

  // Presence of a trailing comment used to change indentation of b.
  verifyFormat("return aaaaaaaaaaaaaaaaaaa +\n"
               "       b;\n"
               "return aaaaaaaaaaaaaaaaaaa +\n"
               "       b; //",
               getLLVMStyleWithColumns(30));
}

TEST_F(FormatTest, ExpressionIndentationBreakingBeforeOperators) {
  // Not sure what the best system is here. Like this, the LHS can be found
  // immediately above an operator (everything with the same or a higher
  // indent). The RHS is aligned right of the operator and so compasses
  // everything until something with the same indent as the operator is found.
  // FIXME: Is this a good system?
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat(
      "bool value = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                     + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                     + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                 == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                            * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
      "                        + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
      "             && aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                        * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                    > ccccccccccccccccccccccccccccccccccccccccc;",
      Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "              * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "               * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "           + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}",
               Style);
  verifyFormat("if () {\n"
               "} else if (aaaaa\n"
               "           && bbbbb // break\n"
               "                  > ccccc) {\n"
               "}",
               Style);
  verifyFormat("return (a)\n"
               "       // comment\n"
               "       + b;",
               Style);
  verifyFormat(
      "int aaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                 * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
      "             + cc;",
      Style);

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    = aaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);

  // Forced by comments.
  verifyFormat(
      "unsigned ContentSize =\n"
      "    sizeof(int16_t)   // DWARF ARange version number\n"
      "    + sizeof(int32_t) // Offset of CU in the .debug_info section\n"
      "    + sizeof(int8_t)  // Pointer Size (in bytes)\n"
      "    + sizeof(int8_t); // Segment Size (in bytes)");

  verifyFormat("return boost::fusion::at_c<0>(iiii).second\n"
               "       == boost::fusion::at_c<1>(iiii).second;",
               Style);

  Style.ColumnLimit = 60;
  verifyFormat("zzzzzzzzzz\n"
               "    = bbbbbbbbbbbbbbbbb\n"
               "      >> aaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
}

TEST_F(FormatTest, NoOperandAlignment) {
  FormatStyle Style = getLLVMStyle();
  Style.AlignOperands = false;
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
  verifyFormat("bool value = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "            + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "    && aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "            * aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        > ccccccccccccccccccccccccccccccccccccccccc;",
               Style);

  verifyFormat("int aaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "    + cc;",
               Style);
  verifyFormat("int a = aa\n"
               "    + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "        * cccccccccccccccccccccccccccccccccccc;",
               Style);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  verifyFormat("return (a > b\n"
               "    // comment1\n"
               "    // comment2\n"
               "    || c);",
               Style);
}

TEST_F(FormatTest, BreakingBeforeNonAssigmentOperators) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_NonAssignment;
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    + bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               Style);
}

TEST_F(FormatTest, ConstructorInitializers) {
  verifyFormat("Constructor() : Initializer(FitsOnTheLine) {}");
  verifyFormat("Constructor() : Inttializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(45));
  verifyFormat("Constructor()\n"
               "    : Inttializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(44));
  verifyFormat("Constructor()\n"
               "    : Inttializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(43));

  verifyFormat("template <typename T>\n"
               "Constructor() : Initializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(45));

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}");

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}");
  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {}");
  verifyFormat("Constructor(aaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    : aaaaaaaaaa(aaaaaa) {}");

  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                               aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaa() {}");

  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}");

  verifyFormat("Constructor(int Parameter = 0)\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaa(aaaaaaaaaaaaaaaaa) {}");
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a), bbbbbbbbbbbbbbbbbbbbbbbb(b) {\n"
               "}",
               getLLVMStyleWithColumns(60));
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaa(aaaa, aaaa)) {}");

  // Here a line could be saved by splitting the second initializer onto two
  // lines, but that is not desirable.
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaa(aaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaat(aaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}");

  FormatStyle OnePerLine = getLLVMStyle();
  OnePerLine.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
               OnePerLine);
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), // Some comment\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
               OnePerLine);
  verifyFormat("MyClass::MyClass(int var)\n"
               "    : some_var_(var),            // 4 space indent\n"
               "      some_other_var_(var + 1) { // lined up\n"
               "}",
               OnePerLine);
  verifyFormat("Constructor()\n"
               "    : aaaaa(aaaaaa),\n"
               "      aaaaa(aaaaaa),\n"
               "      aaaaa(aaaaaa),\n"
               "      aaaaa(aaaaaa),\n"
               "      aaaaa(aaaaaa) {}",
               OnePerLine);
  verifyFormat("Constructor()\n"
               "    : aaaaa(aaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaaaaaaaaaaaaaaaaaa) {}",
               OnePerLine);
  OnePerLine.ColumnLimit = 60;
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaa(a),\n"
               "      bbbbbbbbbbbbbbbbbbbbbbbb(b) {}",
               OnePerLine);

  EXPECT_EQ("Constructor()\n"
            "    : // Comment forcing unwanted break.\n"
            "      aaaa(aaaa) {}",
            format("Constructor() :\n"
                   "    // Comment forcing unwanted break.\n"
                   "    aaaa(aaaa) {}"));
}

TEST_F(FormatTest, MemoizationTests) {
  // This breaks if the memoization lookup does not take \c Indent and
  // \c LastSpace into account.
  verifyFormat(
      "extern CFRunLoopTimerRef\n"
      "CFRunLoopTimerCreate(CFAllocatorRef allocato, CFAbsoluteTime fireDate,\n"
      "                     CFTimeInterval interval, CFOptionFlags flags,\n"
      "                     CFIndex order, CFRunLoopTimerCallBack callout,\n"
      "                     CFRunLoopTimerContext *context) {}");

  // Deep nesting somewhat works around our memoization.
  verifyFormat(
      "aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(\n"
      "    aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(\n"
      "        aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(\n"
      "            aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(aaaaa(\n"
      "                aaaaa())))))))))))))))))))))))))))))))))))))));",
      getLLVMStyleWithColumns(65));
  verifyFormat(
      "aaaaa(\n"
      "    aaaaa,\n"
      "    aaaaa(\n"
      "        aaaaa,\n"
      "        aaaaa(\n"
      "            aaaaa,\n"
      "            aaaaa(\n"
      "                aaaaa,\n"
      "                aaaaa(\n"
      "                    aaaaa,\n"
      "                    aaaaa(\n"
      "                        aaaaa,\n"
      "                        aaaaa(\n"
      "                            aaaaa,\n"
      "                            aaaaa(\n"
      "                                aaaaa,\n"
      "                                aaaaa(\n"
      "                                    aaaaa,\n"
      "                                    aaaaa(\n"
      "                                        aaaaa,\n"
      "                                        aaaaa(\n"
      "                                            aaaaa,\n"
      "                                            aaaaa(\n"
      "                                                aaaaa,\n"
      "                                                aaaaa))))))))))));",
      getLLVMStyleWithColumns(65));
  verifyFormat(
      "a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(a(), a), a), a), a),\n"
      "                                  a),\n"
      "                                a),\n"
      "                              a),\n"
      "                            a),\n"
      "                          a),\n"
      "                        a),\n"
      "                      a),\n"
      "                    a),\n"
      "                  a),\n"
      "                a),\n"
      "              a),\n"
      "            a),\n"
      "          a),\n"
      "        a),\n"
      "      a),\n"
      "    a),\n"
      "  a)",
      getLLVMStyleWithColumns(65));

  // This test takes VERY long when memoization is broken.
  FormatStyle OnePerLine = getLLVMStyle();
  OnePerLine.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  OnePerLine.BinPackParameters = false;
  std::string input = "Constructor()\n"
                      "    : aaaa(a,\n";
  for (unsigned i = 0, e = 80; i != e; ++i) {
    input += "           a,\n";
  }
  input += "           a) {}";
  verifyFormat(input, OnePerLine);
}

TEST_F(FormatTest, BreaksAsHighAsPossible) {
  verifyFormat(
      "void f() {\n"
      "  if ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaa && aaaaaaaaaaaaaaaaaaaaaaaaaa) ||\n"
      "      (bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb && bbbbbbbbbbbbbbbbbbbbbbbbbb))\n"
      "    f();\n"
      "}");
  verifyFormat("if (Intervals[i].getRange().getFirst() <\n"
               "    Intervals[i - 1].getRange().getLast()) {\n}");
}

TEST_F(FormatTest, BreaksFunctionDeclarations) {
  // Principially, we break function declarations in a certain order:
  // 1) break amongst arguments.
  verifyFormat("Aaaaaaaaaaaaaa bbbbbbbbbbbbbb(Cccccccccccccc cccccccccccccc,\n"
               "                              Cccccccccccccc cccccccccccccc);");
  verifyFormat("template <class TemplateIt>\n"
               "SomeReturnType SomeFunction(TemplateIt begin, TemplateIt end,\n"
               "                            TemplateIt *stop) {}");

  // 2) break after return type.
  verifyFormat(
      "Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "bbbbbbbbbbbbbb(Cccccccccccccc cccccccccccccccccccccccccc);",
      getGoogleStyle());

  // 3) break after (.
  verifyFormat(
      "Aaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbb(\n"
      "    Cccccccccccccccccccccccccccccc cccccccccccccccccccccccccccccccc);",
      getGoogleStyle());

  // 4) break before after nested name specifiers.
  verifyFormat(
      "Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "SomeClasssssssssssssssssssssssssssssssssssssss::\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(Cccccccccccccc cccccccccc);",
      getGoogleStyle());

  // However, there are exceptions, if a sufficient amount of lines can be
  // saved.
  // FIXME: The precise cut-offs wrt. the number of saved lines might need some
  // more adjusting.
  verifyFormat("Aaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbb(Cccccccccccccc cccccccccc,\n"
               "                                  Cccccccccccccc cccccccccc,\n"
               "                                  Cccccccccccccc cccccccccc,\n"
               "                                  Cccccccccccccc cccccccccc,\n"
               "                                  Cccccccccccccc cccccccccc);");
  verifyFormat(
      "Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "bbbbbbbbbbb(Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
      "            Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
      "            Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc);",
      getGoogleStyle());
  verifyFormat(
      "Aaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc,\n"
      "                                          Cccccccccccccc cccccccccc);");
  verifyFormat("Aaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(\n"
               "    Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
               "    Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
               "    Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc,\n"
               "    Cccccccccccccc cccccccccc, Cccccccccccccc cccccccccc);");

  // Break after multi-line parameters.
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    bbbb bbbb);");
  verifyFormat("void SomeLoooooooooooongFunction(\n"
               "    std::unique_ptr<aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    int bbbbbbbbbbbbb);");

  // Treat overloaded operators like other functions.
  verifyFormat("SomeLoooooooooooooooooooooooooogType\n"
               "operator>(const SomeLoooooooooooooooooooooooooogType &other);");
  verifyFormat("SomeLoooooooooooooooooooooooooogType\n"
               "operator>>(const SomeLooooooooooooooooooooooooogType &other);");
  verifyFormat("SomeLoooooooooooooooooooooooooogType\n"
               "operator<<(const SomeLooooooooooooooooooooooooogType &other);");
  verifyGoogleFormat(
      "SomeLoooooooooooooooooooooooooooooogType operator>>(\n"
      "    const SomeLooooooooogType &a, const SomeLooooooooogType &b);");
  verifyGoogleFormat(
      "SomeLoooooooooooooooooooooooooooooogType operator<<(\n"
      "    const SomeLooooooooogType &a, const SomeLooooooooogType &b);");
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = 1);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaa\n"
               "aaaaaaaaaaaaaaaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaa = 1);");
  verifyGoogleFormat(
      "typename aaaaaaaaaa<aaaaaa>::aaaaaaaaaaa\n"
      "aaaaaaaaaa<aaaaaa>::aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    bool *aaaaaaaaaaaaaaaaaa, bool *aa) {}");

  FormatStyle Style = getLLVMStyle();
  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaa* const aaaaaaaaaaaa) {}",
               Style);
  verifyFormat("void aaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
               Style);
}

TEST_F(FormatTest, TrailingReturnType) {
  verifyFormat("auto foo() -> int;\n");
  verifyFormat("struct S {\n"
               "  auto bar() const -> int;\n"
               "};");
  verifyFormat("template <size_t Order, typename T>\n"
               "auto load_img(const std::string &filename)\n"
               "    -> alias::tensor<Order, T, mem::tag::cpu> {}");
  verifyFormat("auto SomeFunction(A aaaaaaaaaaaaaaaaaaaaa) const\n"
               "    -> decltype(f(aaaaaaaaaaaaaaaaaaaaa)) {}");
  verifyFormat("auto doSomething(Aaaaaa *aaaaaa) -> decltype(aaaaaa->f()) {}");
  verifyFormat("template <typename T>\n"
               "auto aaaaaaaaaaaaaaaaaaaaaa(T t)\n"
               "    -> decltype(eaaaaaaaaaaaaaaa<T>(t.a).aaaaaaaa());");

  // Not trailing return types.
  verifyFormat("void f() { auto a = b->c(); }");
}

TEST_F(FormatTest, BreaksFunctionDeclarationsWithTrailingTokens) {
  // Avoid breaking before trailing 'const' or other trailing annotations, if
  // they are not function-like.
  FormatStyle Style = getGoogleStyle();
  Style.ColumnLimit = 47;
  verifyFormat("void someLongFunction(\n"
               "    int someLoooooooooooooongParameter) const {\n}",
               getLLVMStyleWithColumns(47));
  verifyFormat("LoooooongReturnType\n"
               "someLoooooooongFunction() const {}",
               getLLVMStyleWithColumns(47));
  verifyFormat("LoooooongReturnType someLoooooooongFunction()\n"
               "    const {}",
               Style);
  verifyFormat("void SomeFunction(aaaaa aaaaaaaaaaaaaaaaaaaa,\n"
               "                  aaaaa aaaaaaaaaaaaaaaaaaaa) OVERRIDE;");
  verifyFormat("void SomeFunction(aaaaa aaaaaaaaaaaaaaaaaaaa,\n"
               "                  aaaaa aaaaaaaaaaaaaaaaaaaa) OVERRIDE FINAL;");
  verifyFormat("void SomeFunction(aaaaa aaaaaaaaaaaaaaaaaaaa,\n"
               "                  aaaaa aaaaaaaaaaaaaaaaaaaa) override final;");
  verifyFormat("virtual void aaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaa aaaa,\n"
               "                   aaaaaaaaaaa aaaaa) const override;");
  verifyGoogleFormat(
      "virtual void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
      "    const override;");

  // Even if the first parameter has to be wrapped.
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) const {}",
               getLLVMStyleWithColumns(46));
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) const {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) override {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) OVERRIDE {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) final {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) FINAL {}",
               Style);
  verifyFormat("void someLongFunction(\n"
               "    int parameter) const override {}",
               Style);

  Style.BreakBeforeBraces = FormatStyle::BS_Allman;
  verifyFormat("void someLongFunction(\n"
               "    int someLongParameter) const\n"
               "{\n"
               "}",
               Style);

  // Unless these are unknown annotations.
  verifyFormat("void SomeFunction(aaaaaaaaaa aaaaaaaaaaaaaaa,\n"
               "                  aaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    LONG_AND_UGLY_ANNOTATION;");

  // Breaking before function-like trailing annotations is fine to keep them
  // close to their arguments.
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    LOCKS_EXCLUDED(aaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) const\n"
               "    LOCKS_EXCLUDED(aaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) const\n"
               "    LOCKS_EXCLUDED(aaaaaaaaaaaaa) {}");
  verifyGoogleFormat("void aaaaaaaaaaaaaa(aaaaaaaa aaa) override\n"
                     "    AAAAAAAAAAAAAAAAAAAAAAAA(aaaaaaaaaaaaaaa);");
  verifyFormat("SomeFunction([](int i) LOCKS_EXCLUDED(a) {});");

  verifyFormat(
      "void aaaaaaaaaaaaaaaaaa()\n"
      "    __attribute__((aaaaaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaaaaaaaaaaa));");
  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    __attribute__((unused));");
  verifyGoogleFormat(
      "bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    GUARDED_BY(aaaaaaaaaaaa);");
  verifyGoogleFormat(
      "bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    GUARDED_BY(aaaaaaaaaaaa);");
  verifyGoogleFormat(
      "bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa GUARDED_BY(aaaaaaaaaaaa) =\n"
      "    aaaaaaaa::aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyGoogleFormat(
      "bool aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa GUARDED_BY(aaaaaaaaaaaa) =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaa;");
}

TEST_F(FormatTest, FunctionAnnotations) {
  verifyFormat("DEPRECATED(\"Use NewClass::NewFunction instead.\")\n"
               "int OldFunction(const string &parameter) {}");
  verifyFormat("DEPRECATED(\"Use NewClass::NewFunction instead.\")\n"
               "string OldFunction(const string &parameter) {}");
  verifyFormat("template <typename T>\n"
               "DEPRECATED(\"Use NewClass::NewFunction instead.\")\n"
               "string OldFunction(const string &parameter) {}");

  // Not function annotations.
  verifyFormat("ASSERT(\"aaaaa\") << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                << bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
  verifyFormat("TEST_F(ThisIsATestFixtureeeeeeeeeeeee,\n"
               "       ThisIsATestWithAReallyReallyReallyReallyLongName) {}");
}

TEST_F(FormatTest, BreaksDesireably) {
  verifyFormat("if (aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa) ||\n"
               "    aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa) ||\n"
               "    aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa)) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)) {\n"
               "}");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa));");

  verifyFormat(
      "aaaaaaaa(aaaaaaaaaaaaa, aaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "                            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)),\n"
      "         aaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)));");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "    (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat(
      "void f() {\n"
      "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa &&\n"
      "                                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
      "}");
  verifyFormat(
      "aaaaaa(new Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaa));");
  verifyFormat(
      "aaaaaa(aaa, new Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "                aaaaaaaaaaaaaaaaaaaaaaaaaaaaa));");
  verifyFormat("aaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa +\n"
               "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  // Indent consistently independent of call expression and unary operator.
  verifyFormat("aaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(\n"
               "    dddddddddddddddddddddddddddddd));");
  verifyFormat("aaaaaaaaaaa(!bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(\n"
               "    dddddddddddddddddddddddddddddd));");
  verifyFormat("aaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbb.ccccccccccccccccc(\n"
               "    dddddddddddddddddddddddddddddd));");

  // This test case breaks on an incorrect memoization, i.e. an optimization not
  // taking into account the StopAt value.
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       aaaaaaaaaaa(aaaaaaaaa) || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       aaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("{\n  {\n    {\n"
               "      Annotation.SpaceRequiredBefore =\n"
               "          Line.Tokens[i - 1].Tok.isNot(tok::l_paren) &&\n"
               "          Line.Tokens[i - 1].Tok.isNot(tok::l_square);\n"
               "    }\n  }\n}");

  // Break on an outer level if there was a break on an inner level.
  EXPECT_EQ("f(g(h(a, // comment\n"
            "      b, c),\n"
            "    d, e),\n"
            "  x, y);",
            format("f(g(h(a, // comment\n"
                   "    b, c), d, e), x, y);"));

  // Prefer breaking similar line breaks.
  verifyFormat(
      "const int kTrackingOptions = NSTrackingMouseMoved |\n"
      "                             NSTrackingMouseEnteredAndExited |\n"
      "                             NSTrackingActiveAlways;");
}

TEST_F(FormatTest, FormatsDeclarationsOnePerLine) {
  FormatStyle NoBinPacking = getGoogleStyle();
  NoBinPacking.BinPackParameters = false;
  NoBinPacking.BinPackArguments = true;
  verifyFormat("void f() {\n"
               "  f(aaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
               "}",
               NoBinPacking);
  verifyFormat("void f(int aaaaaaaaaaaaaaaaaaaa,\n"
               "       int aaaaaaaaaaaaaaaaaaaa,\n"
               "       int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {}",
               NoBinPacking);
}

TEST_F(FormatTest, FormatsOneParameterPerLineIfNecessary) {
  FormatStyle NoBinPacking = getGoogleStyle();
  NoBinPacking.BinPackParameters = false;
  NoBinPacking.BinPackArguments = false;
  verifyFormat("f(aaaaaaaaaaaaaaaaaaaa,\n"
               "  aaaaaaaaaaaaaaaaaaaa,\n"
               "  aaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaa);",
               NoBinPacking);
  verifyFormat("aaaaaaa(aaaaaaaaaaaaa,\n"
               "        aaaaaaaaaaaaa,\n"
               "        aaaaaaaaaaaaa(aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa));",
               NoBinPacking);
  verifyFormat(
      "aaaaaaaa(aaaaaaaaaaaaa,\n"
      "         aaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)),\n"
      "         aaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)));",
      NoBinPacking);
  verifyFormat("aaaaaaaaaaaaaaa(aaaaaaaaa, aaaaaaaaa, aaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaaaaaaaaaaaaaa();",
               NoBinPacking);
  verifyFormat("void f() {\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "      aaaaaaaaaa, aaaaaaaaaa, aaaaaaaaaa, aaaaaaaaaaa);\n"
               "}",
               NoBinPacking);

  verifyFormat(
      "aaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "             aaaaaaaaaaaa,\n"
      "             aaaaaaaaaaaa);",
      NoBinPacking);
  verifyFormat(
      "somefunction(someotherFunction(ddddddddddddddddddddddddddddddddddd,\n"
      "                               ddddddddddddddddddddddddddddd),\n"
      "             test);",
      NoBinPacking);

  verifyFormat("std::vector<aaaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaaaaaaaaaaaaaaaaaaa>\n"
               "    aaaaaaaaaaaaaaaaaa;",
               NoBinPacking);
  verifyFormat("a(\"a\"\n"
               "  \"a\",\n"
               "  a);");

  NoBinPacking.AllowAllParametersOfDeclarationOnNextLine = false;
  verifyFormat("void aaaaaaaaaa(aaaaaaaaa,\n"
               "                aaaaaaaaa,\n"
               "                aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               NoBinPacking);
  verifyFormat(
      "void f() {\n"
      "  aaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaa, aaaaaaaaa, aaaaaaaaaaaaaaaaaaaaa)\n"
      "      .aaaaaaa();\n"
      "}",
      NoBinPacking);
  verifyFormat(
      "template <class SomeType, class SomeOtherType>\n"
      "SomeType SomeFunction(SomeType Type, SomeOtherType OtherType) {}",
      NoBinPacking);
}

TEST_F(FormatTest, AdaptiveOnePerLineFormatting) {
  FormatStyle Style = getLLVMStyleWithColumns(15);
  Style.ExperimentalAutoDetectBinPacking = true;
  EXPECT_EQ("aaa(aaaa,\n"
            "    aaaa,\n"
            "    aaaa);\n"
            "aaa(aaaa,\n"
            "    aaaa,\n"
            "    aaaa);",
            format("aaa(aaaa,\n" // one-per-line
                   "  aaaa,\n"
                   "    aaaa  );\n"
                   "aaa(aaaa,  aaaa,  aaaa);", // inconclusive
                   Style));
  EXPECT_EQ("aaa(aaaa, aaaa,\n"
            "    aaaa);\n"
            "aaa(aaaa, aaaa,\n"
            "    aaaa);",
            format("aaa(aaaa,  aaaa,\n" // bin-packed
                   "    aaaa  );\n"
                   "aaa(aaaa,  aaaa,  aaaa);", // inconclusive
                   Style));
}

TEST_F(FormatTest, FormatsBuilderPattern) {
  verifyFormat("return llvm::StringSwitch<Reference::Kind>(name)\n"
               "    .StartsWith(\".eh_frame_hdr\", ORDER_EH_FRAMEHDR)\n"
               "    .StartsWith(\".eh_frame\", ORDER_EH_FRAME)\n"
               "    .StartsWith(\".init\", ORDER_INIT)\n"
               "    .StartsWith(\".fini\", ORDER_FINI)\n"
               "    .StartsWith(\".hash\", ORDER_HASH)\n"
               "    .Default(ORDER_TEXT);\n");

  verifyFormat("return aaaaaaaaaaaaaaaaa->aaaaa().aaaaaaaaaaaaa().aaaaaa() <\n"
               "       aaaaaaaaaaaaaaa->aaaaa().aaaaaaaaaaaaa().aaaaaa();");
  verifyFormat(
      "aaaaaaa->aaaaaaa->aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "    ->aaaaaaaa(aaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaa->aaaaaaa\n"
      "    ->aaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "    ->aaaaaaaa(aaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaa()->aaaaaa(bbbbb)->aaaaaaaaaaaaaaaaaaa( // break\n"
      "    aaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaa *aaaaaaaaa =\n"
      "    aaaaaa->aaaaaaaaaaaa()\n"
      "        ->aaaaaaaaaaaaaaaa(\n"
      "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "        ->aaaaaaaaaaaaaaaaa();");
  verifyGoogleFormat(
      "void f() {\n"
      "  someo->Add((new util::filetools::Handler(dir))\n"
      "                 ->OnEvent1(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBA))\n"
      "                 ->OnEvent2(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBB))\n"
      "                 ->OnEvent3(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBC))\n"
      "                 ->OnEvent5(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBD))\n"
      "                 ->OnEvent6(NewPermanentCallback(\n"
      "                     this, &HandlerHolderClass::EventHandlerCBE)));\n"
      "}");

  verifyFormat(
      "aaaaaaaaaaa().aaaaaaaaaaa().aaaaaaaaaaa().aaaaaaaaaaa().aaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaaaaaaa.aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaaaaaaa.aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa.aaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaaaaa->aaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    ->aaaaaaaaaaaaaae(0)\n"
               "    ->aaaaaaaaaaaaaaa();");

  // Don't linewrap after very short segments.
  verifyFormat("a().aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("aa().aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("aaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaa.aaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "    .has<bbbbbbbbbbbbbbbbbbbbb>();");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaa.aaaaaaaaaaaaa()\n"
               "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>();");

  // Prefer not to break after empty parentheses.
  verifyFormat("FirstToken->WhitespaceRange.getBegin().getLocWithOffset(\n"
               "    First->LastNewlineOffset);");

  // Prefer not to create "hanging" indents.
  verifyFormat(
      "return !soooooooooooooome_map\n"
      "            .insert(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "            .second;");
}

TEST_F(FormatTest, BreaksAccordingToOperatorPrecedence) {
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbb && ccccccccccccccccccccccccc) {\n}");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaa or\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbb and cccccccccccccccccccccccc) {\n}");

  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa && bbbbbbbbbbbbbbbbbbbbbbbbb ||\n"
               "    ccccccccccccccccccccccccc) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa and bbbbbbbbbbbbbbbbbbbbbbbb or\n"
               "    ccccccccccccccccccccccccc) {\n}");

  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa || bbbbbbbbbbbbbbbbbbbbbbbbb ||\n"
               "    ccccccccccccccccccccccccc) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa or bbbbbbbbbbbbbbbbbbbbbbbbb or\n"
               "    ccccccccccccccccccccccccc) {\n}");

  verifyFormat(
      "if ((aaaaaaaaaaaaaaaaaaaaaaaaa || bbbbbbbbbbbbbbbbbbbbbbbbb) &&\n"
      "    ccccccccccccccccccccccccc) {\n}");
  verifyFormat(
      "if ((aaaaaaaaaaaaaaaaaaaaaaaaa or bbbbbbbbbbbbbbbbbbbbbbbbb) and\n"
      "    ccccccccccccccccccccccccc) {\n}");

  verifyFormat("return aaaa & AAAAAAAAAAAAAAAAAAAAAAAAAAAAA ||\n"
               "       bbbb & BBBBBBBBBBBBBBBBBBBBBBBBBBBBB ||\n"
               "       cccc & CCCCCCCCCCCCCCCCCCCCCCCCCC ||\n"
               "       dddd & DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD;");
  verifyFormat("return aaaa & AAAAAAAAAAAAAAAAAAAAAAAAAAAAA or\n"
               "       bbbb & BBBBBBBBBBBBBBBBBBBBBBBBBBBBB or\n"
               "       cccc & CCCCCCCCCCCCCCCCCCCCCCCCCC or\n"
               "       dddd & DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD;");

  verifyFormat("if ((aaaaaaaaaa != aaaaaaaaaaaaaaa ||\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaa() >= aaaaaaaaaaaaaaaaaaaa) &&\n"
               "    aaaaaaaaaaaaaaa != aa) {\n}");
  verifyFormat("if ((aaaaaaaaaa != aaaaaaaaaaaaaaa or\n"
               "     aaaaaaaaaaaaaaaaaaaaaaaa() >= aaaaaaaaaaaaaaaaaaaa) and\n"
               "    aaaaaaaaaaaaaaa != aa) {\n}");
}

TEST_F(FormatTest, BreaksAfterAssignments) {
  verifyFormat(
      "unsigned Cost =\n"
      "    TTI.getMemoryOpCost(I->getOpcode(), VectorTy, SI->getAlignment(),\n"
      "                        SI->getPointerAddressSpaceee());\n");
  verifyFormat(
      "CharSourceRange LineRange = CharSourceRange::getTokenRange(\n"
      "    Line.Tokens.front().Tok.getLo(), Line.Tokens.back().Tok.getLoc());");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaa aaaa = aaaaaaaaaaaaaa(0).aaaa().aaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaa::aaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("unsigned OriginalStartColumn =\n"
               "    SourceMgr.getSpellingColumnNumber(\n"
               "        Current.FormatTok.getStartOfNonWhitespace()) -\n"
               "    1;");
}

TEST_F(FormatTest, AlignsAfterAssignments) {
  verifyFormat(
      "int Result = aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "Result += aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "          aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "Result >>= aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "           aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "int Result = (aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "              aaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "double LooooooooooooooooooooooooongResult = aaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaaaaa;");
}

TEST_F(FormatTest, AlignsAfterReturn) {
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "       aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "return (aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa >=\n"
      "       aaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat(
      "return (aaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa >=\n"
      "        aaaaaaaaaaaaaaaaaaaaaa());");
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) &&\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("return\n"
               "    // true if code is one of a or b.\n"
               "    code == a || code == b;");
}

TEST_F(FormatTest, AlignsAfterOpenBracket) {
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

  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  Style.BinPackArguments = false;
  Style.BinPackParameters = false;
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
  verifyFormat("SomeLongVariableName->someFunction(\n"
               "    foooooooo(\n"
               "        aaaaaaaaaaaaaaa,\n"
               "        aaaaaaaaaaaaaaaaaaaaa,\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa));",
               Style);
}

TEST_F(FormatTest, ParenthesesAndOperandAlignment) {
  FormatStyle Style = getLLVMStyleWithColumns(40);
  verifyFormat("int a = f(aaaaaaaaaaaaaaaaaaaaaa &&\n"
               "          bbbbbbbbbbbbbbbbbbbbbb);",
               Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_Align;
  Style.AlignOperands = false;
  verifyFormat("int a = f(aaaaaaaaaaaaaaaaaaaaaa &&\n"
               "          bbbbbbbbbbbbbbbbbbbbbb);",
               Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  Style.AlignOperands = true;
  verifyFormat("int a = f(aaaaaaaaaaaaaaaaaaaaaa &&\n"
               "          bbbbbbbbbbbbbbbbbbbbbb);",
               Style);
  Style.AlignAfterOpenBracket = FormatStyle::BAS_DontAlign;
  Style.AlignOperands = false;
  verifyFormat("int a = f(aaaaaaaaaaaaaaaaaaaaaa &&\n"
               "    bbbbbbbbbbbbbbbbbbbbbb);",
               Style);
}

TEST_F(FormatTest, BreaksConditionalExpressions) {
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                               ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                               : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                   : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa ? aaaa(aaaaaa)\n"
      "                                                    : aaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                    : aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaa ?: aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaa);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "           ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "           : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "           ?: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    ? aaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        : aaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    ? aaaaaaaaaaaaaaa\n"
      "    : aaaaaaaaaaaaaaa;");
  verifyFormat("f(aaaaaaaaaaaaaaaa == // force break\n"
               "          aaaaaaaaa\n"
               "      ? b\n"
               "      : c);");
  verifyFormat("return aaaa == bbbb\n"
               "           // comment\n"
               "           ? aaaa\n"
               "           : bbbb;");
  verifyFormat("unsigned Indent =\n"
               "    format(TheLine.First, IndentForLevel[TheLine.Level] >= 0\n"
               "                              ? IndentForLevel[TheLine.Level]\n"
               "                              : TheLine * 2,\n"
               "           TheLine.InPPDirective, PreviousEndOfLineColumn);",
               getLLVMStyleWithColumns(70));
  verifyFormat("bool aaaaaa = aaaaaaaaaaaaa //\n"
               "                  ? aaaaaaaaaaaaaaa\n"
               "                  : bbbbbbbbbbbbbbb //\n"
               "                        ? ccccccccccccccc\n"
               "                        : ddddddddddddddd;");
  verifyFormat("bool aaaaaa = aaaaaaaaaaaaa //\n"
               "                  ? aaaaaaaaaaaaaaa\n"
               "                  : (bbbbbbbbbbbbbbb //\n"
               "                         ? ccccccccccccccc\n"
               "                         : ddddddddddddddd);");
  verifyFormat(
      "int aaaaaaaaaaaaaaaaaaaaaaaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                      ? aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaa\n"
      "                                      : aaaaaaaaaa;");
  verifyFormat(
      "aaaaaa = aaaaaaaaaaaa\n"
      "             ? aaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                          : aaaaaaaaaaaaaaaaaaaaaa\n"
      "             : aaaaaaaaaaaaaaaaaaaaaaaaaaaa;");

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackArguments = false;
  verifyFormat(
      "void f() {\n"
      "  g(aaa,\n"
      "    aaaaaaaaaa == aaaaaaaaaa ? aaaa : aaaaa,\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "        ? aaaaaaaaaaaaaaa\n"
      "        : aaaaaaaaaaaaaaa);\n"
      "}",
      NoBinPacking);
  verifyFormat(
      "void f() {\n"
      "  g(aaa,\n"
      "    aaaaaaaaaa == aaaaaaaaaa ? aaaa : aaaaa,\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "        ?: aaaaaaaaaaaaaaa);\n"
      "}",
      NoBinPacking);

  verifyFormat("SomeFunction(aaaaaaaaaaaaaaaaa,\n"
               "             // comment.\n"
               "             ccccccccccccccccccccccccccccccccccccccc\n"
               "                 ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "                 : bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);");

  // Assignments in conditional expressions. Apparently not uncommon :-(.
  verifyFormat("return a != b\n"
               "           // comment\n"
               "           ? a = b\n"
               "           : a = b;");
  verifyFormat("return a != b\n"
               "           // comment\n"
               "           ? a = a != b\n"
               "                     // comment\n"
               "                     ? a = b\n"
               "                     : a\n"
               "           : a;\n");
  verifyFormat("return a != b\n"
               "           // comment\n"
               "           ? a\n"
               "           : a = a != b\n"
               "                     // comment\n"
               "                     ? a = b\n"
               "                     : a;");
}

TEST_F(FormatTest, BreaksConditionalExpressionsAfterOperator) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakBeforeTernaryOperators = false;
  Style.ColumnLimit = 70;
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
      "                               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
      "                               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
      "                                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa ? aaaa(aaaaaa) :\n"
      "                                                      aaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
      "                                      aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaa);",
      Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaa ?: aaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaa);",
      Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  verifyFormat("aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) :\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  verifyFormat("aaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?:\n"
               "           aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
      "    aaaaaaaaaaaaaaa :\n"
      "    aaaaaaaaaaaaaaa;",
      Style);
  verifyFormat("f(aaaaaaaaaaaaaaaa == // force break\n"
               "          aaaaaaaaa ?\n"
               "      b :\n"
               "      c);",
               Style);
  verifyFormat(
      "unsigned Indent =\n"
      "    format(TheLine.First, IndentForLevel[TheLine.Level] >= 0 ?\n"
      "                              IndentForLevel[TheLine.Level] :\n"
      "                              TheLine * 2,\n"
      "           TheLine.InPPDirective, PreviousEndOfLineColumn);",
      Style);
  verifyFormat("bool aaaaaa = aaaaaaaaaaaaa ? //\n"
               "                  aaaaaaaaaaaaaaa :\n"
               "                  bbbbbbbbbbbbbbb ? //\n"
               "                      ccccccccccccccc :\n"
               "                      ddddddddddddddd;",
               Style);
  verifyFormat("bool aaaaaa = aaaaaaaaaaaaa ? //\n"
               "                  aaaaaaaaaaaaaaa :\n"
               "                  (bbbbbbbbbbbbbbb ? //\n"
               "                       ccccccccccccccc :\n"
               "                       ddddddddddddddd);",
               Style);
}

TEST_F(FormatTest, DeclarationsOfMultipleVariables) {
  verifyFormat("bool aaaaaaaaaaaaaaaaa = aaaaaa->aaaaaaaaaaaaaaaaa(),\n"
               "     aaaaaaaaaaa = aaaaaa->aaaaaaaaaaa();");
  verifyFormat("bool a = true, b = false;");

  verifyFormat("bool aaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaa),\n"
               "     bbbbbbbbbbbbbbbbbbbbbbbbb =\n"
               "         bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(bbbbbbbbbbbbbbbb);");
  verifyFormat(
      "bool aaaaaaaaaaaaaaaaaaaaa =\n"
      "         bbbbbbbbbbbbbbbbbbbbbbbbbbbb && cccccccccccccccccccccccccccc,\n"
      "     d = e && f;");
  verifyFormat("aaaaaaaaa a = aaaaaaaaaaaaaaaaaaaa, b = bbbbbbbbbbbbbbbbbbbb,\n"
               "          c = cccccccccccccccccccc, d = dddddddddddddddddddd;");
  verifyFormat("aaaaaaaaa *a = aaaaaaaaaaaaaaaaaaa, *b = bbbbbbbbbbbbbbbbbbb,\n"
               "          *c = ccccccccccccccccccc, *d = ddddddddddddddddddd;");
  verifyFormat("aaaaaaaaa ***a = aaaaaaaaaaaaaaaaaaa, ***b = bbbbbbbbbbbbbbb,\n"
               "          ***c = ccccccccccccccccccc, ***d = ddddddddddddddd;");

  FormatStyle Style = getGoogleStyle();
  Style.PointerAlignment = FormatStyle::PAS_Left;
  Style.DerivePointerAlignment = false;
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    *aaaaaaaaaaaaaaaaaaaaaaaaaaaaa = aaaaaaaaaaaaaaaaaaa,\n"
               "    *b = bbbbbbbbbbbbbbbbbbb;",
               Style);
  verifyFormat("aaaaaaaaa *a = aaaaaaaaaaaaaaaaaaa, *b = bbbbbbbbbbbbbbbbbbb,\n"
               "          *b = bbbbbbbbbbbbbbbbbbb, *d = ddddddddddddddddddd;",
               Style);
}

TEST_F(FormatTest, ConditionalExpressionsInBrackets) {
  verifyFormat("arr[foo ? bar : baz];");
  verifyFormat("f()[foo ? bar : baz];");
  verifyFormat("(a + b)[foo ? bar : baz];");
  verifyFormat("arr[foo ? (4 > 5 ? 4 : 5) : 5 < 5 ? 5 : 7];");
}

TEST_F(FormatTest, AlignsStringLiterals) {
  verifyFormat("loooooooooooooooooooooooooongFunction(\"short literal \"\n"
               "                                      \"short literal\");");
  verifyFormat(
      "looooooooooooooooooooooooongFunction(\n"
      "    \"short literal\"\n"
      "    \"looooooooooooooooooooooooooooooooooooooooooooooooong literal\");");
  verifyFormat("someFunction(\"Always break between multi-line\"\n"
               "             \" string literals\",\n"
               "             and, other, parameters);");
  EXPECT_EQ("fun + \"1243\" /* comment */\n"
            "      \"5678\";",
            format("fun + \"1243\" /* comment */\n"
                   "      \"5678\";",
                   getLLVMStyleWithColumns(28)));
  EXPECT_EQ(
      "aaaaaa = \"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaa \"\n"
      "         \"aaaaaaaaaaaaaaaaaaaaa\"\n"
      "         \"aaaaaaaaaaaaaaaa\";",
      format("aaaaaa ="
             "\"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaa "
             "aaaaaaaaaaaaaaaaaaaaa\" "
             "\"aaaaaaaaaaaaaaaa\";"));
  verifyFormat("a = a + \"a\"\n"
               "        \"a\"\n"
               "        \"a\";");
  verifyFormat("f(\"a\", \"b\"\n"
               "       \"c\");");

  verifyFormat(
      "#define LL_FORMAT \"ll\"\n"
      "printf(\"aaaaa: %d, bbbbbb: %\" LL_FORMAT \"d, cccccccc: %\" LL_FORMAT\n"
      "       \"d, ddddddddd: %\" LL_FORMAT \"d\");");

  verifyFormat("#define A(X)          \\\n"
               "  \"aaaaa\" #X \"bbbbbb\" \\\n"
               "  \"ccccc\"",
               getLLVMStyleWithColumns(23));
  verifyFormat("#define A \"def\"\n"
               "f(\"abc\" A \"ghi\"\n"
               "  \"jkl\");");

  verifyFormat("f(L\"a\"\n"
               "  L\"b\");");
  verifyFormat("#define A(X)            \\\n"
               "  L\"aaaaa\" #X L\"bbbbbb\" \\\n"
               "  L\"ccccc\"",
               getLLVMStyleWithColumns(25));

  verifyFormat("f(@\"a\"\n"
               "  @\"b\");");
  verifyFormat("NSString s = @\"a\"\n"
               "             @\"b\"\n"
               "             @\"c\";");
  verifyFormat("NSString s = @\"a\"\n"
               "              \"b\"\n"
               "              \"c\";");
}

TEST_F(FormatTest, ReturnTypeBreakingStyle) {
  FormatStyle Style = getLLVMStyle();
  // No declarations or definitions should be moved to own line.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_None;
  verifyFormat("class A {\n"
               "  int f() { return 1; }\n"
               "  int g();\n"
               "};\n"
               "int f() { return 1; }\n"
               "int g();\n",
               Style);

  // All declarations and definitions should have the return type moved to its
  // own
  // line.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_All;
  verifyFormat("class E {\n"
               "  int\n"
               "  f() {\n"
               "    return 1;\n"
               "  }\n"
               "  int\n"
               "  g();\n"
               "};\n"
               "int\n"
               "f() {\n"
               "  return 1;\n"
               "}\n"
               "int\n"
               "g();\n",
               Style);

  // Top-level definitions, and no kinds of declarations should have the
  // return type moved to its own line.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_TopLevelDefinitions;
  verifyFormat("class B {\n"
               "  int f() { return 1; }\n"
               "  int g();\n"
               "};\n"
               "int\n"
               "f() {\n"
               "  return 1;\n"
               "}\n"
               "int g();\n",
               Style);

  // Top-level definitions and declarations should have the return type moved
  // to its own line.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_TopLevel;
  verifyFormat("class C {\n"
               "  int f() { return 1; }\n"
               "  int g();\n"
               "};\n"
               "int\n"
               "f() {\n"
               "  return 1;\n"
               "}\n"
               "int\n"
               "g();\n",
               Style);

  // All definitions should have the return type moved to its own line, but no
  // kinds of declarations.
  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_AllDefinitions;
  verifyFormat("class D {\n"
               "  int\n"
               "  f() {\n"
               "    return 1;\n"
               "  }\n"
               "  int g();\n"
               "};\n"
               "int\n"
               "f() {\n"
               "  return 1;\n"
               "}\n"
               "int g();\n",
               Style);
  verifyFormat("const char *\n"
               "f(void) {\n" // Break here.
               "  return \"\";\n"
               "}\n"
               "const char *bar(void);\n", // No break here.
               Style);
  verifyFormat("template <class T>\n"
               "T *\n"
               "f(T &c) {\n" // Break here.
               "  return NULL;\n"
               "}\n"
               "template <class T> T *f(T &c);\n", // No break here.
               Style);
  verifyFormat("class C {\n"
               "  int\n"
               "  operator+() {\n"
               "    return 1;\n"
               "  }\n"
               "  int\n"
               "  operator()() {\n"
               "    return 1;\n"
               "  }\n"
               "};\n",
               Style);
  verifyFormat("void\n"
               "A::operator()() {}\n"
               "void\n"
               "A::operator>>() {}\n"
               "void\n"
               "A::operator+() {}\n",
               Style);
  verifyFormat("void *operator new(std::size_t s);", // No break here.
               Style);
  verifyFormat("void *\n"
               "operator new(std::size_t s) {}",
               Style);
  verifyFormat("void *\n"
               "operator delete[](void *ptr) {}",
               Style);
  Style.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  verifyFormat("const char *\n"
               "f(void)\n" // Break here.
               "{\n"
               "  return \"\";\n"
               "}\n"
               "const char *bar(void);\n", // No break here.
               Style);
  verifyFormat("template <class T>\n"
               "T *\n"     // Problem here: no line break
               "f(T &c)\n" // Break here.
               "{\n"
               "  return NULL;\n"
               "}\n"
               "template <class T> T *f(T &c);\n", // No break here.
               Style);
}

TEST_F(FormatTest, AlwaysBreakBeforeMultilineStrings) {
  FormatStyle NoBreak = getLLVMStyle();
  NoBreak.AlwaysBreakBeforeMultilineStrings = false;
  FormatStyle Break = getLLVMStyle();
  Break.AlwaysBreakBeforeMultilineStrings = true;
  verifyFormat("aaaa = \"bbbb\"\n"
               "       \"cccc\";",
               NoBreak);
  verifyFormat("aaaa =\n"
               "    \"bbbb\"\n"
               "    \"cccc\";",
               Break);
  verifyFormat("aaaa(\"bbbb\"\n"
               "     \"cccc\");",
               NoBreak);
  verifyFormat("aaaa(\n"
               "    \"bbbb\"\n"
               "    \"cccc\");",
               Break);
  verifyFormat("aaaa(qqq, \"bbbb\"\n"
               "          \"cccc\");",
               NoBreak);
  verifyFormat("aaaa(qqq,\n"
               "     \"bbbb\"\n"
               "     \"cccc\");",
               Break);
  verifyFormat("aaaa(qqq,\n"
               "     L\"bbbb\"\n"
               "     L\"cccc\");",
               Break);
  verifyFormat("aaaaa(aaaaaa, aaaaaaa(\"aaaa\"\n"
               "                      \"bbbb\"));",
               Break);
  verifyFormat("string s = someFunction(\n"
               "    \"abc\"\n"
               "    \"abc\");",
               Break);

  // As we break before unary operators, breaking right after them is bad.
  verifyFormat("string foo = abc ? \"x\"\n"
               "                   \"blah blah blah blah blah blah\"\n"
               "                 : \"y\";",
               Break);

  // Don't break if there is no column gain.
  verifyFormat("f(\"aaaa\"\n"
               "  \"bbbb\");",
               Break);

  // Treat literals with escaped newlines like multi-line string literals.
  EXPECT_EQ("x = \"a\\\n"
            "b\\\n"
            "c\";",
            format("x = \"a\\\n"
                   "b\\\n"
                   "c\";",
                   NoBreak));
  EXPECT_EQ("xxxx =\n"
            "    \"a\\\n"
            "b\\\n"
            "c\";",
            format("xxxx = \"a\\\n"
                   "b\\\n"
                   "c\";",
                   Break));

  // Exempt ObjC strings for now.
  EXPECT_EQ("NSString *const kString = @\"aaaa\"\n"
            "                          @\"bbbb\";",
            format("NSString *const kString = @\"aaaa\"\n"
                   "@\"bbbb\";",
                   Break));

  Break.ColumnLimit = 0;
  verifyFormat("const char *hello = \"hello llvm\";", Break);
}

TEST_F(FormatTest, AlignsPipes) {
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaa\n"
      "                     << aaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                 << aaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "llvm::outs() << \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
      "                \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"\n"
      "             << \"ccccccccccccccccccccccccccccccccccccccccccccccccc\";");
  verifyFormat(
      "aaaaaaaa << (aaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                 << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "         << aaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("llvm::errs() << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "             << bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;");
  verifyFormat("llvm::errs() << \"aaaaaaaaaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaaaaaaaaaaa(aaaaaaaa, aaaaaaaaaaa);");
  verifyFormat(
      "llvm::errs() << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("return out << \"somepacket = {\\n\"\n"
               "           << \" aaaaaa = \" << pkt.aaaaaa << \"\\n\"\n"
               "           << \" bbbb = \" << pkt.bbbb << \"\\n\"\n"
               "           << \" cccccc = \" << pkt.cccccc << \"\\n\"\n"
               "           << \" ddd = [\" << pkt.ddd << \"]\\n\"\n"
               "           << \"}\";");

  verifyFormat("llvm::outs() << \"aaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaa\n"
               "             << \"aaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaa\n"
               "             << \"aaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaa;");
  verifyFormat(
      "llvm::outs() << \"aaaaaaaaaaaaaaaaa = \" << aaaaaaaaaaaaaaaaa\n"
      "             << \"bbbbbbbbbbbbbbbbb = \" << bbbbbbbbbbbbbbbbb\n"
      "             << \"ccccccccccccccccc = \" << ccccccccccccccccc\n"
      "             << \"ddddddddddddddddd = \" << ddddddddddddddddd\n"
      "             << \"eeeeeeeeeeeeeeeee = \" << eeeeeeeeeeeeeeeee;");
  verifyFormat("llvm::outs() << aaaaaaaaaaaaaaaaaaaaaaaa << \"=\"\n"
               "             << bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;");
  verifyFormat(
      "void f() {\n"
      "  llvm::outs() << \"aaaaaaaaaaaaaaaaaaaa: \"\n"
      "               << aaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaa);\n"
      "}");
  verifyFormat("llvm::outs() << \"aaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaa.aaaaaaaaaaaa(aaa)->aaaaaaaaaaaaaa();");
  verifyFormat("llvm::errs() << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                    aaaaaaaaaaaaaaaaaaaaa)\n"
               "             << aaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("LOG_IF(aaa == //\n"
               "       bbb)\n"
               "    << a << b;");

  // Breaking before the first "<<" is generally not desirable.
  verifyFormat(
      "llvm::errs()\n"
      "    << \"aaaaaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << \"aaaaaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << \"aaaaaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << \"aaaaaaaaaaaaaaaaaaa: \" << aaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
      getLLVMStyleWithColumns(70));
  verifyFormat("llvm::errs() << \"aaaaaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "             << \"aaaaaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "             << \"aaaaaaaaaaaaaaaaaaa: \"\n"
               "             << aaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               getLLVMStyleWithColumns(70));

  // But sometimes, breaking before the first "<<" is desirable.
  verifyFormat("Diag(aaaaaaaaaaaaaaaaaaaa, aaaaaaaa)\n"
               "    << aaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaa);");
  verifyFormat("Diag(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbb)\n"
               "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("SemaRef.Diag(Loc, diag::note_for_range_begin_end)\n"
               "    << BEF << IsTemplate << Description << E->getType();");

  verifyFormat(
      "llvm::errs() << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");

  // Incomplete string literal.
  EXPECT_EQ("llvm::errs() << \"\n"
            "             << a;",
            format("llvm::errs() << \"\n<<a;"));

  verifyFormat("void f() {\n"
               "  CHECK_EQ(aaaa, (*bbbbbbbbb)->cccccc)\n"
               "      << \"qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\";\n"
               "}");

  // Handle 'endl'.
  verifyFormat("llvm::errs() << aaaaaaaaaaaaaaaaaaaaaa << endl\n"
               "             << bbbbbbbbbbbbbbbbbbbbbb << endl;");
  verifyFormat("llvm::errs() << endl << bbbbbbbbbbbbbbbbbbbbbb << endl;");
}

TEST_F(FormatTest, UnderstandsEquals) {
  verifyFormat(
      "aaaaaaaaaaaaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat(
      "if (a) {\n"
      "  f();\n"
      "} else if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
      "}");

  verifyFormat("if (int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "        100000000 + 10000000) {\n}");
}

TEST_F(FormatTest, WrapsAtFunctionCallsIfNecessary) {
  verifyFormat("LoooooooooooooooooooooooooooooooooooooongObject\n"
               "    .looooooooooooooooooooooooooooooooooooooongFunction();");

  verifyFormat("LoooooooooooooooooooooooooooooooooooooongObject\n"
               "    ->looooooooooooooooooooooooooooooooooooooongFunction();");

  verifyFormat(
      "LooooooooooooooooooooooooooooooooongObject->shortFunction(Parameter1,\n"
      "                                                          Parameter2);");

  verifyFormat(
      "ShortObject->shortFunction(\n"
      "    LooooooooooooooooooooooooooooooooooooooooooooooongParameter1,\n"
      "    LooooooooooooooooooooooooooooooooooooooooooooooongParameter2);");

  verifyFormat("loooooooooooooongFunction(\n"
               "    LoooooooooooooongObject->looooooooooooooooongFunction());");

  verifyFormat(
      "function(LoooooooooooooooooooooooooooooooooooongObject\n"
      "             ->loooooooooooooooooooooooooooooooooooooooongFunction());");

  verifyFormat("EXPECT_CALL(SomeObject, SomeFunction(Parameter))\n"
               "    .WillRepeatedly(Return(SomeValue));");
  verifyFormat("void f() {\n"
               "  EXPECT_CALL(SomeObject, SomeFunction(Parameter))\n"
               "      .Times(2)\n"
               "      .WillRepeatedly(Return(SomeValue));\n"
               "}");
  verifyFormat("SomeMap[std::pair(aaaaaaaaaaaa, bbbbbbbbbbbbbbb)].insert(\n"
               "    ccccccccccccccccccccccc);");
  verifyFormat("aaaaa(aaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "          .aaaaa(aaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("void f() {\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "      aaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa)->aaaaaaaaa());\n"
               "}");
  verifyFormat("aaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaaaaaaaaaaa(aa(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                        aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                        aaaaaaaaaaaaaaaaaaaaaaaaaaa));");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()) {\n"
               "}");

  // Here, it is not necessary to wrap at "." or "->".
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaa) ||\n"
               "    aaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat(
      "aaaaaaaaaaa->aaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaaa->aaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa));\n");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa().aaaaaaaaaaaaaaaaa());");
  verifyFormat("a->aaaaaa()->aaaaaaaaaaa(aaaaaaaa()->aaaaaa()->aaaaa() *\n"
               "                         aaaaaaaaa()->aaaaaa()->aaaaa());");
  verifyFormat("a->aaaaaa()->aaaaaaaaaaa(aaaaaaaa()->aaaaaa()->aaaaa() ||\n"
               "                         aaaaaaaaa()->aaaaaa()->aaaaa());");

  verifyFormat("aaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    .a();");

  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = false;
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa)\n"
               "    .aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaa,\n"
               "                         aaaaaaaaaaaaaaaaaaa,\n"
               "                         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);",
               NoBinPacking);

  // If there is a subsequent call, change to hanging indentation.
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "                         aaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa))\n"
      "    .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa));");
  verifyFormat("aaaaaaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "                 .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("aaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "               .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa());");
}

TEST_F(FormatTest, WrapsTemplateDeclarations) {
  verifyFormat("template <typename T>\n"
               "virtual void loooooooooooongFunction(int Param1, int Param2);");
  verifyFormat("template <typename T>\n"
               "// T should be one of {A, B}.\n"
               "virtual void loooooooooooongFunction(int Param1, int Param2);");
  verifyFormat(
      "template <typename T>\n"
      "using comment_to_xml_conversion = comment_to_xml_conversion<T, int>;");
  verifyFormat("template <typename T>\n"
               "void f(int Paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaram1,\n"
               "       int Paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaram2);");
  verifyFormat(
      "template <typename T>\n"
      "void looooooooooooooooooooongFunction(int Paaaaaaaaaaaaaaaaaaaaram1,\n"
      "                                      int Paaaaaaaaaaaaaaaaaaaaram2);");
  verifyFormat(
      "template <typename T>\n"
      "aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaa<T>::aaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("template <typename T>\n"
               "void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    int aaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "template <typename T1, typename T2 = char, typename T3 = char,\n"
      "          typename T4 = char>\n"
      "void f();");
  verifyFormat("template <typename aaaaaaaaaaa, typename bbbbbbbbbbbbb,\n"
               "          template <typename> class cccccccccccccccccccccc,\n"
               "          typename ddddddddddddd>\n"
               "class C {};");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa>(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("void f() {\n"
               "  a<aaaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaa>(\n"
               "      a(aaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaa));\n"
               "}");

  verifyFormat("template <typename T> class C {};");
  verifyFormat("template <typename T> void f();");
  verifyFormat("template <typename T> void f() {}");
  verifyFormat(
      "aaaaaaaaaaaaa<aaaaaaaaaa, aaaaaaaaaaa,\n"
      "              aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "              aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa> *aaaa =\n"
      "    new aaaaaaaaaaaaa<aaaaaaaaaa, aaaaaaaaaaa,\n"
      "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>(\n"
      "        bbbbbbbbbbbbbbbbbbbbbbbb);",
      getLLVMStyleWithColumns(72));
  EXPECT_EQ("static_cast<A< //\n"
            "    B> *>(\n"
            "\n"
            "    );",
            format("static_cast<A<//\n"
                   "    B>*>(\n"
                   "\n"
                   "    );"));
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    const typename aaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaa);");

  FormatStyle AlwaysBreak = getLLVMStyle();
  AlwaysBreak.AlwaysBreakTemplateDeclarations = true;
  verifyFormat("template <typename T>\nclass C {};", AlwaysBreak);
  verifyFormat("template <typename T>\nvoid f();", AlwaysBreak);
  verifyFormat("template <typename T>\nvoid f() {}", AlwaysBreak);
  verifyFormat("void aaaaaaaaaaaaaaaaaaa<aaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                         bbbbbbbbbbbbbbbbbbbbbbbbbbbb>(\n"
               "    ccccccccccccccccccccccccccccccccccccccccccccccc);");
  verifyFormat("template <template <typename> class Fooooooo,\n"
               "          template <typename> class Baaaaaaar>\n"
               "struct C {};",
               AlwaysBreak);
  verifyFormat("template <typename T> // T can be A, B or C.\n"
               "struct C {};",
               AlwaysBreak);
}

TEST_F(FormatTest, WrapsAtNestedNameSpecifiers) {
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa());");

  // FIXME: Should we have the extra indent after the second break?
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");

  verifyFormat(
      "aaaaaaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb::\n"
      "                    cccccccccccccccccccccccccccccccccccccccccccccc());");

  // Breaking at nested name specifiers is generally not desirable.
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaa(aaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
      "                                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                   aaaaaaaaaaaaaaaaaaaaa);",
      getLLVMStyleWithColumns(74));

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa::\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
}

TEST_F(FormatTest, UnderstandsTemplateParameters) {
  verifyFormat("A<int> a;");
  verifyFormat("A<A<A<int>>> a;");
  verifyFormat("A<A<A<int, 2>, 3>, 4> a;");
  verifyFormat("bool x = a < 1 || 2 > a;");
  verifyFormat("bool x = 5 < f<int>();");
  verifyFormat("bool x = f<int>() > 5;");
  verifyFormat("bool x = 5 < a<int>::x;");
  verifyFormat("bool x = a < 4 ? a > 2 : false;");
  verifyFormat("bool x = f() ? a < 2 : a > 2;");

  verifyGoogleFormat("A<A<int>> a;");
  verifyGoogleFormat("A<A<A<int>>> a;");
  verifyGoogleFormat("A<A<A<A<int>>>> a;");
  verifyGoogleFormat("A<A<int> > a;");
  verifyGoogleFormat("A<A<A<int> > > a;");
  verifyGoogleFormat("A<A<A<A<int> > > > a;");
  verifyGoogleFormat("A<::A<int>> a;");
  verifyGoogleFormat("A<::A> a;");
  verifyGoogleFormat("A< ::A> a;");
  verifyGoogleFormat("A< ::A<int> > a;");
  EXPECT_EQ("A<A<A<A>>> a;", format("A<A<A<A> >> a;", getGoogleStyle()));
  EXPECT_EQ("A<A<A<A>>> a;", format("A<A<A<A>> > a;", getGoogleStyle()));
  EXPECT_EQ("A<::A<int>> a;", format("A< ::A<int>> a;", getGoogleStyle()));
  EXPECT_EQ("A<::A<int>> a;", format("A<::A<int> > a;", getGoogleStyle()));
  EXPECT_EQ("auto x = [] { A<A<A<A>>> a; };",
            format("auto x=[]{A<A<A<A> >> a;};", getGoogleStyle()));

  verifyFormat("A<A>> a;", getChromiumStyle(FormatStyle::LK_Cpp));

  verifyFormat("test >> a >> b;");
  verifyFormat("test << a >> b;");

  verifyFormat("f<int>();");
  verifyFormat("template <typename T> void f() {}");
  verifyFormat("struct A<std::enable_if<sizeof(T2) < sizeof(int32)>::type>;");
  verifyFormat("struct A<std::enable_if<sizeof(T2) ? sizeof(int32) : "
               "sizeof(char)>::type>;");
  verifyFormat("template <class T> struct S<std::is_arithmetic<T>{}> {};");

  // Not template parameters.
  verifyFormat("return a < b && c > d;");
  verifyFormat("void f() {\n"
               "  while (a < b && c > d) {\n"
               "  }\n"
               "}");
  verifyFormat("template <typename... Types>\n"
               "typename enable_if<0 < sizeof...(Types)>::type Foo() {}");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaa >> aaaaa);",
               getLLVMStyleWithColumns(60));
  verifyFormat("static_assert(is_convertible<A &&, B>::value, \"AAA\");");
  verifyFormat("Constructor(A... a) : a_(X<A>{std::forward<A>(a)}...) {}");
  verifyFormat("< < < < < < < < < < < < < < < < < < < < < < < < < < < < < <");
}

TEST_F(FormatTest, UnderstandsBinaryOperators) {
  verifyFormat("COMPARE(a, ==, b);");
}

TEST_F(FormatTest, UnderstandsPointersToMembers) {
  verifyFormat("int A::*x;");
  verifyFormat("int (S::*func)(void *);");
  verifyFormat("void f() { int (S::*func)(void *); }");
  verifyFormat("typedef bool *(Class::*Member)() const;");
  verifyFormat("void f() {\n"
               "  (a->*f)();\n"
               "  a->*x;\n"
               "  (a.*f)();\n"
               "  ((*a).*f)();\n"
               "  a.*x;\n"
               "}");
  verifyFormat("void f() {\n"
               "  (a->*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)(\n"
               "      aaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb);\n"
               "}");
  verifyFormat(
      "(aaaaaaaaaa->*bbbbbbb)(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa));");
  FormatStyle Style = getLLVMStyle();
  Style.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("typedef bool* (Class::*Member)() const;", Style);
}

TEST_F(FormatTest, UnderstandsUnaryOperators) {
  verifyFormat("int a = -2;");
  verifyFormat("f(-1, -2, -3);");
  verifyFormat("a[-1] = 5;");
  verifyFormat("int a = 5 + -2;");
  verifyFormat("if (i == -1) {\n}");
  verifyFormat("if (i != -1) {\n}");
  verifyFormat("if (i > -1) {\n}");
  verifyFormat("if (i < -1) {\n}");
  verifyFormat("++(a->f());");
  verifyFormat("--(a->f());");
  verifyFormat("(a->f())++;");
  verifyFormat("a[42]++;");
  verifyFormat("if (!(a->f())) {\n}");

  verifyFormat("a-- > b;");
  verifyFormat("b ? -a : c;");
  verifyFormat("n * sizeof char16;");
  verifyFormat("n * alignof char16;", getGoogleStyle());
  verifyFormat("sizeof(char);");
  verifyFormat("alignof(char);", getGoogleStyle());

  verifyFormat("return -1;");
  verifyFormat("switch (a) {\n"
               "case -1:\n"
               "  break;\n"
               "}");
  verifyFormat("#define X -1");
  verifyFormat("#define X -kConstant");

  verifyFormat("const NSPoint kBrowserFrameViewPatternOffset = {-5, +3};");
  verifyFormat("const NSPoint kBrowserFrameViewPatternOffset = {+5, -3};");

  verifyFormat("int a = /* confusing comment */ -1;");
  // FIXME: The space after 'i' is wrong, but hopefully, this is a rare case.
  verifyFormat("int a = i /* confusing comment */++;");
}

TEST_F(FormatTest, DoesNotIndentRelativeToUnaryOperators) {
  verifyFormat("if (!aaaaaaaaaa( // break\n"
               "        aaaaa)) {\n"
               "}");
  verifyFormat("aaaaaaaaaa(!aaaaaaaaaa( // break\n"
               "    aaaaa));");
  verifyFormat("*aaa = aaaaaaa( // break\n"
               "    bbbbbb);");
}

TEST_F(FormatTest, UnderstandsOverloadedOperators) {
  verifyFormat("bool operator<();");
  verifyFormat("bool operator>();");
  verifyFormat("bool operator=();");
  verifyFormat("bool operator==();");
  verifyFormat("bool operator!=();");
  verifyFormat("int operator+();");
  verifyFormat("int operator++();");
  verifyFormat("bool operator();");
  verifyFormat("bool operator()();");
  verifyFormat("bool operator[]();");
  verifyFormat("operator bool();");
  verifyFormat("operator int();");
  verifyFormat("operator void *();");
  verifyFormat("operator SomeType<int>();");
  verifyFormat("operator SomeType<int, int>();");
  verifyFormat("operator SomeType<SomeType<int>>();");
  verifyFormat("void *operator new(std::size_t size);");
  verifyFormat("void *operator new[](std::size_t size);");
  verifyFormat("void operator delete(void *ptr);");
  verifyFormat("void operator delete[](void *ptr);");
  verifyFormat("template <typename AAAAAAA, typename BBBBBBB>\n"
               "AAAAAAA operator/(const AAAAAAA &a, BBBBBBB &b);");

  verifyFormat(
      "ostream &operator<<(ostream &OutputStream,\n"
      "                    SomeReallyLongType WithSomeReallyLongValue);");
  verifyFormat("bool operator<(const aaaaaaaaaaaaaaaaaaaaa &left,\n"
               "               const aaaaaaaaaaaaaaaaaaaaa &right) {\n"
               "  return left.group < right.group;\n"
               "}");
  verifyFormat("SomeType &operator=(const SomeType &S);");
  verifyFormat("f.template operator()<int>();");

  verifyGoogleFormat("operator void*();");
  verifyGoogleFormat("operator SomeType<SomeType<int>>();");
  verifyGoogleFormat("operator ::A();");

  verifyFormat("using A::operator+;");
  verifyFormat("inline A operator^(const A &lhs, const A &rhs) {}\n"
               "int i;");
}

TEST_F(FormatTest, UnderstandsFunctionRefQualification) {
  verifyFormat("Deleted &operator=(const Deleted &) & = default;");
  verifyFormat("Deleted &operator=(const Deleted &) && = delete;");
  verifyFormat("SomeType MemberFunction(const Deleted &) & = delete;");
  verifyFormat("SomeType MemberFunction(const Deleted &) && = delete;");
  verifyFormat("Deleted &operator=(const Deleted &) &;");
  verifyFormat("Deleted &operator=(const Deleted &) &&;");
  verifyFormat("SomeType MemberFunction(const Deleted &) &;");
  verifyFormat("SomeType MemberFunction(const Deleted &) &&;");
  verifyFormat("SomeType MemberFunction(const Deleted &) && {}");
  verifyFormat("SomeType MemberFunction(const Deleted &) && final {}");
  verifyFormat("SomeType MemberFunction(const Deleted &) && override {}");

  FormatStyle AlignLeft = getLLVMStyle();
  AlignLeft.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("Deleted& operator=(const Deleted&) & = default;", AlignLeft);
  verifyFormat("SomeType MemberFunction(const Deleted&) & = delete;",
               AlignLeft);
  verifyFormat("Deleted& operator=(const Deleted&) &;", AlignLeft);
  verifyFormat("SomeType MemberFunction(const Deleted&) &;", AlignLeft);

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInCStyleCastParentheses = true;
  verifyFormat("Deleted &operator=(const Deleted &) & = default;", Spaces);
  verifyFormat("SomeType MemberFunction(const Deleted &) & = delete;", Spaces);
  verifyFormat("Deleted &operator=(const Deleted &) &;", Spaces);
  verifyFormat("SomeType MemberFunction(const Deleted &) &;", Spaces);

  Spaces.SpacesInCStyleCastParentheses = false;
  Spaces.SpacesInParentheses = true;
  verifyFormat("Deleted &operator=( const Deleted & ) & = default;", Spaces);
  verifyFormat("SomeType MemberFunction( const Deleted & ) & = delete;", Spaces);
  verifyFormat("Deleted &operator=( const Deleted & ) &;", Spaces);
  verifyFormat("SomeType MemberFunction( const Deleted & ) &;", Spaces);
}

TEST_F(FormatTest, UnderstandsNewAndDelete) {
  verifyFormat("void f() {\n"
               "  A *a = new A;\n"
               "  A *a = new (placement) A;\n"
               "  delete a;\n"
               "  delete (A *)a;\n"
               "}");
  verifyFormat("new (aaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaa))\n"
               "    typename aaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("auto aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    new (aaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaa))\n"
               "        typename aaaaaaaaaaaaaaaaaaaaaaaa();");
  verifyFormat("delete[] h->p;");
}

TEST_F(FormatTest, UnderstandsUsesOfStarAndAmp) {
  verifyFormat("int *f(int *a) {}");
  verifyFormat("int main(int argc, char **argv) {}");
  verifyFormat("Test::Test(int b) : a(b * b) {}");
  verifyIndependentOfContext("f(a, *a);");
  verifyFormat("void g() { f(*a); }");
  verifyIndependentOfContext("int a = b * 10;");
  verifyIndependentOfContext("int a = 10 * b;");
  verifyIndependentOfContext("int a = b * c;");
  verifyIndependentOfContext("int a += b * c;");
  verifyIndependentOfContext("int a -= b * c;");
  verifyIndependentOfContext("int a *= b * c;");
  verifyIndependentOfContext("int a /= b * c;");
  verifyIndependentOfContext("int a = *b;");
  verifyIndependentOfContext("int a = *b * c;");
  verifyIndependentOfContext("int a = b * *c;");
  verifyIndependentOfContext("int a = b * (10);");
  verifyIndependentOfContext("S << b * (10);");
  verifyIndependentOfContext("return 10 * b;");
  verifyIndependentOfContext("return *b * *c;");
  verifyIndependentOfContext("return a & ~b;");
  verifyIndependentOfContext("f(b ? *c : *d);");
  verifyIndependentOfContext("int a = b ? *c : *d;");
  verifyIndependentOfContext("*b = a;");
  verifyIndependentOfContext("a * ~b;");
  verifyIndependentOfContext("a * !b;");
  verifyIndependentOfContext("a * +b;");
  verifyIndependentOfContext("a * -b;");
  verifyIndependentOfContext("a * ++b;");
  verifyIndependentOfContext("a * --b;");
  verifyIndependentOfContext("a[4] * b;");
  verifyIndependentOfContext("a[a * a] = 1;");
  verifyIndependentOfContext("f() * b;");
  verifyIndependentOfContext("a * [self dostuff];");
  verifyIndependentOfContext("int x = a * (a + b);");
  verifyIndependentOfContext("(a *)(a + b);");
  verifyIndependentOfContext("*(int *)(p & ~3UL) = 0;");
  verifyIndependentOfContext("int *pa = (int *)&a;");
  verifyIndependentOfContext("return sizeof(int **);");
  verifyIndependentOfContext("return sizeof(int ******);");
  verifyIndependentOfContext("return (int **&)a;");
  verifyIndependentOfContext("f((*PointerToArray)[10]);");
  verifyFormat("void f(Type (*parameter)[10]) {}");
  verifyFormat("void f(Type (&parameter)[10]) {}");
  verifyGoogleFormat("return sizeof(int**);");
  verifyIndependentOfContext("Type **A = static_cast<Type **>(P);");
  verifyGoogleFormat("Type** A = static_cast<Type**>(P);");
  verifyFormat("auto a = [](int **&, int ***) {};");
  verifyFormat("auto PointerBinding = [](const char *S) {};");
  verifyFormat("typedef typeof(int(int, int)) *MyFunc;");
  verifyFormat("[](const decltype(*a) &value) {}");
  verifyFormat("decltype(a * b) F();");
  verifyFormat("#define MACRO() [](A *a) { return 1; }");
  verifyIndependentOfContext("typedef void (*f)(int *a);");
  verifyIndependentOfContext("int i{a * b};");
  verifyIndependentOfContext("aaa && aaa->f();");
  verifyIndependentOfContext("int x = ~*p;");
  verifyFormat("Constructor() : a(a), area(width * height) {}");
  verifyFormat("Constructor() : a(a), area(a, width * height) {}");
  verifyGoogleFormat("MACRO Constructor(const int& i) : a(a), b(b) {}");
  verifyFormat("void f() { f(a, c * d); }");
  verifyFormat("void f() { f(new a(), c * d); }");

  verifyIndependentOfContext("InvalidRegions[*R] = 0;");

  verifyIndependentOfContext("A<int *> a;");
  verifyIndependentOfContext("A<int **> a;");
  verifyIndependentOfContext("A<int *, int *> a;");
  verifyIndependentOfContext("A<int *[]> a;");
  verifyIndependentOfContext(
      "const char *const p = reinterpret_cast<const char *const>(q);");
  verifyIndependentOfContext("A<int **, int **> a;");
  verifyIndependentOfContext("void f(int *a = d * e, int *b = c * d);");
  verifyFormat("for (char **a = b; *a; ++a) {\n}");
  verifyFormat("for (; a && b;) {\n}");
  verifyFormat("bool foo = true && [] { return false; }();");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa, *aaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyGoogleFormat("**outparam = 1;");
  verifyGoogleFormat("*outparam = a * b;");
  verifyGoogleFormat("int main(int argc, char** argv) {}");
  verifyGoogleFormat("A<int*> a;");
  verifyGoogleFormat("A<int**> a;");
  verifyGoogleFormat("A<int*, int*> a;");
  verifyGoogleFormat("A<int**, int**> a;");
  verifyGoogleFormat("f(b ? *c : *d);");
  verifyGoogleFormat("int a = b ? *c : *d;");
  verifyGoogleFormat("Type* t = **x;");
  verifyGoogleFormat("Type* t = *++*x;");
  verifyGoogleFormat("*++*x;");
  verifyGoogleFormat("Type* t = const_cast<T*>(&*x);");
  verifyGoogleFormat("Type* t = x++ * y;");
  verifyGoogleFormat(
      "const char* const p = reinterpret_cast<const char* const>(q);");
  verifyGoogleFormat("void f(int i = 0, SomeType** temps = NULL);");
  verifyGoogleFormat("void f(Bar* a = nullptr, Bar* b);");
  verifyGoogleFormat("template <typename T>\n"
                     "void f(int i = 0, SomeType** temps = NULL);");

  FormatStyle Left = getLLVMStyle();
  Left.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("x = *a(x) = *a(y);", Left);
  verifyFormat("for (;; * = b) {\n}", Left);
  verifyFormat("return *this += 1;", Left);

  verifyIndependentOfContext("a = *(x + y);");
  verifyIndependentOfContext("a = &(x + y);");
  verifyIndependentOfContext("*(x + y).call();");
  verifyIndependentOfContext("&(x + y)->call();");
  verifyFormat("void f() { &(*I).first; }");

  verifyIndependentOfContext("f(b * /* confusing comment */ ++c);");
  verifyFormat(
      "int *MyValues = {\n"
      "    *A, // Operator detection might be confused by the '{'\n"
      "    *BB // Operator detection might be confused by previous comment\n"
      "};");

  verifyIndependentOfContext("if (int *a = &b)");
  verifyIndependentOfContext("if (int &a = *b)");
  verifyIndependentOfContext("if (a & b[i])");
  verifyIndependentOfContext("if (a::b::c::d & b[i])");
  verifyIndependentOfContext("if (*b[i])");
  verifyIndependentOfContext("if (int *a = (&b))");
  verifyIndependentOfContext("while (int *a = &b)");
  verifyIndependentOfContext("size = sizeof *a;");
  verifyIndependentOfContext("if (a && (b = c))");
  verifyFormat("void f() {\n"
               "  for (const int &v : Values) {\n"
               "  }\n"
               "}");
  verifyFormat("for (int i = a * a; i < 10; ++i) {\n}");
  verifyFormat("for (int i = 0; i < a * a; ++i) {\n}");
  verifyGoogleFormat("for (int i = 0; i * 2 < z; i *= 2) {\n}");

  verifyFormat("#define A (!a * b)");
  verifyFormat("#define MACRO     \\\n"
               "  int *i = a * b; \\\n"
               "  void f(a *b);",
               getLLVMStyleWithColumns(19));

  verifyIndependentOfContext("A = new SomeType *[Length];");
  verifyIndependentOfContext("A = new SomeType *[Length]();");
  verifyIndependentOfContext("T **t = new T *;");
  verifyIndependentOfContext("T **t = new T *();");
  verifyGoogleFormat("A = new SomeType*[Length]();");
  verifyGoogleFormat("A = new SomeType*[Length];");
  verifyGoogleFormat("T** t = new T*;");
  verifyGoogleFormat("T** t = new T*();");

  FormatStyle PointerLeft = getLLVMStyle();
  PointerLeft.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("delete *x;", PointerLeft);
  verifyFormat("STATIC_ASSERT((a & b) == 0);");
  verifyFormat("STATIC_ASSERT(0 == (a & b));");
  verifyFormat("template <bool a, bool b> "
               "typename t::if<x && y>::type f() {}");
  verifyFormat("template <int *y> f() {}");
  verifyFormat("vector<int *> v;");
  verifyFormat("vector<int *const> v;");
  verifyFormat("vector<int *const **const *> v;");
  verifyFormat("vector<int *volatile> v;");
  verifyFormat("vector<a * b> v;");
  verifyFormat("foo<b && false>();");
  verifyFormat("foo<b & 1>();");
  verifyFormat("decltype(*::std::declval<const T &>()) void F();");
  verifyFormat(
      "template <class T, class = typename std::enable_if<\n"
      "                       std::is_integral<T>::value &&\n"
      "                       (sizeof(T) > 1 || sizeof(T) < 8)>::type>\n"
      "void F();",
      getLLVMStyleWithColumns(76));
  verifyFormat(
      "template <class T,\n"
      "          class = typename ::std::enable_if<\n"
      "              ::std::is_array<T>{} && ::std::is_array<T>{}>::type>\n"
      "void F();",
      getGoogleStyleWithColumns(68));

  verifyIndependentOfContext("MACRO(int *i);");
  verifyIndependentOfContext("MACRO(auto *a);");
  verifyIndependentOfContext("MACRO(const A *a);");
  verifyIndependentOfContext("MACRO('0' <= c && c <= '9');");
  // FIXME: Is there a way to make this work?
  // verifyIndependentOfContext("MACRO(A *a);");

  verifyFormat("DatumHandle const *operator->() const { return input_; }");
  verifyFormat("return options != nullptr && operator==(*options);");

  EXPECT_EQ("#define OP(x)                                    \\\n"
            "  ostream &operator<<(ostream &s, const A &a) {  \\\n"
            "    return s << a.DebugString();                 \\\n"
            "  }",
            format("#define OP(x) \\\n"
                   "  ostream &operator<<(ostream &s, const A &a) { \\\n"
                   "    return s << a.DebugString(); \\\n"
                   "  }",
                   getLLVMStyleWithColumns(50)));

  // FIXME: We cannot handle this case yet; we might be able to figure out that
  // foo<x> d > v; doesn't make sense.
  verifyFormat("foo<a<b && c> d> v;");

  FormatStyle PointerMiddle = getLLVMStyle();
  PointerMiddle.PointerAlignment = FormatStyle::PAS_Middle;
  verifyFormat("delete *x;", PointerMiddle);
  verifyFormat("int * x;", PointerMiddle);
  verifyFormat("template <int * y> f() {}", PointerMiddle);
  verifyFormat("int * f(int * a) {}", PointerMiddle);
  verifyFormat("int main(int argc, char ** argv) {}", PointerMiddle);
  verifyFormat("Test::Test(int b) : a(b * b) {}", PointerMiddle);
  verifyFormat("A<int *> a;", PointerMiddle);
  verifyFormat("A<int **> a;", PointerMiddle);
  verifyFormat("A<int *, int *> a;", PointerMiddle);
  verifyFormat("A<int * []> a;", PointerMiddle);
  verifyFormat("A = new SomeType *[Length]();", PointerMiddle);
  verifyFormat("A = new SomeType *[Length];", PointerMiddle);
  verifyFormat("T ** t = new T *;", PointerMiddle);

  // Member function reference qualifiers aren't binary operators.
  verifyFormat("string // break\n"
               "operator()() & {}");
  verifyFormat("string // break\n"
               "operator()() && {}");
  verifyGoogleFormat("template <typename T>\n"
                     "auto x() & -> int {}");
}

TEST_F(FormatTest, UnderstandsAttributes) {
  verifyFormat("SomeType s __attribute__((unused)) (InitValue);");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa __attribute__((unused))\n"
               "aaaaaaaaaaaaaaaaaaaaaaa(int i);");
  FormatStyle AfterType = getLLVMStyle();
  AfterType.AlwaysBreakAfterReturnType = FormatStyle::RTBS_AllDefinitions;
  verifyFormat("__attribute__((nodebug)) void\n"
               "foo() {}\n",
               AfterType);
}

TEST_F(FormatTest, UnderstandsEllipsis) {
  verifyFormat("int printf(const char *fmt, ...);");
  verifyFormat("template <class... Ts> void Foo(Ts... ts) { Foo(ts...); }");
  verifyFormat("template <class... Ts> void Foo(Ts *... ts) {}");

  FormatStyle PointersLeft = getLLVMStyle();
  PointersLeft.PointerAlignment = FormatStyle::PAS_Left;
  verifyFormat("template <class... Ts> void Foo(Ts*... ts) {}", PointersLeft);
}

TEST_F(FormatTest, AdaptivelyFormatsPointersAndReferences) {
  EXPECT_EQ("int *a;\n"
            "int *a;\n"
            "int *a;",
            format("int *a;\n"
                   "int* a;\n"
                   "int *a;",
                   getGoogleStyle()));
  EXPECT_EQ("int* a;\n"
            "int* a;\n"
            "int* a;",
            format("int* a;\n"
                   "int* a;\n"
                   "int *a;",
                   getGoogleStyle()));
  EXPECT_EQ("int *a;\n"
            "int *a;\n"
            "int *a;",
            format("int *a;\n"
                   "int * a;\n"
                   "int *  a;",
                   getGoogleStyle()));
  EXPECT_EQ("auto x = [] {\n"
            "  int *a;\n"
            "  int *a;\n"
            "  int *a;\n"
            "};",
            format("auto x=[]{int *a;\n"
                   "int * a;\n"
                   "int *  a;};",
                   getGoogleStyle()));
}

TEST_F(FormatTest, UnderstandsRvalueReferences) {
  verifyFormat("int f(int &&a) {}");
  verifyFormat("int f(int a, char &&b) {}");
  verifyFormat("void f() { int &&a = b; }");
  verifyGoogleFormat("int f(int a, char&& b) {}");
  verifyGoogleFormat("void f() { int&& a = b; }");

  verifyIndependentOfContext("A<int &&> a;");
  verifyIndependentOfContext("A<int &&, int &&> a;");
  verifyGoogleFormat("A<int&&> a;");
  verifyGoogleFormat("A<int&&, int&&> a;");

  // Not rvalue references:
  verifyFormat("template <bool B, bool C> class A {\n"
               "  static_assert(B && C, \"Something is wrong\");\n"
               "};");
  verifyGoogleFormat("#define IF(a, b, c) if (a && (b == c))");
  verifyGoogleFormat("#define WHILE(a, b, c) while (a && (b == c))");
  verifyFormat("#define A(a, b) (a && b)");
}

TEST_F(FormatTest, FormatsBinaryOperatorsPrecedingEquals) {
  verifyFormat("void f() {\n"
               "  x[aaaaaaaaa -\n"
               "    b] = 23;\n"
               "}",
               getLLVMStyleWithColumns(15));
}

TEST_F(FormatTest, FormatsCasts) {
  verifyFormat("Type *A = static_cast<Type *>(P);");
  verifyFormat("Type *A = (Type *)P;");
  verifyFormat("Type *A = (vector<Type *, int *>)P;");
  verifyFormat("int a = (int)(2.0f);");
  verifyFormat("int a = (int)2.0f;");
  verifyFormat("x[(int32)y];");
  verifyFormat("x = (int32)y;");
  verifyFormat("#define AA(X) sizeof(((X *)NULL)->a)");
  verifyFormat("int a = (int)*b;");
  verifyFormat("int a = (int)2.0f;");
  verifyFormat("int a = (int)~0;");
  verifyFormat("int a = (int)++a;");
  verifyFormat("int a = (int)sizeof(int);");
  verifyFormat("int a = (int)+2;");
  verifyFormat("my_int a = (my_int)2.0f;");
  verifyFormat("my_int a = (my_int)sizeof(int);");
  verifyFormat("return (my_int)aaa;");
  verifyFormat("#define x ((int)-1)");
  verifyFormat("#define LENGTH(x, y) (x) - (y) + 1");
  verifyFormat("#define p(q) ((int *)&q)");
  verifyFormat("fn(a)(b) + 1;");

  verifyFormat("void f() { my_int a = (my_int)*b; }");
  verifyFormat("void f() { return P ? (my_int)*P : (my_int)0; }");
  verifyFormat("my_int a = (my_int)~0;");
  verifyFormat("my_int a = (my_int)++a;");
  verifyFormat("my_int a = (my_int)-2;");
  verifyFormat("my_int a = (my_int)1;");
  verifyFormat("my_int a = (my_int *)1;");
  verifyFormat("my_int a = (const my_int)-1;");
  verifyFormat("my_int a = (const my_int *)-1;");
  verifyFormat("my_int a = (my_int)(my_int)-1;");
  verifyFormat("my_int a = (ns::my_int)-2;");
  verifyFormat("case (my_int)ONE:");

  // FIXME: single value wrapped with paren will be treated as cast.
  verifyFormat("void f(int i = (kValue)*kMask) {}");

  verifyFormat("{ (void)F; }");

  // Don't break after a cast's
  verifyFormat("int aaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    (aaaaaaaaaaaaaaaaaaaaaaaaaa *)(aaaaaaaaaaaaaaaaaaaaaa +\n"
               "                                   bbbbbbbbbbbbbbbbbbbbbb);");

  // These are not casts.
  verifyFormat("void f(int *) {}");
  verifyFormat("f(foo)->b;");
  verifyFormat("f(foo).b;");
  verifyFormat("f(foo)(b);");
  verifyFormat("f(foo)[b];");
  verifyFormat("[](foo) { return 4; }(bar);");
  verifyFormat("(*funptr)(foo)[4];");
  verifyFormat("funptrs[4](foo)[4];");
  verifyFormat("void f(int *);");
  verifyFormat("void f(int *) = 0;");
  verifyFormat("void f(SmallVector<int>) {}");
  verifyFormat("void f(SmallVector<int>);");
  verifyFormat("void f(SmallVector<int>) = 0;");
  verifyFormat("void f(int i = (kA * kB) & kMask) {}");
  verifyFormat("int a = sizeof(int) * b;");
  verifyFormat("int a = alignof(int) * b;", getGoogleStyle());
  verifyFormat("template <> void f<int>(int i) SOME_ANNOTATION;");
  verifyFormat("f(\"%\" SOME_MACRO(ll) \"d\");");
  verifyFormat("aaaaa &operator=(const aaaaa &) LLVM_DELETED_FUNCTION;");

  // These are not casts, but at some point were confused with casts.
  verifyFormat("virtual void foo(int *) override;");
  verifyFormat("virtual void foo(char &) const;");
  verifyFormat("virtual void foo(int *a, char *) const;");
  verifyFormat("int a = sizeof(int *) + b;");
  verifyFormat("int a = alignof(int *) + b;", getGoogleStyle());
  verifyFormat("bool b = f(g<int>) && c;");
  verifyFormat("typedef void (*f)(int i) func;");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa *foo = (aaaaaaaaaaaaaaaaa *)\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;");
  // FIXME: The indentation here is not ideal.
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    [bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] = (*cccccccccccccccc)\n"
      "        [dddddddddddddddddddddddddddddddddddddddddddddddddddddddd];");
}

TEST_F(FormatTest, FormatsFunctionTypes) {
  verifyFormat("A<bool()> a;");
  verifyFormat("A<SomeType()> a;");
  verifyFormat("A<void (*)(int, std::string)> a;");
  verifyFormat("A<void *(int)>;");
  verifyFormat("void *(*a)(int *, SomeType *);");
  verifyFormat("int (*func)(void *);");
  verifyFormat("void f() { int (*func)(void *); }");
  verifyFormat("template <class CallbackClass>\n"
               "using MyCallback = void (CallbackClass::*)(SomeObject *Data);");

  verifyGoogleFormat("A<void*(int*, SomeType*)>;");
  verifyGoogleFormat("void* (*a)(int);");
  verifyGoogleFormat(
      "template <class CallbackClass>\n"
      "using MyCallback = void (CallbackClass::*)(SomeObject* Data);");

  // Other constructs can look somewhat like function types:
  verifyFormat("A<sizeof(*x)> a;");
  verifyFormat("#define DEREF_AND_CALL_F(x) f(*x)");
  verifyFormat("some_var = function(*some_pointer_var)[0];");
  verifyFormat("void f() { function(*some_pointer_var)[0] = 10; }");
  verifyFormat("int x = f(&h)();");
}

TEST_F(FormatTest, FormatsPointersToArrayTypes) {
  verifyFormat("A (*foo_)[6];");
  verifyFormat("vector<int> (*foo_)[6];");
}

TEST_F(FormatTest, BreaksLongVariableDeclarations) {
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable;");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType const\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable;");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    *LoooooooooooooooooooooooooooooooooooooooongVariable;");

  // Different ways of ()-initializiation.
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable(1);");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable(a);");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooooongVariable({});");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    LoooooooooooooooooooooooooooooooooooooongVariable([A a]);");
}

TEST_F(FormatTest, BreaksLongDeclarations) {
  verifyFormat("typedef LoooooooooooooooooooooooooooooooooooooooongType\n"
               "    AnotherNameForTheLongType;");
  verifyFormat("typedef LongTemplateType<aaaaaaaaaaaaaaaaaaa()>\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
               "LoooooooooooooooooooooooooooooooongFunctionDeclaration();");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType *\n"
               "LoooooooooooooooooooooooooooooooongFunctionDeclaration();");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType MACRO\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType const\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  verifyFormat("decltype(LoooooooooooooooooooooooooooooooooooooooongName)\n"
               "LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}");
  FormatStyle Indented = getLLVMStyle();
  Indented.IndentWrappedFunctionNames = true;
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
               "    LoooooooooooooooooooooooooooooooongFunctionDeclaration();",
               Indented);
  verifyFormat(
      "LoooooooooooooooooooooooooooooooooooooooongReturnType\n"
      "    LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}",
      Indented);
  verifyFormat(
      "LoooooooooooooooooooooooooooooooooooooooongReturnType const\n"
      "    LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}",
      Indented);
  verifyFormat(
      "decltype(LoooooooooooooooooooooooooooooooooooooooongName)\n"
      "    LooooooooooooooooooooooooooooooooooongFunctionDefinition() {}",
      Indented);

  // FIXME: Without the comment, this breaks after "(".
  verifyFormat("LoooooooooooooooooooooooooooooooooooooooongType  // break\n"
               "    (*LoooooooooooooooooooooooooooongFunctionTypeVarialbe)();",
               getGoogleStyle());

  verifyFormat("int *someFunction(int LoooooooooooooooooooongParam1,\n"
               "                  int LoooooooooooooooooooongParam2) {}");
  verifyFormat(
      "TypeSpecDecl *TypeSpecDecl::Create(ASTContext &C, DeclContext *DC,\n"
      "                                   SourceLocation L, IdentifierIn *II,\n"
      "                                   Type *T) {}");
  verifyFormat("ReallyLongReturnType<TemplateParam1, TemplateParam2>\n"
               "ReallyReaaallyLongFunctionName(\n"
               "    const std::string &SomeParameter,\n"
               "    const SomeType<string, SomeOtherTemplateParameter>\n"
               "        &ReallyReallyLongParameterName,\n"
               "    const SomeType<string, SomeOtherTemplateParameter>\n"
               "        &AnotherLongParameterName) {}");
  verifyFormat("template <typename A>\n"
               "SomeLoooooooooooooooooooooongType<\n"
               "    typename some_namespace::SomeOtherType<A>::Type>\n"
               "Function() {}");

  verifyGoogleFormat(
      "aaaaaaaaaaaaaaaa::aaaaaaaaaaaaaaaa<aaaaaaaaaaaaa, aaaaaaaaaaaa>\n"
      "    aaaaaaaaaaaaaaaaaaaaaaa;");
  verifyGoogleFormat(
      "TypeSpecDecl* TypeSpecDecl::Create(ASTContext& C, DeclContext* DC,\n"
      "                                   SourceLocation L) {}");
  verifyGoogleFormat(
      "some_namespace::LongReturnType\n"
      "long_namespace::SomeVeryLongClass::SomeVeryLongFunction(\n"
      "    int first_long_parameter, int second_parameter) {}");

  verifyGoogleFormat("template <typename T>\n"
                     "aaaaaaaa::aaaaa::aaaaaa<T, aaaaaaaaaaaaaaaaaaaaaaaaa>\n"
                     "aaaaaaaaaaaaaaaaaaaaaaaa<T>::aaaaaaa() {}");
  verifyGoogleFormat("A<A<A>> aaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
                     "                   int aaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("typedef size_t (*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)(\n"
               "    const aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "        *aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    vector<aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    vector<aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa>>\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
}

TEST_F(FormatTest, FormatsArrays) {
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaa[aaaaaaaaaaaaaaaaaaaaaaaaa]\n"
               "                         [bbbbbbbbbbbbbbbbbbbbbbbbb] = c;");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaa[aaaaaaaaaaa(aaaaaaaaaaaa)]\n"
               "                         [bbbbbbbbbbb(bbbbbbbbbbbb)] = c;");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaa &&\n"
               "    aaaaaaaaaaaaaaaaaaa[aaaaaaaaaaaaa][aaaaaaaaaaaaa]) {\n}");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    [bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] = ccccccccccc;");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    [a][bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] = cccccccc;");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]\n"
               "    [bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] = ccccccccccc;");
  verifyFormat(
      "llvm::outs() << \"aaaaaaaaaaaa: \"\n"
      "             << (*aaaaaaaiaaaaaaa)[aaaaaaaaaaaaaaaaaaaaaaaaa]\n"
      "                                  [aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa];");

  verifyGoogleFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa<int>\n"
                     "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa[aaaaaaaaaaaa];");
  verifyFormat(
      "aaaaaaaaaaa aaaaaaaaaaaaaaa = aaaaaaaaaaaaaaaaaaaaaaaaaa->aaaaaaaaa[0]\n"
      "                                  .aaaaaaa[0]\n"
      "                                  .aaaaaaaaaaaaaaaaaaaaaa();");

  verifyNoCrash("a[,Y?)]", getLLVMStyleWithColumns(10));
}

TEST_F(FormatTest, LineStartsWithSpecialCharacter) {
  verifyFormat("(a)->b();");
  verifyFormat("--a;");
}

TEST_F(FormatTest, HandlesIncludeDirectives) {
  verifyFormat("#include <string>\n"
               "#include <a/b/c.h>\n"
               "#include \"a/b/string\"\n"
               "#include \"string.h\"\n"
               "#include \"string.h\"\n"
               "#include <a-a>\n"
               "#include < path with space >\n"
               "#include_next <test.h>"
               "#include \"abc.h\" // this is included for ABC\n"
               "#include \"some long include\" // with a comment\n"
               "#include \"some very long include paaaaaaaaaaaaaaaaaaaaaaath\"",
               getLLVMStyleWithColumns(35));
  EXPECT_EQ("#include \"a.h\"", format("#include  \"a.h\""));
  EXPECT_EQ("#include <a>", format("#include<a>"));

  verifyFormat("#import <string>");
  verifyFormat("#import <a/b/c.h>");
  verifyFormat("#import \"a/b/string\"");
  verifyFormat("#import \"string.h\"");
  verifyFormat("#import \"string.h\"");
  verifyFormat("#if __has_include(<strstream>)\n"
               "#include <strstream>\n"
               "#endif");

  verifyFormat("#define MY_IMPORT <a/b>");

  // Protocol buffer definition or missing "#".
  verifyFormat("import \"aaaaaaaaaaaaaaaaa/aaaaaaaaaaaaaaa\";",
               getLLVMStyleWithColumns(30));

  FormatStyle Style = getLLVMStyle();
  Style.AlwaysBreakBeforeMultilineStrings = true;
  Style.ColumnLimit = 0;
  verifyFormat("#import \"abc.h\"", Style);

  // But 'import' might also be a regular C++ namespace.
  verifyFormat("import::SomeFunction(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                     aaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
}

//===----------------------------------------------------------------------===//
// Error recovery tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, IncompleteParameterLists) {
  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackParameters = false;
  verifyFormat("void aaaaaaaaaaaaaaaaaa(int level,\n"
               "                        double *min_x,\n"
               "                        double *max_x,\n"
               "                        double *min_y,\n"
               "                        double *max_y,\n"
               "                        double *min_z,\n"
               "                        double *max_z, ) {}",
               NoBinPacking);
}

TEST_F(FormatTest, IncorrectCodeTrailingStuff) {
  verifyFormat("void f() { return; }\n42");
  verifyFormat("void f() {\n"
               "  if (0)\n"
               "    return;\n"
               "}\n"
               "42");
  verifyFormat("void f() { return }\n42");
  verifyFormat("void f() {\n"
               "  if (0)\n"
               "    return\n"
               "}\n"
               "42");
}

TEST_F(FormatTest, IncorrectCodeMissingSemicolon) {
  EXPECT_EQ("void f() { return }", format("void  f ( )  {  return  }"));
  EXPECT_EQ("void f() {\n"
            "  if (a)\n"
            "    return\n"
            "}",
            format("void  f  (  )  {  if  ( a )  return  }"));
  EXPECT_EQ("namespace N {\n"
            "void f()\n"
            "}",
            format("namespace  N  {  void f()  }"));
  EXPECT_EQ("namespace N {\n"
            "void f() {}\n"
            "void g()\n"
            "}",
            format("namespace N  { void f( ) { } void g( ) }"));
}

TEST_F(FormatTest, IndentationWithinColumnLimitNotPossible) {
  verifyFormat("int aaaaaaaa =\n"
               "    // Overlylongcomment\n"
               "    b;",
               getLLVMStyleWithColumns(20));
  verifyFormat("function(\n"
               "    ShortArgument,\n"
               "    LoooooooooooongArgument);\n",
               getLLVMStyleWithColumns(20));
}

TEST_F(FormatTest, IncorrectAccessSpecifier) {
  verifyFormat("public:");
  verifyFormat("class A {\n"
               "public\n"
               "  void f() {}\n"
               "};");
  verifyFormat("public\n"
               "int qwerty;");
  verifyFormat("public\n"
               "B {}");
  verifyFormat("public\n"
               "{}");
  verifyFormat("public\n"
               "B { int x; }");
}

TEST_F(FormatTest, IncorrectCodeUnbalancedBraces) {
  verifyFormat("{");
  verifyFormat("#})");
  verifyNoCrash("(/**/[:!] ?[).");
}

TEST_F(FormatTest, IncorrectCodeDoNoWhile) {
  verifyFormat("do {\n}");
  verifyFormat("do {\n}\n"
               "f();");
  verifyFormat("do {\n}\n"
               "wheeee(fun);");
  verifyFormat("do {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, IncorrectCodeMissingParens) {
  verifyFormat("if {\n  foo;\n  foo();\n}");
  verifyFormat("switch {\n  foo;\n  foo();\n}");
  verifyIncompleteFormat("for {\n  foo;\n  foo();\n}");
  verifyFormat("while {\n  foo;\n  foo();\n}");
  verifyFormat("do {\n  foo;\n  foo();\n} while;");
}

TEST_F(FormatTest, DoesNotTouchUnwrappedLinesWithErrors) {
  verifyIncompleteFormat("namespace {\n"
                         "class Foo { Foo (\n"
                         "};\n"
                         "} // comment");
}

TEST_F(FormatTest, IncorrectCodeErrorDetection) {
  EXPECT_EQ("{\n  {}\n", format("{\n{\n}\n"));
  EXPECT_EQ("{\n  {}\n", format("{\n  {\n}\n"));
  EXPECT_EQ("{\n  {}\n", format("{\n  {\n  }\n"));
  EXPECT_EQ("{\n  {}\n}\n}\n", format("{\n  {\n    }\n  }\n}\n"));

  EXPECT_EQ("{\n"
            "  {\n"
            "    breakme(\n"
            "        qwe);\n"
            "  }\n",
            format("{\n"
                   "    {\n"
                   " breakme(qwe);\n"
                   "}\n",
                   getLLVMStyleWithColumns(10)));
}

TEST_F(FormatTest, LayoutCallsInsideBraceInitializers) {
  verifyFormat("int x = {\n"
               "    avariable,\n"
               "    b(alongervariable)};",
               getLLVMStyleWithColumns(25));
}

TEST_F(FormatTest, LayoutBraceInitializersInReturnStatement) {
  verifyFormat("return (a)(b){1, 2, 3};");
}

TEST_F(FormatTest, LayoutCxx11BraceInitializers) {
  verifyFormat("vector<int> x{1, 2, 3, 4};");
  verifyFormat("vector<int> x{\n"
               "    1, 2, 3, 4,\n"
               "};");
  verifyFormat("vector<T> x{{}, {}, {}, {}};");
  verifyFormat("f({1, 2});");
  verifyFormat("auto v = Foo{-1};");
  verifyFormat("f({1, 2}, {{2, 3}, {4, 5}}, c, {d});");
  verifyFormat("Class::Class : member{1, 2, 3} {}");
  verifyFormat("new vector<int>{1, 2, 3};");
  verifyFormat("new int[3]{1, 2, 3};");
  verifyFormat("new int{1};");
  verifyFormat("return {arg1, arg2};");
  verifyFormat("return {arg1, SomeType{parameter}};");
  verifyFormat("int count = set<int>{f(), g(), h()}.size();");
  verifyFormat("new T{arg1, arg2};");
  verifyFormat("f(MyMap[{composite, key}]);");
  verifyFormat("class Class {\n"
               "  T member = {arg1, arg2};\n"
               "};");
  verifyFormat("vector<int> foo = {::SomeGlobalFunction()};");
  verifyFormat("static_assert(std::is_integral<int>{} + 0, \"\");");
  verifyFormat("int a = std::is_integral<int>{} + 0;");

  verifyFormat("int foo(int i) { return fo1{}(i); }");
  verifyFormat("int foo(int i) { return fo1{}(i); }");
  verifyFormat("auto i = decltype(x){};");
  verifyFormat("std::vector<int> v = {1, 0 /* comment */};");
  verifyFormat("Node n{1, Node{1000}, //\n"
               "       2};");
  verifyFormat("Aaaa aaaaaaa{\n"
               "    {\n"
               "        aaaa,\n"
               "    },\n"
               "};");
  verifyFormat("class C : public D {\n"
               "  SomeClass SC{2};\n"
               "};");
  verifyFormat("class C : public A {\n"
               "  class D : public B {\n"
               "    void f() { int i{2}; }\n"
               "  };\n"
               "};");
  verifyFormat("#define A {a, a},");

  // In combination with BinPackArguments = false.
  FormatStyle NoBinPacking = getLLVMStyle();
  NoBinPacking.BinPackArguments = false;
  verifyFormat("const Aaaaaa aaaaa = {aaaaa,\n"
               "                      bbbbb,\n"
               "                      ccccc,\n"
               "                      ddddd,\n"
               "                      eeeee,\n"
               "                      ffffff,\n"
               "                      ggggg,\n"
               "                      hhhhhh,\n"
               "                      iiiiii,\n"
               "                      jjjjjj,\n"
               "                      kkkkkk};",
               NoBinPacking);
  verifyFormat("const Aaaaaa aaaaa = {\n"
               "    aaaaa,\n"
               "    bbbbb,\n"
               "    ccccc,\n"
               "    ddddd,\n"
               "    eeeee,\n"
               "    ffffff,\n"
               "    ggggg,\n"
               "    hhhhhh,\n"
               "    iiiiii,\n"
               "    jjjjjj,\n"
               "    kkkkkk,\n"
               "};",
               NoBinPacking);
  verifyFormat(
      "const Aaaaaa aaaaa = {\n"
      "    aaaaa,  bbbbb,  ccccc,  ddddd,  eeeee,  ffffff, ggggg, hhhhhh,\n"
      "    iiiiii, jjjjjj, kkkkkk, aaaaa,  bbbbb,  ccccc,  ddddd, eeeee,\n"
      "    ffffff, ggggg,  hhhhhh, iiiiii, jjjjjj, kkkkkk,\n"
      "};",
      NoBinPacking);

  // FIXME: The alignment of these trailing comments might be bad. Then again,
  // this might be utterly useless in real code.
  verifyFormat("Constructor::Constructor()\n"
               "    : some_value{         //\n"
               "                 aaaaaaa, //\n"
               "                 bbbbbbb} {}");

  // In braced lists, the first comment is always assumed to belong to the
  // first element. Thus, it can be moved to the next or previous line as
  // appropriate.
  EXPECT_EQ("function({// First element:\n"
            "          1,\n"
            "          // Second element:\n"
            "          2});",
            format("function({\n"
                   "    // First element:\n"
                   "    1,\n"
                   "    // Second element:\n"
                   "    2});"));
  EXPECT_EQ("std::vector<int> MyNumbers{\n"
            "    // First element:\n"
            "    1,\n"
            "    // Second element:\n"
            "    2};",
            format("std::vector<int> MyNumbers{// First element:\n"
                   "                           1,\n"
                   "                           // Second element:\n"
                   "                           2};",
                   getLLVMStyleWithColumns(30)));
  // A trailing comma should still lead to an enforced line break.
  EXPECT_EQ("vector<int> SomeVector = {\n"
            "    // aaa\n"
            "    1, 2,\n"
            "};",
            format("vector<int> SomeVector = { // aaa\n"
                   "    1, 2, };"));

  FormatStyle ExtraSpaces = getLLVMStyle();
  ExtraSpaces.Cpp11BracedListStyle = false;
  ExtraSpaces.ColumnLimit = 75;
  verifyFormat("vector<int> x{ 1, 2, 3, 4 };", ExtraSpaces);
  verifyFormat("vector<T> x{ {}, {}, {}, {} };", ExtraSpaces);
  verifyFormat("f({ 1, 2 });", ExtraSpaces);
  verifyFormat("auto v = Foo{ 1 };", ExtraSpaces);
  verifyFormat("f({ 1, 2 }, { { 2, 3 }, { 4, 5 } }, c, { d });", ExtraSpaces);
  verifyFormat("Class::Class : member{ 1, 2, 3 } {}", ExtraSpaces);
  verifyFormat("new vector<int>{ 1, 2, 3 };", ExtraSpaces);
  verifyFormat("new int[3]{ 1, 2, 3 };", ExtraSpaces);
  verifyFormat("return { arg1, arg2 };", ExtraSpaces);
  verifyFormat("return { arg1, SomeType{ parameter } };", ExtraSpaces);
  verifyFormat("int count = set<int>{ f(), g(), h() }.size();", ExtraSpaces);
  verifyFormat("new T{ arg1, arg2 };", ExtraSpaces);
  verifyFormat("f(MyMap[{ composite, key }]);", ExtraSpaces);
  verifyFormat("class Class {\n"
               "  T member = { arg1, arg2 };\n"
               "};",
               ExtraSpaces);
  verifyFormat(
      "foo = aaaaaaaaaaa ? vector<int>{ aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                                 aaaaaaaaaaaaaaaaaaaa, aaaaa }\n"
      "                  : vector<int>{ bbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
      "                                 bbbbbbbbbbbbbbbbbbbb, bbbbb };",
      ExtraSpaces);
  verifyFormat("DoSomethingWithVector({} /* No data */);", ExtraSpaces);
  verifyFormat("DoSomethingWithVector({ {} /* No data */ }, { { 1, 2 } });",
               ExtraSpaces);
  verifyFormat(
      "someFunction(OtherParam,\n"
      "             BracedList{ // comment 1 (Forcing interesting break)\n"
      "                         param1, param2,\n"
      "                         // comment 2\n"
      "                         param3, param4 });",
      ExtraSpaces);
  verifyFormat(
      "std::this_thread::sleep_for(\n"
      "    std::chrono::nanoseconds{ std::chrono::seconds{ 1 } } / 5);",
      ExtraSpaces);
  verifyFormat("std::vector<MyValues> aaaaaaaaaaaaaaaaaaa{\n"
               "    aaaaaaa,\n"
               "    aaaaaaaaaa,\n"
               "    aaaaa,\n"
               "    aaaaaaaaaaaaaaa,\n"
               "    aaa,\n"
               "    aaaaaaaaaa,\n"
               "    a,\n"
               "    aaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaa,\n"
               "    a};");
  verifyFormat("vector<int> foo = { ::SomeGlobalFunction() };", ExtraSpaces);
}

TEST_F(FormatTest, FormatsBracedListsInColumnLayout) {
  verifyFormat("vector<int> x = {1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777};");
  verifyFormat("vector<int> x = {1, 22, 333, 4444, 55555, 666666, 7777777, //\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, //\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777,\n"
               "                 1, 22, 333, 4444, 55555, 666666, 7777777};");
  verifyFormat(
      "vector<int> x = {1,       22, 333, 4444, 55555, 666666, 7777777,\n"
      "                 1,       22, 333, 4444, 55555, 666666, 7777777,\n"
      "                 1,       22, 333, 4444, 55555, 666666, // comment\n"
      "                 7777777, 1,  22,  333,  4444,  55555,  666666,\n"
      "                 7777777, 1,  22,  333,  4444,  55555,  666666,\n"
      "                 7777777, 1,  22,  333,  4444,  55555,  666666,\n"
      "                 7777777};");
  verifyFormat("static const uint16_t CallerSavedRegs64Bittttt[] = {\n"
               "    X86::RAX, X86::RDX, X86::RCX, X86::RSI, X86::RDI,\n"
               "    X86::R8,  X86::R9,  X86::R10, X86::R11, 0};");
  verifyFormat("static const uint16_t CallerSavedRegs64Bittttt[] = {\n"
               "    X86::RAX, X86::RDX, X86::RCX, X86::RSI, X86::RDI,\n"
               "    // Separating comment.\n"
               "    X86::R8, X86::R9, X86::R10, X86::R11, 0};");
  verifyFormat("static const uint16_t CallerSavedRegs64Bittttt[] = {\n"
               "    // Leading comment\n"
               "    X86::RAX, X86::RDX, X86::RCX, X86::RSI, X86::RDI,\n"
               "    X86::R8,  X86::R9,  X86::R10, X86::R11, 0};");
  verifyFormat("vector<int> x = {1, 1, 1, 1,\n"
               "                 1, 1, 1, 1};",
               getLLVMStyleWithColumns(39));
  verifyFormat("vector<int> x = {1, 1, 1, 1,\n"
               "                 1, 1, 1, 1};",
               getLLVMStyleWithColumns(38));
  verifyFormat("vector<int> aaaaaaaaaaaaaaaaaaaaaa = {\n"
               "    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};",
               getLLVMStyleWithColumns(43));
  verifyFormat(
      "static unsigned SomeValues[10][3] = {\n"
      "    {1, 4, 0},  {4, 9, 0},  {4, 5, 9},  {8, 5, 4}, {1, 8, 4},\n"
      "    {10, 1, 6}, {11, 0, 9}, {2, 11, 9}, {5, 2, 9}, {11, 2, 7}};");
  verifyFormat("static auto fields = new vector<string>{\n"
               "    \"aaaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaaaaaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaa\",\n"
               "    \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\n"
               "};");
  verifyFormat("vector<int> x = {1, 2, 3, 4, aaaaaaaaaaaaaaaaa, 6};");
  verifyFormat("vector<int> x = {1, aaaaaaaaaaaaaaaaaaaaaa,\n"
               "                 2, bbbbbbbbbbbbbbbbbbbbbb,\n"
               "                 3, cccccccccccccccccccccc};",
               getLLVMStyleWithColumns(60));

  // Trailing commas.
  verifyFormat("vector<int> x = {\n"
               "    1, 1, 1, 1, 1, 1, 1, 1,\n"
               "};",
               getLLVMStyleWithColumns(39));
  verifyFormat("vector<int> x = {\n"
               "    1, 1, 1, 1, 1, 1, 1, 1, //\n"
               "};",
               getLLVMStyleWithColumns(39));
  verifyFormat("vector<int> x = {1, 1, 1, 1,\n"
               "                 1, 1, 1, 1,\n"
               "                 /**/ /**/};",
               getLLVMStyleWithColumns(39));

  // Trailing comment in the first line.
  verifyFormat("vector<int> iiiiiiiiiiiiiii = {                      //\n"
               "    1111111111, 2222222222, 33333333333, 4444444444, //\n"
               "    111111111,  222222222,  3333333333,  444444444,  //\n"
               "    11111111,   22222222,   333333333,   44444444};");
  // Trailing comment in the last line.
  verifyFormat("int aaaaa[] = {\n"
               "    1, 2, 3, // comment\n"
               "    4, 5, 6  // comment\n"
               "};");

  // With nested lists, we should either format one item per line or all nested
  // lists one on line.
  // FIXME: For some nested lists, we can do better.
  verifyFormat("return {{aaaaaaaaaaaaaaaaaaaaa},\n"
               "        {aaaaaaaaaaaaaaaaaaa},\n"
               "        {aaaaaaaaaaaaaaaaaaaaa},\n"
               "        {aaaaaaaaaaaaaaaaa}};",
               getLLVMStyleWithColumns(60));
  verifyFormat(
      "SomeStruct my_struct_array = {\n"
      "    {aaaaaa, aaaaaaaa, aaaaaaaaaa, aaaaaaaaa, aaaaaaaaa, aaaaaaaaaa,\n"
      "     aaaaaaaaaaaaa, aaaaaaa, aaa},\n"
      "    {aaa, aaa},\n"
      "    {aaa, aaa},\n"
      "    {aaaa, aaaa, aaaa, aaaa, aaaa, aaaa, aaaa, aaa},\n"
      "    {aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "     aaaaaaaaaaaa, a, aaaaaaaaaa, aaaaaaaaa, aaa}};");

  // No column layout should be used here.
  verifyFormat("aaaaaaaaaaaaaaa = {aaaaaaaaaaaaaaaaaaaaaaaaaaa, 0, 0,\n"
               "                   bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb};");

  verifyNoCrash("a<,");
  
  // No braced initializer here.
  verifyFormat("void f() {\n"
               "  struct Dummy {};\n"
               "  f(v);\n"
               "}");

  // Long lists should be formatted in columns even if they are nested.
  verifyFormat(
      "vector<int> x = function({1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777,\n"
      "                          1, 22, 333, 4444, 55555, 666666, 7777777});");
}

TEST_F(FormatTest, PullTrivialFunctionDefinitionsIntoSingleLine) {
  FormatStyle DoNotMerge = getLLVMStyle();
  DoNotMerge.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;

  verifyFormat("void f() { return 42; }");
  verifyFormat("void f() {\n"
               "  return 42;\n"
               "}",
               DoNotMerge);
  verifyFormat("void f() {\n"
               "  // Comment\n"
               "}");
  verifyFormat("{\n"
               "#error {\n"
               "  int a;\n"
               "}");
  verifyFormat("{\n"
               "  int a;\n"
               "#error {\n"
               "}");
  verifyFormat("void f() {} // comment");
  verifyFormat("void f() { int a; } // comment");
  verifyFormat("void f() {\n"
               "} // comment",
               DoNotMerge);
  verifyFormat("void f() {\n"
               "  int a;\n"
               "} // comment",
               DoNotMerge);
  verifyFormat("void f() {\n"
               "} // comment",
               getLLVMStyleWithColumns(15));

  verifyFormat("void f() { return 42; }", getLLVMStyleWithColumns(23));
  verifyFormat("void f() {\n  return 42;\n}", getLLVMStyleWithColumns(22));

  verifyFormat("void f() {}", getLLVMStyleWithColumns(11));
  verifyFormat("void f() {\n}", getLLVMStyleWithColumns(10));
  verifyFormat("class C {\n"
               "  C()\n"
               "      : iiiiiiii(nullptr),\n"
               "        kkkkkkk(nullptr),\n"
               "        mmmmmmm(nullptr),\n"
               "        nnnnnnn(nullptr) {}\n"
               "};",
               getGoogleStyle());

  FormatStyle NoColumnLimit = getLLVMStyle();
  NoColumnLimit.ColumnLimit = 0;
  EXPECT_EQ("A() : b(0) {}", format("A():b(0){}", NoColumnLimit));
  EXPECT_EQ("class C {\n"
            "  A() : b(0) {}\n"
            "};",
            format("class C{A():b(0){}};", NoColumnLimit));
  EXPECT_EQ("A()\n"
            "    : b(0) {\n"
            "}",
            format("A()\n:b(0)\n{\n}", NoColumnLimit));

  FormatStyle DoNotMergeNoColumnLimit = NoColumnLimit;
  DoNotMergeNoColumnLimit.AllowShortFunctionsOnASingleLine =
      FormatStyle::SFS_None;
  EXPECT_EQ("A()\n"
            "    : b(0) {\n"
            "}",
            format("A():b(0){}", DoNotMergeNoColumnLimit));
  EXPECT_EQ("A()\n"
            "    : b(0) {\n"
            "}",
            format("A()\n:b(0)\n{\n}", DoNotMergeNoColumnLimit));

  verifyFormat("#define A          \\\n"
               "  void f() {       \\\n"
               "    int i;         \\\n"
               "  }",
               getLLVMStyleWithColumns(20));
  verifyFormat("#define A           \\\n"
               "  void f() { int i; }",
               getLLVMStyleWithColumns(21));
  verifyFormat("#define A            \\\n"
               "  void f() {         \\\n"
               "    int i;           \\\n"
               "  }                  \\\n"
               "  int j;",
               getLLVMStyleWithColumns(22));
  verifyFormat("#define A             \\\n"
               "  void f() { int i; } \\\n"
               "  int j;",
               getLLVMStyleWithColumns(23));
}

TEST_F(FormatTest, PullInlineFunctionDefinitionsIntoSingleLine) {
  FormatStyle MergeInlineOnly = getLLVMStyle();
  MergeInlineOnly.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  verifyFormat("class C {\n"
               "  int f() { return 42; }\n"
               "};",
               MergeInlineOnly);
  verifyFormat("int f() {\n"
               "  return 42;\n"
               "}",
               MergeInlineOnly);
}

TEST_F(FormatTest, UnderstandContextOfRecordTypeKeywords) {
  // Elaborate type variable declarations.
  verifyFormat("struct foo a = {bar};\nint n;");
  verifyFormat("class foo a = {bar};\nint n;");
  verifyFormat("union foo a = {bar};\nint n;");

  // Elaborate types inside function definitions.
  verifyFormat("struct foo f() {}\nint n;");
  verifyFormat("class foo f() {}\nint n;");
  verifyFormat("union foo f() {}\nint n;");

  // Templates.
  verifyFormat("template <class X> void f() {}\nint n;");
  verifyFormat("template <struct X> void f() {}\nint n;");
  verifyFormat("template <union X> void f() {}\nint n;");

  // Actual definitions...
  verifyFormat("struct {\n} n;");
  verifyFormat(
      "template <template <class T, class Y>, class Z> class X {\n} n;");
  verifyFormat("union Z {\n  int n;\n} x;");
  verifyFormat("class MACRO Z {\n} n;");
  verifyFormat("class MACRO(X) Z {\n} n;");
  verifyFormat("class __attribute__(X) Z {\n} n;");
  verifyFormat("class __declspec(X) Z {\n} n;");
  verifyFormat("class A##B##C {\n} n;");
  verifyFormat("class alignas(16) Z {\n} n;");
  verifyFormat("class MACRO(X) alignas(16) Z {\n} n;");
  verifyFormat("class MACROA MACRO(X) Z {\n} n;");

  // Redefinition from nested context:
  verifyFormat("class A::B::C {\n} n;");

  // Template definitions.
  verifyFormat(
      "template <typename F>\n"
      "Matcher(const Matcher<F> &Other,\n"
      "        typename enable_if_c<is_base_of<F, T>::value &&\n"
      "                             !is_same<F, T>::value>::type * = 0)\n"
      "    : Implementation(new ImplicitCastMatcher<F>(Other)) {}");

  // FIXME: This is still incorrectly handled at the formatter side.
  verifyFormat("template <> struct X < 15, i<3 && 42 < 50 && 33 < 28> {};");
  verifyFormat("int i = SomeFunction(a<b, a> b);");

  // FIXME:
  // This now gets parsed incorrectly as class definition.
  // verifyFormat("class A<int> f() {\n}\nint n;");

  // Elaborate types where incorrectly parsing the structural element would
  // break the indent.
  verifyFormat("if (true)\n"
               "  class X x;\n"
               "else\n"
               "  f();\n");

  // This is simply incomplete. Formatting is not important, but must not crash.
  verifyFormat("class A:");
}

TEST_F(FormatTest, DoNotInterfereWithErrorAndWarning) {
  EXPECT_EQ("#error Leave     all         white!!!!! space* alone!\n",
            format("#error Leave     all         white!!!!! space* alone!\n"));
  EXPECT_EQ(
      "#warning Leave     all         white!!!!! space* alone!\n",
      format("#warning Leave     all         white!!!!! space* alone!\n"));
  EXPECT_EQ("#error 1", format("  #  error   1"));
  EXPECT_EQ("#warning 1", format("  #  warning 1"));
}

TEST_F(FormatTest, FormatHashIfExpressions) {
  verifyFormat("#if AAAA && BBBB");
  verifyFormat("#if (AAAA && BBBB)");
  verifyFormat("#elif (AAAA && BBBB)");
  // FIXME: Come up with a better indentation for #elif.
  verifyFormat(
      "#if !defined(AAAAAAA) && (defined CCCCCC || defined DDDDDD) &&  \\\n"
      "    defined(BBBBBBBB)\n"
      "#elif !defined(AAAAAA) && (defined CCCCC || defined DDDDDD) &&  \\\n"
      "    defined(BBBBBBBB)\n"
      "#endif",
      getLLVMStyleWithColumns(65));
}

TEST_F(FormatTest, MergeHandlingInTheFaceOfPreprocessorDirectives) {
  FormatStyle AllowsMergedIf = getGoogleStyle();
  AllowsMergedIf.AllowShortIfStatementsOnASingleLine = true;
  verifyFormat("void f() { f(); }\n#error E", AllowsMergedIf);
  verifyFormat("if (true) return 42;\n#error E", AllowsMergedIf);
  verifyFormat("if (true)\n#error E\n  return 42;", AllowsMergedIf);
  EXPECT_EQ("if (true) return 42;",
            format("if (true)\nreturn 42;", AllowsMergedIf));
  FormatStyle ShortMergedIf = AllowsMergedIf;
  ShortMergedIf.ColumnLimit = 25;
  verifyFormat("#define A \\\n"
               "  if (true) return 42;",
               ShortMergedIf);
  verifyFormat("#define A \\\n"
               "  f();    \\\n"
               "  if (true)\n"
               "#define B",
               ShortMergedIf);
  verifyFormat("#define A \\\n"
               "  f();    \\\n"
               "  if (true)\n"
               "g();",
               ShortMergedIf);
  verifyFormat("{\n"
               "#ifdef A\n"
               "  // Comment\n"
               "  if (true) continue;\n"
               "#endif\n"
               "  // Comment\n"
               "  if (true) continue;\n"
               "}",
               ShortMergedIf);
  ShortMergedIf.ColumnLimit = 29;
  verifyFormat("#define A                   \\\n"
               "  if (aaaaaaaaaa) return 1; \\\n"
               "  return 2;",
               ShortMergedIf);
  ShortMergedIf.ColumnLimit = 28;
  verifyFormat("#define A         \\\n"
               "  if (aaaaaaaaaa) \\\n"
               "    return 1;     \\\n"
               "  return 2;",
               ShortMergedIf);
}

TEST_F(FormatTest, BlockCommentsInControlLoops) {
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

TEST_F(FormatTest, BlockComments) {
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
            "*\n"
            " * aaaaaa\n"
            " * aaaaaa\n"
            "*/",
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

TEST_F(FormatTest, BlockCommentsInMacros) {
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

TEST_F(FormatTest, BlockCommentsAtEndOfLine) {
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

  // FIXME: The formatting is still wrong here.
  EXPECT_EQ("a = {\n"
            "    1111 /*      a\n"
            "            */\n"
            "};",
            format("a = {1111 /*      a */\n"
                   "};",
                   getLLVMStyleWithColumns(15)));
}

TEST_F(FormatTest, IndentLineCommentsInStartOfBlockAtEndOfFile) {
  // FIXME: This is not what we want...
  verifyFormat("{\n"
               "// a"
               "// b");
}

TEST_F(FormatTest, FormatStarDependingOnContext) {
  verifyFormat("void f(int *a);");
  verifyFormat("void f() { f(fint * b); }");
  verifyFormat("class A {\n  void f(int *a);\n};");
  verifyFormat("class A {\n  int *a;\n};");
  verifyFormat("namespace a {\n"
               "namespace b {\n"
               "class A {\n"
               "  void f() {}\n"
               "  int *a;\n"
               "};\n"
               "}\n"
               "}");
}

TEST_F(FormatTest, SpecialTokensAtEndOfLine) {
  verifyFormat("while");
  verifyFormat("operator");
}

//===----------------------------------------------------------------------===//
// Objective-C tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, FormatForObjectiveCMethodDecls) {
  verifyFormat("- (void)sendAction:(SEL)aSelector to:(BOOL)anObject;");
  EXPECT_EQ("- (NSUInteger)indexOfObject:(id)anObject;",
            format("-(NSUInteger)indexOfObject:(id)anObject;"));
  EXPECT_EQ("- (NSInteger)Mthod1;", format("-(NSInteger)Mthod1;"));
  EXPECT_EQ("+ (id)Mthod2;", format("+(id)Mthod2;"));
  EXPECT_EQ("- (NSInteger)Method3:(id)anObject;",
            format("-(NSInteger)Method3:(id)anObject;"));
  EXPECT_EQ("- (NSInteger)Method4:(id)anObject;",
            format("-(NSInteger)Method4:(id)anObject;"));
  EXPECT_EQ("- (NSInteger)Method5:(id)anObject:(id)AnotherObject;",
            format("-(NSInteger)Method5:(id)anObject:(id)AnotherObject;"));
  EXPECT_EQ("- (id)Method6:(id)A:(id)B:(id)C:(id)D;",
            format("- (id)Method6:(id)A:(id)B:(id)C:(id)D;"));
  EXPECT_EQ("- (void)sendAction:(SEL)aSelector to:(id)anObject "
            "forAllCells:(BOOL)flag;",
            format("- (void)sendAction:(SEL)aSelector to:(id)anObject "
                   "forAllCells:(BOOL)flag;"));

  // Very long objectiveC method declaration.
  verifyFormat("- (void)aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:\n"
               "    (SoooooooooooooooooooooomeType *)bbbbbbbbbb;");
  verifyFormat("- (NSUInteger)indexOfObject:(id)anObject\n"
               "                    inRange:(NSRange)range\n"
               "                   outRange:(NSRange)out_range\n"
               "                  outRange1:(NSRange)out_range1\n"
               "                  outRange2:(NSRange)out_range2\n"
               "                  outRange3:(NSRange)out_range3\n"
               "                  outRange4:(NSRange)out_range4\n"
               "                  outRange5:(NSRange)out_range5\n"
               "                  outRange6:(NSRange)out_range6\n"
               "                  outRange7:(NSRange)out_range7\n"
               "                  outRange8:(NSRange)out_range8\n"
               "                  outRange9:(NSRange)out_range9;");

  // When the function name has to be wrapped.
  FormatStyle Style = getLLVMStyle();
  Style.IndentWrappedFunctionNames = false;
  verifyFormat("- (SomeLooooooooooooooooooooongType *)\n"
               "veryLooooooooooongName:(NSString)aaaaaaaaaaaaaa\n"
               "           anotherName:(NSString)bbbbbbbbbbbbbb {\n"
               "}",
               Style);
  Style.IndentWrappedFunctionNames = true;
  verifyFormat("- (SomeLooooooooooooooooooooongType *)\n"
               "    veryLooooooooooongName:(NSString)aaaaaaaaaaaaaa\n"
               "               anotherName:(NSString)bbbbbbbbbbbbbb {\n"
               "}",
               Style);

  verifyFormat("- (int)sum:(vector<int>)numbers;");
  verifyGoogleFormat("- (void)setDelegate:(id<Protocol>)delegate;");
  // FIXME: In LLVM style, there should be a space in front of a '<' for ObjC
  // protocol lists (but not for template classes):
  // verifyFormat("- (void)setDelegate:(id <Protocol>)delegate;");

  verifyFormat("- (int (*)())foo:(int (*)())f;");
  verifyGoogleFormat("- (int (*)())foo:(int (*)())foo;");

  // If there's no return type (very rare in practice!), LLVM and Google style
  // agree.
  verifyFormat("- foo;");
  verifyFormat("- foo:(int)f;");
  verifyGoogleFormat("- foo:(int)foo;");
}

TEST_F(FormatTest, FormatObjCInterface) {
  verifyFormat("@interface Foo : NSObject <NSSomeDelegate> {\n"
               "@public\n"
               "  int field1;\n"
               "@protected\n"
               "  int field2;\n"
               "@private\n"
               "  int field3;\n"
               "@package\n"
               "  int field4;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyGoogleFormat("@interface Foo : NSObject<NSSomeDelegate> {\n"
                     " @public\n"
                     "  int field1;\n"
                     " @protected\n"
                     "  int field2;\n"
                     " @private\n"
                     "  int field3;\n"
                     " @package\n"
                     "  int field4;\n"
                     "}\n"
                     "+ (id)init;\n"
                     "@end");

  verifyFormat("@interface /* wait for it */ Foo\n"
               "+ (id)init;\n"
               "// Look, a comment!\n"
               "- (int)answerWith:(int)i;\n"
               "@end");

  verifyFormat("@interface Foo\n"
               "@end\n"
               "@interface Bar\n"
               "@end");

  verifyFormat("@interface Foo : Bar\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : /**/ Bar /**/ <Baz, /**/ Quux>\n"
               "+ (id)init;\n"
               "@end");

  verifyGoogleFormat("@interface Foo : Bar<Baz, Quux>\n"
                     "+ (id)init;\n"
                     "@end");

  verifyFormat("@interface Foo (HackStuff)\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo ()\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) <MyProtocol>\n"
               "+ (id)init;\n"
               "@end");

  verifyGoogleFormat("@interface Foo (HackStuff)<MyProtocol>\n"
                     "+ (id)init;\n"
                     "@end");

  verifyFormat("@interface Foo {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : Bar {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : Bar <Baz, Quux> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo () {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) <MyProtocol> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  FormatStyle OnePerLine = getGoogleStyle();
  OnePerLine.BinPackParameters = false;
  verifyFormat("@interface aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ()<\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa> {\n"
               "}",
               OnePerLine);
}

TEST_F(FormatTest, FormatObjCImplementation) {
  verifyFormat("@implementation Foo : NSObject {\n"
               "@public\n"
               "  int field1;\n"
               "@protected\n"
               "  int field2;\n"
               "@private\n"
               "  int field3;\n"
               "@package\n"
               "  int field4;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyGoogleFormat("@implementation Foo : NSObject {\n"
                     " @public\n"
                     "  int field1;\n"
                     " @protected\n"
                     "  int field2;\n"
                     " @private\n"
                     "  int field3;\n"
                     " @package\n"
                     "  int field4;\n"
                     "}\n"
                     "+ (id)init {\n}\n"
                     "@end");

  verifyFormat("@implementation Foo\n"
               "+ (id)init {\n"
               "  if (true)\n"
               "    return nil;\n"
               "}\n"
               "// Look, a comment!\n"
               "- (int)answerWith:(int)i {\n"
               "  return i;\n"
               "}\n"
               "+ (int)answerWith:(int)i {\n"
               "  return i;\n"
               "}\n"
               "@end");

  verifyFormat("@implementation Foo\n"
               "@end\n"
               "@implementation Bar\n"
               "@end");

  EXPECT_EQ("@implementation Foo : Bar\n"
            "+ (id)init {\n}\n"
            "- (void)foo {\n}\n"
            "@end",
            format("@implementation Foo : Bar\n"
                   "+(id)init{}\n"
                   "-(void)foo{}\n"
                   "@end"));

  verifyFormat("@implementation Foo {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyFormat("@implementation Foo : Bar {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyFormat("@implementation Foo (HackStuff)\n"
               "+ (id)init {\n}\n"
               "@end");
  verifyFormat("@implementation ObjcClass\n"
               "- (void)method;\n"
               "{}\n"
               "@end");
}

TEST_F(FormatTest, FormatObjCProtocol) {
  verifyFormat("@protocol Foo\n"
               "@property(weak) id delegate;\n"
               "- (NSUInteger)numberOfThings;\n"
               "@end");

  verifyFormat("@protocol MyProtocol <NSObject>\n"
               "- (NSUInteger)numberOfThings;\n"
               "@end");

  verifyGoogleFormat("@protocol MyProtocol<NSObject>\n"
                     "- (NSUInteger)numberOfThings;\n"
                     "@end");

  verifyFormat("@protocol Foo;\n"
               "@protocol Bar;\n");

  verifyFormat("@protocol Foo\n"
               "@end\n"
               "@protocol Bar\n"
               "@end");

  verifyFormat("@protocol myProtocol\n"
               "- (void)mandatoryWithInt:(int)i;\n"
               "@optional\n"
               "- (void)optional;\n"
               "@required\n"
               "- (void)required;\n"
               "@optional\n"
               "@property(assign) int madProp;\n"
               "@end\n");

  verifyFormat("@property(nonatomic, assign, readonly)\n"
               "    int *looooooooooooooooooooooooooooongNumber;\n"
               "@property(nonatomic, assign, readonly)\n"
               "    NSString *looooooooooooooooooooooooooooongName;");

  verifyFormat("@implementation PR18406\n"
               "}\n"
               "@end");
}

TEST_F(FormatTest, FormatObjCMethodDeclarations) {
  verifyFormat("- (void)doSomethingWith:(GTMFoo *)theFoo\n"
               "                   rect:(NSRect)theRect\n"
               "               interval:(float)theInterval {\n"
               "}");
  verifyFormat("- (void)shortf:(GTMFoo *)theFoo\n"
               "      longKeyword:(NSRect)theRect\n"
               "    longerKeyword:(float)theInterval\n"
               "            error:(NSError **)theError {\n"
               "}");
  verifyFormat("- (void)shortf:(GTMFoo *)theFoo\n"
               "          longKeyword:(NSRect)theRect\n"
               "    evenLongerKeyword:(float)theInterval\n"
               "                error:(NSError **)theError {\n"
               "}");
  verifyFormat("- (instancetype)initXxxxxx:(id<x>)x\n"
               "                         y:(id<yyyyyyyyyyyyyyyyyyyy>)y\n"
               "    NS_DESIGNATED_INITIALIZER;",
               getLLVMStyleWithColumns(60));

  // Continuation indent width should win over aligning colons if the function
  // name is long.
  FormatStyle continuationStyle = getGoogleStyle();
  continuationStyle.ColumnLimit = 40;
  continuationStyle.IndentWrappedFunctionNames = true;
  verifyFormat("- (void)shortf:(GTMFoo *)theFoo\n"
               "    dontAlignNamef:(NSRect)theRect {\n"
               "}",
               continuationStyle);

  // Make sure we don't break aligning for short parameter names.
  verifyFormat("- (void)shortf:(GTMFoo *)theFoo\n"
               "       aShortf:(NSRect)theRect {\n"
               "}",
               continuationStyle);
}

TEST_F(FormatTest, FormatObjCMethodExpr) {
  verifyFormat("[foo bar:baz];");
  verifyFormat("return [foo bar:baz];");
  verifyFormat("return (a)[foo bar:baz];");
  verifyFormat("f([foo bar:baz]);");
  verifyFormat("f(2, [foo bar:baz]);");
  verifyFormat("f(2, a ? b : c);");
  verifyFormat("[[self initWithInt:4] bar:[baz quux:arrrr]];");

  // Unary operators.
  verifyFormat("int a = +[foo bar:baz];");
  verifyFormat("int a = -[foo bar:baz];");
  verifyFormat("int a = ![foo bar:baz];");
  verifyFormat("int a = ~[foo bar:baz];");
  verifyFormat("int a = ++[foo bar:baz];");
  verifyFormat("int a = --[foo bar:baz];");
  verifyFormat("int a = sizeof [foo bar:baz];");
  verifyFormat("int a = alignof [foo bar:baz];", getGoogleStyle());
  verifyFormat("int a = &[foo bar:baz];");
  verifyFormat("int a = *[foo bar:baz];");
  // FIXME: Make casts work, without breaking f()[4].
  // verifyFormat("int a = (int)[foo bar:baz];");
  // verifyFormat("return (int)[foo bar:baz];");
  // verifyFormat("(void)[foo bar:baz];");
  verifyFormat("return (MyType *)[self.tableView cellForRowAtIndexPath:cell];");

  // Binary operators.
  verifyFormat("[foo bar:baz], [foo bar:baz];");
  verifyFormat("[foo bar:baz] = [foo bar:baz];");
  verifyFormat("[foo bar:baz] *= [foo bar:baz];");
  verifyFormat("[foo bar:baz] /= [foo bar:baz];");
  verifyFormat("[foo bar:baz] %= [foo bar:baz];");
  verifyFormat("[foo bar:baz] += [foo bar:baz];");
  verifyFormat("[foo bar:baz] -= [foo bar:baz];");
  verifyFormat("[foo bar:baz] <<= [foo bar:baz];");
  verifyFormat("[foo bar:baz] >>= [foo bar:baz];");
  verifyFormat("[foo bar:baz] &= [foo bar:baz];");
  verifyFormat("[foo bar:baz] ^= [foo bar:baz];");
  verifyFormat("[foo bar:baz] |= [foo bar:baz];");
  verifyFormat("[foo bar:baz] ? [foo bar:baz] : [foo bar:baz];");
  verifyFormat("[foo bar:baz] || [foo bar:baz];");
  verifyFormat("[foo bar:baz] && [foo bar:baz];");
  verifyFormat("[foo bar:baz] | [foo bar:baz];");
  verifyFormat("[foo bar:baz] ^ [foo bar:baz];");
  verifyFormat("[foo bar:baz] & [foo bar:baz];");
  verifyFormat("[foo bar:baz] == [foo bar:baz];");
  verifyFormat("[foo bar:baz] != [foo bar:baz];");
  verifyFormat("[foo bar:baz] >= [foo bar:baz];");
  verifyFormat("[foo bar:baz] <= [foo bar:baz];");
  verifyFormat("[foo bar:baz] > [foo bar:baz];");
  verifyFormat("[foo bar:baz] < [foo bar:baz];");
  verifyFormat("[foo bar:baz] >> [foo bar:baz];");
  verifyFormat("[foo bar:baz] << [foo bar:baz];");
  verifyFormat("[foo bar:baz] - [foo bar:baz];");
  verifyFormat("[foo bar:baz] + [foo bar:baz];");
  verifyFormat("[foo bar:baz] * [foo bar:baz];");
  verifyFormat("[foo bar:baz] / [foo bar:baz];");
  verifyFormat("[foo bar:baz] % [foo bar:baz];");
  // Whew!

  verifyFormat("return in[42];");
  verifyFormat("for (auto v : in[1]) {\n}");
  verifyFormat("for (int i = 0; i < in[a]; ++i) {\n}");
  verifyFormat("for (int i = 0; in[a] < i; ++i) {\n}");
  verifyFormat("for (int i = 0; i < n; ++i, ++in[a]) {\n}");
  verifyFormat("for (int i = 0; i < n; ++i, in[a]++) {\n}");
  verifyFormat("for (int i = 0; i < f(in[a]); ++i, in[a]++) {\n}");
  verifyFormat("for (id foo in [self getStuffFor:bla]) {\n"
               "}");
  verifyFormat("[self aaaaa:MACRO(a, b:, c:)];");
  verifyFormat("[self aaaaa:(1 + 2) bbbbb:3];");
  verifyFormat("[self aaaaa:(Type)a bbbbb:3];");

  verifyFormat("[self stuffWithInt:(4 + 2) float:4.5];");
  verifyFormat("[self stuffWithInt:a ? b : c float:4.5];");
  verifyFormat("[self stuffWithInt:a ? [self foo:bar] : c];");
  verifyFormat("[self stuffWithInt:a ? (e ? f : g) : c];");
  verifyFormat("[cond ? obj1 : obj2 methodWithParam:param]");
  verifyFormat("[button setAction:@selector(zoomOut:)];");
  verifyFormat("[color getRed:&r green:&g blue:&b alpha:&a];");

  verifyFormat("arr[[self indexForFoo:a]];");
  verifyFormat("throw [self errorFor:a];");
  verifyFormat("@throw [self errorFor:a];");

  verifyFormat("[(id)foo bar:(id)baz quux:(id)snorf];");
  verifyFormat("[(id)foo bar:(id) ? baz : quux];");
  verifyFormat("4 > 4 ? (id)a : (id)baz;");

  // This tests that the formatter doesn't break after "backing" but before ":",
  // which would be at 80 columns.
  verifyFormat(
      "void f() {\n"
      "  if ((self = [super initWithContentRect:contentRect\n"
      "                               styleMask:styleMask ?: otherMask\n"
      "                                 backing:NSBackingStoreBuffered\n"
      "                                   defer:YES]))");

  verifyFormat(
      "[foo checkThatBreakingAfterColonWorksOk:\n"
      "         [bar ifItDoes:reduceOverallLineLengthLikeInThisCase]];");

  verifyFormat("[myObj short:arg1 // Force line break\n"
               "          longKeyword:arg2 != nil ? arg2 : @\"longKeyword\"\n"
               "    evenLongerKeyword:arg3 ?: @\"evenLongerKeyword\"\n"
               "                error:arg4];");
  verifyFormat(
      "void f() {\n"
      "  popup_window_.reset([[RenderWidgetPopupWindow alloc]\n"
      "      initWithContentRect:NSMakeRect(origin_global.x, origin_global.y,\n"
      "                                     pos.width(), pos.height())\n"
      "                styleMask:NSBorderlessWindowMask\n"
      "                  backing:NSBackingStoreBuffered\n"
      "                    defer:NO]);\n"
      "}");
  verifyFormat(
      "void f() {\n"
      "  popup_wdow_.reset([[RenderWidgetPopupWindow alloc]\n"
      "      iniithContentRect:NSMakRet(origin_global.x, origin_global.y,\n"
      "                                 pos.width(), pos.height())\n"
      "                syeMask:NSBorderlessWindowMask\n"
      "                  bking:NSBackingStoreBuffered\n"
      "                    der:NO]);\n"
      "}",
      getLLVMStyleWithColumns(70));
  verifyFormat(
      "void f() {\n"
      "  popup_window_.reset([[RenderWidgetPopupWindow alloc]\n"
      "      initWithContentRect:NSMakeRect(origin_global.x, origin_global.y,\n"
      "                                     pos.width(), pos.height())\n"
      "                styleMask:NSBorderlessWindowMask\n"
      "                  backing:NSBackingStoreBuffered\n"
      "                    defer:NO]);\n"
      "}",
      getChromiumStyle(FormatStyle::LK_Cpp));
  verifyFormat("[contentsContainer replaceSubview:[subviews objectAtIndex:0]\n"
               "                             with:contentsNativeView];");

  verifyFormat(
      "[pboard addTypes:[NSArray arrayWithObject:kBookmarkButtonDragType]\n"
      "           owner:nillllll];");

  verifyFormat(
      "[pboard setData:[NSData dataWithBytes:&button length:sizeof(button)]\n"
      "        forType:kBookmarkButtonDragType];");

  verifyFormat("[defaultCenter addObserver:self\n"
               "                  selector:@selector(willEnterFullscreen)\n"
               "                      name:kWillEnterFullscreenNotification\n"
               "                    object:nil];");
  verifyFormat("[image_rep drawInRect:drawRect\n"
               "             fromRect:NSZeroRect\n"
               "            operation:NSCompositeCopy\n"
               "             fraction:1.0\n"
               "       respectFlipped:NO\n"
               "                hints:nil];");
  verifyFormat("[aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa];");
  verifyFormat("[aaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa)\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa];");
  verifyFormat("[aaaaaaaaaaaaaaaaaaaaaaa.aaaaaaaa[aaaaaaaaaaaaaaaaaaaaa]\n"
               "    aaaaaaaaaaaaaaaaaaaaaa];");
  verifyFormat("[call aaaaaaaa.aaaaaa.aaaaaaaa.aaaaaaaa.aaaaaaaa\n"
               "        .aaaaaaaa.aaaaaaaa];", // FIXME: Indentation seems off.
               getLLVMStyleWithColumns(60));

  verifyFormat(
      "scoped_nsobject<NSTextField> message(\n"
      "    // The frame will be fixed up when |-setMessageText:| is called.\n"
      "    [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 0, 0)]);");
  verifyFormat("[self aaaaaa:bbbbbbbbbbbbb\n"
               "    aaaaaaaaaa:bbbbbbbbbbbbbbbbb\n"
               "         aaaaa:bbbbbbbbbbb + bbbbbbbbbbbb\n"
               "          aaaa:bbb];");
  verifyFormat("[self param:function( //\n"
               "                parameter)]");
  verifyFormat(
      "[self aaaaaaaaaa:aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa |\n"
      "                 aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa |\n"
      "                 aaaaaaaaaaaaaaa | aaaaaaaaaaaaaaa];");

  // FIXME: This violates the column limit.
  verifyFormat(
      "[aaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    aaaaaaaaaaaaaaaaa:aaaaaaaa\n"
      "                  aaa:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa];",
      getLLVMStyleWithColumns(60));

  // Variadic parameters.
  verifyFormat(
      "NSArray *myStrings = [NSArray stringarray:@\"a\", @\"b\", nil];");
  verifyFormat(
      "[self aaaaaaaaaaaaa:aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaa, aaaaaaaaaaaaaaa];");
  verifyFormat("[self // break\n"
               "      a:a\n"
               "    aaa:aaa];");
  verifyFormat("bool a = ([aaaaaaaa aaaaa] == aaaaaaaaaaaaaaaaa ||\n"
               "          [aaaaaaaa aaaaa] == aaaaaaaaaaaaaaaaaaaa);");
}

TEST_F(FormatTest, ObjCAt) {
  verifyFormat("@autoreleasepool");
  verifyFormat("@catch");
  verifyFormat("@class");
  verifyFormat("@compatibility_alias");
  verifyFormat("@defs");
  verifyFormat("@dynamic");
  verifyFormat("@encode");
  verifyFormat("@end");
  verifyFormat("@finally");
  verifyFormat("@implementation");
  verifyFormat("@import");
  verifyFormat("@interface");
  verifyFormat("@optional");
  verifyFormat("@package");
  verifyFormat("@private");
  verifyFormat("@property");
  verifyFormat("@protected");
  verifyFormat("@protocol");
  verifyFormat("@public");
  verifyFormat("@required");
  verifyFormat("@selector");
  verifyFormat("@synchronized");
  verifyFormat("@synthesize");
  verifyFormat("@throw");
  verifyFormat("@try");

  EXPECT_EQ("@interface", format("@ interface"));

  // The precise formatting of this doesn't matter, nobody writes code like
  // this.
  verifyFormat("@ /*foo*/ interface");
}

TEST_F(FormatTest, ObjCSnippets) {
  verifyFormat("@autoreleasepool {\n"
               "  foo();\n"
               "}");
  verifyFormat("@class Foo, Bar;");
  verifyFormat("@compatibility_alias AliasName ExistingClass;");
  verifyFormat("@dynamic textColor;");
  verifyFormat("char *buf1 = @encode(int *);");
  verifyFormat("char *buf1 = @encode(typeof(4 * 5));");
  verifyFormat("char *buf1 = @encode(int **);");
  verifyFormat("Protocol *proto = @protocol(p1);");
  verifyFormat("SEL s = @selector(foo:);");
  verifyFormat("@synchronized(self) {\n"
               "  f();\n"
               "}");

  verifyFormat("@synthesize dropArrowPosition = dropArrowPosition_;");
  verifyGoogleFormat("@synthesize dropArrowPosition = dropArrowPosition_;");

  verifyFormat("@property(assign, nonatomic) CGFloat hoverAlpha;");
  verifyFormat("@property(assign, getter=isEditable) BOOL editable;");
  verifyGoogleFormat("@property(assign, getter=isEditable) BOOL editable;");
  verifyFormat("@property (assign, getter=isEditable) BOOL editable;",
               getMozillaStyle());
  verifyFormat("@property BOOL editable;", getMozillaStyle());
  verifyFormat("@property (assign, getter=isEditable) BOOL editable;",
               getWebKitStyle());
  verifyFormat("@property BOOL editable;", getWebKitStyle());

  verifyFormat("@import foo.bar;\n"
               "@import baz;");
}

TEST_F(FormatTest, ObjCForIn) {
  verifyFormat("- (void)test {\n"
               "  for (NSString *n in arrayOfStrings) {\n"
               "    foo(n);\n"
               "  }\n"
               "}");
  verifyFormat("- (void)test {\n"
               "  for (NSString *n in (__bridge NSArray *)arrayOfStrings) {\n"
               "    foo(n);\n"
               "  }\n"
               "}");
}

TEST_F(FormatTest, ObjCLiterals) {
  verifyFormat("@\"String\"");
  verifyFormat("@1");
  verifyFormat("@+4.8");
  verifyFormat("@-4");
  verifyFormat("@1LL");
  verifyFormat("@.5");
  verifyFormat("@'c'");
  verifyFormat("@true");

  verifyFormat("NSNumber *smallestInt = @(-INT_MAX - 1);");
  verifyFormat("NSNumber *piOverTwo = @(M_PI / 2);");
  verifyFormat("NSNumber *favoriteColor = @(Green);");
  verifyFormat("NSString *path = @(getenv(\"PATH\"));");

  verifyFormat("[dictionary setObject:@(1) forKey:@\"number\"];");
}

TEST_F(FormatTest, ObjCDictLiterals) {
  verifyFormat("@{");
  verifyFormat("@{}");
  verifyFormat("@{@\"one\" : @1}");
  verifyFormat("return @{@\"one\" : @1;");
  verifyFormat("@{@\"one\" : @1}");

  verifyFormat("@{@\"one\" : @{@2 : @1}}");
  verifyFormat("@{\n"
               "  @\"one\" : @{@2 : @1},\n"
               "}");

  verifyFormat("@{1 > 2 ? @\"one\" : @\"two\" : 1 > 2 ? @1 : @2}");
  verifyIncompleteFormat("[self setDict:@{}");
  verifyIncompleteFormat("[self setDict:@{@1 : @2}");
  verifyFormat("NSLog(@\"%@\", @{@1 : @2, @2 : @3}[@1]);");
  verifyFormat(
      "NSDictionary *masses = @{@\"H\" : @1.0078, @\"He\" : @4.0026};");
  verifyFormat(
      "NSDictionary *settings = @{AVEncoderKey : @(AVAudioQualityMax)};");

  verifyFormat("NSDictionary *d = @{\n"
               "  @\"nam\" : NSUserNam(),\n"
               "  @\"dte\" : [NSDate date],\n"
               "  @\"processInfo\" : [NSProcessInfo processInfo]\n"
               "};");
  verifyFormat(
      "@{\n"
      "  NSFontAttributeNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee : "
      "regularFont,\n"
      "};");
  verifyGoogleFormat(
      "@{\n"
      "  NSFontAttributeNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee : "
      "regularFont,\n"
      "};");
  verifyFormat(
      "@{\n"
      "  NSFontAttributeNameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee :\n"
      "      reeeeeeeeeeeeeeeeeeeeeeeegularFont,\n"
      "};");

  // We should try to be robust in case someone forgets the "@".
  verifyFormat("NSDictionary *d = {\n"
               "  @\"nam\" : NSUserNam(),\n"
               "  @\"dte\" : [NSDate date],\n"
               "  @\"processInfo\" : [NSProcessInfo processInfo]\n"
               "};");
  verifyFormat("NSMutableDictionary *dictionary =\n"
               "    [NSMutableDictionary dictionaryWithDictionary:@{\n"
               "      aaaaaaaaaaaaaaaaaaaaa : aaaaaaaaaaaaa,\n"
               "      bbbbbbbbbbbbbbbbbb : bbbbb,\n"
               "      cccccccccccccccc : ccccccccccccccc\n"
               "    }];");

  // Ensure that casts before the key are kept on the same line as the key.
  verifyFormat(
      "NSDictionary *d = @{\n"
      "  (aaaaaaaa id)aaaaaaaaa : (aaaaaaaa id)aaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "  (aaaaaaaa id)aaaaaaaaaaaaaa : (aaaaaaaa id)aaaaaaaaaaaaaa,\n"
      "};");
}

TEST_F(FormatTest, ObjCArrayLiterals) {
  verifyIncompleteFormat("@[");
  verifyFormat("@[]");
  verifyFormat(
      "NSArray *array = @[ @\" Hey \", NSApp, [NSNumber numberWithInt:42] ];");
  verifyFormat("return @[ @3, @[], @[ @4, @5 ] ];");
  verifyFormat("NSArray *array = @[ [foo description] ];");

  verifyFormat(
      "NSArray *some_variable = @[\n"
      "  aaaa == bbbbbbbbbbb ? @\"aaaaaaaaaaaa\" : @\"aaaaaaaaaaaaaa\",\n"
      "  @\"aaaaaaaaaaaaaaaaa\",\n"
      "  @\"aaaaaaaaaaaaaaaaa\",\n"
      "  @\"aaaaaaaaaaaaaaaaa\"\n"
      "];");
  verifyFormat("NSArray *some_variable = @[\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "];");
  verifyGoogleFormat("NSArray *some_variable = @[\n"
                     "  @\"aaaaaaaaaaaaaaaaa\",\n"
                     "  @\"aaaaaaaaaaaaaaaaa\",\n"
                     "  @\"aaaaaaaaaaaaaaaaa\",\n"
                     "  @\"aaaaaaaaaaaaaaaaa\"\n"
                     "];");
  verifyFormat("NSArray *array = @[\n"
               "  @\"a\",\n"
               "  @\"a\",\n" // Trailing comma -> one per line.
               "];");

  // We should try to be robust in case someone forgets the "@".
  verifyFormat("NSArray *some_variable = [\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "  @\"aaaaaaaaaaaaaaaaa\",\n"
               "];");
  verifyFormat(
      "- (NSAttributedString *)attributedStringForSegment:(NSUInteger)segment\n"
      "                                             index:(NSUInteger)index\n"
      "                                nonDigitAttributes:\n"
      "                                    (NSDictionary *)noDigitAttributes;");
  verifyFormat("[someFunction someLooooooooooooongParameter:@[\n"
               "  NSBundle.mainBundle.infoDictionary[@\"a\"]\n"
               "]];");
}

TEST_F(FormatTest, BreaksStringLiterals) {
  EXPECT_EQ("\"some text \"\n"
            "\"other\";",
            format("\"some text other\";", getLLVMStyleWithColumns(12)));
  EXPECT_EQ("\"some text \"\n"
            "\"other\";",
            format("\\\n\"some text other\";", getLLVMStyleWithColumns(12)));
  EXPECT_EQ(
      "#define A  \\\n"
      "  \"some \"  \\\n"
      "  \"text \"  \\\n"
      "  \"other\";",
      format("#define A \"some text other\";", getLLVMStyleWithColumns(12)));
  EXPECT_EQ(
      "#define A  \\\n"
      "  \"so \"    \\\n"
      "  \"text \"  \\\n"
      "  \"other\";",
      format("#define A \"so text other\";", getLLVMStyleWithColumns(12)));

  EXPECT_EQ("\"some text\"",
            format("\"some text\"", getLLVMStyleWithColumns(1)));
  EXPECT_EQ("\"some text\"",
            format("\"some text\"", getLLVMStyleWithColumns(11)));
  EXPECT_EQ("\"some \"\n"
            "\"text\"",
            format("\"some text\"", getLLVMStyleWithColumns(10)));
  EXPECT_EQ("\"some \"\n"
            "\"text\"",
            format("\"some text\"", getLLVMStyleWithColumns(7)));
  EXPECT_EQ("\"some\"\n"
            "\" tex\"\n"
            "\"t\"",
            format("\"some text\"", getLLVMStyleWithColumns(6)));
  EXPECT_EQ("\"some\"\n"
            "\" tex\"\n"
            "\" and\"",
            format("\"some tex and\"", getLLVMStyleWithColumns(6)));
  EXPECT_EQ("\"some\"\n"
            "\"/tex\"\n"
            "\"/and\"",
            format("\"some/tex/and\"", getLLVMStyleWithColumns(6)));

  EXPECT_EQ("variable =\n"
            "    \"long string \"\n"
            "    \"literal\";",
            format("variable = \"long string literal\";",
                   getLLVMStyleWithColumns(20)));

  EXPECT_EQ("variable = f(\n"
            "    \"long string \"\n"
            "    \"literal\",\n"
            "    short,\n"
            "    loooooooooooooooooooong);",
            format("variable = f(\"long string literal\", short, "
                   "loooooooooooooooooooong);",
                   getLLVMStyleWithColumns(20)));

  EXPECT_EQ(
      "f(g(\"long string \"\n"
      "    \"literal\"),\n"
      "  b);",
      format("f(g(\"long string literal\"), b);", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("f(g(\"long string \"\n"
            "    \"literal\",\n"
            "    a),\n"
            "  b);",
            format("f(g(\"long string literal\", a), b);",
                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ(
      "f(\"one two\".split(\n"
      "    variable));",
      format("f(\"one two\".split(variable));", getLLVMStyleWithColumns(20)));
  EXPECT_EQ("f(\"one two three four five six \"\n"
            "  \"seven\".split(\n"
            "      really_looooong_variable));",
            format("f(\"one two three four five six seven\"."
                   "split(really_looooong_variable));",
                   getLLVMStyleWithColumns(33)));

  EXPECT_EQ("f(\"some \"\n"
            "  \"text\",\n"
            "  other);",
            format("f(\"some text\", other);", getLLVMStyleWithColumns(10)));

  // Only break as a last resort.
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaa,\n"
      "    aaaaaa(\"aaa aaaaa aaa aaa aaaaa aaa aaaaa aaa aaa aaaaaa\"));");

  EXPECT_EQ("\"splitmea\"\n"
            "\"trandomp\"\n"
            "\"oint\"",
            format("\"splitmeatrandompoint\"", getLLVMStyleWithColumns(10)));

  EXPECT_EQ("\"split/\"\n"
            "\"pathat/\"\n"
            "\"slashes\"",
            format("\"split/pathat/slashes\"", getLLVMStyleWithColumns(10)));

  EXPECT_EQ("\"split/\"\n"
            "\"pathat/\"\n"
            "\"slashes\"",
            format("\"split/pathat/slashes\"", getLLVMStyleWithColumns(10)));
  EXPECT_EQ("\"split at \"\n"
            "\"spaces/at/\"\n"
            "\"slashes.at.any$\"\n"
            "\"non-alphanumeric%\"\n"
            "\"1111111111characte\"\n"
            "\"rs\"",
            format("\"split at "
                   "spaces/at/"
                   "slashes.at."
                   "any$non-"
                   "alphanumeric%"
                   "1111111111characte"
                   "rs\"",
                   getLLVMStyleWithColumns(20)));

  // Verify that splitting the strings understands
  // Style::AlwaysBreakBeforeMultilineStrings.
  EXPECT_EQ(
      "aaaaaaaaaaaa(\n"
      "    \"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaa \"\n"
      "    \"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaa\");",
      format("aaaaaaaaaaaa(\"aaaaaaaaaaaaaaaaaaaaaaaaaa "
             "aaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaa "
             "aaaaaaaaaaaaaaaaaaaaaa\");",
             getGoogleStyle()));
  EXPECT_EQ("return \"aaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaa \"\n"
            "       \"aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaa\";",
            format("return \"aaaaaaaaaaaaaaaaaaaaaa "
                   "aaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaa "
                   "aaaaaaaaaaaaaaaaaaaaaa\";",
                   getGoogleStyle()));
  EXPECT_EQ("llvm::outs() << \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \"\n"
            "                \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";",
            format("llvm::outs() << "
                   "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaa\";"));
  EXPECT_EQ("ffff(\n"
            "    {\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \"\n"
            "     \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"});",
            format("ffff({\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"});",
                   getGoogleStyle()));

  FormatStyle AlignLeft = getLLVMStyleWithColumns(12);
  AlignLeft.AlignEscapedNewlinesLeft = true;
  EXPECT_EQ("#define A \\\n"
            "  \"some \" \\\n"
            "  \"text \" \\\n"
            "  \"other\";",
            format("#define A \"some text other\";", AlignLeft));
}

TEST_F(FormatTest, FullyRemoveEmptyLines) {
  FormatStyle NoEmptyLines = getLLVMStyleWithColumns(80);
  NoEmptyLines.MaxEmptyLinesToKeep = 0;
  EXPECT_EQ("int i = a(b());",
            format("int i=a(\n\n b(\n\n\n )\n\n);", NoEmptyLines));
}

TEST_F(FormatTest, BreaksStringLiteralsWithTabs) {
  EXPECT_EQ(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "(\n"
      "    \"x\t\");",
      format("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
             "aaaaaaa("
             "\"x\t\");"));
}

TEST_F(FormatTest, BreaksWideAndNSStringLiterals) {
  EXPECT_EQ(
      "u8\"utf8 string \"\n"
      "u8\"literal\";",
      format("u8\"utf8 string literal\";", getGoogleStyleWithColumns(16)));
  EXPECT_EQ(
      "u\"utf16 string \"\n"
      "u\"literal\";",
      format("u\"utf16 string literal\";", getGoogleStyleWithColumns(16)));
  EXPECT_EQ(
      "U\"utf32 string \"\n"
      "U\"literal\";",
      format("U\"utf32 string literal\";", getGoogleStyleWithColumns(16)));
  EXPECT_EQ("L\"wide string \"\n"
            "L\"literal\";",
            format("L\"wide string literal\";", getGoogleStyleWithColumns(16)));
  EXPECT_EQ("@\"NSString \"\n"
            "@\"literal\";",
            format("@\"NSString literal\";", getGoogleStyleWithColumns(19)));

  // This input makes clang-format try to split the incomplete unicode escape
  // sequence, which used to lead to a crasher.
  verifyNoCrash(
      "aaaaaaaaaaaaaaaaaaaa = L\"\\udff\"'; // aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      getLLVMStyleWithColumns(60));
}

TEST_F(FormatTest, DoesNotBreakRawStringLiterals) {
  FormatStyle Style = getGoogleStyleWithColumns(15);
  EXPECT_EQ("R\"x(raw literal)x\";", format("R\"x(raw literal)x\";", Style));
  EXPECT_EQ("uR\"x(raw literal)x\";", format("uR\"x(raw literal)x\";", Style));
  EXPECT_EQ("LR\"x(raw literal)x\";", format("LR\"x(raw literal)x\";", Style));
  EXPECT_EQ("UR\"x(raw literal)x\";", format("UR\"x(raw literal)x\";", Style));
  EXPECT_EQ("u8R\"x(raw literal)x\";",
            format("u8R\"x(raw literal)x\";", Style));
}

TEST_F(FormatTest, BreaksStringLiteralsWithin_TMacro) {
  FormatStyle Style = getLLVMStyleWithColumns(20);
  EXPECT_EQ(
      "_T(\"aaaaaaaaaaaaaa\")\n"
      "_T(\"aaaaaaaaaaaaaa\")\n"
      "_T(\"aaaaaaaaaaaa\")",
      format("  _T(\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\")", Style));
  EXPECT_EQ("f(x, _T(\"aaaaaaaaa\")\n"
            "     _T(\"aaaaaa\"),\n"
            "  z);",
            format("f(x, _T(\"aaaaaaaaaaaaaaa\"), z);", Style));

  // FIXME: Handle embedded spaces in one iteration.
  //  EXPECT_EQ("_T(\"aaaaaaaaaaaaa\")\n"
  //            "_T(\"aaaaaaaaaaaaa\")\n"
  //            "_T(\"aaaaaaaaaaaaa\")\n"
  //            "_T(\"a\")",
  //            format("  _T ( \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" )",
  //                   getLLVMStyleWithColumns(20)));
  EXPECT_EQ(
      "_T ( \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" )",
      format("  _T ( \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" )", Style));
  EXPECT_EQ("f(\n"
            "#if !TEST\n"
            "    _T(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXn\")\n"
            "#endif\n"
            "    );",
            format("f(\n"
                   "#if !TEST\n"
                   "_T(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXn\")\n"
                   "#endif\n"
                   ");"));
  EXPECT_EQ("f(\n"
            "\n"
            "    _T(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXn\"));",
            format("f(\n"
                   "\n"
                   "_T(\"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXn\"));"));
}

TEST_F(FormatTest, DontSplitStringLiteralsWithEscapedNewlines) {
  EXPECT_EQ(
      "aaaaaaaaaaa = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
      "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
      "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";",
      format("aaaaaaaaaaa  =  \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
             "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\n"
             "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\";"));
}

TEST_F(FormatTest, CountsCharactersInMultilineRawStringLiterals) {
  EXPECT_EQ("f(g(R\"x(raw literal)x\", a), b);",
            format("f(g(R\"x(raw literal)x\",   a), b);", getGoogleStyle()));
  EXPECT_EQ("fffffffffff(g(R\"x(\n"
            "multiline raw string literal xxxxxxxxxxxxxx\n"
            ")x\",\n"
            "              a),\n"
            "            b);",
            format("fffffffffff(g(R\"x(\n"
                   "multiline raw string literal xxxxxxxxxxxxxx\n"
                   ")x\", a), b);",
                   getGoogleStyleWithColumns(20)));
  EXPECT_EQ("fffffffffff(\n"
            "    g(R\"x(qqq\n"
            "multiline raw string literal xxxxxxxxxxxxxx\n"
            ")x\",\n"
            "      a),\n"
            "    b);",
            format("fffffffffff(g(R\"x(qqq\n"
                   "multiline raw string literal xxxxxxxxxxxxxx\n"
                   ")x\", a), b);",
                   getGoogleStyleWithColumns(20)));

  EXPECT_EQ("fffffffffff(R\"x(\n"
            "multiline raw string literal xxxxxxxxxxxxxx\n"
            ")x\");",
            format("fffffffffff(R\"x(\n"
                   "multiline raw string literal xxxxxxxxxxxxxx\n"
                   ")x\");",
                   getGoogleStyleWithColumns(20)));
  EXPECT_EQ("fffffffffff(R\"x(\n"
            "multiline raw string literal xxxxxxxxxxxxxx\n"
            ")x\" + bbbbbb);",
            format("fffffffffff(R\"x(\n"
                   "multiline raw string literal xxxxxxxxxxxxxx\n"
                   ")x\" +   bbbbbb);",
                   getGoogleStyleWithColumns(20)));
  EXPECT_EQ("fffffffffff(\n"
            "    R\"x(\n"
            "multiline raw string literal xxxxxxxxxxxxxx\n"
            ")x\" +\n"
            "    bbbbbb);",
            format("fffffffffff(\n"
                   " R\"x(\n"
                   "multiline raw string literal xxxxxxxxxxxxxx\n"
                   ")x\" + bbbbbb);",
                   getGoogleStyleWithColumns(20)));
}

TEST_F(FormatTest, SkipsUnknownStringLiterals) {
  verifyFormat("string a = \"unterminated;");
  EXPECT_EQ("function(\"unterminated,\n"
            "         OtherParameter);",
            format("function(  \"unterminated,\n"
                   "    OtherParameter);"));
}

TEST_F(FormatTest, DoesNotTryToParseUDLiteralsInPreCpp11Code) {
  FormatStyle Style = getLLVMStyle();
  Style.Standard = FormatStyle::LS_Cpp03;
  EXPECT_EQ("#define x(_a) printf(\"foo\" _a);",
            format("#define x(_a) printf(\"foo\"_a);", Style));
}

TEST_F(FormatTest, UnderstandsCpp1y) { verifyFormat("int bi{1'000'000};"); }

TEST_F(FormatTest, BreakStringLiteralsBeforeUnbreakableTokenSequence) {
  EXPECT_EQ("someFunction(\"aaabbbcccd\"\n"
            "             \"ddeeefff\");",
            format("someFunction(\"aaabbbcccdddeeefff\");",
                   getLLVMStyleWithColumns(25)));
  EXPECT_EQ("someFunction1234567890(\n"
            "    \"aaabbbcccdddeeefff\");",
            format("someFunction1234567890(\"aaabbbcccdddeeefff\");",
                   getLLVMStyleWithColumns(26)));
  EXPECT_EQ("someFunction1234567890(\n"
            "    \"aaabbbcccdddeeeff\"\n"
            "    \"f\");",
            format("someFunction1234567890(\"aaabbbcccdddeeefff\");",
                   getLLVMStyleWithColumns(25)));
  EXPECT_EQ("someFunction1234567890(\n"
            "    \"aaabbbcccdddeeeff\"\n"
            "    \"f\");",
            format("someFunction1234567890(\"aaabbbcccdddeeefff\");",
                   getLLVMStyleWithColumns(24)));
  EXPECT_EQ("someFunction(\"aaabbbcc \"\n"
            "             \"ddde \"\n"
            "             \"efff\");",
            format("someFunction(\"aaabbbcc ddde efff\");",
                   getLLVMStyleWithColumns(25)));
  EXPECT_EQ("someFunction(\"aaabbbccc \"\n"
            "             \"ddeeefff\");",
            format("someFunction(\"aaabbbccc ddeeefff\");",
                   getLLVMStyleWithColumns(25)));
  EXPECT_EQ("someFunction1234567890(\n"
            "    \"aaabb \"\n"
            "    \"cccdddeeefff\");",
            format("someFunction1234567890(\"aaabb cccdddeeefff\");",
                   getLLVMStyleWithColumns(25)));
  EXPECT_EQ("#define A          \\\n"
            "  string s =       \\\n"
            "      \"123456789\"  \\\n"
            "      \"0\";         \\\n"
            "  int i;",
            format("#define A string s = \"1234567890\"; int i;",
                   getLLVMStyleWithColumns(20)));
  // FIXME: Put additional penalties on breaking at non-whitespace locations.
  EXPECT_EQ("someFunction(\"aaabbbcc \"\n"
            "             \"dddeeeff\"\n"
            "             \"f\");",
            format("someFunction(\"aaabbbcc dddeeefff\");",
                   getLLVMStyleWithColumns(25)));
}

TEST_F(FormatTest, DoNotBreakStringLiteralsInEscapeSequence) {
  EXPECT_EQ("\"\\a\"", format("\"\\a\"", getLLVMStyleWithColumns(3)));
  EXPECT_EQ("\"\\\"", format("\"\\\"", getLLVMStyleWithColumns(2)));
  EXPECT_EQ("\"test\"\n"
            "\"\\n\"",
            format("\"test\\n\"", getLLVMStyleWithColumns(7)));
  EXPECT_EQ("\"tes\\\\\"\n"
            "\"n\"",
            format("\"tes\\\\n\"", getLLVMStyleWithColumns(7)));
  EXPECT_EQ("\"\\\\\\\\\"\n"
            "\"\\n\"",
            format("\"\\\\\\\\\\n\"", getLLVMStyleWithColumns(7)));
  EXPECT_EQ("\"\\uff01\"", format("\"\\uff01\"", getLLVMStyleWithColumns(7)));
  EXPECT_EQ("\"\\uff01\"\n"
            "\"test\"",
            format("\"\\uff01test\"", getLLVMStyleWithColumns(8)));
  EXPECT_EQ("\"\\Uff01ff02\"",
            format("\"\\Uff01ff02\"", getLLVMStyleWithColumns(11)));
  EXPECT_EQ("\"\\x000000000001\"\n"
            "\"next\"",
            format("\"\\x000000000001next\"", getLLVMStyleWithColumns(16)));
  EXPECT_EQ("\"\\x000000000001next\"",
            format("\"\\x000000000001next\"", getLLVMStyleWithColumns(15)));
  EXPECT_EQ("\"\\x000000000001\"",
            format("\"\\x000000000001\"", getLLVMStyleWithColumns(7)));
  EXPECT_EQ("\"test\"\n"
            "\"\\000000\"\n"
            "\"000001\"",
            format("\"test\\000000000001\"", getLLVMStyleWithColumns(9)));
  EXPECT_EQ("\"test\\000\"\n"
            "\"00000000\"\n"
            "\"1\"",
            format("\"test\\000000000001\"", getLLVMStyleWithColumns(10)));
}

TEST_F(FormatTest, DoNotCreateUnreasonableUnwrappedLines) {
  verifyFormat("void f() {\n"
               "  return g() {}\n"
               "  void h() {}");
  verifyFormat("int a[] = {void forgot_closing_brace(){f();\n"
               "g();\n"
               "}");
}

TEST_F(FormatTest, DoNotPrematurelyEndUnwrappedLineForReturnStatements) {
  verifyFormat(
      "void f() { return C{param1, param2}.SomeCall(param1, param2); }");
}

TEST_F(FormatTest, FormatsClosingBracesInEmptyNestedBlocks) {
  verifyFormat("class X {\n"
               "  void f() {\n"
               "  }\n"
               "};",
               getLLVMStyleWithColumns(12));
}

TEST_F(FormatTest, ConfigurableIndentWidth) {
  FormatStyle EightIndent = getLLVMStyleWithColumns(18);
  EightIndent.IndentWidth = 8;
  EightIndent.ContinuationIndentWidth = 8;
  verifyFormat("void f() {\n"
               "        someFunction();\n"
               "        if (true) {\n"
               "                f();\n"
               "        }\n"
               "}",
               EightIndent);
  verifyFormat("class X {\n"
               "        void f() {\n"
               "        }\n"
               "};",
               EightIndent);
  verifyFormat("int x[] = {\n"
               "        call(),\n"
               "        call()};",
               EightIndent);
}

TEST_F(FormatTest, ConfigurableFunctionDeclarationIndentAfterType) {
  verifyFormat("double\n"
               "f();",
               getLLVMStyleWithColumns(8));
}

TEST_F(FormatTest, ConfigurableUseOfTab) {
  FormatStyle Tab = getLLVMStyleWithColumns(42);
  Tab.IndentWidth = 8;
  Tab.UseTab = FormatStyle::UT_Always;
  Tab.AlignEscapedNewlinesLeft = true;

  EXPECT_EQ("if (aaaaaaaa && // q\n"
            "    bb)\t\t// w\n"
            "\t;",
            format("if (aaaaaaaa &&// q\n"
                   "bb)// w\n"
                   ";",
                   Tab));
  EXPECT_EQ("if (aaa && bbb) // w\n"
            "\t;",
            format("if(aaa&&bbb)// w\n"
                   ";",
                   Tab));

  verifyFormat("class X {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t\t     parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  verifyFormat("#define A                        \\\n"
               "\tvoid f() {               \\\n"
               "\t\tsomeFunction(    \\\n"
               "\t\t    parameter1,  \\\n"
               "\t\t    parameter2); \\\n"
               "\t}",
               Tab);

  Tab.TabWidth = 4;
  Tab.IndentWidth = 8;
  verifyFormat("class TabWidth4Indent8 {\n"
               "\t\tvoid f() {\n"
               "\t\t\t\tsomeFunction(parameter1,\n"
               "\t\t\t\t\t\t\t parameter2);\n"
               "\t\t}\n"
               "};",
               Tab);

  Tab.TabWidth = 4;
  Tab.IndentWidth = 4;
  verifyFormat("class TabWidth4Indent4 {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t\t\t\t parameter2);\n"
               "\t}\n"
               "};",
               Tab);

  Tab.TabWidth = 8;
  Tab.IndentWidth = 4;
  verifyFormat("class TabWidth8Indent4 {\n"
               "    void f() {\n"
               "\tsomeFunction(parameter1,\n"
               "\t\t     parameter2);\n"
               "    }\n"
               "};",
               Tab);

  Tab.TabWidth = 8;
  Tab.IndentWidth = 8;
  EXPECT_EQ("/*\n"
            "\t      a\t\tcomment\n"
            "\t      in multiple lines\n"
            "       */",
            format("   /*\t \t \n"
                   " \t \t a\t\tcomment\t \t\n"
                   " \t \t in multiple lines\t\n"
                   " \t  */",
                   Tab));

  Tab.UseTab = FormatStyle::UT_ForIndentation;
  verifyFormat("{\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "\taaaaaaaaaaaaaaaaaaaaaaaaaaaa();\n"
               "};",
               Tab);
  verifyFormat("enum A {\n"
               "\ta1, // Force multiple lines\n"
               "\ta2,\n"
               "\ta3\n"
               "};",
               Tab);
  EXPECT_EQ("if (aaaaaaaa && // q\n"
            "    bb)         // w\n"
            "\t;",
            format("if (aaaaaaaa &&// q\n"
                   "bb)// w\n"
                   ";",
                   Tab));
  verifyFormat("class X {\n"
               "\tvoid f() {\n"
               "\t\tsomeFunction(parameter1,\n"
               "\t\t             parameter2);\n"
               "\t}\n"
               "};",
               Tab);
  verifyFormat("{\n"
               "\tQ(\n"
               "\t    {\n"
               "\t\t    int a;\n"
               "\t\t    someFunction(aaaaaaaa,\n"
               "\t\t                 bbbbbbb);\n"
               "\t    },\n"
               "\t    p);\n"
               "}",
               Tab);
  EXPECT_EQ("{\n"
            "\t/* aaaa\n"
            "\t   bbbb */\n"
            "}",
            format("{\n"
                   "/* aaaa\n"
                   "   bbbb */\n"
                   "}",
                   Tab));
  EXPECT_EQ("{\n"
            "\t/*\n"
            "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
            "\t  bbbbbbbbbbbbb\n"
            "\t*/\n"
            "}",
            format("{\n"
                   "/*\n"
                   "  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
                   "*/\n"
                   "}",
                   Tab));
  EXPECT_EQ("{\n"
            "\t// aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
            "\t// bbbbbbbbbbbbb\n"
            "}",
            format("{\n"
                   "\t// aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
                   "}",
                   Tab));
  EXPECT_EQ("{\n"
            "\t/*\n"
            "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa\n"
            "\t  bbbbbbbbbbbbb\n"
            "\t*/\n"
            "}",
            format("{\n"
                   "\t/*\n"
                   "\t  aaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb\n"
                   "\t*/\n"
                   "}",
                   Tab));
  EXPECT_EQ("{\n"
            "\t/*\n"
            "\n"
            "\t*/\n"
            "}",
            format("{\n"
                   "\t/*\n"
                   "\n"
                   "\t*/\n"
                   "}",
                   Tab));
  EXPECT_EQ("{\n"
            "\t/*\n"
            " asdf\n"
            "\t*/\n"
            "}",
            format("{\n"
                   "\t/*\n"
                   " asdf\n"
                   "\t*/\n"
                   "}",
                   Tab));

  Tab.UseTab = FormatStyle::UT_Never;
  EXPECT_EQ("/*\n"
            "              a\t\tcomment\n"
            "              in multiple lines\n"
            "       */",
            format("   /*\t \t \n"
                   " \t \t a\t\tcomment\t \t\n"
                   " \t \t in multiple lines\t\n"
                   " \t  */",
                   Tab));
  EXPECT_EQ("/* some\n"
            "   comment */",
            format(" \t \t /* some\n"
                   " \t \t    comment */",
                   Tab));
  EXPECT_EQ("int a; /* some\n"
            "   comment */",
            format(" \t \t int a; /* some\n"
                   " \t \t    comment */",
                   Tab));

  EXPECT_EQ("int a; /* some\n"
            "comment */",
            format(" \t \t int\ta; /* some\n"
                   " \t \t    comment */",
                   Tab));
  EXPECT_EQ("f(\"\t\t\"); /* some\n"
            "    comment */",
            format(" \t \t f(\"\t\t\"); /* some\n"
                   " \t \t    comment */",
                   Tab));
  EXPECT_EQ("{\n"
            "  /*\n"
            "   * Comment\n"
            "   */\n"
            "  int i;\n"
            "}",
            format("{\n"
                   "\t/*\n"
                   "\t * Comment\n"
                   "\t */\n"
                   "\t int i;\n"
                   "}"));
}

TEST_F(FormatTest, CalculatesOriginalColumn) {
  EXPECT_EQ("\"qqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
            "q\"; /* some\n"
            "       comment */",
            format("  \"qqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
                   "q\"; /* some\n"
                   "       comment */",
                   getLLVMStyle()));
  EXPECT_EQ("// qqqqqqqqqqqqqqqqqqqqqqqqqq\n"
            "/* some\n"
            "   comment */",
            format("// qqqqqqqqqqqqqqqqqqqqqqqqqq\n"
                   " /* some\n"
                   "    comment */",
                   getLLVMStyle()));
  EXPECT_EQ("// qqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
            "qqq\n"
            "/* some\n"
            "   comment */",
            format("// qqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
                   "qqq\n"
                   " /* some\n"
                   "    comment */",
                   getLLVMStyle()));
  EXPECT_EQ("inttt qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
            "wwww; /* some\n"
            "         comment */",
            format("  inttt qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\\\n"
                   "wwww; /* some\n"
                   "         comment */",
                   getLLVMStyle()));
}

TEST_F(FormatTest, ConfigurableSpaceBeforeParens) {
  FormatStyle NoSpace = getLLVMStyle();
  NoSpace.SpaceBeforeParens = FormatStyle::SBPO_Never;

  verifyFormat("while(true)\n"
               "  continue;",
               NoSpace);
  verifyFormat("for(;;)\n"
               "  continue;",
               NoSpace);
  verifyFormat("if(true)\n"
               "  f();\n"
               "else if(true)\n"
               "  f();",
               NoSpace);
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while(something());",
               NoSpace);
  verifyFormat("switch(x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               NoSpace);
  verifyFormat("auto i = std::make_unique<int>(5);", NoSpace);
  verifyFormat("size_t x = sizeof(x);", NoSpace);
  verifyFormat("auto f(int x) -> decltype(x);", NoSpace);
  verifyFormat("int f(T x) noexcept(x.create());", NoSpace);
  verifyFormat("alignas(128) char a[128];", NoSpace);
  verifyFormat("size_t x = alignof(MyType);", NoSpace);
  verifyFormat("static_assert(sizeof(char) == 1, \"Impossible!\");", NoSpace);
  verifyFormat("int f() throw(Deprecated);", NoSpace);
  verifyFormat("typedef void (*cb)(int);", NoSpace);
  verifyFormat("T A::operator()();", NoSpace);
  verifyFormat("X A::operator++(T);", NoSpace);

  FormatStyle Space = getLLVMStyle();
  Space.SpaceBeforeParens = FormatStyle::SBPO_Always;

  verifyFormat("int f ();", Space);
  verifyFormat("void f (int a, T b) {\n"
               "  while (true)\n"
               "    continue;\n"
               "}",
               Space);
  verifyFormat("if (true)\n"
               "  f ();\n"
               "else if (true)\n"
               "  f ();",
               Space);
  verifyFormat("do {\n"
               "  do_something ();\n"
               "} while (something ());",
               Space);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Space);
  verifyFormat("A::A () : a (1) {}", Space);
  verifyFormat("void f () __attribute__ ((asdf));", Space);
  verifyFormat("*(&a + 1);\n"
               "&((&a)[1]);\n"
               "a[(b + c) * d];\n"
               "(((a + 1) * 2) + 3) * 4;",
               Space);
  verifyFormat("#define A(x) x", Space);
  verifyFormat("#define A (x) x", Space);
  verifyFormat("#if defined(x)\n"
               "#endif",
               Space);
  verifyFormat("auto i = std::make_unique<int> (5);", Space);
  verifyFormat("size_t x = sizeof (x);", Space);
  verifyFormat("auto f (int x) -> decltype (x);", Space);
  verifyFormat("int f (T x) noexcept (x.create ());", Space);
  verifyFormat("alignas (128) char a[128];", Space);
  verifyFormat("size_t x = alignof (MyType);", Space);
  verifyFormat("static_assert (sizeof (char) == 1, \"Impossible!\");", Space);
  verifyFormat("int f () throw (Deprecated);", Space);
  verifyFormat("typedef void (*cb) (int);", Space);
  verifyFormat("T A::operator() ();", Space);
  verifyFormat("X A::operator++ (T);", Space);
}

TEST_F(FormatTest, ConfigurableSpacesInParentheses) {
  FormatStyle Spaces = getLLVMStyle();

  Spaces.SpacesInParentheses = true;
  verifyFormat("call( x, y, z );", Spaces);
  verifyFormat("call();", Spaces);
  verifyFormat("std::function<void( int, int )> callback;", Spaces);
  verifyFormat("void inFunction() { std::function<void( int, int )> fct; }",
               Spaces);
  verifyFormat("while ( (bool)1 )\n"
               "  continue;",
               Spaces);
  verifyFormat("for ( ;; )\n"
               "  continue;",
               Spaces);
  verifyFormat("if ( true )\n"
               "  f();\n"
               "else if ( true )\n"
               "  f();",
               Spaces);
  verifyFormat("do {\n"
               "  do_something( (int)i );\n"
               "} while ( something() );",
               Spaces);
  verifyFormat("switch ( x ) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Spaces);

  Spaces.SpacesInParentheses = false;
  Spaces.SpacesInCStyleCastParentheses = true;
  verifyFormat("Type *A = ( Type * )P;", Spaces);
  verifyFormat("Type *A = ( vector<Type *, int *> )P;", Spaces);
  verifyFormat("x = ( int32 )y;", Spaces);
  verifyFormat("int a = ( int )(2.0f);", Spaces);
  verifyFormat("#define AA(X) sizeof((( X * )NULL)->a)", Spaces);
  verifyFormat("my_int a = ( my_int )sizeof(int);", Spaces);
  verifyFormat("#define x (( int )-1)", Spaces);

  // Run the first set of tests again with:
  Spaces.SpacesInParentheses = false, Spaces.SpaceInEmptyParentheses = true;
  Spaces.SpacesInCStyleCastParentheses = true;
  verifyFormat("call(x, y, z);", Spaces);
  verifyFormat("call( );", Spaces);
  verifyFormat("std::function<void(int, int)> callback;", Spaces);
  verifyFormat("while (( bool )1)\n"
               "  continue;",
               Spaces);
  verifyFormat("for (;;)\n"
               "  continue;",
               Spaces);
  verifyFormat("if (true)\n"
               "  f( );\n"
               "else if (true)\n"
               "  f( );",
               Spaces);
  verifyFormat("do {\n"
               "  do_something(( int )i);\n"
               "} while (something( ));",
               Spaces);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Spaces);

  // Run the first set of tests again with:
  Spaces.SpaceAfterCStyleCast = true;
  verifyFormat("call(x, y, z);", Spaces);
  verifyFormat("call( );", Spaces);
  verifyFormat("std::function<void(int, int)> callback;", Spaces);
  verifyFormat("while (( bool ) 1)\n"
               "  continue;",
               Spaces);
  verifyFormat("for (;;)\n"
               "  continue;",
               Spaces);
  verifyFormat("if (true)\n"
               "  f( );\n"
               "else if (true)\n"
               "  f( );",
               Spaces);
  verifyFormat("do {\n"
               "  do_something(( int ) i);\n"
               "} while (something( ));",
               Spaces);
  verifyFormat("switch (x) {\n"
               "default:\n"
               "  break;\n"
               "}",
               Spaces);

  // Run subset of tests again with:
  Spaces.SpacesInCStyleCastParentheses = false;
  Spaces.SpaceAfterCStyleCast = true;
  verifyFormat("while ((bool) 1)\n"
               "  continue;",
               Spaces);
  verifyFormat("do {\n"
               "  do_something((int) i);\n"
               "} while (something( ));",
               Spaces);
}

TEST_F(FormatTest, ConfigurableSpacesInSquareBrackets) {
  verifyFormat("int a[5];");
  verifyFormat("a[3] += 42;");

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInSquareBrackets = true;
  // Lambdas unchanged.
  verifyFormat("int c = []() -> int { return 2; }();\n", Spaces);
  verifyFormat("return [i, args...] {};", Spaces);

  // Not lambdas.
  verifyFormat("int a[ 5 ];", Spaces);
  verifyFormat("a[ 3 ] += 42;", Spaces);
  verifyFormat("constexpr char hello[]{\"hello\"};", Spaces);
  verifyFormat("double &operator[](int i) { return 0; }\n"
               "int i;",
               Spaces);
  verifyFormat("std::unique_ptr<int[]> foo() {}", Spaces);
  verifyFormat("int i = a[ a ][ a ]->f();", Spaces);
  verifyFormat("int i = (*b)[ a ]->f();", Spaces);
}

TEST_F(FormatTest, ConfigurableSpaceBeforeAssignmentOperators) {
  verifyFormat("int a = 5;");
  verifyFormat("a += 42;");
  verifyFormat("a or_eq 8;");

  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpaceBeforeAssignmentOperators = false;
  verifyFormat("int a= 5;", Spaces);
  verifyFormat("a+= 42;", Spaces);
  verifyFormat("a or_eq 8;", Spaces);
}

TEST_F(FormatTest, AlignConsecutiveAssignments) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveAssignments = false;
  verifyFormat("int a = 5;\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a = 5;\n"
               "int oneTwoThree = 123;",
               Alignment);

  Alignment.AlignConsecutiveAssignments = true;
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
               "int i = 1, j = 10;\n",
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
               "method();\n",
               Alignment);
  verifyFormat("int oneTwoThree = 123; // comment\n"
               "int oneTwo      = 12;  // comment",
               Alignment);
  EXPECT_EQ("int a = 5;\n"
            "\n"
            "int oneTwoThree = 123;",
            format("int a       = 5;\n"
                   "\n"
                   "int oneTwoThree= 123;",
                   Alignment));
  EXPECT_EQ("int a   = 5;\n"
            "int one = 1;\n"
            "\n"
            "int oneTwoThree = 123;",
            format("int a = 5;\n"
                   "int one = 1;\n"
                   "\n"
                   "int oneTwoThree = 123;",
                   Alignment));
  EXPECT_EQ("int a   = 5;\n"
            "int one = 1;\n"
            "\n"
            "int oneTwoThree = 123;\n"
            "int oneTwo      = 12;",
            format("int a = 5;\n"
                   "int one = 1;\n"
                   "\n"
                   "int oneTwoThree = 123;\n"
                   "int oneTwo = 12;",
                   Alignment));
  Alignment.AlignEscapedNewlinesLeft = true;
  verifyFormat("#define A               \\\n"
               "  int aaaa       = 12;  \\\n"
               "  int b          = 23;  \\\n"
               "  int ccc        = 234; \\\n"
               "  int dddddddddd = 2345;",
               Alignment);
  Alignment.AlignEscapedNewlinesLeft = false;
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

  // FIXME: Should align all three assignments
  verifyFormat(
      "int i      = 1;\n"
      "SomeType a = SomeFunction(looooooooooooooooooooooongParameterA,\n"
      "                          loooooooooooooooooooooongParameterB);\n"
      "int j = 2;",
      Alignment);

  verifyFormat("template <typename T, typename T_0 = very_long_type_name_0,\n"
               "          typename B   = very_long_type_name_1,\n"
               "          typename T_2 = very_long_type_name_2>\n"
               "auto foo() {}\n",
               Alignment);
  verifyFormat("int a, b = 1;\n"
               "int c  = 2;\n"
               "int dd = 3;\n",
               Alignment);
  verifyFormat("int aa       = ((1 > 2) ? 3 : 4);\n"
               "float b[1][] = {{3.f}};\n",
               Alignment);
}

TEST_F(FormatTest, AlignConsecutiveDeclarations) {
  FormatStyle Alignment = getLLVMStyle();
  Alignment.AlignConsecutiveDeclarations = false;
  verifyFormat("float const a = 5;\n"
               "int oneTwoThree = 123;",
               Alignment);
  verifyFormat("int a = 5;\n"
               "float const oneTwoThree = 123;",
               Alignment);

  Alignment.AlignConsecutiveDeclarations = true;
  verifyFormat("float const a = 5;\n"
               "int         oneTwoThree = 123;",
               Alignment);
  verifyFormat("int         a = method();\n"
               "float const oneTwoThree = 133;",
               Alignment);
  verifyFormat("int i = 1, j = 10;\n"
               "something = 2000;",
               Alignment);
  verifyFormat("something = 2000;\n"
               "int i = 1, j = 10;\n",
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
  EXPECT_EQ("float const a = 5;\n"
            "\n"
            "int oneTwoThree = 123;",
            format("float const   a = 5;\n"
                   "\n"
                   "int           oneTwoThree= 123;",
                   Alignment));
  EXPECT_EQ("float a = 5;\n"
            "int   one = 1;\n"
            "\n"
            "unsigned oneTwoThree = 123;",
            format("float    a = 5;\n"
                   "int      one = 1;\n"
                   "\n"
                   "unsigned oneTwoThree = 123;",
                   Alignment));
  EXPECT_EQ("float a = 5;\n"
            "int   one = 1;\n"
            "\n"
            "unsigned oneTwoThree = 123;\n"
            "int      oneTwo = 12;",
            format("float    a = 5;\n"
                   "int one = 1;\n"
                   "\n"
                   "unsigned oneTwoThree = 123;\n"
                   "int oneTwo = 12;",
                   Alignment));
  Alignment.AlignConsecutiveAssignments = true;
  verifyFormat("float      something = 2000;\n"
               "double     another   = 911;\n"
               "int        i = 1, j = 10;\n"
               "const int *oneMore = 1;\n"
               "unsigned   i       = 2;",
               Alignment);
  verifyFormat("int      oneTwoThree = {0}; // comment\n"
               "unsigned oneTwo      = 0;   // comment",
               Alignment);
  EXPECT_EQ("void SomeFunction(int parameter = 0) {\n"
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
            format("void SomeFunction(int parameter= 0) {\n"
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
                   Alignment));
  Alignment.AlignConsecutiveAssignments = false;
  Alignment.AlignEscapedNewlinesLeft = true;
  verifyFormat("#define A              \\\n"
               "  int       aaaa = 12; \\\n"
               "  float     b = 23;    \\\n"
               "  const int ccc = 234; \\\n"
               "  unsigned  dddddddddd = 2345;",
               Alignment);
  Alignment.AlignEscapedNewlinesLeft = false;
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
               "  int *     j = 2;\n"
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

  Alignment.AlignConsecutiveAssignments = true;
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
  Alignment.AlignConsecutiveAssignments = false;

  // FIXME: Should align all three declarations
  verifyFormat(
      "int      i = 1;\n"
      "SomeType a = SomeFunction(looooooooooooooooooooooongParameterA,\n"
      "                          loooooooooooooooooooooongParameterB);\n"
      "int j = 2;",
      Alignment);

  // Test interactions with ColumnLimit and AlignConsecutiveAssignments:
  // We expect declarations and assignments to align, as long as it doesn't
  // exceed the column limit, starting a new alignemnt sequence whenever it
  // happens.
  Alignment.AlignConsecutiveAssignments = true;
  Alignment.ColumnLimit = 30;
  verifyFormat("float    ii              = 1;\n"
               "unsigned j               = 2;\n"
               "int someVerylongVariable = 1;\n"
               "AnotherLongType  ll = 123456;\n"
               "VeryVeryLongType k  = 2;\n"
               "int              myvar = 1;",
               Alignment);
  Alignment.ColumnLimit = 80;
  Alignment.AlignConsecutiveAssignments = false;

  verifyFormat(
      "template <typename LongTemplate, typename VeryLongTemplateTypeName,\n"
      "          typename LongType, typename B>\n"
      "auto foo() {}\n",
      Alignment);
  verifyFormat("float a, b = 1;\n"
               "int   c = 2;\n"
               "int   dd = 3;\n",
               Alignment);
  verifyFormat("int   aa = ((1 > 2) ? 3 : 4);\n"
               "float b[1][] = {{3.f}};\n",
               Alignment);
  Alignment.AlignConsecutiveAssignments = true;
  verifyFormat("float a, b = 1;\n"
               "int   c  = 2;\n"
               "int   dd = 3;\n",
               Alignment);
  verifyFormat("int   aa     = ((1 > 2) ? 3 : 4);\n"
               "float b[1][] = {{3.f}};\n",
               Alignment);
  Alignment.AlignConsecutiveAssignments = false;

  Alignment.ColumnLimit = 30;
  Alignment.BinPackParameters = false;
  verifyFormat("void foo(float     a,\n"
               "         float     b,\n"
               "         int       c,\n"
               "         uint32_t *d) {\n"
               "  int *  e = 0;\n"
               "  float  f = 0;\n"
               "  double g = 0;\n"
               "}\n"
               "void bar(ino_t     a,\n"
               "         int       b,\n"
               "         uint32_t *c,\n"
               "         bool      d) {}\n",
               Alignment);
  Alignment.BinPackParameters = true;
  Alignment.ColumnLimit = 80;
}

TEST_F(FormatTest, LinuxBraceBreaking) {
  FormatStyle LinuxBraceStyle = getLLVMStyle();
  LinuxBraceStyle.BreakBeforeBraces = FormatStyle::BS_Linux;
  verifyFormat("namespace a\n"
               "{\n"
               "class A\n"
               "{\n"
               "  void f()\n"
               "  {\n"
               "    if (true) {\n"
               "      a();\n"
               "      b();\n"
               "    } else {\n"
               "      a();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "struct B {\n"
               "  int x;\n"
               "};\n"
               "}\n",
               LinuxBraceStyle);
  verifyFormat("enum X {\n"
               "  Y = 0,\n"
               "}\n",
               LinuxBraceStyle);
  verifyFormat("struct S {\n"
               "  int Type;\n"
               "  union {\n"
               "    int x;\n"
               "    double y;\n"
               "  } Value;\n"
               "  class C\n"
               "  {\n"
               "    MyFavoriteType Value;\n"
               "  } Class;\n"
               "}\n",
               LinuxBraceStyle);
}

TEST_F(FormatTest, MozillaBraceBreaking) {
  FormatStyle MozillaBraceStyle = getLLVMStyle();
  MozillaBraceStyle.BreakBeforeBraces = FormatStyle::BS_Mozilla;
  verifyFormat("namespace a {\n"
               "class A\n"
               "{\n"
               "  void f()\n"
               "  {\n"
               "    if (true) {\n"
               "      a();\n"
               "      b();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "enum E\n"
               "{\n"
               "  A,\n"
               "  // foo\n"
               "  B,\n"
               "  C\n"
               "};\n"
               "struct B\n"
               "{\n"
               "  int x;\n"
               "};\n"
               "}\n",
               MozillaBraceStyle);
  verifyFormat("struct S\n"
               "{\n"
               "  int Type;\n"
               "  union\n"
               "  {\n"
               "    int x;\n"
               "    double y;\n"
               "  } Value;\n"
               "  class C\n"
               "  {\n"
               "    MyFavoriteType Value;\n"
               "  } Class;\n"
               "}\n",
               MozillaBraceStyle);
}

TEST_F(FormatTest, StroustrupBraceBreaking) {
  FormatStyle StroustrupBraceStyle = getLLVMStyle();
  StroustrupBraceStyle.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  verifyFormat("namespace a {\n"
               "class A {\n"
               "  void f()\n"
               "  {\n"
               "    if (true) {\n"
               "      a();\n"
               "      b();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "struct B {\n"
               "  int x;\n"
               "};\n"
               "}\n",
               StroustrupBraceStyle);

  verifyFormat("void foo()\n"
               "{\n"
               "  if (a) {\n"
               "    a();\n"
               "  }\n"
               "  else {\n"
               "    b();\n"
               "  }\n"
               "}\n",
               StroustrupBraceStyle);

  verifyFormat("#ifdef _DEBUG\n"
               "int foo(int i = 0)\n"
               "#else\n"
               "int foo(int i = 5)\n"
               "#endif\n"
               "{\n"
               "  return i;\n"
               "}",
               StroustrupBraceStyle);

  verifyFormat("void foo() {}\n"
               "void bar()\n"
               "#ifdef _DEBUG\n"
               "{\n"
               "  foo();\n"
               "}\n"
               "#else\n"
               "{\n"
               "}\n"
               "#endif",
               StroustrupBraceStyle);

  verifyFormat("void foobar() { int i = 5; }\n"
               "#ifdef _DEBUG\n"
               "void bar() {}\n"
               "#else\n"
               "void bar() { foobar(); }\n"
               "#endif",
               StroustrupBraceStyle);
}

TEST_F(FormatTest, AllmanBraceBreaking) {
  FormatStyle AllmanBraceStyle = getLLVMStyle();
  AllmanBraceStyle.BreakBeforeBraces = FormatStyle::BS_Allman;
  verifyFormat("namespace a\n"
               "{\n"
               "class A\n"
               "{\n"
               "  void f()\n"
               "  {\n"
               "    if (true)\n"
               "    {\n"
               "      a();\n"
               "      b();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "struct B\n"
               "{\n"
               "  int x;\n"
               "};\n"
               "}",
               AllmanBraceStyle);

  verifyFormat("void f()\n"
               "{\n"
               "  if (true)\n"
               "  {\n"
               "    a();\n"
               "  }\n"
               "  else if (false)\n"
               "  {\n"
               "    b();\n"
               "  }\n"
               "  else\n"
               "  {\n"
               "    c();\n"
               "  }\n"
               "}\n",
               AllmanBraceStyle);

  verifyFormat("void f()\n"
               "{\n"
               "  for (int i = 0; i < 10; ++i)\n"
               "  {\n"
               "    a();\n"
               "  }\n"
               "  while (false)\n"
               "  {\n"
               "    b();\n"
               "  }\n"
               "  do\n"
               "  {\n"
               "    c();\n"
               "  } while (false)\n"
               "}\n",
               AllmanBraceStyle);

  verifyFormat("void f(int a)\n"
               "{\n"
               "  switch (a)\n"
               "  {\n"
               "  case 0:\n"
               "    break;\n"
               "  case 1:\n"
               "  {\n"
               "    break;\n"
               "  }\n"
               "  case 2:\n"
               "  {\n"
               "  }\n"
               "  break;\n"
               "  default:\n"
               "    break;\n"
               "  }\n"
               "}\n",
               AllmanBraceStyle);

  verifyFormat("enum X\n"
               "{\n"
               "  Y = 0,\n"
               "}\n",
               AllmanBraceStyle);
  verifyFormat("enum X\n"
               "{\n"
               "  Y = 0\n"
               "}\n",
               AllmanBraceStyle);

  verifyFormat("@interface BSApplicationController ()\n"
               "{\n"
               "@private\n"
               "  id _extraIvar;\n"
               "}\n"
               "@end\n",
               AllmanBraceStyle);

  verifyFormat("#ifdef _DEBUG\n"
               "int foo(int i = 0)\n"
               "#else\n"
               "int foo(int i = 5)\n"
               "#endif\n"
               "{\n"
               "  return i;\n"
               "}",
               AllmanBraceStyle);

  verifyFormat("void foo() {}\n"
               "void bar()\n"
               "#ifdef _DEBUG\n"
               "{\n"
               "  foo();\n"
               "}\n"
               "#else\n"
               "{\n"
               "}\n"
               "#endif",
               AllmanBraceStyle);

  verifyFormat("void foobar() { int i = 5; }\n"
               "#ifdef _DEBUG\n"
               "void bar() {}\n"
               "#else\n"
               "void bar() { foobar(); }\n"
               "#endif",
               AllmanBraceStyle);

  // This shouldn't affect ObjC blocks..
  verifyFormat("[self doSomeThingWithACompletionHandler:^{\n"
               "  // ...\n"
               "  int i;\n"
               "}];",
               AllmanBraceStyle);
  verifyFormat("void (^block)(void) = ^{\n"
               "  // ...\n"
               "  int i;\n"
               "};",
               AllmanBraceStyle);
  // .. or dict literals.
  verifyFormat("void f()\n"
               "{\n"
               "  [object someMethod:@{ @\"a\" : @\"b\" }];\n"
               "}",
               AllmanBraceStyle);
  verifyFormat("int f()\n"
               "{ // comment\n"
               "  return 42;\n"
               "}",
               AllmanBraceStyle);

  AllmanBraceStyle.ColumnLimit = 19;
  verifyFormat("void f() { int i; }", AllmanBraceStyle);
  AllmanBraceStyle.ColumnLimit = 18;
  verifyFormat("void f()\n"
               "{\n"
               "  int i;\n"
               "}",
               AllmanBraceStyle);
  AllmanBraceStyle.ColumnLimit = 80;

  FormatStyle BreakBeforeBraceShortIfs = AllmanBraceStyle;
  BreakBeforeBraceShortIfs.AllowShortIfStatementsOnASingleLine = true;
  BreakBeforeBraceShortIfs.AllowShortLoopsOnASingleLine = true;
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  if (b)\n"
               "  {\n"
               "    return;\n"
               "  }\n"
               "}\n",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  if (b) return;\n"
               "}\n",
               BreakBeforeBraceShortIfs);
  verifyFormat("void f(bool b)\n"
               "{\n"
               "  while (b)\n"
               "  {\n"
               "    return;\n"
               "  }\n"
               "}\n",
               BreakBeforeBraceShortIfs);
}

TEST_F(FormatTest, GNUBraceBreaking) {
  FormatStyle GNUBraceStyle = getLLVMStyle();
  GNUBraceStyle.BreakBeforeBraces = FormatStyle::BS_GNU;
  verifyFormat("namespace a\n"
               "{\n"
               "class A\n"
               "{\n"
               "  void f()\n"
               "  {\n"
               "    int a;\n"
               "    {\n"
               "      int b;\n"
               "    }\n"
               "    if (true)\n"
               "      {\n"
               "        a();\n"
               "        b();\n"
               "      }\n"
               "  }\n"
               "  void g() { return; }\n"
               "}\n"
               "}",
               GNUBraceStyle);

  verifyFormat("void f()\n"
               "{\n"
               "  if (true)\n"
               "    {\n"
               "      a();\n"
               "    }\n"
               "  else if (false)\n"
               "    {\n"
               "      b();\n"
               "    }\n"
               "  else\n"
               "    {\n"
               "      c();\n"
               "    }\n"
               "}\n",
               GNUBraceStyle);

  verifyFormat("void f()\n"
               "{\n"
               "  for (int i = 0; i < 10; ++i)\n"
               "    {\n"
               "      a();\n"
               "    }\n"
               "  while (false)\n"
               "    {\n"
               "      b();\n"
               "    }\n"
               "  do\n"
               "    {\n"
               "      c();\n"
               "    }\n"
               "  while (false);\n"
               "}\n",
               GNUBraceStyle);

  verifyFormat("void f(int a)\n"
               "{\n"
               "  switch (a)\n"
               "    {\n"
               "    case 0:\n"
               "      break;\n"
               "    case 1:\n"
               "      {\n"
               "        break;\n"
               "      }\n"
               "    case 2:\n"
               "      {\n"
               "      }\n"
               "      break;\n"
               "    default:\n"
               "      break;\n"
               "    }\n"
               "}\n",
               GNUBraceStyle);

  verifyFormat("enum X\n"
               "{\n"
               "  Y = 0,\n"
               "}\n",
               GNUBraceStyle);

  verifyFormat("@interface BSApplicationController ()\n"
               "{\n"
               "@private\n"
               "  id _extraIvar;\n"
               "}\n"
               "@end\n",
               GNUBraceStyle);

  verifyFormat("#ifdef _DEBUG\n"
               "int foo(int i = 0)\n"
               "#else\n"
               "int foo(int i = 5)\n"
               "#endif\n"
               "{\n"
               "  return i;\n"
               "}",
               GNUBraceStyle);

  verifyFormat("void foo() {}\n"
               "void bar()\n"
               "#ifdef _DEBUG\n"
               "{\n"
               "  foo();\n"
               "}\n"
               "#else\n"
               "{\n"
               "}\n"
               "#endif",
               GNUBraceStyle);

  verifyFormat("void foobar() { int i = 5; }\n"
               "#ifdef _DEBUG\n"
               "void bar() {}\n"
               "#else\n"
               "void bar() { foobar(); }\n"
               "#endif",
               GNUBraceStyle);
}

TEST_F(FormatTest, WebKitBraceBreaking) {
  FormatStyle WebKitBraceStyle = getLLVMStyle();
  WebKitBraceStyle.BreakBeforeBraces = FormatStyle::BS_WebKit;
  verifyFormat("namespace a {\n"
               "class A {\n"
               "  void f()\n"
               "  {\n"
               "    if (true) {\n"
               "      a();\n"
               "      b();\n"
               "    }\n"
               "  }\n"
               "  void g() { return; }\n"
               "};\n"
               "enum E {\n"
               "  A,\n"
               "  // foo\n"
               "  B,\n"
               "  C\n"
               "};\n"
               "struct B {\n"
               "  int x;\n"
               "};\n"
               "}\n",
               WebKitBraceStyle);
  verifyFormat("struct S {\n"
               "  int Type;\n"
               "  union {\n"
               "    int x;\n"
               "    double y;\n"
               "  } Value;\n"
               "  class C {\n"
               "    MyFavoriteType Value;\n"
               "  } Class;\n"
               "};\n",
               WebKitBraceStyle);
}

TEST_F(FormatTest, CatchExceptionReferenceBinding) {
  verifyFormat("void f() {\n"
               "  try {\n"
               "  } catch (const Exception &e) {\n"
               "  }\n"
               "}\n",
               getLLVMStyle());
}

TEST_F(FormatTest, UnderstandsPragmas) {
  verifyFormat("#pragma omp reduction(| : var)");
  verifyFormat("#pragma omp reduction(+ : var)");

  EXPECT_EQ("#pragma mark Any non-hyphenated or hyphenated string "
            "(including parentheses).",
            format("#pragma    mark   Any non-hyphenated or hyphenated string "
                   "(including parentheses)."));
}

TEST_F(FormatTest, UnderstandPragmaOption) {
  verifyFormat("#pragma option -C -A");

  EXPECT_EQ("#pragma option -C -A", format("#pragma    option   -C   -A"));
}

#define EXPECT_ALL_STYLES_EQUAL(Styles)                                        \
  for (size_t i = 1; i < Styles.size(); ++i)                                   \
  EXPECT_EQ(Styles[0], Styles[i]) << "Style #" << i << " of " << Styles.size() \
                                  << " differs from Style #0"

TEST_F(FormatTest, GetsPredefinedStyleByName) {
  SmallVector<FormatStyle, 3> Styles;
  Styles.resize(3);

  Styles[0] = getLLVMStyle();
  EXPECT_TRUE(getPredefinedStyle("LLVM", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("lLvM", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getGoogleStyle();
  EXPECT_TRUE(getPredefinedStyle("Google", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("gOOgle", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getGoogleStyle(FormatStyle::LK_JavaScript);
  EXPECT_TRUE(
      getPredefinedStyle("Google", FormatStyle::LK_JavaScript, &Styles[1]));
  EXPECT_TRUE(
      getPredefinedStyle("gOOgle", FormatStyle::LK_JavaScript, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getChromiumStyle(FormatStyle::LK_Cpp);
  EXPECT_TRUE(getPredefinedStyle("Chromium", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("cHRoMiUM", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getMozillaStyle();
  EXPECT_TRUE(getPredefinedStyle("Mozilla", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("moZILla", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getWebKitStyle();
  EXPECT_TRUE(getPredefinedStyle("WebKit", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("wEbKit", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles[0] = getGNUStyle();
  EXPECT_TRUE(getPredefinedStyle("GNU", FormatStyle::LK_Cpp, &Styles[1]));
  EXPECT_TRUE(getPredefinedStyle("gnU", FormatStyle::LK_Cpp, &Styles[2]));
  EXPECT_ALL_STYLES_EQUAL(Styles);

  EXPECT_FALSE(getPredefinedStyle("qwerty", FormatStyle::LK_Cpp, &Styles[0]));
}

TEST_F(FormatTest, GetsCorrectBasedOnStyle) {
  SmallVector<FormatStyle, 8> Styles;
  Styles.resize(2);

  Styles[0] = getGoogleStyle();
  Styles[1] = getLLVMStyle();
  EXPECT_EQ(0, parseConfiguration("BasedOnStyle: Google", &Styles[1]).value());
  EXPECT_ALL_STYLES_EQUAL(Styles);

  Styles.resize(5);
  Styles[0] = getGoogleStyle(FormatStyle::LK_JavaScript);
  Styles[1] = getLLVMStyle();
  Styles[1].Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(0, parseConfiguration("BasedOnStyle: Google", &Styles[1]).value());

  Styles[2] = getLLVMStyle();
  Styles[2].Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(0, parseConfiguration("Language: JavaScript\n"
                                  "BasedOnStyle: Google",
                                  &Styles[2])
                   .value());

  Styles[3] = getLLVMStyle();
  Styles[3].Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(0, parseConfiguration("BasedOnStyle: Google\n"
                                  "Language: JavaScript",
                                  &Styles[3])
                   .value());

  Styles[4] = getLLVMStyle();
  Styles[4].Language = FormatStyle::LK_JavaScript;
  EXPECT_EQ(0, parseConfiguration("---\n"
                                  "BasedOnStyle: LLVM\n"
                                  "IndentWidth: 123\n"
                                  "---\n"
                                  "BasedOnStyle: Google\n"
                                  "Language: JavaScript",
                                  &Styles[4])
                   .value());
  EXPECT_ALL_STYLES_EQUAL(Styles);
}

#define CHECK_PARSE_BOOL_FIELD(FIELD, CONFIG_NAME)                             \
  Style.FIELD = false;                                                         \
  EXPECT_EQ(0, parseConfiguration(CONFIG_NAME ": true", &Style).value());      \
  EXPECT_TRUE(Style.FIELD);                                                    \
  EXPECT_EQ(0, parseConfiguration(CONFIG_NAME ": false", &Style).value());     \
  EXPECT_FALSE(Style.FIELD);

#define CHECK_PARSE_BOOL(FIELD) CHECK_PARSE_BOOL_FIELD(FIELD, #FIELD)

#define CHECK_PARSE_NESTED_BOOL_FIELD(STRUCT, FIELD, CONFIG_NAME)              \
  Style.STRUCT.FIELD = false;                                                  \
  EXPECT_EQ(0,                                                                 \
            parseConfiguration(#STRUCT ":\n  " CONFIG_NAME ": true", &Style)   \
                .value());                                                     \
  EXPECT_TRUE(Style.STRUCT.FIELD);                                             \
  EXPECT_EQ(0,                                                                 \
            parseConfiguration(#STRUCT ":\n  " CONFIG_NAME ": false", &Style)  \
                .value());                                                     \
  EXPECT_FALSE(Style.STRUCT.FIELD);

#define CHECK_PARSE_NESTED_BOOL(STRUCT, FIELD)                                 \
  CHECK_PARSE_NESTED_BOOL_FIELD(STRUCT, FIELD, #FIELD)

#define CHECK_PARSE(TEXT, FIELD, VALUE)                                        \
  EXPECT_NE(VALUE, Style.FIELD);                                               \
  EXPECT_EQ(0, parseConfiguration(TEXT, &Style).value());                      \
  EXPECT_EQ(VALUE, Style.FIELD)

TEST_F(FormatTest, ParsesConfigurationBools) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE_BOOL(AlignEscapedNewlinesLeft);
  CHECK_PARSE_BOOL(AlignOperands);
  CHECK_PARSE_BOOL(AlignTrailingComments);
  CHECK_PARSE_BOOL(AlignConsecutiveAssignments);
  CHECK_PARSE_BOOL(AlignConsecutiveDeclarations);
  CHECK_PARSE_BOOL(AllowAllParametersOfDeclarationOnNextLine);
  CHECK_PARSE_BOOL(AllowShortBlocksOnASingleLine);
  CHECK_PARSE_BOOL(AllowShortCaseLabelsOnASingleLine);
  CHECK_PARSE_BOOL(AllowShortIfStatementsOnASingleLine);
  CHECK_PARSE_BOOL(AllowShortLoopsOnASingleLine);
  CHECK_PARSE_BOOL(AlwaysBreakTemplateDeclarations);
  CHECK_PARSE_BOOL(BinPackArguments);
  CHECK_PARSE_BOOL(BinPackParameters);
  CHECK_PARSE_BOOL(BreakBeforeTernaryOperators);
  CHECK_PARSE_BOOL(BreakConstructorInitializersBeforeComma);
  CHECK_PARSE_BOOL(ConstructorInitializerAllOnOneLineOrOnePerLine);
  CHECK_PARSE_BOOL(DerivePointerAlignment);
  CHECK_PARSE_BOOL_FIELD(DerivePointerAlignment, "DerivePointerBinding");
  CHECK_PARSE_BOOL(DisableFormat);
  CHECK_PARSE_BOOL(IndentCaseLabels);
  CHECK_PARSE_BOOL(IndentWrappedFunctionNames);
  CHECK_PARSE_BOOL(KeepEmptyLinesAtTheStartOfBlocks);
  CHECK_PARSE_BOOL(ObjCSpaceAfterProperty);
  CHECK_PARSE_BOOL(ObjCSpaceBeforeProtocolList);
  CHECK_PARSE_BOOL(Cpp11BracedListStyle);
  CHECK_PARSE_BOOL(ReflowComments);
  CHECK_PARSE_BOOL(SortIncludes);
  CHECK_PARSE_BOOL(SpacesInParentheses);
  CHECK_PARSE_BOOL(SpacesInSquareBrackets);
  CHECK_PARSE_BOOL(SpacesInAngles);
  CHECK_PARSE_BOOL(SpaceInEmptyParentheses);
  CHECK_PARSE_BOOL(SpacesInContainerLiterals);
  CHECK_PARSE_BOOL(SpacesInCStyleCastParentheses);
  CHECK_PARSE_BOOL(SpaceAfterCStyleCast);
  CHECK_PARSE_BOOL(SpaceBeforeAssignmentOperators);

  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterClass);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterControlStatement);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterEnum);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterFunction);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterNamespace);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterObjCDeclaration);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterStruct);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, AfterUnion);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, BeforeCatch);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, BeforeElse);
  CHECK_PARSE_NESTED_BOOL(BraceWrapping, IndentBraces);
}

#undef CHECK_PARSE_BOOL

TEST_F(FormatTest, ParsesConfiguration) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("AccessModifierOffset: -1234", AccessModifierOffset, -1234);
  CHECK_PARSE("ConstructorInitializerIndentWidth: 1234",
              ConstructorInitializerIndentWidth, 1234u);
  CHECK_PARSE("ObjCBlockIndentWidth: 1234", ObjCBlockIndentWidth, 1234u);
  CHECK_PARSE("ColumnLimit: 1234", ColumnLimit, 1234u);
  CHECK_PARSE("MaxEmptyLinesToKeep: 1234", MaxEmptyLinesToKeep, 1234u);
  CHECK_PARSE("PenaltyBreakBeforeFirstCallParameter: 1234",
              PenaltyBreakBeforeFirstCallParameter, 1234u);
  CHECK_PARSE("PenaltyExcessCharacter: 1234", PenaltyExcessCharacter, 1234u);
  CHECK_PARSE("PenaltyReturnTypeOnItsOwnLine: 1234",
              PenaltyReturnTypeOnItsOwnLine, 1234u);
  CHECK_PARSE("SpacesBeforeTrailingComments: 1234",
              SpacesBeforeTrailingComments, 1234u);
  CHECK_PARSE("IndentWidth: 32", IndentWidth, 32u);
  CHECK_PARSE("ContinuationIndentWidth: 11", ContinuationIndentWidth, 11u);

  Style.PointerAlignment = FormatStyle::PAS_Middle;
  CHECK_PARSE("PointerAlignment: Left", PointerAlignment,
              FormatStyle::PAS_Left);
  CHECK_PARSE("PointerAlignment: Right", PointerAlignment,
              FormatStyle::PAS_Right);
  CHECK_PARSE("PointerAlignment: Middle", PointerAlignment,
              FormatStyle::PAS_Middle);
  // For backward compatibility:
  CHECK_PARSE("PointerBindsToType: Left", PointerAlignment,
              FormatStyle::PAS_Left);
  CHECK_PARSE("PointerBindsToType: Right", PointerAlignment,
              FormatStyle::PAS_Right);
  CHECK_PARSE("PointerBindsToType: Middle", PointerAlignment,
              FormatStyle::PAS_Middle);

  Style.Standard = FormatStyle::LS_Auto;
  CHECK_PARSE("Standard: Cpp03", Standard, FormatStyle::LS_Cpp03);
  CHECK_PARSE("Standard: Cpp11", Standard, FormatStyle::LS_Cpp11);
  CHECK_PARSE("Standard: C++03", Standard, FormatStyle::LS_Cpp03);
  CHECK_PARSE("Standard: C++11", Standard, FormatStyle::LS_Cpp11);
  CHECK_PARSE("Standard: Auto", Standard, FormatStyle::LS_Auto);

  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  CHECK_PARSE("BreakBeforeBinaryOperators: NonAssignment",
              BreakBeforeBinaryOperators, FormatStyle::BOS_NonAssignment);
  CHECK_PARSE("BreakBeforeBinaryOperators: None", BreakBeforeBinaryOperators,
              FormatStyle::BOS_None);
  CHECK_PARSE("BreakBeforeBinaryOperators: All", BreakBeforeBinaryOperators,
              FormatStyle::BOS_All);
  // For backward compatibility:
  CHECK_PARSE("BreakBeforeBinaryOperators: false", BreakBeforeBinaryOperators,
              FormatStyle::BOS_None);
  CHECK_PARSE("BreakBeforeBinaryOperators: true", BreakBeforeBinaryOperators,
              FormatStyle::BOS_All);

  Style.AlignAfterOpenBracket = FormatStyle::BAS_AlwaysBreak;
  CHECK_PARSE("AlignAfterOpenBracket: Align", AlignAfterOpenBracket,
              FormatStyle::BAS_Align);
  CHECK_PARSE("AlignAfterOpenBracket: DontAlign", AlignAfterOpenBracket,
              FormatStyle::BAS_DontAlign);
  CHECK_PARSE("AlignAfterOpenBracket: AlwaysBreak", AlignAfterOpenBracket,
              FormatStyle::BAS_AlwaysBreak);
  // For backward compatibility:
  CHECK_PARSE("AlignAfterOpenBracket: false", AlignAfterOpenBracket,
              FormatStyle::BAS_DontAlign);
  CHECK_PARSE("AlignAfterOpenBracket: true", AlignAfterOpenBracket,
              FormatStyle::BAS_Align);

  Style.UseTab = FormatStyle::UT_ForIndentation;
  CHECK_PARSE("UseTab: Never", UseTab, FormatStyle::UT_Never);
  CHECK_PARSE("UseTab: ForIndentation", UseTab, FormatStyle::UT_ForIndentation);
  CHECK_PARSE("UseTab: Always", UseTab, FormatStyle::UT_Always);
  // For backward compatibility:
  CHECK_PARSE("UseTab: false", UseTab, FormatStyle::UT_Never);
  CHECK_PARSE("UseTab: true", UseTab, FormatStyle::UT_Always);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: None",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_None);
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: Inline",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_Inline);
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: Empty",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_Empty);
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: All",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_All);
  // For backward compatibility:
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: false",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_None);
  CHECK_PARSE("AllowShortFunctionsOnASingleLine: true",
              AllowShortFunctionsOnASingleLine, FormatStyle::SFS_All);

  Style.SpaceBeforeParens = FormatStyle::SBPO_Always;
  CHECK_PARSE("SpaceBeforeParens: Never", SpaceBeforeParens,
              FormatStyle::SBPO_Never);
  CHECK_PARSE("SpaceBeforeParens: Always", SpaceBeforeParens,
              FormatStyle::SBPO_Always);
  CHECK_PARSE("SpaceBeforeParens: ControlStatements", SpaceBeforeParens,
              FormatStyle::SBPO_ControlStatements);
  // For backward compatibility:
  CHECK_PARSE("SpaceAfterControlStatementKeyword: false", SpaceBeforeParens,
              FormatStyle::SBPO_Never);
  CHECK_PARSE("SpaceAfterControlStatementKeyword: true", SpaceBeforeParens,
              FormatStyle::SBPO_ControlStatements);

  Style.ColumnLimit = 123;
  FormatStyle BaseStyle = getLLVMStyle();
  CHECK_PARSE("BasedOnStyle: LLVM", ColumnLimit, BaseStyle.ColumnLimit);
  CHECK_PARSE("BasedOnStyle: LLVM\nColumnLimit: 1234", ColumnLimit, 1234u);

  Style.BreakBeforeBraces = FormatStyle::BS_Stroustrup;
  CHECK_PARSE("BreakBeforeBraces: Attach", BreakBeforeBraces,
              FormatStyle::BS_Attach);
  CHECK_PARSE("BreakBeforeBraces: Linux", BreakBeforeBraces,
              FormatStyle::BS_Linux);
  CHECK_PARSE("BreakBeforeBraces: Mozilla", BreakBeforeBraces,
              FormatStyle::BS_Mozilla);
  CHECK_PARSE("BreakBeforeBraces: Stroustrup", BreakBeforeBraces,
              FormatStyle::BS_Stroustrup);
  CHECK_PARSE("BreakBeforeBraces: Allman", BreakBeforeBraces,
              FormatStyle::BS_Allman);
  CHECK_PARSE("BreakBeforeBraces: GNU", BreakBeforeBraces, FormatStyle::BS_GNU);
  CHECK_PARSE("BreakBeforeBraces: WebKit", BreakBeforeBraces,
              FormatStyle::BS_WebKit);
  CHECK_PARSE("BreakBeforeBraces: Custom", BreakBeforeBraces,
              FormatStyle::BS_Custom);

  Style.AlwaysBreakAfterReturnType = FormatStyle::RTBS_All;
  CHECK_PARSE("AlwaysBreakAfterReturnType: None", AlwaysBreakAfterReturnType,
              FormatStyle::RTBS_None);
  CHECK_PARSE("AlwaysBreakAfterReturnType: All", AlwaysBreakAfterReturnType,
              FormatStyle::RTBS_All);
  CHECK_PARSE("AlwaysBreakAfterReturnType: TopLevel",
              AlwaysBreakAfterReturnType, FormatStyle::RTBS_TopLevel);
  CHECK_PARSE("AlwaysBreakAfterReturnType: AllDefinitions",
              AlwaysBreakAfterReturnType, FormatStyle::RTBS_AllDefinitions);
  CHECK_PARSE("AlwaysBreakAfterReturnType: TopLevelDefinitions",
              AlwaysBreakAfterReturnType,
              FormatStyle::RTBS_TopLevelDefinitions);

  Style.AlwaysBreakAfterDefinitionReturnType = FormatStyle::DRTBS_All;
  CHECK_PARSE("AlwaysBreakAfterDefinitionReturnType: None",
              AlwaysBreakAfterDefinitionReturnType, FormatStyle::DRTBS_None);
  CHECK_PARSE("AlwaysBreakAfterDefinitionReturnType: All",
              AlwaysBreakAfterDefinitionReturnType, FormatStyle::DRTBS_All);
  CHECK_PARSE("AlwaysBreakAfterDefinitionReturnType: TopLevel",
              AlwaysBreakAfterDefinitionReturnType,
              FormatStyle::DRTBS_TopLevel);

  Style.NamespaceIndentation = FormatStyle::NI_All;
  CHECK_PARSE("NamespaceIndentation: None", NamespaceIndentation,
              FormatStyle::NI_None);
  CHECK_PARSE("NamespaceIndentation: Inner", NamespaceIndentation,
              FormatStyle::NI_Inner);
  CHECK_PARSE("NamespaceIndentation: All", NamespaceIndentation,
              FormatStyle::NI_All);

  // FIXME: This is required because parsing a configuration simply overwrites
  // the first N elements of the list instead of resetting it.
  Style.ForEachMacros.clear();
  std::vector<std::string> BoostForeach;
  BoostForeach.push_back("BOOST_FOREACH");
  CHECK_PARSE("ForEachMacros: [BOOST_FOREACH]", ForEachMacros, BoostForeach);
  std::vector<std::string> BoostAndQForeach;
  BoostAndQForeach.push_back("BOOST_FOREACH");
  BoostAndQForeach.push_back("Q_FOREACH");
  CHECK_PARSE("ForEachMacros: [BOOST_FOREACH, Q_FOREACH]", ForEachMacros,
              BoostAndQForeach);

  Style.IncludeCategories.clear();
  std::vector<FormatStyle::IncludeCategory> ExpectedCategories = {{"abc/.*", 2},
                                                                  {".*", 1}};
  CHECK_PARSE("IncludeCategories:\n"
              "  - Regex: abc/.*\n"
              "    Priority: 2\n"
              "  - Regex: .*\n"
              "    Priority: 1",
              IncludeCategories, ExpectedCategories);
}

TEST_F(FormatTest, ParsesConfigurationWithLanguages) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("Language: Cpp\n"
              "IndentWidth: 12",
              IndentWidth, 12u);
  EXPECT_EQ(parseConfiguration("Language: JavaScript\n"
                               "IndentWidth: 34",
                               &Style),
            ParseError::Unsuitable);
  EXPECT_EQ(12u, Style.IndentWidth);
  CHECK_PARSE("IndentWidth: 56", IndentWidth, 56u);
  EXPECT_EQ(FormatStyle::LK_Cpp, Style.Language);

  Style.Language = FormatStyle::LK_JavaScript;
  CHECK_PARSE("Language: JavaScript\n"
              "IndentWidth: 12",
              IndentWidth, 12u);
  CHECK_PARSE("IndentWidth: 23", IndentWidth, 23u);
  EXPECT_EQ(parseConfiguration("Language: Cpp\n"
                               "IndentWidth: 34",
                               &Style),
            ParseError::Unsuitable);
  EXPECT_EQ(23u, Style.IndentWidth);
  CHECK_PARSE("IndentWidth: 56", IndentWidth, 56u);
  EXPECT_EQ(FormatStyle::LK_JavaScript, Style.Language);

  CHECK_PARSE("BasedOnStyle: LLVM\n"
              "IndentWidth: 67",
              IndentWidth, 67u);

  CHECK_PARSE("---\n"
              "Language: JavaScript\n"
              "IndentWidth: 12\n"
              "---\n"
              "Language: Cpp\n"
              "IndentWidth: 34\n"
              "...\n",
              IndentWidth, 12u);

  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("---\n"
              "Language: JavaScript\n"
              "IndentWidth: 12\n"
              "---\n"
              "Language: Cpp\n"
              "IndentWidth: 34\n"
              "...\n",
              IndentWidth, 34u);
  CHECK_PARSE("---\n"
              "IndentWidth: 78\n"
              "---\n"
              "Language: JavaScript\n"
              "IndentWidth: 56\n"
              "...\n",
              IndentWidth, 78u);

  Style.ColumnLimit = 123;
  Style.IndentWidth = 234;
  Style.BreakBeforeBraces = FormatStyle::BS_Linux;
  Style.TabWidth = 345;
  EXPECT_FALSE(parseConfiguration("---\n"
                                  "IndentWidth: 456\n"
                                  "BreakBeforeBraces: Allman\n"
                                  "---\n"
                                  "Language: JavaScript\n"
                                  "IndentWidth: 111\n"
                                  "TabWidth: 111\n"
                                  "---\n"
                                  "Language: Cpp\n"
                                  "BreakBeforeBraces: Stroustrup\n"
                                  "TabWidth: 789\n"
                                  "...\n",
                                  &Style));
  EXPECT_EQ(123u, Style.ColumnLimit);
  EXPECT_EQ(456u, Style.IndentWidth);
  EXPECT_EQ(FormatStyle::BS_Stroustrup, Style.BreakBeforeBraces);
  EXPECT_EQ(789u, Style.TabWidth);

  EXPECT_EQ(parseConfiguration("---\n"
                               "Language: JavaScript\n"
                               "IndentWidth: 56\n"
                               "---\n"
                               "IndentWidth: 78\n"
                               "...\n",
                               &Style),
            ParseError::Error);
  EXPECT_EQ(parseConfiguration("---\n"
                               "Language: JavaScript\n"
                               "IndentWidth: 56\n"
                               "---\n"
                               "Language: JavaScript\n"
                               "IndentWidth: 78\n"
                               "...\n",
                               &Style),
            ParseError::Error);

  EXPECT_EQ(FormatStyle::LK_Cpp, Style.Language);
}

#undef CHECK_PARSE

TEST_F(FormatTest, UsesLanguageForBasedOnStyle) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_JavaScript;
  Style.BreakBeforeTernaryOperators = true;
  EXPECT_EQ(0, parseConfiguration("BasedOnStyle: Google", &Style).value());
  EXPECT_FALSE(Style.BreakBeforeTernaryOperators);

  Style.BreakBeforeTernaryOperators = true;
  EXPECT_EQ(0, parseConfiguration("---\n"
                                  "BasedOnStyle: Google\n"
                                  "---\n"
                                  "Language: JavaScript\n"
                                  "IndentWidth: 76\n"
                                  "...\n",
                                  &Style)
                   .value());
  EXPECT_FALSE(Style.BreakBeforeTernaryOperators);
  EXPECT_EQ(76u, Style.IndentWidth);
  EXPECT_EQ(FormatStyle::LK_JavaScript, Style.Language);
}

TEST_F(FormatTest, ConfigurationRoundTripTest) {
  FormatStyle Style = getLLVMStyle();
  std::string YAML = configurationAsText(Style);
  FormatStyle ParsedStyle = {};
  ParsedStyle.Language = FormatStyle::LK_Cpp;
  EXPECT_EQ(0, parseConfiguration(YAML, &ParsedStyle).value());
  EXPECT_EQ(Style, ParsedStyle);
}

TEST_F(FormatTest, WorksFor8bitEncodings) {
  EXPECT_EQ("\"\xce\xe4\xed\xe0\xe6\xe4\xfb \xe2 \"\n"
            "\"\xf1\xf2\xf3\xe4\xb8\xed\xf3\xfe \"\n"
            "\"\xe7\xe8\xec\xed\xfe\xfe \"\n"
            "\"\xef\xee\xf0\xf3...\"",
            format("\"\xce\xe4\xed\xe0\xe6\xe4\xfb \xe2 "
                   "\xf1\xf2\xf3\xe4\xb8\xed\xf3\xfe \xe7\xe8\xec\xed\xfe\xfe "
                   "\xef\xee\xf0\xf3...\"",
                   getLLVMStyleWithColumns(12)));
}

TEST_F(FormatTest, HandlesUTF8BOM) {
  EXPECT_EQ("\xef\xbb\xbf", format("\xef\xbb\xbf"));
  EXPECT_EQ("\xef\xbb\xbf#include <iostream>",
            format("\xef\xbb\xbf#include <iostream>"));
  EXPECT_EQ("\xef\xbb\xbf\n#include <iostream>",
            format("\xef\xbb\xbf\n#include <iostream>"));
}

// FIXME: Encode Cyrillic and CJK characters below to appease MS compilers.
#if !defined(_MSC_VER)

TEST_F(FormatTest, CountsUTF8CharactersProperly) {
  verifyFormat("\"Однажды в студёную зимнюю пору...\"",
               getLLVMStyleWithColumns(35));
  verifyFormat("\"一 二 三 四 五 六 七 八 九 十\"",
               getLLVMStyleWithColumns(31));
  verifyFormat("// Однажды в студёную зимнюю пору...",
               getLLVMStyleWithColumns(36));
  verifyFormat("// 一 二 三 四 五 六 七 八 九 十", getLLVMStyleWithColumns(32));
  verifyFormat("/* Однажды в студёную зимнюю пору... */",
               getLLVMStyleWithColumns(39));
  verifyFormat("/* 一 二 三 四 五 六 七 八 九 十 */",
               getLLVMStyleWithColumns(35));
}

TEST_F(FormatTest, SplitsUTF8Strings) {
  // Non-printable characters' width is currently considered to be the length in
  // bytes in UTF8. The characters can be displayed in very different manner
  // (zero-width, single width with a substitution glyph, expanded to their code
  // (e.g. "<8d>"), so there's no single correct way to handle them.
  EXPECT_EQ("\"aaaaÄ\"\n"
            "\"\xc2\x8d\";",
            format("\"aaaaÄ\xc2\x8d\";", getLLVMStyleWithColumns(10)));
  EXPECT_EQ("\"aaaaaaaÄ\"\n"
            "\"\xc2\x8d\";",
            format("\"aaaaaaaÄ\xc2\x8d\";", getLLVMStyleWithColumns(10)));
  EXPECT_EQ("\"Однажды, в \"\n"
            "\"студёную \"\n"
            "\"зимнюю \"\n"
            "\"пору,\"",
            format("\"Однажды, в студёную зимнюю пору,\"",
                   getLLVMStyleWithColumns(13)));
  EXPECT_EQ(
      "\"一 二 三 \"\n"
      "\"四 五六 \"\n"
      "\"七 八 九 \"\n"
      "\"十\"",
      format("\"一 二 三 四 五六 七 八 九 十\"", getLLVMStyleWithColumns(11)));
  EXPECT_EQ("\"一\t二 \"\n"
            "\"\t三 \"\n"
            "\"四 五\t六 \"\n"
            "\"\t七 \"\n"
            "\"八九十\tqq\"",
            format("\"一\t二 \t三 四 五\t六 \t七 八九十\tqq\"",
                   getLLVMStyleWithColumns(11)));

  // UTF8 character in an escape sequence.
  EXPECT_EQ("\"aaaaaa\"\n"
            "\"\\\xC2\x8D\"",
            format("\"aaaaaa\\\xC2\x8D\"", getLLVMStyleWithColumns(10)));
}

TEST_F(FormatTest, HandlesDoubleWidthCharsInMultiLineStrings) {
  EXPECT_EQ("const char *sssss =\n"
            "    \"一二三四五六七八\\\n"
            " 九 十\";",
            format("const char *sssss = \"一二三四五六七八\\\n"
                   " 九 十\";",
                   getLLVMStyleWithColumns(30)));
}

TEST_F(FormatTest, SplitsUTF8LineComments) {
  EXPECT_EQ("// aaaaÄ\xc2\x8d",
            format("// aaaaÄ\xc2\x8d", getLLVMStyleWithColumns(10)));
  EXPECT_EQ("// Я из лесу\n"
            "// вышел; был\n"
            "// сильный\n"
            "// мороз.",
            format("// Я из лесу вышел; был сильный мороз.",
                   getLLVMStyleWithColumns(13)));
  EXPECT_EQ("// 一二三\n"
            "// 四五六七\n"
            "// 八  九\n"
            "// 十",
            format("// 一二三 四五六七 八  九 十", getLLVMStyleWithColumns(9)));
}

TEST_F(FormatTest, SplitsUTF8BlockComments) {
  EXPECT_EQ("/* Гляжу,\n"
            " * поднимается\n"
            " * медленно в\n"
            " * гору\n"
            " * Лошадка,\n"
            " * везущая\n"
            " * хворосту\n"
            " * воз. */",
            format("/* Гляжу, поднимается медленно в гору\n"
                   " * Лошадка, везущая хворосту воз. */",
                   getLLVMStyleWithColumns(13)));
  EXPECT_EQ(
      "/* 一二三\n"
      " * 四五六七\n"
      " * 八  九\n"
      " * 十  */",
      format("/* 一二三 四五六七 八  九 十  */", getLLVMStyleWithColumns(9)));
  EXPECT_EQ("/* 𝓣𝓮𝓼𝓽 𝔣𝔬𝔲𝔯\n"
            " * 𝕓𝕪𝕥𝕖\n"
            " * 𝖀𝕿𝕱-𝟠 */",
            format("/* 𝓣𝓮𝓼𝓽 𝔣𝔬𝔲𝔯 𝕓𝕪𝕥𝕖 𝖀𝕿𝕱-𝟠 */", getLLVMStyleWithColumns(12)));
}

#endif // _MSC_VER

TEST_F(FormatTest, ConstructorInitializerIndentWidth) {
  FormatStyle Style = getLLVMStyle();

  Style.ConstructorInitializerIndentWidth = 4;
  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
      Style);

  Style.ConstructorInitializerIndentWidth = 2;
  verifyFormat(
      "SomeClass::Constructor()\n"
      "  : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "    aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
      Style);

  Style.ConstructorInitializerIndentWidth = 0;
  verifyFormat(
      "SomeClass::Constructor()\n"
      ": aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "  aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {}",
      Style);
}

TEST_F(FormatTest, BreakConstructorInitializersBeforeComma) {
  FormatStyle Style = getLLVMStyle();
  Style.BreakConstructorInitializersBeforeComma = true;
  Style.ConstructorInitializerIndentWidth = 4;
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "    , b(b)\n"
               "    , c(c) {}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a) {}",
               Style);

  Style.ColumnLimit = 0;
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a) {}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "    , b(b)\n"
               "    , c(c) {}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a) {\n"
               "  foo();\n"
               "  bar();\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "    , b(b)\n"
               "    , c(c) {\n}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a) {\n}",
               Style);

  Style.ColumnLimit = 80;
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  Style.ConstructorInitializerIndentWidth = 2;
  verifyFormat("SomeClass::Constructor()\n"
               "  : a(a)\n"
               "  , b(b)\n"
               "  , c(c) {}",
               Style);

  Style.ConstructorInitializerIndentWidth = 0;
  verifyFormat("SomeClass::Constructor()\n"
               ": a(a)\n"
               ", b(b)\n"
               ", c(c) {}",
               Style);

  Style.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  Style.ConstructorInitializerIndentWidth = 4;
  verifyFormat("SomeClass::Constructor() : aaaaaaaa(aaaaaaaa) {}", Style);
  verifyFormat(
      "SomeClass::Constructor() : aaaaa(aaaaa), aaaaa(aaaaa), aaaaa(aaaaa)\n",
      Style);
  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaa(aaaaaaaa), aaaaaaaa(aaaaaaaa), aaaaaaaa(aaaaaaaa) {}",
      Style);
  Style.ConstructorInitializerIndentWidth = 4;
  Style.ColumnLimit = 60;
  verifyFormat("SomeClass::Constructor()\n"
               "    : aaaaaaaa(aaaaaaaa)\n"
               "    , aaaaaaaa(aaaaaaaa)\n"
               "    , aaaaaaaa(aaaaaaaa) {}",
               Style);
}

TEST_F(FormatTest, Destructors) {
  verifyFormat("void F(int &i) { i.~int(); }");
  verifyFormat("void F(int &i) { i->~int(); }");
}

TEST_F(FormatTest, FormatsWithWebKitStyle) {
  FormatStyle Style = getWebKitStyle();

  // Don't indent in outer namespaces.
  verifyFormat("namespace outer {\n"
               "int i;\n"
               "namespace inner {\n"
               "    int i;\n"
               "} // namespace inner\n"
               "} // namespace outer\n"
               "namespace other_outer {\n"
               "int i;\n"
               "}",
               Style);

  // Don't indent case labels.
  verifyFormat("switch (variable) {\n"
               "case 1:\n"
               "case 2:\n"
               "    doSomething();\n"
               "    break;\n"
               "default:\n"
               "    ++variable;\n"
               "}",
               Style);

  // Wrap before binary operators.
  EXPECT_EQ("void f()\n"
            "{\n"
            "    if (aaaaaaaaaaaaaaaa\n"
            "        && bbbbbbbbbbbbbbbbbbbbbbbb\n"
            "        && (cccccccccccccccccccccccccc || dddddddddddddddddddd))\n"
            "        return;\n"
            "}",
            format("void f() {\n"
                   "if (aaaaaaaaaaaaaaaa\n"
                   "&& bbbbbbbbbbbbbbbbbbbbbbbb\n"
                   "&& (cccccccccccccccccccccccccc || dddddddddddddddddddd))\n"
                   "return;\n"
                   "}",
                   Style));

  // Allow functions on a single line.
  verifyFormat("void f() { return; }", Style);

  // Constructor initializers are formatted one per line with the "," on the
  // new line.
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    , aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaa, // break\n"
               "          aaaaaaaaaaaaaa)\n"
               "    , aaaaaaaaaaaaaaaaaaaaaaa()\n"
               "{\n"
               "}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "{\n"
               "}",
               Style);
  EXPECT_EQ("SomeClass::Constructor()\n"
            "    : a(a)\n"
            "{\n"
            "}",
            format("SomeClass::Constructor():a(a){}", Style));
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "    , b(b)\n"
               "    , c(c)\n"
               "{\n"
               "}",
               Style);
  verifyFormat("SomeClass::Constructor()\n"
               "    : a(a)\n"
               "{\n"
               "    foo();\n"
               "    bar();\n"
               "}",
               Style);

  // Access specifiers should be aligned left.
  verifyFormat("class C {\n"
               "public:\n"
               "    int i;\n"
               "};",
               Style);

  // Do not align comments.
  verifyFormat("int a; // Do not\n"
               "double b; // align comments.",
               Style);

  // Do not align operands.
  EXPECT_EQ("ASSERT(aaaa\n"
            "    || bbbb);",
            format("ASSERT ( aaaa\n||bbbb);", Style));

  // Accept input's line breaks.
  EXPECT_EQ("if (aaaaaaaaaaaaaaa\n"
            "    || bbbbbbbbbbbbbbb) {\n"
            "    i++;\n"
            "}",
            format("if (aaaaaaaaaaaaaaa\n"
                   "|| bbbbbbbbbbbbbbb) { i++; }",
                   Style));
  EXPECT_EQ("if (aaaaaaaaaaaaaaa || bbbbbbbbbbbbbbb) {\n"
            "    i++;\n"
            "}",
            format("if (aaaaaaaaaaaaaaa || bbbbbbbbbbbbbbb) { i++; }", Style));

  // Don't automatically break all macro definitions (llvm.org/PR17842).
  verifyFormat("#define aNumber 10", Style);
  // However, generally keep the line breaks that the user authored.
  EXPECT_EQ("#define aNumber \\\n"
            "    10",
            format("#define aNumber \\\n"
                   " 10",
                   Style));

  // Keep empty and one-element array literals on a single line.
  EXPECT_EQ("NSArray* a = [[NSArray alloc] initWithArray:@[]\n"
            "                                  copyItems:YES];",
            format("NSArray*a=[[NSArray alloc] initWithArray:@[]\n"
                   "copyItems:YES];",
                   Style));
  EXPECT_EQ("NSArray* a = [[NSArray alloc] initWithArray:@[ @\"a\" ]\n"
            "                                  copyItems:YES];",
            format("NSArray*a=[[NSArray alloc]initWithArray:@[ @\"a\" ]\n"
                   "             copyItems:YES];",
                   Style));
  // FIXME: This does not seem right, there should be more indentation before
  // the array literal's entries. Nested blocks have the same problem.
  EXPECT_EQ("NSArray* a = [[NSArray alloc] initWithArray:@[\n"
            "    @\"a\",\n"
            "    @\"a\"\n"
            "]\n"
            "                                  copyItems:YES];",
            format("NSArray* a = [[NSArray alloc] initWithArray:@[\n"
                   "     @\"a\",\n"
                   "     @\"a\"\n"
                   "     ]\n"
                   "       copyItems:YES];",
                   Style));
  EXPECT_EQ(
      "NSArray* a = [[NSArray alloc] initWithArray:@[ @\"a\", @\"a\" ]\n"
      "                                  copyItems:YES];",
      format("NSArray* a = [[NSArray alloc] initWithArray:@[ @\"a\", @\"a\" ]\n"
             "   copyItems:YES];",
             Style));

  verifyFormat("[self.a b:c c:d];", Style);
  EXPECT_EQ("[self.a b:c\n"
            "        c:d];",
            format("[self.a b:c\n"
                   "c:d];",
                   Style));
}

TEST_F(FormatTest, FormatsLambdas) {
  verifyFormat("int c = [b]() mutable { return [&b] { return b++; }(); }();\n");
  verifyFormat("int c = [&] { [=] { return b++; }(); }();\n");
  verifyFormat("int c = [&, &a, a] { [=, c, &d] { return b++; }(); }();\n");
  verifyFormat("int c = [&a, &a, a] { [=, a, b, &c] { return b++; }(); }();\n");
  verifyFormat("auto c = {[&a, &a, a] { [=, a, b, &c] { return b++; }(); }}\n");
  verifyFormat("auto c = {[&a, &a, a] { [=, a, b, &c] {}(); }}\n");
  verifyFormat("void f() {\n"
               "  other(x.begin(), x.end(), [&](int, int) { return 1; });\n"
               "}\n");
  verifyFormat("void f() {\n"
               "  other(x.begin(), //\n"
               "        x.end(),   //\n"
               "        [&](int, int) { return 1; });\n"
               "}\n");
  verifyFormat("SomeFunction([]() { // A cool function...\n"
               "  return 43;\n"
               "});");
  EXPECT_EQ("SomeFunction([]() {\n"
            "#define A a\n"
            "  return 43;\n"
            "});",
            format("SomeFunction([](){\n"
                   "#define A a\n"
                   "return 43;\n"
                   "});"));
  verifyFormat("void f() {\n"
               "  SomeFunction([](decltype(x), A *a) {});\n"
               "}");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    [](const aaaaaaaaaa &a) { return a; });");
  verifyFormat("string abc = SomeFunction(aaaaaaaaaaaaa, aaaaa, []() {\n"
               "  SomeOtherFunctioooooooooooooooooooooooooon();\n"
               "});");
  verifyFormat("Constructor()\n"
               "    : Field([] { // comment\n"
               "        int i;\n"
               "      }) {}");
  verifyFormat("auto my_lambda = [](const string &some_parameter) {\n"
               "  return some_parameter.size();\n"
               "};");
  verifyFormat("std::function<std::string(const std::string &)> my_lambda =\n"
               "    [](const string &s) { return s; };");
  verifyFormat("int i = aaaaaa ? 1 //\n"
               "               : [] {\n"
               "                   return 2; //\n"
               "                 }();");
  verifyFormat("llvm::errs() << \"number of twos is \"\n"
               "             << std::count_if(v.begin(), v.end(), [](int x) {\n"
               "                  return x == 2; // force break\n"
               "                });");
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa([=](\n"
               "    int iiiiiiiiiiii) {\n"
               "  return aaaaaaaaaaaaaaaaaaaaaaa != aaaaaaaaaaaaaaaaaaaaaaa;\n"
               "});",
               getLLVMStyleWithColumns(60));
  verifyFormat("SomeFunction({[&] {\n"
               "                // comment\n"
               "              },\n"
               "              [&] {\n"
               "                // comment\n"
               "              }});");
  verifyFormat("SomeFunction({[&] {\n"
               "  // comment\n"
               "}});");
  verifyFormat("virtual aaaaaaaaaaaaaaaa(std::function<bool()> bbbbbbbbbbbb =\n"
               "                             [&]() { return true; },\n"
               "                         aaaaa aaaaaaaaa);");

  // Lambdas with return types.
  verifyFormat("int c = []() -> int { return 2; }();\n");
  verifyFormat("int c = []() -> int * { return 2; }();\n");
  verifyFormat("int c = []() -> vector<int> { return {2}; }();\n");
  verifyFormat("Foo([]() -> std::vector<int> { return {2}; }());");
  verifyGoogleFormat("auto a = [&b, c](D* d) -> D* {};");
  verifyGoogleFormat("auto a = [&b, c](D* d) -> pair<D*, D*> {};");
  verifyGoogleFormat("auto a = [&b, c](D* d) -> D& {};");
  verifyGoogleFormat("auto a = [&b, c](D* d) -> const D* {};");
  verifyFormat("[a, a]() -> a<1> {};");
  verifyFormat("auto aaaaaaaa = [](int i, // break for some reason\n"
               "                   int j) -> int {\n"
               "  return ffffffffffffffffffffffffffffffffffffffffffff(i * j);\n"
               "};");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaa(\n"
      "    [](aaaaaaaaaaaaaaaaaaaaaaaaaaa &aaa) -> aaaaaaaaaaaaaaaa {\n"
      "      return aaaaaaaaaaaaaaaaa;\n"
      "    });",
      getLLVMStyleWithColumns(70));

  // Multiple lambdas in the same parentheses change indentation rules.
  verifyFormat("SomeFunction(\n"
               "    []() {\n"
               "      int i = 42;\n"
               "      return i;\n"
               "    },\n"
               "    []() {\n"
               "      int j = 43;\n"
               "      return j;\n"
               "    });");

  // More complex introducers.
  verifyFormat("return [i, args...] {};");

  // Not lambdas.
  verifyFormat("constexpr char hello[]{\"hello\"};");
  verifyFormat("double &operator[](int i) { return 0; }\n"
               "int i;");
  verifyFormat("std::unique_ptr<int[]> foo() {}");
  verifyFormat("int i = a[a][a]->f();");
  verifyFormat("int i = (*b)[a]->f();");

  // Other corner cases.
  verifyFormat("void f() {\n"
               "  bar([]() {} // Did not respect SpacesBeforeTrailingComments\n"
               "      );\n"
               "}");

  // Lambdas created through weird macros.
  verifyFormat("void f() {\n"
               "  MACRO((const AA &a) { return 1; });\n"
               "}");

  verifyFormat("if (blah_blah(whatever, whatever, [] {\n"
               "      doo_dah();\n"
               "      doo_dah();\n"
               "    })) {\n"
               "}");
  verifyFormat("auto lambda = []() {\n"
               "  int a = 2\n"
               "#if A\n"
               "          + 2\n"
               "#endif\n"
               "      ;\n"
               "};");
}

TEST_F(FormatTest, FormatsBlocks) {
  FormatStyle ShortBlocks = getLLVMStyle();
  ShortBlocks.AllowShortBlocksOnASingleLine = true;
  verifyFormat("int (^Block)(int, int);", ShortBlocks);
  verifyFormat("int (^Block1)(int, int) = ^(int i, int j)", ShortBlocks);
  verifyFormat("void (^block)(int) = ^(id test) { int i; };", ShortBlocks);
  verifyFormat("void (^block)(int) = ^(int test) { int i; };", ShortBlocks);
  verifyFormat("void (^block)(int) = ^id(int test) { int i; };", ShortBlocks);
  verifyFormat("void (^block)(int) = ^int(int test) { int i; };", ShortBlocks);

  verifyFormat("foo(^{ bar(); });", ShortBlocks);
  verifyFormat("foo(a, ^{ bar(); });", ShortBlocks);
  verifyFormat("{ void (^block)(Object *x); }", ShortBlocks);

  verifyFormat("[operation setCompletionBlock:^{\n"
               "  [self onOperationDone];\n"
               "}];");
  verifyFormat("int i = {[operation setCompletionBlock:^{\n"
               "  [self onOperationDone];\n"
               "}]};");
  verifyFormat("[operation setCompletionBlock:^(int *i) {\n"
               "  f();\n"
               "}];");
  verifyFormat("int a = [operation block:^int(int *i) {\n"
               "  return 1;\n"
               "}];");
  verifyFormat("[myObject doSomethingWith:arg1\n"
               "                      aaa:^int(int *a) {\n"
               "                        return 1;\n"
               "                      }\n"
               "                      bbb:f(a * bbbbbbbb)];");

  verifyFormat("[operation setCompletionBlock:^{\n"
               "  [self.delegate newDataAvailable];\n"
               "}];",
               getLLVMStyleWithColumns(60));
  verifyFormat("dispatch_async(_fileIOQueue, ^{\n"
               "  NSString *path = [self sessionFilePath];\n"
               "  if (path) {\n"
               "    // ...\n"
               "  }\n"
               "});");
  verifyFormat("[[SessionService sharedService]\n"
               "    loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
               "      if (window) {\n"
               "        [self windowDidLoad:window];\n"
               "      } else {\n"
               "        [self errorLoadingWindow];\n"
               "      }\n"
               "    }];");
  verifyFormat("void (^largeBlock)(void) = ^{\n"
               "  // ...\n"
               "};\n",
               getLLVMStyleWithColumns(40));
  verifyFormat("[[SessionService sharedService]\n"
               "    loadWindowWithCompletionBlock: //\n"
               "        ^(SessionWindow *window) {\n"
               "          if (window) {\n"
               "            [self windowDidLoad:window];\n"
               "          } else {\n"
               "            [self errorLoadingWindow];\n"
               "          }\n"
               "        }];",
               getLLVMStyleWithColumns(60));
  verifyFormat("[myObject doSomethingWith:arg1\n"
               "    firstBlock:^(Foo *a) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }\n"
               "    secondBlock:^(Bar *b) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }\n"
               "    thirdBlock:^Foo(Bar *b) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }];");
  verifyFormat("[myObject doSomethingWith:arg1\n"
               "               firstBlock:-1\n"
               "              secondBlock:^(Bar *b) {\n"
               "                // ...\n"
               "                int i;\n"
               "              }];");

  verifyFormat("f(^{\n"
               "  @autoreleasepool {\n"
               "    if (a) {\n"
               "      g();\n"
               "    }\n"
               "  }\n"
               "});");
  verifyFormat("Block b = ^int *(A *a, B *b) {}");

  FormatStyle FourIndent = getLLVMStyle();
  FourIndent.ObjCBlockIndentWidth = 4;
  verifyFormat("[operation setCompletionBlock:^{\n"
               "    [self onOperationDone];\n"
               "}];",
               FourIndent);
}

TEST_F(FormatTest, FormatsBlocksWithZeroColumnWidth) {
  FormatStyle ZeroColumn = getLLVMStyle();
  ZeroColumn.ColumnLimit = 0;

  verifyFormat("[[SessionService sharedService] "
               "loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
               "  if (window) {\n"
               "    [self windowDidLoad:window];\n"
               "  } else {\n"
               "    [self errorLoadingWindow];\n"
               "  }\n"
               "}];",
               ZeroColumn);
  EXPECT_EQ("[[SessionService sharedService]\n"
            "    loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
            "      if (window) {\n"
            "        [self windowDidLoad:window];\n"
            "      } else {\n"
            "        [self errorLoadingWindow];\n"
            "      }\n"
            "    }];",
            format("[[SessionService sharedService]\n"
                   "loadWindowWithCompletionBlock:^(SessionWindow *window) {\n"
                   "                if (window) {\n"
                   "    [self windowDidLoad:window];\n"
                   "  } else {\n"
                   "    [self errorLoadingWindow];\n"
                   "  }\n"
                   "}];",
                   ZeroColumn));
  verifyFormat("[myObject doSomethingWith:arg1\n"
               "    firstBlock:^(Foo *a) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }\n"
               "    secondBlock:^(Bar *b) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }\n"
               "    thirdBlock:^Foo(Bar *b) {\n"
               "      // ...\n"
               "      int i;\n"
               "    }];",
               ZeroColumn);
  verifyFormat("f(^{\n"
               "  @autoreleasepool {\n"
               "    if (a) {\n"
               "      g();\n"
               "    }\n"
               "  }\n"
               "});",
               ZeroColumn);
  verifyFormat("void (^largeBlock)(void) = ^{\n"
               "  // ...\n"
               "};",
               ZeroColumn);

  ZeroColumn.AllowShortBlocksOnASingleLine = true;
  EXPECT_EQ("void (^largeBlock)(void) = ^{ int i; };",
            format("void   (^largeBlock)(void) = ^{ int   i; };", ZeroColumn));
  ZeroColumn.AllowShortBlocksOnASingleLine = false;
  EXPECT_EQ("void (^largeBlock)(void) = ^{\n"
            "  int i;\n"
            "};",
            format("void   (^largeBlock)(void) = ^{ int   i; };", ZeroColumn));
}

TEST_F(FormatTest, SupportsCRLF) {
  EXPECT_EQ("int a;\r\n"
            "int b;\r\n"
            "int c;\r\n",
            format("int a;\r\n"
                   "  int b;\r\n"
                   "    int c;\r\n",
                   getLLVMStyle()));
  EXPECT_EQ("int a;\r\n"
            "int b;\r\n"
            "int c;\r\n",
            format("int a;\r\n"
                   "  int b;\n"
                   "    int c;\r\n",
                   getLLVMStyle()));
  EXPECT_EQ("int a;\n"
            "int b;\n"
            "int c;\n",
            format("int a;\r\n"
                   "  int b;\n"
                   "    int c;\n",
                   getLLVMStyle()));
  EXPECT_EQ("\"aaaaaaa \"\r\n"
            "\"bbbbbbb\";\r\n",
            format("\"aaaaaaa bbbbbbb\";\r\n", getLLVMStyleWithColumns(10)));
  EXPECT_EQ("#define A \\\r\n"
            "  b;      \\\r\n"
            "  c;      \\\r\n"
            "  d;\r\n",
            format("#define A \\\r\n"
                   "  b; \\\r\n"
                   "  c; d; \r\n",
                   getGoogleStyle()));

  EXPECT_EQ("/*\r\n"
            "multi line block comments\r\n"
            "should not introduce\r\n"
            "an extra carriage return\r\n"
            "*/\r\n",
            format("/*\r\n"
                   "multi line block comments\r\n"
                   "should not introduce\r\n"
                   "an extra carriage return\r\n"
                   "*/\r\n"));
}

TEST_F(FormatTest, MunchSemicolonAfterBlocks) {
  verifyFormat("MY_CLASS(C) {\n"
               "  int i;\n"
               "  int j;\n"
               "};");
}

TEST_F(FormatTest, ConfigurableContinuationIndentWidth) {
  FormatStyle TwoIndent = getLLVMStyleWithColumns(15);
  TwoIndent.ContinuationIndentWidth = 2;

  EXPECT_EQ("int i =\n"
            "  longFunction(\n"
            "    arg);",
            format("int i = longFunction(arg);", TwoIndent));

  FormatStyle SixIndent = getLLVMStyleWithColumns(20);
  SixIndent.ContinuationIndentWidth = 6;

  EXPECT_EQ("int i =\n"
            "      longFunction(\n"
            "            arg);",
            format("int i = longFunction(arg);", SixIndent));
}

TEST_F(FormatTest, SpacesInAngles) {
  FormatStyle Spaces = getLLVMStyle();
  Spaces.SpacesInAngles = true;

  verifyFormat("static_cast< int >(arg);", Spaces);
  verifyFormat("template < typename T0, typename T1 > void f() {}", Spaces);
  verifyFormat("f< int, float >();", Spaces);
  verifyFormat("template <> g() {}", Spaces);
  verifyFormat("template < std::vector< int > > f() {}", Spaces);
  verifyFormat("std::function< void(int, int) > fct;", Spaces);
  verifyFormat("void inFunction() { std::function< void(int, int) > fct; }",
               Spaces);

  Spaces.Standard = FormatStyle::LS_Cpp03;
  Spaces.SpacesInAngles = true;
  verifyFormat("A< A< int > >();", Spaces);

  Spaces.SpacesInAngles = false;
  verifyFormat("A<A<int> >();", Spaces);

  Spaces.Standard = FormatStyle::LS_Cpp11;
  Spaces.SpacesInAngles = true;
  verifyFormat("A< A< int > >();", Spaces);

  Spaces.SpacesInAngles = false;
  verifyFormat("A<A<int>>();", Spaces);
}

TEST_F(FormatTest, TripleAngleBrackets) {
  verifyFormat("f<<<1, 1>>>();");
  verifyFormat("f<<<1, 1, 1, s>>>();");
  verifyFormat("f<<<a, b, c, d>>>();");
  EXPECT_EQ("f<<<1, 1>>>();", format("f <<< 1, 1 >>> ();"));
  verifyFormat("f<param><<<1, 1>>>();");
  verifyFormat("f<1><<<1, 1>>>();");
  EXPECT_EQ("f<param><<<1, 1>>>();", format("f< param > <<< 1, 1 >>> ();"));
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaaaaaaaaaa<<<\n    1, 1>>>();");
}

TEST_F(FormatTest, MergeLessLessAtEnd) {
  verifyFormat("<<");
  EXPECT_EQ("< < <", format("\\\n<<<"));
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaallvm::outs() <<");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               "aaaallvm::outs()\n    <<");
}

TEST_F(FormatTest, HandleUnbalancedImplicitBracesAcrossPPBranches) {
  std::string code = "#if A\n"
                     "#if B\n"
                     "a.\n"
                     "#endif\n"
                     "    a = 1;\n"
                     "#else\n"
                     "#endif\n"
                     "#if C\n"
                     "#else\n"
                     "#endif\n";
  EXPECT_EQ(code, format(code));
}

TEST_F(FormatTest, HandleConflictMarkers) {
  // Git/SVN conflict markers.
  EXPECT_EQ("int a;\n"
            "void f() {\n"
            "  callme(some(parameter1,\n"
            "<<<<<<< text by the vcs\n"
            "              parameter2),\n"
            "||||||| text by the vcs\n"
            "              parameter2),\n"
            "         parameter3,\n"
            "======= text by the vcs\n"
            "              parameter2, parameter3),\n"
            ">>>>>>> text by the vcs\n"
            "         otherparameter);\n",
            format("int a;\n"
                   "void f() {\n"
                   "  callme(some(parameter1,\n"
                   "<<<<<<< text by the vcs\n"
                   "  parameter2),\n"
                   "||||||| text by the vcs\n"
                   "  parameter2),\n"
                   "  parameter3,\n"
                   "======= text by the vcs\n"
                   "  parameter2,\n"
                   "  parameter3),\n"
                   ">>>>>>> text by the vcs\n"
                   "  otherparameter);\n"));

  // Perforce markers.
  EXPECT_EQ("void f() {\n"
            "  function(\n"
            ">>>> text by the vcs\n"
            "      parameter,\n"
            "==== text by the vcs\n"
            "      parameter,\n"
            "==== text by the vcs\n"
            "      parameter,\n"
            "<<<< text by the vcs\n"
            "      parameter);\n",
            format("void f() {\n"
                   "  function(\n"
                   ">>>> text by the vcs\n"
                   "  parameter,\n"
                   "==== text by the vcs\n"
                   "  parameter,\n"
                   "==== text by the vcs\n"
                   "  parameter,\n"
                   "<<<< text by the vcs\n"
                   "  parameter);\n"));

  EXPECT_EQ("<<<<<<<\n"
            "|||||||\n"
            "=======\n"
            ">>>>>>>",
            format("<<<<<<<\n"
                   "|||||||\n"
                   "=======\n"
                   ">>>>>>>"));

  EXPECT_EQ("<<<<<<<\n"
            "|||||||\n"
            "int i;\n"
            "=======\n"
            ">>>>>>>",
            format("<<<<<<<\n"
                   "|||||||\n"
                   "int i;\n"
                   "=======\n"
                   ">>>>>>>"));

  // FIXME: Handle parsing of macros around conflict markers correctly:
  EXPECT_EQ("#define Macro \\\n"
            "<<<<<<<\n"
            "Something \\\n"
            "|||||||\n"
            "Else \\\n"
            "=======\n"
            "Other \\\n"
            ">>>>>>>\n"
            "    End int i;\n",
            format("#define Macro \\\n"
                   "<<<<<<<\n"
                   "  Something \\\n"
                   "|||||||\n"
                   "  Else \\\n"
                   "=======\n"
                   "  Other \\\n"
                   ">>>>>>>\n"
                   "  End\n"
                   "int i;\n"));
}

TEST_F(FormatTest, DisableRegions) {
  EXPECT_EQ("int i;\n"
            "// clang-format off\n"
            "  int j;\n"
            "// clang-format on\n"
            "int k;",
            format(" int  i;\n"
                   "   // clang-format off\n"
                   "  int j;\n"
                   " // clang-format on\n"
                   "   int   k;"));
  EXPECT_EQ("int i;\n"
            "/* clang-format off */\n"
            "  int j;\n"
            "/* clang-format on */\n"
            "int k;",
            format(" int  i;\n"
                   "   /* clang-format off */\n"
                   "  int j;\n"
                   " /* clang-format on */\n"
                   "   int   k;"));
}

TEST_F(FormatTest, DoNotCrashOnInvalidInput) {
  format("? ) =");
  verifyNoCrash("#define a\\\n /**/}");
}

TEST_F(FormatTest, FormatsTableGenCode) {
  FormatStyle Style = getLLVMStyle();
  Style.Language = FormatStyle::LK_TableGen;
  verifyFormat("include \"a.td\"\ninclude \"b.td\"", Style);
}

} // end namespace
} // end namespace format
} // end namespace clang
