//===- unittest/Format/FormatTestTableGen.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {

class FormatTestTableGen : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    tooling::Replacements Replaces = reformat(Style, Code, Ranges);
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  static std::string format(llvm::StringRef Code) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_TableGen);
    Style.ColumnLimit = 60; // To make writing tests easier.
    return format(Code, 0, Code.size(), Style);
  }

  static void verifyFormat(llvm::StringRef Code) {
    EXPECT_EQ(Code.str(), format(Code)) << "Expected code is not stable";
    EXPECT_EQ(Code.str(), format(test::messUp(Code)));
  }

  static void verifyFormat(llvm::StringRef Result, llvm::StringRef MessedUp) {
    EXPECT_EQ(Result, format(MessedUp));
  }
};

TEST_F(FormatTestTableGen, FormatStringBreak) {
  verifyFormat("include \"OptParser.td\"\n"
               "def flag : Flag<\"--foo\">,\n"
               "           HelpText<\n"
               "               \"This is a very, very, very, very, \"\n"
               "               \"very, very, very, very, very, very, \"\n"
               "               \"very long help string\">;");
}

TEST_F(FormatTestTableGen, NoSpacesInSquareBracketLists) {
  verifyFormat("def flag : Flag<[\"-\", \"--\"], \"foo\">;");
}

TEST_F(FormatTestTableGen, LiteralsAndIdentifiers) {
  verifyFormat("def LiteralAndIdentifiers {\n"
               "  let someInteger = -42;\n"
               "  let 0startID = $TokVarName;\n"
               "  let 0xstartInteger = 0x42;\n"
               "  let someIdentifier = $TokVarName;\n"
               "}\n");
}

TEST_F(FormatTestTableGen, BangOperators) {
  verifyFormat("def BangOperators {\n"
               "  let IfOpe = !if(\n"
               "      !not(!and(!gt(!add(1, 2), !sub(3, 4)), !isa<Ty>($x))),\n"
               "      !foldl(0, !listconcat(!range(5, 6), !range(7, 8)),\n"
               "             total, rec, !add(total, rec.Number)),\n"
               "      !tail(!range(9, 10)));\n"
               "  let ForeachOpe = !foreach(\n"
               "      arg, arglist,\n"
               "      !if(!isa<SomeType>(arg.Type),\n"
               "          !add(!cast<SomeOtherType>(arg).Number, x), arg));\n"
               "  let CondOpe1 = !cond(!eq(size, 1): 1,\n"
               "                       !eq(size, 2): 1,\n"
               "                       !eq(size, 4): 1,\n"
               "                       !eq(size, 8): 1,\n"
               "                       !eq(size, 16): 1,\n"
               "                       true: 0);\n"
               "  let CondOpe2 = !cond(!lt(x, 0): \"negativenegative\",\n"
               "                       !eq(x, 0): \"zerozero\",\n"
               "                       true: \"positivepositive\");\n"
               "  let CondOpe2WithComment = !cond(!lt(x, 0):  // negative\n"
               "                                  \"negativenegative\",\n"
               "                                  !eq(x, 0):  // zero\n"
               "                                  \"zerozero\",\n"
               "                                  true:  // default\n"
               "                                  \"positivepositive\");\n"
               "}\n");
}

TEST_F(FormatTestTableGen, Include) {
  verifyFormat("include \"test/IncludeFile.h\"\n");
}

TEST_F(FormatTestTableGen, Types) {
  verifyFormat("def Types : list<int>, bits<3>, list<list<string>> {}\n");
}

TEST_F(FormatTestTableGen, SimpleValue1_SingleLiterals) {
  verifyFormat("def SimpleValue {\n"
               "  let Integer = 42;\n"
               "  let String = \"some string\";\n"
               "}\n");
}

TEST_F(FormatTestTableGen, SimpleValue1_MultilineString) {
  // test::messUp does not understand multiline TableGen code-literals.
  // We have to give the result and the strings to format manually.
  StringRef DefWithCode =
      "def SimpleValueCode {\n"
      "  let Code =\n"
      "      [{ A TokCode is  nothing more than a multi-line string literal "
      "delimited by \\[{ and }\\]. It  can break across lines and the line "
      "breaks are retained in the string. \n"
      "(https://llvm.org/docs/TableGen/ProgRef.html#grammar-token-TokCode)}];\n"
      "}\n";
  StringRef DefWithCodeMessedUp =
      "def SimpleValueCode {  let  \n"
      "Code=       \n"
      "               [{ A TokCode is  nothing more than a multi-line string "
      "literal "
      "delimited by \\[{ and }\\]. It  can break across lines and the line "
      "breaks are retained in the string. \n"
      "(https://llvm.org/docs/TableGen/ProgRef.html#grammar-token-TokCode)}] \n"
      " ;  \n"
      "   }    \n";
  verifyFormat(DefWithCode, DefWithCodeMessedUp);
}

TEST_F(FormatTestTableGen, SimpleValue2) {
  verifyFormat("def SimpleValue2 {\n"
               "  let True = true;\n"
               "  let False = false;\n"
               "}\n");
}

TEST_F(FormatTestTableGen, SimpleValue3) {
  verifyFormat("class SimpleValue3<int x> { int Question = ?; }\n");
}

TEST_F(FormatTestTableGen, SimpleValue4) {
  verifyFormat("def SimpleValue4 { let ValueList = {1, 2, 3}; }\n");
}

TEST_F(FormatTestTableGen, SimpleValue5) {
  verifyFormat("def SimpleValue5 {\n"
               "  let SquareList = [1, 4, 9];\n"
               "  let SquareListWithType = [\"a\", \"b\", \"c\"]<string>;\n"
               "  let SquareListListWithType = [[1, 2], [3, 4, 5], [7]]<\n"
               "      list<int>>;\n"
               "  let SquareBitsListWithType = [ {1, 2},\n"
               "                                 {3, 4} ]<list<bits<8>>>;\n"
               "}\n");
}

TEST_F(FormatTestTableGen, SimpleValue6) {
  verifyFormat("def SimpleValue6 {\n"
               "  let DAGArgIns = (ins i32:$src1, i32:$src2);\n"
               "  let DAGArgOuts = (outs i32:$dst1, i32:$dst2, i32:$dst3,\n"
               "      i32:$dst4, i32:$dst5, i32:$dst6, i32:$dst7);\n"
               "  let DAGArgOutsWithComment = (outs i32:$dst1,  // dst1\n"
               "      i32:$dst2,                                // dst2\n"
               "      i32:$dst3,                                // dst3\n"
               "      i32:$dst4,                                // dst4\n"
               "      i32:$dst5,                                // dst5\n"
               "      i32:$dst6,                                // dst6\n"
               "      i32:$dst7                                 // dst7\n"
               "  );\n"
               "  let DAGArgBang = (!cast<SomeType>(\"Some\") i32:$src1,\n"
               "      i32:$src2);\n"
               "}\n");
}

TEST_F(FormatTestTableGen, SimpleValue7) {
  verifyFormat("def SimpleValue7 { let Identifier = SimpleValue; }\n");
}

TEST_F(FormatTestTableGen, SimpleValue8) {
  verifyFormat("def SimpleValue8 { let Class = SimpleValue3<3>; }\n");
}

TEST_F(FormatTestTableGen, ValueSuffix) {
  verifyFormat("def SuffixedValues {\n"
               "  let Bit = value{17};\n"
               "  let Bits = value{8...15};\n"
               "  let List = value[1];\n"
               "  let Slice1 = value[1, ];\n"
               "  let Slice2 = value[4...7, 17, 2...3, 4];\n"
               "  let Field = value.field;\n"
               "}\n");
}

TEST_F(FormatTestTableGen, PasteOperator) {
  verifyFormat(
      "def Paste#\"Operator\" { string Paste = \"Paste\"#operator; }\n");

  verifyFormat("def [\"Traring\", \"Paste\"]# {\n"
               "  string X = Traring#;\n"
               "  string Y = List<\"Operator\">#;\n"
               "  string Z = [\"Traring\", \"Paste\", \"Traring\", \"Paste\",\n"
               "              \"Traring\", \"Paste\"]#;\n"
               "}\n");
}

TEST_F(FormatTestTableGen, ClassDefinition) {
  verifyFormat("class Class<int x, int y = 1, string z = \"z\", int w = -1>\n"
               "    : Parent1, Parent2<x, y> {\n"
               "  int Item1 = 1;\n"
               "  int Item2;\n"
               "  code Item3 = [{ Item3 }];\n"
               "  let Item4 = 4;\n"
               "  let Item5{1, 2} = 5;\n"
               "  defvar Item6 = 6;\n"
               "  let Item7 = ?;\n"
               "  assert !ge(x, 0), \"Assert7\";\n"
               "}\n");

  verifyFormat("class FPFormat<bits<3> val> { bits<3> Value = val; }\n");
}

TEST_F(FormatTestTableGen, Def) {
  verifyFormat("def Def : Parent1<Def>, Parent2(defs Def) {\n"
               "  code Item1 = [{ Item1 }];\n"
               "  let Item2{1, 3...4} = {1, 2};\n"
               "  defvar Item3 = (ops nodty:$node1, nodty:$node2);\n"
               "  assert !le(Item2, 0), \"Assert4\";\n"
               "}\n");

  verifyFormat("class FPFormat<bits<3> val> { bits<3> Value = val; }\n");

  verifyFormat("def NotFP : FPFormat<0>;\n");
}

TEST_F(FormatTestTableGen, Let) {
  verifyFormat("let x = 1, y = value<type>,\n"
               "    z = !and(!gt(!add(1, 2), !sub(3, 4)), !isa<Ty>($x)) in {\n"
               "  class Class1 : Parent<x, y> { let Item1 = z; }\n"
               "}\n");
}

TEST_F(FormatTestTableGen, MultiClass) {
  verifyFormat("multiclass Multiclass<int x> {\n"
               "  def : Def1<(item type:$src1),\n"
               "             (!if(!ge(x, 0), !mul(!add(x, 1), !sub(x, 2)),\n"
               "                  !sub(x, 2)))>;\n"
               "  def Def2 : value<type>;\n"
               "  def Def3 : type { let value = 1; }\n"
               "  defm : SomeMultiClass<Def1, Def2>;\n"
               "  defvar DefVar = 6;\n"
               "  foreach i = [1, 2, 3] in {\n"
               "    def : Foreach#i<(item type:$src1),\n"
               "                    (!if(!gt(x, i),\n"
               "                         !mul(!add(x, i), !sub(x, i)),\n"
               "                         !sub(x, !add(i, 1))))>;\n"
               "  }\n"
               "  if !gt(x, 0) then {\n"
               "    def : IfThen<x>;\n"
               "  } else {\n"
               "    def : IfElse<x>;\n"
               "  }\n"
               "  if (dagid x, 0) then {\n"
               "    def : If2<1>;\n"
               "  }\n"
               "  let y = 1, z = 2 in {\n"
               "    multiclass Multiclass2<int x> {\n"
               "      foreach i = [1, 2, 3] in {\n"
               "        def : Foreach#i<(item type:$src1),\n"
               "                        (!if(!gt(z, i),\n"
               "                             !mul(!add(y, i), !sub(x, i)),\n"
               "                             !sub(z, !add(i, 1))))>;\n"
               "      }\n"
               "    }\n"
               "  }\n"
               "}\n");
}

TEST_F(FormatTestTableGen, Defm) {
  verifyFormat("defm : Multiclass<0>;\n");

  verifyFormat("defm Defm1 : Multiclass<1>;\n");
}

TEST_F(FormatTestTableGen, Defset) {
  verifyFormat("defset list<Class> DefSet1 = {\n"
               "  def Def1 : Class<1>;\n"
               "  def Def2 : Class<2>;\n"
               "}\n");
}

TEST_F(FormatTestTableGen, Defvar) {
  verifyFormat("defvar DefVar1 = !cond(!ge(!size(PaseOperator.Paste), 1): 1,\n"
               "                       true: 0);\n");
}

TEST_F(FormatTestTableGen, ForEach) {
  verifyFormat(
      "foreach i = [1, 2, 3] in {\n"
      "  def : Foreach#i<(item type:$src1),\n"
      "                  (!if(!lt(x, i),\n"
      "                       !shl(!mul(x, i), !size(\"string\")),\n"
      "                       !size(!strconcat(\"a\", \"b\", \"c\"))))>;\n"
      "}\n");
}

TEST_F(FormatTestTableGen, Dump) { verifyFormat("dump \"Dump\";\n"); }

TEST_F(FormatTestTableGen, If) {
  verifyFormat("if !gt(x, 0) then {\n"
               "  def : IfThen<x>;\n"
               "} else {\n"
               "  def : IfElse<x>;\n"
               "}\n");
}

TEST_F(FormatTestTableGen, Assert) {
  verifyFormat("assert !le(DefVar1, 0), \"Assert1\";\n");
}

} // namespace format
} // end namespace clang
