//===- unittest/Tooling/FixitTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/FixIt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Testing/TestAST.h"
#include "gtest/gtest.h"

using namespace clang;

using tooling::fixit::getText;
using tooling::fixit::createRemoval;
using tooling::fixit::createReplacement;

namespace {

const CallExpr &onlyCall(ASTContext &Ctx) {
  using namespace ast_matchers;
  auto Calls = match(callExpr().bind(""), Ctx);
  EXPECT_EQ(Calls.size(), 1u);
  return *Calls.front().getNodeAs<CallExpr>("");
}

TEST(FixItTest, getText) {
  TestAST AST("void foo(int x, int y) { foo(x, y); }");
  const CallExpr &CE = onlyCall(AST.context());
  EXPECT_EQ("foo(x, y)", getText(CE, AST.context()));
  EXPECT_EQ("foo(x, y)", getText(CE.getSourceRange(), AST.context()));
  EXPECT_EQ("x", getText(*CE.getArg(0), AST.context()));
  EXPECT_EQ("y", getText(*CE.getArg(1), AST.context()));

  AST = TestAST("#define APPLY(f, x, y) f(x, y)\n"
                "void foo(int x, int y) { APPLY(foo, x, y); }");
  const CallExpr &CE2 = onlyCall(AST.context());
  EXPECT_EQ("APPLY(foo, x, y)", getText(CE2, AST.context()));
}

TEST(FixItTest, getTextWithMacro) {
  TestAST AST("#define F foo(\n"
              "#define OO x, y)\n"
              "void foo(int x, int y) { F OO ; }");
  const CallExpr &CE = onlyCall(AST.context());
  EXPECT_EQ("F OO", getText(CE, AST.context()));
  EXPECT_EQ("", getText(*CE.getArg(0), AST.context()));
  EXPECT_EQ("", getText(*CE.getArg(1), AST.context()));

  AST = TestAST("#define FOO(x, y) (void)x; (void)y; foo(x, y);\n"
                "void foo(int x, int y) { FOO(x,y) }");
  const CallExpr &CE2 = onlyCall(AST.context());
  EXPECT_EQ("", getText(CE2, AST.context()));
  EXPECT_EQ("x", getText(*CE2.getArg(0), AST.context()));
  EXPECT_EQ("y", getText(*CE2.getArg(1), AST.context()));
}

TEST(FixItTest, createRemoval) {
  TestAST AST("void foo(int x, int y) { foo(x, y); }");
  const CallExpr &CE = onlyCall(AST.context());

  FixItHint Hint = createRemoval(CE);
  EXPECT_EQ("foo(x, y)", getText(Hint.RemoveRange.getAsRange(), AST.context()));
  EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
  EXPECT_TRUE(Hint.CodeToInsert.empty());

  FixItHint Hint0 = createRemoval(*CE.getArg(0));
  EXPECT_EQ("x", getText(Hint0.RemoveRange.getAsRange(), AST.context()));
  EXPECT_TRUE(Hint0.InsertFromRange.isInvalid());
  EXPECT_TRUE(Hint0.CodeToInsert.empty());

  FixItHint Hint1 = createRemoval(*CE.getArg(1));
  EXPECT_EQ("y", getText(Hint1.RemoveRange.getAsRange(), AST.context()));
  EXPECT_TRUE(Hint1.InsertFromRange.isInvalid());
  EXPECT_TRUE(Hint1.CodeToInsert.empty());

  AST = TestAST("void foo(int x, int y) { foo(x + y, y + x); }");
  const CallExpr &CE2 = onlyCall(AST.context());
  Hint0 = createRemoval(*CE2.getArg(0));
  EXPECT_EQ("x + y", getText(Hint0.RemoveRange.getAsRange(), AST.context()));

  Hint1 = createRemoval(*CE2.getArg(1));
  EXPECT_EQ("y + x", getText(Hint1.RemoveRange.getAsRange(), AST.context()));
}

TEST(FixItTest, createRemovalWithMacro) {
  TestAST AST("#define FOO foo(1, 1)\n"
              "void foo(int x, int y) { FOO; }");
  const CallExpr &CE = onlyCall(AST.context());
  FixItHint Hint = createRemoval(CE);
  EXPECT_EQ("FOO", getText(Hint.RemoveRange.getAsRange(), AST.context()));
  EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
  EXPECT_TRUE(Hint.CodeToInsert.empty());

  FixItHint Hint0 = createRemoval(*CE.getArg(0));
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:1:17>",
            Hint0.RemoveRange.getBegin().printToString(AST.sourceManager()));
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:1:17>",
            Hint0.RemoveRange.getEnd().printToString(AST.sourceManager()));
  EXPECT_TRUE(Hint0.InsertFromRange.isInvalid());
  EXPECT_TRUE(Hint0.CodeToInsert.empty());

  FixItHint Hint1 = createRemoval(*CE.getArg(1));
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:1:20>",
            Hint1.RemoveRange.getBegin().printToString(AST.sourceManager()));
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:1:20>",
            Hint1.RemoveRange.getEnd().printToString(AST.sourceManager()));
  EXPECT_TRUE(Hint1.InsertFromRange.isInvalid());
  EXPECT_TRUE(Hint1.CodeToInsert.empty());

  AST = TestAST("#define FOO(x, y) (void)x; (void)y; foo(x, y);\n"
                "void foo(int x, int y) { FOO(x,y) }");
  const CallExpr &CE2 = onlyCall(AST.context());
  Hint = createRemoval(CE2);
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:1:37>",
            Hint.RemoveRange.getBegin().printToString(AST.sourceManager()));
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:1:45>",
            Hint.RemoveRange.getEnd().printToString(AST.sourceManager()));
  EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
  EXPECT_TRUE(Hint.CodeToInsert.empty());
}

TEST(FixItTest, createReplacement) {
  for (const char *Code : {
           "void foo(int x, int y) { foo(x, y); }",

           "#define APPLY(f, x, y) f(x, y)\n"
           "void foo(int x, int y) { APPLY(foo, x, y); }",

           "#define APPLY(f, P) f(P)\n"
           "#define PAIR(x, y) x, y\n"
           "void foo(int x, int y) { APPLY(foo, PAIR(x, y)); }\n",
       }) {
    TestAST AST(Code);
    const CallExpr &CE = onlyCall(AST.context());
    const Expr *P0 = CE.getArg(0);
    const Expr *P1 = CE.getArg(1);
    FixItHint Hint0 = createReplacement(*P0, *P1, AST.context());
    FixItHint Hint1 = createReplacement(*P1, *P0, AST.context());

    // Validate Hint0 fields.
    EXPECT_EQ("x", getText(Hint0.RemoveRange.getAsRange(), AST.context()));
    EXPECT_TRUE(Hint0.InsertFromRange.isInvalid());
    EXPECT_EQ(Hint0.CodeToInsert, "y");

    // Validate Hint1 fields.
    EXPECT_EQ("y", getText(Hint1.RemoveRange.getAsRange(), AST.context()));
    EXPECT_TRUE(Hint1.InsertFromRange.isInvalid());
    EXPECT_EQ(Hint1.CodeToInsert, "x");
  }
}

TEST(FixItTest, createReplacementWithMacro) {
  TestAST AST("#define FOO foo(1, 1)\n"
              "void foo(int x, int y) { FOO; }");
  const CallExpr &CE = onlyCall(AST.context());
  FixItHint Hint =
      createReplacement(*CE.getArg(0), *CE.getArg(1), AST.context());
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:1:17>",
            Hint.RemoveRange.getBegin().printToString(AST.sourceManager()));
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:1:17>",
            Hint.RemoveRange.getEnd().printToString(AST.sourceManager()));
  EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
  EXPECT_TRUE(Hint.CodeToInsert.empty());

  AST = TestAST("#define FOO(x, y) (void)x; (void)y; foo(x, y);\n"
                "void foo(int x, int y) { FOO(x,y) }");
  const CallExpr &CE2 = onlyCall(AST.context());
  Hint = createReplacement(*CE2.getArg(0), *CE2.getArg(1), AST.context());
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:2:30>",
            Hint.RemoveRange.getEnd().printToString(AST.sourceManager()));
  EXPECT_EQ("input.mm:2:26 <Spelling=input.mm:2:30>",
            Hint.RemoveRange.getBegin().printToString(AST.sourceManager()));
  EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
  EXPECT_EQ("y", Hint.CodeToInsert);

  AST = TestAST("void foo(int x, int y) { foo(x + y, y + x); }");
  const CallExpr &CE3 = onlyCall(AST.context());
  Hint = createReplacement(*CE3.getArg(0), *CE3.getArg(1), AST.context());
  EXPECT_EQ("x + y", getText(Hint.RemoveRange.getAsRange(), AST.context()));
  EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
  EXPECT_EQ("y + x", Hint.CodeToInsert);
}

} // end anonymous namespace
