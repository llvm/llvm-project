//===- unittest/Tooling/FixitTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Tooling/FixIt.h"

using namespace clang;

using tooling::fixit::getText;
using tooling::fixit::getExtendedText;
using tooling::fixit::createRemoval;
using tooling::fixit::createReplacement;

namespace {

struct CallsVisitor : TestVisitor<CallsVisitor> {
  bool VisitCallExpr(CallExpr *Expr) {
    OnCall(Expr, Context);
    return true;
  }

  std::function<void(CallExpr *, ASTContext *Context)> OnCall;
};

std::string LocationToString(SourceLocation Loc, ASTContext *Context) {
  return Loc.printToString(Context->getSourceManager());
}

TEST(FixItTest, getText) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("foo(x, y)", getText(*CE, *Context));
    EXPECT_EQ("foo(x, y)", getText(CE->getSourceRange(), *Context));

    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    EXPECT_EQ("x", getText(*P0, *Context));
    EXPECT_EQ("y", getText(*P1, *Context));
  };
  Visitor.runOver("void foo(int x, int y) { foo(x, y); }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("APPLY(foo, x, y)", getText(*CE, *Context));
  };
  Visitor.runOver("#define APPLY(f, x, y) f(x, y)\n"
                  "void foo(int x, int y) { APPLY(foo, x, y); }");
}

TEST(FixItTest, getTextWithMacro) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("F OO", getText(*CE, *Context));
    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    EXPECT_EQ("", getText(*P0, *Context));
    EXPECT_EQ("", getText(*P1, *Context));
  };
  Visitor.runOver("#define F foo(\n"
                  "#define OO x, y)\n"
                  "void foo(int x, int y) { F OO ; }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("", getText(*CE, *Context));
    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    EXPECT_EQ("x", getText(*P0, *Context));
    EXPECT_EQ("y", getText(*P1, *Context));
  };
  Visitor.runOver("#define FOO(x, y) (void)x; (void)y; foo(x, y);\n"
                  "void foo(int x, int y) { FOO(x,y) }");
}

TEST(FixItTest, getExtendedText) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("foo(x, y);",
              getExtendedText(*CE, tok::TokenKind::semi, *Context));

    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    EXPECT_EQ("x", getExtendedText(*P0, tok::TokenKind::semi, *Context));
    EXPECT_EQ("x,", getExtendedText(*P0, tok::TokenKind::comma, *Context));
    EXPECT_EQ("y", getExtendedText(*P1, tok::TokenKind::semi, *Context));
  };
  Visitor.runOver("void foo(int x, int y) { foo(x, y); }");
  Visitor.runOver("void foo(int x, int y) { if (true) foo(x, y); }");
  Visitor.runOver("int foo(int x, int y) { if (true) return 3 + foo(x, y); }");
  Visitor.runOver("void foo(int x, int y) { for (foo(x, y);;) ++x; }");
  Visitor.runOver(
      "bool foo(int x, int y) { for (;foo(x, y);) x = 1; return true; }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("foo()", getExtendedText(*CE, tok::TokenKind::semi, *Context));
  };
  Visitor.runOver("bool foo() { if (foo()) return true; return false; }");
  Visitor.runOver("void foo() { int x; for (;; foo()) ++x; }");
  Visitor.runOver("int foo() { return foo() + 3; }");
}

TEST(FixItTest, createRemoval) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    FixItHint Hint = createRemoval(*CE);
    EXPECT_EQ("foo(x, y)", getText(Hint.RemoveRange.getAsRange(), *Context));
    EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
    EXPECT_TRUE(Hint.CodeToInsert.empty());

    Expr *P0 = CE->getArg(0);
    FixItHint Hint0 = createRemoval(*P0);
    EXPECT_EQ("x", getText(Hint0.RemoveRange.getAsRange(), *Context));
    EXPECT_TRUE(Hint0.InsertFromRange.isInvalid());
    EXPECT_TRUE(Hint0.CodeToInsert.empty());

    Expr *P1 = CE->getArg(1);
    FixItHint Hint1 = createRemoval(*P1);
    EXPECT_EQ("y", getText(Hint1.RemoveRange.getAsRange(), *Context));
    EXPECT_TRUE(Hint1.InsertFromRange.isInvalid());
    EXPECT_TRUE(Hint1.CodeToInsert.empty());
  };
  Visitor.runOver("void foo(int x, int y) { foo(x, y); }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    Expr *P0 = CE->getArg(0);
    FixItHint Hint0 = createRemoval(*P0);
    EXPECT_EQ("x + y", getText(Hint0.RemoveRange.getAsRange(), *Context));

    Expr *P1 = CE->getArg(1);
    FixItHint Hint1 = createRemoval(*P1);
    EXPECT_EQ("y + x", getText(Hint1.RemoveRange.getAsRange(), *Context));
  };
  Visitor.runOver("void foo(int x, int y) { foo(x + y, y + x); }");
}

TEST(FixItTest, createRemovalWithMacro) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    FixItHint Hint = createRemoval(*CE);
    EXPECT_EQ("FOO", getText(Hint.RemoveRange.getAsRange(), *Context));
    EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
    EXPECT_TRUE(Hint.CodeToInsert.empty());

    Expr *P0 = CE->getArg(0);
    FixItHint Hint0 = createRemoval(*P0);
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:1:17>",
              LocationToString(Hint0.RemoveRange.getBegin(), Context));
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:1:17>",
              LocationToString(Hint0.RemoveRange.getEnd(), Context));
    EXPECT_TRUE(Hint0.InsertFromRange.isInvalid());
    EXPECT_TRUE(Hint0.CodeToInsert.empty());

    Expr *P1 = CE->getArg(1);
    FixItHint Hint1 = createRemoval(*P1);
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:1:20>",
              LocationToString(Hint1.RemoveRange.getBegin(), Context));
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:1:20>",
              LocationToString(Hint1.RemoveRange.getEnd(), Context));
    EXPECT_TRUE(Hint1.InsertFromRange.isInvalid());
    EXPECT_TRUE(Hint1.CodeToInsert.empty());
  };
  Visitor.runOver("#define FOO foo(1, 1)\n"
                  "void foo(int x, int y) { FOO; }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    FixItHint Hint = createRemoval(*CE);
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:1:37>",
              LocationToString(Hint.RemoveRange.getBegin(), Context));
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:1:45>",
              LocationToString(Hint.RemoveRange.getEnd(), Context));
    EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
    EXPECT_TRUE(Hint.CodeToInsert.empty());
  };
  Visitor.runOver("#define FOO(x, y) (void)x; (void)y; foo(x, y);\n"
                  "void foo(int x, int y) { FOO(x,y) }");
}

TEST(FixItTest, createReplacement) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    FixItHint Hint0 = createReplacement(*P0, *P1, *Context);
    FixItHint Hint1 = createReplacement(*P1, *P0, *Context);

    // Validate Hint0 fields.
    EXPECT_EQ("x", getText(Hint0.RemoveRange.getAsRange(), *Context));
    EXPECT_TRUE(Hint0.InsertFromRange.isInvalid());
    EXPECT_EQ(Hint0.CodeToInsert, "y");

    // Validate Hint1 fields.
    EXPECT_EQ("y", getText(Hint1.RemoveRange.getAsRange(), *Context));
    EXPECT_TRUE(Hint1.InsertFromRange.isInvalid());
    EXPECT_EQ(Hint1.CodeToInsert, "x");
  };

  Visitor.runOver("void foo(int x, int y) { foo(x, y); }");

  Visitor.runOver("#define APPLY(f, x, y) f(x, y)\n"
                  "void foo(int x, int y) { APPLY(foo, x, y); }");

  Visitor.runOver("#define APPLY(f, P) f(P)\n"
                  "#define PAIR(x, y) x, y\n"
                  "void foo(int x, int y) { APPLY(foo, PAIR(x, y)); }\n");
}

TEST(FixItTest, createReplacementWithMacro) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    FixItHint Hint = createReplacement(*P0, *P1, *Context);
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:1:17>",
              LocationToString(Hint.RemoveRange.getBegin(), Context));
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:1:17>",
              LocationToString(Hint.RemoveRange.getEnd(), Context));
    EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
    EXPECT_TRUE(Hint.CodeToInsert.empty());
  };

  Visitor.runOver("#define FOO foo(1, 1)\n"
                  "void foo(int x, int y) { FOO; }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    FixItHint Hint = createReplacement(*P0, *P1, *Context);
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:2:30>",
              LocationToString(Hint.RemoveRange.getBegin(), Context));
    EXPECT_EQ("input.cc:2:26 <Spelling=input.cc:2:30>",
              LocationToString(Hint.RemoveRange.getEnd(), Context));
    EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
    EXPECT_EQ("y", Hint.CodeToInsert);
  };
  Visitor.runOver("#define FOO(x, y) (void)x; (void)y; foo(x, y);\n"
                  "void foo(int x, int y) { FOO(x,y) }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    Expr *P0 = CE->getArg(0);
    Expr *P1 = CE->getArg(1);
    FixItHint Hint = createReplacement(*P0, *P1, *Context);
    EXPECT_EQ("x + y", getText(Hint.RemoveRange.getAsRange(), *Context));
    EXPECT_TRUE(Hint.InsertFromRange.isInvalid());
    EXPECT_EQ("y + x", Hint.CodeToInsert);
  };
  Visitor.runOver("void foo(int x, int y) { foo(x + y, y + x); }");
}

} // end anonymous namespace
