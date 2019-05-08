//===- unittest/Tooling/SourceCodeTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Tooling/Refactoring/SourceCode.h"

using namespace clang;

using tooling::getText;
using tooling::getExtendedText;

namespace {

struct CallsVisitor : TestVisitor<CallsVisitor> {
  bool VisitCallExpr(CallExpr *Expr) {
    OnCall(Expr, Context);
    return true;
  }

  std::function<void(CallExpr *, ASTContext *Context)> OnCall;
};

TEST(SourceCodeTest, getText) {
  CallsVisitor Visitor;

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("foo(x, y)", getText(*CE, *Context));
  };
  Visitor.runOver("void foo(int x, int y) { foo(x, y); }");

  Visitor.OnCall = [](CallExpr *CE, ASTContext *Context) {
    EXPECT_EQ("APPLY(foo, x, y)", getText(*CE, *Context));
  };
  Visitor.runOver("#define APPLY(f, x, y) f(x, y)\n"
                  "void foo(int x, int y) { APPLY(foo, x, y); }");
}

TEST(SourceCodeTest, getTextWithMacro) {
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

TEST(SourceCodeTest, getExtendedText) {
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

} // end anonymous namespace
