//===- unittest/Tooling/SourceCodeTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/SourceCode.h"
#include "TestVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/Testing/Support/Annotations.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace clang;

using llvm::ValueIs;
using tooling::getExtendedText;
using tooling::getRangeForEdit;
using tooling::getText;

namespace {

struct IntLitVisitor : TestVisitor<IntLitVisitor> {
  bool VisitIntegerLiteral(IntegerLiteral *Expr) {
    OnIntLit(Expr, Context);
    return true;
  }

  std::function<void(IntegerLiteral *, ASTContext *Context)> OnIntLit;
};

struct CallsVisitor : TestVisitor<CallsVisitor> {
  bool VisitCallExpr(CallExpr *Expr) {
    OnCall(Expr, Context);
    return true;
  }

  std::function<void(CallExpr *, ASTContext *Context)> OnCall;
};

// Equality matcher for `clang::CharSourceRange`, which lacks `operator==`.
MATCHER_P(EqualsRange, R, "") {
  return arg.isTokenRange() == R.isTokenRange() &&
         arg.getBegin() == R.getBegin() && arg.getEnd() == R.getEnd();
}

static ::testing::Matcher<CharSourceRange> AsRange(const SourceManager &SM,
                                                   llvm::Annotations::Range R) {
  return EqualsRange(CharSourceRange::getCharRange(
      SM.getLocForStartOfFile(SM.getMainFileID()).getLocWithOffset(R.Begin),
      SM.getLocForStartOfFile(SM.getMainFileID()).getLocWithOffset(R.End)));
}

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

TEST(SourceCodeTest, EditRangeWithMacroExpansionsShouldSucceed) {
  // The call expression, whose range we are extracting, includes two macro
  // expansions.
  llvm::Annotations Code(R"cpp(
#define M(a) a * 13
int foo(int x, int y);
int a = $r[[foo(M(1), M(2))]];
)cpp");

  CallsVisitor Visitor;

  Visitor.OnCall = [&Code](CallExpr *CE, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(CE->getSourceRange());
    EXPECT_THAT(getRangeForEdit(Range, *Context),
                ValueIs(AsRange(Context->getSourceManager(), Code.range("r"))));
  };
  Visitor.runOver(Code.code());
}

TEST(SourceCodeTest, EditWholeMacroExpansionShouldSucceed) {
  llvm::Annotations Code(R"cpp(
#define FOO 10
int a = $r[[FOO]];
)cpp");

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [&Code](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT(getRangeForEdit(Range, *Context),
                ValueIs(AsRange(Context->getSourceManager(), Code.range("r"))));
  };
  Visitor.runOver(Code.code());
}

TEST(SourceCodeTest, EditPartialMacroExpansionShouldFail) {
  std::string Code = R"cpp(
#define BAR 10+
int c = BAR 3.0;
)cpp";

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_FALSE(getRangeForEdit(Range, *Context).hasValue());
  };
  Visitor.runOver(Code);
}

TEST(SourceCodeTest, EditWholeMacroArgShouldSucceed) {
  llvm::Annotations Code(R"cpp(
#define FOO(a) a + 7.0;
int a = FOO($r[[10]]);
)cpp");

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [&Code](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT(getRangeForEdit(Range, *Context),
                ValueIs(AsRange(Context->getSourceManager(), Code.range("r"))));
  };
  Visitor.runOver(Code.code());
}

TEST(SourceCodeTest, EditPartialMacroArgShouldSucceed) {
  llvm::Annotations Code(R"cpp(
#define FOO(a) a + 7.0;
int a = FOO($r[[10]] + 10.0);
)cpp");

  IntLitVisitor Visitor;
  Visitor.OnIntLit = [&Code](IntegerLiteral *Expr, ASTContext *Context) {
    auto Range = CharSourceRange::getTokenRange(Expr->getSourceRange());
    EXPECT_THAT(getRangeForEdit(Range, *Context),
                ValueIs(AsRange(Context->getSourceManager(), Code.range("r"))));
  };
  Visitor.runOver(Code.code());
}

} // end anonymous namespace
