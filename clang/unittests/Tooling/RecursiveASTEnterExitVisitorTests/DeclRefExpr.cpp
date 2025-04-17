//===- unittest/Tooling/RecursiveASTEnterExitVisitorTests/DeclRefExpr.cpp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/AST/RecursiveASTEnterExitVisitor.h"

using namespace clang;

namespace {

class DeclRefExprVisitor : public ExpectedLocationVisitor {
public:
  DeclRefExprVisitor() { ShouldVisitImplicitCode = false; }

  bool VisitDeclRefExpr(DeclRefExpr *Reference) override {
    Match(Reference->getNameInfo().getAsString(), Reference->getLocation());
    return true;
  }
};

TEST(RecursiveASTEnterExitVisitor, VisitsBaseClassTemplateArguments) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 2, 3);
  EXPECT_TRUE(Visitor.runOver(
    "void x(); template <void (*T)()> class X {};\nX<x> y;"));
}

TEST(RecursiveASTEnterExitVisitor, VisitsCXXForRangeStmtRange) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 2, 25);
  Visitor.ExpectMatch("x", 2, 30);
  EXPECT_TRUE(Visitor.runOver(
    "int x[5];\n"
    "void f() { for (int i : x) { x[0] = 1; } }",
    DeclRefExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTEnterExitVisitor, VisitsCallExpr) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 1, 22);
  EXPECT_TRUE(Visitor.runOver(
    "void x(); void y() { x(); }"));
}

TEST(RecursiveASTEnterExitVisitor, VisitsExplicitLambdaCaptureInit) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("i", 1, 20);
  EXPECT_TRUE(Visitor.runOver(
    "void f() { int i; [i]{}; }",
    DeclRefExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTEnterExitVisitor, VisitsUseOfImplicitLambdaCapture) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("i", 1, 24);
  EXPECT_TRUE(Visitor.runOver(
    "void f() { int i; [=]{ i; }; }",
    DeclRefExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTEnterExitVisitor, VisitsImplicitLambdaCaptureInit) {
  DeclRefExprVisitor Visitor;
  Visitor.ShouldVisitImplicitCode = true;
  // We're expecting "i" to be visited twice: once for the initialization expr
  // for the captured variable "i" outside of the lambda body, and again for
  // the use of "i" inside the lambda.
  Visitor.ExpectMatch("i", 1, 20, /*Times=*/1);
  Visitor.ExpectMatch("i", 1, 24, /*Times=*/1);
  EXPECT_TRUE(Visitor.runOver(
    "void f() { int i; [=]{ i; }; }",
    DeclRefExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTEnterExitVisitor, VisitsLambdaInitCaptureInit) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("i", 1, 24);
  EXPECT_TRUE(Visitor.runOver(
    "void f() { int i; [a = i + 1]{}; }",
    DeclRefExprVisitor::Lang_CXX14));
}

/* FIXME: According to Richard Smith this is a bug in the AST.
TEST(RecursiveASTEnterExitVisitor, VisitsBaseClassTemplateArgumentsInInstantiation) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 3, 43);
  EXPECT_TRUE(Visitor.runOver(
    "template <typename T> void x();\n"
    "template <void (*T)()> class X {};\n"
    "template <typename T> class Y : public X< x<T> > {};\n"
    "Y<int> y;"));
}
*/

TEST(RecursiveASTEnterExitVisitor, VisitsExtension) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("s", 1, 24);
  EXPECT_TRUE(Visitor.runOver(
    "int s = __extension__ (s);\n"));
}

TEST(RecursiveASTEnterExitVisitor, VisitsCopyExprOfBlockDeclCapture) {
  DeclRefExprVisitor Visitor;
  Visitor.ExpectMatch("x", 3, 24);
  EXPECT_TRUE(Visitor.runOver("void f(int(^)(int)); \n"
                              "void g() { \n"
                              "  f([&](int x){ return x; }); \n"
                              "}",
                              DeclRefExprVisitor::Lang_OBJCXX11));
}

} // end anonymous namespace
