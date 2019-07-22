//===- unittest/Tooling/RecursiveASTVisitorTests/InitListExprPreOrder.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

// Check to ensure that InitListExpr is visited twice, once each for the
// syntactic and semantic form.
class InitListExprPreOrderVisitor
    : public ExpectedLocationVisitor<InitListExprPreOrderVisitor> {
public:
  InitListExprPreOrderVisitor(bool VisitImplicitCode)
      : VisitImplicitCode(VisitImplicitCode) {}

  bool shouldVisitImplicitCode() const { return VisitImplicitCode; }

  bool VisitInitListExpr(InitListExpr *ILE) {
    Match(ILE->isSemanticForm() ? "semantic" : "syntactic", ILE->getBeginLoc());
    return true;
  }

private:
  bool VisitImplicitCode;
};

TEST(RecursiveASTVisitor, InitListExprIsPreOrderVisitedTwice) {
  InitListExprPreOrderVisitor Visitor(/*VisitImplicitCode=*/true);
  Visitor.ExpectMatch("syntactic", 2, 21);
  Visitor.ExpectMatch("semantic", 2, 21);
  EXPECT_TRUE(Visitor.runOver("struct S { int x; };\n"
                              "static struct S s = {.x = 0};\n",
                              InitListExprPreOrderVisitor::Lang_C));
}

TEST(RecursiveASTVisitor, InitListExprVisitedOnceWhenNoImplicit) {
  InitListExprPreOrderVisitor Visitor(/*VisitImplicitCode=*/false);
  Visitor.ExpectMatch("syntactic", 2, 21);
  Visitor.DisallowMatch("semantic", 2, 21);
  EXPECT_TRUE(Visitor.runOver("struct S { int x; };\n"
                              "static struct S s = {.x = 0};\n",
                              InitListExprPreOrderVisitor::Lang_C));
}

} // end anonymous namespace
