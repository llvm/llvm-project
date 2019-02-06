//===- unittest/Tooling/RecursiveASTVisitorTests/InitListExprPreOrderNoQueue.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class InitListExprPreOrderNoQueueVisitor
    : public ExpectedLocationVisitor<InitListExprPreOrderNoQueueVisitor> {
public:
  bool TraverseInitListExpr(InitListExpr *ILE) {
    return ExpectedLocationVisitor::TraverseInitListExpr(ILE);
  }

  bool VisitInitListExpr(InitListExpr *ILE) {
    Match(ILE->isSemanticForm() ? "semantic" : "syntactic", ILE->getBeginLoc());
    return true;
  }
};

TEST(RecursiveASTVisitor, InitListExprIsPreOrderNoQueueVisitedTwice) {
  InitListExprPreOrderNoQueueVisitor Visitor;
  Visitor.ExpectMatch("syntactic", 2, 21);
  Visitor.ExpectMatch("semantic", 2, 21);
  EXPECT_TRUE(Visitor.runOver("struct S { int x; };\n"
                              "static struct S s = {.x = 0};\n",
                              InitListExprPreOrderNoQueueVisitor::Lang_C));
}

} // end anonymous namespace
