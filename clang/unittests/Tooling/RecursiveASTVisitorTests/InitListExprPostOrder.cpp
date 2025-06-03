//===- unittest/Tooling/RecursiveASTVisitorTests/InitListExprPostOrder.cpp -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CRTPTestVisitor.h"

using namespace clang;

namespace {

class InitListExprPostOrderVisitor
    : public CRTPExpectedLocationVisitor<InitListExprPostOrderVisitor> {
public:
  bool shouldTraversePostOrder() const { return true; }

  bool VisitInitListExpr(InitListExpr *ILE) {
    Match(ILE->isSemanticForm() ? "semantic" : "syntactic", ILE->getBeginLoc());
    return true;
  }
};

TEST(RecursiveASTVisitor, InitListExprIsPostOrderVisitedTwice) {
  InitListExprPostOrderVisitor Visitor;
  Visitor.ExpectMatch("syntactic", 2, 21);
  Visitor.ExpectMatch("semantic", 2, 21);
  EXPECT_TRUE(Visitor.runOver("struct S { int x; };\n"
                              "static struct S s = {.x = 0};\n",
                              InitListExprPostOrderVisitor::Lang_C));
}

} // end anonymous namespace
