//===- unittest/Tooling/RecursiveASTVisitorTests/CXXOperatorCallExprTraverser.cpp -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class CXXOperatorCallExprTraverser
  : public ExpectedLocationVisitor<CXXOperatorCallExprTraverser> {
public:
  // Use Traverse, not Visit, to check that data recursion optimization isn't
  // bypassing the call of this function.
  bool TraverseCXXOperatorCallExpr(CXXOperatorCallExpr *CE) {
    Match(getOperatorSpelling(CE->getOperator()), CE->getExprLoc());
    return ExpectedLocationVisitor<CXXOperatorCallExprTraverser>::
        TraverseCXXOperatorCallExpr(CE);
  }
};

TEST(RecursiveASTVisitor, TraversesOverloadedOperator) {
  CXXOperatorCallExprTraverser Visitor;
  Visitor.ExpectMatch("()", 4, 9);
  EXPECT_TRUE(Visitor.runOver(
    "struct A {\n"
    "  int operator()();\n"
    "} a;\n"
    "int k = a();\n"));
}

} // end anonymous namespace
