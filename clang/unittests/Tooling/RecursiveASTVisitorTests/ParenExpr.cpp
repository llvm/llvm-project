//===- unittest/Tooling/RecursiveASTVisitorTests/ParenExpr.cpp ------------===//
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

class ParenExprVisitor : public ExpectedLocationVisitor<ParenExprVisitor> {
public:
  bool VisitParenExpr(ParenExpr *Parens) {
    Match("", Parens->getExprLoc());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsParensDuringDataRecursion) {
  ParenExprVisitor Visitor;
  Visitor.ExpectMatch("", 1, 9);
  EXPECT_TRUE(Visitor.runOver("int k = (4) + 9;\n"));
}

} // end anonymous namespace
