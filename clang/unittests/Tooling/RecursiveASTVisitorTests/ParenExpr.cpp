//===- unittest/Tooling/RecursiveASTVisitorTests/ParenExpr.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class ParenExprVisitor : public ExpectedLocationVisitor {
public:
  bool VisitParenExpr(ParenExpr *Parens) override {
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
