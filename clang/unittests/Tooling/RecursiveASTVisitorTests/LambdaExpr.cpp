//===- unittest/Tooling/RecursiveASTVisitorTests/LambdaExpr.cpp -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include <stack>

using namespace clang;

namespace {

class LambdaExprVisitor : public ExpectedLocationVisitor<LambdaExprVisitor> {
public:
  bool VisitLambdaExpr(LambdaExpr *Lambda) {
    PendingBodies.push(Lambda);
    Match("", Lambda->getIntroducerRange().getBegin());
    return true;
  }
  /// For each call to VisitLambdaExpr, we expect a subsequent call (with
  /// proper nesting) to TraverseLambdaBody.
  bool TraverseLambdaBody(LambdaExpr *Lambda) {
    EXPECT_FALSE(PendingBodies.empty());
    EXPECT_EQ(PendingBodies.top(), Lambda);
    PendingBodies.pop();
    return TraverseStmt(Lambda->getBody());
  }
  /// Determine whether TraverseLambdaBody has been called for every call to
  /// VisitLambdaExpr.
  bool allBodiesHaveBeenTraversed() const {
    return PendingBodies.empty();
  }
private:
  std::stack<LambdaExpr *> PendingBodies;
};

TEST(RecursiveASTVisitor, VisitsLambdaExpr) {
  LambdaExprVisitor Visitor;
  Visitor.ExpectMatch("", 1, 12);
  EXPECT_TRUE(Visitor.runOver("void f() { []{ return; }(); }",
                              LambdaExprVisitor::Lang_CXX11));
}

TEST(RecursiveASTVisitor, TraverseLambdaBodyCanBeOverridden) {
  LambdaExprVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver("void f() { []{ return; }(); }",
                              LambdaExprVisitor::Lang_CXX11));
  EXPECT_TRUE(Visitor.allBodiesHaveBeenTraversed());
}

TEST(RecursiveASTVisitor, VisitsAttributedLambdaExpr) {
  LambdaExprVisitor Visitor;
  Visitor.ExpectMatch("", 1, 12);
  EXPECT_TRUE(Visitor.runOver(
      "void f() { [] () __attribute__ (( fastcall )) { return; }(); }",
      LambdaExprVisitor::Lang_CXX14));
}

} // end anonymous namespace
