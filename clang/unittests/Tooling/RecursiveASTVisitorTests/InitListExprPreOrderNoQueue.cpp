//===- unittest/Tooling/RecursiveASTVisitorTests/InitListExprPreOrderNoQueue.cpp -===//
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
