//===- unittest/Tooling/RecursiveASTVisitorTests/InitListExprPostOrderNoQueue.cpp -===//
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

class InitListExprPostOrderNoQueueVisitor
    : public ExpectedLocationVisitor<InitListExprPostOrderNoQueueVisitor> {
public:
  bool shouldTraversePostOrder() const { return true; }

  bool TraverseInitListExpr(InitListExpr *ILE) {
    return ExpectedLocationVisitor::TraverseInitListExpr(ILE);
  }

  bool VisitInitListExpr(InitListExpr *ILE) {
    Match(ILE->isSemanticForm() ? "semantic" : "syntactic", ILE->getLocStart());
    return true;
  }
};

TEST(RecursiveASTVisitor, InitListExprIsPostOrderNoQueueVisitedTwice) {
  InitListExprPostOrderNoQueueVisitor Visitor;
  Visitor.ExpectMatch("syntactic", 2, 21);
  Visitor.ExpectMatch("semantic", 2, 21);
  EXPECT_TRUE(Visitor.runOver("struct S { int x; };\n"
                              "static struct S s = {.x = 0};\n",
                              InitListExprPostOrderNoQueueVisitor::Lang_C));
}

} // end anonymous namespace
