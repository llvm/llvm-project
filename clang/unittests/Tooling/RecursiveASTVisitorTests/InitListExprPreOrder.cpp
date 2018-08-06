//===- unittest/Tooling/RecursiveASTVisitorTests/InitListExprPreOrder.cpp -===//
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

// Check to ensure that InitListExpr is visited twice, once each for the
// syntactic and semantic form.
class InitListExprPreOrderVisitor
    : public ExpectedLocationVisitor<InitListExprPreOrderVisitor> {
public:
  bool VisitInitListExpr(InitListExpr *ILE) {
    Match(ILE->isSemanticForm() ? "semantic" : "syntactic", ILE->getLocStart());
    return true;
  }
};

TEST(RecursiveASTVisitor, InitListExprIsPreOrderVisitedTwice) {
  InitListExprPreOrderVisitor Visitor;
  Visitor.ExpectMatch("syntactic", 2, 21);
  Visitor.ExpectMatch("semantic", 2, 21);
  EXPECT_TRUE(Visitor.runOver("struct S { int x; };\n"
                              "static struct S s = {.x = 0};\n",
                              InitListExprPreOrderVisitor::Lang_C));
}

} // end anonymous namespace
