//===- unittest/Tooling/RecursiveASTVisitorTests/LambdaDefaultCapture.cpp -===//
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

// Matches the (optional) capture-default of a lambda-introducer.
class LambdaDefaultCaptureVisitor
  : public ExpectedLocationVisitor<LambdaDefaultCaptureVisitor> {
public:
  bool VisitLambdaExpr(LambdaExpr *Lambda) {
    if (Lambda->getCaptureDefault() != LCD_None) {
      Match("", Lambda->getCaptureDefaultLoc());
    }
    return true;
  }
};

TEST(RecursiveASTVisitor, HasCaptureDefaultLoc) {
  LambdaDefaultCaptureVisitor Visitor;
  Visitor.ExpectMatch("", 1, 20);
  EXPECT_TRUE(Visitor.runOver("void f() { int a; [=]{a;}; }",
                              LambdaDefaultCaptureVisitor::Lang_CXX11));
}

} // end anonymous namespace
