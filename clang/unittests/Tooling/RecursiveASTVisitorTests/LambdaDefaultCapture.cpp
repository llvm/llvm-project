//===- unittest/Tooling/RecursiveASTVisitorTests/LambdaDefaultCapture.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
