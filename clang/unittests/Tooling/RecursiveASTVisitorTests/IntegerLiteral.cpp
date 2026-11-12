//===- unittest/Tooling/RecursiveASTVisitorTests/IntegerLiteral.cpp -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

// Check to ensure that implicit default argument expressions are visited.
class IntegerLiteralVisitor : public ExpectedLocationVisitor {
public:
  bool VisitIntegerLiteral(IntegerLiteral *IL) override {
    Match("literal", IL->getLocation());
    return true;
  }
};

TEST(RecursiveASTVisitor, DefaultArgumentsAreVisited) {
  IntegerLiteralVisitor Visitor;
  Visitor.ExpectMatch("literal", 1, 15, 2);
  EXPECT_TRUE(Visitor.runOver("int f(int i = 1);\n"
                              "static int k = f();\n"));
}

} // end anonymous namespace
