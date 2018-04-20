//===- unittest/Tooling/RecursiveASTVisitorTests/IntegerLiteral.cpp -------===//
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

// Check to ensure that implicit default argument expressions are visited.
class IntegerLiteralVisitor
    : public ExpectedLocationVisitor<IntegerLiteralVisitor> {
public:
  bool VisitIntegerLiteral(const IntegerLiteral *IL) {
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
