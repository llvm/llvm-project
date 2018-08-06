//===- unittest/Tooling/RecursiveASTVisitorTests/CXXBoolLiteralExpr.cpp ---===//
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

class CXXBoolLiteralExprVisitor 
  : public ExpectedLocationVisitor<CXXBoolLiteralExprVisitor> {
public:
  bool VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *BE) {
    if (BE->getValue())
      Match("true", BE->getLocation());
    else
      Match("false", BE->getLocation());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsClassTemplateNonTypeParmDefaultArgument) {
  CXXBoolLiteralExprVisitor Visitor;
  Visitor.ExpectMatch("true", 2, 19);
  EXPECT_TRUE(Visitor.runOver(
    "template<bool B> class X;\n"
    "template<bool B = true> class Y;\n"
    "template<bool B> class Y {};\n"));
}

} // end anonymous namespace
