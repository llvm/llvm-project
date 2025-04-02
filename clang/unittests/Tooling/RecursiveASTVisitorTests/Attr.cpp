//===- unittest/Tooling/RecursiveASTVisitorTests/Attr.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CRTPTestVisitor.h"

using namespace clang;

namespace {

// Check to ensure that attributes and expressions within them are being
// visited.
class AttrVisitor : public CRTPExpectedLocationVisitor<AttrVisitor> {
public:
  bool VisitMemberExpr(MemberExpr *ME) {
    Match(ME->getMemberDecl()->getNameAsString(), ME->getBeginLoc());
    return true;
  }
  bool VisitAttr(Attr *A) {
    Match("Attr", A->getLocation());
    return true;
  }
  bool VisitGuardedByAttr(GuardedByAttr *A) {
    Match("guarded_by", A->getLocation());
    return true;
  }
};

TEST(RecursiveASTVisitor, AttributesAreVisited) {
  AttrVisitor Visitor;
  Visitor.ExpectMatch("Attr", 4, 24);
  Visitor.ExpectMatch("guarded_by", 4, 24);
  Visitor.ExpectMatch("mu1",  4, 35);
  Visitor.ExpectMatch("Attr", 5, 29);
  Visitor.ExpectMatch("mu1",  5, 54);
  Visitor.ExpectMatch("mu2",  5, 59);
  EXPECT_TRUE(Visitor.runOver(
    "class Foo {\n"
    "  int mu1;\n"
    "  int mu2;\n"
    "  int a __attribute__((guarded_by(mu1)));\n"
    "  void bar() __attribute__((exclusive_locks_required(mu1, mu2)));\n"
    "};\n"));
}

} // end anonymous namespace
