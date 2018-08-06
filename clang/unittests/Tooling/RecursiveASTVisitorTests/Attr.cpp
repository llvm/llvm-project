//===- unittest/Tooling/RecursiveASTVisitorTests/Attr.cpp -----------------===//
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

// Check to ensure that attributes and expressions within them are being
// visited.
class AttrVisitor : public ExpectedLocationVisitor<AttrVisitor> {
public:
  bool VisitMemberExpr(MemberExpr *ME) {
    Match(ME->getMemberDecl()->getNameAsString(), ME->getLocStart());
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
