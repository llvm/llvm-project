//===- unittest/Tooling/RecursiveASTVisitorTests/LambdaTemplateParams.cpp -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

// Matches (optional) explicit template parameters.
class LambdaTemplateParametersVisitor
  : public ExpectedLocationVisitor<LambdaTemplateParametersVisitor> {
public:
  bool shouldVisitImplicitCode() const { return false; }

  bool VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
    EXPECT_FALSE(D->isImplicit());
    Match(D->getName(), D->getBeginLoc());
    return true;
  }

  bool VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
    EXPECT_FALSE(D->isImplicit());
    Match(D->getName(), D->getBeginLoc());
    return true;
  }

  bool VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
    EXPECT_FALSE(D->isImplicit());
    Match(D->getName(), D->getBeginLoc());
    return true;
  }
};

TEST(RecursiveASTVisitor, VisitsLambdaExplicitTemplateParameters) {
  LambdaTemplateParametersVisitor Visitor;
  Visitor.ExpectMatch("T",  2, 15);
  Visitor.ExpectMatch("I",  2, 24);
  Visitor.ExpectMatch("TT", 2, 31);
  EXPECT_TRUE(Visitor.runOver(
      "void f() { \n"
      "  auto l = []<class T, int I, template<class> class TT>(auto p) { }; \n"
      "}",
      LambdaTemplateParametersVisitor::Lang_CXX2a));
}

} // end anonymous namespace
