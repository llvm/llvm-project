//===- unittest/Tooling/RecursiveASTVisitorTests/ConstructExpr.cpp --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

/// \brief A visitor that optionally includes implicit code and matches
/// CXXConstructExpr.
///
/// The name recorded for the match is the name of the class whose constructor
/// is invoked by the CXXConstructExpr, not the name of the class whose
/// constructor the CXXConstructExpr is contained in.
class ConstructExprVisitor : public ExpectedLocationVisitor {
public:
  ConstructExprVisitor() { ShouldVisitImplicitCode = false; }

  bool VisitCXXConstructExpr(CXXConstructExpr *Expr) override {
    if (const CXXConstructorDecl* Ctor = Expr->getConstructor()) {
      if (const CXXRecordDecl* Class = Ctor->getParent()) {
        Match(Class->getName(), Expr->getLocation());
      }
    }
    return true;
  }
};

TEST(RecursiveASTVisitor, CanVisitImplicitMemberInitializations) {
  ConstructExprVisitor Visitor;
  Visitor.ShouldVisitImplicitCode = true;
  Visitor.ExpectMatch("WithCtor", 2, 8);
  // Simple has a constructor that implicitly initializes 'w'.  Test
  // that a visitor that visits implicit code visits that initialization.
  // Note: Clang lazily instantiates implicit declarations, so we need
  // to use them in order to force them to appear in the AST.
  EXPECT_TRUE(Visitor.runOver(
      "struct WithCtor { WithCtor(); }; \n"
      "struct Simple { WithCtor w; }; \n"
      "int main() { Simple s; }\n"));
}

// The same as CanVisitImplicitMemberInitializations, but checking that the
// visits are omitted when the visitor does not include implicit code.
TEST(RecursiveASTVisitor, CanSkipImplicitMemberInitializations) {
  ConstructExprVisitor Visitor;
  Visitor.ShouldVisitImplicitCode = false;
  Visitor.DisallowMatch("WithCtor", 2, 8);
  // Simple has a constructor that implicitly initializes 'w'.  Test
  // that a visitor that skips implicit code skips that initialization.
  // Note: Clang lazily instantiates implicit declarations, so we need
  // to use them in order to force them to appear in the AST.
  EXPECT_TRUE(Visitor.runOver(
      "struct WithCtor { WithCtor(); }; \n"
      "struct Simple { WithCtor w; }; \n"
      "int main() { Simple s; }\n"));
}

} // end anonymous namespace
