//===- unittest/Tooling/RecursiveASTVisitorTests/ConstructExpr.cpp --------===//
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

/// \brief A visitor that optionally includes implicit code and matches
/// CXXConstructExpr.
///
/// The name recorded for the match is the name of the class whose constructor
/// is invoked by the CXXConstructExpr, not the name of the class whose
/// constructor the CXXConstructExpr is contained in.
class ConstructExprVisitor
    : public ExpectedLocationVisitor<ConstructExprVisitor> {
public:
  ConstructExprVisitor() : ShouldVisitImplicitCode(false) {}

  bool shouldVisitImplicitCode() const { return ShouldVisitImplicitCode; }

  void setShouldVisitImplicitCode(bool NewValue) {
    ShouldVisitImplicitCode = NewValue;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr* Expr) {
    if (const CXXConstructorDecl* Ctor = Expr->getConstructor()) {
      if (const CXXRecordDecl* Class = Ctor->getParent()) {
        Match(Class->getName(), Expr->getLocation());
      }
    }
    return true;
  }

 private:
  bool ShouldVisitImplicitCode;
};

TEST(RecursiveASTVisitor, CanVisitImplicitMemberInitializations) {
  ConstructExprVisitor Visitor;
  Visitor.setShouldVisitImplicitCode(true);
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
  Visitor.setShouldVisitImplicitCode(false);
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
