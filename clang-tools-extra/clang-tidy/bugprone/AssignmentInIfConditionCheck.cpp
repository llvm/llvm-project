//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AssignmentInIfConditionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void AssignmentInIfConditionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl(), this);
}

void AssignmentInIfConditionCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  class Visitor : public ConstDynamicRecursiveASTVisitor {
    AssignmentInIfConditionCheck &Check;

  public:
    explicit Visitor(AssignmentInIfConditionCheck &Check) : Check(Check) {}
    bool VisitIfStmt(const IfStmt *If) override {
      class ConditionVisitor : public ConstDynamicRecursiveASTVisitor {
        AssignmentInIfConditionCheck &Check;

      public:
        explicit ConditionVisitor(AssignmentInIfConditionCheck &Check)
            : Check(Check) {}

        // Dont traverse into any lambda expressions.
        bool TraverseLambdaExpr(const LambdaExpr *) override { return true; }

        // Dont traverse into any requires expressions.
        bool TraverseRequiresExpr(const RequiresExpr *) override {
          return true;
        }

        bool VisitBinaryOperator(const BinaryOperator *BO) override {
          if (BO->isAssignmentOp())
            Check.report(BO);
          return true;
        }

        bool VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE) override {
          if (OCE->isAssignmentOp())
            Check.report(OCE);
          return true;
        }
      };

      ConditionVisitor(Check).TraverseStmt(If->getCond());
      return true;
    }
  };
  Visitor(*this).TraverseAST(*Result.Context);
}

void AssignmentInIfConditionCheck::report(const Expr *AssignmentExpr) {
  SourceLocation OpLoc =
      isa<BinaryOperator>(AssignmentExpr)
          ? cast<BinaryOperator>(AssignmentExpr)->getOperatorLoc()
          : cast<CXXOperatorCallExpr>(AssignmentExpr)->getOperatorLoc();

  diag(OpLoc, "an assignment within an 'if' condition is bug-prone")
      << AssignmentExpr->getSourceRange();
  diag(OpLoc,
       "if it should be an assignment, move it out of the 'if' condition",
       DiagnosticIDs::Note);
  diag(OpLoc, "if it is meant to be an equality check, change '=' to '=='",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
