//===--- ConditionaltostdminmaxCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConditionaltostdminmaxCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void ConditionaltostdminmaxCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      ifStmt(has(binaryOperator(
                 anyOf(hasOperatorName("<"), hasOperatorName(">")),
                 hasLHS(ignoringImpCasts(declRefExpr().bind("lhsVar1"))),
                 hasRHS(ignoringImpCasts(declRefExpr().bind("rhsVar1"))))),
             hasThen(stmt(binaryOperator(
                 hasOperatorName("="),
                 hasLHS(ignoringImpCasts(declRefExpr().bind("lhsVar2"))),
                 hasRHS(ignoringImpCasts(declRefExpr().bind("rhsVar2")))))))
          .bind("ifStmt"),
      this);
}

void ConditionaltostdminmaxCheck::check(
    const MatchFinder::MatchResult &Result) {
  const DeclRefExpr *lhsVar1 = Result.Nodes.getNodeAs<DeclRefExpr>("lhsVar1");
  const DeclRefExpr *rhsVar1 = Result.Nodes.getNodeAs<DeclRefExpr>("rhsVar1");
  const DeclRefExpr *lhsVar2 = Result.Nodes.getNodeAs<DeclRefExpr>("lhsVar2");
  const DeclRefExpr *rhsVar2 = Result.Nodes.getNodeAs<DeclRefExpr>("rhsVar2");
  const IfStmt *ifStmt = Result.Nodes.getNodeAs<IfStmt>("ifStmt");

  if (!lhsVar1 || !rhsVar1 || !lhsVar2 || !rhsVar2 || !ifStmt)
    return;

  const BinaryOperator *binaryOp = dyn_cast<BinaryOperator>(ifStmt->getCond());
  if (!binaryOp)
    return;

  SourceLocation ifLocation = ifStmt->getIfLoc();
  SourceLocation thenLocation = ifStmt->getEndLoc();

  if (binaryOp->getOpcode() == BO_LT) {
    if (lhsVar1->getDecl() == lhsVar2->getDecl() &&
        rhsVar1->getDecl() == rhsVar2->getDecl()) {
      diag(ifStmt->getIfLoc(), "use std::max instead of <")
          << FixItHint::CreateReplacement(
                 SourceRange(ifLocation, thenLocation),
                 lhsVar2->getNameInfo().getAsString() + " = std::max(" +
                     lhsVar1->getNameInfo().getAsString() + ", " +
                     rhsVar1->getNameInfo().getAsString() + ")");
    } else if (lhsVar1->getDecl() == rhsVar2->getDecl() &&
               rhsVar1->getDecl() == lhsVar2->getDecl()) {
      diag(ifStmt->getIfLoc(), "use std::min instead of <")
          << FixItHint::CreateReplacement(
                 SourceRange(ifLocation, thenLocation),
                 lhsVar2->getNameInfo().getAsString() + " = std::min(" +
                     lhsVar1->getNameInfo().getAsString() + ", " +
                     rhsVar1->getNameInfo().getAsString() + ")");
    }
  } else if (binaryOp->getOpcode() == BO_GT) {
    if (lhsVar1->getDecl() == lhsVar2->getDecl() &&
        rhsVar1->getDecl() == rhsVar2->getDecl()) {
      diag(ifStmt->getIfLoc(), "use std::min instead of >")
          << FixItHint::CreateReplacement(
                 SourceRange(ifLocation, thenLocation),
                 lhsVar2->getNameInfo().getAsString() + " = std::min(" +
                     lhsVar1->getNameInfo().getAsString() + ", " +
                     rhsVar1->getNameInfo().getAsString() + ")");
    } else if (lhsVar1->getDecl() == rhsVar2->getDecl() &&
               rhsVar1->getDecl() == lhsVar2->getDecl()) {
      diag(ifStmt->getIfLoc(), "use std::max instead of >")
          << FixItHint::CreateReplacement(
                 SourceRange(ifLocation, thenLocation),
                 lhsVar2->getNameInfo().getAsString() + " = std::max(" +
                     lhsVar1->getNameInfo().getAsString() + ", " +
                     rhsVar1->getNameInfo().getAsString() + ")");
    }
  }
}

} // namespace clang::tidy::readability
