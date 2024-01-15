//===--- UseStdMinMaxCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdMinMaxCheck.h"
#include "clang/AST/ASTContext.h"
#include "../utils/ASTUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void UseStdMinMaxCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      ifStmt(
          has(binaryOperator(
              anyOf(hasOperatorName("<"), hasOperatorName(">"),
                    hasOperatorName("<="), hasOperatorName(">=")),
              hasLHS(expr().bind("lhsVar1")), hasRHS(expr().bind("rhsVar1")))),
          hasThen(stmt(binaryOperator(hasOperatorName("="),
                                      hasLHS(expr().bind("lhsVar2")),
                                      hasRHS(expr().bind("rhsVar2"))))))
          .bind("ifStmt"),
      this);
}

void UseStdMinMaxCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *lhsVar1 = Result.Nodes.getNodeAs<Expr>("lhsVar1");
  const auto *rhsVar1 = Result.Nodes.getNodeAs<Expr>("rhsVar1");
  const auto *lhsVar2 = Result.Nodes.getNodeAs<Expr>("lhsVar2");
  const auto *rhsVar2 = Result.Nodes.getNodeAs<Expr>("rhsVar2");
  const auto *ifStmt = Result.Nodes.getNodeAs<IfStmt>("ifStmt");
  auto &Context = *Result.Context;

  if (!lhsVar1 || !rhsVar1 || !lhsVar2 || !rhsVar2 || !ifStmt)
    return;

  const auto *binaryOp = dyn_cast<BinaryOperator>(ifStmt->getCond());
  if (!binaryOp)
    return;

  SourceLocation ifLocation = ifStmt->getIfLoc();
  SourceLocation thenLocation = ifStmt->getEndLoc();

  auto lhsVar1Str = Lexer::getSourceText(
      CharSourceRange::getTokenRange(lhsVar1->getSourceRange()),
      Context.getSourceManager(), Context.getLangOpts());

  auto lhsVar2Str = Lexer::getSourceText(
      CharSourceRange::getTokenRange(lhsVar2->getSourceRange()),
      Context.getSourceManager(), Context.getLangOpts());

  auto rhsVar1Str = Lexer::getSourceText(
      CharSourceRange::getTokenRange(rhsVar1->getSourceRange()),
      Context.getSourceManager(), Context.getLangOpts());
  
  auto replacementMax = lhsVar2Str.str() + " = std::max(" + lhsVar1Str.str() + ", " + rhsVar1Str.str() + ")";
  auto replacementMin = lhsVar2Str.str() + " = std::min(" + lhsVar1Str.str() + ", " + rhsVar1Str.str() + ")";
  auto *operatorStr = binaryOp->getOpcodeStr().data();

  if (binaryOp->getOpcode() == BO_LT || binaryOp->getOpcode() == BO_LE) {
    if (tidy::utils::areStatementsIdentical(lhsVar1, lhsVar2,Context) &&
        tidy::utils::areStatementsIdentical(rhsVar1, rhsVar2,Context)) {
      diag(ifStmt->getIfLoc(), "use `std::max` instead of `%0`")<< operatorStr
          << FixItHint::CreateReplacement(SourceRange(ifLocation, thenLocation),
                                    std::move(replacementMax));
    } else if (tidy::utils::areStatementsIdentical(lhsVar1, rhsVar2,Context) &&
               tidy::utils::areStatementsIdentical(rhsVar1, lhsVar2,Context)) {
      diag(ifStmt->getIfLoc(), "use `std::min` instead of `%0`")<< operatorStr
          << FixItHint::CreateReplacement(SourceRange(ifLocation, thenLocation),
                                    std::move(replacementMin));
    }
  } else if (binaryOp->getOpcode() == BO_GT || binaryOp->getOpcode() == BO_GE) {
    if (tidy::utils::areStatementsIdentical(lhsVar1, lhsVar2,Context) &&
        tidy::utils::areStatementsIdentical(rhsVar1, rhsVar2,Context)) {
      diag(ifStmt->getIfLoc(), "use `std::min` instead of `%0`")<< operatorStr
          << FixItHint::CreateReplacement(SourceRange(ifLocation, thenLocation),
                                    std::move(replacementMin));
    } else if (tidy::utils::areStatementsIdentical(lhsVar1, rhsVar2,Context) &&
               tidy::utils::areStatementsIdentical(rhsVar1, lhsVar2,Context)) {
      diag(ifStmt->getIfLoc(), "use `std::max` instead of `%0`")<< operatorStr
          << FixItHint::CreateReplacement(SourceRange(ifLocation, thenLocation),
                                    std::move(replacementMax));
    }
  }
}

} // namespace clang::tidy::readability
