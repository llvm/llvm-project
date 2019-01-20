//===--- MisplacedArrayIndexCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MisplacedArrayIndexCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void MisplacedArrayIndexCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(arraySubscriptExpr(hasLHS(hasType(isInteger())),
                                        hasRHS(hasType(isAnyPointer())))
                         .bind("expr"),
                     this);
}

void MisplacedArrayIndexCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ArraySubscriptE =
      Result.Nodes.getNodeAs<ArraySubscriptExpr>("expr");

  auto Diag = diag(ArraySubscriptE->getBeginLoc(), "confusing array subscript "
                                                   "expression, usually the "
                                                   "index is inside the []");

  // Only try to fixit when LHS and RHS can be swapped directly without changing
  // the logic.
  const Expr *RHSE = ArraySubscriptE->getRHS()->IgnoreParenImpCasts();
  if (!isa<StringLiteral>(RHSE) && !isa<DeclRefExpr>(RHSE) &&
      !isa<MemberExpr>(RHSE))
    return;

  const StringRef LText = tooling::fixit::getText(
      ArraySubscriptE->getLHS()->getSourceRange(), *Result.Context);
  const StringRef RText = tooling::fixit::getText(
      ArraySubscriptE->getRHS()->getSourceRange(), *Result.Context);

  Diag << FixItHint::CreateReplacement(
      ArraySubscriptE->getLHS()->getSourceRange(), RText);
  Diag << FixItHint::CreateReplacement(
      ArraySubscriptE->getRHS()->getSourceRange(), LText);
}

} // namespace readability
} // namespace tidy
} // namespace clang
