//===--- TerminatingContinueCheck.cpp - clang-tidy-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TerminatingContinueCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void TerminatingContinueCheck::registerMatchers(MatchFinder *Finder) {
  const auto DoWithFalse =
      doStmt(hasCondition(ignoringImpCasts(
                 anyOf(cxxBoolLiteral(equals(false)), integerLiteral(equals(0)),
                       cxxNullPtrLiteralExpr(), gnuNullExpr()))),
             equalsBoundNode("closestLoop"));

  Finder->addMatcher(
      continueStmt(
          hasAncestor(stmt(anyOf(forStmt(), whileStmt(), cxxForRangeStmt(),
                                 doStmt(), switchStmt()))
                          .bind("closestLoop")),
          hasAncestor(DoWithFalse))
          .bind("continue"),
      this);
}

void TerminatingContinueCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ContStmt = Result.Nodes.getNodeAs<ContinueStmt>("continue");

  auto Diag =
      diag(ContStmt->getBeginLoc(),
           "'continue' in loop with false condition is equivalent to 'break'")
      << tooling::fixit::createReplacement(*ContStmt, "break");
}

} // namespace clang::tidy::bugprone
