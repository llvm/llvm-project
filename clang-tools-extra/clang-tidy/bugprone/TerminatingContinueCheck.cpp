//===--- TerminatingContinueCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TerminatingContinueCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void TerminatingContinueCheck::registerMatchers(MatchFinder *Finder) {
  const auto doWithFalse =
      doStmt(hasCondition(ignoringImpCasts(
                 anyOf(cxxBoolLiteral(equals(false)), integerLiteral(equals(0)),
                       cxxNullPtrLiteralExpr(), gnuNullExpr()))),
             equalsBoundNode("closestLoop"));

  Finder->addMatcher(
      continueStmt(hasAncestor(stmt(anyOf(forStmt(), whileStmt(),
                                          cxxForRangeStmt(), doStmt()))
                                   .bind("closestLoop")),
                   hasAncestor(doWithFalse))
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

} // namespace bugprone
} // namespace tidy
} // namespace clang
