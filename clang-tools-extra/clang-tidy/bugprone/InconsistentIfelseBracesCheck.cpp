//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InconsistentIfelseBracesCheck.h"
#include "../utils/BracesAroundStatement.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

/// Check that at least one branch of the \p If statement is a \c CompoundStmt.
static bool shouldHaveBraces(const IfStmt *If) {
  const Stmt *const Then = If->getThen();
  if (isa<CompoundStmt>(Then))
    return true;

  if (const Stmt *const Else = If->getElse()) {
    if (const auto *NestedIf = dyn_cast<const IfStmt>(Else))
      return shouldHaveBraces(NestedIf);

    return isa<CompoundStmt>(Else);
  }

  return false;
}

void InconsistentIfelseBracesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               ifStmt(hasElse(anything()),
                      unless(isConsteval()), // 'if consteval' always has braces
                      unless(hasParent(ifStmt())))
                   .bind("if_stmt")),
      this);
}

void InconsistentIfelseBracesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedIf = Result.Nodes.getNodeAs<IfStmt>("if_stmt");
  if (!shouldHaveBraces(MatchedIf))
    return;
  checkIfStmt(Result, MatchedIf);
}

void InconsistentIfelseBracesCheck::checkIfStmt(
    const ast_matchers::MatchFinder::MatchResult &Result, const IfStmt *If) {
  const Stmt *Then = If->getThen();
  if (const auto *NestedIf = dyn_cast<const IfStmt>(Then)) {
    // If the then-branch is a nested IfStmt, first we need to add braces to
    // it, then we need to check the inner IfStmt.
    checkStmt(Result, If->getThen(), If->getRParenLoc(), If->getElseLoc());
    if (shouldHaveBraces(NestedIf))
      checkIfStmt(Result, NestedIf);
  } else if (!isa<CompoundStmt>(Then)) {
    checkStmt(Result, If->getThen(), If->getRParenLoc(), If->getElseLoc());
  }

  if (const Stmt *const Else = If->getElse()) {
    if (const auto *NestedIf = dyn_cast<const IfStmt>(Else))
      checkIfStmt(Result, NestedIf);
    else if (!isa<CompoundStmt>(Else))
      checkStmt(Result, If->getElse(), If->getElseLoc());
  }
}

void InconsistentIfelseBracesCheck::checkStmt(
    const ast_matchers::MatchFinder::MatchResult &Result, const Stmt *S,
    SourceLocation StartLoc, SourceLocation EndLocHint) {
  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = Result.Context->getLangOpts();

  const utils::BraceInsertionHints Hints =
      utils::getBraceInsertionsHints(S, LangOpts, SM, StartLoc, EndLocHint);
  if (Hints) {
    DiagnosticBuilder Diag = diag(Hints.DiagnosticPos, "<message>");
    if (Hints.offersFixIts()) {
      Diag << Hints.openingBraceFixIt() << Hints.closingBraceFixIt();
    }
  }
}

} // namespace clang::tidy::bugprone
