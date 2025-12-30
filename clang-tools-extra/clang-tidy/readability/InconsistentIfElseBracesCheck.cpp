//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InconsistentIfElseBracesCheck.h"
#include "../utils/BracesAroundStatement.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

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

void InconsistentIfElseBracesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      ifStmt(hasElse(anything()),
             unless(isConsteval()), // 'if consteval' always has braces
             unless(hasParent(ifStmt())))
          .bind("if_stmt"),
      this);
}

void InconsistentIfElseBracesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedIf = Result.Nodes.getNodeAs<IfStmt>("if_stmt");
  if (!shouldHaveBraces(MatchedIf))
    return;
  checkIfStmt(Result, MatchedIf);
}

void InconsistentIfElseBracesCheck::checkIfStmt(
    const MatchFinder::MatchResult &Result, const IfStmt *If) {
  const Stmt *Then = If->getThen();
  if (const auto *NestedIf = dyn_cast<const IfStmt>(Then)) {
    // If the then-branch is a nested IfStmt, first we need to add braces to
    // it, then we need to check the inner IfStmt.
    emitDiagnostic(Result, If->getThen(), If->getRParenLoc(), If->getElseLoc());

    if (shouldHaveBraces(NestedIf))
      checkIfStmt(Result, NestedIf);
  } else if (!isa<CompoundStmt>(Then)) {
    emitDiagnostic(Result, Then, If->getRParenLoc(), If->getElseLoc());
  }

  if (const Stmt *const Else = If->getElse()) {
    if (const auto *NestedIf = dyn_cast<const IfStmt>(Else))
      checkIfStmt(Result, NestedIf);
    else if (!isa<CompoundStmt>(Else))
      emitDiagnostic(Result, If->getElse(), If->getElseLoc());
  }
}

void InconsistentIfElseBracesCheck::emitDiagnostic(
    const MatchFinder::MatchResult &Result, const Stmt *S,
    SourceLocation StartLoc, SourceLocation EndLocHint) {
  if (StartLoc.isMacroID()) {
    diag(StartLoc, "statement should have braces");
    return;
  }
  const utils::BraceInsertionHints Hints = utils::getBraceInsertionsHints(
      S, Result.Context->getLangOpts(), *Result.SourceManager, StartLoc,
      EndLocHint);
  assert(Hints && Hints.offersFixIts() && "Expected hints or fix-its");
  diag(Hints.DiagnosticPos, "statement should have braces")
      << Hints.openingBraceFixIt() << Hints.closingBraceFixIt();
}

} // namespace clang::tidy::readability
