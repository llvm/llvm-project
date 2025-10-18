
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InconsistentIfelseBracesCheck.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

/// Check that at least one branch of the \p If statement is a \c CompoundStmt.
static bool shouldHaveBraces(const IfStmt *If) {
  const Stmt *const Then = If->getThen();
  if (isa<CompoundStmt>(Then))
    return true;

  const Stmt *const Else = If->getElse();
  if (const auto *NestedIf = dyn_cast<const IfStmt>(Else))
    return shouldHaveBraces(NestedIf);

  return isa<CompoundStmt>(Else);
}

/// Check that all branchs of the \p If statement is are \c CompoundStmt.
static bool doesHaveBraces(const IfStmt *If) {
  const Stmt *const Then = If->getThen();
  if (!isa<CompoundStmt>(Then))
    return false;

  const Stmt *const Else = If->getElse();
  if (const auto *NestedIf = dyn_cast<const IfStmt>(Else))
    return doesHaveBraces(NestedIf);

  return isa<CompoundStmt>(Else);
}

void InconsistentIfelseBracesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(TK_IgnoreUnlessSpelledInSource,
               ifStmt(hasElse(anything()), unless(hasParent(ifStmt())))
                   .bind("if_stmt")),
      this);
}

void InconsistentIfelseBracesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedIf = Result.Nodes.getNodeAs<IfStmt>("if_stmt");

  if (shouldHaveBraces(MatchedIf) && !doesHaveBraces(MatchedIf)) {
    diag(MatchedIf->getBeginLoc(), "bad!") << MatchedIf->getSourceRange();
  }
}

} // namespace clang::tidy::bugprone
