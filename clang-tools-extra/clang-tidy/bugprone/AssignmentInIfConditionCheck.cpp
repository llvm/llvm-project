//===--- AssignmentInIfConditionCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AssignmentInIfConditionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void AssignmentInIfConditionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(ifStmt(hasCondition(forEachDescendant(
                         binaryOperator(isAssignmentOperator())
                             .bind("assignment_in_if_statement")))),
                     this);
  Finder->addMatcher(ifStmt(hasCondition(forEachDescendant(
                         cxxOperatorCallExpr(isAssignmentOperator())
                             .bind("assignment_in_if_statement")))),
                     this);
}

void AssignmentInIfConditionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<clang::Stmt>("assignment_in_if_statement");
  if (!MatchedDecl) {
    return;
  }
  diag(MatchedDecl->getBeginLoc(),
       "an assignment within an 'if' condition is bug-prone");
  diag(MatchedDecl->getBeginLoc(),
       "if it should be an assignment, move it out of the 'if' condition",
       DiagnosticIDs::Note);
  diag(MatchedDecl->getBeginLoc(),
       "if it is meant to be an equality check, change '=' to '=='",
       DiagnosticIDs::Note);
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
