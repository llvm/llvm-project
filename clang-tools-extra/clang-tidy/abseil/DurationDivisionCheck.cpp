//===--- DurationDivisionCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DurationDivisionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace tidy {
namespace abseil {

using namespace clang::ast_matchers;

void DurationDivisionCheck::registerMatchers(MatchFinder *Finder) {
  const auto DurationExpr =
      expr(hasType(cxxRecordDecl(hasName("::absl::Duration"))));
  Finder->addMatcher(
      traverse(TK_AsIs,
               implicitCastExpr(
                   hasSourceExpression(ignoringParenCasts(
                       cxxOperatorCallExpr(hasOverloadedOperatorName("/"),
                                           hasArgument(0, DurationExpr),
                                           hasArgument(1, DurationExpr))
                           .bind("OpCall"))),
                   hasImplicitDestinationType(qualType(unless(isInteger()))),
                   unless(hasParent(cxxStaticCastExpr())),
                   unless(hasParent(cStyleCastExpr())),
                   unless(isInTemplateInstantiation()))),
      this);
}

void DurationDivisionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *OpCall = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("OpCall");
  diag(OpCall->getOperatorLoc(),
       "operator/ on absl::Duration objects performs integer division; "
       "did you mean to use FDivDuration()?")
      << FixItHint::CreateInsertion(OpCall->getBeginLoc(),
                                    "absl::FDivDuration(")
      << FixItHint::CreateReplacement(
             SourceRange(OpCall->getOperatorLoc(), OpCall->getOperatorLoc()),
             ", ")
      << FixItHint::CreateInsertion(
             Lexer::getLocForEndOfToken(
                 Result.SourceManager->getSpellingLoc(OpCall->getEndLoc()), 0,
                 *Result.SourceManager, Result.Context->getLangOpts()),
             ")");
}

} // namespace abseil
} // namespace tidy
} // namespace clang
