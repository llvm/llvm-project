//===--- DurationDivisionCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DurationDivisionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace tidy {
namespace abseil {

using namespace clang::ast_matchers;

void DurationDivisionCheck::registerMatchers(MatchFinder *finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  const auto DurationExpr =
      expr(hasType(cxxRecordDecl(hasName("::absl::Duration"))));
  finder->addMatcher(
      implicitCastExpr(
          hasSourceExpression(ignoringParenCasts(
              cxxOperatorCallExpr(hasOverloadedOperatorName("/"),
                                  hasArgument(0, DurationExpr),
                                  hasArgument(1, DurationExpr))
                  .bind("OpCall"))),
          hasImplicitDestinationType(qualType(unless(isInteger()))),
          unless(hasParent(cxxStaticCastExpr())),
          unless(hasParent(cStyleCastExpr())),
          unless(isInTemplateInstantiation())),
      this);
}

void DurationDivisionCheck::check(const MatchFinder::MatchResult &result) {
  const auto *OpCall = result.Nodes.getNodeAs<CXXOperatorCallExpr>("OpCall");
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
                 result.SourceManager->getSpellingLoc(OpCall->getEndLoc()), 0,
                 *result.SourceManager, result.Context->getLangOpts()),
             ")");
}

} // namespace abseil
} // namespace tidy
} // namespace clang
