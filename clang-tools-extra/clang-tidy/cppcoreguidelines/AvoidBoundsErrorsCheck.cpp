//===--- AvoidBoundsErrorsCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidBoundsErrorsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

const CXXMethodDecl *findAlternative(const CXXRecordDecl *MatchedParent,
                                     const CXXMethodDecl *MatchedOperator) {
  for (const CXXMethodDecl *Method : MatchedParent->methods()) {
    const bool CorrectName = Method->getNameInfo().getAsString() == "at";
    if (!CorrectName)
      continue;

    const bool SameReturnType =
        Method->getReturnType() == MatchedOperator->getReturnType();
    if (!SameReturnType)
      continue;

    const bool SameNumberOfArguments =
        Method->getNumParams() == MatchedOperator->getNumParams();
    if (!SameNumberOfArguments)
      continue;

    for (unsigned a = 0; a < Method->getNumParams(); a++) {
      const bool SameArgType =
          Method->parameters()[a]->getOriginalType() ==
          MatchedOperator->parameters()[a]->getOriginalType();
      if (!SameArgType)
        continue;
    }

    return Method;
  }
  return static_cast<CXXMethodDecl *>(nullptr);
}

void AvoidBoundsErrorsCheck::registerMatchers(MatchFinder *Finder) {
  // Need a callExpr here to match CXXOperatorCallExpr ``(&a)->operator[](0)``
  // and CXXMemberCallExpr ``a[0]``.
  Finder->addMatcher(
      callExpr(
          callee(
              cxxMethodDecl(hasOverloadedOperatorName("[]")).bind("operator")),
          callee(cxxMethodDecl(hasParent(
              cxxRecordDecl(hasMethod(hasName("at"))).bind("parent")))))
          .bind("caller"),
      this);
}

void AvoidBoundsErrorsCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();
  const CallExpr *MatchedExpr = Result.Nodes.getNodeAs<CallExpr>("caller");
  const CXXMethodDecl *MatchedOperator =
      Result.Nodes.getNodeAs<CXXMethodDecl>("operator");
  const CXXRecordDecl *MatchedParent =
      Result.Nodes.getNodeAs<CXXRecordDecl>("parent");

  const CXXMethodDecl *Alternative =
      findAlternative(MatchedParent, MatchedOperator);
  if (!Alternative)
    return;

  const SourceLocation AlternativeSource(Alternative->getBeginLoc());

  diag(MatchedExpr->getBeginLoc(),
       "found possibly unsafe operator[], consider using at() instead");
  diag(Alternative->getBeginLoc(), "alternative at() defined here",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::cppcoreguidelines
