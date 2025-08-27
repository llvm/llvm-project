//===--- DefaultOperatorNewCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefaultOperatorNewAlignmentCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cert {

void DefaultOperatorNewAlignmentCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxNewExpr(unless(hasAnyPlacementArg(anything()))).bind("new"), this);
}

void DefaultOperatorNewAlignmentCheck::check(
    const MatchFinder::MatchResult &Result) {
  // Get the found 'new' expression.
  const auto *NewExpr = Result.Nodes.getNodeAs<CXXNewExpr>("new");

  QualType T = NewExpr->getAllocatedType();
  // Dependent types do not have fixed alignment.
  if (T->isDependentType())
    return;
  const TagDecl *D = T->getAsTagDecl();
  // Alignment can not be obtained for undefined type.
  if (!D || !D->isCompleteDefinition())
    return;

  ASTContext &Context = D->getASTContext();

  // Check if no alignment was specified for the type.
  if (!Context.isAlignmentRequired(T))
    return;

  // The user-specified alignment (in bits).
  unsigned SpecifiedAlignment = D->getMaxAlignment();
  // Double-check if no alignment was specified.
  if (!SpecifiedAlignment)
    return;
  // The alignment used by default 'operator new' (in bits).
  unsigned DefaultNewAlignment = Context.getTargetInfo().getNewAlign();

  bool OverAligned = SpecifiedAlignment > DefaultNewAlignment;
  bool HasDefaultOperatorNew =
      !NewExpr->getOperatorNew() || NewExpr->getOperatorNew()->isImplicit();

  unsigned CharWidth = Context.getTargetInfo().getCharWidth();
  if (HasDefaultOperatorNew && OverAligned)
    diag(NewExpr->getBeginLoc(),
         "allocation function returns a pointer with alignment %0 but the "
         "over-aligned type being allocated requires alignment %1")
        << (DefaultNewAlignment / CharWidth)
        << (SpecifiedAlignment / CharWidth);
}

} // namespace clang::tidy::cert
