//===--- CastingThroughVoidCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CastingThroughVoidCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void CastingThroughVoidCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      explicitCastExpr(
          hasDestinationType(qualType(pointsTo(qualType(unless(voidType()))))
                                 .bind("target_type")),
          hasSourceExpression(
              explicitCastExpr(hasSourceExpression(expr(
                                   hasType(qualType().bind("source_type")))),
                               hasDestinationType(pointsTo(voidType())))
                  .bind("cast"))),
      this);
}

void CastingThroughVoidCheck::check(const MatchFinder::MatchResult &Result) {
  const auto TT = *Result.Nodes.getNodeAs<QualType>("target_type");
  const auto ST = *Result.Nodes.getNodeAs<QualType>("source_type");
  if (Result.Context->hasSameType(TT, ST))
    return;
  const auto *CE = Result.Nodes.getNodeAs<ExplicitCastExpr>("cast");
  diag(CE->getSourceRange().getBegin(), "do not cast %0 to %1 through 'void*'")
      << ST << TT;
}

} // namespace clang::tidy::bugprone
