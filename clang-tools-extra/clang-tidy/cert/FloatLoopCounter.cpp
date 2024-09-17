//===--- FloatLoopCounter.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FloatLoopCounter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cert {

void FloatLoopCounter::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      forStmt(hasIncrement(forEachDescendant(
                  declRefExpr(hasType(realFloatingPointType()),
                              to(varDecl().bind("var")))
                      .bind("inc"))),
              hasCondition(forEachDescendant(
                  declRefExpr(hasType(realFloatingPointType()),
                              to(varDecl(equalsBoundNode("var"))))
                      .bind("cond"))))
          .bind("for"),
      this);
}

void FloatLoopCounter::check(const MatchFinder::MatchResult &Result) {
  const auto *FS = Result.Nodes.getNodeAs<ForStmt>("for");

  diag(FS->getInc()->getBeginLoc(), "loop induction expression should not have "
                                    "floating-point type")
      << Result.Nodes.getNodeAs<DeclRefExpr>("inc")->getSourceRange()
      << Result.Nodes.getNodeAs<DeclRefExpr>("cond")->getSourceRange();

  if (!FS->getInc()->getType()->isRealFloatingType())
    if (const auto *V = Result.Nodes.getNodeAs<VarDecl>("var"))
      diag(V->getBeginLoc(), "floating-point type loop induction variable",
           DiagnosticIDs::Note);
}

} // namespace clang::tidy::cert
