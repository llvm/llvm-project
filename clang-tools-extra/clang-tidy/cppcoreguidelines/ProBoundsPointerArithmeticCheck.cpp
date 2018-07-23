//===--- ProBoundsPointerArithmeticCheck.cpp - clang-tidy------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProBoundsPointerArithmeticCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void ProBoundsPointerArithmeticCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  const auto AllPointerTypes = anyOf(
      hasType(pointerType()), hasType(autoType(hasDeducedType(pointerType()))),
      hasType(decltypeType(hasUnderlyingType(pointerType()))));

  // Flag all operators +, -, +=, -=, ++, -- that result in a pointer
  Finder->addMatcher(
      binaryOperator(
          anyOf(hasOperatorName("+"), hasOperatorName("-"),
                hasOperatorName("+="), hasOperatorName("-=")),
          AllPointerTypes,
          unless(hasLHS(ignoringImpCasts(declRefExpr(to(isImplicit()))))))
          .bind("expr"),
      this);

  Finder->addMatcher(
      unaryOperator(anyOf(hasOperatorName("++"), hasOperatorName("--")),
                    hasType(pointerType()))
          .bind("expr"),
      this);

  // Array subscript on a pointer (not an array) is also pointer arithmetic
  Finder->addMatcher(
      arraySubscriptExpr(
          hasBase(ignoringImpCasts(
              anyOf(AllPointerTypes,
                    hasType(decayedType(hasDecayedType(pointerType())))))))
          .bind("expr"),
      this);
}

void ProBoundsPointerArithmeticCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<Expr>("expr");

  diag(MatchedExpr->getExprLoc(), "do not use pointer arithmetic");
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
