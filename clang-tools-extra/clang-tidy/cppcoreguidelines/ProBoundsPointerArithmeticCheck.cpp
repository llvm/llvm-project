//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProBoundsPointerArithmeticCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

ProBoundsPointerArithmeticCheck::ProBoundsPointerArithmeticCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowIncrementDecrementOperators(
          Options.get("AllowIncrementDecrementOperators", false)) {}

void ProBoundsPointerArithmeticCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowIncrementDecrementOperators",
                AllowIncrementDecrementOperators);
}

void ProBoundsPointerArithmeticCheck::registerMatchers(MatchFinder *Finder) {
  const auto AllPointerTypes =
      anyOf(hasType(hasUnqualifiedDesugaredType(pointerType())),
            hasType(autoType(
                hasDeducedType(hasUnqualifiedDesugaredType(pointerType())))),
            hasType(decltypeType(hasUnderlyingType(pointerType()))));

  // Flag all operators +, -, +=, -= that result in a pointer
  Finder->addMatcher(
      binaryOperator(
          hasAnyOperatorName("+", "-", "+=", "-="), AllPointerTypes,
          unless(hasLHS(ignoringImpCasts(declRefExpr(to(isImplicit()))))))
          .bind("expr"),
      this);

  // Flag all operators ++, -- that result in a pointer
  if (!AllowIncrementDecrementOperators)
    Finder->addMatcher(
        unaryOperator(hasAnyOperatorName("++", "--"),
                      hasType(hasUnqualifiedDesugaredType(pointerType())),
                      unless(hasUnaryOperand(
                          ignoringImpCasts(declRefExpr(to(isImplicit()))))))
            .bind("expr"),
        this);

  // Array subscript on a pointer (not an array) is also pointer arithmetic
  Finder->addMatcher(
      arraySubscriptExpr(
          hasBase(ignoringImpCasts(
              anyOf(AllPointerTypes,
                    hasType(decayedType(hasDecayedType(pointerType())))))),
          hasIndex(hasType(isInteger())))
          .bind("expr"),
      this);
}

void ProBoundsPointerArithmeticCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<Expr>("expr");

  diag(MatchedExpr->getExprLoc(), "do not use pointer arithmetic");
}

} // namespace clang::tidy::cppcoreguidelines
