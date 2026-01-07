//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectRoundingsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static llvm::APFloat getHalf(const llvm::fltSemantics &Semantics) {
  return llvm::APFloat(Semantics, 1U) / llvm::APFloat(Semantics, 2U);
}

namespace {
AST_MATCHER(FloatingLiteral, floatHalf) {
  return Node.getValue() == getHalf(Node.getSemantics());
}
} // namespace

void IncorrectRoundingsCheck::registerMatchers(MatchFinder *MatchFinder) {
  // Match a floating literal with value 0.5.
  auto FloatHalf = floatLiteral(floatHalf());

  // Match a floating point expression.
  auto FloatType = expr(hasType(realFloatingPointType()));

  // Find expressions of cast to int of the sum of a floating point expression
  // and 0.5.
  MatchFinder->addMatcher(
      traverse(TK_AsIs,
               implicitCastExpr(
                   hasImplicitDestinationType(isInteger()),
                   ignoringParenCasts(binaryOperator(
                       hasOperatorName("+"), hasOperands(FloatType, FloatType),
                       hasEitherOperand(ignoringParenImpCasts(FloatHalf)))))
                   .bind("CastExpr")),
      this);
}

void IncorrectRoundingsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CastExpr = Result.Nodes.getNodeAs<ImplicitCastExpr>("CastExpr");
  diag(CastExpr->getBeginLoc(),
       "casting (double + 0.5) to integer leads to incorrect rounding; "
       "consider using lround (#include <cmath>) instead");
}

} // namespace clang::tidy::bugprone
