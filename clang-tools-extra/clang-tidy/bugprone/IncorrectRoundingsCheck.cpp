//===--- IncorrectRoundingsCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectRoundingsCheck.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
AST_MATCHER(FloatingLiteral, floatHalf) {
  const auto &Literal = Node.getValue();
  if ((&Node.getSemantics()) == &llvm::APFloat::IEEEsingle())
    return Literal.convertToFloat() == 0.5f;
  if ((&Node.getSemantics()) == &llvm::APFloat::IEEEdouble())
    return Literal.convertToDouble() == 0.5;
  return false;
}
} // namespace

void IncorrectRoundingsCheck::registerMatchers(MatchFinder *MatchFinder) {
  // Match a floating literal with value 0.5.
  auto FloatHalf = floatLiteral(floatHalf());

  // Match a floating point expression.
  auto FloatType = expr(hasType(realFloatingPointType()));

  // Match a floating literal of 0.5 or a floating literal of 0.5 implicitly.
  // cast to floating type.
  auto FloatOrCastHalf =
      anyOf(FloatHalf,
            implicitCastExpr(FloatType, has(ignoringParenImpCasts(FloatHalf))));

  // Match if either the LHS or RHS is a floating literal of 0.5 or a floating
  // literal of 0.5 and the other is of type double or vice versa.
  auto OneSideHalf = anyOf(allOf(hasLHS(FloatOrCastHalf), hasRHS(FloatType)),
                           allOf(hasRHS(FloatOrCastHalf), hasLHS(FloatType)));

  // Find expressions of cast to int of the sum of a floating point expression
  // and 0.5.
  MatchFinder->addMatcher(
      traverse(TK_AsIs,
               implicitCastExpr(hasImplicitDestinationType(isInteger()),
                                ignoringParenCasts(binaryOperator(
                                    hasOperatorName("+"), OneSideHalf)))
                   .bind("CastExpr")),
      this);
}

void IncorrectRoundingsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *CastExpr = Result.Nodes.getNodeAs<ImplicitCastExpr>("CastExpr");
  diag(CastExpr->getBeginLoc(),
       "casting (double + 0.5) to integer leads to incorrect rounding; "
       "consider using lround (#include <cmath>) instead");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
