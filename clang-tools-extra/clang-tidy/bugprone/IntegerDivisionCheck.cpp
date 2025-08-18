//===--- IntegerDivisionCheck.cpp - clang-tidy-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IntegerDivisionCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void IntegerDivisionCheck::registerMatchers(MatchFinder *Finder) {
  const auto IntType = hasType(isInteger());

  const auto BinaryOperators = binaryOperator(
      hasAnyOperatorName("%", "<<", ">>", "<<", "^", "|", "&", "||", "&&", "<",
                         ">", "<=", ">=", "==", "!="));

  const auto UnaryOperators = unaryOperator(hasAnyOperatorName("~", "!"));

  const auto Exceptions =
      anyOf(BinaryOperators, conditionalOperator(), binaryConditionalOperator(),
            callExpr(IntType), explicitCastExpr(IntType), UnaryOperators);

  Finder->addMatcher(
      traverse(TK_AsIs,
               binaryOperator(
                   hasOperatorName("/"), hasLHS(expr(IntType)),
                   hasRHS(expr(IntType)),
                   hasAncestor(castExpr(hasCastKind(CK_IntegralToFloating))
                                   .bind("FloatCast")),
                   unless(hasAncestor(expr(
                       Exceptions,
                       hasAncestor(castExpr(equalsBoundNode("FloatCast")))))))
                   .bind("IntDiv")),
      this);
}

void IntegerDivisionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *IntDiv = Result.Nodes.getNodeAs<BinaryOperator>("IntDiv");
  diag(IntDiv->getBeginLoc(), "result of integer division used in a floating "
                              "point context; possible loss of precision");
}

} // namespace clang::tidy::bugprone
