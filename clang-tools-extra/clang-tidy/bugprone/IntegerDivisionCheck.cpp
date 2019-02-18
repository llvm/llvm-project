//===--- IntegerDivisionCheck.cpp - clang-tidy-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IntegerDivisionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void IntegerDivisionCheck::registerMatchers(MatchFinder *Finder) {
  const auto IntType = hasType(isInteger());

  const auto BinaryOperators = binaryOperator(anyOf(
      hasOperatorName("%"), hasOperatorName("<<"), hasOperatorName(">>"),
      hasOperatorName("<<"), hasOperatorName("^"), hasOperatorName("|"),
      hasOperatorName("&"), hasOperatorName("||"), hasOperatorName("&&"),
      hasOperatorName("<"), hasOperatorName(">"), hasOperatorName("<="),
      hasOperatorName(">="), hasOperatorName("=="), hasOperatorName("!=")));

  const auto UnaryOperators =
      unaryOperator(anyOf(hasOperatorName("~"), hasOperatorName("!")));

  const auto Exceptions =
      anyOf(BinaryOperators, conditionalOperator(), binaryConditionalOperator(),
            callExpr(IntType), explicitCastExpr(IntType), UnaryOperators);

  Finder->addMatcher(
      binaryOperator(
          hasOperatorName("/"), hasLHS(expr(IntType)), hasRHS(expr(IntType)),
          hasAncestor(
              castExpr(hasCastKind(CK_IntegralToFloating)).bind("FloatCast")),
          unless(hasAncestor(
              expr(Exceptions,
                   hasAncestor(castExpr(equalsBoundNode("FloatCast")))))))
          .bind("IntDiv"),
      this);
}

void IntegerDivisionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *IntDiv = Result.Nodes.getNodeAs<BinaryOperator>("IntDiv");
  diag(IntDiv->getBeginLoc(), "result of integer division used in a floating "
                              "point context; possible loss of precision");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
