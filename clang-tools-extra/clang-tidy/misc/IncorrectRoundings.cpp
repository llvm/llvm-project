//===--- IncorrectRoundings.cpp - clang-tidy ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IncorrectRoundings.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace ast_matchers {
AST_MATCHER(FloatingLiteral, floatHalf) {
  const auto &literal = Node.getValue();
  if ((&Node.getSemantics()) == &llvm::APFloat::IEEEsingle)
    return literal.convertToFloat() == 0.5f;
  if ((&Node.getSemantics()) == &llvm::APFloat::IEEEdouble)
    return literal.convertToDouble() == 0.5;
  return false;
}

// TODO(hokein): Moving it to ASTMatchers.h
AST_MATCHER(BuiltinType, isFloatingPoint) {
  return Node.isFloatingPoint();
}
} // namespace ast_matchers
} // namespace clang

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {
void IncorrectRoundings::registerMatchers(MatchFinder *MatchFinder) {
  // Match a floating literal with value 0.5.
  auto FloatHalf = floatLiteral(floatHalf());

  // Match a floating point expression.
  auto FloatType = expr(hasType(builtinType(isFloatingPoint())));

  // Match a floating literal of 0.5 or a floating literal of 0.5 implicitly.
  // cast to floating type.
  auto FloatOrCastHalf =
      anyOf(FloatHalf, implicitCastExpr(FloatType, has(FloatHalf)));

  // Match if either the LHS or RHS is a floating literal of 0.5 or a floating
  // literal of 0.5 and the other is of type double or vice versa.
  auto OneSideHalf = anyOf(allOf(hasLHS(FloatOrCastHalf), hasRHS(FloatType)),
                           allOf(hasRHS(FloatOrCastHalf), hasLHS(FloatType)));

  // Find expressions of cast to int of the sum of a floating point expression
  // and 0.5.
  MatchFinder->addMatcher(
      implicitCastExpr(
          hasImplicitDestinationType(isInteger()),
          ignoringParenCasts(binaryOperator(hasOperatorName("+"), OneSideHalf)))
          .bind("CastExpr"),
      this);
}

void IncorrectRoundings::check(const MatchFinder::MatchResult &Result) {
  const auto *CastExpr = Result.Nodes.getStmtAs<ImplicitCastExpr>("CastExpr");
  diag(CastExpr->getLocStart(),
       "casting (double + 0.5) to integer leads to incorrect rounding; "
       "consider using lround (#include <cmath>) instead");
}

} // namespace misc
} // namespace tidy
} // namespace clang
