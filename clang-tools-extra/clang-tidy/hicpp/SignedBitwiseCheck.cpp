//===--- SignedBitwiseCheck.cpp - clang-tidy-------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SignedBitwiseCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang {
namespace tidy {
namespace hicpp {

void SignedBitwiseCheck::registerMatchers(MatchFinder *Finder) {
  const auto SignedIntegerOperand =
      expr(ignoringImpCasts(hasType(isSignedInteger()))).bind("signed_operand");

  // Match binary bitwise operations on signed integer arguments.
  Finder->addMatcher(
      binaryOperator(allOf(anyOf(hasOperatorName("|"), hasOperatorName("&"),
                                 hasOperatorName("^"), hasOperatorName("<<"),
                                 hasOperatorName(">>")),
                           hasEitherOperand(SignedIntegerOperand),
                           hasLHS(hasType(isInteger())),
                           hasRHS(hasType(isInteger()))))
          .bind("binary_signed"),
      this);

  // Match unary operations on signed integer types.
  Finder->addMatcher(unaryOperator(allOf(hasOperatorName("~"),
                                         hasUnaryOperand(SignedIntegerOperand)))
                         .bind("unary_signed"),
                     this);
}

void SignedBitwiseCheck::check(const MatchFinder::MatchResult &Result) {
  const ast_matchers::BoundNodes &N = Result.Nodes;
  const auto *SignedBinary = N.getNodeAs<BinaryOperator>("binary_signed");
  const auto *SignedUnary = N.getNodeAs<UnaryOperator>("unary_signed");
  const auto *SignedOperand = N.getNodeAs<Expr>("signed_operand");

  const bool IsUnary = SignedUnary != nullptr;
  diag(IsUnary ? SignedUnary->getLocStart() : SignedBinary->getLocStart(),
       "use of a signed integer operand with a %select{binary|unary}0 bitwise "
       "operator")
      << IsUnary << SignedOperand->getSourceRange();
}

} // namespace hicpp
} // namespace tidy
} // namespace clang
