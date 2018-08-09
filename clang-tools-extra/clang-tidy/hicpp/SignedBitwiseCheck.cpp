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
      expr(ignoringImpCasts(hasType(isSignedInteger()))).bind("signed-operand");

  // The standard [bitmask.types] allows some integral types to be implemented
  // as signed types. Exclude these types from diagnosing for bitwise or(|) and
  // bitwise and(&). Shifting and complementing such values is still not
  // allowed.
  const auto BitmaskType = namedDecl(anyOf(
      hasName("::std::locale::category"), hasName("::std::ctype_base::mask"),
      hasName("::std::ios_base::fmtflags"), hasName("::std::ios_base::iostate"),
      hasName("::std::ios_base::openmode")));
  const auto IsStdBitmask = ignoringImpCasts(declRefExpr(hasType(BitmaskType)));

  // Match binary bitwise operations on signed integer arguments.
  Finder->addMatcher(
      binaryOperator(
          allOf(anyOf(hasOperatorName("^"), hasOperatorName("|"),
                      hasOperatorName("&"), hasOperatorName("^="),
                      hasOperatorName("|="), hasOperatorName("&=")),

                unless(allOf(hasLHS(IsStdBitmask), hasRHS(IsStdBitmask))),

                hasEitherOperand(SignedIntegerOperand),
                hasLHS(hasType(isInteger())), hasRHS(hasType(isInteger()))))
          .bind("binary-no-sign-interference"),
      this);

  // Shifting and complement is not allowed for any signed integer type because
  // the sign bit may corrupt the result.
  Finder->addMatcher(
      binaryOperator(
          allOf(anyOf(hasOperatorName("<<"), hasOperatorName(">>"),
                      hasOperatorName("<<="), hasOperatorName(">>=")),
                hasEitherOperand(SignedIntegerOperand),
                hasLHS(hasType(isInteger())), hasRHS(hasType(isInteger()))))
          .bind("binary-sign-interference"),
      this);

  // Match unary operations on signed integer types.
  Finder->addMatcher(unaryOperator(allOf(hasOperatorName("~"),
                                         hasUnaryOperand(SignedIntegerOperand)))
                         .bind("unary-signed"),
                     this);
}

void SignedBitwiseCheck::check(const MatchFinder::MatchResult &Result) {
  const ast_matchers::BoundNodes &N = Result.Nodes;
  const auto *SignedOperand = N.getNodeAs<Expr>("signed-operand");
  assert(SignedOperand &&
         "No signed operand found in problematic bitwise operations");

  bool IsUnary = false;
  SourceLocation Location;

  if (const auto *UnaryOp = N.getNodeAs<UnaryOperator>("unary-signed")) {
    IsUnary = true;
    Location = UnaryOp->getBeginLoc();
  } else {
    if (const auto *BinaryOp =
            N.getNodeAs<BinaryOperator>("binary-no-sign-interference"))
      Location = BinaryOp->getBeginLoc();
    else if (const auto *BinaryOp =
                 N.getNodeAs<BinaryOperator>("binary-sign-interference"))
      Location = BinaryOp->getBeginLoc();
    else
      llvm_unreachable("unexpected matcher result");
  }
  diag(Location, "use of a signed integer operand with a "
                 "%select{binary|unary}0 bitwise operator")
      << IsUnary << SignedOperand->getSourceRange();
}

} // namespace hicpp
} // namespace tidy
} // namespace clang
