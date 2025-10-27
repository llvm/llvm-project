//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SignedBitwiseCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang::tidy::hicpp {

SignedBitwiseCheck::SignedBitwiseCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnorePositiveIntegerLiterals(
          Options.get("IgnorePositiveIntegerLiterals", false)) {}

void SignedBitwiseCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnorePositiveIntegerLiterals",
                IgnorePositiveIntegerLiterals);
}

void SignedBitwiseCheck::registerMatchers(MatchFinder *Finder) {
  const auto SignedIntegerOperand =
      (IgnorePositiveIntegerLiterals
           ? expr(ignoringImpCasts(
                 allOf(hasType(isSignedInteger()), unless(integerLiteral()))))
           : expr(ignoringImpCasts(hasType(isSignedInteger()))))
          .bind("signed-operand");

  // The standard [bitmask.types] allows some integral types to be implemented
  // as signed types. Exclude these types from diagnosing for bitwise or(|) and
  // bitwise and(&). Shifting and complementing such values is still not
  // allowed.
  const auto BitmaskType = namedDecl(
      hasAnyName("::std::locale::category", "::std::ctype_base::mask",
                 "::std::ios_base::fmtflags", "::std::ios_base::iostate",
                 "::std::ios_base::openmode"));
  const auto IsStdBitmask = ignoringImpCasts(declRefExpr(hasType(BitmaskType)));

  // Match binary bitwise operations on signed integer arguments.
  Finder->addMatcher(
      binaryOperator(hasAnyOperatorName("^", "|", "&", "^=", "|=", "&="),

                     unless(allOf(hasLHS(IsStdBitmask), hasRHS(IsStdBitmask))),

                     hasEitherOperand(SignedIntegerOperand),
                     hasLHS(hasType(isInteger())), hasRHS(hasType(isInteger())))
          .bind("binary-no-sign-interference"),
      this);

  // Shifting and complement is not allowed for any signed integer type because
  // the sign bit may corrupt the result.
  Finder->addMatcher(
      binaryOperator(hasAnyOperatorName("<<", ">>", "<<=", ">>="),
                     hasEitherOperand(SignedIntegerOperand),
                     hasLHS(hasType(isInteger())), hasRHS(hasType(isInteger())))
          .bind("binary-sign-interference"),
      this);

  // Match unary operations on signed integer types.
  Finder->addMatcher(
      unaryOperator(hasOperatorName("~"), hasUnaryOperand(SignedIntegerOperand))
          .bind("unary-signed"),
      this);
}

void SignedBitwiseCheck::check(const MatchFinder::MatchResult &Result) {
  const ast_matchers::BoundNodes &N = Result.Nodes;
  const auto *SignedOperand = N.getNodeAs<Expr>("signed-operand");
  assert(SignedOperand &&
         "No signed operand found in problematic bitwise operations");

  bool IsUnary = false;
  SourceLocation OperatorLoc;

  if (const auto *UnaryOp = N.getNodeAs<UnaryOperator>("unary-signed")) {
    IsUnary = true;
    OperatorLoc = UnaryOp->getOperatorLoc();
  } else {
    if (const auto *BinaryOp =
            N.getNodeAs<BinaryOperator>("binary-no-sign-interference"))
      OperatorLoc = BinaryOp->getOperatorLoc();
    else if (const auto *BinaryOp =
                 N.getNodeAs<BinaryOperator>("binary-sign-interference"))
      OperatorLoc = BinaryOp->getOperatorLoc();
    else
      llvm_unreachable("unexpected matcher result");
  }
  diag(SignedOperand->getBeginLoc(), "use of a signed integer operand with a "
                                     "%select{binary|unary}0 bitwise operator")
      << IsUnary << SignedOperand->getSourceRange() << OperatorLoc;
}

} // namespace clang::tidy::hicpp
