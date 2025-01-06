//===--- PosixReturnCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PosixReturnCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static StringRef getFunctionSpelling(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>("call");
  const SourceManager &SM = *Result.SourceManager;
  return Lexer::getSourceText(CharSourceRange::getTokenRange(
                                  MatchedCall->getCallee()->getSourceRange()),
                              SM, Result.Context->getLangOpts());
}

void PosixReturnCheck::registerMatchers(MatchFinder *Finder) {
  const auto PosixCall =
      callExpr(callee(functionDecl(
                   anyOf(matchesName("^::posix_"), matchesName("^::pthread_")),
                   unless(hasName("::posix_openpt")))))
          .bind("call");
  const auto ZeroIntegerLiteral = integerLiteral(equals(0));
  const auto NegIntegerLiteral =
      unaryOperator(hasOperatorName("-"), hasUnaryOperand(integerLiteral()));

  Finder->addMatcher(
      binaryOperator(
          anyOf(allOf(hasOperatorName("<"), hasLHS(PosixCall),
                      hasRHS(ZeroIntegerLiteral)),
                allOf(hasOperatorName(">"), hasLHS(ZeroIntegerLiteral),
                      hasRHS(PosixCall))))
          .bind("ltzop"),
      this);
  Finder->addMatcher(
      binaryOperator(
          anyOf(allOf(hasOperatorName(">="), hasLHS(PosixCall),
                      hasRHS(ZeroIntegerLiteral)),
                allOf(hasOperatorName("<="), hasLHS(ZeroIntegerLiteral),
                      hasRHS(PosixCall))))
          .bind("atop"),
      this);
  Finder->addMatcher(binaryOperator(hasAnyOperatorName("==", "!="),
                                    hasOperands(PosixCall, NegIntegerLiteral))
                         .bind("binop"),
                     this);
  Finder->addMatcher(
      binaryOperator(anyOf(allOf(hasAnyOperatorName("<=", "<"),
                                 hasLHS(PosixCall), hasRHS(NegIntegerLiteral)),
                           allOf(hasAnyOperatorName(">", ">="),
                                 hasLHS(NegIntegerLiteral), hasRHS(PosixCall))))
          .bind("binop"),
      this);
}

void PosixReturnCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *LessThanZeroOp =
          Result.Nodes.getNodeAs<BinaryOperator>("ltzop")) {
    SourceLocation OperatorLoc = LessThanZeroOp->getOperatorLoc();
    StringRef NewBinOp =
        LessThanZeroOp->getOpcode() == BinaryOperator::Opcode::BO_LT ? ">"
                                                                     : "<";
    diag(OperatorLoc, "the comparison always evaluates to false because %0 "
                      "always returns non-negative values")
        << getFunctionSpelling(Result)
        << FixItHint::CreateReplacement(OperatorLoc, NewBinOp);
    return;
  }
  if (const auto *AlwaysTrueOp =
          Result.Nodes.getNodeAs<BinaryOperator>("atop")) {
    diag(AlwaysTrueOp->getOperatorLoc(),
         "the comparison always evaluates to true because %0 always returns "
         "non-negative values")
        << getFunctionSpelling(Result);
    return;
  }
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binop");
  diag(BinOp->getOperatorLoc(), "%0 only returns non-negative values")
      << getFunctionSpelling(Result);
}

} // namespace clang::tidy::bugprone
