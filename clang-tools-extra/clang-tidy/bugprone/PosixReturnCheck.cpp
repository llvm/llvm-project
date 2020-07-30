//===--- PosixReturnCheck.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PosixReturnCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

static StringRef getFunctionSpelling(const MatchFinder::MatchResult &Result,
                                     const char *BindingStr) {
  const CallExpr *MatchedCall = cast<CallExpr>(
      (Result.Nodes.getNodeAs<BinaryOperator>(BindingStr))->getLHS());
  const SourceManager &SM = *Result.SourceManager;
  return Lexer::getSourceText(CharSourceRange::getTokenRange(
                                  MatchedCall->getCallee()->getSourceRange()),
                              SM, Result.Context->getLangOpts());
}

void PosixReturnCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      binaryOperator(
          hasOperatorName("<"),
          hasLHS(callExpr(callee(functionDecl(
              anyOf(matchesName("^::posix_"), matchesName("^::pthread_")),
              unless(hasName("::posix_openpt")))))),
          hasRHS(integerLiteral(equals(0))))
          .bind("ltzop"),
      this);
  Finder->addMatcher(
      binaryOperator(
          hasOperatorName(">="),
          hasLHS(callExpr(callee(functionDecl(
              anyOf(matchesName("^::posix_"), matchesName("^::pthread_")),
              unless(hasName("::posix_openpt")))))),
          hasRHS(integerLiteral(equals(0))))
          .bind("atop"),
      this);
  Finder->addMatcher(
      binaryOperator(
          hasAnyOperatorName("==", "!=", "<=", "<"),
          hasLHS(callExpr(callee(functionDecl(
              anyOf(matchesName("^::posix_"), matchesName("^::pthread_")),
              unless(hasName("::posix_openpt")))))),
          hasRHS(unaryOperator(hasOperatorName("-"),
                               hasUnaryOperand(integerLiteral()))))
          .bind("binop"),
      this);
}

void PosixReturnCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *LessThanZeroOp =
          Result.Nodes.getNodeAs<BinaryOperator>("ltzop")) {
    SourceLocation OperatorLoc = LessThanZeroOp->getOperatorLoc();
    diag(OperatorLoc, "the comparison always evaluates to false because %0 "
                      "always returns non-negative values")
        << getFunctionSpelling(Result, "ltzop")
        << FixItHint::CreateReplacement(OperatorLoc, Twine(">").str());
    return;
  }
  if (const auto *AlwaysTrueOp =
          Result.Nodes.getNodeAs<BinaryOperator>("atop")) {
    diag(AlwaysTrueOp->getOperatorLoc(),
         "the comparison always evaluates to true because %0 always returns "
         "non-negative values")
        << getFunctionSpelling(Result, "atop");
    return;
  }
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binop");
  diag(BinOp->getOperatorLoc(), "%0 only returns non-negative values")
      << getFunctionSpelling(Result, "binop");
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
