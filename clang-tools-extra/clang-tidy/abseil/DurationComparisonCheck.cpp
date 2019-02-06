//===--- DurationComparisonCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DurationComparisonCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

void DurationComparisonCheck::registerMatchers(MatchFinder *Finder) {
  auto Matcher =
      binaryOperator(anyOf(hasOperatorName(">"), hasOperatorName(">="),
                           hasOperatorName("=="), hasOperatorName("<="),
                           hasOperatorName("<")),
                     hasEitherOperand(ignoringImpCasts(callExpr(
                         callee(functionDecl(DurationConversionFunction())
                                    .bind("function_decl"))))))
          .bind("binop");

  Finder->addMatcher(Matcher, this);
}

void DurationComparisonCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Binop = Result.Nodes.getNodeAs<BinaryOperator>("binop");

  llvm::Optional<DurationScale> Scale = getScaleForDurationInverse(
      Result.Nodes.getNodeAs<FunctionDecl>("function_decl")->getName());
  if (!Scale)
    return;

  // In most cases, we'll only need to rewrite one of the sides, but we also
  // want to handle the case of rewriting both sides. This is much simpler if
  // we unconditionally try and rewrite both, and let the rewriter determine
  // if nothing needs to be done.
  if (!isNotInMacro(Result, Binop->getLHS()) ||
      !isNotInMacro(Result, Binop->getRHS()))
    return;
  std::string LhsReplacement =
      rewriteExprFromNumberToDuration(Result, *Scale, Binop->getLHS());
  std::string RhsReplacement =
      rewriteExprFromNumberToDuration(Result, *Scale, Binop->getRHS());

  diag(Binop->getBeginLoc(), "perform comparison in the duration domain")
      << FixItHint::CreateReplacement(Binop->getSourceRange(),
                                      (llvm::Twine(LhsReplacement) + " " +
                                       Binop->getOpcodeStr() + " " +
                                       RhsReplacement)
                                          .str());
}

} // namespace abseil
} // namespace tidy
} // namespace clang
