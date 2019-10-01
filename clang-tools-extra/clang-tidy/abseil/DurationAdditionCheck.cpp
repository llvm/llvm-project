//===--- DurationAdditionCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DurationAdditionCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

void DurationAdditionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      binaryOperator(hasOperatorName("+"),
                     hasEitherOperand(expr(ignoringParenImpCasts(
                         callExpr(callee(functionDecl(TimeConversionFunction())
                                             .bind("function_decl")))
                             .bind("call")))))
          .bind("binop"),
      this);
}

void DurationAdditionCheck::check(const MatchFinder::MatchResult &Result) {
  const BinaryOperator *Binop =
      Result.Nodes.getNodeAs<clang::BinaryOperator>("binop");
  const CallExpr *Call = Result.Nodes.getNodeAs<clang::CallExpr>("call");

  // Don't try to replace things inside of macro definitions.
  if (Binop->getExprLoc().isMacroID() || Binop->getExprLoc().isInvalid())
    return;

  llvm::Optional<DurationScale> Scale = getScaleForTimeInverse(
      Result.Nodes.getNodeAs<clang::FunctionDecl>("function_decl")->getName());
  if (!Scale)
    return;

  llvm::StringRef TimeFactory = getTimeInverseForScale(*Scale);

  FixItHint Hint;
  if (Call == Binop->getLHS()->IgnoreParenImpCasts()) {
    Hint = FixItHint::CreateReplacement(
        Binop->getSourceRange(),
        (llvm::Twine(TimeFactory) + "(" +
         tooling::fixit::getText(*Call->getArg(0), *Result.Context) + " + " +
         rewriteExprFromNumberToDuration(Result, *Scale, Binop->getRHS()) + ")")
            .str());
  } else {
    assert(Call == Binop->getRHS()->IgnoreParenImpCasts() &&
           "Call should be found on the RHS");
    Hint = FixItHint::CreateReplacement(
        Binop->getSourceRange(),
        (llvm::Twine(TimeFactory) + "(" +
         rewriteExprFromNumberToDuration(Result, *Scale, Binop->getLHS()) +
         " + " + tooling::fixit::getText(*Call->getArg(0), *Result.Context) +
         ")")
            .str());
  }

  diag(Binop->getBeginLoc(), "perform addition in the duration domain") << Hint;
}

} // namespace abseil
} // namespace tidy
} // namespace clang
