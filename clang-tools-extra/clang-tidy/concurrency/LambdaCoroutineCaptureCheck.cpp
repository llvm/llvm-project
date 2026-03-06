//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LambdaCoroutineCaptureCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::concurrency {

namespace {

AST_MATCHER(LambdaExpr, hasCoroutineBody) {
  const Stmt *Body = Node.getBody();
  return Body != nullptr && isa<CoroutineBodyStmt>(Body);
}

AST_MATCHER(LambdaExpr, capturesWithoutDeducingThis) {
  if (Node.capture_size() == 0U)
    return false;
  const auto *Call = Node.getCallOperator();
  return !Call->isExplicitObjectMemberFunction();
}

} // namespace

void LambdaCoroutineCaptureCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      lambdaExpr(hasCoroutineBody(), capturesWithoutDeducingThis())
          .bind("lambda"),
      this);
}

void LambdaCoroutineCaptureCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedLambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  const auto *Call = MatchedLambda->getCallOperator();
  const bool HasExplicitParams = MatchedLambda->hasExplicitParameters();

  auto DiagBuilder =
      diag(MatchedLambda->getExprLoc(),
           "lambda coroutine with captures may cause use-after-free; use "
           "'this auto' as the first parameter to move captures into the "
           "coroutine frame");

  if (HasExplicitParams) {
    const bool HasParams = !Call->param_empty();
    if (HasParams) {
      const ParmVarDecl *FirstParam = Call->parameters().front();
      DiagBuilder << FixItHint::CreateInsertion(FirstParam->getBeginLoc(),
                                                "this auto, ");
    } else {
      // Empty parens `()` — insert `this auto` before the closing paren.
      DiagBuilder << FixItHint::CreateInsertion(
          Call->getFunctionTypeLoc().getRParenLoc(), "this auto");
    }
  } else {
    // No explicit parameter list — insert `(this auto)` after the
    // capture list closing `]`.
    auto IntroRange = MatchedLambda->getIntroducerRange();
    DiagBuilder << FixItHint::CreateInsertion(
        IntroRange.getEnd().getLocWithOffset(1), "(this auto)");
  }
}

} // namespace clang::tidy::concurrency
