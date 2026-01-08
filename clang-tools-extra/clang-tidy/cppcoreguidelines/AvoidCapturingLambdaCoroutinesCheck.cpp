//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidCapturingLambdaCoroutinesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {
AST_MATCHER(LambdaExpr, hasCoroutineBody) {
  const Stmt *Body = Node.getBody();
  return Body != nullptr && CoroutineBodyStmt::classof(Body);
}

AST_MATCHER(LambdaExpr, hasCaptures) { return Node.capture_size() != 0U; }
} // namespace

void AvoidCapturingLambdaCoroutinesCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      lambdaExpr(hasCaptures(), hasCoroutineBody()).bind("lambda"), this);
}

bool AvoidCapturingLambdaCoroutinesCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus20;
}

void AvoidCapturingLambdaCoroutinesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedLambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  diag(MatchedLambda->getExprLoc(),
       "coroutine lambda may cause use-after-free, avoid captures or ensure "
       "lambda closure object has guaranteed lifetime");
}

} // namespace clang::tidy::cppcoreguidelines
