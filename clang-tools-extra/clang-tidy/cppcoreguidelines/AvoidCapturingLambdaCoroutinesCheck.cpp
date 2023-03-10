//===--- AvoidCapturingLambdaCoroutinesCheck.cpp - clang-tidy -------------===//
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

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void AvoidCapturingLambdaCoroutinesCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(lambdaExpr().bind("lambda"), this);
}

void AvoidCapturingLambdaCoroutinesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  if (!Lambda) {
    return;
  }

  const auto *Body = dyn_cast<CoroutineBodyStmt>(Lambda->getBody());
  if (!Body) {
    return;
  }

  if (Lambda->captures().empty()) {
    return;
  }

  diag(Lambda->getBeginLoc(), "found capturing coroutine lambda");
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
