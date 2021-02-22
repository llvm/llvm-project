//===--- ThreadCanceltypeAsynchronousCheck.cpp - clang-tidy ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadCanceltypeAsynchronousCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace concurrency {

void ThreadCanceltypeAsynchronousCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          allOf(callee(functionDecl(hasName("::pthread_setcanceltype"))),
                argumentCountIs(2)),
          hasArgument(0, isExpandedFromMacro("PTHREAD_CANCEL_ASYNCHRONOUS")))
          .bind("setcanceltype"),
      this);
}

void ThreadCanceltypeAsynchronousCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<Expr>("setcanceltype");
  diag(MatchedExpr->getBeginLoc(), "the cancel type for a pthread should not "
                                   "be 'PTHREAD_CANCEL_ASYNCHRONOUS'");
}

} // namespace concurrency
} // namespace tidy
} // namespace clang
