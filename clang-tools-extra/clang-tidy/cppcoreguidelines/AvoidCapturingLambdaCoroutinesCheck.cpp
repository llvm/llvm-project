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
  return Body != nullptr && isa<CoroutineBodyStmt>(Body);
}

AST_MATCHER(LambdaExpr, hasCaptures) { return Node.capture_size() != 0U; }

AST_MATCHER(LambdaExpr, hasDeducingThis) {
  return Node.getCallOperator()->isExplicitObjectMemberFunction();
}
} // namespace

AvoidCapturingLambdaCoroutinesCheck::AvoidCapturingLambdaCoroutinesCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowExplicitObjectParameters(
          Options.get("AllowExplicitObjectParameters", false)) {}

void AvoidCapturingLambdaCoroutinesCheck::registerMatchers(
    MatchFinder *Finder) {
  using LambdaExprMatcher = ast_matchers::internal::Matcher<LambdaExpr>;
  const auto ExplicitObjectFilter =
      AllowExplicitObjectParameters
          ? LambdaExprMatcher(unless(hasDeducingThis()))
          : LambdaExprMatcher(anything());
  Finder->addMatcher(
      lambdaExpr(hasCaptures(), hasCoroutineBody(), ExplicitObjectFilter)
          .bind("lambda"),
      this);
}

bool AvoidCapturingLambdaCoroutinesCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus20;
}

void AvoidCapturingLambdaCoroutinesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowExplicitObjectParameters",
                AllowExplicitObjectParameters);
}

void AvoidCapturingLambdaCoroutinesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedLambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  diag(MatchedLambda->getExprLoc(),
       "coroutine lambda may cause use-after-free, avoid captures or ensure "
       "lambda closure object has guaranteed lifetime");
}

} // namespace clang::tidy::cppcoreguidelines
