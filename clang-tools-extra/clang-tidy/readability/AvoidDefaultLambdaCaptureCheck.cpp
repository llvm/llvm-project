//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidDefaultLambdaCaptureCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {
AST_MATCHER(LambdaExpr, hasDefaultCapture) {
  return Node.getCaptureDefault() != LCD_None;
}

} // namespace

void AvoidDefaultLambdaCaptureCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(lambdaExpr(hasDefaultCapture()).bind("lambda"), this);
}

void AvoidDefaultLambdaCaptureCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  assert(Lambda);

  const SourceLocation DefaultCaptureLoc = Lambda->getCaptureDefaultLoc();
  if (DefaultCaptureLoc.isInvalid())
    return;

  diag(DefaultCaptureLoc, "lambda default captures are discouraged; "
                          "prefer to capture specific variables explicitly");
}

} // namespace clang::tidy::readability
