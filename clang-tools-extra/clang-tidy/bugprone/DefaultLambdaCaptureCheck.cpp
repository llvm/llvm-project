//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefaultLambdaCaptureCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void DefaultLambdaCaptureCheck::registerMatchers(MatchFinder *Finder) {
  // Match any lambda expression
  Finder->addMatcher(lambdaExpr().bind("lambda"), this);
}

void DefaultLambdaCaptureCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  if (!Lambda)
    return;

  // Check if lambda has a default capture
  if (Lambda->getCaptureDefault() == LCD_None)
    return;

  SourceLocation DefaultCaptureLoc = Lambda->getCaptureDefaultLoc();
  if (DefaultCaptureLoc.isInvalid())
    return;

  const char *CaptureKind =
      (Lambda->getCaptureDefault() == LCD_ByCopy) ? "by-copy" : "by-reference";

  diag(DefaultCaptureLoc, "lambda %0 default capture is discouraged; "
                          "prefer to capture specific variables explicitly")
      << CaptureKind;
}

} // namespace clang::tidy::bugprone
