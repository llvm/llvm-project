//===--- FloatLoopCounter.cpp - clang-tidy---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FloatLoopCounter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void FloatLoopCounter::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      forStmt(hasIncrement(expr(hasType(realFloatingPointType())))).bind("for"),
      this);
}

void FloatLoopCounter::check(const MatchFinder::MatchResult &Result) {
  const auto *FS = Result.Nodes.getNodeAs<ForStmt>("for");

  diag(FS->getInc()->getExprLoc(), "loop induction expression should not have "
                                   "floating-point type");
}

} // namespace cert
} // namespace tidy
} // namespace clang
