//===--- ProTypeReinterpretCastCheck.cpp - clang-tidy----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeReinterpretCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void ProTypeReinterpretCastCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(cxxReinterpretCastExpr().bind("cast"), this);
}

void ProTypeReinterpretCastCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast =
      Result.Nodes.getNodeAs<CXXReinterpretCastExpr>("cast");
  diag(MatchedCast->getOperatorLoc(), "do not use reinterpret_cast");
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
