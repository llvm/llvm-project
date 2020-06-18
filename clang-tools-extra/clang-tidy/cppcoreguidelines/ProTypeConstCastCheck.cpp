//===--- ProTypeConstCastCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeConstCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void ProTypeConstCastCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxConstCastExpr().bind("cast"), this);
}

void ProTypeConstCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<CXXConstCastExpr>("cast");
  diag(MatchedCast->getOperatorLoc(), "do not use const_cast");
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
