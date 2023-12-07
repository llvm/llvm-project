//===--- DefaultArgumentsCallsCheck.cpp - clang-tidy-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefaultArgumentsCallsCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy::fuchsia {

void DefaultArgumentsCallsCheck::registerMatchers(MatchFinder *Finder) {
  // Calling a function which uses default arguments is disallowed.
  Finder->addMatcher(cxxDefaultArgExpr().bind("stmt"), this);
}

void DefaultArgumentsCallsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *S = Result.Nodes.getNodeAs<CXXDefaultArgExpr>("stmt");
  if (!S)
    return;

  diag(S->getUsedLocation(),
       "calling a function that uses a default argument is disallowed");
  diag(S->getParam()->getBeginLoc(), "default parameter was declared here",
       DiagnosticIDs::Note);
}

} // namespace clang::tidy::fuchsia
