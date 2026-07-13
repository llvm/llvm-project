//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TrivialSwitchCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void TrivialSwitchCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(switchStmt().bind("switch"), this);
}

void TrivialSwitchCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Switch = Result.Nodes.getNodeAs<SwitchStmt>("switch");
  std::size_t CaseCount = 0;
  bool HasDefault = false;

  for (const SwitchCase *CurrentCase = Switch->getSwitchCaseList(); CurrentCase;
       CurrentCase = CurrentCase->getNextSwitchCase()) {
    ++CaseCount;
    if (isa<DefaultStmt>(CurrentCase))
      HasDefault = true;
  }

  // FIXME: Try to add fix-it for each case.
  switch (const SourceLocation Loc = Switch->getBeginLoc(); CaseCount) {
  case 0:
    diag(Loc, "switch statement without labels has no effect");
    return;
  case 1:
    if (HasDefault)
      diag(Loc, "switch with default label only");
    else
      diag(Loc, "switch with only one case; use an if statement");
    return;
  case 2:
    if (HasDefault)
      diag(Loc, "switch could be better written as an if-else statement");
    [[fallthrough]];
  default:
    break;
  }
}

} // namespace clang::tidy::readability
