//===--- suspiciousCopyInRangeLoopCheck.cpp - clang-tidy
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousCopyInRangeLoopCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

// TODO: Possibly rename to "SuspiciousAutoCopyInRangeLoop"
void SuspiciousCopyInRangeLoopCheck::registerMatchers(MatchFinder *Finder) {
  // TODO: make sure this catches `const auto` as well.
  auto auto_copy_in_range_based_for_loops =
      cxxForRangeStmt(hasLoopVariable(varDecl(hasType(autoType()))));
  Finder->addMatcher(auto_copy_in_range_based_for_loops.bind("for_range"),
                     this);
}

void SuspiciousCopyInRangeLoopCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<CXXForRangeStmt>("for_range");
  std::string VarName = MatchedDecl->getLoopVariable()->getNameAsString();
  if (MatchedDecl) {
    diag(MatchedDecl->getBeginLoc(),
         "Found potentially-spurious copy in range loop:\n* It is unlikely you "
         "intended to copy '%0' for each iteration of the loop\n* To avoid "
         "copying, use `auto&`\n* If this copy was intentional, do not use "
         "`auto` and instead spell out the type explicitly")
        << VarName
        << FixItHint::CreateInsertion(
               MatchedDecl->getLoopVariable()->getLocation(),
               "suspicious copy here");
  }
}

} // namespace clang::tidy::bugprone
