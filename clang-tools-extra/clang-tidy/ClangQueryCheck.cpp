//===--- ClangQueryCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangQueryCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void ClangQueryCheck::registerMatchers(MatchFinder *Finder) {
  for (const auto &Matcher : Matchers) {
    bool Ok = Finder->addDynamicMatcher(Matcher, this);
    assert(Ok && "Expected to get top level matcher from query parser");
  }
}

void ClangQueryCheck::check(const MatchFinder::MatchResult &Result) {
  auto Map = Result.Nodes.getMap();
  for (const auto &[k, v] : Map) {
    diag(v.getSourceRange().getBegin(), k) << v.getSourceRange();
  }
}

} // namespace clang::tidy::misc
