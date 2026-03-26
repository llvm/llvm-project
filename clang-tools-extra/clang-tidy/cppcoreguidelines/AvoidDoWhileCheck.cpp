//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidDoWhileCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

AvoidDoWhileCheck::AvoidDoWhileCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.get("IgnoreMacros", false)) {}

void AvoidDoWhileCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void AvoidDoWhileCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(doStmt().bind("x"), this);
}

void AvoidDoWhileCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *MatchedDecl = Result.Nodes.getNodeAs<DoStmt>("x")) {
    if (IgnoreMacros && MatchedDecl->getBeginLoc().isMacroID())
      return;
    diag(MatchedDecl->getBeginLoc(), "avoid do-while loops");
  }
}

} // namespace clang::tidy::cppcoreguidelines
