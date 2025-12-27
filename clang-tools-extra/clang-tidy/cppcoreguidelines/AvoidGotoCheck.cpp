//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidGotoCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {
AST_MATCHER(GotoStmt, isForwardJumping) {
  return Node.getBeginLoc() < Node.getLabel()->getBeginLoc();
}

AST_MATCHER(GotoStmt, isInMacro) {
  return Node.getBeginLoc().isMacroID() && Node.getEndLoc().isMacroID();
}
} // namespace

AvoidGotoCheck::AvoidGotoCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.get("IgnoreMacros", false)) {}

void AvoidGotoCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void AvoidGotoCheck::registerMatchers(MatchFinder *Finder) {
  // TODO: This check does not recognize `IndirectGotoStmt` which is a
  // GNU extension. These must be matched separately and an AST matcher
  // is currently missing for them.

  // Check if the 'goto' is used for control flow other than jumping
  // out of a nested loop.
  auto Loop = mapAnyOf(forStmt, cxxForRangeStmt, whileStmt, doStmt);
  auto NestedLoop = Loop.with(hasAncestor(Loop));

  const ast_matchers::internal::Matcher<GotoStmt> Anything = anything();

  Finder->addMatcher(gotoStmt(IgnoreMacros ? unless(isInMacro()) : Anything,
                              anyOf(unless(hasAncestor(NestedLoop)),
                                    unless(isForwardJumping())))
                         .bind("goto"),
                     this);
}

void AvoidGotoCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Goto = Result.Nodes.getNodeAs<GotoStmt>("goto");

  diag(Goto->getGotoLoc(), "avoid using 'goto' for flow control")
      << Goto->getSourceRange();
  diag(Goto->getLabel()->getBeginLoc(), "label defined here",
       DiagnosticIDs::Note);
}
} // namespace clang::tidy::cppcoreguidelines
