//===--- AvoidReturnWithVoidValueCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidReturnWithVoidValueCheck.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static constexpr auto IgnoreMacrosName = "IgnoreMacros";
static constexpr auto IgnoreMacrosDefault = true;

static constexpr auto StrictModeName = "StrictMode";
static constexpr auto StrictModeDefault = true;

AvoidReturnWithVoidValueCheck::AvoidReturnWithVoidValueCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(
          Options.getLocalOrGlobal(IgnoreMacrosName, IgnoreMacrosDefault)),
      StrictMode(Options.getLocalOrGlobal(StrictModeName, StrictModeDefault)) {}

void AvoidReturnWithVoidValueCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      returnStmt(
          hasReturnValue(allOf(hasType(voidType()), unless(initListExpr()))),
          optionally(hasParent(compoundStmt().bind("compound_parent"))))
          .bind("void_return"),
      this);
}

void AvoidReturnWithVoidValueCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *VoidReturn = Result.Nodes.getNodeAs<ReturnStmt>("void_return");
  if (IgnoreMacros && VoidReturn->getBeginLoc().isMacroID())
    return;
  if (!StrictMode && !Result.Nodes.getNodeAs<CompoundStmt>("compound_parent"))
    return;
  diag(VoidReturn->getBeginLoc(), "return statement within a void function "
                                  "should not have a specified return value");
}

void AvoidReturnWithVoidValueCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, IgnoreMacrosName, IgnoreMacros);
  Options.store(Opts, StrictModeName, StrictMode);
}

} // namespace clang::tidy::readability
