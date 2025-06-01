//===--- SwitchMissingDefaultCaseCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SwitchMissingDefaultCaseCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(SwitchStmt, hasDefaultCase) {
  const SwitchCase *Case = Node.getSwitchCaseList();
  while (Case) {
    if (DefaultStmt::classof(Case))
      return true;

    Case = Case->getNextSwitchCase();
  }
  return false;
}
} // namespace

void SwitchMissingDefaultCaseCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      switchStmt(hasCondition(expr(unless(isInstantiationDependent()),
                                   hasType(qualType(hasCanonicalType(
                                       unless(hasDeclaration(enumDecl()))))))),
                 unless(hasDefaultCase()))
          .bind("switch"),
      this);
}

void SwitchMissingDefaultCaseCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *Switch = Result.Nodes.getNodeAs<SwitchStmt>("switch");

  diag(Switch->getSwitchLoc(), "switching on non-enum value without "
                               "default case may not cover all cases");
}
} // namespace clang::tidy::bugprone
