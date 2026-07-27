//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NestedSwitchLabelCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

static const char SwitchBodyId[] = "switch-body";
static const char SwitchLabelId[] = "switch-label";

AST_MATCHER_P(SwitchStmt, hasBodyIgnoringAttrs,
              clang::ast_matchers::internal::Matcher<Stmt>, InnerMatcher) {
  const Stmt *Body = Node.getBody();
  while (const auto *Attributed = dyn_cast<AttributedStmt>(Body))
    Body = Attributed->getSubStmt();

  return InnerMatcher.matches(*Body, Finder, Builder);
}

} // namespace

void NestedSwitchLabelCheck::registerMatchers(MatchFinder *Finder) {
  const auto BoundSwitchBody = stmt().bind(SwitchBodyId);
  const auto CompoundNestedInSwitchBody =
      compoundStmt(hasAncestor(stmt(equalsBoundNode(SwitchBodyId))));
  const auto LabelInNestedCompound =
      switchCase(hasAncestor(CompoundNestedInSwitchBody)).bind(SwitchLabelId);

  Finder->addMatcher(switchStmt(hasBodyIgnoringAttrs(BoundSwitchBody),
                                forEachSwitchCase(LabelInNestedCompound)),
                     this);
}

void NestedSwitchLabelCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Label = Result.Nodes.getNodeAs<SwitchCase>(SwitchLabelId);
  diag(Label->getKeywordLoc(),
       "switch label is nested inside a compound statement other than the "
       "switch body");
}

} // namespace clang::tidy::bugprone
