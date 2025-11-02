//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OverloadedOperatorCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy::fuchsia {

namespace {
AST_MATCHER(FunctionDecl, isFuchsiaOverloadedOperator) {
  if (const auto *CXXMethodNode = dyn_cast<CXXMethodDecl>(&Node)) {
    if (CXXMethodNode->isCopyAssignmentOperator() ||
        CXXMethodNode->isMoveAssignmentOperator())
      return false;
    if (CXXMethodNode->getParent()->isLambda())
      return false;
  }
  return Node.isOverloadedOperator();
}
} // namespace

void OverloadedOperatorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(isFuchsiaOverloadedOperator()).bind("decl"),
                     this);
}

void OverloadedOperatorCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *D = Result.Nodes.getNodeAs<FunctionDecl>("decl");
  assert(D && "No FunctionDecl captured!");

  SourceLocation Loc = D->getBeginLoc();
  if (Loc.isValid())
    diag(Loc, "overloading %0 is disallowed") << D;
}

} // namespace clang::tidy::fuchsia
