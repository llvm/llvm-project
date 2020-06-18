//===--- UnaryStaticAssertCheck.cpp - clang-tidy---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnaryStaticAssertCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

void UnaryStaticAssertCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(staticAssertDecl().bind("static_assert"), this);
}

void UnaryStaticAssertCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl =
      Result.Nodes.getNodeAs<StaticAssertDecl>("static_assert");
  const StringLiteral *AssertMessage = MatchedDecl->getMessage();

  SourceLocation Loc = MatchedDecl->getLocation();

  if (!AssertMessage || AssertMessage->getLength() ||
      AssertMessage->getBeginLoc().isMacroID() || Loc.isMacroID())
    return;

  diag(Loc,
       "use unary 'static_assert' when the string literal is an empty string")
      << FixItHint::CreateRemoval(AssertMessage->getSourceRange());
}

} // namespace modernize
} // namespace tidy
} // namespace clang
