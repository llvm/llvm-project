//===--- ProTypeUnionAccessCheck.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeUnionAccessCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void ProTypeUnionAccessCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(
      memberExpr(hasObjectExpression(hasType(recordDecl(isUnion()))))
          .bind("expr"),
      this);
}

void ProTypeUnionAccessCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Matched = Result.Nodes.getNodeAs<MemberExpr>("expr");
  diag(Matched->getMemberLoc(),
       "do not access members of unions; use (boost::)variant instead");
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
