//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeUnionAccessCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

void ProTypeUnionAccessCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      memberExpr(hasObjectExpression(hasType(recordDecl(isUnion()))))
          .bind("expr"),
      this);
}

void ProTypeUnionAccessCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Matched = Result.Nodes.getNodeAs<MemberExpr>("expr");
  SourceLocation Loc = Matched->getMemberLoc();
  if (Loc.isInvalid())
    Loc = Matched->getBeginLoc();
  diag(Loc, "do not access members of unions; consider using (boost::)variant "
            "instead");
}

} // namespace clang::tidy::cppcoreguidelines
