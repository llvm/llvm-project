//===--- ProTypeUnionAccessCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
