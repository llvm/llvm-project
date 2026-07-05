//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TrailingReturnCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"

using namespace clang::ast_matchers;

namespace clang::tidy::fuchsia {

void TrailingReturnCheck::registerMatchers(MatchFinder *Finder) {
  // Functions that have trailing returns are disallowed, except for those
  // using decltype specifiers and lambda with otherwise unutterable
  // return types.
  Finder->addMatcher(
      functionDecl(hasTrailingReturn(),
                   unless(anyOf(returns(decltypeType()),
                                hasParent(cxxRecordDecl(isLambda())),
                                cxxDeductionGuideDecl())))
          .bind("decl"),
      this);
}

void TrailingReturnCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<FunctionDecl>("decl"))
    diag(D->getBeginLoc(),
         "a trailing return type is disallowed for this function declaration");
}

} // namespace clang::tidy::fuchsia
