//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PredictableRandCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void PredictableRandCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(callExpr(callee(functionDecl(namedDecl(hasName("::rand")),
                                                  parameterCountIs(0))))
                         .bind("randomGenerator"),
                     this);
}

void PredictableRandCheck::check(const MatchFinder::MatchResult &Result) {
  std::string Msg;
  if (getLangOpts().CPlusPlus)
    Msg = "; use C++11 random library instead";

  const auto *MatchedDecl = Result.Nodes.getNodeAs<CallExpr>("randomGenerator");
  diag(MatchedDecl->getBeginLoc(), "rand() has limited randomness" + Msg);
}

} // namespace clang::tidy::misc
