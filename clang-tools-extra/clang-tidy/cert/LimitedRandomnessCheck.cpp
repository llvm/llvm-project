//===--- LimitedRandomnessCheck.cpp - clang-tidy---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LimitedRandomnessCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void LimitedRandomnessCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(callExpr(callee(functionDecl(namedDecl(hasName("::rand")),
                                                  parameterCountIs(0))))
                         .bind("randomGenerator"),
                     this);
}

void LimitedRandomnessCheck::check(const MatchFinder::MatchResult &Result) {
  std::string Msg;
  if (getLangOpts().CPlusPlus)
    Msg = "; use C++11 random library instead";

  const auto *MatchedDecl = Result.Nodes.getNodeAs<CallExpr>("randomGenerator");
  diag(MatchedDecl->getBeginLoc(), "rand() has limited randomness" + Msg);
}

} // namespace cert
} // namespace tidy
} // namespace clang
