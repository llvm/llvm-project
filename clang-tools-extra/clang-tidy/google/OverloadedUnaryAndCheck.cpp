//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OverloadedUnaryAndCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::google::runtime {

void OverloadedUnaryAndCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  // Match unary methods that overload operator&.
  Finder->addMatcher(
      cxxMethodDecl(parameterCountIs(0), hasOverloadedOperatorName("&"))
          .bind("overload"),
      this);
  // Also match freestanding unary operator& overloads. Be careful not to match
  // binary methods.
  Finder->addMatcher(functionDecl(unless(cxxMethodDecl()), parameterCountIs(1),
                                  hasOverloadedOperatorName("&"))
                         .bind("overload"),
                     this);
}

void OverloadedUnaryAndCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Decl = Result.Nodes.getNodeAs<FunctionDecl>("overload");
  diag(Decl->getBeginLoc(),
       "do not overload unary operator&, it is dangerous.");
}

} // namespace clang::tidy::google::runtime
