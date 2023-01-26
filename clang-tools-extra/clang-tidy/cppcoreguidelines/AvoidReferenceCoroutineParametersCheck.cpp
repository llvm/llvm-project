//===--- AvoidReferenceCoroutineParametersCheck.cpp - clang-tidy ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidReferenceCoroutineParametersCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

void AvoidReferenceCoroutineParametersCheck::registerMatchers(
    MatchFinder *Finder) {
  auto IsCoroMatcher =
      hasDescendant(expr(anyOf(coyieldExpr(), coreturnStmt(), coawaitExpr())));
  Finder->addMatcher(parmVarDecl(hasType(type(referenceType())),
                                 hasAncestor(functionDecl(IsCoroMatcher)))
                         .bind("param"),
                     this);
}

void AvoidReferenceCoroutineParametersCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param")) {
    diag(Param->getBeginLoc(), "coroutine parameters should not be references");
  }
}

} // namespace clang::tidy::cppcoreguidelines
