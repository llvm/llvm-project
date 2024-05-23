//===--- ReturnConstRefFromParameterCheck.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReturnConstRefFromParameterCheck.h"
#include "../utils/Matchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void ReturnConstRefFromParameterCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      returnStmt(
          hasReturnValue(declRefExpr(to(parmVarDecl(hasType(hasCanonicalType(
              qualType(matchers::isReferenceToConst()).bind("type"))))))),
          hasAncestor(functionDecl(hasReturnTypeLoc(
              loc(qualType(hasCanonicalType(equalsBoundNode("type"))))))))
          .bind("ret"),
      this);
}

void ReturnConstRefFromParameterCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *R = Result.Nodes.getNodeAs<ReturnStmt>("ret");
  diag(R->getRetValue()->getBeginLoc(),
       "returning a constant reference parameter may cause a use-after-free "
       "when the parameter is constructed from a temporary")
      << R->getRetValue()->getSourceRange();
}

} // namespace clang::tidy::bugprone
