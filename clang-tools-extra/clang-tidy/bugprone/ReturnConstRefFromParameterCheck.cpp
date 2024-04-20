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

std::optional<TraversalKind>
ReturnConstRefFromParameterCheck::getCheckTraversalKind() const {
  // Use 'AsIs' to make sure the return type is exactly the same as the
  // parameter type.
  return TK_AsIs;
}

void ReturnConstRefFromParameterCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      returnStmt(hasReturnValue(declRefExpr(to(parmVarDecl(hasType(
                     hasCanonicalType(matchers::isReferenceToConst())))))))
          .bind("ret"),
      this);
}

void ReturnConstRefFromParameterCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *R = Result.Nodes.getNodeAs<ReturnStmt>("ret");
  diag(R->getRetValue()->getBeginLoc(),
       "return const reference parameter cause potential use-after-free "
       "when function accepts immediately constructed value.");
}

} // namespace clang::tidy::bugprone
