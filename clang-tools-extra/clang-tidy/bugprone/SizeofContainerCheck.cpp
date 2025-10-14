//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SizeofContainerCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void SizeofContainerCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      expr(unless(isInTemplateInstantiation()),
           expr(sizeOfExpr(has(ignoringParenImpCasts(
                    expr(hasType(hasCanonicalType(hasDeclaration(cxxRecordDecl(
                        matchesName("^(::std::|::string)"),
                        unless(matchesName("^::std::(bitset|array)$")),
                        hasMethod(cxxMethodDecl(hasName("size"), isPublic(),
                                                isConst())))))))))))
               .bind("sizeof"),
           // Ignore ARRAYSIZE(<array of containers>) pattern.
           unless(hasAncestor(binaryOperator(
               hasAnyOperatorName("/", "%"),
               hasLHS(ignoringParenCasts(sizeOfExpr(expr()))),
               hasRHS(ignoringParenCasts(equalsBoundNode("sizeof"))))))),
      this);
}

void SizeofContainerCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *SizeOf =
      Result.Nodes.getNodeAs<UnaryExprOrTypeTraitExpr>("sizeof");

  auto Diag =
      diag(SizeOf->getBeginLoc(), "sizeof() doesn't return the size of the "
                                  "container; did you mean .size()?");
}

} // namespace clang::tidy::bugprone
