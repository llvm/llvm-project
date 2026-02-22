//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReferenceToConstructedTemporaryCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER_P(MaterializeTemporaryExpr, isExtendedBy,
              ast_matchers::internal::Matcher<ValueDecl>, InnerMatcher) {
  const ValueDecl *ExtendingDecl = Node.getExtendingDecl();
  return ExtendingDecl && InnerMatcher.matches(*ExtendingDecl, Finder, Builder);
}

} // namespace

void ReferenceToConstructedTemporaryCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      materializeTemporaryExpr(
          hasType(qualType().bind("type")),
          isExtendedBy(varDecl(hasType(qualType(references(
                                   qualType(equalsBoundNode("type"))))))
                           .bind("var")))
          .bind("temporary"),
      this);
}

void ReferenceToConstructedTemporaryCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("var");
  const auto *MatchedTemporary = Result.Nodes.getNodeAs<Expr>("temporary");

  diag(MatchedDecl->getLocation(),
       "reference variable %0 extends the lifetime of a just-constructed "
       "temporary object %1, consider changing reference to value")
      << MatchedDecl << MatchedTemporary->getType();
}

} // namespace clang::tidy::readability
