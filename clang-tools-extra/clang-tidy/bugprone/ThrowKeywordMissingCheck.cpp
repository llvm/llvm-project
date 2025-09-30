//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThrowKeywordMissingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void ThrowKeywordMissingCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxConstructExpr(
          hasType(cxxRecordDecl(anyOf(
              matchesName("[Ee]xception|EXCEPTION"),
              hasAnyBase(hasType(hasCanonicalType(recordType(hasDeclaration(
                  cxxRecordDecl(matchesName("[Ee]xception|EXCEPTION"))
                      .bind("base"))))))))),
          unless(anyOf(
              hasAncestor(
                  stmt(anyOf(cxxThrowExpr(), callExpr(), returnStmt()))),
              hasAncestor(decl(anyOf(varDecl(), fieldDecl()))),
              hasAncestor(expr(cxxNewExpr(hasAnyPlacementArg(anything())))),
              allOf(hasAncestor(cxxConstructorDecl()),
                    unless(hasAncestor(cxxCatchStmt()))))))
          .bind("temporary-exception-not-thrown"),
      this);
}

void ThrowKeywordMissingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *TemporaryExpr =
      Result.Nodes.getNodeAs<Expr>("temporary-exception-not-thrown");

  diag(TemporaryExpr->getBeginLoc(), "suspicious exception object created but "
                                     "not thrown; did you mean 'throw %0'?")
      << TemporaryExpr->getType().getBaseTypeIdentifier()->getName();

  if (const auto *BaseDecl = Result.Nodes.getNodeAs<Decl>("base"))
    diag(BaseDecl->getLocation(),
         "object type inherits from base class declared here",
         DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
