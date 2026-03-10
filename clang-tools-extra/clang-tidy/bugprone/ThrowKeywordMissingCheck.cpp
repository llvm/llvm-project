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
using namespace clang::ast_matchers::internal;

namespace clang::tidy::bugprone {

void ThrowKeywordMissingCheck::registerMatchers(MatchFinder *Finder) {
  const VariadicDynCastAllOfMatcher<Stmt, AttributedStmt> AttributedStmt;
  // Matches an 'expression-statement', as defined in [stmt.expr]/1.
  // Not to be confused with the similarly-named GNU extension, the
  // statement expression.
  const auto ExprStmt = [&](const Matcher<Expr> &InnerMatcher) {
    return expr(hasParent(stmt(anyOf(doStmt(), whileStmt(), forStmt(),
                                     compoundStmt(), ifStmt(), switchStmt(),
                                     labelStmt(), AttributedStmt()))),
                InnerMatcher);
  };

  Finder->addMatcher(
      ExprStmt(
          cxxConstructExpr(hasType(cxxRecordDecl(anyOf(
              matchesName("[Ee]xception|EXCEPTION"),
              hasAnyBase(hasType(hasCanonicalType(recordType(hasDeclaration(
                  cxxRecordDecl(matchesName("[Ee]xception|EXCEPTION"))
                      .bind("base")))))))))))
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
