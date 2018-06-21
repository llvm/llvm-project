//===--- ThrowKeywordMissingCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ThrowKeywordMissingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void ThrowKeywordMissingCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  auto CtorInitializerList =
      cxxConstructorDecl(hasAnyConstructorInitializer(anything()));

  Finder->addMatcher(
      expr(anyOf(cxxFunctionalCastExpr(), cxxBindTemporaryExpr(),
                 cxxTemporaryObjectExpr()),
           hasType(cxxRecordDecl(
               isSameOrDerivedFrom(matchesName("[Ee]xception|EXCEPTION")))),
           unless(anyOf(hasAncestor(stmt(
                            anyOf(cxxThrowExpr(), callExpr(), returnStmt()))),
                        hasAncestor(varDecl()),
                        allOf(hasAncestor(CtorInitializerList),
                              unless(hasAncestor(cxxCatchStmt()))))))
          .bind("temporary-exception-not-thrown"),
      this); 
}

void ThrowKeywordMissingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *TemporaryExpr =
      Result.Nodes.getNodeAs<Expr>("temporary-exception-not-thrown");

  diag(TemporaryExpr->getLocStart(), "suspicious exception object created but "
                                     "not thrown; did you mean 'throw %0'?")
      << TemporaryExpr->getType().getBaseTypeIdentifier()->getName();
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
