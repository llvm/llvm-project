//===--- ProBoundsArrayToPointerDecayCheck.cpp - clang-tidy----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProBoundsArrayToPointerDecayCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {
AST_MATCHER_P(CXXForRangeStmt, hasRangeBeginEndStmt,
              ast_matchers::internal::Matcher<DeclStmt>, InnerMatcher) {
  for (const DeclStmt *Stmt : {Node.getBeginStmt(), Node.getEndStmt()})
    if (Stmt != nullptr && InnerMatcher.matches(*Stmt, Finder, Builder))
      return true;
  return false;
}

AST_MATCHER(Stmt, isInsideOfRangeBeginEndStmt) {
  return stmt(hasAncestor(cxxForRangeStmt(
                  hasRangeBeginEndStmt(hasDescendant(equalsNode(&Node))))))
      .matches(Node, Finder, Builder);
}

AST_MATCHER_P(Expr, hasParentIgnoringImpCasts,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  const Expr *E = &Node;
  do {
    DynTypedNodeList Parents = Finder->getASTContext().getParents(*E);
    if (Parents.size() != 1)
      return false;
    E = Parents[0].get<Expr>();
    if (!E)
      return false;
  } while (isa<ImplicitCastExpr>(E));

  return InnerMatcher.matches(*E, Finder, Builder);
}
} // namespace

void ProBoundsArrayToPointerDecayCheck::registerMatchers(MatchFinder *Finder) {
  // The only allowed array to pointer decay
  // 1) just before array subscription
  // 2) inside a range-for over an array
  // 3) if it converts a string literal to a pointer
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          implicitCastExpr(
              unless(hasParent(arraySubscriptExpr())),
              unless(hasSourceExpression(predefinedExpr())),
              unless(hasParentIgnoringImpCasts(explicitCastExpr())),
              unless(isInsideOfRangeBeginEndStmt()),
              unless(hasSourceExpression(ignoringParens(stringLiteral()))),
              unless(hasSourceExpression(ignoringParens(
                  conditionalOperator(hasTrueExpression(stringLiteral()),
                                      hasFalseExpression(stringLiteral()))))))
              .bind("cast")),
      this);
}

void ProBoundsArrayToPointerDecayCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast = Result.Nodes.getNodeAs<ImplicitCastExpr>("cast");
  if (MatchedCast->getCastKind() != CK_ArrayToPointerDecay)
    return;

  diag(MatchedCast->getExprLoc(), "do not implicitly decay an array into a "
                                  "pointer; consider using gsl::array_view or "
                                  "an explicit cast instead");
}

} // namespace clang::tidy::cppcoreguidelines
