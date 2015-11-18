//===--- ProBoundsArrayToPointerDecayCheck.cpp - clang-tidy----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProBoundsArrayToPointerDecayCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {

AST_MATCHER_P(CXXForRangeStmt, hasRangeBeginEndStmt,
              ast_matchers::internal::Matcher<DeclStmt>, InnerMatcher) {
  const DeclStmt *const Stmt = Node.getBeginEndStmt();
  return (Stmt != nullptr && InnerMatcher.matches(*Stmt, Finder, Builder));
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
    ASTContext::DynTypedNodeList Parents =
        Finder->getASTContext().getParents(*E);
    if (Parents.size() != 1)
      return false;
    E = Parents[0].get<Expr>();
    if (!E)
      return false;
  } while (isa<ImplicitCastExpr>(E));

  return InnerMatcher.matches(*E, Finder, Builder);
}

void ProBoundsArrayToPointerDecayCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  // The only allowed array to pointer decay
  // 1) just before array subscription
  // 2) inside a range-for over an array
  // 3) if it converts a string literal to a pointer
  Finder->addMatcher(
      implicitCastExpr(unless(hasParent(arraySubscriptExpr())),
                       unless(hasParentIgnoringImpCasts(explicitCastExpr())),
                       unless(isInsideOfRangeBeginEndStmt()),
                       unless(hasSourceExpression(stringLiteral())))
          .bind("cast"),
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

} // namespace tidy
} // namespace clang
