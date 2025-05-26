//===--- LostStdMoveCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LostStdMoveCheck.h"
#include "../utils/DeclRefExprUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

using utils::decl_ref_expr::allDeclRefExprs;

static const Expr* getLastVarUsage(const VarDecl& Var, const Decl& Func,
                                   ASTContext& Context) {
  auto Exprs = allDeclRefExprs(Var, Func, Context);

  const Expr* LastExpr = nullptr;
  for (const clang::DeclRefExpr* Expr : Exprs) {
    if (!LastExpr) LastExpr = Expr;

    if (LastExpr->getBeginLoc() < Expr->getBeginLoc()) LastExpr = Expr;
  }

  return LastExpr;
}

AST_MATCHER(CXXRecordDecl, hasTrivialMoveConstructor) {
  return Node.hasDefinition() && Node.hasTrivialMoveConstructor();
}

void LostStdMoveCheck::registerMatchers(MatchFinder* Finder) {
  auto returnParent =
      hasParent(expr(hasParent(cxxConstructExpr(hasParent(returnStmt())))));

  auto outermostExpr = expr(unless(hasParent(expr())));
  auto leafStatement =
      stmt(outermostExpr, unless(hasDescendant(outermostExpr)));

  Finder->addMatcher(
      declRefExpr(
          // not "return x;"
          unless(returnParent),

          unless(hasType(namedDecl(hasName("::std::string_view")))),

          // non-trivial type
          hasType(hasCanonicalType(hasDeclaration(cxxRecordDecl()))),

          // non-trivial X(X&&)
          unless(hasType(hasCanonicalType(
              hasDeclaration(cxxRecordDecl(hasTrivialMoveConstructor()))))),

          // Not in a cycle
          unless(hasAncestor(forStmt())), unless(hasAncestor(doStmt())),
          unless(hasAncestor(whileStmt())),

          // only non-X&
          unless(hasDeclaration(
              varDecl(hasType(qualType(lValueReferenceType()))))),

          hasAncestor(leafStatement.bind("leaf_statement")),

          hasDeclaration(
              varDecl(hasAncestor(functionDecl().bind("func"))).bind("decl")),

          hasParent(expr(hasParent(cxxConstructExpr())).bind("use_parent")))
          .bind("use"),
      this);
}

template <typename Node>
void extractNodesByIdTo(ArrayRef<BoundNodes> Matches, StringRef ID,
                        llvm::SmallPtrSet<const Node*, 16>& Nodes) {
  for (const BoundNodes& Match : Matches)
    Nodes.insert(Match.getNodeAs<Node>(ID));
}

void LostStdMoveCheck::check(const MatchFinder::MatchResult& Result) {
  const auto* MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("decl");
  const auto* MatchedFunc = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto* MatchedUse = Result.Nodes.getNodeAs<Expr>("use");
  const auto* MatchedUseCall = Result.Nodes.getNodeAs<CallExpr>("use_parent");
  const auto* MatchedLeafStatement =
      Result.Nodes.getNodeAs<Stmt>("leaf_statement");

  if (MatchedUseCall) return;

  const Expr* LastUsage =
      getLastVarUsage(*MatchedDecl, *MatchedFunc, *Result.Context);
  if (LastUsage == nullptr) return;

  if (LastUsage->getBeginLoc() > MatchedUse->getBeginLoc()) {
    // "use" is not the last reference to x
    return;
  }

  // Calculate X usage count in the statement
  llvm::SmallPtrSet<const DeclRefExpr*, 16> DeclRefs;
  ArrayRef<BoundNodes> Matches = match(
      findAll(declRefExpr(to(varDecl(equalsNode(MatchedDecl)))).bind("ref")),
      *MatchedLeafStatement, *Result.Context);
  extractNodesByIdTo(Matches, "ref", DeclRefs);
  if (DeclRefs.size() > 1) {
    // Unspecified order of evaluation, e.g. f(x, x)
    return;
  }

  diag(LastUsage->getBeginLoc(), "could be std::move()");
}

}  // namespace clang::tidy::performance
