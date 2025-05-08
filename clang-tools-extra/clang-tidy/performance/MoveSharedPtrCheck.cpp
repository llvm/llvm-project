//===--- MoveSharedPtrCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MoveSharedPtrCheck.h"
#include "../utils/DeclRefExprUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

using utils::decl_ref_expr::allDeclRefExprs;

void MoveSharedPtrCheck::registerMatchers(MatchFinder* Finder) {
  Finder->addMatcher(
      declRefExpr(
          hasDeclaration(
              varDecl(hasAncestor(functionDecl().bind("func"))).bind("decl")),
          hasParent(expr(hasParent(cxxConstructExpr())).bind("use_parent"))

              )
          .bind("use"),
      this);
}

const Expr* MoveSharedPtrCheck::getLastVarUsage(const VarDecl& Var,
                                                const Decl& Func,
                                                ASTContext& Context) {
  auto Exprs = allDeclRefExprs(Var, Func, Context);

  const Expr* LastExpr = nullptr;
  for (const auto& Expr : Exprs) {
    if (!LastExpr) LastExpr = Expr;

    if (LastExpr->getBeginLoc() < Expr->getBeginLoc()) LastExpr = Expr;
  }

  // diag(LastExpr->getBeginLoc(), "last usage");
  return LastExpr;
}

const std::string_view kSharedPtr = "std::shared_ptr<";

void MoveSharedPtrCheck::check(const MatchFinder::MatchResult& Result) {
  const auto* MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("decl");
  const auto* MatchedFunc = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto* MatchedUse = Result.Nodes.getNodeAs<Expr>("use");
  const auto* MatchedUseCall = Result.Nodes.getNodeAs<CallExpr>("use_parent");

  if (MatchedUseCall) return;

  auto Type = MatchedDecl->getType().getAsString();
  if (std::string_view(Type).substr(0, kSharedPtr.size()) != kSharedPtr) return;

  const auto* LastUsage =
      getLastVarUsage(*MatchedDecl, *MatchedFunc, *Result.Context);
  if (LastUsage == nullptr) return;

  if (LastUsage->getBeginLoc() > MatchedUse->getBeginLoc()) {
    // "use" is not the last reference to x
    return;
  }

  diag(LastUsage->getBeginLoc(), Type);
  diag(LastUsage->getBeginLoc(), "Could be std::move()");
}

}  // namespace clang::tidy::performance
