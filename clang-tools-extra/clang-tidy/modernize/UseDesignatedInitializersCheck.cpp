//===--- UseDesignatedInitializersCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseDesignatedInitializersCheck.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include <algorithm>
#include <iterator>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static std::vector<Stmt *>
getAllUndesignatedInits(const InitListExpr *SyntacticInitList) {
  std::vector<Stmt *> Result;
  std::copy_if(SyntacticInitList->begin(), SyntacticInitList->end(),
               std::back_inserter(Result),
               [](auto S) { return !isa<DesignatedInitExpr>(S); });
  return Result;
}

void UseDesignatedInitializersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(varDecl(allOf(has(initListExpr().bind("init")),
                                   hasType(recordDecl().bind("type")))),
                     this);
}

void UseDesignatedInitializersCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InitList = Result.Nodes.getNodeAs<InitListExpr>("init");
  const auto *Type = Result.Nodes.getNodeAs<CXXRecordDecl>("type");
  if (!Type || !InitList || !Type->isAggregate())
    return;
  if (const auto *SyntacticInitList = InitList->getSyntacticForm()) {
    const auto UndesignatedParts = getAllUndesignatedInits(SyntacticInitList);
    if (UndesignatedParts.empty())
      return;
    if (UndesignatedParts.size() == SyntacticInitList->getNumInits()) {
      diag(InitList->getLBraceLoc(), "use designated initializer list");
      return;
    }
    for (const auto *InitExpr : UndesignatedParts) {
      diag(InitExpr->getBeginLoc(), "use designated init expression");
    }
  }
}

} // namespace clang::tidy::modernize
