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

static constexpr auto IgnoreSingleElementAggregatesName =
    "IgnoreSingleElementAggregates";
static constexpr auto IgnoreSingleElementAggregatesDefault = true;

static std::vector<Stmt *>
getUndesignatedComponents(const InitListExpr *SyntacticInitList) {
  std::vector<Stmt *> Result;
  std::copy_if(SyntacticInitList->begin(), SyntacticInitList->end(),
               std::back_inserter(Result),
               [](auto S) { return !isa<DesignatedInitExpr>(S); });
  return Result;
}

UseDesignatedInitializersCheck::UseDesignatedInitializersCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreSingleElementAggregates(
          Options.getLocalOrGlobal(IgnoreSingleElementAggregatesName,
                                   IgnoreSingleElementAggregatesDefault)) {}

void UseDesignatedInitializersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      initListExpr(hasType(recordDecl().bind("type"))).bind("init"), this);
}

void UseDesignatedInitializersCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InitList = Result.Nodes.getNodeAs<InitListExpr>("init");
  const auto *Type = Result.Nodes.getNodeAs<CXXRecordDecl>("type");
  if (!Type || !InitList || !Type->isAggregate())
    return;
  if (IgnoreSingleElementAggregates && InitList->getNumInits() == 1)
    return;
  if (const auto *SyntacticInitList = InitList->getSyntacticForm()) {
    const auto UndesignatedComponents =
        getUndesignatedComponents(SyntacticInitList);
    if (UndesignatedComponents.empty())
      return;
    if (UndesignatedComponents.size() == SyntacticInitList->getNumInits()) {
      diag(InitList->getLBraceLoc(), "use designated initializer list");
      return;
    }
    for (const auto *InitExpr : UndesignatedComponents) {
      diag(InitExpr->getBeginLoc(), "use designated init expression");
    }
  }
}

void UseDesignatedInitializersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, IgnoreSingleElementAggregatesName,
                IgnoreSingleElementAggregates);
}

} // namespace clang::tidy::modernize
