//===--- UseDesignatedInitializersCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseDesignatedInitializersCheck.h"
#include "clang/AST/APValue.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/DesignatedInitializers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static constexpr char IgnoreSingleElementAggregatesName[] =
    "IgnoreSingleElementAggregates";
static constexpr bool IgnoreSingleElementAggregatesDefault = true;

static constexpr char RestrictToPODTypesName[] = "RestrictToPODTypes";
static constexpr bool RestrictToPODTypesDefault = false;

static constexpr char IgnoreMacrosName[] = "IgnoreMacros";
static constexpr bool IgnoreMacrosDefault = true;

namespace {

AST_MATCHER(CXXRecordDecl, isAggregate) { return Node.isAggregate(); }

AST_MATCHER(CXXRecordDecl, isPOD) { return Node.isPOD(); }

AST_MATCHER(InitListExpr, isFullyDesignated) {
  if (const InitListExpr *SyntacticForm =
          Node.isSyntacticForm() ? &Node : Node.getSyntacticForm()) {
    return llvm::all_of(SyntacticForm->children(), [](auto *InitExpr) {
      return isa<DesignatedInitExpr>(InitExpr);
    });
  }
  return true;
}

AST_MATCHER(InitListExpr, hasSingleElement) { return Node.getNumInits() == 1; }

} // namespace

UseDesignatedInitializersCheck::UseDesignatedInitializersCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), IgnoreSingleElementAggregates(Options.get(
                                         IgnoreSingleElementAggregatesName,
                                         IgnoreSingleElementAggregatesDefault)),
      RestrictToPODTypes(
          Options.get(RestrictToPODTypesName, RestrictToPODTypesDefault)),
      IgnoreMacros(Options.get(IgnoreMacrosName, IgnoreMacrosDefault)) {}

void UseDesignatedInitializersCheck::registerMatchers(MatchFinder *Finder) {
  const auto HasBaseWithFields =
      hasAnyBase(hasType(cxxRecordDecl(has(fieldDecl()))));
  Finder->addMatcher(
      initListExpr(
          hasType(cxxRecordDecl(RestrictToPODTypes ? isPOD() : isAggregate(),
                                unless(HasBaseWithFields))
                      .bind("type")),
          unless(IgnoreSingleElementAggregates ? hasSingleElement()
                                               : unless(anything())),
          unless(isFullyDesignated()))
          .bind("init"),
      this);
}

static bool isFullyUndesignated(const InitListExpr *SyntacticInitList) {
  return std::all_of(
      SyntacticInitList->begin(), SyntacticInitList->end(),
      [](auto *InitExpr) { return !isa<DesignatedInitExpr>(InitExpr); });
}

void UseDesignatedInitializersCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InitList = Result.Nodes.getNodeAs<InitListExpr>("init");
  const auto *Type = Result.Nodes.getNodeAs<CXXRecordDecl>("type");
  if (!Type || !InitList)
    return;
  if (const auto *SyntacticInitList = InitList->getSyntacticForm()) {
    const llvm::DenseMap<clang::SourceLocation, std::string> Designators =
        clang::tooling::getUnwrittenDesignators(SyntacticInitList);
    if (isFullyUndesignated(SyntacticInitList)) {
      if (IgnoreMacros && InitList->getBeginLoc().isMacroID()) {
        return;
      }
      DiagnosticBuilder Diag =
          diag(InitList->getLBraceLoc(), "use designated initializer list");
      for (const Stmt *InitExpr : *SyntacticInitList) {
        Diag << FixItHint::CreateInsertion(
            InitExpr->getBeginLoc(),
            Designators.at(InitExpr->getBeginLoc()) + "=");
      }
      return;
    }
    for (const auto *InitExpr : *SyntacticInitList) {
      if (isa<DesignatedInitExpr>(InitExpr)) {
        continue;
      }
      if (IgnoreMacros && InitExpr->getBeginLoc().isMacroID()) {
        continue;
      }
      diag(InitExpr->getBeginLoc(), "use designated init expression")
          << FixItHint::CreateInsertion(
                 InitExpr->getBeginLoc(),
                 Designators.at(InitExpr->getBeginLoc()) + "=");
    }
  }
}

void UseDesignatedInitializersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, IgnoreSingleElementAggregatesName,
                IgnoreSingleElementAggregates);
  Options.store(Opts, RestrictToPODTypesName, RestrictToPODTypes);
  Options.store(Opts, IgnoreMacrosName, IgnoreMacros);
}

} // namespace clang::tidy::modernize
