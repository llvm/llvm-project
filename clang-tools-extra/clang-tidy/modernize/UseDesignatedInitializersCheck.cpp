//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseDesignatedInitializersCheck.h"
#include "../utils/DesignatedInitializers.h"
#include "clang/AST/APValue.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static constexpr char IgnoreSingleElementAggregatesName[] =
    "IgnoreSingleElementAggregates";
static constexpr bool IgnoreSingleElementAggregatesDefault = true;

static constexpr char RestrictToPODTypesName[] = "RestrictToPODTypes";
static constexpr bool RestrictToPODTypesDefault = false;

static constexpr char IgnoreMacrosName[] = "IgnoreMacros";
static constexpr bool IgnoreMacrosDefault = true;

static constexpr char StrictCStandardComplianceName[] =
    "StrictCStandardCompliance";
static constexpr bool StrictCStandardComplianceDefault = true;

static constexpr char StrictCppStandardComplianceName[] =
    "StrictCppStandardCompliance";
static constexpr bool StrictCppStandardComplianceDefault = true;

namespace {

struct Designators {

  Designators(const InitListExpr *InitList) : InitList(InitList) {
    assert(InitList->isSyntacticForm());
  };

  unsigned size() { return getCached().size(); }

  std::optional<llvm::StringRef> operator[](const SourceLocation &Location) {
    const auto &Designators = getCached();
    const auto Result = Designators.find(Location);
    if (Result == Designators.end())
      return {};
    const llvm::StringRef Designator = Result->getSecond();
    return (Designator.front() == '.' ? Designator.substr(1) : Designator)
        .trim("\0"); // Trim NULL characters appearing on Windows in the
                     // name.
  }

private:
  using LocationToNameMap = llvm::DenseMap<clang::SourceLocation, std::string>;

  std::optional<LocationToNameMap> CachedDesignators;
  const InitListExpr *InitList;

  LocationToNameMap &getCached() {
    return CachedDesignators ? *CachedDesignators
                             : CachedDesignators.emplace(
                                   utils::getUnwrittenDesignators(InitList));
  }
};

unsigned getNumberOfDesignated(const InitListExpr *SyntacticInitList) {
  return llvm::count_if(*SyntacticInitList, [](auto *InitExpr) {
    return isa<DesignatedInitExpr>(InitExpr);
  });
}

AST_MATCHER(CXXRecordDecl, isAggregate) {
  return Node.hasDefinition() && Node.isAggregate();
}

AST_MATCHER(CXXRecordDecl, isPOD) {
  return Node.hasDefinition() && Node.isPOD();
}

AST_MATCHER(InitListExpr, isFullyDesignated) {
  if (const InitListExpr *SyntacticForm =
          Node.isSyntacticForm() ? &Node : Node.getSyntacticForm())
    return getNumberOfDesignated(SyntacticForm) == SyntacticForm->getNumInits();
  return true;
}

AST_MATCHER(InitListExpr, hasMoreThanOneElement) {
  return Node.getNumInits() > 1;
}

} // namespace

UseDesignatedInitializersCheck::UseDesignatedInitializersCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), IgnoreSingleElementAggregates(Options.get(
                                         IgnoreSingleElementAggregatesName,
                                         IgnoreSingleElementAggregatesDefault)),
      RestrictToPODTypes(
          Options.get(RestrictToPODTypesName, RestrictToPODTypesDefault)),
      IgnoreMacros(Options.get(IgnoreMacrosName, IgnoreMacrosDefault)),
      StrictCStandardCompliance(Options.get(StrictCStandardComplianceName,
                                            StrictCStandardComplianceDefault)),
      StrictCppStandardCompliance(
          Options.get(StrictCppStandardComplianceName,
                      StrictCppStandardComplianceDefault)) {}

void UseDesignatedInitializersCheck::registerMatchers(MatchFinder *Finder) {
  const auto HasBaseWithFields =
      hasAnyBase(hasType(cxxRecordDecl(has(fieldDecl()))));
  Finder->addMatcher(
      initListExpr(
          hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
              cxxRecordDecl(
                  RestrictToPODTypes ? isPOD() : isAggregate(),
                  unless(anyOf(HasBaseWithFields, hasName("::std::array"))))
                  .bind("type"))))),
          IgnoreSingleElementAggregates ? hasMoreThanOneElement() : anything(),
          unless(isFullyDesignated()))
          .bind("init"),
      this);
}

void UseDesignatedInitializersCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InitList = Result.Nodes.getNodeAs<InitListExpr>("init");
  const auto *Type = Result.Nodes.getNodeAs<CXXRecordDecl>("type");
  if (!Type || !InitList)
    return;
  const auto *SyntacticInitList = InitList->getSyntacticForm();
  if (!SyntacticInitList)
    return;
  Designators Designators{SyntacticInitList};
  const unsigned NumberOfDesignated = getNumberOfDesignated(SyntacticInitList);
  if (SyntacticInitList->getNumInits() - NumberOfDesignated >
      Designators.size())
    return;

  // If the whole initializer list is un-designated, issue only one warning and
  // a single fix-it for the whole expression.
  if (0 == NumberOfDesignated) {
    if (IgnoreMacros && InitList->getBeginLoc().isMacroID())
      return;
    {
      DiagnosticBuilder Diag =
          diag(InitList->getLBraceLoc(),
               "use designated initializer list to initialize %0");
      Diag << InitList->getType() << InitList->getSourceRange();
      for (const Stmt *InitExpr : *SyntacticInitList) {
        const auto Designator = Designators[InitExpr->getBeginLoc()];
        if (Designator && !Designator->empty())
          Diag << FixItHint::CreateInsertion(InitExpr->getBeginLoc(),
                                             ("." + *Designator + "=").str());
      }
    }
    diag(Type->getBeginLoc(), "aggregate type is defined here",
         DiagnosticIDs::Note);
    return;
  }

  // In case that only a few elements are un-designated (not all as before), the
  // check offers dedicated issues and fix-its for each of them.
  for (const auto *InitExpr : *SyntacticInitList) {
    if (isa<DesignatedInitExpr>(InitExpr))
      continue;
    if (IgnoreMacros && InitExpr->getBeginLoc().isMacroID())
      continue;
    const auto Designator = Designators[InitExpr->getBeginLoc()];
    if (!Designator || Designator->empty()) {
      // There should always be a designator. If there's unexpectedly none, we
      // at least report a generic diagnostic.
      diag(InitExpr->getBeginLoc(), "use designated init expression")
          << InitExpr->getSourceRange();
    } else {
      diag(InitExpr->getBeginLoc(),
           "use designated init expression to initialize field '%0'")
          << InitExpr->getSourceRange() << *Designator
          << FixItHint::CreateInsertion(InitExpr->getBeginLoc(),
                                        ("." + *Designator + "=").str());
    }
  }
}

void UseDesignatedInitializersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, IgnoreSingleElementAggregatesName,
                IgnoreSingleElementAggregates);
  Options.store(Opts, RestrictToPODTypesName, RestrictToPODTypes);
  Options.store(Opts, IgnoreMacrosName, IgnoreMacros);
  Options.store(Opts, StrictCStandardComplianceName, StrictCStandardCompliance);
  Options.store(Opts, StrictCppStandardComplianceName,
                StrictCppStandardCompliance);
}

} // namespace clang::tidy::modernize
