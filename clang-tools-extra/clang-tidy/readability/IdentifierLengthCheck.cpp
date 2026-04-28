//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IdentifierLengthCheck.h"
#include "../utils/DeclRefExprUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

const unsigned DefaultMinimumVariableNameLength = 3;
const unsigned DefaultMinimumBindingNameLength = 2;
const unsigned DefaultMinimumLoopCounterNameLength = 2;
const unsigned DefaultMinimumExceptionNameLength = 2;
const unsigned DefaultMinimumParameterNameLength = 3;
const char DefaultIgnoredVariableNames[] = "";
const char DefaultIgnoredBindingNames[] = "^[_]$";
const char DefaultIgnoredLoopCounterNames[] = "^[ijk_]$";
const char DefaultIgnoredExceptionVariableNames[] = "^[e]$";
const char DefaultIgnoredParameterNames[] = "^[n]$";
const unsigned DefaultLineCountThreshold = 0;

const char ErrorMessage[] =
    "%select{variable|binding variable|exception variable|loop variable|"
    "parameter}0 name %1 is too short, expected at least %2 characters";

IdentifierLengthCheck::IdentifierLengthCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MinimumVariableNameLength(Options.get("MinimumVariableNameLength",
                                            DefaultMinimumVariableNameLength)),
      MinimumBindingNameLength(Options.get("MinimumBindingNameLength",
                                           DefaultMinimumBindingNameLength)),
      MinimumLoopCounterNameLength(Options.get(
          "MinimumLoopCounterNameLength", DefaultMinimumLoopCounterNameLength)),
      MinimumExceptionNameLength(Options.get(
          "MinimumExceptionNameLength", DefaultMinimumExceptionNameLength)),
      MinimumParameterNameLength(Options.get(
          "MinimumParameterNameLength", DefaultMinimumParameterNameLength)),
      IgnoredVariableNamesInput(
          Options.get("IgnoredVariableNames", DefaultIgnoredVariableNames)),
      IgnoredVariableNames(IgnoredVariableNamesInput),
      IgnoredBindingNamesInput(
          Options.get("IgnoredBindingNames", DefaultIgnoredBindingNames)),
      IgnoredBindingNames(IgnoredBindingNamesInput),
      IgnoredLoopCounterNamesInput(Options.get("IgnoredLoopCounterNames",
                                               DefaultIgnoredLoopCounterNames)),
      IgnoredLoopCounterNames(IgnoredLoopCounterNamesInput),
      IgnoredExceptionVariableNamesInput(
          Options.get("IgnoredExceptionVariableNames",
                      DefaultIgnoredExceptionVariableNames)),
      IgnoredExceptionVariableNames(IgnoredExceptionVariableNamesInput),
      IgnoredParameterNamesInput(
          Options.get("IgnoredParameterNames", DefaultIgnoredParameterNames)),
      IgnoredParameterNames(IgnoredParameterNamesInput),
      LineCountThreshold(
          Options.get("LineCountThreshold", DefaultLineCountThreshold)) {}

void IdentifierLengthCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MinimumVariableNameLength", MinimumVariableNameLength);
  Options.store(Opts, "MinimumBindingNameLength", MinimumBindingNameLength);
  Options.store(Opts, "MinimumLoopCounterNameLength",
                MinimumLoopCounterNameLength);
  Options.store(Opts, "MinimumExceptionNameLength", MinimumExceptionNameLength);
  Options.store(Opts, "MinimumParameterNameLength", MinimumParameterNameLength);
  Options.store(Opts, "IgnoredVariableNames", IgnoredVariableNamesInput);
  Options.store(Opts, "IgnoredBindingNames", IgnoredBindingNamesInput);
  Options.store(Opts, "IgnoredLoopCounterNames", IgnoredLoopCounterNamesInput);
  Options.store(Opts, "IgnoredExceptionVariableNames",
                IgnoredExceptionVariableNamesInput);
  Options.store(Opts, "IgnoredParameterNames", IgnoredParameterNamesInput);
  Options.store(Opts, "LineCountThreshold", LineCountThreshold);
}

void IdentifierLengthCheck::registerMatchers(MatchFinder *Finder) {
  if (MinimumLoopCounterNameLength > 1)
    Finder->addMatcher(
        forStmt(hasLoopInit(declStmt(forEach(varDecl().bind("loopVar"))))),
        this);

  if (MinimumExceptionNameLength > 1)
    Finder->addMatcher(varDecl(hasParent(cxxCatchStmt())).bind("exceptionVar"),
                       this);

  if (MinimumParameterNameLength > 1)
    Finder->addMatcher(parmVarDecl().bind("paramVar"), this);

  if (MinimumBindingNameLength > 1)
    Finder->addMatcher(bindingDecl().bind("bindingVar"), this);

  if (MinimumVariableNameLength > 1)
    Finder->addMatcher(
        varDecl(unless(anyOf(hasParent(declStmt(hasParent(forStmt()))),
                             hasParent(cxxCatchStmt()), parmVarDecl())))
            .bind("standaloneVar"),
        this);
}

static std::optional<unsigned> countLinesToLastUse(const ValueDecl *Var,
                                                   const SourceManager *SrcMgr,
                                                   ASTContext *Ctx) {
  const auto *ParentScope = llvm::dyn_cast<FunctionDecl>(Var->getDeclContext());
  if (ParentScope == nullptr)
    return std::nullopt;

  auto AllRefs =
      utils::decl_ref_expr::allDeclRefExprs(*Var, *ParentScope, *Ctx);

  auto AllRefLines =
      llvm::map_range(AllRefs, [&](const DeclRefExpr *RefToVar) -> unsigned {
        return SrcMgr->getSpellingLineNumber(RefToVar->getLocation());
      });

  const unsigned DeclLine = SrcMgr->getSpellingLineNumber(Var->getLocation());
  const unsigned LastUseLine =
      AllRefLines.empty() ? DeclLine
                          : std::max(DeclLine, *llvm::max_element(AllRefLines));

  return LastUseLine - DeclLine + 1;
}

static bool isShortLived(const ValueDecl *Var, const SourceManager *SrcMgr,
                         ASTContext *Ctx, unsigned LineCountThreshold) {
  if (LineCountThreshold == 0)
    return false;

  std::optional<unsigned> LineCount = countLinesToLastUse(Var, SrcMgr, Ctx);
  if (LineCount && LineCount.value() <= LineCountThreshold)
    return true;

  return false;
}

static bool shouldWarn(const ValueDecl *Var, unsigned MinNameLength,
                       llvm::Regex const &IgnoredNames,
                       const MatchFinder::MatchResult &Result,
                       unsigned LineCountThreshold) {
  if (!Var->getIdentifier())
    return false;

  const StringRef VarName = Var->getName();

  if (VarName.size() >= MinNameLength || IgnoredNames.match(VarName))
    return false;

  if (isShortLived(Var, Result.SourceManager, Result.Context,
                   LineCountThreshold))
    return false;

  return true;
}

void IdentifierLengthCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *StandaloneVar =
      Result.Nodes.getNodeAs<ValueDecl>("standaloneVar");
  if (StandaloneVar) {
    if (shouldWarn(StandaloneVar, MinimumVariableNameLength,
                   IgnoredVariableNames, Result, LineCountThreshold))
      diag(StandaloneVar->getLocation(), ErrorMessage)
          << 0 << StandaloneVar << MinimumVariableNameLength;
    return;
  }

  const auto *BindingVar = Result.Nodes.getNodeAs<ValueDecl>("bindingVar");
  if (BindingVar) {
    if (shouldWarn(BindingVar, MinimumBindingNameLength, IgnoredBindingNames,
                   Result, LineCountThreshold))
      diag(BindingVar->getLocation(), ErrorMessage)
          << 1 << BindingVar << MinimumBindingNameLength;
    return;
  }

  auto *ExceptionVar = Result.Nodes.getNodeAs<ValueDecl>("exceptionVar");
  if (ExceptionVar) {
    if (shouldWarn(ExceptionVar, MinimumExceptionNameLength,
                   IgnoredExceptionVariableNames, Result, LineCountThreshold))
      diag(ExceptionVar->getLocation(), ErrorMessage)
          << 2 << ExceptionVar << MinimumExceptionNameLength;
    return;
  }

  const auto *LoopVar = Result.Nodes.getNodeAs<ValueDecl>("loopVar");
  if (LoopVar) {
    if (shouldWarn(LoopVar, MinimumLoopCounterNameLength,
                   IgnoredLoopCounterNames, Result, LineCountThreshold))
      diag(LoopVar->getLocation(), ErrorMessage)
          << 3 << LoopVar << MinimumLoopCounterNameLength;
    return;
  }

  const auto *ParamVar = Result.Nodes.getNodeAs<ValueDecl>("paramVar");
  if (ParamVar) {
    if (shouldWarn(ParamVar, MinimumParameterNameLength, IgnoredParameterNames,
                   Result, LineCountThreshold))
      diag(ParamVar->getLocation(), ErrorMessage)
          << 4 << ParamVar << MinimumParameterNameLength;
    return;
  }
}

} // namespace clang::tidy::readability
