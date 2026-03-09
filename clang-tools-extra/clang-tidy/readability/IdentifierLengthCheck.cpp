//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IdentifierLengthCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

const unsigned DefaultMinimumVariableNameLength = 3;
const unsigned DefaultMinimumLoopCounterNameLength = 2;
const unsigned DefaultMinimumExceptionNameLength = 2;
const unsigned DefaultMinimumParameterNameLength = 3;
const char DefaultIgnoredLoopCounterNames[] = "^[ijk_]$";
const char DefaultIgnoredVariableNames[] = "";
const char DefaultIgnoredExceptionVariableNames[] = "^[e]$";
const char DefaultIgnoredParameterNames[] = "^[n]$";
const unsigned DefaultLineCountThreshold = 0;

const char ErrorMessage[] =
    "%select{variable|exception variable|loop variable|"
    "parameter}0 name %1 is too short, expected at least %2 characters";

IdentifierLengthCheck::IdentifierLengthCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MinimumVariableNameLength(Options.get("MinimumVariableNameLength",
                                            DefaultMinimumVariableNameLength)),
      MinimumLoopCounterNameLength(Options.get(
          "MinimumLoopCounterNameLength", DefaultMinimumLoopCounterNameLength)),
      MinimumExceptionNameLength(Options.get(
          "MinimumExceptionNameLength", DefaultMinimumExceptionNameLength)),
      MinimumParameterNameLength(Options.get(
          "MinimumParameterNameLength", DefaultMinimumParameterNameLength)),
      IgnoredVariableNamesInput(
          Options.get("IgnoredVariableNames", DefaultIgnoredVariableNames)),
      IgnoredVariableNames(IgnoredVariableNamesInput),
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
  Options.store(Opts, "MinimumLoopCounterNameLength",
                MinimumLoopCounterNameLength);
  Options.store(Opts, "MinimumExceptionNameLength", MinimumExceptionNameLength);
  Options.store(Opts, "MinimumParameterNameLength", MinimumParameterNameLength);
  Options.store(Opts, "IgnoredLoopCounterNames", IgnoredLoopCounterNamesInput);
  Options.store(Opts, "IgnoredVariableNames", IgnoredVariableNamesInput);
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

  if (MinimumVariableNameLength > 1)
    Finder->addMatcher(
        varDecl(unless(anyOf(hasParent(declStmt(hasParent(forStmt()))),
                             hasParent(cxxCatchStmt()), parmVarDecl())))
            .bind("standaloneVar"),
        this);
}

static unsigned countLinesToLastUse(const VarDecl *Var,
                                    const SourceManager *SrcMgr,
                                    ASTContext *Ctx) {
  const unsigned DeclLine = SrcMgr->getSpellingLineNumber(Var->getLocation());

  class VarUseCallback : public MatchFinder::MatchCallback {
  private:
    unsigned *LastUseLineNumber;

  public:
    explicit VarUseCallback(unsigned *Output) : LastUseLineNumber{Output} {}

    void run(const MatchFinder::MatchResult &Result) override {
      const auto *Use = Result.Nodes.getNodeAs<DeclRefExpr>("varUse");
      if (Use && LastUseLineNumber) {
        auto Loc = Use->getLocation();
        const unsigned UseLine =
            Result.SourceManager->getSpellingLineNumber(Loc);
        *LastUseLineNumber = std::max(*LastUseLineNumber, UseLine);
      }
    }
  };

  unsigned LastUseLine = DeclLine;
  VarUseCallback Callback{&LastUseLine};

  auto Matcher = declRefExpr(to(varDecl(equalsNode(Var)))).bind("varUse");

  MatchFinder Finder;
  Finder.addMatcher(Matcher, &Callback);
  Finder.matchAST(*Ctx);

  return LastUseLine - DeclLine + 1;
}

void IdentifierLengthCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *StandaloneVar = Result.Nodes.getNodeAs<VarDecl>("standaloneVar");
  if (StandaloneVar) {
    if (!StandaloneVar->getIdentifier())
      return;

    const StringRef VarName = StandaloneVar->getName();

    if (VarName.size() >= MinimumVariableNameLength ||
        IgnoredVariableNames.match(VarName))
      return;

    if (LineCountThreshold > 0 &&
        countLinesToLastUse(StandaloneVar, Result.SourceManager,
                            Result.Context) <= LineCountThreshold)
      return;

    diag(StandaloneVar->getLocation(), ErrorMessage)
        << 0 << StandaloneVar << MinimumVariableNameLength;
  }

  auto *ExceptionVarName = Result.Nodes.getNodeAs<VarDecl>("exceptionVar");
  if (ExceptionVarName) {
    if (!ExceptionVarName->getIdentifier())
      return;

    const StringRef VarName = ExceptionVarName->getName();
    if (VarName.size() >= MinimumExceptionNameLength ||
        IgnoredExceptionVariableNames.match(VarName))
      return;

    if (LineCountThreshold > 0 &&
        countLinesToLastUse(ExceptionVarName, Result.SourceManager,
                            Result.Context) <= LineCountThreshold)
      return;

    diag(ExceptionVarName->getLocation(), ErrorMessage)
        << 1 << ExceptionVarName << MinimumExceptionNameLength;
  }

  const auto *LoopVar = Result.Nodes.getNodeAs<VarDecl>("loopVar");
  if (LoopVar) {
    if (!LoopVar->getIdentifier())
      return;

    const StringRef VarName = LoopVar->getName();

    if (VarName.size() >= MinimumLoopCounterNameLength ||
        IgnoredLoopCounterNames.match(VarName))
      return;

    if (LineCountThreshold > 0 &&
        countLinesToLastUse(LoopVar, Result.SourceManager, Result.Context) <=
            LineCountThreshold)
      return;

    diag(LoopVar->getLocation(), ErrorMessage)
        << 2 << LoopVar << MinimumLoopCounterNameLength;
  }

  const auto *ParamVar = Result.Nodes.getNodeAs<VarDecl>("paramVar");
  if (ParamVar) {
    if (!ParamVar->getIdentifier())
      return;

    const StringRef VarName = ParamVar->getName();

    if (VarName.size() >= MinimumParameterNameLength ||
        IgnoredParameterNames.match(VarName))
      return;

    if (LineCountThreshold > 0 &&
        countLinesToLastUse(ParamVar, Result.SourceManager, Result.Context) <=
            LineCountThreshold)
      return;

    diag(ParamVar->getLocation(), ErrorMessage)
        << 3 << ParamVar << MinimumParameterNameLength;
  }
}

} // namespace clang::tidy::readability
