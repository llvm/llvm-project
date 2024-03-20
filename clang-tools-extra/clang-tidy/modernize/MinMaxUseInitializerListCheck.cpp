//===--- MinMaxUseInitializerListCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinMaxUseInitializerListCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

MinMaxUseInitializerListCheck::MinMaxUseInitializerListCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void MinMaxUseInitializerListCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

void MinMaxUseInitializerListCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("::std::max"))),
          anyOf(hasArgument(
                    0, callExpr(callee(functionDecl(hasName("::std::max"))))),
                hasArgument(
                    1, callExpr(callee(functionDecl(hasName("::std::max")))))),
          unless(
              hasParent(callExpr(callee(functionDecl(hasName("::std::max")))))))
          .bind("topCall"),
      this);

  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("::std::min"))),
          anyOf(hasArgument(
                    0, callExpr(callee(functionDecl(hasName("::std::min"))))),
                hasArgument(
                    1, callExpr(callee(functionDecl(hasName("::std::min")))))),
          unless(
              hasParent(callExpr(callee(functionDecl(hasName("::std::min")))))))
          .bind("topCall"),
      this);
}

void MinMaxUseInitializerListCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void MinMaxUseInitializerListCheck::check(
    const MatchFinder::MatchResult &Match) {
  const CallExpr *TopCall = Match.Nodes.getNodeAs<CallExpr>("topCall");
  MinMaxUseInitializerListCheck::FindArgsResult Result =
      findArgs(Match, TopCall);

  if (!Result.First || !Result.Last || Result.Args.size() <= 2) {
    return;
  }

  std::string ReplacementText = generateReplacement(Match, TopCall, Result);

  diag(TopCall->getBeginLoc(),
       "do not use nested std::%0 calls, use %1 instead")
      << TopCall->getDirectCallee()->getName() << ReplacementText
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(TopCall->getBeginLoc(),
                                            TopCall->getEndLoc()),
             ReplacementText)
      << Inserter.createMainFileIncludeInsertion("<algorithm>");
}

MinMaxUseInitializerListCheck::FindArgsResult
MinMaxUseInitializerListCheck::findArgs(const MatchFinder::MatchResult &Match,
                                        const CallExpr *Call) {
  FindArgsResult Result;
  Result.First = nullptr;
  Result.Last = nullptr;
  Result.Compare = nullptr;

  if (Call->getNumArgs() > 2) {
    auto argIterator = Call->arguments().begin();
    std::advance(argIterator, 2);
    Result.Compare = *argIterator;
  }

  for (const Expr *Arg : Call->arguments()) {
    if (!Result.First)
      Result.First = Arg;

    const CallExpr *InnerCall = dyn_cast<CallExpr>(Arg);
    if (InnerCall && InnerCall->getDirectCallee() &&
        InnerCall->getDirectCallee()->getNameAsString() ==
            Call->getDirectCallee()->getNameAsString()) {
      FindArgsResult InnerResult = findArgs(Match, InnerCall);

      bool processInnerResult = false;

      if (!Result.Compare && !InnerResult.Compare)
        processInnerResult = true;
      else if (Result.Compare && InnerResult.Compare &&
               Lexer::getSourceText(CharSourceRange::getTokenRange(
                                        Result.Compare->getSourceRange()),
                                    *Match.SourceManager,
                                    Match.Context->getLangOpts()) ==
                   Lexer::getSourceText(
                       CharSourceRange::getTokenRange(
                           InnerResult.Compare->getSourceRange()),
                       *Match.SourceManager, Match.Context->getLangOpts()))
        processInnerResult = true;

      if (processInnerResult) {
        Result.Args.insert(Result.Args.end(), InnerResult.Args.begin(),
                           InnerResult.Args.end());
        continue;
      }
    }

    if (Arg == Result.Compare)
      continue;

    Result.Args.push_back(Arg);
    Result.Last = Arg;
  }

  return Result;
}

std::string MinMaxUseInitializerListCheck::generateReplacement(
    const MatchFinder::MatchResult &Match, const CallExpr *TopCall,
    const FindArgsResult Result) {
  std::string ReplacementText =
      Lexer::getSourceText(
          CharSourceRange::getTokenRange(
              TopCall->getBeginLoc(),
              Result.First->getBeginLoc().getLocWithOffset(-1)),
          *Match.SourceManager, Match.Context->getLangOpts())
          .str() +
      "{";
  const QualType ResultType =
      TopCall->getDirectCallee()->getReturnType().getNonReferenceType();

  for (const Expr *Arg : Result.Args) {
    QualType ArgType = Arg->getType();

    // check if expression is std::min or std::max
    if (const auto *InnerCall = dyn_cast<CallExpr>(Arg)) {
      if (InnerCall->getDirectCallee() &&
          InnerCall->getDirectCallee()->getNameAsString() !=
              TopCall->getDirectCallee()->getNameAsString()) {
        FindArgsResult innerResult = findArgs(Match, InnerCall);
        ReplacementText += generateReplacement(Match, InnerCall, innerResult) +=
            "})";
        continue;
      }
    }

    bool CastNeeded =
        ArgType.getCanonicalType() != ResultType.getCanonicalType();

    if (CastNeeded)
      ReplacementText += "static_cast<" + ResultType.getAsString() + ">(";

    ReplacementText += Lexer::getSourceText(
        CharSourceRange::getTokenRange(Arg->getSourceRange()),
        *Match.SourceManager, Match.Context->getLangOpts());

    if (CastNeeded)
      ReplacementText += ")";
    ReplacementText += ", ";
  }
  ReplacementText = ReplacementText.substr(0, ReplacementText.size() - 2) + "}";
  if (Result.Compare) {
    ReplacementText += ", ";
    ReplacementText += Lexer::getSourceText(
        CharSourceRange::getTokenRange(Result.Compare->getSourceRange()),
        *Match.SourceManager, Match.Context->getLangOpts());
  }
  ReplacementText += ")";

  return ReplacementText;
}

} // namespace clang::tidy::modernize
