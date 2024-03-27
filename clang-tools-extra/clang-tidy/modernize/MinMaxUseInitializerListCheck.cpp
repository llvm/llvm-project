//===--- MinMaxUseInitializerListCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinMaxUseInitializerListCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

struct FindArgsResult {
  const Expr *First;
  const Expr *Last;
  const Expr *Compare;
  std::vector<const Expr *> Args;
};

static const FindArgsResult findArgs(const MatchFinder::MatchResult &Match,
                                     const CallExpr *Call);
static const std::string
generateReplacement(const MatchFinder::MatchResult &Match,
                    const CallExpr *TopCall, const FindArgsResult &Result);

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
  auto createMatcher = [](const std::string &functionName) {
    auto funcDecl = functionDecl(hasName(functionName));
    auto expr = callExpr(callee(funcDecl));

    return callExpr(callee(funcDecl),
                    anyOf(hasArgument(0, expr), hasArgument(1, expr)),
                    unless(hasParent(expr)))
        .bind("topCall");
  };

  Finder->addMatcher(createMatcher("::std::max"), this);
  Finder->addMatcher(createMatcher("::std::min"), this);
}

void MinMaxUseInitializerListCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void MinMaxUseInitializerListCheck::check(
    const MatchFinder::MatchResult &Match) {

  const auto *TopCall = Match.Nodes.getNodeAs<CallExpr>("topCall");
  FindArgsResult Result = findArgs(Match, TopCall);

  if (Result.Args.size() <= 2) {
    return;
  }

  const std::string ReplacementText =
      generateReplacement(Match, TopCall, Result);

  diag(TopCall->getBeginLoc(),
       "do not use nested 'std::%0' calls, use '%1' instead")
      << TopCall->getDirectCallee()->getName() << ReplacementText
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(TopCall->getSourceRange()),
             ReplacementText)
      << Inserter.createIncludeInsertion(
             Match.SourceManager->getFileID(TopCall->getBeginLoc()),
             "<algorithm>");
}

static const FindArgsResult findArgs(const MatchFinder::MatchResult &Match,
                                     const CallExpr *Call) {
  FindArgsResult Result;
  Result.First = nullptr;
  Result.Last = nullptr;
  Result.Compare = nullptr;

  if (Call->getNumArgs() == 3) {
    auto ArgIterator = Call->arguments().begin();
    std::advance(ArgIterator, 2);
    Result.Compare = *ArgIterator;
  } else {
    auto ArgIterator = Call->arguments().begin();

    if (const auto *InitListExpr =
            dyn_cast<CXXStdInitializerListExpr>(*ArgIterator)) {
      if (const auto *TempExpr =
              dyn_cast<MaterializeTemporaryExpr>(InitListExpr->getSubExpr())) {
        if (const auto *InitList =
                dyn_cast<clang::InitListExpr>(TempExpr->getSubExpr())) {
          for (const Expr *Init : InitList->inits()) {
            Result.Args.push_back(Init);
          }
          Result.First = *ArgIterator;
          Result.Last = *ArgIterator;

          std::advance(ArgIterator, 1);
          if (ArgIterator != Call->arguments().end()) {
            Result.Compare = *ArgIterator;
          }
          return Result;
        }
      }
    }
  }

  for (const Expr *Arg : Call->arguments()) {
    if (!Result.First)
      Result.First = Arg;

    if (Arg == Result.Compare)
      continue;

    const auto *InnerCall = dyn_cast<CallExpr>(Arg->IgnoreParenImpCasts());

    if (InnerCall) {
      printf("InnerCall: %s\n",
             InnerCall->getDirectCallee()->getQualifiedNameAsString().c_str());
      printf("Call: %s\n",
             Call->getDirectCallee()->getQualifiedNameAsString().c_str());
    }

    if (InnerCall && InnerCall->getDirectCallee() &&
        InnerCall->getDirectCallee()->getQualifiedNameAsString() ==
            Call->getDirectCallee()->getQualifiedNameAsString()) {
      FindArgsResult InnerResult = findArgs(Match, InnerCall);

      const bool ProcessInnerResult =
          (!Result.Compare && !InnerResult.Compare) ||
          utils::areStatementsIdentical(Result.Compare, InnerResult.Compare,
                                        *Match.Context);

      if (ProcessInnerResult) {
        Result.Args.insert(Result.Args.end(), InnerResult.Args.begin(),
                           InnerResult.Args.end());
        Result.Last = InnerResult.Last;
        continue;
      }
    }

    Result.Args.push_back(Arg);
    Result.Last = Arg;
  }

  return Result;
}

static const std::string
generateReplacement(const MatchFinder::MatchResult &Match,
                    const CallExpr *TopCall, const FindArgsResult &Result) {

  const QualType ResultType = TopCall->getDirectCallee()
                                  ->getReturnType()
                                  .getNonReferenceType()
                                  .getUnqualifiedType()
                                  .getCanonicalType();

  std::string ReplacementText =
      Lexer::getSourceText(
          CharSourceRange::getTokenRange(
              TopCall->getBeginLoc(),
              Result.First->getBeginLoc().getLocWithOffset(-1)),
          *Match.SourceManager, Match.Context->getLangOpts())
          .str() +
      "{";

  for (const Expr *Arg : Result.Args) {
    const QualType ArgType = Arg->IgnoreParenImpCasts()
                                 ->getType()
                                 .getUnqualifiedType()
                                 .getCanonicalType();

    if (const auto *InnerCall = dyn_cast<CallExpr>(Arg)) {
      if (InnerCall->getDirectCallee()) {
        const std::string InnerCallNameStr =
            InnerCall->getDirectCallee()->getQualifiedNameAsString();

        if (InnerCallNameStr !=
                TopCall->getDirectCallee()->getQualifiedNameAsString() &&
            (InnerCallNameStr == "std::min" ||
             InnerCallNameStr == "std::max")) {
          FindArgsResult innerResult = findArgs(Match, InnerCall);
          if (innerResult.Args.size() > 2) {
            ReplacementText +=
                generateReplacement(Match, InnerCall, innerResult) + ", ";

            continue;
          }
        }
      }
    }

    const bool CastNeeded = ArgType != ResultType;

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
