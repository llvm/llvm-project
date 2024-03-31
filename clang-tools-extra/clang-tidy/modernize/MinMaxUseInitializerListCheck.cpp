//===--- MinMaxUseInitializerListCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MinMaxUseInitializerListCheck.h"
#include "../utils/ASTUtils.h"
#include "../utils/LexerUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"

using namespace clang;

namespace {

struct FindArgsResult {
  const Expr *First;
  const Expr *Last;
  const Expr *Compare;
  std::vector<const clang::Expr *> Args;
};

} // anonymous namespace

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static FindArgsResult findArgs(const CallExpr *Call) {
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
      if (const auto *InitList = dyn_cast<clang::InitListExpr>(
              InitListExpr->getSubExpr()->IgnoreImplicit())) {
        Result.Args.insert(Result.Args.begin(), InitList->inits().begin(),
                           InitList->inits().end());

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

  for (const Expr *Arg : Call->arguments()) {
    if (!Result.First)
      Result.First = Arg;

    if (Arg == Result.Compare)
      continue;

    Result.Args.push_back(Arg);
    Result.Last = Arg;
  }

  return Result;
}

static std::vector<FixItHint>
generateReplacement(const MatchFinder::MatchResult &Match,
                    const CallExpr *TopCall, const FindArgsResult &Result) {
  std::vector<FixItHint> FixItHints;

  const QualType ResultType = TopCall->getDirectCallee()
                                  ->getReturnType()
                                  .getNonReferenceType()
                                  .getUnqualifiedType()
                                  .getCanonicalType();
  const bool IsInitializerList = Result.First == Result.Last;

  if (!IsInitializerList)
    FixItHints.push_back(
        FixItHint::CreateInsertion(Result.First->getBeginLoc(), "{"));

  for (const Expr *Arg : Result.Args) {
    if (const auto *InnerCall =
            dyn_cast<CallExpr>(Arg->IgnoreParenImpCasts())) {
      const FindArgsResult InnerResult = findArgs(InnerCall);
      const std::vector<FixItHint> InnerReplacements =
          generateReplacement(Match, InnerCall, InnerResult);
      if (InnerCall->getDirectCallee()->getQualifiedNameAsString() ==
              TopCall->getDirectCallee()->getQualifiedNameAsString() &&
          ((!Result.Compare && !InnerResult.Compare) ||
           utils::areStatementsIdentical(Result.Compare, InnerResult.Compare,
                                         *Match.Context))) {

        FixItHints.push_back(
            FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                InnerCall->getCallee()->getSourceRange())));

        const auto LParen = utils::lexer::findNextTokenSkippingComments(
            InnerCall->getCallee()->getEndLoc(), *Match.SourceManager,
            Match.Context->getLangOpts());
        if (LParen && LParen->getKind() == tok::l_paren)
          FixItHints.push_back(
              FixItHint::CreateRemoval(SourceRange(LParen->getLocation())));

        FixItHints.push_back(
            FixItHint::CreateRemoval(SourceRange(InnerCall->getRParenLoc())));

        if (InnerResult.First == InnerResult.Last) {
          FixItHints.insert(FixItHints.end(), InnerReplacements.begin(),
                            InnerReplacements.end());

          FixItHints.push_back(
              FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                  InnerResult.First->getBeginLoc())));
          FixItHints.push_back(FixItHint::CreateRemoval(
              CharSourceRange::getTokenRange(InnerResult.First->getEndLoc())));
        } else
          FixItHints.insert(FixItHints.end(), InnerReplacements.begin() + 1,
                            InnerReplacements.end() - 1);

        if (InnerResult.Compare) {
          const auto Comma = utils::lexer::findNextTokenSkippingComments(
              InnerResult.Last->getEndLoc(), *Match.SourceManager,
              Match.Context->getLangOpts());
          if (Comma && Comma->getKind() == tok::comma)
            FixItHints.push_back(
                FixItHint::CreateRemoval(SourceRange(Comma->getLocation())));

          if (utils::lexer::getPreviousToken(
                  InnerResult.Compare->getExprLoc(), *Match.SourceManager,
                  Match.Context->getLangOpts(), false)
                  .getLocation() == Comma->getLocation())
            FixItHints.push_back(
                FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                    Comma->getLocation(), InnerResult.Compare->getEndLoc())));
          else {
            FixItHints.push_back(
                FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                    InnerResult.Compare->getSourceRange())));
          }
        }
      }
      continue;
    }

    const QualType ArgType = Arg->IgnoreParenImpCasts()
                                 ->getType()
                                 .getUnqualifiedType()
                                 .getCanonicalType();

    if (ArgType != ResultType) {
      const std::string ArgText =
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(Arg->getSourceRange()),
              *Match.SourceManager, Match.Context->getLangOpts())
              .str();

      FixItHints.push_back(FixItHint::CreateReplacement(
          Arg->getSourceRange(),
          "static_cast<" + ResultType.getAsString() + ">(" + ArgText + ")"));
    }
  }

  if (!IsInitializerList) {
    if (Result.Compare)
      FixItHints.push_back(FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(Result.Last->getEndLoc(), 0,
                                     *Match.SourceManager,
                                     Match.Context->getLangOpts()),
          "}"));
    else
      FixItHints.push_back(
          FixItHint::CreateInsertion(TopCall->getEndLoc(), "}"));
  }

  return FixItHints;
}

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
  auto CreateMatcher = [](const StringRef FunctionName) {
    auto FuncDecl = functionDecl(hasName(FunctionName));
    auto Expression = callExpr(callee(FuncDecl));

    return callExpr(callee(FuncDecl),
                    anyOf(hasArgument(0, Expression),
                          hasArgument(1, Expression),
                          hasArgument(0, cxxStdInitializerListExpr())),
                    unless(hasParent(Expression)))
        .bind("topCall");
  };

  Finder->addMatcher(CreateMatcher("::std::max"), this);
  Finder->addMatcher(CreateMatcher("::std::min"), this);
}

void MinMaxUseInitializerListCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void MinMaxUseInitializerListCheck::check(
    const MatchFinder::MatchResult &Match) {

  const auto *TopCall = Match.Nodes.getNodeAs<CallExpr>("topCall");

  const FindArgsResult Result = findArgs(TopCall);
  const std::vector<FixItHint> Replacement =
      generateReplacement(Match, TopCall, Result);

  if (Replacement.size() <= 2) {
    return;
  }

  const DiagnosticBuilder Diagnostic =
      diag(TopCall->getBeginLoc(),
           "do not use nested 'std::%0' calls, use an initializer list instead")
      << TopCall->getDirectCallee()->getName()
      << Inserter.createIncludeInsertion(
             Match.SourceManager->getFileID(TopCall->getBeginLoc()),
             "<algorithm>");

  for (const auto &FixIt : Replacement) {
    Diagnostic << FixIt;
  }
}

} // namespace clang::tidy::modernize
