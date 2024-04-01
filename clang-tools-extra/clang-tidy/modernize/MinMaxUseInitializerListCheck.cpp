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
  SmallVector<const clang::Expr *, 2> Args;
};

} // anonymous namespace

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static FindArgsResult findArgs(const CallExpr *Call) {
  FindArgsResult Result;
  Result.First = nullptr;
  Result.Last = nullptr;
  Result.Compare = nullptr;

  //   check if the function has initializer list argument
  if (Call->getNumArgs() < 3) {
    auto ArgIterator = Call->arguments().begin();

    const auto *InitListExpr =
        dyn_cast<CXXStdInitializerListExpr>(*ArgIterator);
    const auto *InitList =
        InitListExpr != nullptr
            ? dyn_cast<clang::InitListExpr>(
                  InitListExpr->getSubExpr()->IgnoreImplicit())
            : nullptr;

    if (InitListExpr && InitList) {
      Result.Args.insert(Result.Args.begin(), InitList->inits().begin(),
                         InitList->inits().end());
      Result.First = *ArgIterator;
      Result.Last = *ArgIterator;

      // check if there is a comparison argument
      std::advance(ArgIterator, 1);
      if (ArgIterator != Call->arguments().end()) {
        Result.Compare = *ArgIterator;
      }

      return Result;
    }
    // if it has 3 arguments then the last will be the comparison
  } else {
    Result.Compare = *(std::next(Call->arguments().begin(), 2));
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

static SmallVector<FixItHint>
generateReplacement(const MatchFinder::MatchResult &Match,
                    const CallExpr *TopCall, const FindArgsResult &Result) {
  SmallVector<FixItHint> FixItHints;

  const QualType ResultType = TopCall->getDirectCallee()
                                  ->getReturnType()
                                  .getNonReferenceType()
                                  .getUnqualifiedType()
                                  .getCanonicalType();
  const auto &SourceMngr = *Match.SourceManager;
  const auto LanguageOpts = Match.Context->getLangOpts();
  const bool IsInitializerList = Result.First == Result.Last;

  // add { and } if the top call doesn't have an initializer list arg
  if (!IsInitializerList) {
    FixItHints.push_back(
        FixItHint::CreateInsertion(Result.First->getBeginLoc(), "{"));

    if (Result.Compare)
      FixItHints.push_back(FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(Result.Last->getEndLoc(), 0, SourceMngr,
                                     LanguageOpts),
          "}"));
    else
      FixItHints.push_back(
          FixItHint::CreateInsertion(TopCall->getEndLoc(), "}"));
  }

  for (const Expr *Arg : Result.Args) {
    const auto *InnerCall = dyn_cast<CallExpr>(Arg->IgnoreParenImpCasts());

    // If the argument is not a nested call
    if (!InnerCall) {
      // check if typecast is required
      const QualType ArgType = Arg->IgnoreParenImpCasts()
                                   ->getType()
                                   .getUnqualifiedType()
                                   .getCanonicalType();

      if (ArgType == ResultType)
        continue;

      const StringRef ArgText = Lexer::getSourceText(
          CharSourceRange::getTokenRange(Arg->getSourceRange()), SourceMngr,
          LanguageOpts);

      Twine Replacement = llvm::Twine("static_cast<")
                              .concat(ResultType.getAsString(LanguageOpts))
                              .concat(">(")
                              .concat(ArgText)
                              .concat(")");

      FixItHints.push_back(FixItHint::CreateReplacement(Arg->getSourceRange(),
                                                        Replacement.str()));

      continue;
    }

    const auto InnerResult = findArgs(InnerCall);
    const auto InnerReplacements =
        generateReplacement(Match, InnerCall, InnerResult);
    const bool IsInnerInitializerList = InnerResult.First == InnerResult.Last;

    // if the nested call doesn't have arguments skip it
    if (!InnerResult.First || !InnerResult.Last)
      continue;

    // if the nested call is not the same as the top call
    if (InnerCall->getDirectCallee()->getQualifiedNameAsString() !=
        TopCall->getDirectCallee()->getQualifiedNameAsString())
      continue;

    // if the nested call doesn't have the same compare function
    if ((Result.Compare || InnerResult.Compare) &&
        !utils::areStatementsIdentical(Result.Compare, InnerResult.Compare,
                                       *Match.Context))
      continue;

    // remove the function call
    FixItHints.push_back(
        FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
            InnerCall->getCallee()->getSourceRange())));

    // remove the parentheses
    const auto LParen = utils::lexer::findNextTokenSkippingComments(
        InnerCall->getCallee()->getEndLoc(), SourceMngr, LanguageOpts);
    FixItHints.push_back(
        FixItHint::CreateRemoval(SourceRange(LParen->getLocation())));
    FixItHints.push_back(
        FixItHint::CreateRemoval(SourceRange(InnerCall->getRParenLoc())));

    // if the inner call has an initializer list arg
    if (IsInnerInitializerList) {
      // remove the initializer list braces
      FixItHints.push_back(FixItHint::CreateRemoval(
          CharSourceRange::getTokenRange(InnerResult.First->getBeginLoc())));
      FixItHints.push_back(FixItHint::CreateRemoval(
          CharSourceRange::getTokenRange(InnerResult.First->getEndLoc())));
    }

    FixItHints.insert(FixItHints.end(),
                      // ignore { and } insertions for the inner call if it does
                      // not have an initializer list arg
                      InnerReplacements.begin() + (!IsInnerInitializerList) * 2,
                      InnerReplacements.end());

    if (InnerResult.Compare) {
      // find the comma after the value arguments
      const auto Comma = utils::lexer::findNextTokenSkippingComments(
          InnerResult.Last->getEndLoc(), SourceMngr, LanguageOpts);

      // if there are comments between the comma and the comparison
      if (utils::lexer::getPreviousToken(InnerResult.Compare->getExprLoc(),
                                         SourceMngr, LanguageOpts, false)
              .getLocation() != Comma->getLocation()) {
        // remove the comma and the comparison
        FixItHints.push_back(
            FixItHint::CreateRemoval(SourceRange(Comma->getLocation())));

        FixItHints.push_back(
            FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                InnerResult.Compare->getSourceRange())));
      } else
        // remove everything after the last argument
        FixItHints.push_back(
            FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                Comma->getLocation(), InnerResult.Compare->getEndLoc())));
    }
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
  const SmallVector<FixItHint> Replacement =
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
