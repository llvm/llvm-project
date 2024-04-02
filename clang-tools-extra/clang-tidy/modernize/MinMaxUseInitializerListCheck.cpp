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

    if (InitList) {
      Result.Args.append(InitList->inits().begin(), InitList->inits().end());
      Result.First = *ArgIterator;
      Result.Last = *ArgIterator;

      // check if there is a comparison argument
      std::advance(ArgIterator, 1);
      if (ArgIterator != Call->arguments().end())
        Result.Compare = *ArgIterator;

      return Result;
    }
  } else {
    // if it has 3 arguments then the last will be the comparison
    Result.Compare = *(std::next(Call->arguments().begin(), 2));
  }

  if (Result.Compare)
    Result.Args = SmallVector<const Expr *>(llvm::drop_end(Call->arguments()));
  else
    Result.Args = SmallVector<const Expr *>(Call->arguments());

  Result.First = Result.Args.front();
  Result.Last = Result.Args.back();

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
  const SourceManager &SourceMngr = *Match.SourceManager;
  const LangOptions &LanguageOpts = Match.Context->getLangOpts();

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

    const FindArgsResult InnerResult = findArgs(InnerCall);

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
        FixItHint::CreateRemoval(InnerCall->getCallee()->getSourceRange()));

    // remove the parentheses
    const auto LParen = utils::lexer::findNextTokenSkippingComments(
        InnerCall->getCallee()->getEndLoc(), SourceMngr, LanguageOpts);
    FixItHints.push_back(
        FixItHint::CreateRemoval(SourceRange(LParen->getLocation())));
    FixItHints.push_back(
        FixItHint::CreateRemoval(SourceRange(InnerCall->getRParenLoc())));

    // if the inner call has an initializer list arg
    if (InnerResult.First == InnerResult.Last) {
      // remove the initializer list braces
      FixItHints.push_back(FixItHint::CreateRemoval(
          CharSourceRange::getTokenRange(InnerResult.First->getBeginLoc())));
      FixItHints.push_back(FixItHint::CreateRemoval(
          CharSourceRange::getTokenRange(InnerResult.First->getEndLoc())));
    }

    const SmallVector<FixItHint> InnerReplacements =
        generateReplacement(Match, InnerCall, InnerResult);

    FixItHints.insert(FixItHints.end(),
                      // ignore { and } insertions for the inner call if it does
                      // not have an initializer list arg
                      InnerReplacements.begin(), InnerReplacements.end());

    if (InnerResult.Compare) {
      // find the comma after the value arguments
      const auto Comma = utils::lexer::findNextTokenSkippingComments(
          InnerResult.Last->getEndLoc(), SourceMngr, LanguageOpts);

      // remove the comma and the comparison
      FixItHints.push_back(
          FixItHint::CreateRemoval(SourceRange(Comma->getLocation())));

      FixItHints.push_back(
          FixItHint::CreateRemoval(InnerResult.Compare->getSourceRange()));
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

  if (Replacement.empty())
    return;

  const DiagnosticBuilder Diagnostic =
      diag(TopCall->getBeginLoc(),
           "do not use nested 'std::%0' calls, use an initializer list instead")
      << TopCall->getDirectCallee()->getName()
      << Inserter.createIncludeInsertion(
             Match.SourceManager->getFileID(TopCall->getBeginLoc()),
             "<algorithm>");

  // if the top call doesn't have an initializer list argument
  if (Result.First != Result.Last) {
    // add { and } insertions
    Diagnostic << FixItHint::CreateInsertion(Result.First->getBeginLoc(), "{");

      Diagnostic << FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(Result.Last->getEndLoc(), 0,
                                     *Match.SourceManager,
                                     Match.Context->getLangOpts()),
          "}");
  }

  for (const auto &FixIt : Replacement)
    Diagnostic << FixIt;
}

} // namespace clang::tidy::modernize
