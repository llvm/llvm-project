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
          hasAnyArgument(callExpr(callee(functionDecl(hasName("::std::max"))))),
          unless(
              hasParent(callExpr(callee(functionDecl(hasName("::std::max")))))))
          .bind("maxCall"),
      this);

  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("::std::min"))),
          hasAnyArgument(callExpr(callee(functionDecl(hasName("::std::min"))))),
          unless(
              hasParent(callExpr(callee(functionDecl(hasName("::std::min")))))))
          .bind("minCall"),
      this);
}

void MinMaxUseInitializerListCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void MinMaxUseInitializerListCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MaxCall = Result.Nodes.getNodeAs<CallExpr>("maxCall");
  const auto *MinCall = Result.Nodes.getNodeAs<CallExpr>("minCall");

  const CallExpr *TopCall = MaxCall ? MaxCall : MinCall;
  if (!TopCall) {
    return;
  }
  const QualType ResultType =
      TopCall->getDirectCallee()->getReturnType().getNonReferenceType();

  const Expr *FirstArg = nullptr;
  const Expr *LastArg = nullptr;
  std::vector<const Expr *> Args;
  findArgs(TopCall, &FirstArg, &LastArg, Args);

  if (!FirstArg || !LastArg || Args.size() <= 2) {
    return;
  }

  std::string ReplacementText = "{";
  for (const Expr *Arg : Args) {
    QualType ArgType = Arg->getType();
    bool CastNeeded =
        ArgType.getCanonicalType() != ResultType.getCanonicalType();

    if (CastNeeded)
      ReplacementText += "static_cast<" + ResultType.getAsString() + ">(";

    ReplacementText += Lexer::getSourceText(
        CharSourceRange::getTokenRange(Arg->getSourceRange()),
        *Result.SourceManager, Result.Context->getLangOpts());

    if (CastNeeded)
      ReplacementText += ")";
    ReplacementText += ", ";
  }
  ReplacementText = ReplacementText.substr(0, ReplacementText.size() - 2) + "}";

  diag(TopCall->getBeginLoc(),
       "do not use nested std::%0 calls, use %1 instead")
      << TopCall->getDirectCallee()->getName() << ReplacementText
      << FixItHint::CreateReplacement(
             CharSourceRange::getTokenRange(
                 FirstArg->getBeginLoc(),
                 Lexer::getLocForEndOfToken(TopCall->getEndLoc(), 0,
                                            Result.Context->getSourceManager(),
                                            Result.Context->getLangOpts())
                     .getLocWithOffset(-2)),
             ReplacementText)
      << Inserter.createMainFileIncludeInsertion("<algorithm>");
}

void MinMaxUseInitializerListCheck::findArgs(const CallExpr *Call,
                                             const Expr **First,
                                             const Expr **Last,
                                             std::vector<const Expr *> &Args) {
  if (!Call) {
    return;
  }

  const FunctionDecl *Callee = Call->getDirectCallee();
  if (!Callee) {
    return;
  }

  for (const Expr *Arg : Call->arguments()) {
    if (!*First)
      *First = Arg;

    const CallExpr *InnerCall = dyn_cast<CallExpr>(Arg);
    if (InnerCall && InnerCall->getDirectCallee() &&
        InnerCall->getDirectCallee()->getNameAsString() ==
            Call->getDirectCallee()->getNameAsString()) {
      findArgs(InnerCall, First, Last, Args);
    } else
      Args.push_back(Arg);

    *Last = Arg;
  }
}

} // namespace clang::tidy::modernize
