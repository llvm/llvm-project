//===--- UndefinedSprintfOverlapCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UndefinedSprintfOverlapCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

AST_MATCHER_P(IntegerLiteral, hasSameValueAs, std::string, ID) {
  return Builder->removeBindings(
      [this, &Node](const ast_matchers::internal::BoundNodesMap &Nodes) {
        const DynTypedNode &BN = Nodes.getNode(ID);
        if (const auto *Lit = BN.get<IntegerLiteral>())
          return Lit->getValue() != Node.getValue();
        return true;
      });
}

// Similar to forEachArgumentWithParam. forEachArgumentWithParam does not work
// with variadic functions like sprintf, since there is no `decl()` to match
// against in the parameter list `...`.
AST_MATCHER_P(CallExpr, forEachArgument, ast_matchers::internal::Matcher<Expr>,
              ArgMatcher) {
  using namespace clang::ast_matchers::internal;
  BoundNodesTreeBuilder Result;
  int ParamIndex = 0;
  bool Matched = false;
  for (unsigned ArgIndex = 0; ArgIndex < Node.getNumArgs(); ++ArgIndex) {
    BoundNodesTreeBuilder ArgMatches(*Builder);
    if (ArgMatcher.matches(*(Node.getArg(ArgIndex)->IgnoreParenCasts()), Finder,
                           &ArgMatches)) {
      BoundNodesTreeBuilder ParamMatches(ArgMatches);
      Result.addMatch(ArgMatches);
      Matched = true;
    }
    ++ParamIndex;
  }
  *Builder = std::move(Result);
  return Matched;
}

UndefinedSprintfOverlapCheck::UndefinedSprintfOverlapCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SprintfRegex(Options.get("SprintfFunction", "(::std)?::sn?printf")) {}

void UndefinedSprintfOverlapCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(matchesName(SprintfRegex)).bind("decl")),
               hasArgument(0, expr().bind("firstArgExpr")),
               forEachArgument(expr(unless(equalsBoundNode("firstArgExpr")))
                                   .bind("secondArgExpr"))),
      this);
}

void UndefinedSprintfOverlapCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FirstArg = Result.Nodes.getNodeAs<Expr>("firstArgExpr");
  const auto *SecondArg = Result.Nodes.getNodeAs<Expr>("secondArgExpr");
  const auto *FnDecl = Result.Nodes.getNodeAs<FunctionDecl>("decl");

  clang::ASTContext &Context = *Result.Context;

  if (!FirstArg || !SecondArg)
    return;

  if (!utils::areStatementsIdentical(FirstArg, SecondArg, Context))
    return;
  if (FirstArg->HasSideEffects(Context) || SecondArg->HasSideEffects(Context))
    return;

  const llvm::StringRef FirstArgText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(FirstArg->getSourceRange()),
      *Result.SourceManager, getLangOpts());

  diag(SecondArg->getBeginLoc(), "argument '%0' overlaps the first argument in "
                                 "%1, which is undefined behavior")
      << FirstArgText << FnDecl << FirstArg->getSourceRange()
      << SecondArg->getSourceRange();
}

void UndefinedSprintfOverlapCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SprintfRegex", SprintfRegex);
}

} // namespace clang::tidy::bugprone
