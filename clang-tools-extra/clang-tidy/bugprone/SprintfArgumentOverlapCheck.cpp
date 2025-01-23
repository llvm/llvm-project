//===--- SprintfArgumentOverlapCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SprintfArgumentOverlapCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

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

AST_MATCHER_P(Stmt, identicalTo, std::string, ID) {
  return Builder->removeBindings(
      [this, &Node,
       &Finder](const ast_matchers::internal::BoundNodesMap &Nodes) {
        const DynTypedNode &BN = Nodes.getNode(ID);
        if (const auto *BoundStmt = BN.get<Stmt>())
          return !utils::areStatementsIdentical(&Node, BoundStmt,
                                                Finder->getASTContext());
        return true;
      });
}

AST_MATCHER(Expr, hasSideEffects) {
  return Node.HasSideEffects(Finder->getASTContext());
}

SprintfArgumentOverlapCheck::SprintfArgumentOverlapCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SprintfRegex(Options.get("SprintfFunction", "(::std)?::sn?printf")) {}

void SprintfArgumentOverlapCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(matchesName(SprintfRegex)).bind("decl")),
          hasArgument(0, expr(unless(hasSideEffects())).bind("firstArgExpr")),
          forEachArgument(expr(unless(equalsBoundNode("firstArgExpr")),
                               identicalTo("firstArgExpr"))
                              .bind("otherArgExpr")))
          .bind("call"),
      this);
}

void SprintfArgumentOverlapCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FirstArg = Result.Nodes.getNodeAs<Expr>("firstArgExpr");
  const auto *OtherArg = Result.Nodes.getNodeAs<Expr>("otherArgExpr");
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  const auto *FnDecl = Result.Nodes.getNodeAs<FunctionDecl>("decl");

  if (!FirstArg || !OtherArg || !Call || !FnDecl)
    return;

  std::optional<unsigned> ArgIndex;
  for (unsigned I = 0; I != Call->getNumArgs(); ++I) {
    if (Call->getArg(I)->IgnoreUnlessSpelledInSource() == OtherArg) {
      ArgIndex = I;
      break;
    }
  }
  if (!ArgIndex)
    return;

  diag(OtherArg->getBeginLoc(),
       "the %ordinal0 argument in %1 overlaps the 1st argument, "
       "which is undefined behavior")
      << (*ArgIndex + 1) << FnDecl << FirstArg->getSourceRange()
      << OtherArg->getSourceRange();
}

void SprintfArgumentOverlapCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SprintfRegex", SprintfRegex);
}

} // namespace clang::tidy::bugprone
