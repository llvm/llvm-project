//===--- UndefinedSprintfOverlapCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UndefinedSprintfOverlapCheck.h"
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

UndefinedSprintfOverlapCheck::UndefinedSprintfOverlapCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SprintfRegex(Options.get("SprintfFunction", "(::std)?::sn?printf")) {}

void UndefinedSprintfOverlapCheck::registerMatchers(MatchFinder *Finder) {
  auto FirstArg = declRefExpr(to(varDecl().bind("firstArgDecl")));
  auto OtherRefToArg =
      declRefExpr(to(varDecl(equalsBoundNode("firstArgDecl"))));
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(matchesName(SprintfRegex)).bind("decl")),
          hasArgument(0,
                      expr(anyOf(FirstArg,
                                 arraySubscriptExpr(
                                     hasBase(FirstArg),
                                     hasIndex(integerLiteral().bind("index"))),
                                 memberExpr(member(decl().bind("member")),
                                            hasObjectExpression(FirstArg))))
                          .bind("firstArgExpr")),
          hasAnyArgument(
              expr(unless(equalsBoundNode("firstArgExpr")),
                   anyOf(OtherRefToArg,
                         arraySubscriptExpr(
                             hasBase(OtherRefToArg),
                             hasIndex(integerLiteral(hasSameValueAs("index")))),
                         memberExpr(member(decl(equalsBoundNode("member"))),
                                    hasObjectExpression(OtherRefToArg))))
                  .bind("secondArgExpr"))),
      this);
}

void UndefinedSprintfOverlapCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FirstArg = Result.Nodes.getNodeAs<Expr>("firstArgExpr");
  const auto *SecondArg = Result.Nodes.getNodeAs<Expr>("secondArgExpr");
  const auto *FnDecl = Result.Nodes.getNodeAs<FunctionDecl>("decl");

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
