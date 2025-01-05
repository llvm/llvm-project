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
        if (const auto *Lit = BN.get<IntegerLiteral>()) {
          return Lit->getValue() != Node.getValue();
        }
        return true;
      });
}

UndefinedSprintfOverlapCheck::UndefinedSprintfOverlapCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SprintfRegex(Options.get("SprintfFunction", "(::std)?::(sn?printf)")) {}

void UndefinedSprintfOverlapCheck::registerMatchers(MatchFinder *Finder) {
  auto FirstArg = declRefExpr(to(varDecl().bind("firstArgDecl")));
  auto OtherRefToArg = declRefExpr(to(varDecl(equalsBoundNode("firstArgDecl"))))
                           .bind("overlappingArg");
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(matchesName(SprintfRegex)).bind("decl")),
          allOf(hasArgument(
                    0, expr(anyOf(FirstArg,
                                  arraySubscriptExpr(
                                      hasBase(FirstArg),
                                      hasIndex(integerLiteral().bind("index"))),
                                  memberExpr(member(decl().bind("member")),
                                             hasObjectExpression(FirstArg))))
                           .bind("firstArgExpr")),
                hasAnyArgument(expr(
                    unless(equalsBoundNode("firstArgExpr")),
                    anyOf(OtherRefToArg,
                          arraySubscriptExpr(hasBase(OtherRefToArg),
                                             hasIndex(integerLiteral(
                                                 hasSameValueAs("index")))),
                          memberExpr(member(decl(equalsBoundNode("member"))),
                                     hasObjectExpression(OtherRefToArg))))))),
      this);
}

void UndefinedSprintfOverlapCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *OverlappingArg =
      Result.Nodes.getNodeAs<DeclRefExpr>("overlappingArg");
  const auto *FirstArg = Result.Nodes.getNodeAs<Expr>("firstArgExpr");
  const auto *FnDecl = Result.Nodes.getNodeAs<FunctionDecl>("decl");

  const llvm::StringRef FirstArgText =
      Lexer::getSourceText(CharSourceRange::getTokenRange(
                               FirstArg->getBeginLoc(), FirstArg->getEndLoc()),
                           *Result.SourceManager, getLangOpts());

  diag(OverlappingArg->getLocation(),
       "argument '%0' overlaps the first argument "
       "in %1, which is undefined behavior")
      << FirstArgText << FnDecl;
}

void UndefinedSprintfOverlapCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SprintfRegex", SprintfRegex);
}

} // namespace clang::tidy::bugprone
