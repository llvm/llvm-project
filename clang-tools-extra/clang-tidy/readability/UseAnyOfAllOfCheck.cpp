//===--- UseAnyOfAllOfCheck.cpp - clang-tidy-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseAnyOfAllOfCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Frontend/CompilerInstance.h"

using namespace clang::ast_matchers;

namespace clang {
namespace {
/// Matches a Stmt whose parent is a CompoundStmt, and which is directly
/// followed by a Stmt matching the inner matcher.
AST_MATCHER_P(Stmt, nextStmt, ast_matchers::internal::Matcher<Stmt>,
              InnerMatcher) {
  DynTypedNodeList Parents = Finder->getASTContext().getParents(Node);
  if (Parents.size() != 1)
    return false;

  auto *C = Parents[0].get<CompoundStmt>();
  if (!C)
    return false;

  const auto *I = llvm::find(C->body(), &Node);
  assert(I != C->body_end() && "C is parent of Node");
  if (++I == C->body_end())
    return false; // Node is last statement.

  return InnerMatcher.matches(**I, Finder, Builder);
}
} // namespace

namespace tidy {
namespace readability {

void UseAnyOfAllOfCheck::registerMatchers(MatchFinder *Finder) {
  auto Returns = [](bool V) {
    return returnStmt(hasReturnValue(cxxBoolLiteral(equals(V))));
  };

  auto ReturnsButNotTrue =
      returnStmt(hasReturnValue(unless(cxxBoolLiteral(equals(true)))));
  auto ReturnsButNotFalse =
      returnStmt(hasReturnValue(unless(cxxBoolLiteral(equals(false)))));

  Finder->addMatcher(
      cxxForRangeStmt(
          nextStmt(Returns(false).bind("final_return")),
          hasBody(allOf(hasDescendant(Returns(true)),
                        unless(anyOf(hasDescendant(breakStmt()),
                                     hasDescendant(gotoStmt()),
                                     hasDescendant(ReturnsButNotTrue))))))
          .bind("any_of_loop"),
      this);

  Finder->addMatcher(
      cxxForRangeStmt(
          nextStmt(Returns(true).bind("final_return")),
          hasBody(allOf(hasDescendant(Returns(false)),
                        unless(anyOf(hasDescendant(breakStmt()),
                                     hasDescendant(gotoStmt()),
                                     hasDescendant(ReturnsButNotFalse))))))
          .bind("all_of_loop"),
      this);
}

static bool isViableLoop(const CXXForRangeStmt &S, ASTContext &Context) {

  ExprMutationAnalyzer Mutations(*S.getBody(), Context);
  if (Mutations.isMutated(S.getLoopVariable()))
    return false;
  const auto Matches =
      match(findAll(declRefExpr().bind("decl_ref")), *S.getBody(), Context);

  return llvm::none_of(Matches, [&Mutations](auto &DeclRef) {
    // TODO: allow modifications of loop-local variables
    return Mutations.isMutated(
        DeclRef.template getNodeAs<DeclRefExpr>("decl_ref")->getDecl());
  });
}

void UseAnyOfAllOfCheck::check(const MatchFinder::MatchResult &Result) {

  if (const auto *S = Result.Nodes.getNodeAs<CXXForRangeStmt>("any_of_loop")) {
    if (!isViableLoop(*S, *Result.Context))
      return;

    diag(S->getForLoc(), "replace loop by 'std%select{|::ranges}0::any_of()'")
        << getLangOpts().CPlusPlus20;
  } else if (const auto *S =
                 Result.Nodes.getNodeAs<CXXForRangeStmt>("all_of_loop")) {
    if (!isViableLoop(*S, *Result.Context))
      return;

    diag(S->getForLoc(), "replace loop by 'std%select{|::ranges}0::all_of()'")
        << getLangOpts().CPlusPlus20;
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
