//===--- UseAnyOfAllOfCheck.cpp - clang-tidy-------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  auto returns = [](bool V) {
    return returnStmt(hasReturnValue(cxxBoolLiteral(equals(V))));
  };

  auto returnsButNotTrue =
      returnStmt(hasReturnValue(unless(cxxBoolLiteral(equals(true)))));
  auto returnsButNotFalse =
      returnStmt(hasReturnValue(unless(cxxBoolLiteral(equals(false)))));

  Finder->addMatcher(
      cxxForRangeStmt(
          nextStmt(returns(false).bind("final_return")),
          hasBody(allOf(hasDescendant(returns(true)),
                        unless(anyOf(hasDescendant(breakStmt()),
                                     hasDescendant(gotoStmt()),
                                     hasDescendant(returnsButNotTrue))))))
          .bind("any_of_loop"),
      this);

  Finder->addMatcher(
      cxxForRangeStmt(
          nextStmt(returns(true).bind("final_return")),
          hasBody(allOf(hasDescendant(returns(false)),
                        unless(anyOf(hasDescendant(breakStmt()),
                                     hasDescendant(gotoStmt()),
                                     hasDescendant(returnsButNotFalse))))))
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
  StringRef Ranges = getLangOpts().CPlusPlus20 ? "::ranges" : "";

  if (const auto *S = Result.Nodes.getNodeAs<CXXForRangeStmt>("any_of_loop")) {
    if (!isViableLoop(*S, *Result.Context))
      return;

    diag(S->getForLoc(), "replace loop by 'std%0::any_of()'") << Ranges;
  } else if (const auto *S =
                 Result.Nodes.getNodeAs<CXXForRangeStmt>("all_of_loop")) {
    if (!isViableLoop(*S, *Result.Context))
      return;

    diag(S->getForLoc(), "replace loop by 'std%0::all_of()'") << Ranges;
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
