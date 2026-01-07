//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MultipleStatementMacroCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER(Expr, isInMacro) { return Node.getBeginLoc().isMacroID(); }

} // namespace

/// Find the next statement after `S`.
static const Stmt *nextStmt(const MatchFinder::MatchResult &Result,
                            const Stmt *S) {
  auto Parents = Result.Context->getParents(*S);
  if (Parents.empty())
    return nullptr;
  const auto *Parent = Parents[0].get<Stmt>();
  if (!Parent)
    return nullptr;
  const Stmt *Prev = nullptr;
  for (const Stmt *Child : Parent->children()) {
    if (Prev == S)
      return Child;
    Prev = Child;
  }
  return nextStmt(Result, Parent);
}

using ExpansionRanges = std::vector<SourceRange>;

/// \brief Get all the macro expansion ranges related to `Loc`.
///
/// The result is ordered from most inner to most outer.
static ExpansionRanges
getExpansionRanges(SourceLocation Loc, const MatchFinder::MatchResult &Result) {
  ExpansionRanges Locs;
  while (Loc.isMacroID()) {
    Locs.push_back(
        Result.SourceManager->getImmediateExpansionRange(Loc).getAsRange());
    Loc = Locs.back().getBegin();
  }
  return Locs;
}

void MultipleStatementMacroCheck::registerMatchers(MatchFinder *Finder) {
  const auto Inner = expr(isInMacro(), unless(compoundStmt())).bind("inner");
  Finder->addMatcher(
      stmt(anyOf(ifStmt(hasThen(Inner)), ifStmt(hasElse(Inner)).bind("else"),
                 whileStmt(hasBody(Inner)), forStmt(hasBody(Inner))))
          .bind("outer"),
      this);
}

void MultipleStatementMacroCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Inner = Result.Nodes.getNodeAs<Expr>("inner");
  const auto *Outer = Result.Nodes.getNodeAs<Stmt>("outer");
  const auto *Next = nextStmt(Result, Outer);
  if (!Next)
    return;

  SourceLocation OuterLoc = Outer->getBeginLoc();
  if (Result.Nodes.getNodeAs<Stmt>("else"))
    OuterLoc = cast<IfStmt>(Outer)->getElseLoc();

  auto InnerRanges = getExpansionRanges(Inner->getBeginLoc(), Result);
  auto OuterRanges = getExpansionRanges(OuterLoc, Result);
  auto NextRanges = getExpansionRanges(Next->getBeginLoc(), Result);

  // Remove all the common ranges, starting from the top (the last ones in the
  // list).
  while (!InnerRanges.empty() && !OuterRanges.empty() && !NextRanges.empty() &&
         InnerRanges.back() == OuterRanges.back() &&
         InnerRanges.back() == NextRanges.back()) {
    InnerRanges.pop_back();
    OuterRanges.pop_back();
    NextRanges.pop_back();
  }

  // Inner and Next must have at least one more macro that Outer doesn't have,
  // and that range must be common to both.
  if (InnerRanges.empty() || NextRanges.empty() ||
      InnerRanges.back() != NextRanges.back())
    return;

  diag(InnerRanges.back().getBegin(), "multiple statement macro used without "
                                      "braces; some statements will be "
                                      "unconditionally executed");
}

} // namespace clang::tidy::bugprone
