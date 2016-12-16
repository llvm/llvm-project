//===--- MultipleStatementMacroCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MultipleStatementMacroCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {

AST_MATCHER(Expr, isInMacro) { return Node.getLocStart().isMacroID(); }

/// \brief Find the next statement after `S`.
const Stmt *nextStmt(const MatchFinder::MatchResult &Result, const Stmt *S) {
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

using ExpansionRanges = std::vector<std::pair<SourceLocation, SourceLocation>>;

/// \bried Get all the macro expansion ranges related to `Loc`.
///
/// The result is ordered from most inner to most outer.
ExpansionRanges getExpansionRanges(SourceLocation Loc,
                                   const MatchFinder::MatchResult &Result) {
  ExpansionRanges Locs;
  while (Loc.isMacroID()) {
    Locs.push_back(Result.SourceManager->getImmediateExpansionRange(Loc));
    Loc = Locs.back().first;
  }
  return Locs;
}

} // namespace

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

  SourceLocation OuterLoc = Outer->getLocStart();
  if (Result.Nodes.getNodeAs<Stmt>("else"))
    OuterLoc = cast<IfStmt>(Outer)->getElseLoc();

  auto InnerRanges = getExpansionRanges(Inner->getLocStart(), Result);
  auto OuterRanges = getExpansionRanges(OuterLoc, Result);
  auto NextRanges = getExpansionRanges(Next->getLocStart(), Result);

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

  diag(InnerRanges.back().first, "multiple statement macro used without "
                                 "braces; some statements will be "
                                 "unconditionally executed");
}

} // namespace misc
} // namespace tidy
} // namespace clang
