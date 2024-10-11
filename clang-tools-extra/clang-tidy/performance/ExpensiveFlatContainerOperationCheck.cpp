//===--- ExpensiveFlatContainerOperationCheck.cpp - clang-tidy ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpensiveFlatContainerOperationCheck.h"

#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {
// TODO: folly::heap_vector_map?
const auto DefaultFlatContainers =
    "::std::flat_map; ::std::flat_multimap;"
    "::std::flat_set; ::std::flat_multiset;"
    "::boost::container::flat_map; ::boost::container::flat_multimap;"
    "::boost::container::flat_set; ::boost::container::flat_multiset;"
    "::folly::sorted_vector_map; ::folly::sorted_vector_set;";
} // namespace

ExpensiveFlatContainerOperationCheck::ExpensiveFlatContainerOperationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WarnOutsideLoops(Options.get("WarnOutsideLoops", false)),
      FlatContainers(utils::options::parseStringList(
          Options.get("FlatContainers", DefaultFlatContainers))) {}

void ExpensiveFlatContainerOperationCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto OnSoughtFlatContainer =
      callee(cxxMethodDecl(ofClass(cxxRecordDecl(hasAnyName(FlatContainers)))));

  // Any emplace-style or insert_or_assign call is a single-element operation.
  const auto HasEmplaceOrInsertorAssignCall = callee(cxxMethodDecl(hasAnyName(
      "emplace", "emplace_hint", "try_emplace", "insert_or_assign")));

  // Erase calls with a single argument are single-element operations.
  const auto HasEraseCallWithOneArgument = cxxMemberCallExpr(
      argumentCountIs(1), callee(cxxMethodDecl(hasName("erase"))));

  // TODO: insert.

  const auto SoughtFlatContainerOperation =
      cxxMemberCallExpr(
          OnSoughtFlatContainer,
          anyOf(HasEmplaceOrInsertorAssignCall, HasEraseCallWithOneArgument))
          .bind("call");

  if (WarnOutsideLoops) {
    Finder->addMatcher(SoughtFlatContainerOperation, this);
    return;
  }

  Finder->addMatcher(
      mapAnyOf(whileStmt, forStmt, cxxForRangeStmt, doStmt)
          .with(stmt(
              stmt().bind("loop"),
              forEachDescendant(cxxMemberCallExpr(
                  SoughtFlatContainerOperation,
                  // Common false positive: variable is declared directly within
                  // the loop. Note that this won't catch cases where the
                  // container is a member of a class declared in the loop.
                  // More robust lifetime analysis would be required to catch
                  // those cases, but this should filter out the most common
                  // false positives.
                  unless(onImplicitObjectArgument(declRefExpr(hasDeclaration(
                      decl(hasAncestor(stmt(equalsBoundNode("loop")))))))))))),
      this);
}

void ExpensiveFlatContainerOperationCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");

  diag(Call->getExprLoc(),
       "Single element operations are expensive for flat containers. "
       "Consider using available bulk operations instead, aggregating values "
       "beforehand if needed.");
}

void ExpensiveFlatContainerOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WarnOutsideLoops", WarnOutsideLoops);
  Options.store(Opts, "FlatContainers",
                utils::options::serializeStringList(FlatContainers));
}

bool ExpensiveFlatContainerOperationCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}

std::optional<TraversalKind>
ExpensiveFlatContainerOperationCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}
} // namespace clang::tidy::performance
