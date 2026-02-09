//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeToAllowExceptionsCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {
namespace {

AST_MATCHER(FunctionDecl, isExplicitThrow) {
  return isExplicitThrowExceptionSpec(Node.getExceptionSpecType()) &&
         Node.getExceptionSpecSourceRange().isValid();
}

} // namespace

UnsafeToAllowExceptionsCheck::UnsafeToAllowExceptionsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      CheckedSwapFunctions(utils::options::parseStringList(
          Options.get("CheckedSwapFunctions", "swap;iter_swap;iter_move"))) {}

void UnsafeToAllowExceptionsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckedSwapFunctions",
                utils::options::serializeStringList(CheckedSwapFunctions));
}

void UnsafeToAllowExceptionsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isDefinition(), isExplicitThrow(),
                   anyOf(cxxDestructorDecl(),
                         cxxConstructorDecl(isMoveConstructor()),
                         cxxMethodDecl(isMoveAssignmentOperator()),
                         allOf(hasAnyName(CheckedSwapFunctions),
                               unless(parameterCountIs(0))),
                         isMain()))
          .bind("f"),
      this);
}

void UnsafeToAllowExceptionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("f");
  assert(MatchedDecl);

  diag(MatchedDecl->getLocation(),
       "function %0 should not throw exceptions but "
       "it is still marked as potentially throwing")
      << MatchedDecl;
}

} // namespace clang::tidy::bugprone
