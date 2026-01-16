//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsafeToAllowExceptionsCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {
namespace {

AST_MATCHER_P(FunctionDecl, isEnabled, llvm::StringSet<>,
              FunctionsThatShouldNotThrow) {
  return FunctionsThatShouldNotThrow.contains(Node.getNameAsString());
}

AST_MATCHER(FunctionDecl, isExplicitThrow) {
  return isExplicitThrowExceptionSpec(Node.getExceptionSpecType()) &&
         Node.getExceptionSpecSourceRange().isValid();
}

AST_MATCHER(FunctionDecl, hasAtLeastOneParameter) {
  return Node.getNumParams() > 0;
}

} // namespace

UnsafeToAllowExceptionsCheck::UnsafeToAllowExceptionsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawCheckedSwapFunctions(
          Options.get("CheckedSwapFunctions", "swap,iter_swap,iter_move")) {
  llvm::SmallVector<StringRef, 4> CheckedSwapFunctionsVec;
  RawCheckedSwapFunctions.split(CheckedSwapFunctionsVec, ",", -1, false);
  CheckedSwapFunctions.insert_range(CheckedSwapFunctionsVec);
}

void UnsafeToAllowExceptionsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "CheckedSwapFunctions", RawCheckedSwapFunctions);
}

void UnsafeToAllowExceptionsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(allOf(isDefinition(), isExplicitThrow(),
                         anyOf(cxxDestructorDecl(),
                               cxxConstructorDecl(isMoveConstructor()),
                               cxxMethodDecl(isMoveAssignmentOperator()),
                               allOf(isEnabled(CheckedSwapFunctions),
                                     hasAtLeastOneParameter()),
                               isMain())))
          .bind("f"),
      this);
}

void UnsafeToAllowExceptionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("f");

  if (!MatchedDecl)
    return;

  diag(MatchedDecl->getLocation(),
       "function %0 should not throw exceptions but "
       "it is still marked as throwable")
      << MatchedDecl;
}

} // namespace clang::tidy::bugprone
