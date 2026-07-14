//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidReferenceCoroutineParametersCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

AvoidReferenceCoroutineParametersCheck::AvoidReferenceCoroutineParametersCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowedReturnTypes(utils::options::parseStringList(
          Options.get("AllowedReturnTypes", ""))) {}

void AvoidReferenceCoroutineParametersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowedReturnTypes",
                utils::options::serializeStringList(AllowedReturnTypes));
}

void AvoidReferenceCoroutineParametersCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(unless(parameterCountIs(0)), hasBody(coroutineBodyStmt()),
                   unless(returns(
                       matchers::matchesAnyListedTypeName(AllowedReturnTypes))))
          .bind("fnt"),
      this);
}

void AvoidReferenceCoroutineParametersCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("fnt");
  for (const ParmVarDecl *Param : Function->parameters()) {
    if (!Param->getType().getCanonicalType()->isReferenceType())
      continue;

    diag(Param->getBeginLoc(), "coroutine parameters should not be references");
  }
}

} // namespace clang::tidy::cppcoreguidelines
