//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeprecatedPosixFunctionsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang::ast_matchers;
using namespace llvm;

namespace clang::tidy::portability {

static constexpr StringRef DeprecatedFunctionId = "DeprecatedFunctions";
static constexpr StringRef DeclRefId = "DRE";

static StringRef getReplacementFor(StringRef FunctionName) {
  // TODO: Suggest Annex K replacements when available.
  return StringSwitch<StringRef>(FunctionName)
      .Case("bcmp", "memcmp")
      .Case("bcopy", "memmove")
      .Case("bzero", "memset")
      .Case("getpw", "getpwuid")
      .Case("vfork", "posix_spawn")
      .Default({});
}

void DeprecatedPosixFunctionsCheck::registerMatchers(MatchFinder *Finder) {
  const auto FunctionNamesMatcher =
      hasAnyName("::bcmp", "::bcopy", "::bzero", "::getpw", "::vfork");
  Finder->addMatcher(
      declRefExpr(
          to(functionDecl(FunctionNamesMatcher).bind(DeprecatedFunctionId)))
          .bind(DeclRefId),
      this);
}

void DeprecatedPosixFunctionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *SourceExpr = Result.Nodes.getNodeAs<DeclRefExpr>(DeclRefId);
  const auto *FuncDecl =
      Result.Nodes.getNodeAs<FunctionDecl>(DeprecatedFunctionId);
  if (!SourceExpr || !FuncDecl)
    return;

  const StringRef FunctionName = FuncDecl->getName();
  diag(SourceExpr->getBeginLoc(),
       "function '%0' is deprecated; '%1' should be used instead")
      << FunctionName << getReplacementFor(FunctionName);
}

} // namespace clang::tidy::portability
