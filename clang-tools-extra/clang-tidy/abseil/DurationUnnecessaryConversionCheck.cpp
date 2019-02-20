//===--- DurationUnnecessaryConversionCheck.cpp - clang-tidy
//-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DurationUnnecessaryConversionCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

void DurationUnnecessaryConversionCheck::registerMatchers(MatchFinder *Finder) {
  for (const auto &Scale : {"Hours", "Minutes", "Seconds", "Milliseconds",
                            "Microseconds", "Nanoseconds"}) {
    std::string DurationFactory = (llvm::Twine("::absl::") + Scale).str();
    std::string FloatConversion =
        (llvm::Twine("::absl::ToDouble") + Scale).str();
    std::string IntegerConversion =
        (llvm::Twine("::absl::ToInt64") + Scale).str();

    Finder->addMatcher(
        callExpr(
            callee(functionDecl(hasName(DurationFactory))),
            hasArgument(0, callExpr(callee(functionDecl(hasAnyName(
                                        FloatConversion, IntegerConversion))),
                                    hasArgument(0, expr().bind("arg")))))
            .bind("call"),
        this);
  }
}

void DurationUnnecessaryConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *OuterCall = Result.Nodes.getNodeAs<Expr>("call");
  const auto *Arg = Result.Nodes.getNodeAs<Expr>("arg");

  if (!isNotInMacro(Result, OuterCall))
    return;

  diag(OuterCall->getBeginLoc(), "remove unnecessary absl::Duration conversions")
      << FixItHint::CreateReplacement(
             OuterCall->getSourceRange(),
             tooling::fixit::getText(*Arg, *Result.Context));
}

} // namespace abseil
} // namespace tidy
} // namespace clang
