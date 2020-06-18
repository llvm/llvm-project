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

    // Matcher which matches the current scale's factory with a `1` argument,
    // e.g. `absl::Seconds(1)`.
    auto factory_matcher = ignoringElidableConstructorCall(
        callExpr(callee(functionDecl(hasName(DurationFactory))),
                 hasArgument(0, ignoringImpCasts(integerLiteral(equals(1))))));

    // Matcher which matches either inverse function and binds its argument,
    // e.g. `absl::ToDoubleSeconds(dur)`.
    auto inverse_function_matcher = callExpr(
        callee(functionDecl(hasAnyName(FloatConversion, IntegerConversion))),
        hasArgument(0, expr().bind("arg")));

    // Matcher which matches a duration divided by the factory_matcher above,
    // e.g. `dur / absl::Seconds(1)`.
    auto division_operator_matcher = cxxOperatorCallExpr(
        hasOverloadedOperatorName("/"), hasArgument(0, expr().bind("arg")),
        hasArgument(1, factory_matcher));

    // Matcher which matches a duration argument to `FDivDuration`,
    // e.g. `absl::FDivDuration(dur, absl::Seconds(1))`
    auto fdiv_matcher = callExpr(
        callee(functionDecl(hasName("::absl::FDivDuration"))),
        hasArgument(0, expr().bind("arg")), hasArgument(1, factory_matcher));

    // Matcher which matches a duration argument being scaled,
    // e.g. `absl::ToDoubleSeconds(dur) * 2`
    auto scalar_matcher = ignoringImpCasts(
        binaryOperator(hasOperatorName("*"),
                       hasEitherOperand(expr(ignoringParenImpCasts(
                           callExpr(callee(functionDecl(hasAnyName(
                                        FloatConversion, IntegerConversion))),
                                    hasArgument(0, expr().bind("arg")))
                               .bind("inner_call")))))
            .bind("binop"));

    Finder->addMatcher(
        callExpr(callee(functionDecl(hasName(DurationFactory))),
                 hasArgument(0, anyOf(inverse_function_matcher,
                                      division_operator_matcher, fdiv_matcher,
                                      scalar_matcher)))
            .bind("call"),
        this);
  }
}

void DurationUnnecessaryConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *OuterCall = Result.Nodes.getNodeAs<Expr>("call");

  if (isInMacro(Result, OuterCall))
    return;

  FixItHint Hint;
  if (const auto *Binop = Result.Nodes.getNodeAs<BinaryOperator>("binop")) {
    const auto *Arg = Result.Nodes.getNodeAs<Expr>("arg");
    const auto *InnerCall = Result.Nodes.getNodeAs<Expr>("inner_call");
    const Expr *LHS = Binop->getLHS();
    const Expr *RHS = Binop->getRHS();

    if (LHS->IgnoreParenImpCasts() == InnerCall) {
      Hint = FixItHint::CreateReplacement(
          OuterCall->getSourceRange(),
          (llvm::Twine(tooling::fixit::getText(*Arg, *Result.Context)) + " * " +
           tooling::fixit::getText(*RHS, *Result.Context))
              .str());
    } else {
      assert(RHS->IgnoreParenImpCasts() == InnerCall &&
             "Inner call should be find on the RHS");

      Hint = FixItHint::CreateReplacement(
          OuterCall->getSourceRange(),
          (llvm::Twine(tooling::fixit::getText(*LHS, *Result.Context)) + " * " +
           tooling::fixit::getText(*Arg, *Result.Context))
              .str());
    }
  } else if (const auto *Arg = Result.Nodes.getNodeAs<Expr>("arg")) {
    Hint = FixItHint::CreateReplacement(
        OuterCall->getSourceRange(),
        tooling::fixit::getText(*Arg, *Result.Context));
  }
  diag(OuterCall->getBeginLoc(),
       "remove unnecessary absl::Duration conversions")
      << Hint;
}

} // namespace abseil
} // namespace tidy
} // namespace clang
