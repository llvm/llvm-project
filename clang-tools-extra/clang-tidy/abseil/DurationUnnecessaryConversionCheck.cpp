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

namespace clang::tidy::abseil {

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
    auto FactoryMatcher = ignoringElidableConstructorCall(
        callExpr(callee(functionDecl(hasName(DurationFactory))),
                 hasArgument(0, ignoringImpCasts(integerLiteral(equals(1))))));

    // Matcher which matches either inverse function and binds its argument,
    // e.g. `absl::ToDoubleSeconds(dur)`.
    auto InverseFunctionMatcher = callExpr(
        callee(functionDecl(hasAnyName(FloatConversion, IntegerConversion))),
        hasArgument(0, expr().bind("arg")));

    // Matcher which matches a duration divided by the factory_matcher above,
    // e.g. `dur / absl::Seconds(1)`.
    auto DivisionOperatorMatcher = cxxOperatorCallExpr(
        hasOverloadedOperatorName("/"), hasArgument(0, expr().bind("arg")),
        hasArgument(1, FactoryMatcher));

    // Matcher which matches a duration argument to `FDivDuration`,
    // e.g. `absl::FDivDuration(dur, absl::Seconds(1))`
    auto FdivMatcher = callExpr(
        callee(functionDecl(hasName("::absl::FDivDuration"))),
        hasArgument(0, expr().bind("arg")), hasArgument(1, FactoryMatcher));

    // Matcher which matches a duration argument being scaled,
    // e.g. `absl::ToDoubleSeconds(dur) * 2`
    auto ScalarMatcher = ignoringImpCasts(
        binaryOperator(hasOperatorName("*"),
                       hasEitherOperand(expr(ignoringParenImpCasts(
                           callExpr(callee(functionDecl(hasAnyName(
                                        FloatConversion, IntegerConversion))),
                                    hasArgument(0, expr().bind("arg")))
                               .bind("inner_call")))))
            .bind("binop"));

    Finder->addMatcher(
        callExpr(callee(functionDecl(hasName(DurationFactory))),
                 hasArgument(0, anyOf(InverseFunctionMatcher,
                                      DivisionOperatorMatcher, FdivMatcher,
                                      ScalarMatcher)))
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

} // namespace clang::tidy::abseil
