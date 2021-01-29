//===--- DurationFactoryScaleCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DurationFactoryScaleCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

// Given the name of a duration factory function, return the appropriate
// `DurationScale` for that factory.  If no factory can be found for
// `FactoryName`, return `None`.
static llvm::Optional<DurationScale>
getScaleForFactory(llvm::StringRef FactoryName) {
  return llvm::StringSwitch<llvm::Optional<DurationScale>>(FactoryName)
      .Case("Nanoseconds", DurationScale::Nanoseconds)
      .Case("Microseconds", DurationScale::Microseconds)
      .Case("Milliseconds", DurationScale::Milliseconds)
      .Case("Seconds", DurationScale::Seconds)
      .Case("Minutes", DurationScale::Minutes)
      .Case("Hours", DurationScale::Hours)
      .Default(llvm::None);
}

// Given either an integer or float literal, return its value.
// One and only one of `IntLit` and `FloatLit` should be provided.
static double getValue(const IntegerLiteral *IntLit,
                       const FloatingLiteral *FloatLit) {
  if (IntLit)
    return IntLit->getValue().getLimitedValue();

  assert(FloatLit != nullptr && "Neither IntLit nor FloatLit set");
  return FloatLit->getValueAsApproximateDouble();
}

// Given the scale of a duration and a `Multiplier`, determine if `Multiplier`
// would produce a new scale.  If so, return a tuple containing the new scale
// and a suitable Multiplier for that scale, otherwise `None`.
static llvm::Optional<std::tuple<DurationScale, double>>
getNewScaleSingleStep(DurationScale OldScale, double Multiplier) {
  switch (OldScale) {
  case DurationScale::Hours:
    if (Multiplier <= 1.0 / 60.0)
      return std::make_tuple(DurationScale::Minutes, Multiplier * 60.0);
    break;

  case DurationScale::Minutes:
    if (Multiplier >= 60.0)
      return std::make_tuple(DurationScale::Hours, Multiplier / 60.0);
    if (Multiplier <= 1.0 / 60.0)
      return std::make_tuple(DurationScale::Seconds, Multiplier * 60.0);
    break;

  case DurationScale::Seconds:
    if (Multiplier >= 60.0)
      return std::make_tuple(DurationScale::Minutes, Multiplier / 60.0);
    if (Multiplier <= 1e-3)
      return std::make_tuple(DurationScale::Milliseconds, Multiplier * 1e3);
    break;

  case DurationScale::Milliseconds:
    if (Multiplier >= 1e3)
      return std::make_tuple(DurationScale::Seconds, Multiplier / 1e3);
    if (Multiplier <= 1e-3)
      return std::make_tuple(DurationScale::Microseconds, Multiplier * 1e3);
    break;

  case DurationScale::Microseconds:
    if (Multiplier >= 1e3)
      return std::make_tuple(DurationScale::Milliseconds, Multiplier / 1e3);
    if (Multiplier <= 1e-3)
      return std::make_tuple(DurationScale::Nanoseconds, Multiplier * 1e-3);
    break;

  case DurationScale::Nanoseconds:
    if (Multiplier >= 1e3)
      return std::make_tuple(DurationScale::Microseconds, Multiplier / 1e3);
    break;
  }

  return llvm::None;
}

// Given the scale of a duration and a `Multiplier`, determine if `Multiplier`
// would produce a new scale.  If so, return it, otherwise `None`.
static llvm::Optional<DurationScale> getNewScale(DurationScale OldScale,
                                                 double Multiplier) {
  while (Multiplier != 1.0) {
    llvm::Optional<std::tuple<DurationScale, double>> Result =
        getNewScaleSingleStep(OldScale, Multiplier);
    if (!Result)
      break;
    if (std::get<1>(*Result) == 1.0)
      return std::get<0>(*Result);
    Multiplier = std::get<1>(*Result);
    OldScale = std::get<0>(*Result);
  }

  return llvm::None;
}

void DurationFactoryScaleCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(DurationFactoryFunction()).bind("call_decl")),
          hasArgument(
              0,
              ignoringImpCasts(anyOf(
                  cxxFunctionalCastExpr(
                      hasDestinationType(
                          anyOf(isInteger(), realFloatingPointType())),
                      hasSourceExpression(initListExpr())),
                  integerLiteral(equals(0)), floatLiteral(equals(0.0)),
                  binaryOperator(hasOperatorName("*"),
                                 hasEitherOperand(ignoringImpCasts(
                                     anyOf(integerLiteral(), floatLiteral()))))
                      .bind("mult_binop"),
                  binaryOperator(hasOperatorName("/"), hasRHS(floatLiteral()))
                      .bind("div_binop")))))
          .bind("call"),
      this);
}

void DurationFactoryScaleCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");

  // Don't try to replace things inside of macro definitions.
  if (Call->getExprLoc().isMacroID())
    return;

  const Expr *Arg = Call->getArg(0)->IgnoreParenImpCasts();
  // Arguments which are macros are ignored.
  if (Arg->getBeginLoc().isMacroID())
    return;

  // We first handle the cases of literal zero (both float and integer).
  if (IsLiteralZero(Result, *Arg)) {
    diag(Call->getBeginLoc(),
         "use ZeroDuration() for zero-length time intervals")
        << FixItHint::CreateReplacement(Call->getSourceRange(),
                                        "absl::ZeroDuration()");
    return;
  }

  const auto *CallDecl = Result.Nodes.getNodeAs<FunctionDecl>("call_decl");
  llvm::Optional<DurationScale> MaybeScale =
      getScaleForFactory(CallDecl->getName());
  if (!MaybeScale)
    return;

  DurationScale Scale = *MaybeScale;
  const Expr *Remainder;
  llvm::Optional<DurationScale> NewScale;

  // We next handle the cases of multiplication and division.
  if (const auto *MultBinOp =
          Result.Nodes.getNodeAs<BinaryOperator>("mult_binop")) {
    // For multiplication, we need to look at both operands, and consider the
    // cases where a user is multiplying by something such as 1e-3.

    // First check the LHS
    const auto *IntLit = llvm::dyn_cast<IntegerLiteral>(MultBinOp->getLHS());
    const auto *FloatLit = llvm::dyn_cast<FloatingLiteral>(MultBinOp->getLHS());
    if (IntLit || FloatLit) {
      NewScale = getNewScale(Scale, getValue(IntLit, FloatLit));
      if (NewScale)
        Remainder = MultBinOp->getRHS();
    }

    // If we weren't able to scale based on the LHS, check the RHS
    if (!NewScale) {
      IntLit = llvm::dyn_cast<IntegerLiteral>(MultBinOp->getRHS());
      FloatLit = llvm::dyn_cast<FloatingLiteral>(MultBinOp->getRHS());
      if (IntLit || FloatLit) {
        NewScale = getNewScale(Scale, getValue(IntLit, FloatLit));
        if (NewScale)
          Remainder = MultBinOp->getLHS();
      }
    }
  } else if (const auto *DivBinOp =
                 Result.Nodes.getNodeAs<BinaryOperator>("div_binop")) {
    // We next handle division.
    // For division, we only check the RHS.
    const auto *FloatLit = llvm::dyn_cast<FloatingLiteral>(DivBinOp->getRHS());

    llvm::Optional<DurationScale> NewScale =
        getNewScale(Scale, 1.0 / FloatLit->getValueAsApproximateDouble());
    if (NewScale) {
      const Expr *Remainder = DivBinOp->getLHS();

      // We've found an appropriate scaling factor and the new scale, so output
      // the relevant fix.
      diag(Call->getBeginLoc(), "internal duration scaling can be removed")
          << FixItHint::CreateReplacement(
                 Call->getSourceRange(),
                 (llvm::Twine(getDurationFactoryForScale(*NewScale)) + "(" +
                  tooling::fixit::getText(*Remainder, *Result.Context) + ")")
                     .str());
    }
  }

  if (NewScale) {
    assert(Remainder && "No remainder found");
    // We've found an appropriate scaling factor and the new scale, so output
    // the relevant fix.
    diag(Call->getBeginLoc(), "internal duration scaling can be removed")
        << FixItHint::CreateReplacement(
               Call->getSourceRange(),
               (llvm::Twine(getDurationFactoryForScale(*NewScale)) + "(" +
                tooling::fixit::getText(*Remainder, *Result.Context) + ")")
                   .str());
  }
  return;
}

} // namespace abseil
} // namespace tidy
} // namespace clang
