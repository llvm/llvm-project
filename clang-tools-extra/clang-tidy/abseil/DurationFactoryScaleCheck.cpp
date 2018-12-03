//===--- DurationFactoryScaleCheck.cpp - clang-tidy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  static const std::unordered_map<std::string, DurationScale> ScaleMap(
      {{"Nanoseconds", DurationScale::Nanoseconds},
       {"Microseconds", DurationScale::Microseconds},
       {"Milliseconds", DurationScale::Milliseconds},
       {"Seconds", DurationScale::Seconds},
       {"Minutes", DurationScale::Minutes},
       {"Hours", DurationScale::Hours}});

  auto ScaleIter = ScaleMap.find(FactoryName);
  if (ScaleIter == ScaleMap.end())
    return llvm::None;

  return ScaleIter->second;
}

// Given either an integer or float literal, return its value.
// One and only one of `IntLit` and `FloatLit` should be provided.
static double GetValue(const IntegerLiteral *IntLit,
                       const FloatingLiteral *FloatLit) {
  if (IntLit)
    return IntLit->getValue().getLimitedValue();

  assert(FloatLit != nullptr && "Neither IntLit nor FloatLit set");
  return FloatLit->getValueAsApproximateDouble();
}

// Given the scale of a duration and a `Multiplier`, determine if `Multiplier`
// would produce a new scale.  If so, return a tuple containing the new scale
// and a suitable Multipler for that scale, otherwise `None`.
static llvm::Optional<std::tuple<DurationScale, double>>
GetNewScaleSingleStep(DurationScale OldScale, double Multiplier) {
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
static llvm::Optional<DurationScale> GetNewScale(DurationScale OldScale,
                                                 double Multiplier) {
  while (Multiplier != 1.0) {
    llvm::Optional<std::tuple<DurationScale, double>> result =
        GetNewScaleSingleStep(OldScale, Multiplier);
    if (!result)
      break;
    if (std::get<1>(*result) == 1.0)
      return std::get<0>(*result);
    Multiplier = std::get<1>(*result);
    OldScale = std::get<0>(*result);
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
      NewScale = GetNewScale(Scale, GetValue(IntLit, FloatLit));
      if (NewScale)
        Remainder = MultBinOp->getRHS();
    }

    // If we weren't able to scale based on the LHS, check the RHS
    if (!NewScale) {
      IntLit = llvm::dyn_cast<IntegerLiteral>(MultBinOp->getRHS());
      FloatLit = llvm::dyn_cast<FloatingLiteral>(MultBinOp->getRHS());
      if (IntLit || FloatLit) {
        NewScale = GetNewScale(Scale, GetValue(IntLit, FloatLit));
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
        GetNewScale(Scale, 1.0 / FloatLit->getValueAsApproximateDouble());
    if (NewScale) {
      const Expr *Remainder = DivBinOp->getLHS();

      // We've found an appropriate scaling factor and the new scale, so output
      // the relevant fix.
      diag(Call->getBeginLoc(), "internal duration scaling can be removed")
          << FixItHint::CreateReplacement(
                 Call->getSourceRange(),
                 (llvm::Twine(getFactoryForScale(*NewScale)) + "(" +
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
               (llvm::Twine(getFactoryForScale(*NewScale)) + "(" +
                tooling::fixit::getText(*Remainder, *Result.Context) + ")")
                   .str());
  }
  return;
}

} // namespace abseil
} // namespace tidy
} // namespace clang
