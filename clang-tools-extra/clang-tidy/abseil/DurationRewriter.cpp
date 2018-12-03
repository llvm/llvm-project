//===--- DurationRewriter.cpp - clang-tidy --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DurationRewriter.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

/// Returns an integer if the fractional part of a `FloatingLiteral` is `0`.
static llvm::Optional<llvm::APSInt>
truncateIfIntegral(const FloatingLiteral &FloatLiteral) {
  double Value = FloatLiteral.getValueAsApproximateDouble();
  if (std::fmod(Value, 1) == 0) {
    if (Value >= static_cast<double>(1u << 31))
      return llvm::None;

    return llvm::APSInt::get(static_cast<int64_t>(Value));
  }
  return llvm::None;
}

/// Returns the factory function name for a given `Scale`.
llvm::StringRef getFactoryForScale(DurationScale Scale) {
  switch (Scale) {
  case DurationScale::Hours:
    return "absl::Hours";
  case DurationScale::Minutes:
    return "absl::Minutes";
  case DurationScale::Seconds:
    return "absl::Seconds";
  case DurationScale::Milliseconds:
    return "absl::Milliseconds";
  case DurationScale::Microseconds:
    return "absl::Microseconds";
  case DurationScale::Nanoseconds:
    return "absl::Nanoseconds";
  }
  llvm_unreachable("unknown scaling factor");
}

/// Returns `true` if `Node` is a value which evaluates to a literal `0`.
bool IsLiteralZero(const MatchFinder::MatchResult &Result, const Expr &Node) {
  return selectFirst<const clang::Expr>(
             "val",
             match(expr(ignoringImpCasts(anyOf(integerLiteral(equals(0)),
                                               floatLiteral(equals(0.0)))))
                       .bind("val"),
                   Node, *Result.Context)) != nullptr;
}

llvm::Optional<std::string>
stripFloatCast(const ast_matchers::MatchFinder::MatchResult &Result,
               const Expr &Node) {
  if (const Expr *MaybeCastArg = selectFirst<const Expr>(
          "cast_arg",
          match(expr(anyOf(cxxStaticCastExpr(
                               hasDestinationType(realFloatingPointType()),
                               hasSourceExpression(expr().bind("cast_arg"))),
                           cStyleCastExpr(
                               hasDestinationType(realFloatingPointType()),
                               hasSourceExpression(expr().bind("cast_arg"))),
                           cxxFunctionalCastExpr(
                               hasDestinationType(realFloatingPointType()),
                               hasSourceExpression(expr().bind("cast_arg"))))),
                Node, *Result.Context)))
    return tooling::fixit::getText(*MaybeCastArg, *Result.Context).str();

  return llvm::None;
}

llvm::Optional<std::string>
stripFloatLiteralFraction(const MatchFinder::MatchResult &Result,
                          const Expr &Node) {
  if (const auto *LitFloat = llvm::dyn_cast<FloatingLiteral>(&Node))
    // Attempt to simplify a `Duration` factory call with a literal argument.
    if (llvm::Optional<llvm::APSInt> IntValue = truncateIfIntegral(*LitFloat))
      return IntValue->toString(/*radix=*/10);

  return llvm::None;
}

std::string simplifyDurationFactoryArg(const MatchFinder::MatchResult &Result,
                                       const Expr &Node) {
  // Check for an explicit cast to `float` or `double`.
  if (llvm::Optional<std::string> MaybeArg = stripFloatCast(Result, Node))
    return *MaybeArg;

  // Check for floats without fractional components.
  if (llvm::Optional<std::string> MaybeArg =
          stripFloatLiteralFraction(Result, Node))
    return *MaybeArg;

  // We couldn't simplify any further, so return the argument text.
  return tooling::fixit::getText(Node, *Result.Context).str();
}

} // namespace abseil
} // namespace tidy
} // namespace clang
