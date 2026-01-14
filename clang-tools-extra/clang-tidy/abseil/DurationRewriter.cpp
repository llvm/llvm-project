//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cmath>
#include <optional>

#include "DurationRewriter.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::abseil {

/// Returns an integer if the fractional part of a `FloatingLiteral` is `0`.
static std::optional<llvm::APSInt>
truncateIfIntegral(const FloatingLiteral &FloatLiteral) {
  const double Value = FloatLiteral.getValueAsApproximateDouble();
  if (std::fmod(Value, 1) == 0) {
    if (Value >= static_cast<double>(1U << 31))
      return std::nullopt;

    return llvm::APSInt::get(static_cast<int64_t>(Value));
  }
  return std::nullopt;
}

const std::pair<llvm::StringRef, llvm::StringRef> &
getDurationInverseForScale(DurationScale Scale) {
  static constexpr std::array<std::pair<llvm::StringRef, llvm::StringRef>, 6>
      InverseMap = {{
          {"::absl::ToDoubleHours", "::absl::ToInt64Hours"},
          {"::absl::ToDoubleMinutes", "::absl::ToInt64Minutes"},
          {"::absl::ToDoubleSeconds", "::absl::ToInt64Seconds"},
          {"::absl::ToDoubleMilliseconds", "::absl::ToInt64Milliseconds"},
          {"::absl::ToDoubleMicroseconds", "::absl::ToInt64Microseconds"},
          {"::absl::ToDoubleNanoseconds", "::absl::ToInt64Nanoseconds"},
      }};

  return InverseMap[llvm::to_underlying(Scale)];
}

/// If `Node` is a call to the inverse of `Scale`, return that inverse's
/// argument, otherwise std::nullopt.
static std::optional<std::string>
rewriteInverseDurationCall(const MatchFinder::MatchResult &Result,
                           DurationScale Scale, const Expr &Node) {
  const std::pair<llvm::StringRef, llvm::StringRef> &InverseFunctions =
      getDurationInverseForScale(Scale);
  if (const auto *MaybeCallArg = selectFirst<const Expr>(
          "e",
          match(callExpr(callee(functionDecl(hasAnyName(
                             InverseFunctions.first, InverseFunctions.second))),
                         hasArgument(0, expr().bind("e"))),
                Node, *Result.Context))) {
    return tooling::fixit::getText(*MaybeCallArg, *Result.Context).str();
  }

  return std::nullopt;
}

/// If `Node` is a call to the inverse of `Scale`, return that inverse's
/// argument, otherwise std::nullopt.
static std::optional<std::string>
rewriteInverseTimeCall(const MatchFinder::MatchResult &Result,
                       DurationScale Scale, const Expr &Node) {
  const llvm::StringRef InverseFunction = getTimeInverseForScale(Scale);
  if (const auto *MaybeCallArg = selectFirst<const Expr>(
          "e", match(callExpr(callee(functionDecl(hasName(InverseFunction))),
                              hasArgument(0, expr().bind("e"))),
                     Node, *Result.Context))) {
    return tooling::fixit::getText(*MaybeCallArg, *Result.Context).str();
  }

  return std::nullopt;
}

/// Returns the factory function name for a given `Scale`.
llvm::StringRef getDurationFactoryForScale(DurationScale Scale) {
  static constexpr std::array<llvm::StringRef, 6> FactoryMap = {
      "absl::Hours",        "absl::Minutes",      "absl::Seconds",
      "absl::Milliseconds", "absl::Microseconds", "absl::Nanoseconds",
  };

  return FactoryMap[llvm::to_underlying(Scale)];
}

llvm::StringRef getTimeFactoryForScale(DurationScale Scale) {
  static constexpr std::array<llvm::StringRef, 6> FactoryMap = {
      "absl::FromUnixHours",  "absl::FromUnixMinutes", "absl::FromUnixSeconds",
      "absl::FromUnixMillis", "absl::FromUnixMicros",  "absl::FromUnixNanos",
  };

  return FactoryMap[llvm::to_underlying(Scale)];
}

/// Returns the Time factory function name for a given `Scale`.
llvm::StringRef getTimeInverseForScale(DurationScale Scale) {
  static constexpr std::array<llvm::StringRef, 6> InverseMap = {
      "absl::ToUnixHours",  "absl::ToUnixMinutes", "absl::ToUnixSeconds",
      "absl::ToUnixMillis", "absl::ToUnixMicros",  "absl::ToUnixNanos",
  };

  return InverseMap[llvm::to_underlying(Scale)];
}

/// Returns `true` if `Node` is a value which evaluates to a literal `0`.
bool isLiteralZero(const MatchFinder::MatchResult &Result, const Expr &Node) {
  auto ZeroMatcher =
      anyOf(integerLiteral(equals(0)), floatLiteral(equals(0.0)));

  // Check to see if we're using a zero directly.
  if (selectFirst<const clang::Expr>(
          "val", match(expr(ignoringImpCasts(ZeroMatcher)).bind("val"), Node,
                       *Result.Context)) != nullptr)
    return true;

  // Now check to see if we're using a functional cast with a scalar
  // initializer expression, e.g. `int{0}`.
  if (selectFirst<const clang::Expr>(
          "val", match(cxxFunctionalCastExpr(
                           hasDestinationType(
                               anyOf(isInteger(), realFloatingPointType())),
                           hasSourceExpression(initListExpr(
                               hasInit(0, ignoringParenImpCasts(ZeroMatcher)))))
                           .bind("val"),
                       Node, *Result.Context)) != nullptr)
    return true;

  return false;
}

std::optional<std::string>
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

  return std::nullopt;
}

std::optional<std::string>
stripFloatLiteralFraction(const MatchFinder::MatchResult &Result,
                          const Expr &Node) {
  if (const auto *LitFloat = llvm::dyn_cast<FloatingLiteral>(&Node))
    // Attempt to simplify a `Duration` factory call with a literal argument.
    if (std::optional<llvm::APSInt> IntValue = truncateIfIntegral(*LitFloat))
      return toString(*IntValue, /*radix=*/10);

  return std::nullopt;
}

std::string simplifyDurationFactoryArg(const MatchFinder::MatchResult &Result,
                                       const Expr &Node) {
  // Check for an explicit cast to `float` or `double`.
  if (std::optional<std::string> MaybeArg = stripFloatCast(Result, Node))
    return *MaybeArg;

  // Check for floats without fractional components.
  if (std::optional<std::string> MaybeArg =
          stripFloatLiteralFraction(Result, Node))
    return *MaybeArg;

  // We couldn't simplify any further, so return the argument text.
  return tooling::fixit::getText(Node, *Result.Context).str();
}

std::optional<DurationScale> getScaleForDurationInverse(llvm::StringRef Name) {
  static const llvm::StringMap<DurationScale> ScaleMap(
      {{"ToDoubleHours", DurationScale::Hours},
       {"ToInt64Hours", DurationScale::Hours},
       {"ToDoubleMinutes", DurationScale::Minutes},
       {"ToInt64Minutes", DurationScale::Minutes},
       {"ToDoubleSeconds", DurationScale::Seconds},
       {"ToInt64Seconds", DurationScale::Seconds},
       {"ToDoubleMilliseconds", DurationScale::Milliseconds},
       {"ToInt64Milliseconds", DurationScale::Milliseconds},
       {"ToDoubleMicroseconds", DurationScale::Microseconds},
       {"ToInt64Microseconds", DurationScale::Microseconds},
       {"ToDoubleNanoseconds", DurationScale::Nanoseconds},
       {"ToInt64Nanoseconds", DurationScale::Nanoseconds}});

  auto ScaleIter = ScaleMap.find(Name);
  if (ScaleIter == ScaleMap.end())
    return std::nullopt;

  return ScaleIter->second;
}

std::optional<DurationScale> getScaleForTimeInverse(llvm::StringRef Name) {
  static const llvm::StringMap<DurationScale> ScaleMap(
      {{"ToUnixHours", DurationScale::Hours},
       {"ToUnixMinutes", DurationScale::Minutes},
       {"ToUnixSeconds", DurationScale::Seconds},
       {"ToUnixMillis", DurationScale::Milliseconds},
       {"ToUnixMicros", DurationScale::Microseconds},
       {"ToUnixNanos", DurationScale::Nanoseconds}});

  auto ScaleIter = ScaleMap.find(Name);
  if (ScaleIter == ScaleMap.end())
    return std::nullopt;

  return ScaleIter->second;
}

std::string rewriteExprFromNumberToDuration(
    const ast_matchers::MatchFinder::MatchResult &Result, DurationScale Scale,
    const Expr *Node) {
  const Expr &RootNode = *Node->IgnoreParenImpCasts();

  // First check to see if we can undo a complementary function call.
  if (std::optional<std::string> MaybeRewrite =
          rewriteInverseDurationCall(Result, Scale, RootNode))
    return *MaybeRewrite;

  if (isLiteralZero(Result, RootNode))
    return {"absl::ZeroDuration()"};

  return (llvm::Twine(getDurationFactoryForScale(Scale)) + "(" +
          simplifyDurationFactoryArg(Result, RootNode) + ")")
      .str();
}

std::string rewriteExprFromNumberToTime(
    const ast_matchers::MatchFinder::MatchResult &Result, DurationScale Scale,
    const Expr *Node) {
  const Expr &RootNode = *Node->IgnoreParenImpCasts();

  // First check to see if we can undo a complementary function call.
  if (std::optional<std::string> MaybeRewrite =
          rewriteInverseTimeCall(Result, Scale, RootNode))
    return *MaybeRewrite;

  if (isLiteralZero(Result, RootNode))
    return {"absl::UnixEpoch()"};

  return (llvm::Twine(getTimeFactoryForScale(Scale)) + "(" +
          tooling::fixit::getText(RootNode, *Result.Context) + ")")
      .str();
}

bool isInMacro(const MatchFinder::MatchResult &Result, const Expr *E) {
  if (!E->getBeginLoc().isMacroID())
    return false;

  SourceLocation Loc = E->getBeginLoc();
  // We want to get closer towards the initial macro typed into the source only
  // if the location is being expanded as a macro argument.
  while (Result.SourceManager->isMacroArgExpansion(Loc)) {
    // We are calling getImmediateMacroCallerLoc, but note it is essentially
    // equivalent to calling getImmediateSpellingLoc in this context according
    // to Clang implementation. We are not calling getImmediateSpellingLoc
    // because Clang comment says it "should not generally be used by clients."
    Loc = Result.SourceManager->getImmediateMacroCallerLoc(Loc);
  }
  return Loc.isMacroID();
}

} // namespace clang::tidy::abseil
