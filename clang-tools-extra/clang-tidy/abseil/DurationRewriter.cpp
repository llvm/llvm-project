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
#include "llvm/ADT/IndexedMap.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

struct DurationScale2IndexFunctor {
  using argument_type = DurationScale;
  unsigned operator()(DurationScale Scale) const {
    return static_cast<unsigned>(Scale);
  }
};

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

/// Given a `Scale` return the inverse functions for it.
static const std::pair<llvm::StringRef, llvm::StringRef> &
getInverseForScale(DurationScale Scale) {
  static const llvm::IndexedMap<std::pair<llvm::StringRef, llvm::StringRef>,
                                DurationScale2IndexFunctor>
      InverseMap = []() {
        // TODO: Revisit the immediately invoked lamba technique when
        // IndexedMap gets an initializer list constructor.
        llvm::IndexedMap<std::pair<llvm::StringRef, llvm::StringRef>,
                         DurationScale2IndexFunctor>
            InverseMap;
        InverseMap.resize(6);
        InverseMap[DurationScale::Hours] =
            std::make_pair("::absl::ToDoubleHours", "::absl::ToInt64Hours");
        InverseMap[DurationScale::Minutes] =
            std::make_pair("::absl::ToDoubleMinutes", "::absl::ToInt64Minutes");
        InverseMap[DurationScale::Seconds] =
            std::make_pair("::absl::ToDoubleSeconds", "::absl::ToInt64Seconds");
        InverseMap[DurationScale::Milliseconds] = std::make_pair(
            "::absl::ToDoubleMilliseconds", "::absl::ToInt64Milliseconds");
        InverseMap[DurationScale::Microseconds] = std::make_pair(
            "::absl::ToDoubleMicroseconds", "::absl::ToInt64Microseconds");
        InverseMap[DurationScale::Nanoseconds] = std::make_pair(
            "::absl::ToDoubleNanoseconds", "::absl::ToInt64Nanoseconds");
        return InverseMap;
      }();

  return InverseMap[Scale];
}

/// If `Node` is a call to the inverse of `Scale`, return that inverse's
/// argument, otherwise None.
static llvm::Optional<std::string>
rewriteInverseDurationCall(const MatchFinder::MatchResult &Result,
                           DurationScale Scale, const Expr &Node) {
  const std::pair<llvm::StringRef, llvm::StringRef> &InverseFunctions =
      getInverseForScale(Scale);
  if (const auto *MaybeCallArg = selectFirst<const Expr>(
          "e",
          match(callExpr(callee(functionDecl(hasAnyName(
                             InverseFunctions.first, InverseFunctions.second))),
                         hasArgument(0, expr().bind("e"))),
                Node, *Result.Context))) {
    return tooling::fixit::getText(*MaybeCallArg, *Result.Context).str();
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

llvm::Optional<DurationScale> getScaleForInverse(llvm::StringRef Name) {
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

  auto ScaleIter = ScaleMap.find(std::string(Name));
  if (ScaleIter == ScaleMap.end())
    return llvm::None;

  return ScaleIter->second;
}

std::string rewriteExprFromNumberToDuration(
    const ast_matchers::MatchFinder::MatchResult &Result, DurationScale Scale,
    const Expr *Node) {
  const Expr &RootNode = *Node->IgnoreParenImpCasts();

  // First check to see if we can undo a complimentary function call.
  if (llvm::Optional<std::string> MaybeRewrite =
          rewriteInverseDurationCall(Result, Scale, RootNode))
    return *MaybeRewrite;

  if (IsLiteralZero(Result, RootNode))
    return std::string("absl::ZeroDuration()");

  return (llvm::Twine(getFactoryForScale(Scale)) + "(" +
          simplifyDurationFactoryArg(Result, RootNode) + ")")
      .str();
}

} // namespace abseil
} // namespace tidy
} // namespace clang
