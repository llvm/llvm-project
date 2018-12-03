//===--- DurationFactoryFloatCheck.cpp - clang-tidy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DurationFactoryFloatCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

// Returns an integer if the fractional part of a `FloatingLiteral` is `0`.
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

// Returns `true` if `Range` is inside a macro definition.
static bool InsideMacroDefinition(const MatchFinder::MatchResult &Result,
                                  SourceRange Range) {
  return !clang::Lexer::makeFileCharRange(
              clang::CharSourceRange::getCharRange(Range),
              *Result.SourceManager, Result.Context->getLangOpts())
              .isValid();
}

void DurationFactoryFloatCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasAnyName(
              "absl::Nanoseconds", "absl::Microseconds", "absl::Milliseconds",
              "absl::Seconds", "absl::Minutes", "absl::Hours"))),
          hasArgument(0,
                      anyOf(cxxStaticCastExpr(
                                hasDestinationType(realFloatingPointType()),
                                hasSourceExpression(expr().bind("cast_arg"))),
                            cStyleCastExpr(
                                hasDestinationType(realFloatingPointType()),
                                hasSourceExpression(expr().bind("cast_arg"))),
                            cxxFunctionalCastExpr(
                                hasDestinationType(realFloatingPointType()),
                                hasSourceExpression(expr().bind("cast_arg"))),
                            floatLiteral().bind("float_literal"))))
          .bind("call"),
      this);
}

void DurationFactoryFloatCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedCall = Result.Nodes.getNodeAs<CallExpr>("call");

  // Don't try and replace things inside of macro definitions.
  if (InsideMacroDefinition(Result, MatchedCall->getSourceRange()))
    return;

  const Expr *Arg = MatchedCall->getArg(0)->IgnoreImpCasts();
  // Arguments which are macros are ignored.
  if (Arg->getBeginLoc().isMacroID())
    return;

  // Check for casts to `float` or `double`.
  if (const auto *MaybeCastArg = Result.Nodes.getNodeAs<Expr>("cast_arg")) {
    diag(MatchedCall->getBeginLoc(),
         (llvm::Twine("use the integer version of absl::") +
          MatchedCall->getDirectCallee()->getName())
             .str())
        << FixItHint::CreateReplacement(
               Arg->getSourceRange(),
               tooling::fixit::getText(*MaybeCastArg, *Result.Context));
    return;
  }

  // Check for floats without fractional components.
  if (const auto *LitFloat =
          Result.Nodes.getNodeAs<FloatingLiteral>("float_literal")) {
    // Attempt to simplify a `Duration` factory call with a literal argument.
    if (llvm::Optional<llvm::APSInt> IntValue = truncateIfIntegral(*LitFloat)) {
      diag(MatchedCall->getBeginLoc(),
           (llvm::Twine("use the integer version of absl::") +
            MatchedCall->getDirectCallee()->getName())
               .str())
          << FixItHint::CreateReplacement(LitFloat->getSourceRange(),
                                          IntValue->toString(/*radix=*/10));
      return;
    }
  }
}

} // namespace abseil
} // namespace tidy
} // namespace clang
