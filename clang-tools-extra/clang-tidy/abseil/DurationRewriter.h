//===--- DurationRewriter.h - clang-tidy ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONREWRITER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONREWRITER_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include <cinttypes>

namespace clang {
namespace tidy {
namespace abseil {

/// Duration factory and conversion scales
enum class DurationScale : std::uint8_t {
  Hours = 0,
  Minutes,
  Seconds,
  Milliseconds,
  Microseconds,
  Nanoseconds,
};

/// Given a `Scale`, return the appropriate factory function call for
/// constructing a `Duration` for that scale.
llvm::StringRef getFactoryForScale(DurationScale Scale);

// Determine if `Node` represents a literal floating point or integral zero.
bool IsLiteralZero(const ast_matchers::MatchFinder::MatchResult &Result,
                   const Expr &Node);

/// Possibly strip a floating point cast expression.
///
/// If `Node` represents an explicit cast to a floating point type, return
/// the textual context of the cast argument, otherwise `None`.
llvm::Optional<std::string>
stripFloatCast(const ast_matchers::MatchFinder::MatchResult &Result,
               const Expr &Node);

/// Possibly remove the fractional part of a floating point literal.
///
/// If `Node` represents a floating point literal with a zero fractional part,
/// return the textual context of the integral part, otherwise `None`.
llvm::Optional<std::string>
stripFloatLiteralFraction(const ast_matchers::MatchFinder::MatchResult &Result,
                          const Expr &Node);

/// Possibly further simplify a duration factory function's argument, without
/// changing the scale of the factory function. Return that simplification or
/// the text of the argument if no simplification is possible.
std::string
simplifyDurationFactoryArg(const ast_matchers::MatchFinder::MatchResult &Result,
                           const Expr &Node);

/// Given the name of an inverse Duration function (e.g., `ToDoubleSeconds`),
/// return its `DurationScale`, or `None` if a match is not found.
llvm::Optional<DurationScale> getScaleForInverse(llvm::StringRef Name);

/// Assuming `Node` has type `double` or `int` representing a time interval of
/// `Scale`, return the expression to make it a suitable `Duration`.
llvm::Optional<std::string> rewriteExprFromNumberToDuration(
    const ast_matchers::MatchFinder::MatchResult &Result, DurationScale Scale,
    const Expr *Node);

AST_MATCHER_FUNCTION(ast_matchers::internal::Matcher<FunctionDecl>,
                     DurationConversionFunction) {
  using namespace clang::ast_matchers;
  return functionDecl(
      hasAnyName("::absl::ToDoubleHours", "::absl::ToDoubleMinutes",
                 "::absl::ToDoubleSeconds", "::absl::ToDoubleMilliseconds",
                 "::absl::ToDoubleMicroseconds", "::absl::ToDoubleNanoseconds",
                 "::absl::ToInt64Hours", "::absl::ToInt64Minutes",
                 "::absl::ToInt64Seconds", "::absl::ToInt64Milliseconds",
                 "::absl::ToInt64Microseconds", "::absl::ToInt64Nanoseconds"));
}

AST_MATCHER_FUNCTION(ast_matchers::internal::Matcher<FunctionDecl>,
                     DurationFactoryFunction) {
  using namespace clang::ast_matchers;
  return functionDecl(hasAnyName("::absl::Nanoseconds", "::absl::Microseconds",
                                 "::absl::Milliseconds", "::absl::Seconds",
                                 "::absl::Minutes", "::absl::Hours"));
}

} // namespace abseil
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_DURATIONCOMPARISONCHECK_H
