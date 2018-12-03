//===--- DurationComparisonCheck.cpp - clang-tidy -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DurationComparisonCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

/// Given the name of an inverse Duration function (e.g., `ToDoubleSeconds`),
/// return its `DurationScale`, or `None` if a match is not found.
static llvm::Optional<DurationScale> getScaleForInverse(llvm::StringRef Name) {
  static const llvm::DenseMap<llvm::StringRef, DurationScale> ScaleMap(
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

/// Given a `Scale` return the inverse functions for it.
static const std::pair<llvm::StringRef, llvm::StringRef> &
getInverseForScale(DurationScale Scale) {
  static const std::unordered_map<DurationScale,
                                  std::pair<llvm::StringRef, llvm::StringRef>>
      InverseMap(
          {{DurationScale::Hours,
            std::make_pair("::absl::ToDoubleHours", "::absl::ToInt64Hours")},
           {DurationScale::Minutes, std::make_pair("::absl::ToDoubleMinutes",
                                                   "::absl::ToInt64Minutes")},
           {DurationScale::Seconds, std::make_pair("::absl::ToDoubleSeconds",
                                                   "::absl::ToInt64Seconds")},
           {DurationScale::Milliseconds,
            std::make_pair("::absl::ToDoubleMilliseconds",
                           "::absl::ToInt64Milliseconds")},
           {DurationScale::Microseconds,
            std::make_pair("::absl::ToDoubleMicroseconds",
                           "::absl::ToInt64Microseconds")},
           {DurationScale::Nanoseconds,
            std::make_pair("::absl::ToDoubleNanoseconds",
                           "::absl::ToInt64Nanoseconds")}});

  // We know our map contains all the Scale values, so we can skip the
  // nonexistence check.
  auto InverseIter = InverseMap.find(Scale);
  assert(InverseIter != InverseMap.end() && "Unexpected scale found");
  return InverseIter->second;
}

/// If `Node` is a call to the inverse of `Scale`, return that inverse's
/// argument, otherwise None.
static llvm::Optional<std::string>
maybeRewriteInverseDurationCall(const MatchFinder::MatchResult &Result,
                                DurationScale Scale, const Expr &Node) {
  const std::pair<std::string, std::string> &InverseFunctions =
      getInverseForScale(Scale);
  if (const Expr *MaybeCallArg = selectFirst<const Expr>(
          "e", match(callExpr(callee(functionDecl(
                                  hasAnyName(InverseFunctions.first.c_str(),
                                             InverseFunctions.second.c_str()))),
                              hasArgument(0, expr().bind("e"))),
                     Node, *Result.Context))) {
    return tooling::fixit::getText(*MaybeCallArg, *Result.Context).str();
  }

  return llvm::None;
}

/// Assuming `Node` has type `double` or `int` representing a time interval of
/// `Scale`, return the expression to make it a suitable `Duration`.
static llvm::Optional<std::string> rewriteExprFromNumberToDuration(
    const ast_matchers::MatchFinder::MatchResult &Result, DurationScale Scale,
    const Expr *Node) {
  const Expr &RootNode = *Node->IgnoreParenImpCasts();

  if (RootNode.getBeginLoc().isMacroID())
    return llvm::None;

  // First check to see if we can undo a complimentary function call.
  if (llvm::Optional<std::string> MaybeRewrite =
          maybeRewriteInverseDurationCall(Result, Scale, RootNode))
    return *MaybeRewrite;

  if (IsLiteralZero(Result, RootNode))
    return std::string("absl::ZeroDuration()");

  return (llvm::Twine(getFactoryForScale(Scale)) + "(" +
          simplifyDurationFactoryArg(Result, RootNode) + ")")
      .str();
}

void DurationComparisonCheck::registerMatchers(MatchFinder *Finder) {
  auto Matcher =
      binaryOperator(anyOf(hasOperatorName(">"), hasOperatorName(">="),
                           hasOperatorName("=="), hasOperatorName("<="),
                           hasOperatorName("<")),
                     hasEitherOperand(ignoringImpCasts(callExpr(
                         callee(functionDecl(DurationConversionFunction())
                                    .bind("function_decl"))))))
          .bind("binop");

  Finder->addMatcher(Matcher, this);
}

void DurationComparisonCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Binop = Result.Nodes.getNodeAs<BinaryOperator>("binop");

  // Don't try to replace things inside of macro definitions.
  if (Binop->getExprLoc().isMacroID())
    return;

  llvm::Optional<DurationScale> Scale = getScaleForInverse(
      Result.Nodes.getNodeAs<FunctionDecl>("function_decl")->getName());
  if (!Scale)
    return;

  // In most cases, we'll only need to rewrite one of the sides, but we also
  // want to handle the case of rewriting both sides. This is much simpler if
  // we unconditionally try and rewrite both, and let the rewriter determine
  // if nothing needs to be done.
  llvm::Optional<std::string> LhsReplacement =
      rewriteExprFromNumberToDuration(Result, *Scale, Binop->getLHS());
  llvm::Optional<std::string> RhsReplacement =
      rewriteExprFromNumberToDuration(Result, *Scale, Binop->getRHS());

  if (!(LhsReplacement && RhsReplacement))
    return;

  diag(Binop->getBeginLoc(), "perform comparison in the duration domain")
      << FixItHint::CreateReplacement(Binop->getSourceRange(),
                                      (llvm::Twine(*LhsReplacement) + " " +
                                       Binop->getOpcodeStr() + " " +
                                       *RhsReplacement)
                                          .str());
}

} // namespace abseil
} // namespace tidy
} // namespace clang
