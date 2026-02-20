//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MissingEndComparisonCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

constexpr llvm::StringRef IteratorAlgorithms[] = {
    "::std::find",          "::std::find_if",
    "::std::find_if_not",   "::std::search",
    "::std::search_n",      "::std::find_end",
    "::std::find_first_of", "::std::lower_bound",
    "::std::upper_bound",   "::std::partition_point",
    "::std::min_element",   "::std::max_element",
    "::std::adjacent_find", "::std::is_sorted_until"};

constexpr llvm::StringRef RangeAlgorithms[] = {
    "::std::ranges::find",        "::std::ranges::find_if",
    "::std::ranges::find_if_not", "::std::ranges::lower_bound",
    "::std::ranges::upper_bound", "::std::ranges::min_element",
    "::std::ranges::max_element"};

} // namespace

void MissingEndComparisonCheck::registerMatchers(MatchFinder *Finder) {
  const auto StdAlgoCall = callExpr(
      callee(functionDecl(hasAnyName(IteratorAlgorithms), isInStdNamespace())));

  const auto RangesCall = cxxOperatorCallExpr(
      hasOverloadedOperatorName("()"),
      hasArgument(0, declRefExpr(to(
                         varDecl(hasAnyName(RangeAlgorithms)).bind("cpo")))));

  const auto AnyAlgoCall =
      getLangOpts().CPlusPlus20
          ? expr(anyOf(StdAlgoCall, RangesCall)).bind("algoCall")
          : expr(StdAlgoCall).bind("algoCall");

  // Captures implicit pointer-to-bool casts and operator bool() calls.
  const auto IsBoolUsage = anyOf(
      implicitCastExpr(hasCastKind(CK_PointerToBoolean),
                       hasSourceExpression(ignoringParenImpCasts(AnyAlgoCall))),
      cxxMemberCallExpr(callee(cxxConversionDecl(returns(booleanType()))),
                        on(ignoringParenImpCasts(AnyAlgoCall))));

  // Captures variable usage: `auto it = std::find(...); if (it)`
  // FIXME: This only handles variables initialized directly by the algorithm.
  // We may need to introduce more accurate dataflow analysis in the future.
  const auto VarWithAlgoInit =
      varDecl(hasInitializer(ignoringParenImpCasts(AnyAlgoCall)));

  const auto IsVariableBoolUsage =
      anyOf(implicitCastExpr(hasCastKind(CK_PointerToBoolean),
                             hasSourceExpression(ignoringParenImpCasts(
                                 declRefExpr(to(VarWithAlgoInit))))),
            cxxMemberCallExpr(
                callee(cxxConversionDecl(returns(booleanType()))),
                on(ignoringParenImpCasts(declRefExpr(to(VarWithAlgoInit))))));

  Finder->addMatcher(
      expr(anyOf(IsBoolUsage, IsVariableBoolUsage)).bind("boolOp"), this);
}

void MissingEndComparisonCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("algoCall");
  const auto *BoolOp = Result.Nodes.getNodeAs<Expr>("boolOp");
  const auto *CPO = Result.Nodes.getNodeAs<VarDecl>("cpo");

  if (!Call || !BoolOp)
    return;

  std::string EndExprText;

  if (!CPO) {
    if (Call->getNumArgs() < 2)
      return;

    const Expr *EndArg = Call->getArg(1);
    // Filters nullptr, we assume the intent might be a valid check against null
    if (EndArg->IgnoreParenCasts()->isNullPointerConstant(
            *Result.Context, Expr::NPC_ValueDependentIsNull))
      return;

    EndExprText = tooling::fixit::getText(*EndArg, *Result.Context).str();
  } else {
    const FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee || Callee->getNumParams() == 0)
      return;

    // Range overloads take a reference (R&&), Iterator overloads pass by value.
    const bool IsIterPair =
        !Callee->getParamDecl(0)->getType()->isReferenceType();

    if (IsIterPair) {
      if (Call->getNumArgs() < 3)
        return;
      // find(CPO, Iter, Sent, Val...) -> Sent is Arg 2.
      const Expr *EndArg = Call->getArg(2);
      EndExprText = tooling::fixit::getText(*EndArg, *Result.Context).str();
    } else {
      if (Call->getNumArgs() < 2)
        return;
      // find(CPO, Range, Val, Proj) -> Range is Arg 1.
      const Expr *RangeArg = Call->getArg(1);
      // Avoid potential side-effects
      const Expr *InnerRange = RangeArg->IgnoreParenImpCasts();
      if (isa<DeclRefExpr>(InnerRange) || isa<MemberExpr>(InnerRange)) {
        const StringRef RangeText =
            tooling::fixit::getText(*RangeArg, *Result.Context);
        if (!RangeText.empty())
          EndExprText = ("std::ranges::end(" + RangeText + ")").str();
      }
    }
  }

  bool IsNegated = false;
  const UnaryOperator *NotOp = nullptr;
  const Expr *CurrentExpr = BoolOp;
  while (true) {
    auto Parents = Result.Context->getParents(*CurrentExpr);
    if (Parents.empty())
      break;
    if (const auto *P = Parents[0].get<ParenExpr>()) {
      CurrentExpr = P;
      continue;
    }
    if (const auto *U = Parents[0].get<UnaryOperator>()) {
      if (U->getOpcode() == UO_LNot) {
        NotOp = U;
        IsNegated = true;
      }
    }
    break;
  }

  const auto Diag =
      diag(BoolOp->getBeginLoc(),
           "result of standard algorithm used in boolean context; did "
           "you mean to compare with the end iterator?");

  if (!EndExprText.empty()) {
    if (IsNegated) {
      // !it -> (it == end)
      Diag << FixItHint::CreateReplacement(NotOp->getOperatorLoc(), "(");
      Diag << FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(BoolOp->getEndLoc(), 0,
                                     *Result.SourceManager,
                                     Result.Context->getLangOpts()),
          " == " + EndExprText + ")");
    } else {
      // it -> (it != end)
      Diag << FixItHint::CreateInsertion(BoolOp->getBeginLoc(), "(");
      Diag << FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(BoolOp->getEndLoc(), 0,
                                     *Result.SourceManager,
                                     Result.Context->getLangOpts()),
          " != " + EndExprText + ")");
    }
  }
}

} // namespace clang::tidy::bugprone
