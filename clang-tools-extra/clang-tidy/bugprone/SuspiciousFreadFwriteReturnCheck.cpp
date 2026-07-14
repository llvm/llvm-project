//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousFreadFwriteReturnCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void SuspiciousFreadFwriteReturnCheck::registerMatchers(MatchFinder *Finder) {
  auto FreadFwriteDecl = functionDecl(hasAnyName("::fread", "::fwrite"));
  auto FreadFwriteCall =
      callExpr(callee(FreadFwriteDecl), hasArgument(2, expr().bind("nmemb")))
          .bind("call");

  // 1. Direct comparison or Variable Initialization
  auto CallOrVarInit =
      anyOf(FreadFwriteCall, declRefExpr(to(varDecl(hasInitializer(
                                 ignoringParenImpCasts(FreadFwriteCall))))));

  auto CallOrVarInitExpr = ignoringParenImpCasts(CallOrVarInit);

  // We only match IntegerLiteral (and unary '-' applied to IntegerLiteral) to
  // avoid firing the matcher on complex expressions that EvaluateAsInt would
  // later reject anyway.
  auto OtherOperand = ignoringParenImpCasts(
      expr(anyOf(integerLiteral(),
                 unaryOperator(hasOperatorName("-"),
                               hasUnaryOperand(integerLiteral()))))
          .bind("other"));

  auto CompLHS =
      binaryOperator(hasAnyOperatorName("==", "!=", "<", "<=", ">", ">="),
                     hasLHS(CallOrVarInitExpr), hasRHS(OtherOperand))
          .bind("suspicious_lhs");

  auto CompRHS =
      binaryOperator(hasAnyOperatorName("==", "!=", "<", "<=", ">", ">="),
                     hasLHS(OtherOperand), hasRHS(CallOrVarInitExpr))
          .bind("suspicious_rhs");

  Finder->addMatcher(CompLHS, this);
  Finder->addMatcher(CompRHS, this);
}

void SuspiciousFreadFwriteReturnCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  if (!Call || !Call->getDirectCallee())
    return;

  StringRef FuncName = Call->getDirectCallee()->getName();

  const BinaryOperator *BinOp = nullptr;
  bool IsCallLHS = false;

  if (const auto *Op =
          Result.Nodes.getNodeAs<BinaryOperator>("suspicious_lhs")) {
    BinOp = Op;
    IsCallLHS = true;
  } else if (const auto *Op =
                 Result.Nodes.getNodeAs<BinaryOperator>("suspicious_rhs")) {
    BinOp = Op;
    IsCallLHS = false;
  }

  if (!BinOp)
    return;

  const auto *OtherExpr = Result.Nodes.getNodeAs<Expr>("other");
  if (!OtherExpr)
    return;

  Expr::EvalResult Eval;
  if (!OtherExpr->EvaluateAsInt(Eval, *Result.Context) || Eval.HasSideEffects)
    return;

  const llvm::APSInt Val = Eval.Val.getInt();
  const BinaryOperatorKind Op = BinOp->getOpcode();

  const Expr *CallOperand = IsCallLHS ? BinOp->getLHS() : BinOp->getRHS();
  const bool IsDirectComparison =
      isa<CallExpr>(CallOperand->IgnoreParenImpCasts());

  const auto *NmembExpr = Result.Nodes.getNodeAs<Expr>("nmemb");
  assert(NmembExpr && "nmemb should always be bound");

  bool IsNmembOne = false;
  Expr::EvalResult NmembEval;
  if (NmembExpr->EvaluateAsInt(NmembEval, *Result.Context) &&
      !NmembEval.HasSideEffects) {
    if (NmembEval.Val.getInt().isOne())
      IsNmembOne = true;
  }

  enum class DiagnosticKind {
    None,
    AlwaysFalse,
    AlwaysTrue,
    Suspicious,
    Discarded
  };
  DiagnosticKind DiagKind = DiagnosticKind::None;

  if (Val.isNegative()) {
    if (Op == BO_EQ) {
      DiagKind = DiagnosticKind::AlwaysFalse;
    } else if (Op == BO_NE) {
      DiagKind = DiagnosticKind::AlwaysTrue;
    } else if (IsCallLHS) {
      if (Op == BO_LT || Op == BO_LE)
        DiagKind = DiagnosticKind::AlwaysFalse;
      else if (Op == BO_GT || Op == BO_GE)
        DiagKind = DiagnosticKind::AlwaysTrue;
    } else {
      if (Op == BO_GT || Op == BO_GE)
        DiagKind = DiagnosticKind::AlwaysFalse;
      else if (Op == BO_LT || Op == BO_LE)
        DiagKind = DiagnosticKind::AlwaysTrue;
    }
  } else if (Val.isZero()) {
    if (IsCallLHS) {
      if (Op == BO_LT) {
        DiagKind = DiagnosticKind::AlwaysFalse;
      } else if (Op == BO_GE) {
        DiagKind = DiagnosticKind::AlwaysTrue;
      } else if (Op == BO_LE) {
        DiagKind = DiagnosticKind::Suspicious;
      } else if (IsDirectComparison && !IsNmembOne &&
                 (Op == BO_EQ || Op == BO_NE || Op == BO_GT)) {
        DiagKind = DiagnosticKind::Discarded;
      }
    } else if (Op == BO_GT) {
      DiagKind = DiagnosticKind::AlwaysFalse;
    } else if (Op == BO_LE) {
      DiagKind = DiagnosticKind::AlwaysTrue;
    } else if (Op == BO_GE) {
      DiagKind = DiagnosticKind::Suspicious;
    } else if (IsDirectComparison && !IsNmembOne &&
               (Op == BO_EQ || Op == BO_NE || Op == BO_LT)) {
      DiagKind = DiagnosticKind::Discarded;
    }
  }

  if (DiagKind == DiagnosticKind::None)
    return;

  std::optional<FixItHint> Hint;
  StringRef NmembStr = Lexer::getSourceText(
      CharSourceRange::getTokenRange(NmembExpr->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  if (!NmembStr.empty()) {
    if (IsCallLHS) {
      Hint = FixItHint::CreateReplacement(
          SourceRange(BinOp->getOperatorLoc(), OtherExpr->getEndLoc()),
          ("!= " + NmembStr).str());
    } else {
      Hint = FixItHint::CreateReplacement(
          SourceRange(OtherExpr->getBeginLoc(), BinOp->getOperatorLoc()),
          (NmembStr.str() + " !="));
    }
  }

  if (DiagKind == DiagnosticKind::AlwaysFalse) {
    auto D = diag(BinOp->getOperatorLoc(),
                  "return value of '%0' is an unsigned 'size_t'; this "
                  "comparison is always false");
    D << FuncName;
    if (Hint)
      D << *Hint;
  } else if (DiagKind == DiagnosticKind::AlwaysTrue) {
    auto D = diag(BinOp->getOperatorLoc(),
                  "return value of '%0' is an unsigned 'size_t'; this "
                  "comparison is always true");
    D << FuncName;
    if (Hint)
      D << *Hint;
  } else if (DiagKind == DiagnosticKind::Suspicious) {
    StringRef BadExpr = Op == BO_LE ? "<= 0" : "0 >=";
    auto D = diag(BinOp->getOperatorLoc(),
                  "suspicious comparison against 0; '%0' returns an unsigned "
                  "'size_t', so comparing it with '%1' is equivalent to "
                  "comparing it with '== 0'. To detect short reads or writes, "
                  "compare against the 'nmemb' argument");
    D << FuncName << BadExpr;
    if (Hint)
      D << *Hint;
  } else if (DiagKind == DiagnosticKind::Discarded) {
    auto D = diag(BinOp->getOperatorLoc(),
                  "return value of '%0' is compared to 0; since 'nmemb' is not "
                  "1, partial reads or writes cannot be handled. Compare "
                  "against the 'nmemb' argument instead");
    D << FuncName;
    if (Hint)
      D << *Hint;
  }
}

} // namespace clang::tidy::bugprone
