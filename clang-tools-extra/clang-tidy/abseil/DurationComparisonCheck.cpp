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

/// Return `true` if `E` is a either: not a macro at all; or an argument to
/// one. In the latter case, we should still transform it.
static bool IsValidMacro(const MatchFinder::MatchResult &Result,
                         const Expr *E) {
  if (!E->getBeginLoc().isMacroID())
    return true;

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
  return !Loc.isMacroID();
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

  llvm::Optional<DurationScale> Scale = getScaleForInverse(
      Result.Nodes.getNodeAs<FunctionDecl>("function_decl")->getName());
  if (!Scale)
    return;

  // In most cases, we'll only need to rewrite one of the sides, but we also
  // want to handle the case of rewriting both sides. This is much simpler if
  // we unconditionally try and rewrite both, and let the rewriter determine
  // if nothing needs to be done.
  if (!IsValidMacro(Result, Binop->getLHS()) ||
      !IsValidMacro(Result, Binop->getRHS()))
    return;
  std::string LhsReplacement =
      rewriteExprFromNumberToDuration(Result, *Scale, Binop->getLHS());
  std::string RhsReplacement =
      rewriteExprFromNumberToDuration(Result, *Scale, Binop->getRHS());

  diag(Binop->getBeginLoc(), "perform comparison in the duration domain")
      << FixItHint::CreateReplacement(Binop->getSourceRange(),
                                      (llvm::Twine(LhsReplacement) + " " +
                                       Binop->getOpcodeStr() + " " +
                                       RhsReplacement)
                                          .str());
}

} // namespace abseil
} // namespace tidy
} // namespace clang
