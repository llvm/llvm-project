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
