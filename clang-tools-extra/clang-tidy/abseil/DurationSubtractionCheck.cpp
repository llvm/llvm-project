//===--- DurationSubtractionCheck.cpp - clang-tidy ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DurationSubtractionCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

void DurationSubtractionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      binaryOperator(
          hasOperatorName("-"),
          hasLHS(callExpr(callee(functionDecl(DurationConversionFunction())
                                     .bind("function_decl")),
                          hasArgument(0, expr().bind("lhs_arg")))))
          .bind("binop"),
      this);
}

void DurationSubtractionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Binop = Result.Nodes.getNodeAs<BinaryOperator>("binop");
  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>("function_decl");

  // Don't try to replace things inside of macro definitions.
  if (Binop->getExprLoc().isMacroID() || Binop->getExprLoc().isInvalid())
    return;

  llvm::Optional<DurationScale> Scale = getScaleForInverse(FuncDecl->getName());
  if (!Scale)
    return;

  llvm::Optional<std::string> RhsReplacement =
      rewriteExprFromNumberToDuration(Result, *Scale, Binop->getRHS());
  if (!RhsReplacement)
    return;

  const Expr *LhsArg = Result.Nodes.getNodeAs<Expr>("lhs_arg");

  diag(Binop->getBeginLoc(), "perform subtraction in the duration domain")
      << FixItHint::CreateReplacement(
             Binop->getSourceRange(),
             (llvm::Twine("absl::") + FuncDecl->getName() + "(" +
              tooling::fixit::getText(*LhsArg, *Result.Context) + " - " +
              *RhsReplacement + ")")
                 .str());
}

} // namespace abseil
} // namespace tidy
} // namespace clang
