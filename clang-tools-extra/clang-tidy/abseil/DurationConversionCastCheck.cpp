//===--- DurationConversionCastCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DurationConversionCastCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

void DurationConversionCastCheck::registerMatchers(MatchFinder *Finder) {
  auto CallMatcher = ignoringImpCasts(callExpr(
      callee(functionDecl(DurationConversionFunction()).bind("func_decl")),
      hasArgument(0, expr().bind("arg"))));

  Finder->addMatcher(
      expr(anyOf(
          cxxStaticCastExpr(hasSourceExpression(CallMatcher)).bind("cast_expr"),
          cStyleCastExpr(hasSourceExpression(CallMatcher)).bind("cast_expr"),
          cxxFunctionalCastExpr(hasSourceExpression(CallMatcher))
              .bind("cast_expr"))),
      this);
}

void DurationConversionCastCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast =
      Result.Nodes.getNodeAs<ExplicitCastExpr>("cast_expr");

  if (!isNotInMacro(Result, MatchedCast))
    return;

  const auto *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>("func_decl");
  const auto *Arg = Result.Nodes.getNodeAs<Expr>("arg");
  StringRef ConversionFuncName = FuncDecl->getName();

  llvm::Optional<DurationScale> Scale = getScaleForInverse(ConversionFuncName);
  if (!Scale)
    return;

  // Casting a double to an integer.
  if (MatchedCast->getTypeAsWritten()->isIntegerType() &&
      ConversionFuncName.contains("Double")) {
    llvm::StringRef NewFuncName = getInverseForScale(*Scale).second;

    diag(MatchedCast->getBeginLoc(),
         "duration should be converted directly to an integer rather than "
         "through a type cast")
        << FixItHint::CreateReplacement(
               MatchedCast->getSourceRange(),
               (llvm::Twine(NewFuncName.substr(2)) + "(" +
                tooling::fixit::getText(*Arg, *Result.Context) + ")")
                   .str());
  }

  // Casting an integer to a double.
  if (MatchedCast->getTypeAsWritten()->isRealFloatingType() &&
      ConversionFuncName.contains("Int64")) {
    llvm::StringRef NewFuncName = getInverseForScale(*Scale).first;

    diag(MatchedCast->getBeginLoc(), "duration should be converted directly to "
                                     "a floating-piont number rather than "
                                     "through a type cast")
        << FixItHint::CreateReplacement(
               MatchedCast->getSourceRange(),
               (llvm::Twine(NewFuncName.substr(2)) + "(" +
                tooling::fixit::getText(*Arg, *Result.Context) + ")")
                   .str());
  }
}

} // namespace abseil
} // namespace tidy
} // namespace clang
