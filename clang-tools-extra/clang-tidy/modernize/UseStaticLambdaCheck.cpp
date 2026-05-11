//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStaticLambdaCheck.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/LambdaCapture.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void UseStaticLambdaCheck::registerMatchers(MatchFinder *Finder) {
  // Match lambdas that have no captures and no capture-default.
  Finder->addMatcher(
      lambdaExpr(unless(hasAnyCapture(lambdaCapture()))).bind("lambda"), this);
}

void UseStaticLambdaCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  if (!Lambda)
    return;

  // Reject lambdas with a capture-default (e.g. [=] or [&] with no actual
  // captures): they are not truly capture-free.
  if (Lambda->getCaptureDefault() != LCD_None)
    return;

  const CXXMethodDecl *CallOp = Lambda->getCallOperator();

  // Already static
  if (CallOp->isStatic())
    return;

  // `static` and `mutable` are mutually exclusive lambda-specifiers (C++23).
  // A mutable lambda has a non-const call operator that is not already static.
  if (!CallOp->isConst())
    return;

  const SourceLocation LambdaLoc = Lambda->getBeginLoc();
  if (LambdaLoc.isInvalid() || LambdaLoc.isMacroID())
    return;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = getLangOpts();

  if (Lambda->hasExplicitParameters()) {
    // Lambda has an explicit parameter list: [...](params) { ... }.
    // Insert static right after the closing ).
    const TypeSourceInfo *TSI = CallOp->getTypeSourceInfo();
    if (!TSI)
      return;
    auto FTL = TSI->getTypeLoc().IgnoreParens().getAs<FunctionTypeLoc>();
    if (!FTL)
      return;
    const SourceLocation RParenLoc = FTL.getRParenLoc();
    if (RParenLoc.isInvalid())
      return;
    const SourceLocation InsertLoc =
        Lexer::getLocForEndOfToken(RParenLoc, 0, SM, LangOpts);
    diag(LambdaLoc, "lambda with empty capture list can be marked 'static'")
        << FixItHint::CreateInsertion(InsertLoc, " static");
  } else {
    // Lambda has no explicit parameter list: [...] { ... }.
    // We must add () as well since specifiers require a parameter list.
    const SourceLocation IntroEnd = Lambda->getIntroducerRange().getEnd();
    if (IntroEnd.isInvalid())
      return;
    const SourceLocation InsertLoc =
        Lexer::getLocForEndOfToken(IntroEnd, 0, SM, LangOpts);
    diag(LambdaLoc, "lambda with empty capture list can be marked 'static'")
        << FixItHint::CreateInsertion(InsertLoc, "() static");
  }
}

} // namespace clang::tidy::modernize