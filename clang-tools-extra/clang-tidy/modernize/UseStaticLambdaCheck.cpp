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

namespace {
AST_MATCHER(LambdaExpr, hasNoCaptureDefault) {
  return Node.getCaptureDefault() == LCD_None;
}
AST_MATCHER(LambdaExpr, callOperatorIsStatic) {
  return Node.getCallOperator()->isStatic();
}
AST_MATCHER(LambdaExpr, callOperatorIsConst) {
  return Node.getCallOperator()->isConst();
}
} // namespace

void UseStaticLambdaCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      lambdaExpr(hasNoCaptureDefault(), unless(hasAnyCapture(lambdaCapture())),
                 unless(callOperatorIsStatic()), callOperatorIsConst())
          .bind("lambda"),
      this);
}

void UseStaticLambdaCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  assert(Lambda && "lambda should be bound by the matcher");

  const SourceLocation LambdaLoc = Lambda->getBeginLoc();
  if (LambdaLoc.isInvalid() || LambdaLoc.isMacroID())
    return;

  const CXXMethodDecl *CallOp = Lambda->getCallOperator();
  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = getLangOpts();

  if (Lambda->hasExplicitParameters()) {
    // Lambda has an explicit parameter list: [...](params) { ... }.
    // Insert 'static' right after the closing ')'.
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
    // In C++23 (form 4), specs such as 'static' may appear without '()'.
    const SourceLocation IntroEnd = Lambda->getIntroducerRange().getEnd();
    if (IntroEnd.isInvalid())
      return;
    const SourceLocation InsertLoc =
        Lexer::getLocForEndOfToken(IntroEnd, 0, SM, LangOpts);
    diag(LambdaLoc, "lambda with empty capture list can be marked 'static'")
        << FixItHint::CreateInsertion(InsertLoc, " static");
  }
}

} // namespace clang::tidy::modernize
