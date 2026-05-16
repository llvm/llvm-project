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

  SourceLocation InsertLoc;
  StringRef InsertStr;

  if (Lambda->hasExplicitParameters()) {
    const TypeSourceInfo *TSI = CallOp->getTypeSourceInfo();
    if (!TSI)
      return;
    auto FTL = TSI->getTypeLoc().IgnoreParens().getAs<FunctionTypeLoc>();
    if (!FTL)
      return;
    const SourceLocation RParenLoc = FTL.getRParenLoc();
    if (RParenLoc.isInvalid())
      return;
    InsertLoc = Lexer::getLocForEndOfToken(RParenLoc, 0, SM, LangOpts);
    InsertStr = " static";
  } else {
    SourceLocation ScanStart = Lambda->getIntroducerRange().getEnd();
    if (Lambda->isGenericLambda()) {
      if (const TemplateParameterList *TPL = Lambda->getTemplateParameterList()) {
        // Skip past the template requires-clause if present, otherwise past '>'.
        if (const Expr *Req = TPL->getRequiresClause())
          ScanStart = Req->getEndLoc();
        else
          ScanStart = TPL->getRAngleLoc();
      }
    }
    ScanStart = Lexer::getLocForEndOfToken(ScanStart, 0, SM, LangOpts);
    if (ScanStart.isInvalid())
      return;

    // Scan forward, tracking '[' / ']' depth to skip [[attr]] blocks.
    SourceLocation CurLoc = ScanStart;
    int Depth = 0;
    while (true) {
      Token RawTok;
      if (Lexer::getRawToken(CurLoc, RawTok, SM, LangOpts,
                             /*IgnoreWhiteSpace=*/true))
        break;
      if (RawTok.is(tok::l_square)) {
        ++Depth;
      } else if (RawTok.is(tok::r_square)) {
        if (--Depth < 0)
          break; // malformed source
        if (Depth == 0) {
          // Finished consuming one [[...]] block; keep scanning.
          CurLoc = Lexer::getLocForEndOfToken(RawTok.getLocation(), 0, SM,
                                              LangOpts);
          continue;
        }
      } else if (Depth == 0) {
        // Outside any attribute block: this is the insertion point.
        InsertLoc = RawTok.getLocation();
        break;
      }
      CurLoc =
          Lexer::getLocForEndOfToken(RawTok.getLocation(), 0, SM, LangOpts);
    }
    InsertStr = "static ";
  }

  if (InsertLoc.isInvalid())
    return;

  diag(LambdaLoc, "lambda with empty capture list can be marked 'static'")
      << FixItHint::CreateInsertion(InsertLoc, InsertStr);
}

} // namespace clang::tidy::modernize
