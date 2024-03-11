//===--- MathMissingParenthesesCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MathMissingParenthesesCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <iostream>
using namespace std;

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void MathMissingParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(binaryOperator(unless(hasParent(binaryOperator())),
                                    hasDescendant(binaryOperator()))
                         .bind("binOp"),
                     this);
}

void addParantheses(const BinaryOperator *BinOp, clang::Rewriter &Rewrite,
                    const BinaryOperator *ParentBinOp, bool &NeedToDiagnose) {
  if (!BinOp)
    return;
  if (ParentBinOp != nullptr &&
      ParentBinOp->getOpcode() != BinOp->getOpcode()) {
    NeedToDiagnose = true;
  }
  clang::SourceLocation StartLoc = BinOp->getLHS()->getBeginLoc();
  clang::SourceLocation EndLoc = BinOp->getRHS()->getEndLoc();
  Rewrite.InsertText(StartLoc, "(");
  Rewrite.InsertTextAfterToken(EndLoc, ")");
  addParantheses(dyn_cast<BinaryOperator>(BinOp->getLHS()), Rewrite, BinOp,
                 NeedToDiagnose);
  addParantheses(dyn_cast<BinaryOperator>(BinOp->getRHS()), Rewrite, BinOp,
                 NeedToDiagnose);
}

void MathMissingParenthesesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  if (!BinOp)
    return;
  clang::SourceManager &SM = *Result.SourceManager;
  clang::LangOptions LO = Result.Context->getLangOpts();
  Rewriter Rewrite(SM, LO);
  bool NeedToDiagnose = false;
  addParantheses(BinOp, Rewrite, nullptr, NeedToDiagnose);
  clang::SourceLocation StartLoc = BinOp->getLHS()->getBeginLoc();
  clang::SourceLocation EndLoc =
      BinOp->getRHS()->getEndLoc().getLocWithOffset(1);
  clang::SourceRange range(StartLoc, EndLoc);
  std::string NewExpression = Rewrite.getRewrittenText(range);
  if (NeedToDiagnose) {
    diag(StartLoc, "add parantheses to clarify the precedence of operations")
        << FixItHint::CreateReplacement(range, NewExpression);
  }
}

} // namespace clang::tidy::readability
