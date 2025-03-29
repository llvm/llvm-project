//===--- MathMissingParenthesesCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MathMissingParenthesesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void MathMissingParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(binaryOperator(unless(hasParent(binaryOperator())),
                                    unless(isAssignmentOperator()),
                                    unless(isComparisonOperator()),
                                    unless(hasAnyOperatorName("&&", "||")),
                                    hasDescendant(binaryOperator()))
                         .bind("binOp"),
                     this);
}

static int getPrecedence(const BinaryOperator *BinOp) {
  if (!BinOp)
    return 0;
  switch (BinOp->getOpcode()) {
  case BO_Mul:
  case BO_Div:
  case BO_Rem:
    return 5;
  case BO_Add:
  case BO_Sub:
    return 4;
  case BO_And:
    return 3;
  case BO_Xor:
    return 2;
  case BO_Or:
    return 1;
  default:
    return 0;
  }
}
static void addParantheses(const BinaryOperator *BinOp,
                           const BinaryOperator *ParentBinOp,
                           ClangTidyCheck *Check,
                           const clang::SourceManager &SM,
                           const clang::LangOptions &LangOpts) {
  if (!BinOp)
    return;

  int Precedence1 = getPrecedence(BinOp);
  int Precedence2 = getPrecedence(ParentBinOp);

  if (ParentBinOp != nullptr && Precedence1 != Precedence2 && Precedence1 > 0 &&
      Precedence2 > 0) {
    const clang::SourceLocation StartLoc = BinOp->getBeginLoc();
    const clang::SourceLocation EndLoc =
        clang::Lexer::getLocForEndOfToken(BinOp->getEndLoc(), 0, SM, LangOpts);

    auto Diag =
        Check->diag(StartLoc,
                    "'%0' has higher precedence than '%1'; add parentheses to "
                    "explicitly specify the order of operations")
        << (Precedence1 > Precedence2 ? BinOp->getOpcodeStr()
                                      : ParentBinOp->getOpcodeStr())
        << (Precedence1 > Precedence2 ? ParentBinOp->getOpcodeStr()
                                      : BinOp->getOpcodeStr())
        << SourceRange(StartLoc, EndLoc);

    if (EndLoc.isValid()) {
      Diag << FixItHint::CreateInsertion(StartLoc, "(")
           << FixItHint::CreateInsertion(EndLoc, ")");
    }
  }

  addParantheses(dyn_cast<BinaryOperator>(BinOp->getLHS()->IgnoreImpCasts()),
                 BinOp, Check, SM, LangOpts);
  addParantheses(dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreImpCasts()),
                 BinOp, Check, SM, LangOpts);
}

void MathMissingParenthesesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  std::vector<
      std::pair<clang::SourceRange, std::pair<const clang::BinaryOperator *,
                                              const clang::BinaryOperator *>>>
      Insertions;
  const SourceManager &SM = *Result.SourceManager;
  const clang::LangOptions &LO = Result.Context->getLangOpts();
  addParantheses(BinOp, nullptr, this, SM, LO);
}

} // namespace clang::tidy::readability
