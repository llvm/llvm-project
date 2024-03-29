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
                                    unless(allOf(hasOperatorName("&&"),
                                                 hasOperatorName("||"))),
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
static bool addParantheses(
    const BinaryOperator *BinOp, const BinaryOperator *ParentBinOp,
    std::vector<
        std::pair<clang::SourceRange, std::pair<const clang::BinaryOperator *,
                                                const clang::BinaryOperator *>>>
        &Insertions,
    const clang::SourceManager &SM, const clang::LangOptions &LangOpts) {
  bool NeedToDiagnose = false;
  if (!BinOp)
    return NeedToDiagnose;

  if (ParentBinOp != nullptr &&
      getPrecedence(BinOp) != getPrecedence(ParentBinOp)) {
    NeedToDiagnose = true;
    const clang::SourceLocation StartLoc = BinOp->getBeginLoc();
    clang::SourceLocation EndLoc =
        clang::Lexer::getLocForEndOfToken(BinOp->getEndLoc(), 0, SM, LangOpts);
    Insertions.push_back(
        {clang::SourceRange(StartLoc, EndLoc), {BinOp, ParentBinOp}});
  }

  NeedToDiagnose |= addParantheses(
      dyn_cast<BinaryOperator>(BinOp->getLHS()->IgnoreImpCasts()), BinOp,
      Insertions, SM, LangOpts);
  NeedToDiagnose |= addParantheses(
      dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreImpCasts()), BinOp,
      Insertions, SM, LangOpts);
  return NeedToDiagnose;
}

void MathMissingParenthesesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  std::vector<
      std::pair<clang::SourceRange, std::pair<const clang::BinaryOperator *,
                                              const clang::BinaryOperator *>>>
      Insertions;
  const clang::SourceLocation StartLoc = BinOp->getBeginLoc();
  const SourceManager &SM = *Result.SourceManager;
  const clang::LangOptions &LO = Result.Context->getLangOpts();

  if (addParantheses(BinOp, nullptr, Insertions, SM, LO)) {
    for (const auto &Insertion : Insertions) {
      const clang::BinaryOperator *BinOp1 = Insertion.second.first;
      const clang::BinaryOperator *BinOp2 = Insertion.second.second;

      int Precedence1 = getPrecedence(BinOp1);
      int Precedence2 = getPrecedence(BinOp2);

      auto Diag = diag(Insertion.first.getBegin(),
                       "'%0' has higher precedence than '%1'; add parentheses "
                       "to make the precedence of operations explicit")
                  << (Precedence1 > Precedence2 ? BinOp1->getOpcodeStr()
                                                : BinOp2->getOpcodeStr())
                  << (Precedence1 > Precedence2 ? BinOp2->getOpcodeStr()
                                                : BinOp1->getOpcodeStr());

      Diag << FixItHint::CreateInsertion(Insertion.first.getBegin(), "(");
      Diag << FixItHint::CreateInsertion(Insertion.first.getEnd(), ")");
      Diag << Insertion.first;
    }
  }
}

} // namespace clang::tidy::readability
