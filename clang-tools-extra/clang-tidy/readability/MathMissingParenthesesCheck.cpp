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

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void MathMissingParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(binaryOperator(unless(hasParent(binaryOperator())),
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
static bool addParantheses(const BinaryOperator *BinOp,
                           const BinaryOperator *ParentBinOp,
                           std::vector<clang::SourceRange> &Insertions) {
  bool NeedToDiagnose = false;
  if (!BinOp)
    return NeedToDiagnose;

  if (ParentBinOp != nullptr &&
      getPrecedence(BinOp) != getPrecedence(ParentBinOp)) {
    NeedToDiagnose = true;
    const clang::SourceLocation StartLoc = BinOp->getBeginLoc();
    const clang::SourceLocation EndLoc = BinOp->getEndLoc().getLocWithOffset(1);
    Insertions.push_back(clang::SourceRange(StartLoc, EndLoc));
  }

  NeedToDiagnose |= addParantheses(
      dyn_cast<BinaryOperator>(BinOp->getLHS()->IgnoreImpCasts()), BinOp,
      Insertions);
  NeedToDiagnose |= addParantheses(
      dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreImpCasts()), BinOp,
      Insertions);
  return NeedToDiagnose;
}

void MathMissingParenthesesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  bool NeedToDiagnose = false;
  std::vector<clang::SourceRange> Insertions;
  const clang::SourceLocation StartLoc = BinOp->getBeginLoc();

  if (addParantheses(BinOp, nullptr, Insertions)) {
    auto const &Diag = diag(
        StartLoc, "add parantheses to clarify the precedence of operations");
    for (const auto &Insertion : Insertions) {
      Diag << FixItHint::CreateInsertion(Insertion.getBegin(), "(");
      Diag << FixItHint::CreateInsertion(Insertion.getEnd(), ")");
      Diag << SourceRange(Insertion.getBegin(), Insertion.getEnd());
    }
  }
}

} // namespace clang::tidy::readability
