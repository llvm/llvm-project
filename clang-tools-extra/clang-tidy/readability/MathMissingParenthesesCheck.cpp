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

void addParantheses(
    const BinaryOperator *BinOp, const BinaryOperator *ParentBinOp,
    bool &NeedToDiagnose,
    std::vector<std::pair<clang::SourceLocation, clang::SourceLocation>>
        &Insertions) {
  if (!BinOp)
    return;

  if (ParentBinOp != nullptr &&
      ParentBinOp->getOpcode() != BinOp->getOpcode()) {
    NeedToDiagnose = true;
  }
  const clang::SourceLocation StartLoc = BinOp->getBeginLoc();
  const clang::SourceLocation EndLoc = BinOp->getEndLoc().getLocWithOffset(1);
  Insertions.push_back({StartLoc, EndLoc});
  addParantheses(dyn_cast<BinaryOperator>(BinOp->getLHS()->IgnoreImpCasts()),
                 BinOp, NeedToDiagnose, Insertions);
  addParantheses(dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreImpCasts()),
                 BinOp, NeedToDiagnose, Insertions);
}

void MathMissingParenthesesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  bool NeedToDiagnose = false;
  std::vector<std::pair<clang::SourceLocation, clang::SourceLocation>>
      Insertions;
  addParantheses(BinOp, nullptr, NeedToDiagnose, Insertions);
  const clang::SourceLocation StartLoc = BinOp->getBeginLoc();
  const clang::SourceLocation EndLoc = BinOp->getEndLoc().getLocWithOffset(1);
  const clang::SourceRange range(StartLoc, EndLoc);
  if (!Insertions.empty()) {
    Insertions.erase(Insertions.begin());
  }
  if (NeedToDiagnose) {
    auto const &Diag = diag(
        StartLoc, "add parantheses to clarify the precedence of operations");
    for (const auto &Insertion : Insertions) {
      Diag << FixItHint::CreateInsertion(Insertion.first, "(");
      Diag << FixItHint::CreateInsertion(Insertion.second, ")");
    }
  }
}

} // namespace clang::tidy::readability
