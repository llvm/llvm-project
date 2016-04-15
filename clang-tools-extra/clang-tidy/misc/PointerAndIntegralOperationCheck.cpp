//===--- PointerAndIntegralOperationCheck.cpp - clang-tidy-----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PointerAndIntegralOperationCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void PointerAndIntegralOperationCheck::registerMatchers(MatchFinder *Finder) {
  const auto PointerExpr = expr(hasType(pointerType()));
  const auto BoolExpr = ignoringParenImpCasts(hasType(booleanType()));
  const auto CharExpr = ignoringParenImpCasts(hasType(isAnyCharacter()));

  const auto BinOpWithPointerExpr =
      binaryOperator(unless(anyOf(hasOperatorName(","), hasOperatorName("="))),
                     hasEitherOperand(PointerExpr));

  const auto AssignToPointerExpr =
      binaryOperator(hasOperatorName("="), hasLHS(PointerExpr));

  const auto CompareToPointerExpr =
      binaryOperator(anyOf(hasOperatorName("<"), hasOperatorName("<="),
                           hasOperatorName(">"), hasOperatorName(">=")),
                     hasEitherOperand(PointerExpr));

  // Detect expression like: ptr = (x != y);
  Finder->addMatcher(binaryOperator(AssignToPointerExpr, hasRHS(BoolExpr))
                         .bind("assign-bool-to-pointer"),
                     this);

  // Detect expression like: ptr = A[i]; where A is char*.
  Finder->addMatcher(binaryOperator(AssignToPointerExpr, hasRHS(CharExpr))
                         .bind("assign-char-to-pointer"),
                     this);

  // Detect expression like: ptr < false;
  Finder->addMatcher(
      binaryOperator(BinOpWithPointerExpr,
                     hasEitherOperand(ignoringParenImpCasts(cxxBoolLiteral())))
          .bind("pointer-and-bool-literal"),
      this);

  // Detect expression like: ptr < 'a';
  Finder->addMatcher(binaryOperator(BinOpWithPointerExpr,
                                    hasEitherOperand(ignoringParenImpCasts(
                                        characterLiteral())))
                         .bind("pointer-and-char-literal"),
                     this);

  // Detect expression like: ptr < 0;
  Finder->addMatcher(binaryOperator(CompareToPointerExpr,
                                    hasEitherOperand(ignoringParenImpCasts(
                                        integerLiteral(equals(0)))))
                         .bind("compare-pointer-to-zero"),
                     this);

  // Detect expression like: ptr < nullptr;
  Finder->addMatcher(binaryOperator(CompareToPointerExpr,
                                    hasEitherOperand(ignoringParenImpCasts(
                                        cxxNullPtrLiteralExpr())))
                         .bind("compare-pointer-to-null"),
                     this);
}

void PointerAndIntegralOperationCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *E =
          Result.Nodes.getNodeAs<BinaryOperator>("assign-bool-to-pointer")) {
    diag(E->getOperatorLoc(), "suspicious assignment from bool to pointer");
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "assign-char-to-pointer")) {
    diag(E->getOperatorLoc(), "suspicious assignment from char to pointer");
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "pointer-and-bool-literal")) {
    diag(E->getOperatorLoc(),
         "suspicious operation between pointer and bool literal");
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "pointer-and-char-literal")) {
    diag(E->getOperatorLoc(),
         "suspicious operation between pointer and character literal");
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "compare-pointer-to-zero")) {
    diag(E->getOperatorLoc(), "suspicious comparison of pointer with zero");
  } else if (const auto *E = Result.Nodes.getNodeAs<BinaryOperator>(
                 "compare-pointer-to-null")) {
    diag(E->getOperatorLoc(),
         "suspicious comparison of pointer with null expression");
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
