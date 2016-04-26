//===--- RedundantExpressionCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantExpressionCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

static bool AreIdenticalExpr(const Expr *Left, const Expr *Right) {
  if (!Left || !Right)
    return !Left && !Right;

  Left = Left->IgnoreParens();
  Right = Right->IgnoreParens();

  // Compare classes.
  if (Left->getStmtClass() != Right->getStmtClass())
    return false;

  // Compare children.
  Expr::const_child_iterator LeftIter = Left->child_begin();
  Expr::const_child_iterator RightIter = Right->child_begin();
  while (LeftIter != Left->child_end() && RightIter != Right->child_end()) {
    if (!AreIdenticalExpr(dyn_cast<Expr>(*LeftIter),
                          dyn_cast<Expr>(*RightIter)))
      return false;
    ++LeftIter;
    ++RightIter;
  }
  if (LeftIter != Left->child_end() || RightIter != Right->child_end())
    return false;

  // Perform extra checks.
  switch (Left->getStmtClass()) {
  default:
    return false;

  case Stmt::CharacterLiteralClass:
    return cast<CharacterLiteral>(Left)->getValue() ==
           cast<CharacterLiteral>(Right)->getValue();
  case Stmt::IntegerLiteralClass: {
    llvm::APInt LeftLit = cast<IntegerLiteral>(Left)->getValue();
    llvm::APInt RightLit = cast<IntegerLiteral>(Right)->getValue();
    return LeftLit.getBitWidth() == RightLit.getBitWidth() && LeftLit == RightLit;
  }
  case Stmt::FloatingLiteralClass:
    return cast<FloatingLiteral>(Left)->getValue().bitwiseIsEqual(
        cast<FloatingLiteral>(Right)->getValue());
  case Stmt::StringLiteralClass:
    return cast<StringLiteral>(Left)->getBytes() ==
           cast<StringLiteral>(Right)->getBytes();

  case Stmt::DeclRefExprClass:
    return cast<DeclRefExpr>(Left)->getDecl() ==
           cast<DeclRefExpr>(Right)->getDecl();
  case Stmt::MemberExprClass:
    return cast<MemberExpr>(Left)->getMemberDecl() ==
           cast<MemberExpr>(Right)->getMemberDecl();

  case Stmt::CStyleCastExprClass:
    return cast<CStyleCastExpr>(Left)->getTypeAsWritten() ==
           cast<CStyleCastExpr>(Right)->getTypeAsWritten();

  case Stmt::CallExprClass:
  case Stmt::ImplicitCastExprClass:
  case Stmt::ArraySubscriptExprClass:
    return true;

  case Stmt::UnaryOperatorClass:
    if (cast<UnaryOperator>(Left)->isIncrementDecrementOp())
      return false;
    return cast<UnaryOperator>(Left)->getOpcode() ==
           cast<UnaryOperator>(Right)->getOpcode();
  case Stmt::BinaryOperatorClass:
    return cast<BinaryOperator>(Left)->getOpcode() ==
           cast<BinaryOperator>(Right)->getOpcode();
  }
}

AST_MATCHER(BinaryOperator, OperandsAreEquivalent) {
  return AreIdenticalExpr(Node.getLHS(), Node.getRHS());
}

AST_MATCHER(BinaryOperator, isInMacro) {
  return Node.getOperatorLoc().isMacroID();
}

AST_MATCHER(Expr, isInstantiationDependent) {
  return Node.isInstantiationDependent();
}

void RedundantExpressionCheck::registerMatchers(MatchFinder *Finder) {
  const auto AnyLiteralExpr = ignoringParenImpCasts(
      anyOf(cxxBoolLiteral(), characterLiteral(), integerLiteral()));

  Finder->addMatcher(
      binaryOperator(anyOf(hasOperatorName("-"), hasOperatorName("/"),
                           hasOperatorName("%"), hasOperatorName("|"),
                           hasOperatorName("&"), hasOperatorName("^"),
                           matchers::isComparisonOperator(),
                           hasOperatorName("&&"), hasOperatorName("||"),
                           hasOperatorName("=")),
                     OperandsAreEquivalent(),
                     // Filter noisy false positives.
                     unless(isInstantiationDependent()),
                     unless(isInMacro()),
                     unless(hasType(realFloatingPointType())),
                     unless(hasEitherOperand(hasType(realFloatingPointType()))),
                     unless(hasLHS(AnyLiteralExpr)))
          .bind("binary"),
      this);
}

void RedundantExpressionCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binary"))
    diag(BinOp->getOperatorLoc(), "both side of operator are equivalent");
}

} // namespace misc
} // namespace tidy
} // namespace clang
