#pragma once

#include <cstddef>

#include "clang/AST/RecursiveASTVisitor.h"

class FloatOpCounter : public clang::RecursiveASTVisitor<FloatOpCounter> {
public:
  explicit FloatOpCounter() : count(0) {}

  bool VisitBinaryOperator(clang::BinaryOperator *bo) {
    using namespace clang;

    if (bo->getLHS()->getType()->isFloatingType() &&
        bo->getRHS()->getType()->isFloatingType()) {
      ++count;
    }

    return true;
  }

  bool VisitCompoundAssignOperator(clang::CompoundAssignOperator *cao) {
    if (cao->getLHS()->getType()->isFloatingType() &&
        cao->getRHS()->getType()->isFloatingType()) {
      ++count;
    }

    return true;
  }

  bool VisitUnaryOperator(clang::UnaryOperator *uo) {
    using namespace clang;

    if (uo->getSubExpr()->getType()->isFloatingType()) {
      switch (uo->getOpcode()) {
      case UO_PreInc:
      case UO_PreDec:
      case UO_PostInc:
      case UO_PostDec:
      case UO_Plus:
      case UO_Minus:
        ++count;
        break;
      default:
        break;
      }
    }

    return true;
  }

  bool VisitFloatingLiteral(clang::FloatingLiteral * /* fl */) {
    ++count;
    return true;
  }

  bool VisitImplicitCastExpr(clang::ImplicitCastExpr *ice) {
    if (ice->getType()->isFloatingType() &&
        !ice->getSubExpr()->getType()->isFloatingType()) {
      ++count;
    }

    return true;
  }

  void traverse(clang::Stmt *S) {
    count = 0;
    TraverseStmt(S);
  }

  std::size_t get_count() const { return count; }

private:
  std::size_t count;
};
