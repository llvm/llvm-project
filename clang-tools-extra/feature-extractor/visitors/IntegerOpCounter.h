#pragma once

#include <cstddef>

#include "clang/AST/RecursiveASTVisitor.h"

class IntegerOpCounter : public clang::RecursiveASTVisitor<IntegerOpCounter> {
public:
  explicit IntegerOpCounter() : count(0) {}

  bool VisitBinaryOperator(clang::BinaryOperator *bo) {
    if (bo->getLHS()->getType()->isIntegerType() &&
        bo->getRHS()->getType()->isIntegerType()) {
      ++count;
    }

    return true;
  }

  bool VisitCompoundAssignOperator(clang::CompoundAssignOperator *cao) {
    if (cao->getLHS()->getType()->isIntegerType() &&
        cao->getRHS()->getType()->isIntegerType()) {
      ++count;
    }

    return true;
  }

  bool VisitUnaryOperator(clang::UnaryOperator *uo) {
    using namespace clang;

    if (uo->getSubExpr()->getType()->isIntegerType()) {
      switch (uo->getOpcode()) {
      case UO_PreInc:
      case UO_PostInc:
      case UO_PreDec:
      case UO_PostDec:
      case UO_Plus:
      case UO_Minus:
      case UO_Not:
        ++count;
        break;
      default:
        break;
      }
    }

    return true;
  }

  void traverse(clang::Stmt *s) {
    count = 0;
    TraverseStmt(s);
  }

  std::size_t get_count() const { return count; }

private:
  std::size_t count;
};
