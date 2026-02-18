#pragma once

#include <cstddef>

#include "clang/AST/RecursiveASTVisitor.h"

class MemoryAccessCounter
    : public clang::RecursiveASTVisitor<MemoryAccessCounter> {
public:
  MemoryAccessCounter() : load_count(0), store_count(0) {}

  bool VisitBinaryOperator(clang::BinaryOperator *bo) {
    using namespace clang;

    if (bo->isAssignmentOp()) {
      if (is_memory_access(bo->getLHS()))
        store_count++;

      if (is_memory_access(bo->getRHS()))
        load_count++;
    }

    return true;
  }

  bool VisitCompoundAssignOperator(clang::CompoundAssignOperator *cao) {
    if (is_memory_access(cao->getLHS())) {
      load_count++;
      store_count++;
    }

    if (is_memory_access(cao->getRHS()))
      load_count++;

    return true;
  }

  bool VisitUnaryOperator(clang::UnaryOperator *uo) {
    using namespace clang;

    if (uo->isIncrementDecrementOp()) {
      if (is_memory_access(uo->getSubExpr())) {
        load_count++;
        store_count++;
      }
    }

    return true;
  }

  void traverse(clang::Stmt *s) {
    load_count = 0;
    store_count = 0;
    TraverseStmt(s);
  }

  std::size_t get_load_count() const { return load_count; }
  std::size_t get_store_count() const { return store_count; }

private:
  std::size_t load_count, store_count;

  bool is_memory_access(clang::Expr *E) {
    E = E->IgnoreParenImpCasts();

    return llvm::isa<clang::DeclRefExpr>(E) ||
           llvm::isa<clang::UnaryOperator>(E) || // *p
           llvm::isa<clang::ArraySubscriptExpr>(E);
  }
};
