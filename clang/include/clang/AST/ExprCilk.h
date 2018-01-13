//===--- ExprCilk.h - Classes for representing expressions ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::Expr interface and subclasses for Cilk
/// expressions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPRCILK_H
#define LLVM_CLANG_AST_EXPRCILK_H

#include "clang/AST/Expr.h"

namespace clang {

/// CilkSpawnExpr - Wrapper for expressions whose evaluation is spawned.
///
class CilkSpawnExpr : public Expr {
  Stmt *SpawnedExpr;
  SourceLocation SpawnLoc;

public:
  CilkSpawnExpr(SourceLocation SpawnLoc, Expr *SpawnedExpr)
      : Expr(CilkSpawnExprClass, SpawnedExpr->getType(),
             SpawnedExpr->getValueKind(), SpawnedExpr->getObjectKind(),
             SpawnedExpr->isTypeDependent(), SpawnedExpr->isValueDependent(),
             SpawnedExpr->isInstantiationDependent(),
             SpawnedExpr->containsUnexpandedParameterPack()),
        SpawnedExpr(SpawnedExpr), SpawnLoc(SpawnLoc) {
  }

  explicit CilkSpawnExpr(EmptyShell Empty)
    : Expr(CilkSpawnExprClass, Empty) { }

  const Expr *getSpawnedExpr() const { return cast<Expr>(SpawnedExpr); }
  Expr *getSpawnedExpr() { return cast<Expr>(SpawnedExpr); }
  void setSpawnedExpr(Expr *E) { SpawnedExpr = E; }

  /// \brief Retrieve the location of this expression.
  SourceLocation getSpawnLoc() const { return SpawnLoc; }
  void setSpawnLoc(SourceLocation L) { SpawnLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY {
    return SpawnedExpr->getLocStart();
  }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return SpawnedExpr->getLocEnd();
  }
  SourceLocation getExprLoc() const LLVM_READONLY {
    return cast<Expr>(SpawnedExpr)->getExprLoc();
  }

  // Iterators
  child_range children() {
    return child_range(&SpawnedExpr, &SpawnedExpr+1);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CilkSpawnExprClass;
  }
};

}  // end namespace clang

#endif
