//===--- ExprCilk.h - Classes for representing Cilk expressions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Defines the clang::Expr interface and subclasses for Cilk expressions.
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

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return SpawnedExpr->getBeginLoc();
  }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return SpawnedExpr->getEndLoc();
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
