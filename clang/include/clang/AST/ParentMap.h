//===--- ParentMap.h - Mappings from Stmts to their Parents -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ParentMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_PARENTMAP_H
#define LLVM_CLANG_AST_PARENTMAP_H

#include <utility>

namespace clang {
class Stmt;
class Expr;

class ParentMap {
  void* Impl;
public:
  ParentMap(Stmt* ASTRoot);
  ~ParentMap();

  using ValueT = std::pair<Stmt *, unsigned>;

  /// Adds and/or updates the parent/child-relations of the complete
  /// stmt tree of S. All children of S including indirect descendants are
  /// visited and updated or inserted but not the parents of S.
  void addStmt(Stmt *S, unsigned Depth);

  /// Manually sets the parent of \p S to \p Parent.
  ///
  /// If \p S is already in the map, this method will update the mapping.
  void setParent(const Stmt *S, const Stmt *Parent);

  ValueT lookup(Stmt *) const;
  Stmt *getParent(Stmt*) const;
  unsigned getParentDepth(Stmt *) const;
  Stmt *getParentIgnoreParens(Stmt *) const;
  Stmt *getParentIgnoreParenCasts(Stmt *) const;
  Stmt *getParentIgnoreParenImpCasts(Stmt *) const;
  Stmt *getOuterParenParent(Stmt *) const;

  const Stmt *getParent(const Stmt* S) const {
    return getParent(const_cast<Stmt*>(S));
  }

  unsigned getParentDepth(const Stmt *S) const {
    return getParentDepth(const_cast<Stmt *>(S));
  }

  const Stmt *getParentIgnoreParens(const Stmt *S) const {
    return getParentIgnoreParens(const_cast<Stmt*>(S));
  }

  const Stmt *getParentIgnoreParenCasts(const Stmt *S) const {
    return getParentIgnoreParenCasts(const_cast<Stmt*>(S));
  }

  bool hasParent(const Stmt *S) const { return getParent(S) != nullptr; }

  bool isConsumedExpr(Expr *E) const;

  bool isConsumedExpr(const Expr *E) const {
    return isConsumedExpr(const_cast<Expr*>(E));
  }
};

} // end namespace clang
#endif
