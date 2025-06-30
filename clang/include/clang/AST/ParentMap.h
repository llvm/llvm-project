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

#include "llvm/Support/Casting.h"

namespace clang {
class Stmt;
class Expr;

class ParentMap {
  void* Impl;
public:
  ParentMap(Stmt* ASTRoot);
  ~ParentMap();

  /// Adds and/or updates the parent/child-relations of the complete
  /// stmt tree of S. All children of S including indirect descendants are
  /// visited and updated or inserted but not the parents of S.
  void addStmt(Stmt* S);

  /// Manually sets the parent of \p S to \p Parent.
  ///
  /// If \p S is already in the map, this method will update the mapping.
  void setParent(const Stmt *S, const Stmt *Parent);

  Stmt *getParent(Stmt*) const;
  Stmt *getParentIgnoreParens(Stmt *) const;
  Stmt *getParentIgnoreParenCasts(Stmt *) const;
  Stmt *getParentIgnoreParenImpCasts(Stmt *) const;
  Stmt *getOuterParenParent(Stmt *) const;

  template <typename... Ts> Stmt *getOuterMostAncestor(Stmt *S) const {
    Stmt *Res = nullptr;
    while (S) {
      if (llvm::isa<Ts...>(S))
        Res = S;
      S = getParent(S);
    }
    return Res;
  }

  template <typename... Ts> Stmt *getInnerMostAncestor(Stmt *S) const {
    while (S) {
      if (llvm::isa<Ts...>(S))
        return S;
      S = getParent(S);
    }
    return nullptr;
  }

  const Stmt *getParent(const Stmt* S) const {
    return getParent(const_cast<Stmt*>(S));
  }

  const Stmt *getParentIgnoreParens(const Stmt *S) const {
    return getParentIgnoreParens(const_cast<Stmt*>(S));
  }

  const Stmt *getParentIgnoreParenCasts(const Stmt *S) const {
    return getParentIgnoreParenCasts(const_cast<Stmt*>(S));
  }

  template <typename... Ts>
  const Stmt *getOuterMostAncestor(const Stmt *S) const {
    return getOuterMostAncestor<Ts...>(const_cast<Stmt *>(S));
  }

  template <typename... Ts>
  const Stmt *getInnerMostAncestor(const Stmt *S) const {
    return getInnerMostAncestor<Ts...>(const_cast<Stmt *>(S));
  }

  bool hasParent(const Stmt *S) const { return getParent(S) != nullptr; }

  bool isConsumedExpr(Expr *E) const;

  bool isConsumedExpr(const Expr *E) const {
    return isConsumedExpr(const_cast<Expr*>(E));
  }
};

} // end clang namespace
#endif
