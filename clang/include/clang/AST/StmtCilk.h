//===- StmtCilk.h - Classes for Cilk statements -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines Cilk AST classes for executable statements and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTCILK_H
#define LLVM_CLANG_AST_STMTCILK_H

// #include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

/// CilkSpawnStmt - This represents a _Cilk_spawn.
///
class CilkSpawnStmt : public Stmt {
  SourceLocation SpawnLoc;
  Stmt *SpawnedStmt;

public:
  explicit CilkSpawnStmt(SourceLocation SL)
      : CilkSpawnStmt(SL, nullptr) {}

  CilkSpawnStmt(SourceLocation SL, Stmt *S)
      : Stmt(CilkSpawnStmtClass), SpawnLoc(SL), SpawnedStmt(S) { }

  // \brief Build an empty _Cilk_spawn statement.
  explicit CilkSpawnStmt(EmptyShell Empty)
      : Stmt(CilkSpawnStmtClass, Empty) { }

  const Stmt *getSpawnedStmt() const;
  Stmt *getSpawnedStmt();
  void setSpawnedStmt(Stmt *S) { SpawnedStmt = S; }

  SourceLocation getSpawnLoc() const { return SpawnLoc; }
  void setSpawnLoc(SourceLocation L) { SpawnLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return SpawnLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return SpawnedStmt->getLocEnd();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CilkSpawnStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SpawnedStmt, &SpawnedStmt+1);
  }
};

/// CilkSyncStmt - This represents a _Cilk_sync.
///
class CilkSyncStmt : public Stmt {
  SourceLocation SyncLoc;

public:
  CilkSyncStmt(SourceLocation SL) : Stmt(CilkSyncStmtClass), SyncLoc(SL) {}

  // \brief Build an empty _Cilk_sync statement.
  explicit CilkSyncStmt(EmptyShell Empty) : Stmt(CilkSyncStmtClass, Empty) { }

  SourceLocation getSyncLoc() const { return SyncLoc; }
  void setSyncLoc(SourceLocation L) { SyncLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return SyncLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY { return SyncLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CilkSyncStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

/// CilkForStmt - This represents a '_Cilk_for(init;cond;inc)' stmt.
class CilkForStmt : public Stmt {
  SourceLocation CilkForLoc;
  enum { INIT, COND, INC, LOOPVAR, BODY,
         END_EXPR };
  Stmt* SubExprs[END_EXPR]; // SubExprs[INIT] is an expression or declstmt.
  SourceLocation LParenLoc, RParenLoc;

public:
  CilkForStmt(const ASTContext &C, Stmt *Init,
              Expr *Cond, Expr *Inc, VarDecl *LoopVar, Stmt *Body,
              SourceLocation CFL, SourceLocation LP, SourceLocation RP);

  /// \brief Build an empty for statement.
  explicit CilkForStmt(EmptyShell Empty) : Stmt(CilkForStmtClass, Empty) { }

  Stmt *getInit() { return SubExprs[INIT]; }

  // /// \brief Retrieve the variable declared in this "for" statement, if any.
  // ///
  // /// In the following example, "y" is the condition variable.
  // /// \code
  // /// for (int x = random(); int y = mangle(x); ++x) {
  // ///   // ...
  // /// }
  // /// \endcode
  // VarDecl *getConditionVariable() const;
  // void setConditionVariable(const ASTContext &C, VarDecl *V);

  // /// If this CilkForStmt has a condition variable, return the faux DeclStmt
  // /// associated with the creation of that condition variable.
  // const DeclStmt *getConditionVariableDeclStmt() const {
  //   return reinterpret_cast<DeclStmt*>(SubExprs[CONDVAR]);
  // }

  VarDecl *getLoopVariable() const;
  void setLoopVariable(const ASTContext &C, VarDecl *V);

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Expr *getInc()  { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const Stmt *getInit() const { return SubExprs[INIT]; }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Expr *getInc()  const { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  const DeclStmt *getLoopVarDecl() const {
    return reinterpret_cast<DeclStmt*>(SubExprs[LOOPVAR]);
  }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  void setInit(Stmt *S) { SubExprs[INIT] = S; }
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  void setInc(Expr *E) { SubExprs[INC] = reinterpret_cast<Stmt*>(E); }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getCilkForLoc() const { return CilkForLoc; }
  void setCilkForLoc(SourceLocation L) { CilkForLoc = L; }
  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  SourceLocation getLocStart() const LLVM_READONLY { return CilkForLoc; }
  SourceLocation getLocEnd() const LLVM_READONLY {
    return SubExprs[BODY]->getLocEnd();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CilkForStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }
};

// /// CilkSpawnExpr - Wrapper for expressions whose evaluation is spawned.
// ///
// class CilkSpawnExpr : public Expr {
//   SourceLocation SpawnLoc;
//   Expr *SpawnedExpr;

// public:
//   CilkSpawnExpr(Expr *SpawnedExpr, SourceLocation SpawnLoc)
//       : Expr(CilkSpawnExprClass, SpawnedExpr->getType(),
//              SpawnedExpr->getValueKind(), SpawnedExpr->getObjectKind(),
//              SpawnedExpr->isTypeDependent(), SpawnedExpr->isValueDependent(),
//              SpawnedExpr->isInstantiationDependen(),
//              SpawnedExpr->containsUnexpandedParameterPack()),
//         SpawnedExpr(SpawnedExpr), SpawnLoc(SpawnLoc) {
//   }

//   const Expr *getSpawnedExpr() const;
//   Expr *getSpawnedExpr();
//   void setSpawnedExpr(Expr *E) { SpawnedExpr = E; }

//   SourceLocation getSpawnLoc() const { return SpawnLoc; }
//   void setSpawnLoc(SourceLocation L) { SpawnLoc = L; }

//   SourceLocation getLocStart() const LLVM_READONLY {
//     return SpawnedExpr->getLocStart();
//   }
//   SourceLocation getLocEnd() const LLVM_READONLY {
//     return SpawnedExpr->getLocEnd();
//   }

//   static bool classof(const Stmt *T) {
//     return T->getStmtClass() == CilkSpawnExprClass;
//   }

//   // Iterators
//   child_range children() {
//     return child_range(child_iterator(), child_iterator());
//   }
// };

}  // end namespace clang

#endif
