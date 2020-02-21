//===- StmtTapir.h - Classes for Tapir statements -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines Tapir AST classes for executable statements and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTTAPIR_H
#define LLVM_CLANG_AST_STMTTAPIR_H

// #include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

class SpawnStmt : public Stmt {
  SourceLocation SpawnLoc;
  StringRef SyncVar;
  Stmt *SpawnedStmt;

public:
  explicit SpawnStmt(SourceLocation SL, StringRef sv)
      : SpawnStmt(SL, sv, nullptr) {}

  SpawnStmt(SourceLocation SL, StringRef SV, Stmt *S)
      : Stmt(SpawnStmtClass), SpawnLoc(SL), SyncVar(SV), SpawnedStmt(S) { }

  // \brief Build an empty spawn statement.
  explicit SpawnStmt(EmptyShell Empty)
      : Stmt(SpawnStmtClass,  Empty) { }

  StringRef getSyncVar() const;

  const Stmt *getSpawnedStmt() const;
  Stmt *getSpawnedStmt();
  void setSpawnedStmt(Stmt *S) { SpawnedStmt = S; }

  SourceLocation getSpawnLoc() const { return SpawnLoc; }
  void setSpawnLoc(SourceLocation L) { SpawnLoc = L; }

  SourceLocation getBeginLoc() const LLVM_READONLY { return SpawnLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return SpawnedStmt->getEndLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SpawnStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SpawnedStmt, &SpawnedStmt+1);
  }
};

class SyncStmt : public Stmt {
  SourceLocation SyncLoc;
  StringRef SyncVar; 

public:
  SyncStmt(SourceLocation SL, StringRef SV) : 
    Stmt(SyncStmtClass), SyncLoc(SL), SyncVar(SV) {}

  // \brief Build an empty __sync statement.
  explicit SyncStmt(EmptyShell Empty) : Stmt(SyncStmtClass, Empty) { }

  StringRef getSyncVar() const;

  SourceLocation getSyncLoc() const { return SyncLoc; }
  void setSyncLoc(SourceLocation L) { SyncLoc = L; }

  SourceLocation getBeginLoc() const LLVM_READONLY { return SyncLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY { return SyncLoc; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == SyncStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};

}

#endif
