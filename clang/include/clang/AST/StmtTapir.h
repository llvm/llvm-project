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

class ForallStmt : public Stmt {
  enum { INIT, CONDVAR, COND, INC, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // SubExprs[INIT] is an expression or declstmt.
  SourceLocation LParenLoc, RParenLoc;

public:
  ForallStmt(const ASTContext &C, Stmt *Init, Expr *Cond, VarDecl *condVar,
          Expr *Inc, Stmt *Body, SourceLocation FL, SourceLocation LP,
          SourceLocation RP);

  /// Build an empty for statement.
  explicit ForallStmt(EmptyShell Empty) : Stmt(ForallStmtClass, Empty) {}

  Stmt *getInit() { return SubExprs[INIT]; }

  /// Retrieve the variable declared in this "for" statement, if any.
  ///
  /// In the following example, "y" is the condition variable.
  /// \code
  /// for (int x = random(); int y = mangle(x); ++x) {
  ///   // ...
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const;
  void setConditionVariable(const ASTContext &C, VarDecl *V);

  /// If this ForStmt has a condition variable, return the faux DeclStmt
  /// associated with the creation of that condition variable.
  const DeclStmt *getConditionVariableDeclStmt() const {
    return reinterpret_cast<DeclStmt*>(SubExprs[CONDVAR]);
  }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Expr *getInc()  { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const Stmt *getInit() const { return SubExprs[INIT]; }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Expr *getInc()  const { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  void setInit(Stmt *S) { SubExprs[INIT] = S; }
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  void setInc(Expr *E) { SubExprs[INC] = reinterpret_cast<Stmt*>(E); }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getForallLoc() const { return ForStmtBits.ForLoc; }
  void setForallLoc(SourceLocation L) { ForStmtBits.ForLoc = L; }
  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  SourceLocation getBeginLoc() const { return getForallLoc(); }
  SourceLocation getEndLoc() const { return getBody()->getEndLoc(); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ForallStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[0]+END_EXPR);
  }

  const_child_range children() const {
    return const_child_range(&SubExprs[0], &SubExprs[0] + END_EXPR);
  }
};
/// CXXForallRangeStmt - This represents C++0x [stmt.ranged]'s ranged for
/// statement, represented as 'for (range-declarator : range-expression)'
/// or 'for (init-statement range-declarator : range-expression)'.
///
/// This is stored in a partially-desugared form to allow full semantic
/// analysis of the constituent components. The original syntactic components
/// can be extracted using getLoopVariable and getRangeInit.
class CXXForallRangeStmt : public Stmt {
  SourceLocation ForLoc;
  enum { INIT, RANGE, BEGINSTMT, ENDSTMT, COND, INC, LOOPVAR, BODY, END };
  // SubExprs[RANGE] is an expression or declstmt.
  // SubExprs[COND] and SubExprs[INC] are expressions.
  Stmt *SubExprs[END];
  SourceLocation CoawaitLoc;
  SourceLocation ColonLoc;
  SourceLocation RParenLoc;

  friend class ASTStmtReader;
public:
  CXXForallRangeStmt(Stmt *InitStmt, DeclStmt *Range, DeclStmt *Begin,
                  DeclStmt *End, Expr *Cond, Expr *Inc, DeclStmt *LoopVar,
                  Stmt *Body, SourceLocation FL, SourceLocation CAL,
                  SourceLocation CL, SourceLocation RPL);
  CXXForallRangeStmt(EmptyShell Empty) : Stmt(CXXForallRangeStmtClass, Empty) { }

  Stmt *getInit() { return SubExprs[INIT]; }
  VarDecl *getLoopVariable();
  Expr *getRangeInit();

  const Stmt *getInit() const { return SubExprs[INIT]; }
  const VarDecl *getLoopVariable() const;
  const Expr *getRangeInit() const;


  DeclStmt *getRangeStmt() { return cast<DeclStmt>(SubExprs[RANGE]); }
  DeclStmt *getBeginStmt() {
    return cast_or_null<DeclStmt>(SubExprs[BEGINSTMT]);
  }
  DeclStmt *getEndStmt() { return cast_or_null<DeclStmt>(SubExprs[ENDSTMT]); }
  Expr *getCond() { return cast_or_null<Expr>(SubExprs[COND]); }
  Expr *getInc() { return cast_or_null<Expr>(SubExprs[INC]); }
  DeclStmt *getLoopVarStmt() { return cast<DeclStmt>(SubExprs[LOOPVAR]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const DeclStmt *getRangeStmt() const {
    return cast<DeclStmt>(SubExprs[RANGE]);
  }
  const DeclStmt *getBeginStmt() const {
    return cast_or_null<DeclStmt>(SubExprs[BEGINSTMT]);
  }
  const DeclStmt *getEndStmt() const {
    return cast_or_null<DeclStmt>(SubExprs[ENDSTMT]);
  }
  const Expr *getCond() const {
    return cast_or_null<Expr>(SubExprs[COND]);
  }
  const Expr *getInc() const {
    return cast_or_null<Expr>(SubExprs[INC]);
  }
  const DeclStmt *getLoopVarStmt() const {
    return cast<DeclStmt>(SubExprs[LOOPVAR]);
  }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  void setInit(Stmt *S) { SubExprs[INIT] = S; }
  void setRangeInit(Expr *E) { SubExprs[RANGE] = reinterpret_cast<Stmt*>(E); }
  void setRangeStmt(Stmt *S) { SubExprs[RANGE] = S; }
  void setBeginStmt(Stmt *S) { SubExprs[BEGINSTMT] = S; }
  void setEndStmt(Stmt *S) { SubExprs[ENDSTMT] = S; }
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  void setInc(Expr *E) { SubExprs[INC] = reinterpret_cast<Stmt*>(E); }
  void setLoopVarStmt(Stmt *S) { SubExprs[LOOPVAR] = S; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getForLoc() const { return ForLoc; }
  SourceLocation getCoawaitLoc() const { return CoawaitLoc; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  SourceLocation getRParenLoc() const { return RParenLoc; }

  SourceLocation getBeginLoc() const LLVM_READONLY { return ForLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return SubExprs[BODY]->getEndLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CXXForallRangeStmtClass;
  }

  // Iterators
  child_range children() {
    return child_range(&SubExprs[0], &SubExprs[END]);
  }

  const_child_range children() const {
    return const_child_range(&SubExprs[0], &SubExprs[END]);
  }
};

}

#endif
