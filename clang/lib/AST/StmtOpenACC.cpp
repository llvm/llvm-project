//===--- StmtOpenACC.cpp - Classes for OpenACC Constructs -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclasses of Stmt class declared in StmtOpenACC.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtOpenACC.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtCXX.h"
using namespace clang;

OpenACCComputeConstruct *
OpenACCComputeConstruct::CreateEmpty(const ASTContext &C, unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCComputeConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCComputeConstruct(NumClauses);
  return Inst;
}

OpenACCComputeConstruct *OpenACCComputeConstruct::Create(
    const ASTContext &C, OpenACCDirectiveKind K, SourceLocation BeginLoc,
    SourceLocation DirLoc, SourceLocation EndLoc,
    ArrayRef<const OpenACCClause *> Clauses, Stmt *StructuredBlock) {
  void *Mem = C.Allocate(
      OpenACCComputeConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst = new (Mem) OpenACCComputeConstruct(K, BeginLoc, DirLoc, EndLoc,
                                                 Clauses, StructuredBlock);
  return Inst;
}

OpenACCLoopConstruct::OpenACCLoopConstruct(unsigned NumClauses)
    : OpenACCAssociatedStmtConstruct(
          OpenACCLoopConstructClass, OpenACCDirectiveKind::Loop,
          SourceLocation{}, SourceLocation{}, SourceLocation{},
          /*AssociatedStmt=*/nullptr) {
  std::uninitialized_value_construct(
      getTrailingObjects<const OpenACCClause *>(),
      getTrailingObjects<const OpenACCClause *>() + NumClauses);
  setClauseList(
      MutableArrayRef(getTrailingObjects<const OpenACCClause *>(), NumClauses));
}

OpenACCLoopConstruct::OpenACCLoopConstruct(
    OpenACCDirectiveKind ParentKind, SourceLocation Start,
    SourceLocation DirLoc, SourceLocation End,
    ArrayRef<const OpenACCClause *> Clauses, Stmt *Loop)
    : OpenACCAssociatedStmtConstruct(OpenACCLoopConstructClass,
                                     OpenACCDirectiveKind::Loop, Start, DirLoc,
                                     End, Loop),
      ParentComputeConstructKind(ParentKind) {
  // accept 'nullptr' for the loop. This is diagnosed somewhere, but this gives
  // us some level of AST fidelity in the error case.
  assert((Loop == nullptr || isa<ForStmt, CXXForRangeStmt>(Loop)) &&
         "Associated Loop not a for loop?");
  // Initialize the trailing storage.
  std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                          getTrailingObjects<const OpenACCClause *>());

  setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                Clauses.size()));
}

OpenACCLoopConstruct *OpenACCLoopConstruct::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses) {
  void *Mem =
      C.Allocate(OpenACCLoopConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCLoopConstruct(NumClauses);
  return Inst;
}

OpenACCLoopConstruct *OpenACCLoopConstruct::Create(
    const ASTContext &C, OpenACCDirectiveKind ParentKind,
    SourceLocation BeginLoc, SourceLocation DirLoc, SourceLocation EndLoc,
    ArrayRef<const OpenACCClause *> Clauses, Stmt *Loop) {
  void *Mem =
      C.Allocate(OpenACCLoopConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst = new (Mem)
      OpenACCLoopConstruct(ParentKind, BeginLoc, DirLoc, EndLoc, Clauses, Loop);
  return Inst;
}

OpenACCCombinedConstruct *
OpenACCCombinedConstruct::CreateEmpty(const ASTContext &C,
                                      unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCCombinedConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCCombinedConstruct(NumClauses);
  return Inst;
}

OpenACCCombinedConstruct *OpenACCCombinedConstruct::Create(
    const ASTContext &C, OpenACCDirectiveKind DK, SourceLocation BeginLoc,
    SourceLocation DirLoc, SourceLocation EndLoc,
    ArrayRef<const OpenACCClause *> Clauses, Stmt *Loop) {
  void *Mem = C.Allocate(
      OpenACCCombinedConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst = new (Mem)
      OpenACCCombinedConstruct(DK, BeginLoc, DirLoc, EndLoc, Clauses, Loop);
  return Inst;
}

OpenACCDataConstruct *OpenACCDataConstruct::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses) {
  void *Mem =
      C.Allocate(OpenACCDataConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCDataConstruct(NumClauses);
  return Inst;
}

OpenACCDataConstruct *
OpenACCDataConstruct::Create(const ASTContext &C, SourceLocation Start,
                             SourceLocation DirectiveLoc, SourceLocation End,
                             ArrayRef<const OpenACCClause *> Clauses,
                             Stmt *StructuredBlock) {
  void *Mem =
      C.Allocate(OpenACCDataConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst = new (Mem)
      OpenACCDataConstruct(Start, DirectiveLoc, End, Clauses, StructuredBlock);
  return Inst;
}

OpenACCEnterDataConstruct *
OpenACCEnterDataConstruct::CreateEmpty(const ASTContext &C,
                                       unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCEnterDataConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCEnterDataConstruct(NumClauses);
  return Inst;
}

OpenACCEnterDataConstruct *OpenACCEnterDataConstruct::Create(
    const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
    SourceLocation End, ArrayRef<const OpenACCClause *> Clauses) {
  void *Mem = C.Allocate(
      OpenACCEnterDataConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst =
      new (Mem) OpenACCEnterDataConstruct(Start, DirectiveLoc, End, Clauses);
  return Inst;
}

OpenACCExitDataConstruct *
OpenACCExitDataConstruct::CreateEmpty(const ASTContext &C,
                                      unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCExitDataConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCExitDataConstruct(NumClauses);
  return Inst;
}

OpenACCExitDataConstruct *OpenACCExitDataConstruct::Create(
    const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
    SourceLocation End, ArrayRef<const OpenACCClause *> Clauses) {
  void *Mem = C.Allocate(
      OpenACCExitDataConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst =
      new (Mem) OpenACCExitDataConstruct(Start, DirectiveLoc, End, Clauses);
  return Inst;
}

OpenACCHostDataConstruct *
OpenACCHostDataConstruct::CreateEmpty(const ASTContext &C,
                                      unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCHostDataConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCHostDataConstruct(NumClauses);
  return Inst;
}

OpenACCHostDataConstruct *OpenACCHostDataConstruct::Create(
    const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
    SourceLocation End, ArrayRef<const OpenACCClause *> Clauses,
    Stmt *StructuredBlock) {
  void *Mem = C.Allocate(
      OpenACCHostDataConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst = new (Mem) OpenACCHostDataConstruct(Start, DirectiveLoc, End,
                                                  Clauses, StructuredBlock);
  return Inst;
}

OpenACCWaitConstruct *OpenACCWaitConstruct::CreateEmpty(const ASTContext &C,
                                                        unsigned NumExprs,
                                                        unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCWaitConstruct::totalSizeToAlloc<Expr *, OpenACCClause *>(
          NumExprs, NumClauses));

  auto *Inst = new (Mem) OpenACCWaitConstruct(NumExprs, NumClauses);
  return Inst;
}

OpenACCWaitConstruct *OpenACCWaitConstruct::Create(
    const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
    SourceLocation LParenLoc, Expr *DevNumExpr, SourceLocation QueuesLoc,
    ArrayRef<Expr *> QueueIdExprs, SourceLocation RParenLoc, SourceLocation End,
    ArrayRef<const OpenACCClause *> Clauses) {

  assert(llvm::all_of(QueueIdExprs, [](Expr *E) { return E != nullptr; }));

  void *Mem = C.Allocate(
      OpenACCWaitConstruct::totalSizeToAlloc<Expr *, OpenACCClause *>(
          QueueIdExprs.size() + 1, Clauses.size()));

  auto *Inst = new (Mem)
      OpenACCWaitConstruct(Start, DirectiveLoc, LParenLoc, DevNumExpr,
                           QueuesLoc, QueueIdExprs, RParenLoc, End, Clauses);
  return Inst;
}
OpenACCInitConstruct *OpenACCInitConstruct::CreateEmpty(const ASTContext &C,
                                                        unsigned NumClauses) {
  void *Mem =
      C.Allocate(OpenACCInitConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCInitConstruct(NumClauses);
  return Inst;
}

OpenACCInitConstruct *
OpenACCInitConstruct::Create(const ASTContext &C, SourceLocation Start,
                             SourceLocation DirectiveLoc, SourceLocation End,
                             ArrayRef<const OpenACCClause *> Clauses) {
  void *Mem =
      C.Allocate(OpenACCInitConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst =
      new (Mem) OpenACCInitConstruct(Start, DirectiveLoc, End, Clauses);
  return Inst;
}
OpenACCShutdownConstruct *
OpenACCShutdownConstruct::CreateEmpty(const ASTContext &C,
                                      unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCShutdownConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCShutdownConstruct(NumClauses);
  return Inst;
}

OpenACCShutdownConstruct *OpenACCShutdownConstruct::Create(
    const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
    SourceLocation End, ArrayRef<const OpenACCClause *> Clauses) {
  void *Mem = C.Allocate(
      OpenACCShutdownConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst =
      new (Mem) OpenACCShutdownConstruct(Start, DirectiveLoc, End, Clauses);
  return Inst;
}

OpenACCSetConstruct *OpenACCSetConstruct::CreateEmpty(const ASTContext &C,
                                                      unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCSetConstruct::totalSizeToAlloc<const OpenACCClause *>(NumClauses));
  auto *Inst = new (Mem) OpenACCSetConstruct(NumClauses);
  return Inst;
}

OpenACCSetConstruct *
OpenACCSetConstruct::Create(const ASTContext &C, SourceLocation Start,
                            SourceLocation DirectiveLoc, SourceLocation End,
                            ArrayRef<const OpenACCClause *> Clauses) {
  void *Mem =
      C.Allocate(OpenACCSetConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst = new (Mem) OpenACCSetConstruct(Start, DirectiveLoc, End, Clauses);
  return Inst;
}

OpenACCUpdateConstruct *
OpenACCUpdateConstruct::CreateEmpty(const ASTContext &C, unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCUpdateConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCUpdateConstruct(NumClauses);
  return Inst;
}

OpenACCUpdateConstruct *
OpenACCUpdateConstruct::Create(const ASTContext &C, SourceLocation Start,
                               SourceLocation DirectiveLoc, SourceLocation End,
                               ArrayRef<const OpenACCClause *> Clauses) {
  void *Mem = C.Allocate(
      OpenACCUpdateConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst =
      new (Mem) OpenACCUpdateConstruct(Start, DirectiveLoc, End, Clauses);
  return Inst;
}
