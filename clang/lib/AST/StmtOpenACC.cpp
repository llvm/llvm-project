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
#include "clang/AST/ExprCXX.h"
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
  std::uninitialized_value_construct_n(getTrailingObjects(), NumClauses);
  setClauseList(getTrailingObjects(NumClauses));
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
  llvm::uninitialized_copy(Clauses, getTrailingObjects());

  setClauseList(getTrailingObjects(Clauses.size()));
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

  assert(!llvm::is_contained(QueueIdExprs, nullptr));

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

OpenACCAtomicConstruct *
OpenACCAtomicConstruct::CreateEmpty(const ASTContext &C, unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCAtomicConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCAtomicConstruct(NumClauses);
  return Inst;
}

OpenACCAtomicConstruct *OpenACCAtomicConstruct::Create(
    const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
    OpenACCAtomicKind AtKind, SourceLocation End,
    ArrayRef<const OpenACCClause *> Clauses, Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(
      OpenACCAtomicConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst = new (Mem) OpenACCAtomicConstruct(Start, DirectiveLoc, AtKind,
                                                End, Clauses, AssociatedStmt);
  return Inst;
}

static std::pair<const Expr *, const Expr *> getBinaryOpArgs(const Expr *Op) {
  if (const auto *BO = dyn_cast<BinaryOperator>(Op)) {
    assert(BO->isAssignmentOp());
    return {BO->getLHS(), BO->getRHS()};
  }

  const auto *OO = cast<CXXOperatorCallExpr>(Op);
  assert(OO->isAssignmentOp());
  return {OO->getArg(0), OO->getArg(1)};
}

static std::pair<bool, const Expr *> getUnaryOpArgs(const Expr *Op) {
  if (const auto *UO = dyn_cast<UnaryOperator>(Op))
    return {true, UO->getSubExpr()};

  if (const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(Op)) {
    // Post-inc/dec have a second unused argument to differentiate it, so we
    // accept -- or ++ as unary, or any operator call with only 1 arg.
    if (OpCall->getNumArgs() == 1 || OpCall->getOperator() != OO_PlusPlus ||
        OpCall->getOperator() != OO_MinusMinus)
      return {true, OpCall->getArg(0)};
  }

  return {false, nullptr};
}

const OpenACCAtomicConstruct::StmtInfo
OpenACCAtomicConstruct::getAssociatedStmtInfo() const {
  // This ends up being a vastly simplified version of SemaOpenACCAtomic, since
  // it doesn't have to worry about erroring out, but we should do a lot of
  // asserts to ensure we don't get off into the weeds.
  assert(getAssociatedStmt() && "invalid associated stmt?");

  const Expr *AssocStmt = cast<const Expr>(getAssociatedStmt());
  switch (AtomicKind) {
  case OpenACCAtomicKind::Capture:
    assert(false && "Only 'read'/'write'/'update' have been implemented here");
    return {};
  case OpenACCAtomicKind::Read: {
    // Read only supports the format 'v = x'; where both sides are a scalar
    // expression. This can come in 2 forms; BinaryOperator or
    // CXXOperatorCallExpr (rarely).
    std::pair<const Expr *, const Expr *> BinaryArgs =
        getBinaryOpArgs(AssocStmt);
    // We want the L-value for each side, so we ignore implicit casts.
    return {BinaryArgs.first->IgnoreImpCasts(),
            BinaryArgs.second->IgnoreImpCasts(), /*expr=*/nullptr};
  }
  case OpenACCAtomicKind::Write: {
    // Write supports only the format 'x = expr', where the expression is scalar
    // type, and 'x' is a scalar l value. As above, this can come in 2 forms;
    // Binary Operator or CXXOperatorCallExpr.
    std::pair<const Expr *, const Expr *> BinaryArgs =
        getBinaryOpArgs(AssocStmt);
    // We want the L-value for ONLY the X side, so we ignore implicit casts. For
    // the right side (the expr), we emit it as an r-value so we need to
    // maintain implicit casts.
    return {/*v=*/nullptr, BinaryArgs.first->IgnoreImpCasts(),
            BinaryArgs.second};
  }
  case OpenACCAtomicKind::None:
  case OpenACCAtomicKind::Update: {
    std::pair<bool, const Expr *> UnaryArgs = getUnaryOpArgs(AssocStmt);
    if (UnaryArgs.first)
      return {/*v=*/nullptr, UnaryArgs.second->IgnoreImpCasts(),
              /*expr=*/nullptr};

    std::pair<const Expr *, const Expr *> BinaryArgs =
        getBinaryOpArgs(AssocStmt);
    // For binary args, we just store the RHS as an expression (in the
    // expression slot), since the codegen just wants the whole thing for a
    // recipe.
    return {/*v=*/nullptr, BinaryArgs.first->IgnoreImpCasts(),
            BinaryArgs.second};
  }
  }

  llvm_unreachable("unknown OpenACC atomic kind");
}

OpenACCCacheConstruct *OpenACCCacheConstruct::CreateEmpty(const ASTContext &C,
                                                          unsigned NumVars) {
  void *Mem =
      C.Allocate(OpenACCCacheConstruct::totalSizeToAlloc<Expr *>(NumVars));
  auto *Inst = new (Mem) OpenACCCacheConstruct(NumVars);
  return Inst;
}

OpenACCCacheConstruct *OpenACCCacheConstruct::Create(
    const ASTContext &C, SourceLocation Start, SourceLocation DirectiveLoc,
    SourceLocation LParenLoc, SourceLocation ReadOnlyLoc,
    ArrayRef<Expr *> VarList, SourceLocation RParenLoc, SourceLocation End) {
  void *Mem = C.Allocate(
      OpenACCCacheConstruct::totalSizeToAlloc<Expr *>(VarList.size()));
  auto *Inst = new (Mem) OpenACCCacheConstruct(
      Start, DirectiveLoc, LParenLoc, ReadOnlyLoc, VarList, RParenLoc, End);
  return Inst;
}
