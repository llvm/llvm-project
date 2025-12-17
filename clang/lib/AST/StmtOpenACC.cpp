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

static std::optional<std::pair<const Expr *, const Expr *>>
getBinaryAssignOpArgs(const Expr *Op, bool &IsCompoundAssign) {
  if (const auto *BO = dyn_cast<BinaryOperator>(Op)) {
    if (!BO->isAssignmentOp())
      return std::nullopt;
    IsCompoundAssign = BO->isCompoundAssignmentOp();
    return std::pair<const Expr *, const Expr *>(BO->getLHS(), BO->getRHS());
  }

  if (const auto *OO = dyn_cast<CXXOperatorCallExpr>(Op)) {
    if (!OO->isAssignmentOp())
      return std::nullopt;
    IsCompoundAssign = OO->getOperator() != OO_Equal;
    return std::pair<const Expr *, const Expr *>(OO->getArg(0), OO->getArg(1));
  }
  return std::nullopt;
}
static std::optional<std::pair<const Expr *, const Expr *>>
getBinaryAssignOpArgs(const Expr *Op) {
  bool IsCompoundAssign;
  return getBinaryAssignOpArgs(Op, IsCompoundAssign);
}

static std::optional<std::pair<const Expr *, bool>>
getUnaryOpArgs(const Expr *Op) {
  if (const auto *UO = dyn_cast<UnaryOperator>(Op))
    return {{UO->getSubExpr(), UO->isPostfix()}};

  if (const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(Op)) {
    // Post-inc/dec have a second unused argument to differentiate it, so we
    // accept -- or ++ as unary, or any operator call with only 1 arg.
    if (OpCall->getNumArgs() == 1 || OpCall->getOperator() == OO_PlusPlus ||
        OpCall->getOperator() == OO_MinusMinus)
      return {{OpCall->getArg(0), /*IsPostfix=*/OpCall->getNumArgs() == 1}};
  }

  return std::nullopt;
}

// Read is of the form `v = x;`, where both sides are scalar L-values. This is a
// BinaryOperator or CXXOperatorCallExpr.
static std::optional<OpenACCAtomicConstruct::SingleStmtInfo>
getReadStmtInfo(const Expr *E, bool ForAtomicComputeSingleStmt = false) {
  std::optional<std::pair<const Expr *, const Expr *>> BinaryArgs =
      getBinaryAssignOpArgs(E);

  if (!BinaryArgs)
    return std::nullopt;

  // We want the L-value for each side, so we ignore implicit casts.
  auto Res = OpenACCAtomicConstruct::SingleStmtInfo::createRead(
      E, BinaryArgs->first->IgnoreImpCasts(),
      BinaryArgs->second->IgnoreImpCasts());

  // The atomic compute single-stmt variant has to do a 'fixup' step for the 'X'
  // value, since it is dependent on the RHS.  So if we're in that version, we
  // skip the checks on X.
  if ((!ForAtomicComputeSingleStmt &&
       (!Res.X->isLValue() || !Res.X->getType()->isScalarType())) ||
      !Res.V->isLValue() || !Res.V->getType()->isScalarType())
    return std::nullopt;

  return Res;
}

// Write supports only the format 'x = expr', where the expression is scalar
// type, and 'x' is a scalar l value. As above, this can come in 2 forms;
// Binary Operator or CXXOperatorCallExpr.
static std::optional<OpenACCAtomicConstruct::SingleStmtInfo>
getWriteStmtInfo(const Expr *E) {
  std::optional<std::pair<const Expr *, const Expr *>> BinaryArgs =
      getBinaryAssignOpArgs(E);
  if (!BinaryArgs)
    return std::nullopt;
  // We want the L-value for ONLY the X side, so we ignore implicit casts. For
  // the right side (the expr), we emit it as an r-value so we need to
  // maintain implicit casts.
  auto Res = OpenACCAtomicConstruct::SingleStmtInfo::createWrite(
      E, BinaryArgs->first->IgnoreImpCasts(), BinaryArgs->second);

  if (!Res.X->isLValue() || !Res.X->getType()->isScalarType())
    return std::nullopt;
  return Res;
}

static std::optional<OpenACCAtomicConstruct::SingleStmtInfo>
getUpdateStmtInfo(const Expr *E) {
  std::optional<std::pair<const Expr *, bool>> UnaryArgs = getUnaryOpArgs(E);
  if (UnaryArgs) {
    auto Res = OpenACCAtomicConstruct::SingleStmtInfo::createUpdate(
        E, UnaryArgs->first->IgnoreImpCasts(), UnaryArgs->second);

    if (!Res.X->isLValue() || !Res.X->getType()->isScalarType())
      return std::nullopt;

    return Res;
  }

  bool IsRHSCompoundAssign = false;
  std::optional<std::pair<const Expr *, const Expr *>> BinaryArgs =
      getBinaryAssignOpArgs(E, IsRHSCompoundAssign);
  if (!BinaryArgs)
    return std::nullopt;

  auto Res = OpenACCAtomicConstruct::SingleStmtInfo::createUpdate(
      E, BinaryArgs->first->IgnoreImpCasts(), /*PostFixIncDec=*/false);

  if (!Res.X->isLValue() || !Res.X->getType()->isScalarType())
    return std::nullopt;

  // 'update' has to be either a compound-assignment operation, or
  // assignment-to-a-binary-op. Return nullopt if these are not the case.
  // If we are already compound-assign, we're done!
  if (IsRHSCompoundAssign)
    return Res;

  // else we have to check that we have a binary operator.
  const Expr *RHS = BinaryArgs->second->IgnoreImpCasts();

  if (isa<BinaryOperator>(RHS)) {
    return Res;
  } else if (const auto *OO = dyn_cast<CXXOperatorCallExpr>(RHS)) {
    if (OO->isInfixBinaryOp())
      return Res;
  }

  return std::nullopt;
}

/// The statement associated with an atomic capture comes in 1 of two forms: A
/// compound statement containing two statements, or a single statement.  In
/// either case, the compound/single statement is decomposed into 2 separate
/// operations, eihter a read/write, read/update, or update/read.  This function
/// figures out that information in the form listed in the standard (filling in
/// V, X, or Expr) for each of these operations.
static OpenACCAtomicConstruct::StmtInfo
getCaptureStmtInfo(const Stmt *AssocStmt) {

  if (const auto *CmpdStmt = dyn_cast<CompoundStmt>(AssocStmt)) {
    // We checked during Sema to ensure we only have 2 statements here, and
    // that both are expressions, we can look at these to see what the valid
    // options are.
    const Expr *Stmt1 = cast<Expr>(*CmpdStmt->body().begin())->IgnoreImpCasts();
    const Expr *Stmt2 =
        cast<Expr>(*(CmpdStmt->body().begin() + 1))->IgnoreImpCasts();

    // The compound statement form allows read/write, read/update, or
    // update/read. First we get the information for a 'Read' to see if this is
    // one of the former two.
    std::optional<OpenACCAtomicConstruct::SingleStmtInfo> Read =
        getReadStmtInfo(Stmt1);

    if (Read) {
      // READ : WRITE
      // v = x; x = expr
      // READ : UPDATE
      // v = x; x binop = expr
      // v = x; x = x binop expr
      // v = x; x = expr binop x
      // v = x; x++
      // v = x; ++x
      // v = x; x--
      // v = x; --x
      std::optional<OpenACCAtomicConstruct::SingleStmtInfo> Update =
          getUpdateStmtInfo(Stmt2);
      // Since we already know the first operation is a read, the second is
      // either an update, which we check, or a write, which we can assume next.
      if (Update)
        return OpenACCAtomicConstruct::StmtInfo::createReadUpdate(*Read,
                                                                  *Update);

      std::optional<OpenACCAtomicConstruct::SingleStmtInfo> Write =
          getWriteStmtInfo(Stmt2);
      return OpenACCAtomicConstruct::StmtInfo::createReadWrite(*Read, *Write);
    }
    // UPDATE: READ
    // x binop = expr; v = x
    // x = x binop expr; v = x
    // x = expr binop x ; v = x
    // ++ x; v = x
    // x++; v = x
    // --x; v = x
    // x--; v = x
    // Otherwise, it is one of the above forms for update/read.
    std::optional<OpenACCAtomicConstruct::SingleStmtInfo> Update =
        getUpdateStmtInfo(Stmt1);
    Read = getReadStmtInfo(Stmt2);

    return OpenACCAtomicConstruct::StmtInfo::createUpdateRead(*Update, *Read);
  } else {
    // All of the forms that can be done in a single line fall into 2
    // categories: update/read, or read/update. The special cases are the
    // postfix unary operators, which we have to make sure we do the 'read'
    // first.  However, we still parse these as the RHS first, so we have a
    // 'reversing' step. READ: UPDATE v = x++; v = x--; UPDATE: READ v = ++x; v
    // = --x; v = x binop=expr v = x = x binop expr v = x = expr binop x

    const Expr *E = cast<const Expr>(AssocStmt);

    std::optional<OpenACCAtomicConstruct::SingleStmtInfo> Read =
        getReadStmtInfo(E, /*ForAtomicComputeSingleStmt=*/true);
    std::optional<OpenACCAtomicConstruct::SingleStmtInfo> Update =
        getUpdateStmtInfo(Read->X);

    // Fixup this, since the 'X' for the read is the result after write, but is
    // the same value as the LHS-most variable of the update(its X).
    Read->X = Update->X;

    // Postfix is a read FIRST, then an update.
    if (Update->IsPostfixIncDec)
      return OpenACCAtomicConstruct::StmtInfo::createReadUpdate(*Read, *Update);

    return OpenACCAtomicConstruct::StmtInfo::createUpdateRead(*Update, *Read);
  }
  return {};
}

const OpenACCAtomicConstruct::StmtInfo
OpenACCAtomicConstruct::getAssociatedStmtInfo() const {
  // This ends up being a vastly simplified version of SemaOpenACCAtomic, since
  // it doesn't have to worry about erroring out, but we should do a lot of
  // asserts to ensure we don't get off into the weeds.
  assert(getAssociatedStmt() && "invalid associated stmt?");

  switch (AtomicKind) {
  case OpenACCAtomicKind::Read:
    return OpenACCAtomicConstruct::StmtInfo{
        OpenACCAtomicConstruct::StmtInfo::StmtForm::Read,
        *getReadStmtInfo(cast<const Expr>(getAssociatedStmt())),
        OpenACCAtomicConstruct::SingleStmtInfo::Empty()};

  case OpenACCAtomicKind::Write:
    return OpenACCAtomicConstruct::StmtInfo{
        OpenACCAtomicConstruct::StmtInfo::StmtForm::Write,
        *getWriteStmtInfo(cast<const Expr>(getAssociatedStmt())),
        OpenACCAtomicConstruct::SingleStmtInfo::Empty()};

  case OpenACCAtomicKind::None:
  case OpenACCAtomicKind::Update:
    return OpenACCAtomicConstruct::StmtInfo{
        OpenACCAtomicConstruct::StmtInfo::StmtForm::Update,
        *getUpdateStmtInfo(cast<const Expr>(getAssociatedStmt())),
        OpenACCAtomicConstruct::SingleStmtInfo::Empty()};

  case OpenACCAtomicKind::Capture:
    return getCaptureStmtInfo(getAssociatedStmt());
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
