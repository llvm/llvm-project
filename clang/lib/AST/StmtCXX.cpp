//===--- StmtCXX.cpp - Classes for representing C++ statements ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Stmt class declared in StmtCXX.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtCXX.h"
#include "clang/AST/ExprCXX.h"

#include "clang/AST/ASTContext.h"

using namespace clang;

QualType CXXCatchStmt::getCaughtType() const {
  if (ExceptionDecl)
    return ExceptionDecl->getType();
  return QualType();
}

CXXTryStmt *CXXTryStmt::Create(const ASTContext &C, SourceLocation tryLoc,
                               CompoundStmt *tryBlock,
                               ArrayRef<Stmt *> handlers) {
  const size_t Size = totalSizeToAlloc<Stmt *>(handlers.size() + 1);
  void *Mem = C.Allocate(Size, alignof(CXXTryStmt));
  return new (Mem) CXXTryStmt(tryLoc, tryBlock, handlers);
}

CXXTryStmt *CXXTryStmt::Create(const ASTContext &C, EmptyShell Empty,
                               unsigned numHandlers) {
  const size_t Size = totalSizeToAlloc<Stmt *>(numHandlers + 1);
  void *Mem = C.Allocate(Size, alignof(CXXTryStmt));
  return new (Mem) CXXTryStmt(Empty, numHandlers);
}

CXXTryStmt::CXXTryStmt(SourceLocation tryLoc, CompoundStmt *tryBlock,
                       ArrayRef<Stmt *> handlers)
    : Stmt(CXXTryStmtClass), TryLoc(tryLoc), NumHandlers(handlers.size()) {
  Stmt **Stmts = getStmts();
  Stmts[0] = tryBlock;
  llvm::copy(handlers, Stmts + 1);
}

CXXForRangeStmt::CXXForRangeStmt(Stmt *Init, DeclStmt *Range,
                                 DeclStmt *BeginStmt, DeclStmt *EndStmt,
                                 Expr *Cond, Expr *Inc, DeclStmt *LoopVar,
                                 Stmt *Body, SourceLocation FL,
                                 SourceLocation CAL, SourceLocation CL,
                                 SourceLocation RPL)
    : Stmt(CXXForRangeStmtClass), ForLoc(FL), CoawaitLoc(CAL), ColonLoc(CL),
      RParenLoc(RPL) {
  SubExprs[INIT] = Init;
  SubExprs[RANGE] = Range;
  SubExprs[BEGINSTMT] = BeginStmt;
  SubExprs[ENDSTMT] = EndStmt;
  SubExprs[COND] = Cond;
  SubExprs[INC] = Inc;
  SubExprs[LOOPVAR] = LoopVar;
  SubExprs[BODY] = Body;
}

Expr *CXXForRangeStmt::getRangeInit() {
  DeclStmt *RangeStmt = getRangeStmt();
  VarDecl *RangeDecl = dyn_cast_or_null<VarDecl>(RangeStmt->getSingleDecl());
  assert(RangeDecl && "for-range should have a single var decl");
  return RangeDecl->getInit();
}

const Expr *CXXForRangeStmt::getRangeInit() const {
  return const_cast<CXXForRangeStmt *>(this)->getRangeInit();
}

VarDecl *CXXForRangeStmt::getLoopVariable() {
  Decl *LV = cast<DeclStmt>(getLoopVarStmt())->getSingleDecl();
  assert(LV && "No loop variable in CXXForRangeStmt");
  return cast<VarDecl>(LV);
}

const VarDecl *CXXForRangeStmt::getLoopVariable() const {
  return const_cast<CXXForRangeStmt *>(this)->getLoopVariable();
}

CoroutineBodyStmt *CoroutineBodyStmt::Create(
    const ASTContext &C, CoroutineBodyStmt::CtorArgs const &Args) {
  std::size_t Size = totalSizeToAlloc<Stmt *>(
      CoroutineBodyStmt::FirstParamMove + Args.ParamMoves.size());

  void *Mem = C.Allocate(Size, alignof(CoroutineBodyStmt));
  return new (Mem) CoroutineBodyStmt(Args);
}

CoroutineBodyStmt *CoroutineBodyStmt::Create(const ASTContext &C, EmptyShell,
                                             unsigned NumParams) {
  std::size_t Size = totalSizeToAlloc<Stmt *>(
      CoroutineBodyStmt::FirstParamMove + NumParams);

  void *Mem = C.Allocate(Size, alignof(CoroutineBodyStmt));
  auto *Result = new (Mem) CoroutineBodyStmt(CtorArgs());
  Result->NumParams = NumParams;
  auto *ParamBegin = Result->getStoredStmts() + SubStmt::FirstParamMove;
  std::uninitialized_fill(ParamBegin, ParamBegin + NumParams,
                          static_cast<Stmt *>(nullptr));
  return Result;
}

CoroutineBodyStmt::CoroutineBodyStmt(CoroutineBodyStmt::CtorArgs const &Args)
    : Stmt(CoroutineBodyStmtClass), NumParams(Args.ParamMoves.size()) {
  Stmt **SubStmts = getStoredStmts();
  SubStmts[CoroutineBodyStmt::Body] = Args.Body;
  SubStmts[CoroutineBodyStmt::Promise] = Args.Promise;
  SubStmts[CoroutineBodyStmt::InitSuspend] = Args.InitialSuspend;
  SubStmts[CoroutineBodyStmt::FinalSuspend] = Args.FinalSuspend;
  SubStmts[CoroutineBodyStmt::OnException] = Args.OnException;
  SubStmts[CoroutineBodyStmt::OnFallthrough] = Args.OnFallthrough;
  SubStmts[CoroutineBodyStmt::Allocate] = Args.Allocate;
  SubStmts[CoroutineBodyStmt::Deallocate] = Args.Deallocate;
  SubStmts[CoroutineBodyStmt::ResultDecl] = Args.ResultDecl;
  SubStmts[CoroutineBodyStmt::ReturnValue] = Args.ReturnValue;
  SubStmts[CoroutineBodyStmt::ReturnStmt] = Args.ReturnStmt;
  SubStmts[CoroutineBodyStmt::ReturnStmtOnAllocFailure] =
      Args.ReturnStmtOnAllocFailure;
  llvm::copy(Args.ParamMoves, const_cast<Stmt **>(getParamMoves().data()));
}

CXXExpansionStmt::CXXExpansionStmt(StmtClass SC, EmptyShell Empty)
    : Stmt(SC, Empty) {}

CXXExpansionStmt::CXXExpansionStmt(StmtClass SC, ExpansionStmtDecl *ESD,
                                   Stmt *Init, DeclStmt *ExpansionVar,
                                   SourceLocation ForLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation ColonLoc,
                                   SourceLocation RParenLoc)
    : Stmt(SC), ParentDecl(ESD), ForLoc(ForLoc), LParenLoc(LParenLoc),
      ColonLoc(ColonLoc), RParenLoc(RParenLoc) {
  setInit(Init);
  setExpansionVarStmt(ExpansionVar);
  setBody(nullptr);
}

CXXEnumeratingExpansionStmt::CXXEnumeratingExpansionStmt(EmptyShell Empty)
    : CXXExpansionStmt(CXXEnumeratingExpansionStmtClass, Empty) {}

CXXEnumeratingExpansionStmt::CXXEnumeratingExpansionStmt(
    ExpansionStmtDecl *ESD, Stmt *Init, DeclStmt *ExpansionVar,
    SourceLocation ForLoc, SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation RParenLoc)
    : CXXExpansionStmt(CXXEnumeratingExpansionStmtClass, ESD, Init,
                       ExpansionVar, ForLoc, LParenLoc, ColonLoc, RParenLoc) {}

SourceLocation CXXExpansionStmt::getBeginLoc() const {
  return ParentDecl->getLocation();
}

// FIXME: Copy-pasted from CXXForRangeStmt. Can we convert this into a helper
// function and put it somewhere else maybe?
VarDecl *CXXExpansionStmt::getExpansionVariable() {
  Decl *LV = cast<DeclStmt>(getExpansionVarStmt())->getSingleDecl();
  assert(LV && "No expansion variable in CXXExpansionStmt");
  return cast<VarDecl>(LV);
}

bool CXXExpansionStmt::hasDependentSize() const {
  if (isa<CXXEnumeratingExpansionStmt>(this))
    return cast<CXXExpansionInitListSelectExpr>(
               getExpansionVariable()->getInit())
        ->getRangeExpr()
        ->containsPackExpansion();

  if (auto *Iterating = dyn_cast<CXXIteratingExpansionStmt>(this)) {
    const Expr* Begin = Iterating->getBeginVar()->getInit();
    const Expr* End = Iterating->getBeginVar()->getInit();
    return Begin->isTypeDependent() || Begin->isValueDependent() ||
           End->isTypeDependent() || End->isValueDependent();
  }

  if (isa<CXXDestructuringExpansionStmt>(this))
    return false;

  if (isa<CXXDependentExpansionStmt>(this))
    return true;

  llvm_unreachable("Invalid expansion statement class");
}

CXXIteratingExpansionStmt::CXXIteratingExpansionStmt(EmptyShell Empty)
  : CXXExpansionStmt(CXXIteratingExpansionStmtClass, Empty) {}

CXXIteratingExpansionStmt::CXXIteratingExpansionStmt(
    ExpansionStmtDecl *ESD, Stmt *Init, DeclStmt *ExpansionVar, DeclStmt *Range,
    DeclStmt *Begin, DeclStmt *End, SourceLocation ForLoc,
    SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation RParenLoc)
    : CXXExpansionStmt(CXXIteratingExpansionStmtClass, ESD, Init, ExpansionVar,
                       ForLoc, LParenLoc, ColonLoc, RParenLoc) {
  setRangeVarStmt(Range);
  setBeginVarStmt(Begin);
  setEndVarStmt(End);
}

CXXDestructuringExpansionStmt::CXXDestructuringExpansionStmt(EmptyShell Empty)
    : CXXExpansionStmt(CXXDestructuringExpansionStmtClass, Empty) {}

CXXDestructuringExpansionStmt::CXXDestructuringExpansionStmt(
    ExpansionStmtDecl *ESD, Stmt *Init, DeclStmt *ExpansionVar,
    Stmt *DecompositionDeclStmt, SourceLocation ForLoc,
    SourceLocation LParenLoc, SourceLocation ColonLoc, SourceLocation RParenLoc)
    : CXXExpansionStmt(CXXDestructuringExpansionStmtClass, ESD, Init, ExpansionVar,
                       ForLoc, LParenLoc, ColonLoc, RParenLoc) {
  setDecompositionDeclStmt(DecompositionDeclStmt);
}

DecompositionDecl* CXXDestructuringExpansionStmt::getDecompositionDecl() {
  return cast<DecompositionDecl>(
      cast<DeclStmt>(getDecompositionDeclStmt())->getSingleDecl());
}

CXXDependentExpansionStmt::CXXDependentExpansionStmt(EmptyShell Empty)
  : CXXExpansionStmt(CXXDependentExpansionStmtClass, Empty) {}

CXXDependentExpansionStmt::CXXDependentExpansionStmt(
    ExpansionStmtDecl *ESD, Stmt *Init, DeclStmt *ExpansionVar,
    Expr *ExpansionInitializer, SourceLocation ForLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation RParenLoc)
    : CXXExpansionStmt(CXXDependentExpansionStmtClass, ESD, Init, ExpansionVar,
                       ForLoc, LParenLoc, ColonLoc, RParenLoc) {
  setExpansionInitializer(ExpansionInitializer);
}

CXXExpansionInstantiationStmt::CXXExpansionInstantiationStmt(
    EmptyShell Empty, unsigned NumInstantiations, unsigned NumSharedStmts)
    : Stmt(CXXExpansionInstantiationStmtClass, Empty),
      NumInstantiations(NumInstantiations), NumSharedStmts(NumSharedStmts) {
  assert(NumSharedStmts <= 4 && "might have to allocate more bits for this");
}

CXXExpansionInstantiationStmt::CXXExpansionInstantiationStmt(
    SourceLocation BeginLoc, SourceLocation EndLoc,
    ArrayRef<Stmt *> Instantiations, ArrayRef<Stmt *> SharedStmts,
    bool ShouldApplyLifetimeExtensionToSharedStmts)
    : Stmt(CXXExpansionInstantiationStmtClass), BeginLoc(BeginLoc),
      EndLoc(EndLoc), NumInstantiations(unsigned(Instantiations.size())),
      NumSharedStmts(unsigned(SharedStmts.size())),
      ShouldApplyLifetimeExtensionToSharedStmts(
          ShouldApplyLifetimeExtensionToSharedStmts) {
  assert(NumSharedStmts <= 4 && "might have to allocate more bits for this");
  llvm::uninitialized_copy(Instantiations, getTrailingObjects());
  llvm::uninitialized_copy(SharedStmts,
                           getTrailingObjects() + NumInstantiations);
}

CXXExpansionInstantiationStmt *CXXExpansionInstantiationStmt::Create(
    ASTContext &C, SourceLocation BeginLoc, SourceLocation EndLoc,
    ArrayRef<Stmt *> Instantiations, ArrayRef<Stmt *> SharedStmts,
    bool ShouldApplyLifetimeExtensionToSharedStmts) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Stmt *>(Instantiations.size() + SharedStmts.size()),
      alignof(CXXExpansionInstantiationStmt));
  return new (Mem) CXXExpansionInstantiationStmt(
      BeginLoc, EndLoc, Instantiations, SharedStmts,
      ShouldApplyLifetimeExtensionToSharedStmts);
}

CXXExpansionInstantiationStmt *
CXXExpansionInstantiationStmt::CreateEmpty(ASTContext &C, EmptyShell Empty,
                                           unsigned NumInstantiations,
                                           unsigned NumSharedStmts) {
  void *Mem =
      C.Allocate(totalSizeToAlloc<Stmt *>(NumInstantiations + NumSharedStmts),
                 alignof(CXXExpansionInstantiationStmt));
  return new (Mem)
      CXXExpansionInstantiationStmt(Empty, NumInstantiations, NumSharedStmts);
}
