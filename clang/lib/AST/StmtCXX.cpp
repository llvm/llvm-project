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

CXXExpansionStmtPattern::CXXExpansionStmtPattern(ExpansionStmtKind PatternKind,
                                                 EmptyShell Empty)
    : Stmt(CXXExpansionStmtPatternClass, Empty), PatternKind(PatternKind) {}

CXXExpansionStmtPattern::CXXExpansionStmtPattern(
    ExpansionStmtKind PatternKind, CXXExpansionStmtDecl *ESD, Stmt *Init,
    DeclStmt *ExpansionVar, SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation RParenLoc)
    : Stmt(CXXExpansionStmtPatternClass), PatternKind(PatternKind),
      LParenLoc(LParenLoc), ColonLoc(ColonLoc), RParenLoc(RParenLoc),
      ParentDecl(ESD) {
  setInit(Init);
  setExpansionVarStmt(ExpansionVar);
  setBody(nullptr);
}

template <typename... Args>
CXXExpansionStmtPattern *CXXExpansionStmtPattern::AllocateAndConstruct(
    ASTContext &Context, ExpansionStmtKind Kind, Args &&...Arguments) {
  std::size_t Size = totalSizeToAlloc<Stmt *>(getNumSubStmts(Kind));
  void *Mem = Context.Allocate(Size, alignof(CXXExpansionStmtPattern));
  return new (Mem)
      CXXExpansionStmtPattern(Kind, std::forward<Args>(Arguments)...);
}

CXXExpansionStmtPattern *CXXExpansionStmtPattern::CreateDependent(
    ASTContext &Context, CXXExpansionStmtDecl *ESD, Stmt *Init,
    DeclStmt *ExpansionVar, Expr *ExpansionInitializer,
    SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation RParenLoc) {
  CXXExpansionStmtPattern *Pattern =
      AllocateAndConstruct(Context, ExpansionStmtKind::Dependent, ESD, Init,
                           ExpansionVar, LParenLoc, ColonLoc, RParenLoc);
  Pattern->setExpansionInitializer(ExpansionInitializer);
  return Pattern;
}

CXXExpansionStmtPattern *CXXExpansionStmtPattern::CreateDestructuring(
    ASTContext &Context, CXXExpansionStmtDecl *ESD, Stmt *Init,
    DeclStmt *ExpansionVar, Stmt *DecompositionDeclStmt,
    SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation RParenLoc) {
  CXXExpansionStmtPattern *Pattern =
      AllocateAndConstruct(Context, ExpansionStmtKind::Destructuring, ESD, Init,
                           ExpansionVar, LParenLoc, ColonLoc, RParenLoc);
  Pattern->setDecompositionDeclStmt(DecompositionDeclStmt);
  return Pattern;
}

CXXExpansionStmtPattern *
CXXExpansionStmtPattern::CreateEmpty(ASTContext &Context, EmptyShell Empty,
                                     ExpansionStmtKind Kind) {
  return AllocateAndConstruct(Context, Kind, Empty);
}

CXXExpansionStmtPattern *CXXExpansionStmtPattern::CreateEnumerating(
    ASTContext &Context, CXXExpansionStmtDecl *ESD, Stmt *Init,
    DeclStmt *ExpansionVar, SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation RParenLoc) {
  return AllocateAndConstruct(Context, ExpansionStmtKind::Enumerating, ESD,
                              Init, ExpansionVar, LParenLoc, ColonLoc,
                              RParenLoc);
}

CXXExpansionStmtPattern *CXXExpansionStmtPattern::CreateIterating(
    ASTContext &Context, CXXExpansionStmtDecl *ESD, Stmt *Init,
    DeclStmt *ExpansionVar, DeclStmt *Range, DeclStmt *Begin, DeclStmt *End,
    SourceLocation LParenLoc, SourceLocation ColonLoc,
    SourceLocation RParenLoc) {
  CXXExpansionStmtPattern *Pattern =
      AllocateAndConstruct(Context, ExpansionStmtKind::Iterating, ESD, Init,
                           ExpansionVar, LParenLoc, ColonLoc, RParenLoc);
  Pattern->setRangeVarStmt(Range);
  Pattern->setBeginVarStmt(Begin);
  Pattern->setEndVarStmt(End);
  return Pattern;
}

SourceLocation CXXExpansionStmtPattern::getBeginLoc() const {
  return ParentDecl->getLocation();
}

DecompositionDecl *
CXXExpansionStmtPattern::getDecompositionDecl() {
  assert(isDestructuring());
  return cast<DecompositionDecl>(
      cast<DeclStmt>(getDecompositionDeclStmt())->getSingleDecl());
}

VarDecl *CXXExpansionStmtPattern::getExpansionVariable() {
  Decl *LV = cast<DeclStmt>(getExpansionVarStmt())->getSingleDecl();
  assert(LV && "No expansion variable in CXXExpansionStmtPattern");
  return cast<VarDecl>(LV);
}

unsigned
CXXExpansionStmtPattern::getNumSubStmts(ExpansionStmtKind PatternKind) {
  switch (PatternKind) {
  case ExpansionStmtKind::Enumerating:
    return COUNT_Enumerating;
  case ExpansionStmtKind::Iterating:
    return COUNT_Iterating;
  case ExpansionStmtKind::Destructuring:
    return COUNT_Destructuring;
  case ExpansionStmtKind::Dependent:
    return COUNT_Dependent;
  }

  llvm_unreachable("invalid pattern kind");
}

CXXExpansionStmtInstantiation::CXXExpansionStmtInstantiation(
    EmptyShell Empty, unsigned NumInstantiations, unsigned NumSharedStmts)
    : Stmt(CXXExpansionStmtInstantiationClass, Empty),
      NumInstantiations(NumInstantiations), NumSharedStmts(NumSharedStmts) {
  assert(NumSharedStmts <= 4 && "might have to allocate more bits for this");
}

CXXExpansionStmtInstantiation::CXXExpansionStmtInstantiation(
    SourceLocation BeginLoc, SourceLocation EndLoc,
    ArrayRef<Stmt *> Instantiations, ArrayRef<Stmt *> SharedStmts,
    bool ShouldApplyLifetimeExtensionToSharedStmts)
    : Stmt(CXXExpansionStmtInstantiationClass), BeginLoc(BeginLoc),
      EndLoc(EndLoc), NumInstantiations(unsigned(Instantiations.size())),
      NumSharedStmts(unsigned(SharedStmts.size())),
      ShouldApplyLifetimeExtensionToSharedStmts(
          ShouldApplyLifetimeExtensionToSharedStmts) {
  assert(NumSharedStmts <= 4 && "might have to allocate more bits for this");
  llvm::uninitialized_copy(Instantiations, getTrailingObjects());
  llvm::uninitialized_copy(SharedStmts,
                           getTrailingObjects() + NumInstantiations);
}

CXXExpansionStmtInstantiation *CXXExpansionStmtInstantiation::Create(
    ASTContext &C, SourceLocation BeginLoc, SourceLocation EndLoc,
    ArrayRef<Stmt *> Instantiations, ArrayRef<Stmt *> SharedStmts,
    bool ShouldApplyLifetimeExtensionToSharedStmts) {
  void *Mem = C.Allocate(
      totalSizeToAlloc<Stmt *>(Instantiations.size() + SharedStmts.size()),
      alignof(CXXExpansionStmtInstantiation));
  return new (Mem) CXXExpansionStmtInstantiation(
      BeginLoc, EndLoc, Instantiations, SharedStmts,
      ShouldApplyLifetimeExtensionToSharedStmts);
}

CXXExpansionStmtInstantiation *
CXXExpansionStmtInstantiation::CreateEmpty(ASTContext &C, EmptyShell Empty,
                                           unsigned NumInstantiations,
                                           unsigned NumSharedStmts) {
  void *Mem =
      C.Allocate(totalSizeToAlloc<Stmt *>(NumInstantiations + NumSharedStmts),
                 alignof(CXXExpansionStmtInstantiation));
  return new (Mem)
      CXXExpansionStmtInstantiation(Empty, NumInstantiations, NumSharedStmts);
}
