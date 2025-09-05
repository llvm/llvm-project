//===---- OpenACCClause.cpp - Classes for OpenACC Clauses  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclasses of the OpenACCClause class declared in
// OpenACCClause.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/OpenACCClause.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"

using namespace clang;

bool OpenACCClauseWithParams::classof(const OpenACCClause *C) {
  return OpenACCDeviceTypeClause::classof(C) ||
         OpenACCClauseWithCondition::classof(C) ||
         OpenACCBindClause::classof(C) || OpenACCClauseWithExprs::classof(C) ||
         OpenACCSelfClause::classof(C);
}
bool OpenACCClauseWithExprs::classof(const OpenACCClause *C) {
  return OpenACCWaitClause::classof(C) || OpenACCNumGangsClause::classof(C) ||
         OpenACCTileClause::classof(C) ||
         OpenACCClauseWithSingleIntExpr::classof(C) ||
         OpenACCGangClause::classof(C) || OpenACCClauseWithVarList::classof(C);
}
bool OpenACCClauseWithVarList::classof(const OpenACCClause *C) {
  return OpenACCPrivateClause::classof(C) ||
         OpenACCFirstPrivateClause::classof(C) ||
         OpenACCDevicePtrClause::classof(C) ||
         OpenACCDeleteClause::classof(C) ||
         OpenACCUseDeviceClause::classof(C) ||
         OpenACCDetachClause::classof(C) || OpenACCAttachClause::classof(C) ||
         OpenACCNoCreateClause::classof(C) ||
         OpenACCPresentClause::classof(C) || OpenACCCopyClause::classof(C) ||
         OpenACCCopyInClause::classof(C) || OpenACCCopyOutClause::classof(C) ||
         OpenACCReductionClause::classof(C) ||
         OpenACCCreateClause::classof(C) || OpenACCDeviceClause::classof(C) ||
         OpenACCLinkClause::classof(C) ||
         OpenACCDeviceResidentClause::classof(C) ||
         OpenACCHostClause::classof(C);
}
bool OpenACCClauseWithCondition::classof(const OpenACCClause *C) {
  return OpenACCIfClause::classof(C);
}
bool OpenACCClauseWithSingleIntExpr::classof(const OpenACCClause *C) {
  return OpenACCNumWorkersClause::classof(C) ||
         OpenACCVectorLengthClause::classof(C) ||
         OpenACCDeviceNumClause::classof(C) ||
         OpenACCDefaultAsyncClause::classof(C) ||
         OpenACCVectorClause::classof(C) || OpenACCWorkerClause::classof(C) ||
         OpenACCCollapseClause::classof(C) || OpenACCAsyncClause::classof(C);
}
OpenACCDefaultClause *OpenACCDefaultClause::Create(const ASTContext &C,
                                                   OpenACCDefaultClauseKind K,
                                                   SourceLocation BeginLoc,
                                                   SourceLocation LParenLoc,
                                                   SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(sizeof(OpenACCDefaultClause), alignof(OpenACCDefaultClause));

  return new (Mem) OpenACCDefaultClause(K, BeginLoc, LParenLoc, EndLoc);
}

OpenACCIfClause *OpenACCIfClause::Create(const ASTContext &C,
                                         SourceLocation BeginLoc,
                                         SourceLocation LParenLoc,
                                         Expr *ConditionExpr,
                                         SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCIfClause), alignof(OpenACCIfClause));
  return new (Mem) OpenACCIfClause(BeginLoc, LParenLoc, ConditionExpr, EndLoc);
}

OpenACCIfClause::OpenACCIfClause(SourceLocation BeginLoc,
                                 SourceLocation LParenLoc, Expr *ConditionExpr,
                                 SourceLocation EndLoc)
    : OpenACCClauseWithCondition(OpenACCClauseKind::If, BeginLoc, LParenLoc,
                                 ConditionExpr, EndLoc) {
  assert(ConditionExpr && "if clause requires condition expr");
  assert((ConditionExpr->isInstantiationDependent() ||
          ConditionExpr->getType()->isScalarType()) &&
         "Condition expression type not scalar/dependent");
}

OpenACCSelfClause *OpenACCSelfClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             Expr *ConditionExpr,
                                             SourceLocation EndLoc) {
  void *Mem = C.Allocate(OpenACCSelfClause::totalSizeToAlloc<Expr *>(1));
  return new (Mem)
      OpenACCSelfClause(BeginLoc, LParenLoc, ConditionExpr, EndLoc);
}

OpenACCSelfClause *OpenACCSelfClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             ArrayRef<Expr *> VarList,
                                             SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCSelfClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCSelfClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCSelfClause::OpenACCSelfClause(SourceLocation BeginLoc,
                                     SourceLocation LParenLoc,
                                     ArrayRef<Expr *> VarList,
                                     SourceLocation EndLoc)
    : OpenACCClauseWithParams(OpenACCClauseKind::Self, BeginLoc, LParenLoc,
                              EndLoc),
      HasConditionExpr(std::nullopt), NumExprs(VarList.size()) {
  llvm::uninitialized_copy(VarList, getTrailingObjects());
}

OpenACCSelfClause::OpenACCSelfClause(SourceLocation BeginLoc,
                                     SourceLocation LParenLoc,
                                     Expr *ConditionExpr, SourceLocation EndLoc)
    : OpenACCClauseWithParams(OpenACCClauseKind::Self, BeginLoc, LParenLoc,
                              EndLoc),
      HasConditionExpr(ConditionExpr != nullptr), NumExprs(1) {
  assert((!ConditionExpr || ConditionExpr->isInstantiationDependent() ||
          ConditionExpr->getType()->isScalarType()) &&
         "Condition expression type not scalar/dependent");
  llvm::uninitialized_copy(ArrayRef(ConditionExpr), getTrailingObjects());
}

OpenACCClause::child_range OpenACCClause::children() {
  switch (getClauseKind()) {
  default:
    assert(false && "Clause children function not implemented");
    break;
#define VISIT_CLAUSE(CLAUSE_NAME)                                              \
  case OpenACCClauseKind::CLAUSE_NAME:                                         \
    return cast<OpenACC##CLAUSE_NAME##Clause>(this)->children();
#define CLAUSE_ALIAS(ALIAS_NAME, CLAUSE_NAME, DEPRECATED)                      \
  case OpenACCClauseKind::ALIAS_NAME:                                          \
    return cast<OpenACC##CLAUSE_NAME##Clause>(this)->children();

#include "clang/Basic/OpenACCClauses.def"
  }
  return child_range(child_iterator(), child_iterator());
}

OpenACCNumWorkersClause::OpenACCNumWorkersClause(SourceLocation BeginLoc,
                                                 SourceLocation LParenLoc,
                                                 Expr *IntExpr,
                                                 SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::NumWorkers, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((!IntExpr || IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "Condition expression type not scalar/dependent");
}

OpenACCGangClause::OpenACCGangClause(SourceLocation BeginLoc,
                                     SourceLocation LParenLoc,
                                     ArrayRef<OpenACCGangKind> GangKinds,
                                     ArrayRef<Expr *> IntExprs,
                                     SourceLocation EndLoc)
    : OpenACCClauseWithExprs(OpenACCClauseKind::Gang, BeginLoc, LParenLoc,
                             EndLoc) {
  assert(GangKinds.size() == IntExprs.size() && "Mismatch exprs/kind?");
  setExprs(getTrailingObjects<Expr *>(IntExprs.size()), IntExprs);
  llvm::uninitialized_copy(GangKinds, getTrailingObjects<OpenACCGangKind>());
}

OpenACCNumWorkersClause *
OpenACCNumWorkersClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                                SourceLocation LParenLoc, Expr *IntExpr,
                                SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCNumWorkersClause),
                         alignof(OpenACCNumWorkersClause));
  return new (Mem)
      OpenACCNumWorkersClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCCollapseClause::OpenACCCollapseClause(SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             bool HasForce, Expr *LoopCount,
                                             SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::Collapse, BeginLoc,
                                     LParenLoc, LoopCount, EndLoc),
      HasForce(HasForce) {}

OpenACCCollapseClause *
OpenACCCollapseClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                              SourceLocation LParenLoc, bool HasForce,
                              Expr *LoopCount, SourceLocation EndLoc) {
  assert((!LoopCount || (LoopCount->isInstantiationDependent() ||
                         isa<ConstantExpr>(LoopCount))) &&
         "Loop count not constant expression");
  void *Mem =
      C.Allocate(sizeof(OpenACCCollapseClause), alignof(OpenACCCollapseClause));
  return new (Mem)
      OpenACCCollapseClause(BeginLoc, LParenLoc, HasForce, LoopCount, EndLoc);
}

OpenACCVectorLengthClause::OpenACCVectorLengthClause(SourceLocation BeginLoc,
                                                     SourceLocation LParenLoc,
                                                     Expr *IntExpr,
                                                     SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::VectorLength, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((!IntExpr || IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "Condition expression type not scalar/dependent");
}

OpenACCVectorLengthClause *
OpenACCVectorLengthClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                                  SourceLocation LParenLoc, Expr *IntExpr,
                                  SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCVectorLengthClause),
                         alignof(OpenACCVectorLengthClause));
  return new (Mem)
      OpenACCVectorLengthClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCAsyncClause::OpenACCAsyncClause(SourceLocation BeginLoc,
                                       SourceLocation LParenLoc, Expr *IntExpr,
                                       SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::Async, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((!IntExpr || IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "Condition expression type not scalar/dependent");
}

OpenACCAsyncClause *OpenACCAsyncClause::Create(const ASTContext &C,
                                               SourceLocation BeginLoc,
                                               SourceLocation LParenLoc,
                                               Expr *IntExpr,
                                               SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(sizeof(OpenACCAsyncClause), alignof(OpenACCAsyncClause));
  return new (Mem) OpenACCAsyncClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCDeviceNumClause::OpenACCDeviceNumClause(SourceLocation BeginLoc,
                                       SourceLocation LParenLoc, Expr *IntExpr,
                                       SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::DeviceNum, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "device_num expression type not scalar/dependent");
}

OpenACCDeviceNumClause *OpenACCDeviceNumClause::Create(const ASTContext &C,
                                               SourceLocation BeginLoc,
                                               SourceLocation LParenLoc,
                                               Expr *IntExpr,
                                               SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(sizeof(OpenACCDeviceNumClause), alignof(OpenACCDeviceNumClause));
  return new (Mem) OpenACCDeviceNumClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCDefaultAsyncClause::OpenACCDefaultAsyncClause(SourceLocation BeginLoc,
                                                     SourceLocation LParenLoc,
                                                     Expr *IntExpr,
                                                     SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::DefaultAsync, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "default_async expression type not scalar/dependent");
}

OpenACCDefaultAsyncClause *
OpenACCDefaultAsyncClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                                  SourceLocation LParenLoc, Expr *IntExpr,
                                  SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCDefaultAsyncClause),
                         alignof(OpenACCDefaultAsyncClause));
  return new (Mem)
      OpenACCDefaultAsyncClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCWaitClause *OpenACCWaitClause::Create(
    const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
    Expr *DevNumExpr, SourceLocation QueuesLoc, ArrayRef<Expr *> QueueIdExprs,
    SourceLocation EndLoc) {
  // Allocates enough room in trailing storage for all the int-exprs, plus a
  // placeholder for the devnum.
  void *Mem = C.Allocate(
      OpenACCWaitClause::totalSizeToAlloc<Expr *>(QueueIdExprs.size() + 1));
  return new (Mem) OpenACCWaitClause(BeginLoc, LParenLoc, DevNumExpr, QueuesLoc,
                                     QueueIdExprs, EndLoc);
}

OpenACCNumGangsClause *OpenACCNumGangsClause::Create(const ASTContext &C,
                                                     SourceLocation BeginLoc,
                                                     SourceLocation LParenLoc,
                                                     ArrayRef<Expr *> IntExprs,
                                                     SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCNumGangsClause::totalSizeToAlloc<Expr *>(IntExprs.size()));
  return new (Mem) OpenACCNumGangsClause(BeginLoc, LParenLoc, IntExprs, EndLoc);
}

OpenACCTileClause *OpenACCTileClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             ArrayRef<Expr *> SizeExprs,
                                             SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCTileClause::totalSizeToAlloc<Expr *>(SizeExprs.size()));
  return new (Mem) OpenACCTileClause(BeginLoc, LParenLoc, SizeExprs, EndLoc);
}

OpenACCPrivateClause *
OpenACCPrivateClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                             SourceLocation LParenLoc, ArrayRef<Expr *> VarList,
                             ArrayRef<OpenACCPrivateRecipe> InitRecipes,
                             SourceLocation EndLoc) {
  assert(VarList.size() == InitRecipes.size());
  void *Mem = C.Allocate(
      OpenACCPrivateClause::totalSizeToAlloc<Expr *, OpenACCPrivateRecipe>(
          VarList.size(), InitRecipes.size()));
  return new (Mem)
      OpenACCPrivateClause(BeginLoc, LParenLoc, VarList, InitRecipes, EndLoc);
}

OpenACCFirstPrivateClause *OpenACCFirstPrivateClause::Create(
    const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
    ArrayRef<Expr *> VarList, ArrayRef<OpenACCFirstPrivateRecipe> InitRecipes,
    SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCFirstPrivateClause::totalSizeToAlloc<Expr *,
                                                  OpenACCFirstPrivateRecipe>(
          VarList.size(), InitRecipes.size()));
  return new (Mem) OpenACCFirstPrivateClause(BeginLoc, LParenLoc, VarList,
                                             InitRecipes, EndLoc);
}

OpenACCAttachClause *OpenACCAttachClause::Create(const ASTContext &C,
                                                 SourceLocation BeginLoc,
                                                 SourceLocation LParenLoc,
                                                 ArrayRef<Expr *> VarList,
                                                 SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCAttachClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCAttachClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCDetachClause *OpenACCDetachClause::Create(const ASTContext &C,
                                                 SourceLocation BeginLoc,
                                                 SourceLocation LParenLoc,
                                                 ArrayRef<Expr *> VarList,
                                                 SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCDetachClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCDetachClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCDeleteClause *OpenACCDeleteClause::Create(const ASTContext &C,
                                                 SourceLocation BeginLoc,
                                                 SourceLocation LParenLoc,
                                                 ArrayRef<Expr *> VarList,
                                                 SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCDeleteClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCDeleteClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCUseDeviceClause *OpenACCUseDeviceClause::Create(const ASTContext &C,
                                                       SourceLocation BeginLoc,
                                                       SourceLocation LParenLoc,
                                                       ArrayRef<Expr *> VarList,
                                                       SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCUseDeviceClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCUseDeviceClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCDevicePtrClause *OpenACCDevicePtrClause::Create(const ASTContext &C,
                                                       SourceLocation BeginLoc,
                                                       SourceLocation LParenLoc,
                                                       ArrayRef<Expr *> VarList,
                                                       SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCDevicePtrClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCDevicePtrClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCNoCreateClause *OpenACCNoCreateClause::Create(const ASTContext &C,
                                                     SourceLocation BeginLoc,
                                                     SourceLocation LParenLoc,
                                                     ArrayRef<Expr *> VarList,
                                                     SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCNoCreateClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCNoCreateClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCPresentClause *OpenACCPresentClause::Create(const ASTContext &C,
                                                   SourceLocation BeginLoc,
                                                   SourceLocation LParenLoc,
                                                   ArrayRef<Expr *> VarList,
                                                   SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCPresentClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCPresentClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCHostClause *OpenACCHostClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             ArrayRef<Expr *> VarList,
                                             SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCHostClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCHostClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCDeviceClause *OpenACCDeviceClause::Create(const ASTContext &C,
                                                 SourceLocation BeginLoc,
                                                 SourceLocation LParenLoc,
                                                 ArrayRef<Expr *> VarList,
                                                 SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCDeviceClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCDeviceClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCCopyClause *
OpenACCCopyClause::Create(const ASTContext &C, OpenACCClauseKind Spelling,
                          SourceLocation BeginLoc, SourceLocation LParenLoc,
                          OpenACCModifierKind Mods, ArrayRef<Expr *> VarList,
                          SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCCopyClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem)
      OpenACCCopyClause(Spelling, BeginLoc, LParenLoc, Mods, VarList, EndLoc);
}

OpenACCLinkClause *OpenACCLinkClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             ArrayRef<Expr *> VarList,
                                             SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCLinkClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCLinkClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCDeviceResidentClause *OpenACCDeviceResidentClause::Create(
    const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
    ArrayRef<Expr *> VarList, SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCDeviceResidentClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem)
      OpenACCDeviceResidentClause(BeginLoc, LParenLoc, VarList, EndLoc);
}

OpenACCCopyInClause *
OpenACCCopyInClause::Create(const ASTContext &C, OpenACCClauseKind Spelling,
                            SourceLocation BeginLoc, SourceLocation LParenLoc,
                            OpenACCModifierKind Mods, ArrayRef<Expr *> VarList,
                            SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCCopyInClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem)
      OpenACCCopyInClause(Spelling, BeginLoc, LParenLoc, Mods, VarList, EndLoc);
}

OpenACCCopyOutClause *
OpenACCCopyOutClause::Create(const ASTContext &C, OpenACCClauseKind Spelling,
                             SourceLocation BeginLoc, SourceLocation LParenLoc,
                             OpenACCModifierKind Mods, ArrayRef<Expr *> VarList,
                             SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCCopyOutClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem) OpenACCCopyOutClause(Spelling, BeginLoc, LParenLoc, Mods,
                                        VarList, EndLoc);
}

OpenACCCreateClause *
OpenACCCreateClause::Create(const ASTContext &C, OpenACCClauseKind Spelling,
                            SourceLocation BeginLoc, SourceLocation LParenLoc,
                            OpenACCModifierKind Mods, ArrayRef<Expr *> VarList,
                            SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCCreateClause::totalSizeToAlloc<Expr *>(VarList.size()));
  return new (Mem)
      OpenACCCreateClause(Spelling, BeginLoc, LParenLoc, Mods, VarList, EndLoc);
}

OpenACCDeviceTypeClause *OpenACCDeviceTypeClause::Create(
    const ASTContext &C, OpenACCClauseKind K, SourceLocation BeginLoc,
    SourceLocation LParenLoc, ArrayRef<DeviceTypeArgument> Archs,
    SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCDeviceTypeClause::totalSizeToAlloc<DeviceTypeArgument>(
          Archs.size()));
  return new (Mem)
      OpenACCDeviceTypeClause(K, BeginLoc, LParenLoc, Archs, EndLoc);
}

OpenACCReductionClause *OpenACCReductionClause::Create(
    const ASTContext &C, SourceLocation BeginLoc, SourceLocation LParenLoc,
    OpenACCReductionOperator Operator, ArrayRef<Expr *> VarList,
    ArrayRef<OpenACCReductionRecipe> Recipes,
    SourceLocation EndLoc) {
  void *Mem = C.Allocate(
      OpenACCReductionClause::totalSizeToAlloc<Expr *, OpenACCReductionRecipe>(
          VarList.size(), Recipes.size()));
  return new (Mem) OpenACCReductionClause(BeginLoc, LParenLoc, Operator,
                                          VarList, Recipes, EndLoc);
}

OpenACCAutoClause *OpenACCAutoClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCAutoClause));
  return new (Mem) OpenACCAutoClause(BeginLoc, EndLoc);
}

OpenACCIndependentClause *
OpenACCIndependentClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                                 SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCIndependentClause));
  return new (Mem) OpenACCIndependentClause(BeginLoc, EndLoc);
}

OpenACCSeqClause *OpenACCSeqClause::Create(const ASTContext &C,
                                           SourceLocation BeginLoc,
                                           SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCSeqClause));
  return new (Mem) OpenACCSeqClause(BeginLoc, EndLoc);
}

OpenACCNoHostClause *OpenACCNoHostClause::Create(const ASTContext &C,
                                                 SourceLocation BeginLoc,
                                                 SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCNoHostClause));
  return new (Mem) OpenACCNoHostClause(BeginLoc, EndLoc);
}

OpenACCGangClause *
OpenACCGangClause::Create(const ASTContext &C, SourceLocation BeginLoc,
                          SourceLocation LParenLoc,
                          ArrayRef<OpenACCGangKind> GangKinds,
                          ArrayRef<Expr *> IntExprs, SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(OpenACCGangClause::totalSizeToAlloc<Expr *, OpenACCGangKind>(
          IntExprs.size(), GangKinds.size()));
  return new (Mem)
      OpenACCGangClause(BeginLoc, LParenLoc, GangKinds, IntExprs, EndLoc);
}

OpenACCWorkerClause::OpenACCWorkerClause(SourceLocation BeginLoc,
                                         SourceLocation LParenLoc,
                                         Expr *IntExpr, SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::Worker, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((!IntExpr || IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "Int expression type not scalar/dependent");
}

OpenACCWorkerClause *OpenACCWorkerClause::Create(const ASTContext &C,
                                                 SourceLocation BeginLoc,
                                                 SourceLocation LParenLoc,
                                                 Expr *IntExpr,
                                                 SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(sizeof(OpenACCWorkerClause), alignof(OpenACCWorkerClause));
  return new (Mem) OpenACCWorkerClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCVectorClause::OpenACCVectorClause(SourceLocation BeginLoc,
                                         SourceLocation LParenLoc,
                                         Expr *IntExpr, SourceLocation EndLoc)
    : OpenACCClauseWithSingleIntExpr(OpenACCClauseKind::Vector, BeginLoc,
                                     LParenLoc, IntExpr, EndLoc) {
  assert((!IntExpr || IntExpr->isInstantiationDependent() ||
          IntExpr->getType()->isIntegerType()) &&
         "Int expression type not scalar/dependent");
}

OpenACCVectorClause *OpenACCVectorClause::Create(const ASTContext &C,
                                                 SourceLocation BeginLoc,
                                                 SourceLocation LParenLoc,
                                                 Expr *IntExpr,
                                                 SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(sizeof(OpenACCVectorClause), alignof(OpenACCVectorClause));
  return new (Mem) OpenACCVectorClause(BeginLoc, LParenLoc, IntExpr, EndLoc);
}

OpenACCFinalizeClause *OpenACCFinalizeClause::Create(const ASTContext &C,
                                                     SourceLocation BeginLoc,
                                                     SourceLocation EndLoc) {
  void *Mem =
      C.Allocate(sizeof(OpenACCFinalizeClause), alignof(OpenACCFinalizeClause));
  return new (Mem) OpenACCFinalizeClause(BeginLoc, EndLoc);
}

OpenACCIfPresentClause *OpenACCIfPresentClause::Create(const ASTContext &C,
                                                       SourceLocation BeginLoc,
                                                       SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCIfPresentClause),
                         alignof(OpenACCIfPresentClause));
  return new (Mem) OpenACCIfPresentClause(BeginLoc, EndLoc);
}

OpenACCBindClause *OpenACCBindClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             const StringLiteral *SL,
                                             SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCBindClause), alignof(OpenACCBindClause));
  return new (Mem) OpenACCBindClause(BeginLoc, LParenLoc, SL, EndLoc);
}

OpenACCBindClause *OpenACCBindClause::Create(const ASTContext &C,
                                             SourceLocation BeginLoc,
                                             SourceLocation LParenLoc,
                                             const IdentifierInfo *ID,
                                             SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OpenACCBindClause), alignof(OpenACCBindClause));
  return new (Mem) OpenACCBindClause(BeginLoc, LParenLoc, ID, EndLoc);
}

bool clang::operator==(const OpenACCBindClause &LHS,
                       const OpenACCBindClause &RHS) {
  if (LHS.isStringArgument() != RHS.isStringArgument())
    return false;

  if (LHS.isStringArgument())
    return LHS.getStringArgument()->getString() ==
           RHS.getStringArgument()->getString();
  return LHS.getIdentifierArgument()->getName() ==
         RHS.getIdentifierArgument()->getName();
}

//===----------------------------------------------------------------------===//
//  OpenACC clauses printing methods
//===----------------------------------------------------------------------===//

void OpenACCClausePrinter::printExpr(const Expr *E) {
  E->printPretty(OS, nullptr, Policy, 0);
}

void OpenACCClausePrinter::VisitDefaultClause(const OpenACCDefaultClause &C) {
  OS << "default(" << C.getDefaultClauseKind() << ")";
}

void OpenACCClausePrinter::VisitIfClause(const OpenACCIfClause &C) {
  OS << "if(";
  printExpr(C.getConditionExpr());
  OS << ")";
}

void OpenACCClausePrinter::VisitSelfClause(const OpenACCSelfClause &C) {
  OS << "self";

  if (C.isConditionExprClause()) {
    if (const Expr *CondExpr = C.getConditionExpr()) {
      OS << "(";
      printExpr(CondExpr);
      OS << ")";
    }
  } else {
    OS << "(";
    llvm::interleaveComma(C.getVarList(), OS,
                          [&](const Expr *E) { printExpr(E); });
    OS << ")";
  }
}

void OpenACCClausePrinter::VisitNumGangsClause(const OpenACCNumGangsClause &C) {
  OS << "num_gangs(";
  llvm::interleaveComma(C.getIntExprs(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitTileClause(const OpenACCTileClause &C) {
  OS << "tile(";
  llvm::interleaveComma(C.getSizeExprs(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitNumWorkersClause(
    const OpenACCNumWorkersClause &C) {
  OS << "num_workers(";
  printExpr(C.getIntExpr());
  OS << ")";
}

void OpenACCClausePrinter::VisitVectorLengthClause(
    const OpenACCVectorLengthClause &C) {
  OS << "vector_length(";
  printExpr(C.getIntExpr());
  OS << ")";
}

void OpenACCClausePrinter::VisitDeviceNumClause(
    const OpenACCDeviceNumClause &C) {
  OS << "device_num(";
  printExpr(C.getIntExpr());
  OS << ")";
}

void OpenACCClausePrinter::VisitDefaultAsyncClause(
    const OpenACCDefaultAsyncClause &C) {
  OS << "default_async(";
  printExpr(C.getIntExpr());
  OS << ")";
}

void OpenACCClausePrinter::VisitAsyncClause(const OpenACCAsyncClause &C) {
  OS << "async";
  if (C.hasIntExpr()) {
    OS << "(";
    printExpr(C.getIntExpr());
    OS << ")";
  }
}

void OpenACCClausePrinter::VisitPrivateClause(const OpenACCPrivateClause &C) {
  OS << "private(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitFirstPrivateClause(
    const OpenACCFirstPrivateClause &C) {
  OS << "firstprivate(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitAttachClause(const OpenACCAttachClause &C) {
  OS << "attach(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitDetachClause(const OpenACCDetachClause &C) {
  OS << "detach(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitDeleteClause(const OpenACCDeleteClause &C) {
  OS << "delete(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitUseDeviceClause(
    const OpenACCUseDeviceClause &C) {
  OS << "use_device(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitDevicePtrClause(
    const OpenACCDevicePtrClause &C) {
  OS << "deviceptr(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitNoCreateClause(const OpenACCNoCreateClause &C) {
  OS << "no_create(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitPresentClause(const OpenACCPresentClause &C) {
  OS << "present(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitHostClause(const OpenACCHostClause &C) {
  OS << "host(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitDeviceClause(const OpenACCDeviceClause &C) {
  OS << "device(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitCopyClause(const OpenACCCopyClause &C) {
  OS << C.getClauseKind() << '(';
  if (C.getModifierList() != OpenACCModifierKind::Invalid)
    OS << C.getModifierList() << ": ";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitLinkClause(const OpenACCLinkClause &C) {
  OS << "link(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitDeviceResidentClause(
    const OpenACCDeviceResidentClause &C) {
  OS << "device_resident(";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitCopyInClause(const OpenACCCopyInClause &C) {
  OS << C.getClauseKind() << '(';
  if (C.getModifierList() != OpenACCModifierKind::Invalid)
    OS << C.getModifierList() << ": ";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitCopyOutClause(const OpenACCCopyOutClause &C) {
  OS << C.getClauseKind() << '(';
  if (C.getModifierList() != OpenACCModifierKind::Invalid)
    OS << C.getModifierList() << ": ";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitCreateClause(const OpenACCCreateClause &C) {
  OS << C.getClauseKind() << '(';
  if (C.getModifierList() != OpenACCModifierKind::Invalid)
    OS << C.getModifierList() << ": ";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitReductionClause(
    const OpenACCReductionClause &C) {
  OS << "reduction(" << C.getReductionOp() << ": ";
  llvm::interleaveComma(C.getVarList(), OS,
                        [&](const Expr *E) { printExpr(E); });
  OS << ")";
}

void OpenACCClausePrinter::VisitWaitClause(const OpenACCWaitClause &C) {
  OS << "wait";
  if (C.hasExprs()) {
    OS << "(";
    if (C.hasDevNumExpr()) {
      OS << "devnum: ";
      printExpr(C.getDevNumExpr());
      OS << " : ";
    }

    if (C.hasQueuesTag())
      OS << "queues: ";

    llvm::interleaveComma(C.getQueueIdExprs(), OS,
                          [&](const Expr *E) { printExpr(E); });
    OS << ")";
  }
}

void OpenACCClausePrinter::VisitDeviceTypeClause(
    const OpenACCDeviceTypeClause &C) {
  OS << C.getClauseKind();
  OS << "(";
  llvm::interleaveComma(C.getArchitectures(), OS,
                        [&](const DeviceTypeArgument &Arch) {
                          if (Arch.getIdentifierInfo() == nullptr)
                            OS << "*";
                          else
                            OS << Arch.getIdentifierInfo()->getName();
                        });
  OS << ")";
}

void OpenACCClausePrinter::VisitAutoClause(const OpenACCAutoClause &C) {
  OS << "auto";
}

void OpenACCClausePrinter::VisitIndependentClause(
    const OpenACCIndependentClause &C) {
  OS << "independent";
}

void OpenACCClausePrinter::VisitSeqClause(const OpenACCSeqClause &C) {
  OS << "seq";
}

void OpenACCClausePrinter::VisitNoHostClause(const OpenACCNoHostClause &C) {
  OS << "nohost";
}

void OpenACCClausePrinter::VisitCollapseClause(const OpenACCCollapseClause &C) {
  OS << "collapse(";
  if (C.hasForce())
    OS << "force:";
  printExpr(C.getLoopCount());
  OS << ")";
}

void OpenACCClausePrinter::VisitGangClause(const OpenACCGangClause &C) {
  OS << "gang";

  if (C.getNumExprs() > 0) {
    OS << "(";
    bool first = true;
    for (unsigned I = 0; I < C.getNumExprs(); ++I) {
      if (!first)
        OS << ", ";
      first = false;

      OS << C.getExpr(I).first << ": ";
      printExpr(C.getExpr(I).second);
    }
    OS << ")";
  }
}

void OpenACCClausePrinter::VisitWorkerClause(const OpenACCWorkerClause &C) {
  OS << "worker";

  if (C.hasIntExpr()) {
    OS << "(num: ";
    printExpr(C.getIntExpr());
    OS << ")";
  }
}

void OpenACCClausePrinter::VisitVectorClause(const OpenACCVectorClause &C) {
  OS << "vector";

  if (C.hasIntExpr()) {
    OS << "(length: ";
    printExpr(C.getIntExpr());
    OS << ")";
  }
}

void OpenACCClausePrinter::VisitFinalizeClause(const OpenACCFinalizeClause &C) {
  OS << "finalize";
}

void OpenACCClausePrinter::VisitIfPresentClause(
    const OpenACCIfPresentClause &C) {
  OS << "if_present";
}

void OpenACCClausePrinter::VisitBindClause(const OpenACCBindClause &C) {
  OS << "bind(";
  if (C.isStringArgument())
    OS << '"' << C.getStringArgument()->getString() << '"';
  else
    OS << C.getIdentifierArgument()->getName();
  OS << ")";
}
