//===-- ASTOps.cc -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Operations on AST nodes that are used in flow-sensitive analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/ASTOps.h"
#include "clang/AST/ComputeDependence.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <iterator>
#include <vector>

#define DEBUG_TYPE "dataflow"

namespace clang::dataflow {

const Expr &ignoreCFGOmittedNodes(const Expr &E) {
  const Expr *Current = &E;
  if (auto *EWC = dyn_cast<ExprWithCleanups>(Current)) {
    Current = EWC->getSubExpr();
    assert(Current != nullptr);
  }
  Current = Current->IgnoreParens();
  assert(Current != nullptr);
  return *Current;
}

const Stmt &ignoreCFGOmittedNodes(const Stmt &S) {
  if (auto *E = dyn_cast<Expr>(&S))
    return ignoreCFGOmittedNodes(*E);
  return S;
}

// FIXME: Does not precisely handle non-virtual diamond inheritance. A single
// field decl will be modeled for all instances of the inherited field.
static void getFieldsFromClassHierarchy(QualType Type, FieldSet &Fields) {
  if (Type->isIncompleteType() || Type->isDependentType() ||
      !Type->isRecordType())
    return;

  for (const FieldDecl *Field : Type->getAsRecordDecl()->fields())
    Fields.insert(Field);
  if (auto *CXXRecord = Type->getAsCXXRecordDecl())
    for (const CXXBaseSpecifier &Base : CXXRecord->bases())
      getFieldsFromClassHierarchy(Base.getType(), Fields);
}

/// Gets the set of all fields in the type.
FieldSet getObjectFields(QualType Type) {
  FieldSet Fields;
  getFieldsFromClassHierarchy(Type, Fields);
  return Fields;
}

bool containsSameFields(const FieldSet &Fields,
                        const RecordStorageLocation::FieldToLoc &FieldLocs) {
  if (Fields.size() != FieldLocs.size())
    return false;
  for ([[maybe_unused]] auto [Field, Loc] : FieldLocs)
    if (!Fields.contains(cast_or_null<FieldDecl>(Field)))
      return false;
  return true;
}

/// Returns the fields of a `RecordDecl` that are initialized by an
/// `InitListExpr`, in the order in which they appear in
/// `InitListExpr::inits()`.
/// `Init->getType()` must be a record type.
static std::vector<const FieldDecl *>
getFieldsForInitListExpr(const InitListExpr *InitList) {
  const RecordDecl *RD = InitList->getType()->getAsRecordDecl();
  assert(RD != nullptr);

  std::vector<const FieldDecl *> Fields;

  if (InitList->getType()->isUnionType()) {
    Fields.push_back(InitList->getInitializedFieldInUnion());
    return Fields;
  }

  // Unnamed bitfields are only used for padding and do not appear in
  // `InitListExpr`'s inits. However, those fields do appear in `RecordDecl`'s
  // field list, and we thus need to remove them before mapping inits to
  // fields to avoid mapping inits to the wrongs fields.
  llvm::copy_if(
      RD->fields(), std::back_inserter(Fields),
      [](const FieldDecl *Field) { return !Field->isUnnamedBitField(); });
  return Fields;
}

RecordInitListHelper::RecordInitListHelper(const InitListExpr *InitList) {
  auto *RD = InitList->getType()->getAsCXXRecordDecl();
  assert(RD != nullptr);

  std::vector<const FieldDecl *> Fields = getFieldsForInitListExpr(InitList);
  ArrayRef<Expr *> Inits = InitList->inits();

  // Unions initialized with an empty initializer list need special treatment.
  // For structs/classes initialized with an empty initializer list, Clang
  // puts `ImplicitValueInitExpr`s in `InitListExpr::inits()`, but for unions,
  // it doesn't do this -- so we create an `ImplicitValueInitExpr` ourselves.
  SmallVector<Expr *> InitsForUnion;
  if (InitList->getType()->isUnionType() && Inits.empty()) {
    assert(Fields.size() == 1);
    ImplicitValueInitForUnion.emplace(Fields.front()->getType());
    InitsForUnion.push_back(&*ImplicitValueInitForUnion);
    Inits = InitsForUnion;
  }

  size_t InitIdx = 0;

  assert(Fields.size() + RD->getNumBases() == Inits.size());
  for (const CXXBaseSpecifier &Base : RD->bases()) {
    assert(InitIdx < Inits.size());
    Expr *Init = Inits[InitIdx++];
    BaseInits.emplace_back(&Base, Init);
  }

  assert(Fields.size() == Inits.size() - InitIdx);
  for (const FieldDecl *Field : Fields) {
    assert(InitIdx < Inits.size());
    Expr *Init = Inits[InitIdx++];
    FieldInits.emplace_back(Field, Init);
  }
}

static void insertIfGlobal(const Decl &D,
                           llvm::DenseSet<const VarDecl *> &Globals) {
  if (auto *V = dyn_cast<VarDecl>(&D))
    if (V->hasGlobalStorage())
      Globals.insert(V);
}

static void insertIfFunction(const Decl &D,
                             llvm::DenseSet<const FunctionDecl *> &Funcs) {
  if (auto *FD = dyn_cast<FunctionDecl>(&D))
    Funcs.insert(FD);
}

static MemberExpr *getMemberForAccessor(const CXXMemberCallExpr &C) {
  // Use getCalleeDecl instead of getMethodDecl in order to handle
  // pointer-to-member calls.
  const auto *MethodDecl = dyn_cast_or_null<CXXMethodDecl>(C.getCalleeDecl());
  if (!MethodDecl)
    return nullptr;
  auto *Body = dyn_cast_or_null<CompoundStmt>(MethodDecl->getBody());
  if (!Body || Body->size() != 1)
    return nullptr;
  if (auto *RS = dyn_cast<ReturnStmt>(*Body->body_begin()))
    if (auto *Return = RS->getRetValue())
      return dyn_cast<MemberExpr>(Return->IgnoreParenImpCasts());
  return nullptr;
}

static void getReferencedDecls(const Decl &D, ReferencedDecls &Referenced) {
  insertIfGlobal(D, Referenced.Globals);
  insertIfFunction(D, Referenced.Functions);
  if (const auto *Decomp = dyn_cast<DecompositionDecl>(&D))
    for (const auto *B : Decomp->bindings())
      if (auto *ME = dyn_cast_or_null<MemberExpr>(B->getBinding()))
        // FIXME: should we be using `E->getFoundDecl()`?
        if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl()))
          Referenced.Fields.insert(FD);
}

/// Traverses `S` and inserts into `Referenced` any declarations that are
/// declared in or referenced from sub-statements.
static void getReferencedDecls(const Stmt &S, ReferencedDecls &Referenced) {
  for (auto *Child : S.children())
    if (Child != nullptr)
      getReferencedDecls(*Child, Referenced);
  if (const auto *DefaultArg = dyn_cast<CXXDefaultArgExpr>(&S))
    getReferencedDecls(*DefaultArg->getExpr(), Referenced);
  if (const auto *DefaultInit = dyn_cast<CXXDefaultInitExpr>(&S))
    getReferencedDecls(*DefaultInit->getExpr(), Referenced);

  if (auto *DS = dyn_cast<DeclStmt>(&S)) {
    if (DS->isSingleDecl())
      getReferencedDecls(*DS->getSingleDecl(), Referenced);
    else
      for (auto *D : DS->getDeclGroup())
        getReferencedDecls(*D, Referenced);
  } else if (auto *E = dyn_cast<DeclRefExpr>(&S)) {
    insertIfGlobal(*E->getDecl(), Referenced.Globals);
    insertIfFunction(*E->getDecl(), Referenced.Functions);
  } else if (const auto *C = dyn_cast<CXXMemberCallExpr>(&S)) {
    // If this is a method that returns a member variable but does nothing else,
    // model the field of the return value.
    if (MemberExpr *E = getMemberForAccessor(*C))
      if (const auto *FD = dyn_cast<FieldDecl>(E->getMemberDecl()))
        Referenced.Fields.insert(FD);
  } else if (auto *E = dyn_cast<MemberExpr>(&S)) {
    // FIXME: should we be using `E->getFoundDecl()`?
    const ValueDecl *VD = E->getMemberDecl();
    insertIfGlobal(*VD, Referenced.Globals);
    insertIfFunction(*VD, Referenced.Functions);
    if (const auto *FD = dyn_cast<FieldDecl>(VD))
      Referenced.Fields.insert(FD);
  } else if (auto *InitList = dyn_cast<InitListExpr>(&S)) {
    if (InitList->getType()->isRecordType())
      for (const auto *FD : getFieldsForInitListExpr(InitList))
        Referenced.Fields.insert(FD);
  }
}

ReferencedDecls getReferencedDecls(const FunctionDecl &FD) {
  ReferencedDecls Result;
  // Look for global variable and field references in the
  // constructor-initializers.
  if (const auto *CtorDecl = dyn_cast<CXXConstructorDecl>(&FD)) {
    for (const auto *Init : CtorDecl->inits()) {
      if (Init->isMemberInitializer()) {
        Result.Fields.insert(Init->getMember());
      } else if (Init->isIndirectMemberInitializer()) {
        for (const auto *I : Init->getIndirectMember()->chain())
          Result.Fields.insert(cast<FieldDecl>(I));
      }
      const Expr *E = Init->getInit();
      assert(E != nullptr);
      getReferencedDecls(*E, Result);
    }
    // Add all fields mentioned in default member initializers.
    for (const FieldDecl *F : CtorDecl->getParent()->fields())
      if (const auto *I = F->getInClassInitializer())
        getReferencedDecls(*I, Result);
  }
  getReferencedDecls(*FD.getBody(), Result);

  return Result;
}

} // namespace clang::dataflow
