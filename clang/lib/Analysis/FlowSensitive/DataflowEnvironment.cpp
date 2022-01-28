//===-- DataflowEnvironment.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines an Environment class that is used by dataflow analyses
//  that run over Control-Flow Graphs (CFGs) to keep track of the state of the
//  program at given program points.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
#include <utility>

namespace clang {
namespace dataflow {

/// Returns a map consisting of key-value entries that are present in both maps.
template <typename K, typename V>
llvm::DenseMap<K, V> intersectDenseMaps(const llvm::DenseMap<K, V> &Map1,
                                        const llvm::DenseMap<K, V> &Map2) {
  llvm::DenseMap<K, V> Result;
  for (auto &Entry : Map1) {
    auto It = Map2.find(Entry.first);
    if (It != Map2.end() && Entry.second == It->second)
      Result.insert({Entry.first, Entry.second});
  }
  return Result;
}

Environment::Environment(DataflowAnalysisContext &DACtx,
                         const DeclContext &DeclCtx)
    : Environment(DACtx) {
  if (const auto *FuncDecl = dyn_cast<FunctionDecl>(&DeclCtx)) {
    for (const auto *ParamDecl : FuncDecl->parameters()) {
      assert(ParamDecl != nullptr);
      auto &ParamLoc = createStorageLocation(*ParamDecl);
      setStorageLocation(*ParamDecl, ParamLoc);
      if (Value *ParamVal = createValue(ParamDecl->getType()))
        setValue(ParamLoc, *ParamVal);
    }
  }

  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(&DeclCtx)) {
    if (!MethodDecl->isStatic()) {
      QualType ThisPointeeType = MethodDecl->getThisObjectType();
      // FIXME: Add support for union types.
      if (!ThisPointeeType->isUnionType()) {
        auto &ThisPointeeLoc = createStorageLocation(ThisPointeeType);
        DACtx.setThisPointeeStorageLocation(ThisPointeeLoc);
        if (Value *ThisPointeeVal = createValue(ThisPointeeType))
          setValue(ThisPointeeLoc, *ThisPointeeVal);
      }
    }
  }
}

bool Environment::operator==(const Environment &Other) const {
  assert(DACtx == Other.DACtx);
  return DeclToLoc == Other.DeclToLoc && LocToVal == Other.LocToVal;
}

LatticeJoinEffect Environment::join(const Environment &Other,
                                    Environment::Merger &Merger) {
  assert(DACtx == Other.DACtx);

  auto Effect = LatticeJoinEffect::Unchanged;

  const unsigned DeclToLocSizeBefore = DeclToLoc.size();
  DeclToLoc = intersectDenseMaps(DeclToLoc, Other.DeclToLoc);
  if (DeclToLocSizeBefore != DeclToLoc.size())
    Effect = LatticeJoinEffect::Changed;

  const unsigned ExprToLocSizeBefore = ExprToLoc.size();
  ExprToLoc = intersectDenseMaps(ExprToLoc, Other.ExprToLoc);
  if (ExprToLocSizeBefore != ExprToLoc.size())
    Effect = LatticeJoinEffect::Changed;

  // Move `LocToVal` so that `Environment::Merger::merge` can safely assign
  // values to storage locations while this code iterates over the current
  // assignments.
  llvm::DenseMap<const StorageLocation *, Value *> OldLocToVal =
      std::move(LocToVal);
  for (auto &Entry : OldLocToVal) {
    const StorageLocation *Loc = Entry.first;
    assert(Loc != nullptr);

    Value *Val = Entry.second;
    assert(Val != nullptr);

    auto It = Other.LocToVal.find(Loc);
    if (It == Other.LocToVal.end())
      continue;
    assert(It->second != nullptr);

    if (It->second == Val) {
      LocToVal.insert({Loc, Val});
      continue;
    }

    if (auto *FirstVal = dyn_cast<PointerValue>(Val)) {
      auto *SecondVal = cast<PointerValue>(It->second);
      if (&FirstVal->getPointeeLoc() == &SecondVal->getPointeeLoc()) {
        LocToVal.insert({Loc, FirstVal});
        continue;
      }
    }

    // FIXME: Consider destroying `MergedValue` immediately if `Merger::merge`
    // returns false to avoid storing unneeded values in `DACtx`.
    if (Value *MergedVal = createValue(Loc->getType()))
      if (Merger.merge(Loc->getType(), *Val, *It->second, *MergedVal, *this))
        LocToVal.insert({Loc, MergedVal});
  }
  if (OldLocToVal.size() != LocToVal.size())
    Effect = LatticeJoinEffect::Changed;

  return Effect;
}

StorageLocation &Environment::createStorageLocation(QualType Type) {
  assert(!Type.isNull());
  if (Type->isStructureOrClassType() || Type->isUnionType()) {
    // FIXME: Explore options to avoid eager initialization of fields as some of
    // them might not be needed for a particular analysis.
    llvm::DenseMap<const ValueDecl *, StorageLocation *> FieldLocs;
    for (const FieldDecl *Field : Type->getAsRecordDecl()->fields()) {
      FieldLocs.insert({Field, &createStorageLocation(Field->getType())});
    }
    return takeOwnership(
        std::make_unique<AggregateStorageLocation>(Type, std::move(FieldLocs)));
  }
  return takeOwnership(std::make_unique<ScalarStorageLocation>(Type));
}

StorageLocation &Environment::createStorageLocation(const VarDecl &D) {
  // Evaluated declarations are always assigned the same storage locations to
  // ensure that the environment stabilizes across loop iterations. Storage
  // locations for evaluated declarations are stored in the analysis context.
  if (auto *Loc = DACtx->getStorageLocation(D))
    return *Loc;
  auto &Loc = createStorageLocation(D.getType());
  DACtx->setStorageLocation(D, Loc);
  return Loc;
}

StorageLocation &Environment::createStorageLocation(const Expr &E) {
  // Evaluated expressions are always assigned the same storage locations to
  // ensure that the environment stabilizes across loop iterations. Storage
  // locations for evaluated expressions are stored in the analysis context.
  if (auto *Loc = DACtx->getStorageLocation(E))
    return *Loc;
  auto &Loc = createStorageLocation(E.getType());
  DACtx->setStorageLocation(E, Loc);
  return Loc;
}

void Environment::setStorageLocation(const ValueDecl &D, StorageLocation &Loc) {
  assert(DeclToLoc.find(&D) == DeclToLoc.end());
  DeclToLoc[&D] = &Loc;
}

StorageLocation *Environment::getStorageLocation(const ValueDecl &D,
                                                 SkipPast SP) const {
  auto It = DeclToLoc.find(&D);
  return It == DeclToLoc.end() ? nullptr : &skip(*It->second, SP);
}

void Environment::setStorageLocation(const Expr &E, StorageLocation &Loc) {
  assert(ExprToLoc.find(&E) == ExprToLoc.end());
  ExprToLoc[&E] = &Loc;
}

StorageLocation *Environment::getStorageLocation(const Expr &E,
                                                 SkipPast SP) const {
  auto It = ExprToLoc.find(&E);
  return It == ExprToLoc.end() ? nullptr : &skip(*It->second, SP);
}

StorageLocation *Environment::getThisPointeeStorageLocation() const {
  return DACtx->getThisPointeeStorageLocation();
}

void Environment::setValue(const StorageLocation &Loc, Value &Val) {
  LocToVal[&Loc] = &Val;

  if (auto *StructVal = dyn_cast<StructValue>(&Val)) {
    auto &AggregateLoc = *cast<AggregateStorageLocation>(&Loc);

    const QualType Type = AggregateLoc.getType();
    assert(Type->isStructureOrClassType());

    for (const FieldDecl *Field : Type->getAsRecordDecl()->fields()) {
      assert(Field != nullptr);
      setValue(AggregateLoc.getChild(*Field), StructVal->getChild(*Field));
    }
  }
}

Value *Environment::getValue(const StorageLocation &Loc) const {
  auto It = LocToVal.find(&Loc);
  return It == LocToVal.end() ? nullptr : It->second;
}

Value *Environment::getValue(const ValueDecl &D, SkipPast SP) const {
  auto *Loc = getStorageLocation(D, SP);
  if (Loc == nullptr)
    return nullptr;
  return getValue(*Loc);
}

Value *Environment::getValue(const Expr &E, SkipPast SP) const {
  auto *Loc = getStorageLocation(E, SP);
  if (Loc == nullptr)
    return nullptr;
  return getValue(*Loc);
}

Value *Environment::createValue(QualType Type) {
  llvm::DenseSet<QualType> Visited;
  return createValueUnlessSelfReferential(Type, Visited);
}

Value *Environment::createValueUnlessSelfReferential(
    QualType Type, llvm::DenseSet<QualType> &Visited) {
  assert(!Type.isNull());

  if (Type->isIntegerType()) {
    return &takeOwnership(std::make_unique<IntegerValue>());
  }

  if (Type->isReferenceType()) {
    QualType PointeeType = Type->getAs<ReferenceType>()->getPointeeType();
    auto &PointeeLoc = createStorageLocation(PointeeType);

    if (!Visited.contains(PointeeType.getCanonicalType())) {
      Visited.insert(PointeeType.getCanonicalType());
      Value *PointeeVal =
          createValueUnlessSelfReferential(PointeeType, Visited);
      Visited.erase(PointeeType.getCanonicalType());

      if (PointeeVal != nullptr)
        setValue(PointeeLoc, *PointeeVal);
    }

    return &takeOwnership(std::make_unique<ReferenceValue>(PointeeLoc));
  }

  if (Type->isPointerType()) {
    QualType PointeeType = Type->getAs<PointerType>()->getPointeeType();
    auto &PointeeLoc = createStorageLocation(PointeeType);

    if (!Visited.contains(PointeeType.getCanonicalType())) {
      Visited.insert(PointeeType.getCanonicalType());
      Value *PointeeVal =
          createValueUnlessSelfReferential(PointeeType, Visited);
      Visited.erase(PointeeType.getCanonicalType());

      if (PointeeVal != nullptr)
        setValue(PointeeLoc, *PointeeVal);
    }

    return &takeOwnership(std::make_unique<PointerValue>(PointeeLoc));
  }

  if (Type->isStructureOrClassType()) {
    // FIXME: Initialize only fields that are accessed in the context that is
    // being analyzed.
    llvm::DenseMap<const ValueDecl *, Value *> FieldValues;
    for (const FieldDecl *Field : Type->getAsRecordDecl()->fields()) {
      assert(Field != nullptr);

      QualType FieldType = Field->getType();
      if (Visited.contains(FieldType.getCanonicalType()))
        continue;

      Visited.insert(FieldType.getCanonicalType());
      FieldValues.insert(
          {Field, createValueUnlessSelfReferential(FieldType, Visited)});
      Visited.erase(FieldType.getCanonicalType());
    }

    return &takeOwnership(
        std::make_unique<StructValue>(std::move(FieldValues)));
  }

  return nullptr;
}

StorageLocation &Environment::skip(StorageLocation &Loc, SkipPast SP) const {
  switch (SP) {
  case SkipPast::None:
    return Loc;
  case SkipPast::Reference:
    // References cannot be chained so we only need to skip past one level of
    // indirection.
    if (auto *Val = dyn_cast_or_null<ReferenceValue>(getValue(Loc)))
      return Val->getPointeeLoc();
    return Loc;
  case SkipPast::ReferenceThenPointer:
    StorageLocation &LocPastRef = skip(Loc, SkipPast::Reference);
    if (auto *Val = dyn_cast_or_null<PointerValue>(getValue(LocPastRef)))
      return Val->getPointeeLoc();
    return LocPastRef;
  }
  llvm_unreachable("bad SkipPast kind");
}

const StorageLocation &Environment::skip(const StorageLocation &Loc,
                                         SkipPast SP) const {
  return skip(*const_cast<StorageLocation *>(&Loc), SP);
}

} // namespace dataflow
} // namespace clang
