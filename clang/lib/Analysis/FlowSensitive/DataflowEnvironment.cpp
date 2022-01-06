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
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

bool Environment::operator==(const Environment &Other) const {
  assert(DACtx == Other.DACtx);
  return DeclToLoc == Other.DeclToLoc && LocToVal == Other.LocToVal;
}

LatticeJoinEffect Environment::join(const Environment &Other) {
  assert(DACtx == Other.DACtx);

  auto Effect = LatticeJoinEffect::Unchanged;

  const unsigned DeclToLocSizeBefore = DeclToLoc.size();
  DeclToLoc = intersectDenseMaps(DeclToLoc, Other.DeclToLoc);
  if (DeclToLocSizeBefore != DeclToLoc.size())
    Effect = LatticeJoinEffect::Changed;

  // FIXME: Add support for joining distinct values that are assigned to the
  // same storage locations in `LocToVal` and `Other.LocToVal`.
  const unsigned LocToValSizeBefore = LocToVal.size();
  LocToVal = intersectDenseMaps(LocToVal, Other.LocToVal);
  if (LocToValSizeBefore != LocToVal.size())
    Effect = LatticeJoinEffect::Changed;

  return Effect;
}

StorageLocation &Environment::createStorageLocation(QualType Type) {
  assert(!Type.isNull());
  if (Type->isStructureOrClassType()) {
    // FIXME: Explore options to avoid eager initialization of fields as some of
    // them might not be needed for a particular analysis.
    llvm::DenseMap<const ValueDecl *, StorageLocation *> FieldLocs;
    for (const FieldDecl *Field : Type->getAsRecordDecl()->fields()) {
      FieldLocs.insert({Field, &createStorageLocation(Field->getType())});
    }
    return DACtx->takeOwnership(
        std::make_unique<AggregateStorageLocation>(Type, std::move(FieldLocs)));
  }
  return DACtx->takeOwnership(std::make_unique<ScalarStorageLocation>(Type));
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

void Environment::setStorageLocation(const ValueDecl &D, StorageLocation &Loc) {
  assert(DeclToLoc.find(&D) == DeclToLoc.end());
  DeclToLoc[&D] = &Loc;
}

StorageLocation *Environment::getStorageLocation(const ValueDecl &D) const {
  auto It = DeclToLoc.find(&D);
  return It == DeclToLoc.end() ? nullptr : It->second;
}

void Environment::setValue(const StorageLocation &Loc, Value &Value) {
  LocToVal[&Loc] = &Value;
}

Value *Environment::getValue(const StorageLocation &Loc) const {
  auto It = LocToVal.find(&Loc);
  return It == LocToVal.end() ? nullptr : It->second;
}

Value *Environment::initValueInStorageLocation(const StorageLocation &Loc,
                                               QualType Type) {
  llvm::DenseSet<QualType> Visited;
  return initValueInStorageLocationUnlessSelfReferential(Loc, Type, Visited);
}

Value *Environment::initValueInStorageLocationUnlessSelfReferential(
    const StorageLocation &Loc, QualType Type,
    llvm::DenseSet<QualType> &Visited) {
  assert(!Type.isNull());

  if (Type->isIntegerType()) {
    auto &Value = DACtx->takeOwnership(std::make_unique<IntegerValue>());
    setValue(Loc, Value);
    return &Value;
  }

  if (Type->isReferenceType()) {
    QualType PointeeType = Type->getAs<ReferenceType>()->getPointeeType();
    auto &PointeeLoc = createStorageLocation(PointeeType);

    if (!Visited.contains(PointeeType.getCanonicalType())) {
      Visited.insert(PointeeType.getCanonicalType());
      initValueInStorageLocationUnlessSelfReferential(PointeeLoc, PointeeType,
                                                      Visited);
      Visited.erase(PointeeType.getCanonicalType());
    }

    auto &Value =
        DACtx->takeOwnership(std::make_unique<ReferenceValue>(PointeeLoc));
    setValue(Loc, Value);
    return &Value;
  }

  if (Type->isPointerType()) {
    QualType PointeeType = Type->getAs<PointerType>()->getPointeeType();
    auto &PointeeLoc = createStorageLocation(PointeeType);

    if (!Visited.contains(PointeeType.getCanonicalType())) {
      Visited.insert(PointeeType.getCanonicalType());
      initValueInStorageLocationUnlessSelfReferential(PointeeLoc, PointeeType,
                                                      Visited);
      Visited.erase(PointeeType.getCanonicalType());
    }

    auto &Value =
        DACtx->takeOwnership(std::make_unique<PointerValue>(PointeeLoc));
    setValue(Loc, Value);
    return &Value;
  }

  if (Type->isStructureOrClassType()) {
    auto *AggregateLoc = cast<AggregateStorageLocation>(&Loc);

    llvm::DenseMap<const ValueDecl *, Value *> FieldValues;
    for (const FieldDecl *Field : Type->getAsRecordDecl()->fields()) {
      assert(Field != nullptr);

      QualType FieldType = Field->getType();
      if (Visited.contains(FieldType.getCanonicalType()))
        continue;

      Visited.insert(FieldType.getCanonicalType());
      FieldValues.insert(
          {Field, initValueInStorageLocationUnlessSelfReferential(
                      AggregateLoc->getChild(*Field), FieldType, Visited)});
      Visited.erase(FieldType.getCanonicalType());
    }

    auto &Value = DACtx->takeOwnership(
        std::make_unique<StructValue>(std::move(FieldValues)));
    setValue(Loc, Value);
    return &Value;
  }

  return nullptr;
}

} // namespace dataflow
} // namespace clang
