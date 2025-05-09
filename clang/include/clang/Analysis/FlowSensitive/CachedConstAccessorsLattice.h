//===-- CachedConstAccessorsLattice.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the lattice mixin that additionally maintains a cache of
// stable method call return values to model const accessor member functions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CACHED_CONST_ACCESSORS_LATTICE_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CACHED_CONST_ACCESSORS_LATTICE_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace clang {
namespace dataflow {

/// A mixin for a lattice that additionally maintains a cache of stable method
/// call return values to model const accessors methods. When a non-const method
/// is called, the cache should be cleared causing the next call to a const
/// method to be considered a different value. NOTE: The user is responsible for
/// clearing the cache.
///
/// For example:
///
/// class Bar {
/// public:
///   const std::optional<Foo>& getFoo() const;
///   void clear();
/// };
//
/// void func(Bar& s) {
///   if (s.getFoo().has_value()) {
///     use(s.getFoo().value()); // safe (checked earlier getFoo())
///     s.clear();
///     use(s.getFoo().value()); // unsafe (invalidate cache for s)
///   }
/// }
template <typename Base> class CachedConstAccessorsLattice : public Base {
public:
  using Base::Base; // inherit all constructors

  /// Creates or returns a previously created `Value` associated with a const
  /// method call `obj.getFoo()` where `RecordLoc` is the
  /// `RecordStorageLocation` of `obj`.
  /// Returns nullptr if unable to find or create a value.
  ///
  /// Requirements:
  ///
  ///  - `CE` should return a value (not a reference or record type)
  Value *
  getOrCreateConstMethodReturnValue(const RecordStorageLocation &RecordLoc,
                                    const CallExpr *CE, Environment &Env);

  /// Creates or returns a previously created `StorageLocation` associated with
  /// a const method call `obj.getFoo()` where `RecordLoc` is the
  /// `RecordStorageLocation` of `obj`, `Callee` is the decl for `getFoo`.
  ///
  /// The callback `Initialize` runs on the storage location if newly created.
  ///
  /// Requirements:
  ///
  ///  - `Callee` should return a location (return type is a reference type or a
  ///     record type).
  StorageLocation &getOrCreateConstMethodReturnStorageLocation(
      const RecordStorageLocation &RecordLoc, const FunctionDecl *Callee,
      Environment &Env, llvm::function_ref<void(StorageLocation &)> Initialize);

  void clearConstMethodReturnValues(const RecordStorageLocation &RecordLoc) {
    ConstMethodReturnValues.erase(&RecordLoc);
  }

  void clearConstMethodReturnStorageLocations(
      const RecordStorageLocation &RecordLoc) {
    ConstMethodReturnStorageLocations.erase(&RecordLoc);
  }

  bool operator==(const CachedConstAccessorsLattice &Other) const {
    return Base::operator==(Other);
  }

  LatticeJoinEffect join(const CachedConstAccessorsLattice &Other);

private:
  // Maps a record storage location and const method to the value to return
  // from that const method.
  using ConstMethodReturnValuesType =
      llvm::SmallDenseMap<const RecordStorageLocation *,
                          llvm::SmallDenseMap<const FunctionDecl *, Value *>>;
  ConstMethodReturnValuesType ConstMethodReturnValues;

  // Maps a record storage location and const method to the record storage
  // location to return from that const method.
  using ConstMethodReturnStorageLocationsType = llvm::SmallDenseMap<
      const RecordStorageLocation *,
      llvm::SmallDenseMap<const FunctionDecl *, StorageLocation *>>;
  ConstMethodReturnStorageLocationsType ConstMethodReturnStorageLocations;
};

namespace internal {

template <typename T>
llvm::SmallDenseMap<const RecordStorageLocation *,
                    llvm::SmallDenseMap<const FunctionDecl *, T *>>
joinConstMethodMap(
    const llvm::SmallDenseMap<const RecordStorageLocation *,
                              llvm::SmallDenseMap<const FunctionDecl *, T *>>
        &Map1,
    const llvm::SmallDenseMap<const RecordStorageLocation *,
                              llvm::SmallDenseMap<const FunctionDecl *, T *>>
        &Map2,
    LatticeEffect &Effect) {
  llvm::SmallDenseMap<const RecordStorageLocation *,
                      llvm::SmallDenseMap<const FunctionDecl *, T *>>
      Result;
  for (auto &[Loc, DeclToT] : Map1) {
    auto It = Map2.find(Loc);
    if (It == Map2.end()) {
      Effect = LatticeJoinEffect::Changed;
      continue;
    }
    const auto &OtherDeclToT = It->second;
    auto &JoinedDeclToT = Result[Loc];
    for (auto [Func, Var] : DeclToT) {
      T *OtherVar = OtherDeclToT.lookup(Func);
      if (OtherVar == nullptr || OtherVar != Var) {
        Effect = LatticeJoinEffect::Changed;
        continue;
      }
      JoinedDeclToT.insert({Func, Var});
    }
  }
  return Result;
}

} // namespace internal

template <typename Base>
LatticeEffect CachedConstAccessorsLattice<Base>::join(
    const CachedConstAccessorsLattice<Base> &Other) {

  LatticeEffect Effect = Base::join(Other);

  // For simplicity, we only retain values that are identical, but not ones that
  // are non-identical but equivalent. This is likely to be sufficient in
  // practice, and it reduces implementation complexity considerably.

  ConstMethodReturnValues =
      clang::dataflow::internal::joinConstMethodMap<dataflow::Value>(
          ConstMethodReturnValues, Other.ConstMethodReturnValues, Effect);

  ConstMethodReturnStorageLocations =
      clang::dataflow::internal::joinConstMethodMap<dataflow::StorageLocation>(
          ConstMethodReturnStorageLocations,
          Other.ConstMethodReturnStorageLocations, Effect);

  return Effect;
}

template <typename Base>
Value *CachedConstAccessorsLattice<Base>::getOrCreateConstMethodReturnValue(
    const RecordStorageLocation &RecordLoc, const CallExpr *CE,
    Environment &Env) {
  QualType Type = CE->getType();
  assert(!Type.isNull());
  assert(!Type->isReferenceType());
  assert(!Type->isRecordType());

  auto &ObjMap = ConstMethodReturnValues[&RecordLoc];
  const FunctionDecl *DirectCallee = CE->getDirectCallee();
  if (DirectCallee == nullptr)
    return nullptr;
  auto it = ObjMap.find(DirectCallee);
  if (it != ObjMap.end())
    return it->second;

  Value *Val = Env.createValue(Type);
  if (Val != nullptr)
    ObjMap.insert({DirectCallee, Val});
  return Val;
}

template <typename Base>
StorageLocation &
CachedConstAccessorsLattice<Base>::getOrCreateConstMethodReturnStorageLocation(
    const RecordStorageLocation &RecordLoc, const FunctionDecl *Callee,
    Environment &Env, llvm::function_ref<void(StorageLocation &)> Initialize) {
  assert(Callee != nullptr);
  QualType Type = Callee->getReturnType();
  assert(!Type.isNull());
  assert(Type->isReferenceType() || Type->isRecordType());
  auto &ObjMap = ConstMethodReturnStorageLocations[&RecordLoc];
  auto it = ObjMap.find(Callee);
  if (it != ObjMap.end())
    return *it->second;

  StorageLocation &Loc = Env.createStorageLocation(Type.getNonReferenceType());
  Initialize(Loc);

  ObjMap.insert({Callee, &Loc});
  return Loc;
}

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CACHED_CONST_ACCESSORS_LATTICE_H
