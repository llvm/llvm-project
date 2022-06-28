//===-- DataflowEnvironment.h -----------------------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWENVIRONMENT_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWENVIRONMENT_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <memory>
#include <type_traits>
#include <utility>

namespace clang {
namespace dataflow {

/// Indicates what kind of indirections should be skipped past when retrieving
/// storage locations or values.
///
/// FIXME: Consider renaming this or replacing it with a more appropriate model.
/// See the discussion in https://reviews.llvm.org/D116596 for context.
enum class SkipPast {
  /// No indirections should be skipped past.
  None,
  /// An optional reference should be skipped past.
  Reference,
  /// An optional reference should be skipped past, then an optional pointer
  /// should be skipped past.
  ReferenceThenPointer,
};

/// Holds the state of the program (store and heap) at a given program point.
///
/// WARNING: Symbolic values that are created by the environment for static
/// local and global variables are not currently invalidated on function calls.
/// This is unsound and should be taken into account when designing dataflow
/// analyses.
class Environment {
public:
  /// Supplements `Environment` with non-standard comparison and join
  /// operations.
  class ValueModel {
  public:
    virtual ~ValueModel() = default;

    /// Returns true if and only if `Val1` is equivalent to `Val2`.
    ///
    /// Requirements:
    ///
    ///  `Val1` and `Val2` must be distinct.
    ///
    ///  `Val1` and `Val2` must model values of type `Type`.
    ///
    ///  `Val1` and `Val2` must be assigned to the same storage location in
    ///  `Env1` and `Env2` respectively.
    virtual bool compareEquivalent(QualType Type, const Value &Val1,
                                   const Environment &Env1, const Value &Val2,
                                   const Environment &Env2) {
      // FIXME: Consider adding QualType to StructValue and removing the Type
      // argument here.
      //
      // FIXME: default to a sound comparison and/or expand the comparison logic
      // built into the framework to support broader forms of equivalence than
      // strict pointer equality.
      return true;
    }

    /// Modifies `MergedVal` to approximate both `Val1` and `Val2`. This could
    /// be a strict lattice join or a more general widening operation.
    ///
    /// If this function returns true, `MergedVal` will be assigned to a storage
    /// location of type `Type` in `MergedEnv`.
    ///
    /// `Env1` and `Env2` can be used to query child values and path condition
    /// implications of `Val1` and `Val2` respectively.
    ///
    /// Requirements:
    ///
    ///  `Val1` and `Val2` must be distinct.
    ///
    ///  `Val1`, `Val2`, and `MergedVal` must model values of type `Type`.
    ///
    ///  `Val1` and `Val2` must be assigned to the same storage location in
    ///  `Env1` and `Env2` respectively.
    virtual bool merge(QualType Type, const Value &Val1,
                       const Environment &Env1, const Value &Val2,
                       const Environment &Env2, Value &MergedVal,
                       Environment &MergedEnv) {
      return true;
    }
  };

  /// Creates an environment that uses `DACtx` to store objects that encompass
  /// the state of a program.
  explicit Environment(DataflowAnalysisContext &DACtx);

  Environment(const Environment &Other);
  Environment &operator=(const Environment &Other);

  Environment(Environment &&Other) = default;
  Environment &operator=(Environment &&Other) = default;

  /// Creates an environment that uses `DACtx` to store objects that encompass
  /// the state of a program.
  ///
  /// If `DeclCtx` is a function, initializes the environment with symbolic
  /// representations of the function parameters.
  ///
  /// If `DeclCtx` is a non-static member function, initializes the environment
  /// with a symbolic representation of the `this` pointee.
  Environment(DataflowAnalysisContext &DACtx, const DeclContext &DeclCtx);

  /// Returns true if and only if the environment is equivalent to `Other`, i.e
  /// the two environments:
  ///  - have the same mappings from declarations to storage locations,
  ///  - have the same mappings from expressions to storage locations,
  ///  - have the same or equivalent (according to `Model`) values assigned to
  ///    the same storage locations.
  ///
  /// Requirements:
  ///
  ///  `Other` and `this` must use the same `DataflowAnalysisContext`.
  bool equivalentTo(const Environment &Other,
                    Environment::ValueModel &Model) const;

  /// Joins the environment with `Other` by taking the intersection of storage
  /// locations and values that are stored in them. Distinct values that are
  /// assigned to the same storage locations in the environment and `Other` are
  /// merged using `Model`.
  ///
  /// Requirements:
  ///
  ///  `Other` and `this` must use the same `DataflowAnalysisContext`.
  LatticeJoinEffect join(const Environment &Other,
                         Environment::ValueModel &Model);

  // FIXME: Rename `createOrGetStorageLocation` to `getOrCreateStorageLocation`,
  // `getStableStorageLocation`, or something more appropriate.

  /// Creates a storage location appropriate for `Type`. Does not assign a value
  /// to the returned storage location in the environment.
  ///
  /// Requirements:
  ///
  ///  `Type` must not be null.
  StorageLocation &createStorageLocation(QualType Type);

  /// Creates a storage location for `D`. Does not assign the returned storage
  /// location to `D` in the environment. Does not assign a value to the
  /// returned storage location in the environment.
  StorageLocation &createStorageLocation(const VarDecl &D);

  /// Creates a storage location for `E`. Does not assign the returned storage
  /// location to `E` in the environment. Does not assign a value to the
  /// returned storage location in the environment.
  StorageLocation &createStorageLocation(const Expr &E);

  /// Assigns `Loc` as the storage location of `D` in the environment.
  ///
  /// Requirements:
  ///
  ///  `D` must not be assigned a storage location in the environment.
  void setStorageLocation(const ValueDecl &D, StorageLocation &Loc);

  /// Returns the storage location assigned to `D` in the environment, applying
  /// the `SP` policy for skipping past indirections, or null if `D` isn't
  /// assigned a storage location in the environment.
  StorageLocation *getStorageLocation(const ValueDecl &D, SkipPast SP) const;

  /// Assigns `Loc` as the storage location of `E` in the environment.
  ///
  /// Requirements:
  ///
  ///  `E` must not be assigned a storage location in the environment.
  void setStorageLocation(const Expr &E, StorageLocation &Loc);

  /// Returns the storage location assigned to `E` in the environment, applying
  /// the `SP` policy for skipping past indirections, or null if `E` isn't
  /// assigned a storage location in the environment.
  StorageLocation *getStorageLocation(const Expr &E, SkipPast SP) const;

  /// Returns the storage location assigned to the `this` pointee in the
  /// environment or null if the `this` pointee has no assigned storage location
  /// in the environment.
  StorageLocation *getThisPointeeStorageLocation() const;

  /// Returns a pointer value that represents a null pointer. Calls with
  /// `PointeeType` that are canonically equivalent will return the same result.
  PointerValue &getOrCreateNullPointerValue(QualType PointeeType);

  /// Creates a value appropriate for `Type`, if `Type` is supported, otherwise
  /// return null. If `Type` is a pointer or reference type, creates all the
  /// necessary storage locations and values for indirections until it finds a
  /// non-pointer/non-reference type.
  ///
  /// Requirements:
  ///
  ///  `Type` must not be null.
  Value *createValue(QualType Type);

  /// Assigns `Val` as the value of `Loc` in the environment.
  void setValue(const StorageLocation &Loc, Value &Val);

  /// Returns the value assigned to `Loc` in the environment or null if `Loc`
  /// isn't assigned a value in the environment.
  Value *getValue(const StorageLocation &Loc) const;

  /// Equivalent to `getValue(getStorageLocation(D, SP), SkipPast::None)` if `D`
  /// is assigned a storage location in the environment, otherwise returns null.
  Value *getValue(const ValueDecl &D, SkipPast SP) const;

  /// Equivalent to `getValue(getStorageLocation(E, SP), SkipPast::None)` if `E`
  /// is assigned a storage location in the environment, otherwise returns null.
  Value *getValue(const Expr &E, SkipPast SP) const;

  /// Transfers ownership of `Loc` to the analysis context and returns a
  /// reference to it.
  ///
  /// Requirements:
  ///
  ///  `Loc` must not be null.
  template <typename T>
  typename std::enable_if<std::is_base_of<StorageLocation, T>::value, T &>::type
  takeOwnership(std::unique_ptr<T> Loc) {
    return DACtx->takeOwnership(std::move(Loc));
  }

  /// Transfers ownership of `Val` to the analysis context and returns a
  /// reference to it.
  ///
  /// Requirements:
  ///
  ///  `Val` must not be null.
  template <typename T>
  typename std::enable_if<std::is_base_of<Value, T>::value, T &>::type
  takeOwnership(std::unique_ptr<T> Val) {
    return DACtx->takeOwnership(std::move(Val));
  }

  /// Returns a symbolic boolean value that models a boolean literal equal to
  /// `Value`
  AtomicBoolValue &getBoolLiteralValue(bool Value) const {
    return DACtx->getBoolLiteralValue(Value);
  }

  /// Returns an atomic boolean value.
  BoolValue &makeAtomicBoolValue() const {
    return DACtx->createAtomicBoolValue();
  }

  /// Returns a boolean value that represents the conjunction of `LHS` and
  /// `RHS`. Subsequent calls with the same arguments, regardless of their
  /// order, will return the same result. If the given boolean values represent
  /// the same value, the result will be the value itself.
  BoolValue &makeAnd(BoolValue &LHS, BoolValue &RHS) const {
    return DACtx->getOrCreateConjunction(LHS, RHS);
  }

  /// Returns a boolean value that represents the disjunction of `LHS` and
  /// `RHS`. Subsequent calls with the same arguments, regardless of their
  /// order, will return the same result. If the given boolean values represent
  /// the same value, the result will be the value itself.
  BoolValue &makeOr(BoolValue &LHS, BoolValue &RHS) const {
    return DACtx->getOrCreateDisjunction(LHS, RHS);
  }

  /// Returns a boolean value that represents the negation of `Val`. Subsequent
  /// calls with the same argument will return the same result.
  BoolValue &makeNot(BoolValue &Val) const {
    return DACtx->getOrCreateNegation(Val);
  }

  /// Returns a boolean value represents `LHS` => `RHS`. Subsequent calls with
  /// the same arguments, will return the same result. If the given boolean
  /// values represent the same value, the result will be a value that
  /// represents the true boolean literal.
  BoolValue &makeImplication(BoolValue &LHS, BoolValue &RHS) const {
    return DACtx->getOrCreateImplication(LHS, RHS);
  }

  /// Returns a boolean value represents `LHS` <=> `RHS`. Subsequent calls with
  /// the same arguments, regardless of their order, will return the same
  /// result. If the given boolean values represent the same value, the result
  /// will be a value that represents the true boolean literal.
  BoolValue &makeIff(BoolValue &LHS, BoolValue &RHS) const {
    return DACtx->getOrCreateIff(LHS, RHS);
  }

  /// Returns the token that identifies the flow condition of the environment.
  AtomicBoolValue &getFlowConditionToken() const { return *FlowConditionToken; }

  /// Adds `Val` to the set of clauses that constitute the flow condition.
  void addToFlowCondition(BoolValue &Val);

  /// Returns true if and only if the clauses that constitute the flow condition
  /// imply that `Val` is true.
  bool flowConditionImplies(BoolValue &Val) const;

private:
  /// Creates a value appropriate for `Type`, if `Type` is supported, otherwise
  /// return null.
  ///
  /// Recursively initializes storage locations and values until it sees a
  /// self-referential pointer or reference type. `Visited` is used to track
  /// which types appeared in the reference/pointer chain in order to avoid
  /// creating a cyclic dependency with self-referential pointers/references.
  ///
  /// Requirements:
  ///
  ///  `Type` must not be null.
  Value *createValueUnlessSelfReferential(QualType Type,
                                          llvm::DenseSet<QualType> &Visited,
                                          int Depth, int &CreatedValuesCount);

  StorageLocation &skip(StorageLocation &Loc, SkipPast SP) const;
  const StorageLocation &skip(const StorageLocation &Loc, SkipPast SP) const;

  // `DACtx` is not null and not owned by this object.
  DataflowAnalysisContext *DACtx;

  // Maps from program declarations and statements to storage locations that are
  // assigned to them. Unlike the maps in `DataflowAnalysisContext`, these
  // include only storage locations that are in scope for a particular basic
  // block.
  llvm::DenseMap<const ValueDecl *, StorageLocation *> DeclToLoc;
  llvm::DenseMap<const Expr *, StorageLocation *> ExprToLoc;

  llvm::DenseMap<const StorageLocation *, Value *> LocToVal;

  // Maps locations of struct members to symbolic values of the structs that own
  // them and the decls of the struct members.
  llvm::DenseMap<const StorageLocation *,
                 std::pair<StructValue *, const ValueDecl *>>
      MemberLocToStruct;

  AtomicBoolValue *FlowConditionToken;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWENVIRONMENT_H
