//===-- DataflowAnalysisContext.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a DataflowAnalysisContext class that owns objects that
//  encompass the state of a program and stores context that is used during
//  dataflow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSISCONTEXT_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSISCONTEXT_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {
namespace dataflow {

/// Owns objects that encompass the state of a program and stores context that
/// is used during dataflow analysis.
class DataflowAnalysisContext {
public:
  /// Constructs a dataflow analysis context.
  ///
  /// Requirements:
  ///
  ///  `S` must not be null.
  DataflowAnalysisContext(std::unique_ptr<Solver> S)
      : S(std::move(S)), TrueVal(createAtomicBoolValue()),
        FalseVal(createAtomicBoolValue()) {
    assert(this->S != nullptr);
  }

  /// Returns the SAT solver instance that is available in this context.
  Solver &getSolver() const { return *S; }

  /// Takes ownership of `Loc` and returns a reference to it.
  ///
  /// Requirements:
  ///
  ///  `Loc` must not be null.
  template <typename T>
  typename std::enable_if<std::is_base_of<StorageLocation, T>::value, T &>::type
  takeOwnership(std::unique_ptr<T> Loc) {
    assert(Loc != nullptr);
    Locs.push_back(std::move(Loc));
    return *cast<T>(Locs.back().get());
  }

  /// Takes ownership of `Val` and returns a reference to it.
  ///
  /// Requirements:
  ///
  ///  `Val` must not be null.
  template <typename T>
  typename std::enable_if<std::is_base_of<Value, T>::value, T &>::type
  takeOwnership(std::unique_ptr<T> Val) {
    assert(Val != nullptr);
    Vals.push_back(std::move(Val));
    return *cast<T>(Vals.back().get());
  }

  /// Assigns `Loc` as the storage location of `D`.
  ///
  /// Requirements:
  ///
  ///  `D` must not be assigned a storage location.
  void setStorageLocation(const ValueDecl &D, StorageLocation &Loc) {
    assert(DeclToLoc.find(&D) == DeclToLoc.end());
    DeclToLoc[&D] = &Loc;
  }

  /// Returns the storage location assigned to `D` or null if `D` has no
  /// assigned storage location.
  StorageLocation *getStorageLocation(const ValueDecl &D) const {
    auto It = DeclToLoc.find(&D);
    return It == DeclToLoc.end() ? nullptr : It->second;
  }

  /// Assigns `Loc` as the storage location of `E`.
  ///
  /// Requirements:
  ///
  ///  `E` must not be assigned a storage location.
  void setStorageLocation(const Expr &E, StorageLocation &Loc) {
    assert(ExprToLoc.find(&E) == ExprToLoc.end());
    ExprToLoc[&E] = &Loc;
  }

  /// Returns the storage location assigned to `E` or null if `E` has no
  /// assigned storage location.
  StorageLocation *getStorageLocation(const Expr &E) const {
    auto It = ExprToLoc.find(&E);
    return It == ExprToLoc.end() ? nullptr : It->second;
  }

  /// Assigns `Loc` as the storage location of the `this` pointee.
  ///
  /// Requirements:
  ///
  ///  The `this` pointee must not be assigned a storage location.
  void setThisPointeeStorageLocation(StorageLocation &Loc) {
    assert(ThisPointeeLoc == nullptr);
    ThisPointeeLoc = &Loc;
  }

  /// Returns the storage location assigned to the `this` pointee or null if the
  /// `this` pointee has no assigned storage location.
  StorageLocation *getThisPointeeStorageLocation() const {
    return ThisPointeeLoc;
  }

  /// Returns a symbolic boolean value that models a boolean literal equal to
  /// `Value`.
  AtomicBoolValue &getBoolLiteralValue(bool Value) const {
    return Value ? TrueVal : FalseVal;
  }

  /// Creates an atomic boolean value.
  AtomicBoolValue &createAtomicBoolValue() {
    return takeOwnership(std::make_unique<AtomicBoolValue>());
  }

  /// Returns a boolean value that represents the conjunction of `LHS` and
  /// `RHS`. Subsequent calls with the same arguments, regardless of their
  /// order, will return the same result. If the given boolean values represent
  /// the same value, the result will be the value itself.
  BoolValue &getOrCreateConjunctionValue(BoolValue &LHS, BoolValue &RHS);

  /// Returns a boolean value that represents the disjunction of `LHS` and
  /// `RHS`. Subsequent calls with the same arguments, regardless of their
  /// order, will return the same result. If the given boolean values represent
  /// the same value, the result will be the value itself.
  BoolValue &getOrCreateDisjunctionValue(BoolValue &LHS, BoolValue &RHS);

  /// Returns a boolean value that represents the negation of `Val`. Subsequent
  /// calls with the same argument will return the same result.
  BoolValue &getOrCreateNegationValue(BoolValue &Val);

private:
  std::unique_ptr<Solver> S;

  // Storage for the state of a program.
  std::vector<std::unique_ptr<StorageLocation>> Locs;
  std::vector<std::unique_ptr<Value>> Vals;

  // Maps from program declarations and statements to storage locations that are
  // assigned to them. These assignments are global (aggregated across all basic
  // blocks) and are used to produce stable storage locations when the same
  // basic blocks are evaluated multiple times. The storage locations that are
  // in scope for a particular basic block are stored in `Environment`.
  llvm::DenseMap<const ValueDecl *, StorageLocation *> DeclToLoc;
  llvm::DenseMap<const Expr *, StorageLocation *> ExprToLoc;

  StorageLocation *ThisPointeeLoc = nullptr;

  AtomicBoolValue &TrueVal;
  AtomicBoolValue &FalseVal;

  // Indices that are used to avoid recreating the same composite boolean
  // values.
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, ConjunctionValue *>
      ConjunctionVals;
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, DisjunctionValue *>
      DisjunctionVals;
  llvm::DenseMap<BoolValue *, NegationValue *> NegationVals;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSISCONTEXT_H
