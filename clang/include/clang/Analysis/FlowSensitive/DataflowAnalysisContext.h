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
#include "llvm/ADT/DenseSet.h"
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {
namespace dataflow {

/// Skip past nodes that the CFG does not emit. These nodes are invisible to
/// flow-sensitive analysis, and should be ignored as they will effectively not
/// exist.
///
///   * `ParenExpr` - The CFG takes the operator precedence into account, but
///   otherwise omits the node afterwards.
///
///   * `ExprWithCleanups` - The CFG will generate the appropriate calls to
///   destructors and then omit the node.
///
const Expr &ignoreCFGOmittedNodes(const Expr &E);
const Stmt &ignoreCFGOmittedNodes(const Stmt &S);

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
    const Expr &CanonE = ignoreCFGOmittedNodes(E);
    assert(ExprToLoc.find(&CanonE) == ExprToLoc.end());
    ExprToLoc[&CanonE] = &Loc;
  }

  /// Returns the storage location assigned to `E` or null if `E` has no
  /// assigned storage location.
  StorageLocation *getStorageLocation(const Expr &E) const {
    auto It = ExprToLoc.find(&ignoreCFGOmittedNodes(E));
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
  BoolValue &getOrCreateConjunction(BoolValue &LHS, BoolValue &RHS);

  /// Returns a boolean value that represents the disjunction of `LHS` and
  /// `RHS`. Subsequent calls with the same arguments, regardless of their
  /// order, will return the same result. If the given boolean values represent
  /// the same value, the result will be the value itself.
  BoolValue &getOrCreateDisjunction(BoolValue &LHS, BoolValue &RHS);

  /// Returns a boolean value that represents the negation of `Val`. Subsequent
  /// calls with the same argument will return the same result.
  BoolValue &getOrCreateNegation(BoolValue &Val);

  /// Returns a boolean value that represents `LHS => RHS`. Subsequent calls
  /// with the same arguments, will return the same result. If the given boolean
  /// values represent the same value, the result will be a value that
  /// represents the true boolean literal.
  BoolValue &getOrCreateImplication(BoolValue &LHS, BoolValue &RHS);

  /// Returns a boolean value that represents `LHS <=> RHS`. Subsequent calls
  /// with the same arguments, regardless of their order, will return the same
  /// result. If the given boolean values represent the same value, the result
  /// will be a value that represents the true boolean literal.
  BoolValue &getOrCreateIff(BoolValue &LHS, BoolValue &RHS);

  /// Creates a fresh flow condition and returns a token that identifies it. The
  /// token can be used to perform various operations on the flow condition such
  /// as adding constraints to it, forking it, joining it with another flow
  /// condition, or checking implications.
  AtomicBoolValue &makeFlowConditionToken();

  /// Adds `Constraint` to the flow condition identified by `Token`.
  void addFlowConditionConstraint(AtomicBoolValue &Token,
                                  BoolValue &Constraint);

  /// Creates a new flow condition with the same constraints as the flow
  /// condition identified by `Token` and returns its token.
  AtomicBoolValue &forkFlowCondition(AtomicBoolValue &Token);

  /// Creates a new flow condition that represents the disjunction of the flow
  /// conditions identified by `FirstToken` and `SecondToken`, and returns its
  /// token.
  AtomicBoolValue &joinFlowConditions(AtomicBoolValue &FirstToken,
                                      AtomicBoolValue &SecondToken);

  /// Returns true if and only if the constraints of the flow condition
  /// identified by `Token` imply that `Val` is true.
  bool flowConditionImplies(AtomicBoolValue &Token, BoolValue &Val);

  /// Returns true if and only if the constraints of the flow condition
  /// identified by `Token` are always true.
  bool flowConditionIsTautology(AtomicBoolValue &Token);

  /// Returns true if `Val1` is equivalent to `Val2`.
  /// Note: This function doesn't take into account constraints on `Val1` and
  /// `Val2` imposed by the flow condition.
  bool equivalentBoolValues(BoolValue &Val1, BoolValue &Val2);

private:
  /// Adds all constraints of the flow condition identified by `Token` and all
  /// of its transitive dependencies to `Constraints`. `VisitedTokens` is used
  /// to track tokens of flow conditions that were already visited by recursive
  /// calls.
  void addTransitiveFlowConditionConstraints(
      AtomicBoolValue &Token, llvm::DenseSet<BoolValue *> &Constraints,
      llvm::DenseSet<AtomicBoolValue *> &VisitedTokens);

  /// Returns the result of satisfiability checking on `Constraints`.
  /// Possible return values are:
  /// - `Satisfiable`: There exists a satisfying assignment for `Constraints`.
  /// - `Unsatisfiable`: There is no satisfying assignment for `Constraints`.
  /// - `TimedOut`: The solver gives up on finding a satisfying assignment.
  Solver::Result querySolver(llvm::DenseSet<BoolValue *> Constraints);

  /// Returns true if the solver is able to prove that there is no satisfying
  /// assignment for `Constraints`
  bool isUnsatisfiable(llvm::DenseSet<BoolValue *> Constraints) {
    return querySolver(std::move(Constraints)) == Solver::Result::Unsatisfiable;
  }

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

  // Flow conditions are tracked symbolically: each unique flow condition is
  // associated with a fresh symbolic variable (token), bound to the clause that
  // defines the flow condition. Conceptually, each binding corresponds to an
  // "iff" of the form `FC <=> (C1 ^ C2 ^ ...)` where `FC` is a flow condition
  // token (an atomic boolean) and `Ci`s are the set of constraints in the flow
  // flow condition clause. The set of constraints (C1 ^ C2 ^ ...) are stored in
  // the `FlowConditionConstraints` map, keyed by the token of the flow
  // condition.
  //
  // Flow conditions depend on other flow conditions if they are created using
  // `forkFlowCondition` or `joinFlowConditions`. The graph of flow condition
  // dependencies is stored in the `FlowConditionDeps` map.
  llvm::DenseMap<AtomicBoolValue *, llvm::DenseSet<AtomicBoolValue *>>
      FlowConditionDeps;
  llvm::DenseMap<AtomicBoolValue *, BoolValue *> FlowConditionConstraints;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSISCONTEXT_H
