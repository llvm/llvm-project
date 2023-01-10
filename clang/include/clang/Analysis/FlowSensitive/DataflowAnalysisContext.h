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
#include "clang/AST/TypeOrdering.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Compiler.h"
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

/// Returns the set of all fields in the type.
llvm::DenseSet<const FieldDecl *> getObjectFields(QualType Type);

struct ContextSensitiveOptions {
  /// The maximum depth to analyze. A value of zero is equivalent to disabling
  /// context-sensitive analysis entirely.
  unsigned Depth = 2;
};

/// Owns objects that encompass the state of a program and stores context that
/// is used during dataflow analysis.
class DataflowAnalysisContext {
public:
  struct Options {
    /// Options for analyzing function bodies when present in the translation
    /// unit, or empty to disable context-sensitive analysis. Note that this is
    /// fundamentally limited: some constructs, such as recursion, are
    /// explicitly unsupported.
    llvm::Optional<ContextSensitiveOptions> ContextSensitiveOpts;
  };

  /// Constructs a dataflow analysis context.
  ///
  /// Requirements:
  ///
  ///  `S` must not be null.
  DataflowAnalysisContext(std::unique_ptr<Solver> S,
                          Options Opts = Options{
                              /*ContextSensitiveOpts=*/std::nullopt})
      : S(std::move(S)), TrueVal(createAtomicBoolValue()),
        FalseVal(createAtomicBoolValue()), Opts(Opts) {
    assert(this->S != nullptr);
  }

  /// Takes ownership of `Loc` and returns a reference to it.
  ///
  /// Requirements:
  ///
  ///  `Loc` must not be null.
  template <typename T>
  std::enable_if_t<std::is_base_of<StorageLocation, T>::value, T &>
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
  std::enable_if_t<std::is_base_of<Value, T>::value, T &>
  takeOwnership(std::unique_ptr<T> Val) {
    assert(Val != nullptr);
    Vals.push_back(std::move(Val));
    return *cast<T>(Vals.back().get());
  }

  /// Returns a new storage location appropriate for `Type`.
  ///
  /// A null `Type` is interpreted as the pointee type of `std::nullptr_t`.
  StorageLocation &createStorageLocation(QualType Type);

  /// Returns a stable storage location for `D`.
  StorageLocation &getStableStorageLocation(const VarDecl &D);

  /// Returns a stable storage location for `E`.
  StorageLocation &getStableStorageLocation(const Expr &E);

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

  /// Returns a pointer value that represents a null pointer. Calls with
  /// `PointeeType` that are canonically equivalent will return the same result.
  /// A null `PointeeType` can be used for the pointee of `std::nullptr_t`.
  PointerValue &getOrCreateNullPointerValue(QualType PointeeType);

  /// Returns a symbolic boolean value that models a boolean literal equal to
  /// `Value`.
  AtomicBoolValue &getBoolLiteralValue(bool Value) const {
    return Value ? TrueVal : FalseVal;
  }

  /// Creates an atomic boolean value.
  AtomicBoolValue &createAtomicBoolValue() {
    return takeOwnership(std::make_unique<AtomicBoolValue>());
  }

  /// Creates a Top value for booleans. Each instance is unique and can be
  /// assigned a distinct truth value during solving.
  ///
  /// FIXME: `Top iff Top` is true when both Tops are identical (by pointer
  /// equality), but not when they are distinct values. We should improve the
  /// implementation so that `Top iff Top` has a consistent meaning, regardless
  /// of the identity of `Top`. Moreover, I think the meaning should be
  /// `false`.
  TopBoolValue &createTopBoolValue() {
    return takeOwnership(std::make_unique<TopBoolValue>());
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

  // FIXME: This function returns the flow condition expressed directly as its
  // constraints: (C1 AND C2 AND ...). This differs from the general approach in
  // the framework where a flow condition is represented as a token (an atomic
  // boolean) with dependencies and constraints tracked in `FlowConditionDeps`
  // and `FlowConditionConstraints`: (FC <=> C1 AND C2 AND ...).
  // Consider if we should make the representation of flow condition consistent,
  // returning an atomic boolean token with separate constraints instead.
  //
  /// Builds and returns the logical formula defining the flow condition
  /// identified by `Token`. If a value in the formula is present as a key in
  /// `Substitutions`, it will be substituted with the value it maps to.
  /// As an example, say we have flow condition tokens FC1, FC2, FC3 and
  /// FlowConditionConstraints: { FC1: C1,
  ///                             FC2: C2,
  ///                             FC3: (FC1 v FC2) ^ C3 }
  /// buildAndSubstituteFlowCondition(FC3, {{C1 -> C1'}}) will return a value
  /// corresponding to (C1' v C2) ^ C3.
  BoolValue &buildAndSubstituteFlowCondition(
      AtomicBoolValue &Token,
      llvm::DenseMap<AtomicBoolValue *, BoolValue *> Substitutions);

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

  LLVM_DUMP_METHOD void dumpFlowCondition(AtomicBoolValue &Token);

  /// Returns the `ControlFlowContext` registered for `F`, if any. Otherwise,
  /// returns null.
  const ControlFlowContext *getControlFlowContext(const FunctionDecl *F);

  void addFieldsReferencedInScope(llvm::DenseSet<const FieldDecl *> Fields);

  const Options &getOptions() { return Opts; }

private:
  friend class Environment;

  struct NullableQualTypeDenseMapInfo : private llvm::DenseMapInfo<QualType> {
    static QualType getEmptyKey() {
      // Allow a NULL `QualType` by using a different value as the empty key.
      return QualType::getFromOpaquePtr(reinterpret_cast<Type *>(1));
    }

    using DenseMapInfo::getHashValue;
    using DenseMapInfo::getTombstoneKey;
    using DenseMapInfo::isEqual;
  };

  /// Returns the subset of fields of `Type` that are referenced in the scope of
  /// the analysis.
  llvm::DenseSet<const FieldDecl *> getReferencedFields(QualType Type);

  /// Adds all constraints of the flow condition identified by `Token` and all
  /// of its transitive dependencies to `Constraints`. `VisitedTokens` is used
  /// to track tokens of flow conditions that were already visited by recursive
  /// calls.
  void addTransitiveFlowConditionConstraints(
      AtomicBoolValue &Token, llvm::DenseSet<BoolValue *> &Constraints,
      llvm::DenseSet<AtomicBoolValue *> &VisitedTokens);

  /// Returns the outcome of satisfiability checking on `Constraints`.
  /// Possible outcomes are:
  /// - `Satisfiable`: A satisfying assignment exists and is returned.
  /// - `Unsatisfiable`: A satisfying assignment does not exist.
  /// - `TimedOut`: The search for a satisfying assignment was not completed.
  Solver::Result querySolver(llvm::DenseSet<BoolValue *> Constraints);

  /// Returns true if the solver is able to prove that there is no satisfying
  /// assignment for `Constraints`
  bool isUnsatisfiable(llvm::DenseSet<BoolValue *> Constraints) {
    return querySolver(std::move(Constraints)).getStatus() ==
           Solver::Result::Status::Unsatisfiable;
  }

  /// Returns a boolean value as a result of substituting `Val` and its sub
  /// values based on entries in `SubstitutionsCache`. Intermediate results are
  /// stored in `SubstitutionsCache` to avoid reprocessing values that have
  /// already been visited.
  BoolValue &substituteBoolValue(
      BoolValue &Val,
      llvm::DenseMap<BoolValue *, BoolValue *> &SubstitutionsCache);

  /// Builds and returns the logical formula defining the flow condition
  /// identified by `Token`, sub values may be substituted based on entries in
  /// `SubstitutionsCache`. Intermediate results are stored in
  /// `SubstitutionsCache` to avoid reprocessing values that have already been
  /// visited.
  BoolValue &buildAndSubstituteFlowConditionWithCache(
      AtomicBoolValue &Token,
      llvm::DenseMap<BoolValue *, BoolValue *> &SubstitutionsCache);

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

  // Null pointer values, keyed by the canonical pointee type.
  //
  // FIXME: The pointer values are indexed by the pointee types which are
  // required to initialize the `PointeeLoc` field in `PointerValue`. Consider
  // creating a type-independent `NullPointerValue` without a `PointeeLoc`
  // field.
  llvm::DenseMap<QualType, PointerValue *, NullableQualTypeDenseMapInfo>
      NullPointerVals;

  AtomicBoolValue &TrueVal;
  AtomicBoolValue &FalseVal;

  Options Opts;

  // Indices that are used to avoid recreating the same composite boolean
  // values.
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, ConjunctionValue *>
      ConjunctionVals;
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, DisjunctionValue *>
      DisjunctionVals;
  llvm::DenseMap<BoolValue *, NegationValue *> NegationVals;
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, ImplicationValue *>
      ImplicationVals;
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, BiconditionalValue *>
      BiconditionalVals;

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

  llvm::DenseMap<const FunctionDecl *, ControlFlowContext> FunctionContexts;

  // All fields referenced (statically) in the scope of the analysis.
  llvm::DenseSet<const FieldDecl *> FieldsReferencedInScope;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSISCONTEXT_H
