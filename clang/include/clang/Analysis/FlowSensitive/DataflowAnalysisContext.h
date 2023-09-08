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
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {
namespace dataflow {
class Logger;

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

/// A set of `FieldDecl *`. Use `SmallSetVector` to guarantee deterministic
/// iteration order.
using FieldSet = llvm::SmallSetVector<const FieldDecl *, 4>;

/// Returns the set of all fields in the type.
FieldSet getObjectFields(QualType Type);

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
    std::optional<ContextSensitiveOptions> ContextSensitiveOpts;

    /// If provided, analysis details will be recorded here.
    /// (This is always non-null within an AnalysisContext, the framework
    /// provides a fallback no-op logger).
    Logger *Log = nullptr;
  };

  /// Constructs a dataflow analysis context.
  ///
  /// Requirements:
  ///
  ///  `S` must not be null.
  DataflowAnalysisContext(std::unique_ptr<Solver> S,
                          Options Opts = Options{
                              /*ContextSensitiveOpts=*/std::nullopt,
                              /*Logger=*/nullptr});
  ~DataflowAnalysisContext();

  /// Returns a new storage location appropriate for `Type`.
  ///
  /// A null `Type` is interpreted as the pointee type of `std::nullptr_t`.
  StorageLocation &createStorageLocation(QualType Type);

  /// Returns a stable storage location for `D`.
  StorageLocation &getStableStorageLocation(const VarDecl &D);

  /// Returns a stable storage location for `E`.
  StorageLocation &getStableStorageLocation(const Expr &E);

  /// Returns a pointer value that represents a null pointer. Calls with
  /// `PointeeType` that are canonically equivalent will return the same result.
  /// A null `PointeeType` can be used for the pointee of `std::nullptr_t`.
  PointerValue &getOrCreateNullPointerValue(QualType PointeeType);

  /// Adds `Constraint` to the flow condition identified by `Token`.
  void addFlowConditionConstraint(Atom Token, const Formula &Constraint);

  /// Creates a new flow condition with the same constraints as the flow
  /// condition identified by `Token` and returns its token.
  Atom forkFlowCondition(Atom Token);

  /// Creates a new flow condition that represents the disjunction of the flow
  /// conditions identified by `FirstToken` and `SecondToken`, and returns its
  /// token.
  Atom joinFlowConditions(Atom FirstToken, Atom SecondToken);

  /// Returns true if and only if the constraints of the flow condition
  /// identified by `Token` imply that `Val` is true.
  bool flowConditionImplies(Atom Token, const Formula &);

  /// Returns true if and only if the constraints of the flow condition
  /// identified by `Token` are always true.
  bool flowConditionIsTautology(Atom Token);

  /// Returns true if `Val1` is equivalent to `Val2`.
  /// Note: This function doesn't take into account constraints on `Val1` and
  /// `Val2` imposed by the flow condition.
  bool equivalentFormulas(const Formula &Val1, const Formula &Val2);

  LLVM_DUMP_METHOD void dumpFlowCondition(Atom Token,
                                          llvm::raw_ostream &OS = llvm::dbgs());

  /// Returns the `ControlFlowContext` registered for `F`, if any. Otherwise,
  /// returns null.
  const ControlFlowContext *getControlFlowContext(const FunctionDecl *F);

  const Options &getOptions() { return Opts; }

  Arena &arena() { return *A; }

  /// Returns the outcome of satisfiability checking on `Constraints`.
  ///
  /// Flow conditions are not incorporated, so they may need to be manually
  /// included in `Constraints` to provide contextually-accurate results, e.g.
  /// if any definitions or relationships of the values in `Constraints` have
  /// been stored in flow conditions.
  Solver::Result querySolver(llvm::SetVector<const Formula *> Constraints);

  /// Returns the fields of `Type`, limited to the set of fields modeled by this
  /// context.
  FieldSet getModeledFields(QualType Type);

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

  // Extends the set of modeled field declarations.
  void addModeledFields(const FieldSet &Fields);

  /// Adds all constraints of the flow condition identified by `Token` and all
  /// of its transitive dependencies to `Constraints`. `VisitedTokens` is used
  /// to track tokens of flow conditions that were already visited by recursive
  /// calls.
  void addTransitiveFlowConditionConstraints(
      Atom Token, llvm::SetVector<const Formula *> &Constraints,
      llvm::DenseSet<Atom> &VisitedTokens);

  /// Returns true if the solver is able to prove that there is no satisfying
  /// assignment for `Constraints`
  bool isUnsatisfiable(llvm::SetVector<const Formula *> Constraints) {
    return querySolver(std::move(Constraints)).getStatus() ==
           Solver::Result::Status::Unsatisfiable;
  }

  std::unique_ptr<Solver> S;
  std::unique_ptr<Arena> A;

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

  Options Opts;

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
  llvm::DenseMap<Atom, llvm::DenseSet<Atom>> FlowConditionDeps;
  llvm::DenseMap<Atom, const Formula *> FlowConditionConstraints;

  llvm::DenseMap<const FunctionDecl *, ControlFlowContext> FunctionContexts;

  // Fields modeled by environments covered by this context.
  FieldSet ModeledFields;

  std::unique_ptr<Logger> LogOwner; // If created via flags.
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSISCONTEXT_H
