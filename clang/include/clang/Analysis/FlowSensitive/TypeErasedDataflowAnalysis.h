//===- TypeErasedDataflowAnalysis.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines type-erased base types and functions for building dataflow
//  analyses that run over Control-Flow Graphs (CFGs).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TYPEERASEDDATAFLOWANALYSIS_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TYPEERASEDDATAFLOWANALYSIS_H

#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace dataflow {

struct DataflowAnalysisOptions {
  /// Options for the built-in model, or empty to not apply them.
  // FIXME: Remove this option once the framework supports composing analyses
  // (at which point the built-in transfer functions can be simply a standalone
  // analysis).
  std::optional<DataflowAnalysisContext::Options> BuiltinOpts =
      DataflowAnalysisContext::Options{};
};

/// Type-erased lattice element container.
///
/// Requirements:
///
///  The type of the object stored in the container must be a bounded
///  join-semilattice.
struct TypeErasedLattice {
  llvm::Any Value;
};

/// Type-erased base class for dataflow analyses built on a single lattice type.
class TypeErasedDataflowAnalysis : public Environment::ValueModel {
  DataflowAnalysisOptions Options;

public:
  TypeErasedDataflowAnalysis() : Options({}) {}

  TypeErasedDataflowAnalysis(DataflowAnalysisOptions Options)
      : Options(Options) {}

  virtual ~TypeErasedDataflowAnalysis() {}

  /// Returns the `ASTContext` that is used by the analysis.
  virtual ASTContext &getASTContext() = 0;

  /// Returns a type-erased lattice element that models the initial state of a
  /// basic block.
  virtual TypeErasedLattice typeErasedInitialElement() = 0;

  /// Joins two type-erased lattice elements by computing their least upper
  /// bound. Places the join result in the left element and returns an effect
  /// indicating whether any changes were made to it.
  virtual LatticeJoinEffect joinTypeErased(TypeErasedLattice &,
                                           const TypeErasedLattice &) = 0;

  /// Chooses a lattice element that approximates the current element at a
  /// program point, given the previous element at that point. Places the
  /// widened result in the current element (`Current`). Widening is optional --
  /// it is only needed to either accelerate convergence (for lattices with
  /// non-trivial height) or guarantee convergence (for lattices with infinite
  /// height).
  ///
  /// Returns an indication of whether any changes were made to `Current` in
  /// order to widen. This saves a separate call to `isEqualTypeErased` after
  /// the widening.
  virtual LatticeJoinEffect
  widenTypeErased(TypeErasedLattice &Current,
                  const TypeErasedLattice &Previous) = 0;

  /// Returns true if and only if the two given type-erased lattice elements are
  /// equal.
  virtual bool isEqualTypeErased(const TypeErasedLattice &,
                                 const TypeErasedLattice &) = 0;

  /// Applies the analysis transfer function for a given control flow graph
  /// element and type-erased lattice element.
  virtual void transferTypeErased(const CFGElement *, TypeErasedLattice &,
                                  Environment &) = 0;

  /// Applies the analysis transfer function for a given edge from a CFG block
  /// of a conditional statement.
  /// @param Stmt The condition which is responsible for the split in the CFG.
  /// @param Branch True if the edge goes to the basic block where the
  /// condition is true.
  virtual void transferBranchTypeErased(bool Branch, const Stmt *,
                                        TypeErasedLattice &, Environment &) = 0;

  /// If the built-in model is enabled, returns the options to be passed to
  /// them. Otherwise returns empty.
  const std::optional<DataflowAnalysisContext::Options> &
  builtinOptions() const {
    return Options.BuiltinOpts;
  }
};

/// Type-erased model of the program at a given program point.
struct TypeErasedDataflowAnalysisState {
  /// Type-erased model of a program property.
  TypeErasedLattice Lattice;

  /// Model of the state of the program (store and heap).
  Environment Env;

  TypeErasedDataflowAnalysisState(TypeErasedLattice Lattice, Environment Env)
      : Lattice(std::move(Lattice)), Env(std::move(Env)) {}
};

/// Transfers the state of a basic block by evaluating each of its elements in
/// the context of `Analysis` and the states of its predecessors that are
/// available in `BlockStates`. `PostVisitCFG` (if provided) will be applied to
/// each element in the block, after it is evaluated.
///
/// Requirements:
///
///   All predecessors of `Block` except those with loop back edges must have
///   already been transferred. States in `BlockStates` that are set to
///   `std::nullopt` represent basic blocks that are not evaluated yet.
TypeErasedDataflowAnalysisState transferBlock(
    const ControlFlowContext &CFCtx,
    llvm::ArrayRef<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates,
    const CFGBlock &Block, const Environment &InitEnv,
    TypeErasedDataflowAnalysis &Analysis,
    std::function<void(const CFGElement &,
                       const TypeErasedDataflowAnalysisState &)>
        PostVisitCFG = nullptr);

/// Performs dataflow analysis and returns a mapping from basic block IDs to
/// dataflow analysis states that model the respective basic blocks. Indices of
/// the returned vector correspond to basic block IDs. Returns an error if the
/// dataflow analysis cannot be performed successfully. Otherwise, calls
/// `PostVisitCFG` on each CFG element with the final analysis results at that
/// program point.
llvm::Expected<std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>>
runTypeErasedDataflowAnalysis(
    const ControlFlowContext &CFCtx, TypeErasedDataflowAnalysis &Analysis,
    const Environment &InitEnv,
    std::function<void(const CFGElement &,
                       const TypeErasedDataflowAnalysisState &)>
        PostVisitCFG = nullptr);

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TYPEERASEDDATAFLOWANALYSIS_H
