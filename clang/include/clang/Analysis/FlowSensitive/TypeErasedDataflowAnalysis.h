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
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Optional.h"

namespace clang {
namespace dataflow {

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
class TypeErasedDataflowAnalysis : public Environment::Merger {
  /// Determines whether to apply the built-in transfer functions.
  // FIXME: Remove this option once the framework supports composing analyses
  // (at which point the built-in transfer functions can be simply a standalone
  // analysis).
  bool ApplyBuiltinTransfer;

public:
  TypeErasedDataflowAnalysis() : ApplyBuiltinTransfer(true) {}
  TypeErasedDataflowAnalysis(bool ApplyBuiltinTransfer)
      : ApplyBuiltinTransfer(ApplyBuiltinTransfer) {}

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

  /// Returns true if and only if the two given type-erased lattice elements are
  /// equal.
  virtual bool isEqualTypeErased(const TypeErasedLattice &,
                                 const TypeErasedLattice &) = 0;

  /// Applies the analysis transfer function for a given statement and
  /// type-erased lattice element.
  virtual void transferTypeErased(const Stmt *, TypeErasedLattice &,
                                  Environment &) = 0;

  /// Determines whether to apply the built-in transfer functions, which model
  /// the heap and stack in the `Environment`.
  bool applyBuiltinTransfer() const { return ApplyBuiltinTransfer; }
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

/// Transfers the state of a basic block by evaluating each of its statements in
/// the context of `Analysis` and the states of its predecessors that are
/// available in `BlockStates`. `HandleTransferredStmt` (if provided) will be
/// applied to each statement in the block, after it is evaluated.
///
/// Requirements:
///
///   All predecessors of `Block` except those with loop back edges must have
///   already been transferred. States in `BlockStates` that are set to
///   `llvm::None` represent basic blocks that are not evaluated yet.
TypeErasedDataflowAnalysisState transferBlock(
    const ControlFlowContext &CFCtx,
    std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> &BlockStates,
    const CFGBlock &Block, const Environment &InitEnv,
    TypeErasedDataflowAnalysis &Analysis,
    std::function<void(const CFGStmt &,
                       const TypeErasedDataflowAnalysisState &)>
        HandleTransferredStmt = nullptr);

/// Performs dataflow analysis and returns a mapping from basic block IDs to
/// dataflow analysis states that model the respective basic blocks. Indices
/// of the returned vector correspond to basic block IDs.
std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>
runTypeErasedDataflowAnalysis(const ControlFlowContext &CFCtx,
                              TypeErasedDataflowAnalysis &Analysis,
                              const Environment &InitEnv);

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TYPEERASEDDATAFLOWANALYSIS_H
