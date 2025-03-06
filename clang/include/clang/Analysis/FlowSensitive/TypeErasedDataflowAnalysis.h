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

#include <optional>
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/AdornedCFG.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
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

/// Base class for dataflow analyses built on a single lattice type. The
/// framework guarantees that the type of lattice elements passed to `transfer`
/// and `transferBranch` always matches the type of elements provided by
/// `initialElement`.
class DataflowAnalysis : public Environment::ValueModel {
  ASTContext &Context;
  DataflowAnalysisOptions Options;

public:
  explicit DataflowAnalysis(ASTContext &Context) : Context(Context) {}

  explicit DataflowAnalysis(ASTContext &Context,
                            DataflowAnalysisOptions Options)
      : Context(Context), Options(Options) {}

  /// Returns the `ASTContext` that is used by the analysis.
  ASTContext &getASTContext() { return Context; }

  /// Returns a lattice element that models the initial state of a
  /// basic block.
  virtual DataflowLatticePtr initialElement() = 0;

  /// Applies the analysis transfer function for a given control flow graph
  /// element and type-erased lattice element. The derived type of Lattice will
  /// always match the type of the lattice returned by `initialElement`.
  virtual void transfer(const CFGElement &, DataflowLattice &Lattice,
                        Environment &) = 0;

  /// Applies the analysis transfer function for a given edge from a CFG block
  /// of a conditional statement. This method is optional and so the default
  /// implementation is a no-op.
  /// @param Stmt The condition which is responsible for the split in the CFG.
  /// @param Branch True if the edge goes to the basic block where the
  /// condition is true.
  virtual void transferBranch(bool Branch, const Stmt &Stmt, DataflowLattice &,
                              Environment &) {};

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
  std::unique_ptr<DataflowLattice> Lattice;

  /// Model of the state of the program (store and heap).
  Environment Env;

  TypeErasedDataflowAnalysisState(std::unique_ptr<DataflowLattice> Lattice,
                                  Environment Env)
      : Lattice(std::move(Lattice)), Env(std::move(Env)) {}

  TypeErasedDataflowAnalysisState fork() const {
    return TypeErasedDataflowAnalysisState(Lattice->clone(), Env.fork());
  }
};

/// A callback to be called with the state before or after visiting a CFG
/// element.
using CFGEltCallbackTypeErased = std::function<void(
    const CFGElement &, const TypeErasedDataflowAnalysisState &)>;

/// A pair of callbacks to be called with the state before and after visiting a
/// CFG element.
/// Either or both of the callbacks may be null.
struct CFGEltCallbacksTypeErased {
  CFGEltCallbackTypeErased Before;
  CFGEltCallbackTypeErased After;
};

/// Performs dataflow analysis and returns a mapping from basic block IDs to
/// dataflow analysis states that model the respective basic blocks. Indices of
/// the returned vector correspond to basic block IDs. Returns an error if the
/// dataflow analysis cannot be performed successfully. Otherwise, calls
/// `PostAnalysisCallbacks` on each CFG element with the final analysis results
/// before and after that program point.
///
/// `MaxBlockVisits` caps the number of block visits during analysis. It doesn't
/// distinguish between repeat visits to the same block and visits to distinct
/// blocks. This parameter is a backstop to prevent infinite loops, in the case
/// of bugs in the lattice and/or transfer functions that prevent the analysis
/// from converging.
llvm::Expected<std::vector<std::optional<TypeErasedDataflowAnalysisState>>>
runTypeErasedDataflowAnalysis(
    const AdornedCFG &ACFG, DataflowAnalysis &Analysis,
    const Environment &InitEnv,
    const CFGEltCallbacksTypeErased &PostAnalysisCallbacks,
    std::int32_t MaxBlockVisits);

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TYPEERASEDDATAFLOWANALYSIS_H
