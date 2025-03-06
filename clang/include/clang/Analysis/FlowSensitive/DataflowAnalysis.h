//===- DataflowAnalysis.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines base types and functions for building dataflow analyses
//  that run over Control-Flow Graphs (CFGs).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSIS_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSIS_H

#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/AdornedCFG.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace dataflow {

// Model of the program at a given program point.
template <typename LatticeT> struct DataflowAnalysisState {
  // Model of a program property.
  LatticeT Lattice;

  // Model of the state of the program (store and heap).
  Environment Env;
};

/// A callback to be called with the state before or after visiting a CFG
/// element.
template <typename AnalysisT>
using CFGEltCallback = std::function<void(
    const CFGElement &,
    const DataflowAnalysisState<typename AnalysisT::Lattice> &)>;

/// A pair of callbacks to be called with the state before and after visiting a
/// CFG element.
/// Either or both of the callbacks may be null.
template <typename AnalysisT> struct CFGEltCallbacks {
  CFGEltCallback<AnalysisT> Before;
  CFGEltCallback<AnalysisT> After;
};

/// A callback for performing diagnosis on a CFG element, called with the state
/// before or after visiting that CFG element. Returns a list of diagnostics
/// to emit (if any).
template <typename AnalysisT, typename Diagnostic>
using DiagnosisCallback = llvm::function_ref<llvm::SmallVector<Diagnostic>(
    const CFGElement &, ASTContext &,
    const TransferStateForDiagnostics<typename AnalysisT::Lattice> &)>;

/// A pair of callbacks for performing diagnosis on a CFG element, called with
/// the state before and after visiting that CFG element.
/// Either or both of the callbacks may be null.
template <typename AnalysisT, typename Diagnostic> struct DiagnosisCallbacks {
  DiagnosisCallback<AnalysisT, Diagnostic> Before;
  DiagnosisCallback<AnalysisT, Diagnostic> After;
};

/// Default for the maximum number of SAT solver iterations during analysis.
inline constexpr std::int64_t kDefaultMaxSATIterations = 1'000'000'000;

/// Default for the maximum number of block visits during analysis.
inline constexpr std::int32_t kDefaultMaxBlockVisits = 20'000;

/// Performs dataflow analysis and returns a mapping from basic block IDs to
/// dataflow analysis states that model the respective basic blocks. The
/// returned vector, if any, will have the same size as the number of CFG
/// blocks, with indices corresponding to basic block IDs. Returns an error if
/// the dataflow analysis cannot be performed successfully. Otherwise, calls
/// `PostAnalysisCallbacks` on each CFG element with the final analysis results
/// before and after that program point.
///
/// `MaxBlockVisits` caps the number of block visits during analysis. See
/// `runTypeErasedDataflowAnalysis` for a full description. The default value is
/// essentially arbitrary -- large enough to accommodate what seems like any
/// reasonable CFG, but still small enough to limit the cost of hitting the
/// limit.
template <typename AnalysisT>
llvm::Expected<std::vector<
    std::optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>>
runDataflowAnalysis(const AdornedCFG &ACFG, AnalysisT &Analysis,
                    const Environment &InitEnv,
                    CFGEltCallbacks<AnalysisT> PostAnalysisCallbacks = {},
                    std::int32_t MaxBlockVisits = kDefaultMaxBlockVisits) {
  CFGEltCallbacksTypeErased TypeErasedCallbacks;
  if (PostAnalysisCallbacks.Before) {
    TypeErasedCallbacks.Before =
        [&PostAnalysisCallbacks](const CFGElement &Element,
                                 const TypeErasedDataflowAnalysisState &State) {
          auto *Lattice =
              llvm::cast<typename AnalysisT::Lattice>(State.Lattice.get());
          // FIXME: we should not be copying the environment here!
          // Ultimately the `CFGEltCallback` only gets a const reference anyway.
          PostAnalysisCallbacks.Before(
              Element, DataflowAnalysisState<typename AnalysisT::Lattice>{
                           *Lattice, State.Env.fork()});
        };
  }
  if (PostAnalysisCallbacks.After) {
    TypeErasedCallbacks.After =
        [&PostAnalysisCallbacks](const CFGElement &Element,
                                 const TypeErasedDataflowAnalysisState &State) {
          auto *Lattice =
              llvm::cast<typename AnalysisT::Lattice>(State.Lattice.get());
          // FIXME: we should not be copying the environment here!
          // Ultimately the `CFGEltCallback` only gets a const reference anyway.
          PostAnalysisCallbacks.After(
              Element, DataflowAnalysisState<typename AnalysisT::Lattice>{
                           *Lattice, State.Env.fork()});
        };
  }

  auto TypeErasedBlockStates = runTypeErasedDataflowAnalysis(
      ACFG, Analysis, InitEnv, TypeErasedCallbacks, MaxBlockVisits);
  if (!TypeErasedBlockStates)
    return TypeErasedBlockStates.takeError();

  std::vector<std::optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>
      BlockStates;
  BlockStates.reserve(TypeErasedBlockStates->size());

  llvm::transform(
      std::move(*TypeErasedBlockStates), std::back_inserter(BlockStates),
      [](auto &OptState) {
        return llvm::transformOptional(
            std::move(OptState), [](TypeErasedDataflowAnalysisState &&State) {
              return DataflowAnalysisState<typename AnalysisT::Lattice>{
                  std::move(*llvm::cast<typename AnalysisT::Lattice>(
                      State.Lattice.get())),
                  std::move(State.Env)};
            });
      });
  return std::move(BlockStates);
}

// Create an analysis class that is derived from `DataflowAnalysis`. This is an
// SFINAE adapter that allows us to call two different variants of constructor
// (either with or without the optional `Environment` parameter).
// FIXME: Make all classes derived from `DataflowAnalysis` take an `Environment`
// parameter in their constructor so that we can get rid of this abomination.
template <typename AnalysisT>
auto createAnalysis(ASTContext &ASTCtx, Environment &Env)
    -> decltype(AnalysisT(ASTCtx, Env)) {
  return AnalysisT(ASTCtx, Env);
}
template <typename AnalysisT>
auto createAnalysis(ASTContext &ASTCtx, Environment &Env)
    -> decltype(AnalysisT(ASTCtx)) {
  return AnalysisT(ASTCtx);
}

/// Runs a dataflow analysis over the given function and then runs `Diagnoser`
/// over the results. Returns a list of diagnostics for `FuncDecl` or an
/// error. Currently, errors can occur (at least) because the analysis requires
/// too many iterations over the CFG or the SAT solver times out.
///
/// The default value of `MaxSATIterations` was chosen based on the following
/// observations:
/// - Non-pathological calls to the solver typically require only a few hundred
///   iterations.
/// - This limit is still low enough to keep runtimes acceptable (on typical
///   machines) in cases where we hit the limit.
///
/// `MaxBlockVisits` caps the number of block visits during analysis. See
/// `runDataflowAnalysis` for a full description and explanation of the default
/// value.
template <typename AnalysisT, typename Diagnostic>
llvm::Expected<llvm::SmallVector<Diagnostic>>
diagnoseFunction(const FunctionDecl &FuncDecl, ASTContext &ASTCtx,
                 DiagnosisCallbacks<AnalysisT, Diagnostic> Diagnoser,
                 std::int64_t MaxSATIterations = kDefaultMaxSATIterations,
                 std::int32_t MaxBlockVisits = kDefaultMaxBlockVisits) {
  llvm::Expected<AdornedCFG> Context = AdornedCFG::build(FuncDecl);
  if (!Context)
    return Context.takeError();

  auto Solver = std::make_unique<WatchedLiteralsSolver>(MaxSATIterations);
  DataflowAnalysisContext AnalysisContext(*Solver);
  Environment Env(AnalysisContext, FuncDecl);
  AnalysisT Analysis = createAnalysis<AnalysisT>(ASTCtx, Env);
  llvm::SmallVector<Diagnostic> Diagnostics;
  CFGEltCallbacksTypeErased PostAnalysisCallbacks;
  if (Diagnoser.Before) {
    PostAnalysisCallbacks.Before =
        [&ASTCtx, &Diagnoser,
         &Diagnostics](const CFGElement &Elt,
                       const TypeErasedDataflowAnalysisState &State) mutable {
          auto EltDiagnostics = Diagnoser.Before(
              Elt, ASTCtx,
              TransferStateForDiagnostics<typename AnalysisT::Lattice>(
                  *llvm::cast<typename AnalysisT::Lattice>(State.Lattice.get()),
                  State.Env));
          llvm::move(EltDiagnostics, std::back_inserter(Diagnostics));
        };
  }
  if (Diagnoser.After) {
    PostAnalysisCallbacks.After =
        [&ASTCtx, &Diagnoser,
         &Diagnostics](const CFGElement &Elt,
                       const TypeErasedDataflowAnalysisState &State) mutable {
          auto EltDiagnostics = Diagnoser.After(
              Elt, ASTCtx,
              TransferStateForDiagnostics<typename AnalysisT::Lattice>(
                  *llvm::cast<typename AnalysisT::Lattice>(State.Lattice.get()),
                  State.Env));
          llvm::move(EltDiagnostics, std::back_inserter(Diagnostics));
        };
  }
  if (llvm::Error Err =
          runTypeErasedDataflowAnalysis(*Context, Analysis, Env,
                                        PostAnalysisCallbacks, MaxBlockVisits)
              .takeError())
    return std::move(Err);

  if (Solver->reachedLimit())
    return llvm::createStringError(llvm::errc::interrupted,
                                   "SAT solver timed out");

  return Diagnostics;
}

/// Overload that takes only one diagnosis callback, which is run on the state
/// after visiting the `CFGElement`. This is provided for backwards
/// compatibility; new callers should call the overload taking
/// `DiagnosisCallbacks` instead.
template <typename AnalysisT, typename Diagnostic>
llvm::Expected<llvm::SmallVector<Diagnostic>>
diagnoseFunction(const FunctionDecl &FuncDecl, ASTContext &ASTCtx,
                 DiagnosisCallback<AnalysisT, Diagnostic> Diagnoser,
                 std::int64_t MaxSATIterations = kDefaultMaxSATIterations,
                 std::int32_t MaxBlockVisits = kDefaultMaxBlockVisits) {
  DiagnosisCallbacks<AnalysisT, Diagnostic> Callbacks = {nullptr, Diagnoser};
  return diagnoseFunction(FuncDecl, ASTCtx, Callbacks, MaxSATIterations,
                          MaxBlockVisits);
}

/// Abstract base class for dataflow "models": reusable analysis components that
/// model a particular aspect of program semantics in the `Environment`. For
/// example, a model may capture a type and its related functions.
class DataflowModel : public Environment::ValueModel {
public:
  /// Return value indicates whether the model processed the `Element`.
  virtual bool transfer(const CFGElement &Element, Environment &Env) = 0;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSIS_H
