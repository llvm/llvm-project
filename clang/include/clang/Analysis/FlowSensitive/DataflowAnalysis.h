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
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace dataflow {

/// Base class template for dataflow analyses built on a single lattice type.
///
/// Requirements:
///
///  `Derived` must be derived from a specialization of this class template and
///  must provide the following public members:
///   * `LatticeT initialElement()` - returns a lattice element that models the
///     initial state of a basic block;
///   * `void transfer(const Stmt *, LatticeT &, Environment &)` - applies the
///     analysis transfer function for a given statement and lattice element.
///
///  `Derived` can optionally override the following members:
///   * `bool merge(QualType, const Value &, const Value &, Value &,
///     Environment &)` -  joins distinct values. This could be a strict
///     lattice join or a more general widening operation.
///
///  `LatticeT` is a bounded join-semilattice that is used by `Derived` and must
///  provide the following public members:
///   * `LatticeJoinEffect join(const LatticeT &)` - joins the object and the
///     argument by computing their least upper bound, modifies the object if
///     necessary, and returns an effect indicating whether any changes were
///     made to it;
///   * `bool operator==(const LatticeT &) const` - returns true if and only if
///     the object is equal to the argument.
template <typename Derived, typename LatticeT>
class DataflowAnalysis : public TypeErasedDataflowAnalysis {
public:
  /// Bounded join-semilattice that is used in the analysis.
  using Lattice = LatticeT;

  explicit DataflowAnalysis(ASTContext &Context) : Context(Context) {}

  /// Deprecated. Use the `DataflowAnalysisOptions` constructor instead.
  explicit DataflowAnalysis(ASTContext &Context, bool ApplyBuiltinTransfer)
      : DataflowAnalysis(Context, DataflowAnalysisOptions{ApplyBuiltinTransfer,
                                                          TransferOptions{}}) {}

  explicit DataflowAnalysis(ASTContext &Context,
                            DataflowAnalysisOptions Options)
      : TypeErasedDataflowAnalysis(Options), Context(Context) {}

  ASTContext &getASTContext() final { return Context; }

  TypeErasedLattice typeErasedInitialElement() final {
    return {static_cast<Derived *>(this)->initialElement()};
  }

  LatticeJoinEffect joinTypeErased(TypeErasedLattice &E1,
                                   const TypeErasedLattice &E2) final {
    Lattice &L1 = llvm::any_cast<Lattice &>(E1.Value);
    const Lattice &L2 = llvm::any_cast<const Lattice &>(E2.Value);
    return L1.join(L2);
  }

  bool isEqualTypeErased(const TypeErasedLattice &E1,
                         const TypeErasedLattice &E2) final {
    const Lattice &L1 = llvm::any_cast<const Lattice &>(E1.Value);
    const Lattice &L2 = llvm::any_cast<const Lattice &>(E2.Value);
    return L1 == L2;
  }

  void transferTypeErased(const Stmt *Stmt, TypeErasedLattice &E,
                          Environment &Env) final {
    Lattice &L = llvm::any_cast<Lattice &>(E.Value);
    static_cast<Derived *>(this)->transfer(Stmt, L, Env);
  }

private:
  ASTContext &Context;
};

// Model of the program at a given program point.
template <typename LatticeT> struct DataflowAnalysisState {
  // Model of a program property.
  LatticeT Lattice;

  // Model of the state of the program (store and heap).
  Environment Env;
};

/// Performs dataflow analysis and returns a mapping from basic block IDs to
/// dataflow analysis states that model the respective basic blocks. The
/// returned vector, if any, will have the same size as the number of CFG
/// blocks, with indices corresponding to basic block IDs. Returns an error if
/// the dataflow analysis cannot be performed successfully. Otherwise, calls
/// `PostVisitStmt` on each statement with the final analysis results at that
/// program point.
template <typename AnalysisT>
llvm::Expected<std::vector<
    llvm::Optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>>
runDataflowAnalysis(
    const ControlFlowContext &CFCtx, AnalysisT &Analysis,
    const Environment &InitEnv,
    std::function<void(const CFGStmt &, const DataflowAnalysisState<
                                            typename AnalysisT::Lattice> &)>
        PostVisitStmt = nullptr) {
  std::function<void(const CFGStmt &, const TypeErasedDataflowAnalysisState &)>
      PostVisitStmtClosure = nullptr;
  if (PostVisitStmt != nullptr) {
    PostVisitStmtClosure = [&PostVisitStmt](
                               const CFGStmt &Stmt,
                               const TypeErasedDataflowAnalysisState &State) {
      auto *Lattice =
          llvm::any_cast<typename AnalysisT::Lattice>(&State.Lattice.Value);
      PostVisitStmt(Stmt, DataflowAnalysisState<typename AnalysisT::Lattice>{
                              *Lattice, State.Env});
    };
  }

  auto TypeErasedBlockStates = runTypeErasedDataflowAnalysis(
      CFCtx, Analysis, InitEnv, PostVisitStmtClosure);
  if (!TypeErasedBlockStates)
    return TypeErasedBlockStates.takeError();

  std::vector<
      llvm::Optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>
      BlockStates;
  BlockStates.reserve(TypeErasedBlockStates->size());

  llvm::transform(std::move(*TypeErasedBlockStates),
                  std::back_inserter(BlockStates), [](auto &OptState) {
                    return std::move(OptState).map([](auto &&State) {
                      return DataflowAnalysisState<typename AnalysisT::Lattice>{
                          llvm::any_cast<typename AnalysisT::Lattice>(
                              std::move(State.Lattice.Value)),
                          std::move(State.Env)};
                    });
                  });
  return BlockStates;
}

/// Abstract base class for dataflow "models": reusable analysis components that
/// model a particular aspect of program semantics in the `Environment`. For
/// example, a model may capture a type and its related functions.
class DataflowModel : public Environment::ValueModel {
public:
  /// Return value indicates whether the model processed the `Stmt`.
  virtual bool transfer(const Stmt *Stmt, Environment &Env) = 0;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSIS_H
