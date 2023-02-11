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
#include <type_traits>
#include <utility>
#include <vector>

#include "clang/AST/ASTContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "llvm/ADT/Any.h"
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
///   * `void transfer(const CFGElement *, LatticeT &, Environment &)` - applies
///     the analysis transfer function for a given CFG element and lattice
///     element.
///
///  `Derived` can optionally provide the following members:
///  * `void transferBranch(bool Branch, const Stmt *Stmt, TypeErasedLattice &E,
///                         Environment &Env)` - applies the analysis transfer
///    function for a given edge from a CFG block of a conditional statement.
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
///
/// `LatticeT` can optionally provide the following members:
///  * `LatticeJoinEffect widen(const LatticeT &Previous)` - replaces the
///    lattice element with an  approximation that can reach a fixed point more
///    quickly than iterated application of the transfer function alone. The
///    previous value is provided to inform the choice of widened value. The
///    function must also serve as a comparison operation, by indicating whether
///    the widened value is equivalent to the previous value with the returned
///    `LatticeJoinEffect`.
template <typename Derived, typename LatticeT>
class DataflowAnalysis : public TypeErasedDataflowAnalysis {
public:
  /// Bounded join-semilattice that is used in the analysis.
  using Lattice = LatticeT;

  explicit DataflowAnalysis(ASTContext &Context) : Context(Context) {}

  /// Deprecated. Use the `DataflowAnalysisOptions` constructor instead.
  explicit DataflowAnalysis(ASTContext &Context, bool ApplyBuiltinTransfer)
      : DataflowAnalysis(
            Context,
            {ApplyBuiltinTransfer
                 ? DataflowAnalysisContext::Options{}
                 : std::optional<DataflowAnalysisContext::Options>()}) {}

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

  LatticeJoinEffect widenTypeErased(TypeErasedLattice &Current,
                                    const TypeErasedLattice &Previous) final {
    Lattice &C = llvm::any_cast<Lattice &>(Current.Value);
    const Lattice &P = llvm::any_cast<const Lattice &>(Previous.Value);
    return widenInternal(Rank0{}, C, P);
  }

  bool isEqualTypeErased(const TypeErasedLattice &E1,
                         const TypeErasedLattice &E2) final {
    const Lattice &L1 = llvm::any_cast<const Lattice &>(E1.Value);
    const Lattice &L2 = llvm::any_cast<const Lattice &>(E2.Value);
    return L1 == L2;
  }

  void transferTypeErased(const CFGElement &Element, TypeErasedLattice &E,
                          Environment &Env) final {
    Lattice &L = llvm::any_cast<Lattice &>(E.Value);
    // FIXME: change the contract of `transfer` to take a reference.
    static_cast<Derived *>(this)->transfer(&Element, L, Env);
  }

  void transferBranchTypeErased(bool Branch, const Stmt *Stmt,
                                TypeErasedLattice &E, Environment &Env) final {
    transferBranchInternal(Rank0{}, *static_cast<Derived *>(this), Branch, Stmt,
                           E, Env);
  }

private:
  // These `Rank` structs are used for template metaprogramming to choose
  // between overloads.
  struct Rank1 {};
  struct Rank0 : Rank1 {};

  // The first-choice implementation: use `widen` when it is available.
  template <typename T>
  static auto widenInternal(Rank0, T &Current, const T &Prev)
      -> decltype(Current.widen(Prev)) {
    return Current.widen(Prev);
  }

  // The second-choice implementation: `widen` is unavailable. Widening is
  // merged with equality checking, so when widening is unimplemented, we
  // default to equality checking.
  static LatticeJoinEffect widenInternal(Rank1, const Lattice &Current,
                                         const Lattice &Prev) {
    return Prev == Current ? LatticeJoinEffect::Unchanged
                           : LatticeJoinEffect::Changed;
  }

  // The first-choice implementation: `transferBranch` is implemented.
  template <typename Analysis>
  static auto transferBranchInternal(Rank0, Analysis &A, bool Branch,
                                     const Stmt *Stmt, TypeErasedLattice &L,
                                     Environment &Env)
      -> std::void_t<decltype(A.transferBranch(
          Branch, Stmt, std::declval<LatticeT &>(), Env))> {
    A.transferBranch(Branch, Stmt, llvm::any_cast<Lattice &>(L.Value), Env);
  }

  // The second-choice implementation: `transferBranch` is unimplemented. No-op.
  template <typename Analysis>
  static void transferBranchInternal(Rank1, Analysis &A, bool, const Stmt *,
                                     TypeErasedLattice &, Environment &) {}

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
/// `PostVisitCFG` on each CFG element with the final analysis results at that
/// program point.
template <typename AnalysisT>
llvm::Expected<std::vector<
    std::optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>>
runDataflowAnalysis(
    const ControlFlowContext &CFCtx, AnalysisT &Analysis,
    const Environment &InitEnv,
    std::function<void(const CFGElement &, const DataflowAnalysisState<
                                               typename AnalysisT::Lattice> &)>
        PostVisitCFG = nullptr) {
  std::function<void(const CFGElement &,
                     const TypeErasedDataflowAnalysisState &)>
      PostVisitCFGClosure = nullptr;
  if (PostVisitCFG) {
    PostVisitCFGClosure = [&PostVisitCFG](
                              const CFGElement &Element,
                              const TypeErasedDataflowAnalysisState &State) {
      auto *Lattice =
          llvm::any_cast<typename AnalysisT::Lattice>(&State.Lattice.Value);
      PostVisitCFG(Element, DataflowAnalysisState<typename AnalysisT::Lattice>{
                                *Lattice, State.Env});
    };
  }

  auto TypeErasedBlockStates = runTypeErasedDataflowAnalysis(
      CFCtx, Analysis, InitEnv, PostVisitCFGClosure);
  if (!TypeErasedBlockStates)
    return TypeErasedBlockStates.takeError();

  std::vector<std::optional<DataflowAnalysisState<typename AnalysisT::Lattice>>>
      BlockStates;
  BlockStates.reserve(TypeErasedBlockStates->size());

  llvm::transform(
      std::move(*TypeErasedBlockStates), std::back_inserter(BlockStates),
      [](auto &OptState) {
        return llvm::transformOptional(std::move(OptState), [](auto &&State) {
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
  /// Return value indicates whether the model processed the `Element`.
  virtual bool transfer(const CFGElement *Element, Environment &Env) = 0;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWANALYSIS_H
