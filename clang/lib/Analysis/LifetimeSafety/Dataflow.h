//===- Dataflow.h - Generic Dataflow Analysis Framework --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic, policy-based driver for dataflow analyses.
// It provides a flexible framework that combines the dataflow runner and
// transfer functions, allowing derived classes to implement specific analyses
// by defining their lattice, join, and transfer functions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_DATAFLOW_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_DATAFLOW_H

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include <optional>

namespace clang::lifetimes::internal {

enum class Direction { Forward, Backward };

/// A `ProgramPoint` identifies a location in the CFG by pointing to a specific
/// `Fact`. identified by a lifetime-related event (`Fact`).
///
/// A `ProgramPoint` has "after" semantics: it represents the location
/// immediately after its corresponding `Fact`.
using ProgramPoint = const Fact *;

/// A generic, policy-based driver for dataflow analyses. It combines
/// the dataflow runner and the transferer logic into a single class hierarchy.
///
/// The derived class is expected to provide:
/// - A `Lattice` type.
/// - `StringRef getAnalysisName() const`
/// - `Lattice getInitialState();` The initial state of the analysis.
/// - `Lattice join(Lattice, Lattice);` Merges states from multiple CFG paths.
/// - `Lattice transfer(Lattice, const FactType&);` Defines how a single
///   lifetime-relevant `Fact` transforms the lattice state. Only overloads
///   for facts relevant to the analysis need to be implemented.
///
/// \tparam Derived The CRTP derived class that implements the specific
/// analysis.
/// \tparam LatticeType The dataflow lattice used by the analysis.
/// \tparam Dir The direction of the analysis (Forward or Backward).
/// TODO: Maybe use the dataflow framework! The framework might need changes
/// to support the current comparison done at block-entry.
template <typename Derived, typename LatticeType, Direction Dir>
class DataflowAnalysis {
public:
  using Lattice = LatticeType;
  using Base = DataflowAnalysis<Derived, Lattice, Dir>;

private:
  const CFG &Cfg;
  AnalysisDeclContext &AC;

  /// The dataflow state before a basic block is processed.
  llvm::DenseMap<const CFGBlock *, Lattice> InStates;
  /// The dataflow state after a basic block is processed.
  llvm::DenseMap<const CFGBlock *, Lattice> OutStates;
  /// The dataflow state at a Program Point.
  /// In a forward analysis, this is the state after the Fact at that point has
  /// been applied, while in a backward analysis, it is the state before.
  llvm::DenseMap<ProgramPoint, Lattice> PerPointStates;

  static constexpr bool isForward() { return Dir == Direction::Forward; }

protected:
  FactManager &FactMgr;

  explicit DataflowAnalysis(const CFG &Cfg, AnalysisDeclContext &AC,
                            FactManager &FactMgr)
      : Cfg(Cfg), AC(AC), FactMgr(FactMgr) {}

public:
  void run() {
    Derived &D = static_cast<Derived &>(*this);
    llvm::TimeTraceScope Time(D.getAnalysisName());

    using Worklist =
        std::conditional_t<Dir == Direction::Forward, ForwardDataflowWorklist,
                           BackwardDataflowWorklist>;
    Worklist W(Cfg, AC);

    const CFGBlock *Start = isForward() ? &Cfg.getEntry() : &Cfg.getExit();
    InStates[Start] = D.getInitialState();
    W.enqueueBlock(Start);

    while (const CFGBlock *B = W.dequeue()) {
      Lattice StateIn = *getInState(B);
      Lattice StateOut = transferBlock(B, StateIn);
      OutStates[B] = StateOut;
      for (const CFGBlock *AdjacentB : isForward() ? B->succs() : B->preds()) {
        if (!AdjacentB)
          continue;
        std::optional<Lattice> OldInState = getInState(AdjacentB);
        Lattice NewInState =
            !OldInState ? StateOut : D.join(*OldInState, StateOut);
        // Enqueue the adjacent block if its in-state has changed or if we have
        // never seen it.
        if (!OldInState || NewInState != *OldInState) {
          InStates[AdjacentB] = NewInState;
          W.enqueueBlock(AdjacentB);
        }
      }
    }
  }

protected:
  Lattice getState(ProgramPoint P) const { return PerPointStates.lookup(P); }

  std::optional<Lattice> getInState(const CFGBlock *B) const {
    auto It = InStates.find(B);
    if (It == InStates.end())
      return std::nullopt;
    return It->second;
  }

  Lattice getOutState(const CFGBlock *B) const { return OutStates.lookup(B); }

  void dump() const {
    const Derived *D = static_cast<const Derived *>(this);
    llvm::dbgs() << "==========================================\n";
    llvm::dbgs() << D->getAnalysisName() << " results:\n";
    llvm::dbgs() << "==========================================\n";
    const CFGBlock &B = isForward() ? Cfg.getExit() : Cfg.getEntry();
    getOutState(&B).dump(llvm::dbgs());
  }

private:
  /// Computes the state at one end of a block by applying all its facts
  /// sequentially to a given state from the other end.
  Lattice transferBlock(const CFGBlock *Block, Lattice State) {
    auto Facts = FactMgr.getFacts(Block);
    if constexpr (isForward()) {
      for (const Fact *F : Facts) {
        State = transferFact(State, F);
        PerPointStates[F] = State;
      }
    } else {
      for (const Fact *F : llvm::reverse(Facts)) {
        // In backward analysis, capture the state before applying the fact.
        PerPointStates[F] = State;
        State = transferFact(State, F);
      }
    }
    return State;
  }

  Lattice transferFact(Lattice In, const Fact *F) {
    assert(F);
    Derived *D = static_cast<Derived *>(this);
    switch (F->getKind()) {
    case Fact::Kind::Issue:
      return D->transfer(In, *F->getAs<IssueFact>());
    case Fact::Kind::Expire:
      return D->transfer(In, *F->getAs<ExpireFact>());
    case Fact::Kind::OriginFlow:
      return D->transfer(In, *F->getAs<OriginFlowFact>());
    case Fact::Kind::ReturnOfOrigin:
      return D->transfer(In, *F->getAs<ReturnOfOriginFact>());
    case Fact::Kind::Use:
      return D->transfer(In, *F->getAs<UseFact>());
    case Fact::Kind::TestPoint:
      return D->transfer(In, *F->getAs<TestPointFact>());
    }
    llvm_unreachable("Unknown fact kind");
  }

public:
  Lattice transfer(Lattice In, const IssueFact &) { return In; }
  Lattice transfer(Lattice In, const ExpireFact &) { return In; }
  Lattice transfer(Lattice In, const OriginFlowFact &) { return In; }
  Lattice transfer(Lattice In, const ReturnOfOriginFact &) { return In; }
  Lattice transfer(Lattice In, const UseFact &) { return In; }
  Lattice transfer(Lattice In, const TestPointFact &) { return In; }
};
} // namespace clang::lifetimes::internal
#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_DATAFLOW_H
