//===- TypeErasedDataflowAnalysis.cpp -------------------------------------===//
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

#include <utility>
#include <vector>

#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace dataflow {

/// Computes the input state for a given basic block by joining the output
/// states of its predecessors.
///
/// Requirements:
///
///   All predecessors of `Block` except those with loop back edges must have
///   already been transferred. States in `BlockStates` that are set to
///   `llvm::None` represent basic blocks that are not evaluated yet.
static TypeErasedDataflowAnalysisState computeBlockInputState(
    std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> &BlockStates,
    const CFGBlock &Block, const Environment &InitEnv,
    TypeErasedDataflowAnalysis &Analysis) {
  // FIXME: Consider passing `Block` to `Analysis.typeErasedInitialElement()`
  // to enable building analyses like computation of dominators that initialize
  // the state of each basic block differently.
  TypeErasedDataflowAnalysisState State = {Analysis.typeErasedInitialElement(),
                                           InitEnv};
  for (const CFGBlock *Pred : Block.preds()) {
    // Skip if the `Block` is unreachable or control flow cannot get past it.
    if (!Pred || Pred->hasNoReturnElement())
      continue;

    // Skip if `Pred` was not evaluated yet. This could happen if `Pred` has a
    // loop back edge to `Block`.
    const llvm::Optional<TypeErasedDataflowAnalysisState> &MaybePredState =
        BlockStates[Pred->getBlockID()];
    if (!MaybePredState.hasValue())
      continue;

    const TypeErasedDataflowAnalysisState &PredState =
        MaybePredState.getValue();
    Analysis.joinTypeErased(State.Lattice, PredState.Lattice);
    State.Env.join(PredState.Env);
  }
  return State;
}

TypeErasedDataflowAnalysisState transferBlock(
    std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> &BlockStates,
    const CFGBlock &Block, const Environment &InitEnv,
    TypeErasedDataflowAnalysis &Analysis,
    std::function<void(const CFGStmt &,
                       const TypeErasedDataflowAnalysisState &)>
        HandleTransferredStmt) {
  TypeErasedDataflowAnalysisState State =
      computeBlockInputState(BlockStates, Block, InitEnv, Analysis);
  for (const CFGElement &Element : Block) {
    // FIXME: Evaluate other kinds of `CFGElement`.
    const llvm::Optional<CFGStmt> Stmt = Element.getAs<CFGStmt>();
    if (!Stmt.hasValue())
      continue;

    // FIXME: Evaluate the statement contained in `Stmt`.

    State.Lattice = Analysis.transferTypeErased(Stmt.getValue().getStmt(),
                                                State.Lattice, State.Env);
    if (HandleTransferredStmt != nullptr)
      HandleTransferredStmt(Stmt.getValue(), State);
  }
  return State;
}

std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>
runTypeErasedDataflowAnalysis(const CFG &Cfg,
                              TypeErasedDataflowAnalysis &Analysis,
                              const Environment &InitEnv) {
  // FIXME: Consider enforcing that `Cfg` meets the requirements that
  // are specified in the header. This could be done by remembering
  // what options were used to build `Cfg` and asserting on them here.

  PostOrderCFGView POV(&Cfg);
  ForwardDataflowWorklist Worklist(Cfg, &POV);

  std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates;
  BlockStates.resize(Cfg.size(), llvm::None);

  // The entry basic block doesn't contain statements so it can be skipped.
  const CFGBlock &Entry = Cfg.getEntry();
  BlockStates[Entry.getBlockID()] = {Analysis.typeErasedInitialElement(),
                                     InitEnv};
  Worklist.enqueueSuccessors(&Entry);

  // Bugs in lattices and transfer functions can prevent the analysis from
  // converging. To limit the damage (infinite loops) that these bugs can cause,
  // limit the number of iterations.
  // FIXME: Consider making the maximum number of iterations configurable.
  // FIXME: Set up statistics (see llvm/ADT/Statistic.h) to count average number
  // of iterations, number of functions that time out, etc.
  unsigned Iterations = 0;
  static constexpr unsigned MaxIterations = 1 << 16;
  while (const CFGBlock *Block = Worklist.dequeue()) {
    if (++Iterations > MaxIterations) {
      llvm::errs() << "Maximum number of iterations reached, giving up.\n";
      break;
    }

    const llvm::Optional<TypeErasedDataflowAnalysisState> &OldBlockState =
        BlockStates[Block->getBlockID()];
    TypeErasedDataflowAnalysisState NewBlockState =
        transferBlock(BlockStates, *Block, InitEnv, Analysis);

    if (OldBlockState.hasValue() &&
        Analysis.isEqualTypeErased(OldBlockState.getValue().Lattice,
                                   NewBlockState.Lattice) &&
        OldBlockState->Env == NewBlockState.Env) {
      // The state of `Block` didn't change after transfer so there's no need to
      // revisit its successors.
      continue;
    }

    BlockStates[Block->getBlockID()] = std::move(NewBlockState);

    // Do not add unreachable successor blocks to `Worklist`.
    if (Block->hasNoReturnElement())
      continue;

    Worklist.enqueueSuccessors(Block);
  }
  // FIXME: Consider evaluating unreachable basic blocks (those that have a
  // state set to `llvm::None` at this point) to also analyze dead code.

  return BlockStates;
}

} // namespace dataflow
} // namespace clang
