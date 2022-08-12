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

#include <algorithm>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

#include "clang/AST/DeclCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "clang/Analysis/FlowSensitive/Transfer.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
namespace dataflow {

class StmtToEnvMapImpl : public StmtToEnvMap {
public:
  StmtToEnvMapImpl(
      const ControlFlowContext &CFCtx,
      llvm::ArrayRef<llvm::Optional<TypeErasedDataflowAnalysisState>>
          BlockToState)
      : CFCtx(CFCtx), BlockToState(BlockToState) {}

  const Environment *getEnvironment(const Stmt &S) const override {
    auto BlockIt = CFCtx.getStmtToBlock().find(&ignoreCFGOmittedNodes(S));
    assert(BlockIt != CFCtx.getStmtToBlock().end());
    const auto &State = BlockToState[BlockIt->getSecond()->getBlockID()];
    assert(State);
    return &State.value().Env;
  }

private:
  const ControlFlowContext &CFCtx;
  llvm::ArrayRef<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockToState;
};

/// Returns the index of `Block` in the successors of `Pred`.
static int blockIndexInPredecessor(const CFGBlock &Pred,
                                   const CFGBlock &Block) {
  auto BlockPos = llvm::find_if(
      Pred.succs(), [&Block](const CFGBlock::AdjacentBlock &Succ) {
        return Succ && Succ->getBlockID() == Block.getBlockID();
      });
  return BlockPos - Pred.succ_begin();
}

/// Extends the flow condition of an environment based on a terminator
/// statement.
class TerminatorVisitor : public ConstStmtVisitor<TerminatorVisitor> {
public:
  TerminatorVisitor(const StmtToEnvMap &StmtToEnv, Environment &Env,
                    int BlockSuccIdx, TransferOptions TransferOpts)
      : StmtToEnv(StmtToEnv), Env(Env), BlockSuccIdx(BlockSuccIdx),
        TransferOpts(TransferOpts) {}

  void VisitIfStmt(const IfStmt *S) {
    auto *Cond = S->getCond();
    assert(Cond != nullptr);
    extendFlowCondition(*Cond);
  }

  void VisitWhileStmt(const WhileStmt *S) {
    auto *Cond = S->getCond();
    assert(Cond != nullptr);
    extendFlowCondition(*Cond);
  }

  void VisitDoStmt(const DoStmt *S) {
    auto *Cond = S->getCond();
    assert(Cond != nullptr);
    extendFlowCondition(*Cond);
  }

  void VisitForStmt(const ForStmt *S) {
    auto *Cond = S->getCond();
    if (Cond != nullptr)
      extendFlowCondition(*Cond);
  }

  void VisitBinaryOperator(const BinaryOperator *S) {
    assert(S->getOpcode() == BO_LAnd || S->getOpcode() == BO_LOr);
    auto *LHS = S->getLHS();
    assert(LHS != nullptr);
    extendFlowCondition(*LHS);
  }

  void VisitConditionalOperator(const ConditionalOperator *S) {
    auto *Cond = S->getCond();
    assert(Cond != nullptr);
    extendFlowCondition(*Cond);
  }

private:
  void extendFlowCondition(const Expr &Cond) {
    // The terminator sub-expression might not be evaluated.
    if (Env.getStorageLocation(Cond, SkipPast::None) == nullptr)
      transfer(StmtToEnv, Cond, Env, TransferOpts);

    // FIXME: The flow condition must be an r-value, so `SkipPast::None` should
    // suffice.
    auto *Val =
        cast_or_null<BoolValue>(Env.getValue(Cond, SkipPast::Reference));
    // Value merging depends on flow conditions from different environments
    // being mutually exclusive -- that is, they cannot both be true in their
    // entirety (even if they may share some clauses). So, we need *some* value
    // for the condition expression, even if just an atom.
    if (Val == nullptr) {
      // FIXME: Consider introducing a helper for this get-or-create pattern.
      auto *Loc = Env.getStorageLocation(Cond, SkipPast::None);
      if (Loc == nullptr) {
        Loc = &Env.createStorageLocation(Cond);
        Env.setStorageLocation(Cond, *Loc);
      }
      Val = &Env.makeAtomicBoolValue();
      Env.setValue(*Loc, *Val);
    }

    // The condition must be inverted for the successor that encompasses the
    // "else" branch, if such exists.
    if (BlockSuccIdx == 1)
      Val = &Env.makeNot(*Val);

    Env.addToFlowCondition(*Val);
  }

  const StmtToEnvMap &StmtToEnv;
  Environment &Env;
  int BlockSuccIdx;
  TransferOptions TransferOpts;
};

/// Computes the input state for a given basic block by joining the output
/// states of its predecessors.
///
/// Requirements:
///
///   All predecessors of `Block` except those with loop back edges must have
///   already been transferred. States in `BlockStates` that are set to
///   `llvm::None` represent basic blocks that are not evaluated yet.
static TypeErasedDataflowAnalysisState computeBlockInputState(
    const ControlFlowContext &CFCtx,
    llvm::ArrayRef<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates,
    const CFGBlock &Block, const Environment &InitEnv,
    TypeErasedDataflowAnalysis &Analysis) {
  llvm::DenseSet<const CFGBlock *> Preds;
  Preds.insert(Block.pred_begin(), Block.pred_end());
  if (Block.getTerminator().isTemporaryDtorsBranch()) {
    // This handles a special case where the code that produced the CFG includes
    // a conditional operator with a branch that constructs a temporary and
    // calls a destructor annotated as noreturn. The CFG models this as follows:
    //
    // B1 (contains the condition of the conditional operator) - succs: B2, B3
    // B2 (contains code that does not call a noreturn destructor) - succs: B4
    // B3 (contains code that calls a noreturn destructor) - succs: B4
    // B4 (has temporary destructor terminator) - succs: B5, B6
    // B5 (noreturn block that is associated with the noreturn destructor call)
    // B6 (contains code that follows the conditional operator statement)
    //
    // The first successor (B5 above) of a basic block with a temporary
    // destructor terminator (B4 above) is the block that evaluates the
    // destructor. If that block has a noreturn element then the predecessor
    // block that constructed the temporary object (B3 above) is effectively a
    // noreturn block and its state should not be used as input for the state
    // of the block that has a temporary destructor terminator (B4 above). This
    // holds regardless of which branch of the ternary operator calls the
    // noreturn destructor. However, it doesn't cases where a nested ternary
    // operator includes a branch that contains a noreturn destructor call.
    //
    // See `NoreturnDestructorTest` for concrete examples.
    if (Block.succ_begin()->getReachableBlock()->hasNoReturnElement()) {
      auto StmtBlock = CFCtx.getStmtToBlock().find(Block.getTerminatorStmt());
      assert(StmtBlock != CFCtx.getStmtToBlock().end());
      Preds.erase(StmtBlock->getSecond());
    }
  }

  llvm::Optional<TypeErasedDataflowAnalysisState> MaybeState;
  auto BuiltinTransferOpts = Analysis.builtinTransferOptions();

  for (const CFGBlock *Pred : Preds) {
    // Skip if the `Block` is unreachable or control flow cannot get past it.
    if (!Pred || Pred->hasNoReturnElement())
      continue;

    // Skip if `Pred` was not evaluated yet. This could happen if `Pred` has a
    // loop back edge to `Block`.
    const llvm::Optional<TypeErasedDataflowAnalysisState> &MaybePredState =
        BlockStates[Pred->getBlockID()];
    if (!MaybePredState)
      continue;

    TypeErasedDataflowAnalysisState PredState = MaybePredState.value();
    if (BuiltinTransferOpts) {
      if (const Stmt *PredTerminatorStmt = Pred->getTerminatorStmt()) {
        const StmtToEnvMapImpl StmtToEnv(CFCtx, BlockStates);
        TerminatorVisitor(StmtToEnv, PredState.Env,
                          blockIndexInPredecessor(*Pred, Block),
                          *BuiltinTransferOpts)
            .Visit(PredTerminatorStmt);
      }
    }

    if (MaybeState) {
      Analysis.joinTypeErased(MaybeState->Lattice, PredState.Lattice);
      MaybeState->Env.join(PredState.Env, Analysis);
    } else {
      MaybeState = std::move(PredState);
    }
  }
  if (!MaybeState) {
    // FIXME: Consider passing `Block` to `Analysis.typeErasedInitialElement()`
    // to enable building analyses like computation of dominators that
    // initialize the state of each basic block differently.
    MaybeState.emplace(Analysis.typeErasedInitialElement(), InitEnv);
  }
  return *MaybeState;
}

/// Transfers `State` by evaluating `CfgStmt` in the context of `Analysis`.
/// `HandleTransferredStmt` (if provided) will be applied to `CfgStmt`, after it
/// is evaluated.
static void transferCFGStmt(
    const ControlFlowContext &CFCtx,
    llvm::ArrayRef<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates,
    const CFGStmt &CfgStmt, TypeErasedDataflowAnalysis &Analysis,
    TypeErasedDataflowAnalysisState &State,
    std::function<void(const CFGStmt &,
                       const TypeErasedDataflowAnalysisState &)>
        HandleTransferredStmt) {
  const Stmt *S = CfgStmt.getStmt();
  assert(S != nullptr);

  auto BuiltinTransferOpts = Analysis.builtinTransferOptions();
  if (BuiltinTransferOpts)
    transfer(StmtToEnvMapImpl(CFCtx, BlockStates), *S, State.Env,
             *BuiltinTransferOpts);
  Analysis.transferTypeErased(S, State.Lattice, State.Env);

  if (HandleTransferredStmt != nullptr)
    HandleTransferredStmt(CfgStmt, State);
}

/// Transfers `State` by evaluating `CfgInit`.
static void transferCFGInitializer(const CFGInitializer &CfgInit,
                                   TypeErasedDataflowAnalysisState &State) {
  const auto &ThisLoc = *cast<AggregateStorageLocation>(
      State.Env.getThisPointeeStorageLocation());

  const CXXCtorInitializer *Initializer = CfgInit.getInitializer();
  assert(Initializer != nullptr);

  const FieldDecl *Member = Initializer->getMember();
  if (Member == nullptr)
    // Not a field initializer.
    return;

  auto *InitStmt = Initializer->getInit();
  assert(InitStmt != nullptr);

  auto *InitStmtLoc =
      State.Env.getStorageLocation(*InitStmt, SkipPast::Reference);
  if (InitStmtLoc == nullptr)
    return;

  auto *InitStmtVal = State.Env.getValue(*InitStmtLoc);
  if (InitStmtVal == nullptr)
    return;

  if (Member->getType()->isReferenceType()) {
    auto &MemberLoc = ThisLoc.getChild(*Member);
    State.Env.setValue(MemberLoc,
                       State.Env.takeOwnership(
                           std::make_unique<ReferenceValue>(*InitStmtLoc)));
  } else {
    auto &MemberLoc = ThisLoc.getChild(*Member);
    State.Env.setValue(MemberLoc, *InitStmtVal);
  }
}

TypeErasedDataflowAnalysisState transferBlock(
    const ControlFlowContext &CFCtx,
    llvm::ArrayRef<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates,
    const CFGBlock &Block, const Environment &InitEnv,
    TypeErasedDataflowAnalysis &Analysis,
    std::function<void(const CFGStmt &,
                       const TypeErasedDataflowAnalysisState &)>
        HandleTransferredStmt) {
  TypeErasedDataflowAnalysisState State =
      computeBlockInputState(CFCtx, BlockStates, Block, InitEnv, Analysis);
  for (const CFGElement &Element : Block) {
    switch (Element.getKind()) {
    case CFGElement::Statement:
      transferCFGStmt(CFCtx, BlockStates, *Element.getAs<CFGStmt>(), Analysis,
                      State, HandleTransferredStmt);
      break;
    case CFGElement::Initializer:
      if (Analysis.builtinTransferOptions())
        transferCFGInitializer(*Element.getAs<CFGInitializer>(), State);
      break;
    default:
      // FIXME: Evaluate other kinds of `CFGElement`.
      break;
    }
  }
  return State;
}

llvm::Expected<std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>>
runTypeErasedDataflowAnalysis(
    const ControlFlowContext &CFCtx, TypeErasedDataflowAnalysis &Analysis,
    const Environment &InitEnv,
    std::function<void(const CFGStmt &,
                       const TypeErasedDataflowAnalysisState &)>
        PostVisitStmt) {
  PostOrderCFGView POV(&CFCtx.getCFG());
  ForwardDataflowWorklist Worklist(CFCtx.getCFG(), &POV);

  std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>> BlockStates(
      CFCtx.getCFG().size(), llvm::None);

  // The entry basic block doesn't contain statements so it can be skipped.
  const CFGBlock &Entry = CFCtx.getCFG().getEntry();
  BlockStates[Entry.getBlockID()] = {Analysis.typeErasedInitialElement(),
                                     InitEnv};
  Worklist.enqueueSuccessors(&Entry);

  // Bugs in lattices and transfer functions can prevent the analysis from
  // converging. To limit the damage (infinite loops) that these bugs can cause,
  // limit the number of iterations.
  // FIXME: Consider making the maximum number of iterations configurable.
  // FIXME: Consider restricting the number of backedges followed, rather than
  // iterations.
  // FIXME: Set up statistics (see llvm/ADT/Statistic.h) to count average number
  // of iterations, number of functions that time out, etc.
  static constexpr uint32_t MaxAverageVisitsPerBlock = 4;
  static constexpr uint32_t AbsoluteMaxIterations = 1 << 16;
  const uint32_t RelativeMaxIterations =
      MaxAverageVisitsPerBlock * BlockStates.size();
  const uint32_t MaxIterations =
      std::min(RelativeMaxIterations, AbsoluteMaxIterations);
  uint32_t Iterations = 0;
  while (const CFGBlock *Block = Worklist.dequeue()) {
    if (++Iterations > MaxIterations) {
      return llvm::createStringError(std::errc::timed_out,
                                     "maximum number of iterations reached");
    }

    const llvm::Optional<TypeErasedDataflowAnalysisState> &OldBlockState =
        BlockStates[Block->getBlockID()];
    TypeErasedDataflowAnalysisState NewBlockState =
        transferBlock(CFCtx, BlockStates, *Block, InitEnv, Analysis);

    if (OldBlockState &&
        Analysis.isEqualTypeErased(OldBlockState.value().Lattice,
                                   NewBlockState.Lattice) &&
        OldBlockState->Env.equivalentTo(NewBlockState.Env, Analysis)) {
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

  if (PostVisitStmt) {
    for (const CFGBlock *Block : CFCtx.getCFG()) {
      // Skip blocks that were not evaluated.
      if (!BlockStates[Block->getBlockID()])
        continue;

      transferBlock(CFCtx, BlockStates, *Block, InitEnv, Analysis,
                    PostVisitStmt);
    }
  }

  return BlockStates;
}

} // namespace dataflow
} // namespace clang
