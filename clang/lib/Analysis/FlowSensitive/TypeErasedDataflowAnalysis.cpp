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
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

#include "clang/AST/ASTDumper.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "clang/Analysis/FlowSensitive/RecordOps.h"
#include "clang/Analysis/FlowSensitive/Transfer.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "clang-dataflow"

namespace clang {
namespace dataflow {

/// Returns the index of `Block` in the successors of `Pred`.
static int blockIndexInPredecessor(const CFGBlock &Pred,
                                   const CFGBlock &Block) {
  auto BlockPos = llvm::find_if(
      Pred.succs(), [&Block](const CFGBlock::AdjacentBlock &Succ) {
        return Succ && Succ->getBlockID() == Block.getBlockID();
      });
  return BlockPos - Pred.succ_begin();
}

static bool isLoopHead(const CFGBlock &B) {
  if (const auto *T = B.getTerminatorStmt())
    switch (T->getStmtClass()) {
      case Stmt::WhileStmtClass:
      case Stmt::DoStmtClass:
      case Stmt::ForStmtClass:
      case Stmt::CXXForRangeStmtClass:
        return true;
      default:
        return false;
    }

  return false;
}

namespace {

// The return type of the visit functions in TerminatorVisitor. The first
// element represents the terminator expression (that is the conditional
// expression in case of a path split in the CFG). The second element
// represents whether the condition was true or false.
using TerminatorVisitorRetTy = std::pair<const Expr *, bool>;

/// Extends the flow condition of an environment based on a terminator
/// statement.
class TerminatorVisitor
    : public ConstStmtVisitor<TerminatorVisitor, TerminatorVisitorRetTy> {
public:
  TerminatorVisitor(const StmtToEnvMap &StmtToEnv, Environment &Env,
                    int BlockSuccIdx)
      : StmtToEnv(StmtToEnv), Env(Env), BlockSuccIdx(BlockSuccIdx) {}

  TerminatorVisitorRetTy VisitIfStmt(const IfStmt *S) {
    auto *Cond = S->getCond();
    assert(Cond != nullptr);
    return extendFlowCondition(*Cond);
  }

  TerminatorVisitorRetTy VisitWhileStmt(const WhileStmt *S) {
    auto *Cond = S->getCond();
    assert(Cond != nullptr);
    return extendFlowCondition(*Cond);
  }

  TerminatorVisitorRetTy VisitDoStmt(const DoStmt *S) {
    auto *Cond = S->getCond();
    assert(Cond != nullptr);
    return extendFlowCondition(*Cond);
  }

  TerminatorVisitorRetTy VisitForStmt(const ForStmt *S) {
    auto *Cond = S->getCond();
    if (Cond != nullptr)
      return extendFlowCondition(*Cond);
    return {nullptr, false};
  }

  TerminatorVisitorRetTy VisitCXXForRangeStmt(const CXXForRangeStmt *) {
    // Don't do anything special for CXXForRangeStmt, because the condition
    // (being implicitly generated) isn't visible from the loop body.
    return {nullptr, false};
  }

  TerminatorVisitorRetTy VisitBinaryOperator(const BinaryOperator *S) {
    assert(S->getOpcode() == BO_LAnd || S->getOpcode() == BO_LOr);
    auto *LHS = S->getLHS();
    assert(LHS != nullptr);
    return extendFlowCondition(*LHS);
  }

  TerminatorVisitorRetTy
  VisitConditionalOperator(const ConditionalOperator *S) {
    auto *Cond = S->getCond();
    assert(Cond != nullptr);
    return extendFlowCondition(*Cond);
  }

private:
  TerminatorVisitorRetTy extendFlowCondition(const Expr &Cond) {
    // The terminator sub-expression might not be evaluated.
    if (Env.getValue(Cond) == nullptr)
      transfer(StmtToEnv, Cond, Env);

    auto *Val = cast_or_null<BoolValue>(Env.getValue(Cond));
    // Value merging depends on flow conditions from different environments
    // being mutually exclusive -- that is, they cannot both be true in their
    // entirety (even if they may share some clauses). So, we need *some* value
    // for the condition expression, even if just an atom.
    if (Val == nullptr) {
      Val = &Env.makeAtomicBoolValue();
      Env.setValue(Cond, *Val);
    }

    bool ConditionValue = true;
    // The condition must be inverted for the successor that encompasses the
    // "else" branch, if such exists.
    if (BlockSuccIdx == 1) {
      Val = &Env.makeNot(*Val);
      ConditionValue = false;
    }

    Env.addToFlowCondition(Val->formula());
    return {&Cond, ConditionValue};
  }

  const StmtToEnvMap &StmtToEnv;
  Environment &Env;
  int BlockSuccIdx;
};

/// Holds data structures required for running dataflow analysis.
struct AnalysisContext {
  AnalysisContext(const ControlFlowContext &CFCtx,
                  TypeErasedDataflowAnalysis &Analysis,
                  const Environment &InitEnv,
                  llvm::ArrayRef<std::optional<TypeErasedDataflowAnalysisState>>
                      BlockStates)
      : CFCtx(CFCtx), Analysis(Analysis), InitEnv(InitEnv),
        Log(*InitEnv.getDataflowAnalysisContext().getOptions().Log),
        BlockStates(BlockStates) {
    Log.beginAnalysis(CFCtx, Analysis);
  }
  ~AnalysisContext() { Log.endAnalysis(); }

  /// Contains the CFG being analyzed.
  const ControlFlowContext &CFCtx;
  /// The analysis to be run.
  TypeErasedDataflowAnalysis &Analysis;
  /// Initial state to start the analysis.
  const Environment &InitEnv;
  Logger &Log;
  /// Stores the state of a CFG block if it has been evaluated by the analysis.
  /// The indices correspond to the block IDs.
  llvm::ArrayRef<std::optional<TypeErasedDataflowAnalysisState>> BlockStates;
};

class PrettyStackTraceAnalysis : public llvm::PrettyStackTraceEntry {
public:
  PrettyStackTraceAnalysis(const ControlFlowContext &CFCtx, const char *Message)
      : CFCtx(CFCtx), Message(Message) {}

  void print(raw_ostream &OS) const override {
    OS << Message << "\n";
    OS << "Decl:\n";
    CFCtx.getDecl().dump(OS);
    OS << "CFG:\n";
    CFCtx.getCFG().print(OS, LangOptions(), false);
  }

private:
  const ControlFlowContext &CFCtx;
  const char *Message;
};

class PrettyStackTraceCFGElement : public llvm::PrettyStackTraceEntry {
public:
  PrettyStackTraceCFGElement(const CFGElement &Element, int BlockIdx,
                             int ElementIdx, const char *Message)
      : Element(Element), BlockIdx(BlockIdx), ElementIdx(ElementIdx),
        Message(Message) {}

  void print(raw_ostream &OS) const override {
    OS << Message << ": Element [B" << BlockIdx << "." << ElementIdx << "]\n";
    if (auto Stmt = Element.getAs<CFGStmt>()) {
      OS << "Stmt:\n";
      ASTDumper Dumper(OS, false);
      Dumper.Visit(Stmt->getStmt());
    }
  }

private:
  const CFGElement &Element;
  int BlockIdx;
  int ElementIdx;
  const char *Message;
};

// Builds a joined TypeErasedDataflowAnalysisState from 0 or more sources,
// each of which may be owned (built as part of the join) or external (a
// reference to an Environment that will outlive the builder).
// Avoids unneccesary copies of the environment.
class JoinedStateBuilder {
  AnalysisContext &AC;
  std::vector<const TypeErasedDataflowAnalysisState *> All;
  std::deque<TypeErasedDataflowAnalysisState> Owned;

  TypeErasedDataflowAnalysisState
  join(const TypeErasedDataflowAnalysisState &L,
       const TypeErasedDataflowAnalysisState &R) {
    return {AC.Analysis.joinTypeErased(L.Lattice, R.Lattice),
            Environment::join(L.Env, R.Env, AC.Analysis)};
  }

public:
  JoinedStateBuilder(AnalysisContext &AC) : AC(AC) {}

  void addOwned(TypeErasedDataflowAnalysisState State) {
    Owned.push_back(std::move(State));
    All.push_back(&Owned.back());
  }
  void addUnowned(const TypeErasedDataflowAnalysisState &State) {
    All.push_back(&State);
  }
  TypeErasedDataflowAnalysisState take() && {
    if (All.empty())
      // FIXME: Consider passing `Block` to Analysis.typeErasedInitialElement
      // to enable building analyses like computation of dominators that
      // initialize the state of each basic block differently.
      return {AC.Analysis.typeErasedInitialElement(), AC.InitEnv.fork()};
    if (All.size() == 1)
      return Owned.empty() ? All.front()->fork() : std::move(Owned.front());

    auto Result = join(*All[0], *All[1]);
    for (unsigned I = 2; I < All.size(); ++I)
      Result = join(Result, *All[I]);
    return Result;
  }
};

} // namespace

/// Computes the input state for a given basic block by joining the output
/// states of its predecessors.
///
/// Requirements:
///
///   All predecessors of `Block` except those with loop back edges must have
///   already been transferred. States in `AC.BlockStates` that are set to
///   `std::nullopt` represent basic blocks that are not evaluated yet.
static TypeErasedDataflowAnalysisState
computeBlockInputState(const CFGBlock &Block, AnalysisContext &AC) {
  std::vector<const CFGBlock *> Preds(Block.pred_begin(), Block.pred_end());
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
    if (Block.succ_begin()->getReachableBlock() != nullptr &&
        Block.succ_begin()->getReachableBlock()->hasNoReturnElement()) {
      auto &StmtToBlock = AC.CFCtx.getStmtToBlock();
      auto StmtBlock = StmtToBlock.find(Block.getTerminatorStmt());
      assert(StmtBlock != StmtToBlock.end());
      llvm::erase_value(Preds, StmtBlock->getSecond());
    }
  }

  JoinedStateBuilder Builder(AC);
  for (const CFGBlock *Pred : Preds) {
    // Skip if the `Block` is unreachable or control flow cannot get past it.
    if (!Pred || Pred->hasNoReturnElement())
      continue;

    // Skip if `Pred` was not evaluated yet. This could happen if `Pred` has a
    // loop back edge to `Block`.
    const std::optional<TypeErasedDataflowAnalysisState> &MaybePredState =
        AC.BlockStates[Pred->getBlockID()];
    if (!MaybePredState)
      continue;

    if (AC.Analysis.builtinOptions()) {
      if (const Stmt *PredTerminatorStmt = Pred->getTerminatorStmt()) {
        // We have a terminator: we need to mutate an environment to describe
        // when the terminator is taken. Copy now.
        TypeErasedDataflowAnalysisState Copy = MaybePredState->fork();

        const StmtToEnvMap StmtToEnv(AC.CFCtx, AC.BlockStates);
        auto [Cond, CondValue] =
            TerminatorVisitor(StmtToEnv, Copy.Env,
                              blockIndexInPredecessor(*Pred, Block))
                .Visit(PredTerminatorStmt);
        if (Cond != nullptr)
          // FIXME: Call transferBranchTypeErased even if BuiltinTransferOpts
          // are not set.
          AC.Analysis.transferBranchTypeErased(CondValue, Cond, Copy.Lattice,
                                               Copy.Env);
        Builder.addOwned(std::move(Copy));
        continue;
      }
    }
    Builder.addUnowned(*MaybePredState);
  }
  return std::move(Builder).take();
}

/// Built-in transfer function for `CFGStmt`.
static void
builtinTransferStatement(const CFGStmt &Elt,
                         TypeErasedDataflowAnalysisState &InputState,
                         AnalysisContext &AC) {
  const Stmt *S = Elt.getStmt();
  assert(S != nullptr);
  transfer(StmtToEnvMap(AC.CFCtx, AC.BlockStates), *S, InputState.Env);
}

/// Built-in transfer function for `CFGInitializer`.
static void
builtinTransferInitializer(const CFGInitializer &Elt,
                           TypeErasedDataflowAnalysisState &InputState) {
  const CXXCtorInitializer *Init = Elt.getInitializer();
  assert(Init != nullptr);

  auto &Env = InputState.Env;
  auto &ThisLoc = *Env.getThisPointeeStorageLocation();

  if (!Init->isAnyMemberInitializer())
    // FIXME: Handle base initialization
    return;

  auto *InitExpr = Init->getInit();
  assert(InitExpr != nullptr);

  const FieldDecl *Member = nullptr;
  RecordStorageLocation *ParentLoc = &ThisLoc;
  StorageLocation *MemberLoc = nullptr;
  if (Init->isMemberInitializer()) {
    Member = Init->getMember();
    MemberLoc = ThisLoc.getChild(*Member);
  } else {
    IndirectFieldDecl *IndirectField = Init->getIndirectMember();
    assert(IndirectField != nullptr);
    MemberLoc = &ThisLoc;
    for (const auto *I : IndirectField->chain()) {
      Member = cast<FieldDecl>(I);
      ParentLoc = cast<RecordStorageLocation>(MemberLoc);
      MemberLoc = ParentLoc->getChild(*Member);
    }
  }
  assert(Member != nullptr);
  assert(MemberLoc != nullptr);

  // FIXME: Instead of these case distinctions, we would ideally want to be able
  // to simply use `Environment::createObject()` here, the same way that we do
  // this in `TransferVisitor::VisitInitListExpr()`. However, this would require
  // us to be able to build a list of fields that we then use to initialize an
  // `RecordStorageLocation` -- and the problem is that, when we get here,
  // the `RecordStorageLocation` already exists. We should explore if there's
  // anything that we can do to change this.
  if (Member->getType()->isReferenceType()) {
    auto *InitExprLoc = Env.getStorageLocation(*InitExpr);
    if (InitExprLoc == nullptr)
      return;

    ParentLoc->setChild(*Member, InitExprLoc);
  } else if (auto *InitExprVal = Env.getValue(*InitExpr)) {
    if (Member->getType()->isRecordType()) {
      auto *InitValStruct = cast<RecordValue>(InitExprVal);
      // FIXME: Rather than performing a copy here, we should really be
      // initializing the field in place. This would require us to propagate the
      // storage location of the field to the AST node that creates the
      // `RecordValue`.
      copyRecord(InitValStruct->getLoc(),
                 *cast<RecordStorageLocation>(MemberLoc), Env);
    } else {
      Env.setValue(*MemberLoc, *InitExprVal);
    }
  }
}

static void builtinTransfer(const CFGElement &Elt,
                            TypeErasedDataflowAnalysisState &State,
                            AnalysisContext &AC) {
  switch (Elt.getKind()) {
  case CFGElement::Statement:
    builtinTransferStatement(Elt.castAs<CFGStmt>(), State, AC);
    break;
  case CFGElement::Initializer:
    builtinTransferInitializer(Elt.castAs<CFGInitializer>(), State);
    break;
  default:
    // FIXME: Evaluate other kinds of `CFGElement`, including:
    // - When encountering `CFGLifetimeEnds`, remove the declaration from
    //   `Environment::DeclToLoc`. This would serve two purposes:
    //   a) Eliminate unnecessary clutter from `Environment::DeclToLoc`
    //   b) Allow us to implement an assertion that, when joining two
    //      `Environments`, the two `DeclToLoc` maps never contain entries that
    //      map the same declaration to different storage locations.
    //   Unfortunately, however, we can't currently process `CFGLifetimeEnds`
    //   because the corresponding CFG option `AddLifetime` is incompatible with
    //   the option 'AddImplicitDtors`, which we already use. We will first
    //   need to modify the CFG implementation to make these two options
    //   compatible before we can process `CFGLifetimeEnds`.
    break;
  }
}

/// Transfers `State` by evaluating each element in the `Block` based on the
/// `AC.Analysis` specified.
///
/// Built-in transfer functions (if the option for `ApplyBuiltinTransfer` is set
/// by the analysis) will be applied to the element before evaluation by the
/// user-specified analysis.
/// `PostVisitCFG` (if provided) will be applied to the element after evaluation
/// by the user-specified analysis.
static TypeErasedDataflowAnalysisState
transferCFGBlock(const CFGBlock &Block, AnalysisContext &AC,
                 std::function<void(const CFGElement &,
                                    const TypeErasedDataflowAnalysisState &)>
                     PostVisitCFG = nullptr) {
  AC.Log.enterBlock(Block, PostVisitCFG != nullptr);
  auto State = computeBlockInputState(Block, AC);
  AC.Log.recordState(State);
  int ElementIdx = 1;
  for (const auto &Element : Block) {
    PrettyStackTraceCFGElement CrashInfo(Element, Block.getBlockID(),
                                         ElementIdx++, "transferCFGBlock");

    AC.Log.enterElement(Element);
    // Built-in analysis
    if (AC.Analysis.builtinOptions()) {
      builtinTransfer(Element, State, AC);
    }

    // User-provided analysis
    AC.Analysis.transferTypeErased(Element, State.Lattice, State.Env);

    // Post processing
    if (PostVisitCFG) {
      PostVisitCFG(Element, State);
    }
    AC.Log.recordState(State);
  }
  return State;
}

llvm::Expected<std::vector<std::optional<TypeErasedDataflowAnalysisState>>>
runTypeErasedDataflowAnalysis(
    const ControlFlowContext &CFCtx, TypeErasedDataflowAnalysis &Analysis,
    const Environment &InitEnv,
    std::function<void(const CFGElement &,
                       const TypeErasedDataflowAnalysisState &)>
        PostVisitCFG) {
  PrettyStackTraceAnalysis CrashInfo(CFCtx, "runTypeErasedDataflowAnalysis");

  PostOrderCFGView POV(&CFCtx.getCFG());
  ForwardDataflowWorklist Worklist(CFCtx.getCFG(), &POV);

  std::vector<std::optional<TypeErasedDataflowAnalysisState>> BlockStates(
      CFCtx.getCFG().size());

  // The entry basic block doesn't contain statements so it can be skipped.
  const CFGBlock &Entry = CFCtx.getCFG().getEntry();
  BlockStates[Entry.getBlockID()] = {Analysis.typeErasedInitialElement(),
                                     InitEnv.fork()};
  Worklist.enqueueSuccessors(&Entry);

  AnalysisContext AC(CFCtx, Analysis, InitEnv, BlockStates);

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
    LLVM_DEBUG(llvm::dbgs()
               << "Processing Block " << Block->getBlockID() << "\n");
    if (++Iterations > MaxIterations) {
      return llvm::createStringError(std::errc::timed_out,
                                     "maximum number of iterations reached");
    }

    const std::optional<TypeErasedDataflowAnalysisState> &OldBlockState =
        BlockStates[Block->getBlockID()];
    TypeErasedDataflowAnalysisState NewBlockState =
        transferCFGBlock(*Block, AC);
    LLVM_DEBUG({
      llvm::errs() << "New Env:\n";
      NewBlockState.Env.dump();
    });

    if (OldBlockState) {
      LLVM_DEBUG({
        llvm::errs() << "Old Env:\n";
        OldBlockState->Env.dump();
      });
      if (isLoopHead(*Block)) {
        LatticeJoinEffect Effect1 = Analysis.widenTypeErased(
            NewBlockState.Lattice, OldBlockState->Lattice);
        LatticeJoinEffect Effect2 =
            NewBlockState.Env.widen(OldBlockState->Env, Analysis);
        if (Effect1 == LatticeJoinEffect::Unchanged &&
            Effect2 == LatticeJoinEffect::Unchanged) {
          // The state of `Block` didn't change from widening so there's no need
          // to revisit its successors.
          AC.Log.blockConverged();
          continue;
        }
      } else if (Analysis.isEqualTypeErased(OldBlockState->Lattice,
                                            NewBlockState.Lattice) &&
                 OldBlockState->Env.equivalentTo(NewBlockState.Env, Analysis)) {
        // The state of `Block` didn't change after transfer so there's no need
        // to revisit its successors.
        AC.Log.blockConverged();
        continue;
      }
    }

    BlockStates[Block->getBlockID()] = std::move(NewBlockState);

    // Do not add unreachable successor blocks to `Worklist`.
    if (Block->hasNoReturnElement())
      continue;

    Worklist.enqueueSuccessors(Block);
  }
  // FIXME: Consider evaluating unreachable basic blocks (those that have a
  // state set to `std::nullopt` at this point) to also analyze dead code.

  if (PostVisitCFG) {
    for (const CFGBlock *Block : CFCtx.getCFG()) {
      // Skip blocks that were not evaluated.
      if (!BlockStates[Block->getBlockID()])
        continue;
      transferCFGBlock(*Block, AC, PostVisitCFG);
    }
  }

  return std::move(BlockStates);
}

} // namespace dataflow
} // namespace clang
