//===- bolt/Passes/CMOVConversion.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CMOV conversion pass.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/CMOVConversion.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <numeric>

#define DEBUG_TYPE "cmov"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static cl::opt<int> BiasThreshold(
    "cmov-conversion-bias-threshold",
    cl::desc("minimum condition bias (pct) to perform a CMOV conversion, "
             "-1 to not account bias"),
    cl::ReallyHidden, cl::init(1), cl::cat(BoltOptCategory));

static cl::opt<int> MispredictionThreshold(
    "cmov-conversion-misprediction-threshold",
    cl::desc("minimum misprediction rate (pct) to perform a CMOV conversion, "
             "-1 to not account misprediction rate"),
    cl::ReallyHidden, cl::init(5), cl::cat(BoltOptCategory));

static cl::opt<bool> ConvertStackMemOperand(
    "cmov-conversion-convert-stack-mem-operand",
    cl::desc("convert moves with stack memory operand (potentially unsafe)"),
    cl::ReallyHidden, cl::init(false), cl::cat(BoltOptCategory));

static cl::opt<bool> ConvertBasePtrStackMemOperand(
    "cmov-conversion-convert-rbp-stack-mem-operand",
    cl::desc("convert moves with rbp stack memory operand (unsafe, must be off "
             "for binaries compiled with -fomit-frame-pointer)"),
    cl::ReallyHidden, cl::init(false), cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

// Return true if the CFG conforms to the following subgraph:
// Predecessor
//   /     \
//  |     RHS
//   \     /
//     LHS
// Caller guarantees that LHS and RHS share the same predecessor.
bool isIfThenSubgraph(const BinaryBasicBlock &LHS,
                      const BinaryBasicBlock &RHS) {
  if (LHS.pred_size() != 2 || RHS.pred_size() != 1)
    return false;

  // Sanity check
  BinaryBasicBlock *Predecessor = *RHS.pred_begin();
  assert(Predecessor && LHS.isPredecessor(Predecessor) && "invalid subgraph");
  (void)Predecessor;

  if (!LHS.isPredecessor(&RHS))
    return false;
  if (RHS.succ_size() != 1)
    return false;
  return true;
}

bool matchCFGSubgraph(BinaryBasicBlock &BB, BinaryBasicBlock *&ConditionalSucc,
                      BinaryBasicBlock *&UnconditionalSucc,
                      bool &IsConditionalTaken) {
  BinaryBasicBlock *TakenSucc = BB.getConditionalSuccessor(true);
  BinaryBasicBlock *FallthroughSucc = BB.getConditionalSuccessor(false);
  bool IsIfThenTaken = isIfThenSubgraph(*FallthroughSucc, *TakenSucc);
  bool IsIfThenFallthrough = isIfThenSubgraph(*TakenSucc, *FallthroughSucc);
  if (!IsIfThenFallthrough && !IsIfThenTaken)
    return false;
  assert((!IsIfThenFallthrough || !IsIfThenTaken) && "Invalid subgraph");

  // Output parameters
  ConditionalSucc = IsIfThenTaken ? TakenSucc : FallthroughSucc;
  UnconditionalSucc = IsIfThenTaken ? FallthroughSucc : TakenSucc;
  IsConditionalTaken = IsIfThenTaken;
  return true;
}

// Return true if basic block instructions can be converted into cmov(s).
bool canConvertInstructions(const BinaryContext &BC, const BinaryBasicBlock &BB,
                            unsigned CC) {
  if (BB.empty())
    return false;
  const MCInst *LastInst = BB.getLastNonPseudoInstr();
  // Only pseudo instructions, can't be converted into CMOV
  if (LastInst == nullptr)
    return false;
  for (const MCInst &Inst : BB) {
    if (BC.MIB->isPseudo(Inst))
      continue;
    // Unconditional branch as a last instruction is OK
    if (&Inst == LastInst && BC.MIB->isUnconditionalBranch(Inst))
      continue;
    MCInst Cmov(Inst);
    // GPR move is OK
    if (!BC.MIB->convertMoveToConditionalMove(
            Cmov, CC, opts::ConvertStackMemOperand,
            opts::ConvertBasePtrStackMemOperand)) {
      LLVM_DEBUG({
        dbgs() << BB.getName() << ": can't convert instruction ";
        BC.printInstruction(dbgs(), Cmov);
      });
      return false;
    }
  }
  return true;
}

void convertMoves(const BinaryContext &BC, BinaryBasicBlock &BB, unsigned CC) {
  for (auto II = BB.begin(), IE = BB.end(); II != IE; ++II) {
    if (BC.MIB->isPseudo(*II))
      continue;
    if (BC.MIB->isUnconditionalBranch(*II)) {
      // XXX: this invalidates II but we return immediately
      BB.eraseInstruction(II);
      return;
    }
    bool Result = BC.MIB->convertMoveToConditionalMove(
        *II, CC, opts::ConvertStackMemOperand,
        opts::ConvertBasePtrStackMemOperand);
    assert(Result && "unexpected instruction");
    (void)Result;
  }
}

// Returns misprediction rate if the profile data is available, -1 otherwise.
std::pair<int, uint64_t>
calculateMispredictionRate(const BinaryBasicBlock &BB) {
  uint64_t TotalExecCount = 0;
  uint64_t TotalMispredictionCount = 0;
  for (auto BI : BB.branch_info()) {
    TotalExecCount += BI.Count;
    if (BI.MispredictedCount != BinaryBasicBlock::COUNT_INFERRED)
      TotalMispredictionCount += BI.MispredictedCount;
  }
  if (!TotalExecCount)
    return {-1, TotalMispredictionCount};
  return {100.0f * TotalMispredictionCount / TotalExecCount,
          TotalMispredictionCount};
}

// Returns conditional succ bias if the profile is available, -1 otherwise.
int calculateConditionBias(const BinaryBasicBlock &BB,
                           const BinaryBasicBlock &ConditionalSucc) {
  if (auto BranchStats = BB.getBranchStats(&ConditionalSucc))
    return BranchStats->first;
  return -1;
}

void CMOVConversion::Stats::dumpTo(raw_ostream &OS) {
  OS << "converted static " << StaticPerformed << "/" << StaticPossible
     << formatv(" ({0:P}) ", getStaticRatio())
     << "hammock(s) into CMOV sequences, with dynamic execution count "
     << DynamicPerformed << "/" << DynamicPossible
     << formatv(" ({0:P}), ", getDynamicRatio()) << "saving " << RemovedMP
     << "/" << PossibleMP << formatv(" ({0:P}) ", getMPRatio())
     << "mispredictions\n";
}

void CMOVConversion::runOnFunction(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  bool Modified = false;
  // Function-local stats
  Stats Local;
  // Traverse blocks in RPO, merging block with a converted cmov with its
  // successor.
  for (BinaryBasicBlock *BB : post_order(&Function)) {
    uint64_t BBExecCount = BB->getKnownExecutionCount();
    if (BB->empty() ||          // The block must have instructions
        BBExecCount == 0 ||     // must be hot
        BB->succ_size() != 2 || // with two successors
        BB->hasJumpTable())     // no jump table
      continue;

    assert(BB->isValid() && "traversal internal error");

    // Check branch instruction
    auto BranchInstrIter = BB->getLastNonPseudo();
    if (BranchInstrIter == BB->rend() ||
        !BC.MIB->isConditionalBranch(*BranchInstrIter))
      continue;

    // Check successors
    BinaryBasicBlock *ConditionalSucc, *UnconditionalSucc;
    bool IsConditionalTaken;
    if (!matchCFGSubgraph(*BB, ConditionalSucc, UnconditionalSucc,
                          IsConditionalTaken)) {
      LLVM_DEBUG(dbgs() << BB->getName() << ": couldn't match hammock\n");
      continue;
    }

    unsigned CC = BC.MIB->getCondCode(*BranchInstrIter);
    if (!IsConditionalTaken)
      CC = BC.MIB->getInvertedCondCode(CC);
    // Check contents of the conditional block
    if (!canConvertInstructions(BC, *ConditionalSucc, CC))
      continue;

    int ConditionBias = calculateConditionBias(*BB, *ConditionalSucc);
    int MispredictionRate = 0;
    uint64_t MispredictionCount = 0;
    std::tie(MispredictionRate, MispredictionCount) =
        calculateMispredictionRate(*BB);

    Local.StaticPossible++;
    Local.DynamicPossible += BBExecCount;
    Local.PossibleMP += MispredictionCount;

    // If the conditional successor is never executed, don't convert it
    if (ConditionBias < opts::BiasThreshold) {
      LLVM_DEBUG(dbgs() << BB->getName() << "->" << ConditionalSucc->getName()
                        << " bias = " << ConditionBias
                        << ", less than threshold " << opts::BiasThreshold
                        << '\n');
      continue;
    }

    // Check the misprediction rate of a branch
    if (MispredictionRate < opts::MispredictionThreshold) {
      LLVM_DEBUG(dbgs() << BB->getName() << " misprediction rate = "
                        << MispredictionRate << ", less than threshold "
                        << opts::MispredictionThreshold << '\n');
      continue;
    }

    // remove conditional branch
    BB->eraseInstruction(std::prev(BranchInstrIter.base()));
    BB->removeAllSuccessors();
    // Convert instructions from the conditional successor into cmov's in BB.
    convertMoves(BC, *ConditionalSucc, CC);
    BB->addInstructions(ConditionalSucc->begin(), ConditionalSucc->end());
    ConditionalSucc->markValid(false);

    // RPO traversal guarantees that the successor is visited and merged if
    // necessary. Merge the unconditional successor into the current block.
    BB->addInstructions(UnconditionalSucc->begin(), UnconditionalSucc->end());
    UnconditionalSucc->moveAllSuccessorsTo(BB);
    UnconditionalSucc->markValid(false);
    Local.StaticPerformed++;
    Local.DynamicPerformed += BBExecCount;
    Local.RemovedMP += MispredictionCount;
    Modified = true;
  }
  if (Modified)
    Function.eraseInvalidBBs();
  if (opts::Verbosity > 1) {
    BC.outs() << "BOLT-INFO: CMOVConversion: " << Function << ", ";
    Local.dumpTo(BC.outs());
  }
  Global = Global + Local;
}

Error CMOVConversion::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    BinaryFunction &Function = It.second;
    if (!shouldOptimize(Function))
      continue;
    runOnFunction(Function);
  }

  BC.outs() << "BOLT-INFO: CMOVConversion total: ";
  Global.dumpTo(BC.outs());
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
