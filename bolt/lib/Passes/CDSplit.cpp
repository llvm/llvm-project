//===- bolt/Passes/CDSplit.cpp - Pass for splitting function code 3-way
//--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CDSplit pass.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/CDSplit.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "bolt-opts"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> UseCDSplit;
extern cl::opt<bool> SplitEH;
extern cl::opt<unsigned> ExecutionCountThreshold;

static cl::opt<double> CallScale("call-scale",
                                 cl::desc("Call score scale coefficient"),
                                 cl::init(0.95), cl::ReallyHidden,
                                 cl::cat(BoltOptCategory));
static cl::opt<double> CallPower("call-power", cl::desc("Call score power"),
                                 cl::init(0.05), cl::ReallyHidden,
                                 cl::cat(BoltOptCategory));
static cl::opt<double> JumpPower("jump-power", cl::desc("Jump score power"),
                                 cl::init(0.15), cl::ReallyHidden,
                                 cl::cat(BoltOptCategory));
} // namespace opts

namespace llvm {
namespace bolt {

namespace {
/// Return true if the function should be considered for building call graph.
bool shouldConsider(const BinaryFunction &BF) {
  return BF.hasValidIndex() && BF.hasValidProfile() && !BF.empty();
}

/// Find (un)conditional branch instruction info of the basic block.
JumpInfo analyzeBranches(BinaryBasicBlock *BB) {
  JumpInfo BBJumpInfo;
  const MCSymbol *TBB = nullptr;
  const MCSymbol *FBB = nullptr;
  MCInst *CondBranch = nullptr;
  MCInst *UncondBranch = nullptr;
  if (BB->analyzeBranch(TBB, FBB, CondBranch, UncondBranch)) {
    BBJumpInfo.HasUncondBranch = UncondBranch != nullptr;
    if (BB->succ_size() == 1) {
      BBJumpInfo.UncondSuccessor = BB->getSuccessor();
    } else if (BB->succ_size() == 2) {
      BBJumpInfo.CondSuccessor = BB->getConditionalSuccessor(true);
      BBJumpInfo.UncondSuccessor = BB->getConditionalSuccessor(false);
    }
  }
  return BBJumpInfo;
}

/// Compute the edge score of a call edge.
double computeCallScore(uint64_t CallCount, size_t CallLength) {
  // Increase call lengths by 1 to avoid raising 0 to a negative power.
  return opts::CallScale * static_cast<double>(CallCount) /
         std::pow(static_cast<double>(CallLength + 1), opts::CallPower);
}

/// Compute the edge score of a jump (branch) edge.
double computeJumpScore(uint64_t JumpCount, size_t JumpLength) {
  // Increase jump lengths by 1 to avoid raising 0 to a negative power.
  return static_cast<double>(JumpCount) /
         std::pow(static_cast<double>(JumpLength + 1), opts::JumpPower);
}
} // anonymous namespace

bool CDSplit::shouldOptimize(const BinaryFunction &BF) const {
  // Do not split functions with a small execution count.
  if (BF.getKnownExecutionCount() < opts::ExecutionCountThreshold)
    return false;

  // Do not split functions with at least one block that has no known
  // execution count due to incomplete information.
  // Do not split functions with only zero-execution count blocks
  // as there is not enough variation in block count to justify splitting.
  if (!BF.hasFullProfile() || BF.allBlocksCold())
    return false;

  return BinaryFunctionPass::shouldOptimize(BF);
}

/// Initialize algorithm's metadata.
void CDSplit::initialize(BinaryContext &BC) {
  // Construct a list of functions that are considered for building call graph.
  // Only those in this list that evaluates true for shouldOptimize are
  // candidates for 3-way splitting.
  std::vector<BinaryFunction *> SortedFunctions = BC.getSortedFunctions();
  FunctionsToConsider.reserve(SortedFunctions.size());
  for (BinaryFunction *BF : SortedFunctions) {
    if (shouldConsider(*BF))
      FunctionsToConsider.push_back(BF);
  }

  // Initialize auxiliary variables.
  for (BinaryFunction *BF : FunctionsToConsider) {
    // Calculate the size of each BB after hot-cold splitting.
    // This populates BinaryBasicBlock::OutputAddressRange which
    // can be used to compute the size of each BB.
    BC.calculateEmittedSize(*BF, /*FixBranches=*/true);

    for (BinaryBasicBlock *BB : BF->getLayout().blocks()) {
      // Unique global index.
      GlobalIndices[BB] = TotalNumBlocks;
      TotalNumBlocks++;

      // Block size after hot-cold splitting.
      BBSizes[BB] = BB->getOutputAddressRange().second -
                    BB->getOutputAddressRange().first;

      // Hot block offset after hot-cold splitting.
      BBOffsets[BB] = OrigHotSectionSize;
      if (!BB->isSplit())
        OrigHotSectionSize += BBSizes[BB];

      // Conditional and unconditional successors.
      JumpInfos[BB] = analyzeBranches(BB);
    }
  }

  // Build call graph.
  Callers.resize(TotalNumBlocks);
  Callees.resize(TotalNumBlocks);
  for (BinaryFunction *SrcFunction : FunctionsToConsider) {
    for (BinaryBasicBlock &SrcBB : SrcFunction->blocks()) {
      // Skip blocks that are not executed
      if (SrcBB.getKnownExecutionCount() == 0)
        continue;

      // Find call instructions and extract target symbols from each one
      for (const MCInst &Inst : SrcBB) {
        if (!BC.MIB->isCall(Inst))
          continue;

        // Call info
        const MCSymbol *DstSym = BC.MIB->getTargetSymbol(Inst);
        // Ignore calls w/o information
        if (!DstSym)
          continue;

        const BinaryFunction *DstFunction = BC.getFunctionForSymbol(DstSym);
        // Ignore calls that do not have a valid target, but do not ignore
        // recursive calls, because caller block could be moved to warm.
        if (!DstFunction || DstFunction->getLayout().block_empty())
          continue;

        const BinaryBasicBlock *DstBB = &(DstFunction->front());

        // Record the call only if DstBB is also in FunctionsToConsider.
        if (GlobalIndices.contains(DstBB)) {
          Callers[GlobalIndices[DstBB]].push_back(&SrcBB);
          Callees[GlobalIndices[&SrcBB]].push_back(DstBB);
        }
      }
    }
  }

  // If X86, long branch instructions take more bytes than short branches.
  // Adjust sizes of branch instructions used to approximate block size
  // increase due to hot-warm splitting.
  if (BC.isX86()) {
    // a short branch takes 2 bytes.
    BRANCH_SIZE = 2;
    // a long uncond branch takes BRANCH_SIZE + 3 bytes.
    LONG_UNCOND_BRANCH_SIZE_DELTA = 3;
    // a long cond branch takes BRANCH_SIZE + 4 bytes.
    LONG_COND_BRANCH_SIZE_DELTA = 4;
  }
}

/// Get a collection of "shortenable" calls, that is, calls of type X->Y
/// when the function order is [... X ... BF ... Y ...].
/// If the hot fragment size of BF is reduced, then such calls are guaranteed
/// to get shorter by the reduced hot fragment size.
std::vector<CallInfo> CDSplit::extractCoverCalls(const BinaryFunction &BF) {
  // Record the length and the count of the calls that can be shortened
  std::vector<CallInfo> CoverCalls;
  if (opts::CallScale == 0)
    return CoverCalls;

  const BinaryFunction *ThisBF = &BF;
  const BinaryBasicBlock *ThisBB = &(ThisBF->front());
  size_t ThisGI = GlobalIndices[ThisBB];

  for (BinaryFunction *DstBF : FunctionsToConsider) {
    const BinaryBasicBlock *DstBB = &(DstBF->front());
    if (DstBB->getKnownExecutionCount() == 0)
      continue;

    size_t DstGI = GlobalIndices[DstBB];
    for (const BinaryBasicBlock *SrcBB : Callers[DstGI]) {
      const BinaryFunction *SrcBF = SrcBB->getFunction();
      if (ThisBF == SrcBF)
        continue;

      const size_t CallCount = SrcBB->getKnownExecutionCount();

      size_t SrcGI = GlobalIndices[SrcBB];

      bool IsCoverCall = (SrcGI < ThisGI && ThisGI < DstGI) ||
                         (DstGI <= ThisGI && ThisGI < SrcGI);
      if (!IsCoverCall)
        continue;

      size_t SrcBBEndAddr = BBOffsets[SrcBB] + BBSizes[SrcBB];
      size_t DstBBStartAddr = BBOffsets[DstBB];
      size_t CallLength = AbsoluteDifference(SrcBBEndAddr, DstBBStartAddr);
      CallInfo CI{CallLength, CallCount};
      CoverCalls.emplace_back(CI);
    }
  }
  return CoverCalls;
}

std::pair<size_t, size_t>
CDSplit::estimatePostSplitBBAddress(const BasicBlockOrder &BlockOrder,
                                    const size_t SplitIndex) {
  assert(SplitIndex < BlockOrder.size() && "Invalid split index");
  // Helper function estimating if a branch needs a longer branch instruction.
  // The function returns true if the following two conditions are satisfied:
  // condition 1. One of SrcBB and DstBB is in hot, the other is in warm.
  // condition 2. The pre-split branch distance is within 8 bytes.
  auto needNewLongBranch = [&](const BinaryBasicBlock *SrcBB,
                               const BinaryBasicBlock *DstBB) {
    if (!SrcBB || !DstBB)
      return false;
    // The following checks for condition 1.
    if (SrcBB->isSplit() || DstBB->isSplit())
      return false;
    if ((SrcBB->getLayoutIndex() <= SplitIndex) ==
        (DstBB->getLayoutIndex() <= SplitIndex))
      return false;
    // The following checks for condition 2.
    return (AbsoluteDifference(BBOffsets[DstBB],
                               BBOffsets[SrcBB] + BBSizes[SrcBB]) <=
            std::numeric_limits<int8_t>::max());
  };

  // Populate BB.OutputAddressRange with estimated new start and end addresses
  // and compute the old end address of the hot section and the new end address
  // of the hot section.
  size_t OldHotEndAddr;
  size_t NewHotEndAddr;
  size_t CurrentAddr = BBOffsets[BlockOrder[0]];
  for (BinaryBasicBlock *BB : BlockOrder) {
    // We only care about new addresses of blocks in hot/warm.
    if (BB->isSplit())
      break;
    size_t NewSize = BBSizes[BB];
    // Need to add a new branch instruction if a fall-through branch is split.
    bool NeedNewUncondBranch =
        (JumpInfos[BB].UncondSuccessor && !JumpInfos[BB].HasUncondBranch &&
         BB->getLayoutIndex() == SplitIndex);

    NewSize += BRANCH_SIZE * NeedNewUncondBranch +
               LONG_UNCOND_BRANCH_SIZE_DELTA *
                   needNewLongBranch(BB, JumpInfos[BB].UncondSuccessor) +
               LONG_COND_BRANCH_SIZE_DELTA *
                   needNewLongBranch(BB, JumpInfos[BB].CondSuccessor);
    BB->setOutputStartAddress(CurrentAddr);
    CurrentAddr += NewSize;
    BB->setOutputEndAddress(CurrentAddr);
    // Temporarily set the start address of the warm fragment of the current
    // function to be 0. We will update it later when we can get a better
    // estimate.
    if (BB->getLayoutIndex() == SplitIndex) {
      NewHotEndAddr = CurrentAddr;
      CurrentAddr = 0;
    }
    OldHotEndAddr = BBOffsets[BB] + BBSizes[BB];
  }

  // Update the start and end addresses of blocks in the warm fragment.
  // First get a better estimate of the start address of the warm fragment.
  assert(OrigHotSectionSize + NewHotEndAddr >= OldHotEndAddr);
  size_t WarmSectionStartAddr =
      OrigHotSectionSize + NewHotEndAddr - OldHotEndAddr;
  // Do the correction.
  for (size_t Index = SplitIndex + 1; Index < BlockOrder.size(); Index++) {
    BinaryBasicBlock *BB = BlockOrder[Index];
    // We only care about new addresses of blocks in warm.
    if (BB->isSplit())
      break;
    size_t StartAddrOffset = BB->getOutputAddressRange().first;
    size_t EndAddrOffset = BB->getOutputAddressRange().second;
    BB->setOutputStartAddress(WarmSectionStartAddr + StartAddrOffset);
    BB->setOutputEndAddress(WarmSectionStartAddr + EndAddrOffset);
  }

  return std::make_pair(OldHotEndAddr, NewHotEndAddr);
}

/// Compute sum of scores over jumps within \p BlockOrder given \p SplitIndex.
/// Increament Score.LocalScore in place by the sum.
void CDSplit::computeJumpScore(const BasicBlockOrder &BlockOrder,
                               const size_t SplitIndex, SplitScore &Score) {

  for (BinaryBasicBlock *SrcBB : BlockOrder) {
    if (SrcBB->getKnownExecutionCount() == 0)
      continue;

    size_t SrcBBEndAddr = SrcBB->getOutputAddressRange().second;

    for (const auto Pair : zip(SrcBB->successors(), SrcBB->branch_info())) {
      const BinaryBasicBlock *DstBB = std::get<0>(Pair);
      const BinaryBasicBlock::BinaryBranchInfo &Branch = std::get<1>(Pair);
      const size_t JumpCount = Branch.Count;

      if (JumpCount == 0)
        continue;

      size_t DstBBStartAddr = DstBB->getOutputAddressRange().first;
      size_t NewJumpLength = AbsoluteDifference(SrcBBEndAddr, DstBBStartAddr);
      Score.LocalScore += ::computeJumpScore(JumpCount, NewJumpLength);
    }
  }
}

/// Compute sum of scores over calls originated in the current function
/// given \p SplitIndex. Increament Score.LocalScore in place by the sum.
void CDSplit::computeLocalCallScore(const BasicBlockOrder &BlockOrder,
                                    const size_t SplitIndex,
                                    SplitScore &Score) {
  if (opts::CallScale == 0)
    return;

  // Global index of the last block in the current function.
  // This is later used to determine whether a call originated in the current
  // function is to a function that comes after the current function.
  size_t LastGlobalIndex = GlobalIndices[BlockOrder.back()];

  // The length of calls originated in the input function can increase /
  // decrease depending on the splitting decision.
  for (BinaryBasicBlock *SrcBB : BlockOrder) {
    const size_t CallCount = SrcBB->getKnownExecutionCount();
    // If SrcBB does not call any functions, skip it.
    if (CallCount == 0)
      continue;

    // Obtain an estimate on the end address of the src basic block
    // after splitting at SplitIndex.
    size_t SrcBBEndAddr = SrcBB->getOutputAddressRange().second;

    for (const BinaryBasicBlock *DstBB : Callees[GlobalIndices[SrcBB]]) {
      // Obtain an estimate on the start address of the dst basic block
      // after splitting at SplitIndex. If DstBB is in a function before
      // the current function, then its start address remains unchanged.
      size_t DstBBStartAddr = BBOffsets[DstBB];
      // If DstBB is in a function after the current function, then its
      // start address should be adjusted based on the reduction in hot size.
      if (GlobalIndices[DstBB] > LastGlobalIndex) {
        assert(DstBBStartAddr >= Score.HotSizeReduction);
        DstBBStartAddr -= Score.HotSizeReduction;
      }
      size_t NewCallLength = AbsoluteDifference(SrcBBEndAddr, DstBBStartAddr);
      Score.LocalScore += ::computeCallScore(CallCount, NewCallLength);
    }
  }
}

/// Compute sum of splitting scores for cover calls of the input function.
/// Increament Score.CoverCallScore in place by the sum.
void CDSplit::computeCoverCallScore(const BasicBlockOrder &BlockOrder,
                                    const size_t SplitIndex,
                                    const std::vector<CallInfo> &CoverCalls,
                                    SplitScore &Score) {
  if (opts::CallScale == 0)
    return;

  for (const CallInfo CI : CoverCalls) {
    assert(CI.Length >= Score.HotSizeReduction &&
           "Length of cover calls must exceed reduced size of hot fragment.");
    // Compute the new length of the call, which is shorter than the original
    // one by the size of the splitted fragment minus the total size increase.
    size_t NewCallLength = CI.Length - Score.HotSizeReduction;
    Score.CoverCallScore += ::computeCallScore(CI.Count, NewCallLength);
  }
}

/// Compute the split score of splitting a function at a given index.
/// The split score consists of local score and cover score. Cover call score is
/// expensive to compute. As a result, we pass in a \p ReferenceScore and
/// compute cover score only when the local score exceeds that in the
/// ReferenceScore or that the size reduction of the hot fragment is larger than
/// that achieved by the split index of the ReferenceScore. This function
/// returns \p Score of SplitScore type. It contains the local score and cover
/// score (if computed) of the current splitting index. For easier book keeping
/// and comparison, it also stores the split index and the resulting reduction
/// in hot fragment size.
SplitScore CDSplit::computeSplitScore(const BinaryFunction &BF,
                                      const BasicBlockOrder &BlockOrder,
                                      const size_t SplitIndex,
                                      const std::vector<CallInfo> &CoverCalls,
                                      const SplitScore &ReferenceScore) {
  // Populate BinaryBasicBlock::OutputAddressRange with estimated
  // new start and end addresses after hot-warm splitting at SplitIndex.
  size_t OldHotEnd;
  size_t NewHotEnd;
  std::tie(OldHotEnd, NewHotEnd) =
      estimatePostSplitBBAddress(BlockOrder, SplitIndex);

  SplitScore Score;
  Score.SplitIndex = SplitIndex;

  // It's not worth splitting if OldHotEnd < NewHotEnd.
  if (OldHotEnd < NewHotEnd)
    return Score;

  // Hot fragment size reduction due to splitting.
  Score.HotSizeReduction = OldHotEnd - NewHotEnd;

  // First part of LocalScore is the sum over call edges originated in the input
  // function. These edges can get shorter or longer depending on SplitIndex.
  // Score.LocalScore is increamented in place.
  computeLocalCallScore(BlockOrder, SplitIndex, Score);

  // Second part of LocalScore is the sum over jump edges with src basic block
  // and dst basic block in the current function. Score.LocalScore is
  // increamented in place.
  computeJumpScore(BlockOrder, SplitIndex, Score);

  // There is no need to compute CoverCallScore if we have already found another
  // split index with a bigger LocalScore and bigger HotSizeReduction.
  if (Score.LocalScore <= ReferenceScore.LocalScore &&
      Score.HotSizeReduction <= ReferenceScore.HotSizeReduction)
    return Score;

  // Compute CoverCallScore and store in Score in place.
  computeCoverCallScore(BlockOrder, SplitIndex, CoverCalls, Score);
  return Score;
}

/// Find the best index for splitting. The returned value is the index of the
/// last hot basic block. Hence, "no splitting" is equivalent to returning the
/// value which is one less than the size of the function.
size_t CDSplit::findSplitIndex(const BinaryFunction &BF,
                               const BasicBlockOrder &BlockOrder) {
  // Find all function calls that can be shortened if we move blocks of the
  // current function to warm/cold
  std::vector<CallInfo> CoverCalls = extractCoverCalls(BF);

  // Try all possible split indices (blocks with Index <= SplitIndex are in hot)
  // and find the one maximizing the splitting score.
  SplitScore BestScore;
  double BestScoreSum = -1.0;
  SplitScore ReferenceScore;
  for (size_t Index = 0; Index < BlockOrder.size(); Index++) {
    const BinaryBasicBlock *LastHotBB = BlockOrder[Index];
    // No need to keep cold blocks in the hot section.
    if (LastHotBB->isSplit())
      break;
    SplitScore Score =
        computeSplitScore(BF, BlockOrder, Index, CoverCalls, ReferenceScore);
    double ScoreSum = Score.LocalScore + Score.CoverCallScore;
    if (ScoreSum > BestScoreSum) {
      BestScoreSum = ScoreSum;
      BestScore = Score;
    }
    if (Score.LocalScore > ReferenceScore.LocalScore)
      ReferenceScore = Score;
  }

  return BestScore.SplitIndex;
}

/// Assign each basic block in the given function to either hot, cold,
/// or warm fragment using the CDSplit algorithm.
void CDSplit::assignFragmentThreeWay(const BinaryFunction &BF,
                                     const BasicBlockOrder &BlockOrder) {
  size_t BestSplitIndex = findSplitIndex(BF, BlockOrder);

  // Assign fragments based on the computed best split index.
  // All basic blocks with index up to the best split index become hot.
  // All remaining blocks are warm / cold depending on if count is
  // greater than 0 or not.
  FragmentNum Main(0);
  FragmentNum Warm(1);
  FragmentNum Cold(2);
  for (size_t Index = 0; Index < BlockOrder.size(); Index++) {
    BinaryBasicBlock *BB = BlockOrder[Index];
    if (Index <= BestSplitIndex)
      BB->setFragmentNum(Main);
    else
      BB->setFragmentNum(BB->getKnownExecutionCount() > 0 ? Warm : Cold);
  }
}

void CDSplit::runOnFunction(BinaryFunction &BF) {
  assert(!BF.empty() && "splitting an empty function");

  FunctionLayout &Layout = BF.getLayout();
  BinaryContext &BC = BF.getBinaryContext();

  BasicBlockOrder NewLayout(Layout.block_begin(), Layout.block_end());
  // Never outline the first basic block.
  NewLayout.front()->setCanOutline(false);
  for (BinaryBasicBlock *BB : NewLayout) {
    if (!BB->canOutline())
      continue;

    // Do not split extra entry points in aarch64. They can be referred by
    // using ADRs and when this happens, these blocks cannot be placed far
    // away due to the limited range in ADR instruction.
    if (BC.isAArch64() && BB->isEntryPoint()) {
      BB->setCanOutline(false);
      continue;
    }

    if (BF.hasEHRanges() && !opts::SplitEH) {
      // We cannot move landing pads (or rather entry points for landing pads).
      if (BB->isLandingPad()) {
        BB->setCanOutline(false);
        continue;
      }
      // We cannot move a block that can throw since exception-handling
      // runtime cannot deal with split functions. However, if we can guarantee
      // that the block never throws, it is safe to move the block to
      // decrease the size of the function.
      for (MCInst &Instr : *BB) {
        if (BC.MIB->isInvoke(Instr)) {
          BB->setCanOutline(false);
          break;
        }
      }
    }
  }

  // Assign each basic block in NewLayout to either hot, warm, or cold fragment.
  assignFragmentThreeWay(BF, NewLayout);

  // Make sure all non-outlineable blocks are in the main-fragment.
  for (BinaryBasicBlock *BB : NewLayout) {
    if (!BB->canOutline())
      BB->setFragmentNum(FragmentNum::main());
  }

  // In case any non-outlineable blocks previously in warm or cold is now set
  // to be in main by the preceding for loop, move them to the end of main.
  llvm::stable_sort(NewLayout,
                    [&](const BinaryBasicBlock *L, const BinaryBasicBlock *R) {
                      return L->getFragmentNum() < R->getFragmentNum();
                    });

  BF.getLayout().update(NewLayout);

  // For shared objects, invoke instructions and corresponding landing pads
  // have to be placed in the same fragment. When we split them, create
  // trampoline landing pads that will redirect the execution to real LPs.
  SplitFunctions::TrampolineSetType Trampolines;
  if (!BC.HasFixedLoadAddress && BF.hasEHRanges() && BF.isSplit())
    Trampolines = SplitFunctions::createEHTrampolines(BF);

  if (BC.isX86() && BF.isSplit()) {
    size_t HotSize;
    size_t ColdSize;
    std::tie(HotSize, ColdSize) = BC.calculateEmittedSize(BF);
    SplitBytesHot += HotSize;
    SplitBytesCold += ColdSize;
  }
}

void CDSplit::runOnFunctions(BinaryContext &BC) {
  if (!opts::UseCDSplit)
    return;

  // Initialize global variables.
  initialize(BC);

  // Only functions satisfying shouldConsider and shouldOptimize are candidates
  // for splitting.
  ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
    return !(shouldConsider(BF) && shouldOptimize(BF));
  };

  // Make function splitting decisions in parallel.
  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_BB_LINEAR,
      [&](BinaryFunction &BF) { runOnFunction(BF); }, SkipFunc, "CDSplit",
      /*ForceSequential=*/false);

  if (SplitBytesHot + SplitBytesCold > 0) {
    outs() << "BOLT-INFO: cdsplit separates " << SplitBytesHot
           << " hot bytes from " << SplitBytesCold << " cold bytes "
           << format("(%.2lf%% of split functions is in the main fragment)\n",
                     100.0 * SplitBytesHot / (SplitBytesHot + SplitBytesCold));

  } else
    outs() << "BOLT-INFO: cdsplit didn't split any functions\n";
}

} // namespace bolt
} // namespace llvm
