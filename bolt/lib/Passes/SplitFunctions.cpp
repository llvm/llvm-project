//===- bolt/Passes/SplitFunctions.cpp - Pass for splitting function code --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SplitFunctions pass.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/SplitFunctions.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#define DEBUG_TYPE "bolt-opts"

using namespace llvm;
using namespace bolt;

namespace {
class DeprecatedSplitFunctionOptionParser : public cl::parser<bool> {
public:
  explicit DeprecatedSplitFunctionOptionParser(cl::Option &O)
      : cl::parser<bool>(O) {}

  bool parse(cl::Option &O, StringRef ArgName, StringRef Arg, bool &Value) {
    if (Arg == "2" || Arg == "3") {
      Value = true;
      errs() << formatv("BOLT-WARNING: specifying non-boolean value \"{0}\" "
                        "for option -{1} is deprecated\n",
                        Arg, ArgName);
      return false;
    }
    return cl::parser<bool>::parse(O, ArgName, Arg, Value);
  }
};
} // namespace

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> SplitEH;
extern cl::opt<unsigned> ExecutionCountThreshold;
extern cl::opt<uint32_t> RandomSeed;

static cl::opt<bool> AggressiveSplitting(
    "split-all-cold", cl::desc("outline as many cold basic blocks as possible"),
    cl::cat(BoltOptCategory));

static cl::opt<unsigned> SplitAlignThreshold(
    "split-align-threshold",
    cl::desc("when deciding to split a function, apply this alignment "
             "while doing the size comparison (see -split-threshold). "
             "Default value: 2."),
    cl::init(2),

    cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool, false, DeprecatedSplitFunctionOptionParser>
    SplitFunctions("split-functions",
                   cl::desc("split functions into fragments"),
                   cl::cat(BoltOptCategory));

static cl::opt<unsigned> SplitThreshold(
    "split-threshold",
    cl::desc("split function only if its main size is reduced by more than "
             "given amount of bytes. Default value: 0, i.e. split iff the "
             "size is reduced. Note that on some architectures the size can "
             "increase after splitting."),
    cl::init(0), cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<double> CallScale(
    "call-scale",
    cl::desc("Call score scale coefficient (when --split-strategy=cdsplit)"),
    cl::init(0.95), cl::ReallyHidden, cl::cat(BoltOptCategory));

static cl::opt<double>
    CallPower("call-power",
              cl::desc("Call score power (when --split-strategy=cdsplit)"),
              cl::init(0.05), cl::ReallyHidden, cl::cat(BoltOptCategory));

static cl::opt<double>
    JumpPower("jump-power",
              cl::desc("Jump score power (when --split-strategy=cdsplit)"),
              cl::init(0.15), cl::ReallyHidden, cl::cat(BoltOptCategory));
} // namespace opts

namespace {
bool hasFullProfile(const BinaryFunction &BF) {
  return llvm::all_of(BF.blocks(), [](const BinaryBasicBlock &BB) {
    return BB.getExecutionCount() != BinaryBasicBlock::COUNT_NO_PROFILE;
  });
}

bool allBlocksCold(const BinaryFunction &BF) {
  return llvm::all_of(BF.blocks(), [](const BinaryBasicBlock &BB) {
    return BB.getExecutionCount() == 0;
  });
}

struct SplitProfile2 final : public SplitStrategy {
  bool canSplit(const BinaryFunction &BF) override {
    return BF.hasValidProfile() && hasFullProfile(BF) && !allBlocksCold(BF);
  }

  bool compactFragments() override { return true; }

  void fragment(const BlockIt Start, const BlockIt End) override {
    for (BinaryBasicBlock *const BB : llvm::make_range(Start, End)) {
      if (BB->getExecutionCount() == 0)
        BB->setFragmentNum(FragmentNum::cold());
    }
  }
};

struct SplitCacheDirected final : public SplitStrategy {
  BinaryContext &BC;
  using BasicBlockOrder = BinaryFunction::BasicBlockOrderType;

  bool canSplit(const BinaryFunction &BF) override {
    return BF.hasValidProfile() && hasFullProfile(BF) && !allBlocksCold(BF);
  }

  explicit SplitCacheDirected(BinaryContext &BC) : BC(BC) {
    initializeAuxiliaryVariables();
    buildCallGraph();
  }

  // When some functions are hot-warm split and others are hot-warm-cold split,
  // we do not want to change the fragment numbers of the blocks in the hot-warm
  // split functions.
  bool compactFragments() override { return false; }

  void fragment(const BlockIt Start, const BlockIt End) override {
    BasicBlockOrder BlockOrder(Start, End);
    BinaryFunction &BF = *BlockOrder.front()->getFunction();
    // No need to re-split small functions.
    if (BlockOrder.size() <= 2)
      return;

    size_t BestSplitIndex = findSplitIndex(BF, BlockOrder);
    assert(BestSplitIndex < BlockOrder.size());

    // Assign fragments based on the computed best split index.
    // All basic blocks with index up to the best split index become hot.
    // All remaining blocks are warm / cold depending on if count is
    // greater than zero or not.
    for (size_t Index = 0; Index < BlockOrder.size(); Index++) {
      BinaryBasicBlock *BB = BlockOrder[Index];
      if (Index <= BestSplitIndex)
        BB->setFragmentNum(FragmentNum::main());
      else
        BB->setFragmentNum(BB->getKnownExecutionCount() > 0
                               ? FragmentNum::warm()
                               : FragmentNum::cold());
    }
  }

private:
  struct CallInfo {
    size_t Length;
    size_t Count;
  };

  struct SplitScore {
    size_t SplitIndex = size_t(-1);
    size_t HotSizeReduction = 0;
    double LocalScore = 0;
    double CoverCallScore = 0;

    double sum() const { return LocalScore + CoverCallScore; }
  };

  // Auxiliary variables used by the algorithm.
  size_t TotalNumBlocks{0};
  size_t OrigHotSectionSize{0};
  DenseMap<const BinaryBasicBlock *, size_t> GlobalIndices;
  DenseMap<const BinaryBasicBlock *, size_t> BBSizes;
  DenseMap<const BinaryBasicBlock *, size_t> BBOffsets;

  // Call graph.
  std::vector<SmallVector<const BinaryBasicBlock *, 0>> Callers;
  std::vector<SmallVector<const BinaryBasicBlock *, 0>> Callees;

  bool shouldConsiderForCallGraph(const BinaryFunction &BF) {
    // Only a subset of the functions in the binary will be considered
    // for initializing auxiliary variables and building call graph.
    return BF.hasValidIndex() && BF.hasValidProfile() && !BF.empty();
  }

  void initializeAuxiliaryVariables() {
    for (BinaryFunction *BF : BC.getSortedFunctions()) {
      if (!shouldConsiderForCallGraph(*BF))
        continue;

      // Calculate the size of each BB after hot-cold splitting.
      // This populates BinaryBasicBlock::OutputAddressRange which
      // can be used to compute the size of each BB.
      BC.calculateEmittedSize(*BF, /*FixBranches=*/true);

      for (BinaryBasicBlock *BB : BF->getLayout().blocks()) {
        // Unique global index.
        GlobalIndices[BB] = TotalNumBlocks;
        TotalNumBlocks++;

        // Block size after hot-cold splitting.
        BBSizes[BB] = BB->getOutputSize();

        // Hot block offset after hot-cold splitting.
        BBOffsets[BB] = OrigHotSectionSize;
        if (!BB->isSplit())
          OrigHotSectionSize += BBSizes[BB];
      }
    }
  }

  void buildCallGraph() {
    Callers.resize(TotalNumBlocks);
    Callees.resize(TotalNumBlocks);
    for (const BinaryFunction *SrcFunction : BC.getSortedFunctions()) {
      if (!shouldConsiderForCallGraph(*SrcFunction))
        continue;

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

          // Record the call only if DstBB is also in functions to consider for
          // call graph.
          if (GlobalIndices.contains(DstBB)) {
            Callers[GlobalIndices[DstBB]].push_back(&SrcBB);
            Callees[GlobalIndices[&SrcBB]].push_back(DstBB);
          }
        }
      }
    }
  }

  /// Populate BinaryBasicBlock::OutputAddressRange with estimated basic block
  /// start and end addresses for hot and warm basic blocks, assuming hot-warm
  /// splitting happens at \p SplitIndex. Also return estimated end addresses
  /// of the hot fragment before and after splitting.
  /// The estimations take into account the potential addition of branch
  /// instructions due to split fall through branches as well as the need to
  /// use longer branch instructions for split (un)conditional branches.
  std::pair<size_t, size_t>
  estimatePostSplitBBAddress(const BasicBlockOrder &BlockOrder,
                             const size_t SplitIndex) {
    assert(SplitIndex < BlockOrder.size() && "Invalid split index");

    // Update function layout assuming hot-warm splitting at SplitIndex.
    for (size_t Index = 0; Index < BlockOrder.size(); Index++) {
      BinaryBasicBlock *BB = BlockOrder[Index];
      if (BB->getFragmentNum() == FragmentNum::cold())
        break;
      BB->setFragmentNum(Index <= SplitIndex ? FragmentNum::main()
                                             : FragmentNum::warm());
    }
    BinaryFunction *BF = BlockOrder[0]->getFunction();
    BF->getLayout().update(BlockOrder);
    // Populate BB.OutputAddressRange under the updated layout.
    BC.calculateEmittedSize(*BF);

    // Populate BB.OutputAddressRange with estimated new start and end addresses
    // and compute the old end address of the hot section and the new end
    // address of the hot section.
    size_t OldHotEndAddr{0};
    size_t NewHotEndAddr{0};
    size_t CurrentAddr = BBOffsets[BlockOrder[0]];
    for (BinaryBasicBlock *BB : BlockOrder) {
      // We only care about new addresses of blocks in hot/warm.
      if (BB->getFragmentNum() == FragmentNum::cold())
        break;
      const size_t NewSize = BB->getOutputSize();
      BB->setOutputStartAddress(CurrentAddr);
      CurrentAddr += NewSize;
      BB->setOutputEndAddress(CurrentAddr);
      if (BB->getLayoutIndex() == SplitIndex) {
        NewHotEndAddr = CurrentAddr;
        // Approximate the start address of the warm fragment of the current
        // function using the original hot section size.
        CurrentAddr = OrigHotSectionSize;
      }
      OldHotEndAddr = BBOffsets[BB] + BBSizes[BB];
    }
    return std::make_pair(OldHotEndAddr, NewHotEndAddr);
  }

  /// Get a collection of "shortenable" calls, that is, calls of type X->Y
  /// when the function order is [... X ... BF ... Y ...].
  /// If the hot fragment size of BF is reduced, then such calls are guaranteed
  /// to get shorter by the reduced hot fragment size.
  std::vector<CallInfo> extractCoverCalls(const BinaryFunction &BF) {
    // Record the length and the count of the calls that can be shortened
    std::vector<CallInfo> CoverCalls;
    if (opts::CallScale == 0)
      return CoverCalls;

    const BinaryFunction *ThisBF = &BF;
    const BinaryBasicBlock *ThisBB = &(ThisBF->front());
    const size_t ThisGI = GlobalIndices[ThisBB];

    for (const BinaryFunction *DstBF : BC.getSortedFunctions()) {
      if (!shouldConsiderForCallGraph(*DstBF))
        continue;

      const BinaryBasicBlock *DstBB = &(DstBF->front());
      if (DstBB->getKnownExecutionCount() == 0)
        continue;

      const size_t DstGI = GlobalIndices[DstBB];
      for (const BinaryBasicBlock *SrcBB : Callers[DstGI]) {
        const BinaryFunction *SrcBF = SrcBB->getFunction();
        if (ThisBF == SrcBF)
          continue;

        const size_t CallCount = SrcBB->getKnownExecutionCount();

        const size_t SrcGI = GlobalIndices[SrcBB];

        const bool IsCoverCall = (SrcGI < ThisGI && ThisGI < DstGI) ||
                                 (DstGI <= ThisGI && ThisGI < SrcGI);
        if (!IsCoverCall)
          continue;

        const size_t SrcBBEndAddr = BBOffsets[SrcBB] + BBSizes[SrcBB];
        const size_t DstBBStartAddr = BBOffsets[DstBB];
        const size_t CallLength =
            AbsoluteDifference(SrcBBEndAddr, DstBBStartAddr);
        const CallInfo CI{CallLength, CallCount};
        CoverCalls.emplace_back(CI);
      }
    }
    return CoverCalls;
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

  /// Compute sum of scores over jumps within \p BlockOrder given \p SplitIndex.
  /// Increament Score.LocalScore in place by the sum.
  void computeJumpScore(const BasicBlockOrder &BlockOrder,
                        const size_t SplitIndex, SplitScore &Score) {

    for (const BinaryBasicBlock *SrcBB : BlockOrder) {
      if (SrcBB->getKnownExecutionCount() == 0)
        continue;

      const size_t SrcBBEndAddr = SrcBB->getOutputAddressRange().second;

      for (const auto Pair : zip(SrcBB->successors(), SrcBB->branch_info())) {
        const BinaryBasicBlock *DstBB = std::get<0>(Pair);
        const BinaryBasicBlock::BinaryBranchInfo &Branch = std::get<1>(Pair);
        const size_t JumpCount = Branch.Count;

        if (JumpCount == 0)
          continue;

        const size_t DstBBStartAddr = DstBB->getOutputAddressRange().first;
        const size_t NewJumpLength =
            AbsoluteDifference(SrcBBEndAddr, DstBBStartAddr);
        Score.LocalScore += computeJumpScore(JumpCount, NewJumpLength);
      }
    }
  }

  /// Compute sum of scores over calls originated in the current function
  /// given \p SplitIndex. Increament Score.LocalScore in place by the sum.
  void computeLocalCallScore(const BasicBlockOrder &BlockOrder,
                             const size_t SplitIndex, SplitScore &Score) {
    if (opts::CallScale == 0)
      return;

    // Global index of the last block in the current function.
    // This is later used to determine whether a call originated in the current
    // function is to a function that comes after the current function.
    const size_t LastGlobalIndex = GlobalIndices[BlockOrder.back()];

    // The length of calls originated in the input function can increase /
    // decrease depending on the splitting decision.
    for (const BinaryBasicBlock *SrcBB : BlockOrder) {
      const size_t CallCount = SrcBB->getKnownExecutionCount();
      // If SrcBB does not call any functions, skip it.
      if (CallCount == 0)
        continue;

      // Obtain an estimate on the end address of the src basic block
      // after splitting at SplitIndex.
      const size_t SrcBBEndAddr = SrcBB->getOutputAddressRange().second;

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
        const size_t NewCallLength =
            AbsoluteDifference(SrcBBEndAddr, DstBBStartAddr);
        Score.LocalScore += computeCallScore(CallCount, NewCallLength);
      }
    }
  }

  /// Compute sum of splitting scores for cover calls of the input function.
  /// Increament Score.CoverCallScore in place by the sum.
  void computeCoverCallScore(const BasicBlockOrder &BlockOrder,
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
      const size_t NewCallLength = CI.Length - Score.HotSizeReduction;
      Score.CoverCallScore += computeCallScore(CI.Count, NewCallLength);
    }
  }

  /// Compute the split score of splitting a function at a given index.
  /// The split score consists of local score and cover score. This function
  /// returns \p Score of SplitScore type. It contains the local score and
  /// cover score of the current splitting index. For easier book keeping and
  /// comparison, it also stores the split index and the resulting reduction
  /// in hot fragment size.
  SplitScore computeSplitScore(const BinaryFunction &BF,
                               const BasicBlockOrder &BlockOrder,
                               const size_t SplitIndex,
                               const std::vector<CallInfo> &CoverCalls) {
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

    // First part of LocalScore is the sum over call edges originated in the
    // input function. These edges can get shorter or longer depending on
    // SplitIndex. Score.LocalScore is increamented in place.
    computeLocalCallScore(BlockOrder, SplitIndex, Score);

    // Second part of LocalScore is the sum over jump edges with src basic block
    // and dst basic block in the current function. Score.LocalScore is
    // increamented in place.
    computeJumpScore(BlockOrder, SplitIndex, Score);

    // Compute CoverCallScore and store in Score in place.
    computeCoverCallScore(BlockOrder, SplitIndex, CoverCalls, Score);
    return Score;
  }

  /// Find the most likely successor of a basic block when it has one or two
  /// successors. Return nullptr otherwise.
  const BinaryBasicBlock *getMostLikelySuccessor(const BinaryBasicBlock *BB) {
    if (BB->succ_size() == 1)
      return BB->getSuccessor();
    if (BB->succ_size() == 2) {
      uint64_t TakenCount = BB->getTakenBranchInfo().Count;
      assert(TakenCount != BinaryBasicBlock::COUNT_NO_PROFILE);
      uint64_t NonTakenCount = BB->getFallthroughBranchInfo().Count;
      assert(NonTakenCount != BinaryBasicBlock::COUNT_NO_PROFILE);
      if (TakenCount > NonTakenCount)
        return BB->getConditionalSuccessor(true);
      else if (TakenCount < NonTakenCount)
        return BB->getConditionalSuccessor(false);
    }
    return nullptr;
  }

  /// Find the best index for splitting. The returned value is the index of the
  /// last hot basic block. Hence, "no splitting" is equivalent to returning the
  /// value which is one less than the size of the function.
  size_t findSplitIndex(const BinaryFunction &BF,
                        const BasicBlockOrder &BlockOrder) {
    assert(BlockOrder.size() > 2);
    // Find all function calls that can be shortened if we move blocks of the
    // current function to warm/cold
    const std::vector<CallInfo> CoverCalls = extractCoverCalls(BF);

    // Find the existing hot-cold splitting index.
    size_t HotColdIndex = 0;
    while (HotColdIndex + 1 < BlockOrder.size()) {
      if (BlockOrder[HotColdIndex + 1]->getFragmentNum() == FragmentNum::cold())
        break;
      HotColdIndex++;
    }
    assert(HotColdIndex + 1 == BlockOrder.size() ||
           (BlockOrder[HotColdIndex]->getFragmentNum() == FragmentNum::main() &&
            BlockOrder[HotColdIndex + 1]->getFragmentNum() ==
                FragmentNum::cold()));

    // Try all possible split indices up to HotColdIndex (blocks that have
    // Index <= SplitIndex are in hot) and find the one maximizing the
    // splitting score.
    SplitScore BestScore;
    for (size_t Index = 0; Index <= HotColdIndex; Index++) {
      const BinaryBasicBlock *LastHotBB = BlockOrder[Index];
      assert(LastHotBB->getFragmentNum() != FragmentNum::cold());

      // Do not break jump to the most likely successor.
      if (Index + 1 < BlockOrder.size() &&
          BlockOrder[Index + 1] == getMostLikelySuccessor(LastHotBB))
        continue;

      const SplitScore Score =
          computeSplitScore(BF, BlockOrder, Index, CoverCalls);
      if (Score.sum() > BestScore.sum())
        BestScore = Score;
    }

    // If we don't find a good splitting point, fallback to the original one.
    if (BestScore.SplitIndex == size_t(-1))
      return HotColdIndex;

    return BestScore.SplitIndex;
  }
};

struct SplitRandom2 final : public SplitStrategy {
  std::minstd_rand0 Gen;

  SplitRandom2() : Gen(opts::RandomSeed.getValue()) {}

  bool canSplit(const BinaryFunction &BF) override { return true; }

  bool compactFragments() override { return true; }

  void fragment(const BlockIt Start, const BlockIt End) override {
    using DiffT = typename std::iterator_traits<BlockIt>::difference_type;
    const DiffT NumBlocks = End - Start;
    assert(NumBlocks > 0 && "Cannot fragment empty function");

    // We want to split at least one block
    const auto LastSplitPoint = std::max<DiffT>(NumBlocks - 1, 1);
    std::uniform_int_distribution<DiffT> Dist(1, LastSplitPoint);
    const DiffT SplitPoint = Dist(Gen);
    for (BinaryBasicBlock *BB : llvm::make_range(Start + SplitPoint, End))
      BB->setFragmentNum(FragmentNum::cold());

    LLVM_DEBUG(dbgs() << formatv("BOLT-DEBUG: randomly chose last {0} (out of "
                                 "{1} possible) blocks to split\n",
                                 NumBlocks - SplitPoint, End - Start));
  }
};

struct SplitRandomN final : public SplitStrategy {
  std::minstd_rand0 Gen;

  SplitRandomN() : Gen(opts::RandomSeed.getValue()) {}

  bool canSplit(const BinaryFunction &BF) override { return true; }

  bool compactFragments() override { return true; }

  void fragment(const BlockIt Start, const BlockIt End) override {
    using DiffT = typename std::iterator_traits<BlockIt>::difference_type;
    const DiffT NumBlocks = End - Start;
    assert(NumBlocks > 0 && "Cannot fragment empty function");

    // With n blocks, there are n-1 places to split them.
    const DiffT MaximumSplits = NumBlocks - 1;
    // We want to generate at least two fragment if possible, but if there is
    // only one block, no splits are possible.
    const auto MinimumSplits = std::min<DiffT>(MaximumSplits, 1);
    std::uniform_int_distribution<DiffT> Dist(MinimumSplits, MaximumSplits);
    // Choose how many splits to perform
    const DiffT NumSplits = Dist(Gen);

    // Draw split points from a lottery
    SmallVector<unsigned, 0> Lottery(MaximumSplits);
    // Start lottery at 1, because there is no meaningful splitpoint before the
    // first block.
    std::iota(Lottery.begin(), Lottery.end(), 1u);
    std::shuffle(Lottery.begin(), Lottery.end(), Gen);
    Lottery.resize(NumSplits);
    llvm::sort(Lottery);

    // Add one past the end entry to lottery
    Lottery.push_back(NumBlocks);

    unsigned LotteryIndex = 0;
    unsigned BBPos = 0;
    for (BinaryBasicBlock *const BB : make_range(Start, End)) {
      // Check whether to start new fragment
      if (BBPos >= Lottery[LotteryIndex])
        ++LotteryIndex;

      // Because LotteryIndex is 0 based and cold fragments are 1 based, we can
      // use the index to assign fragments.
      BB->setFragmentNum(FragmentNum(LotteryIndex));

      ++BBPos;
    }
  }
};

struct SplitAll final : public SplitStrategy {
  bool canSplit(const BinaryFunction &BF) override { return true; }

  bool compactFragments() override {
    // Keeping empty fragments allows us to test, that empty fragments do not
    // generate symbols.
    return false;
  }

  void fragment(const BlockIt Start, const BlockIt End) override {
    unsigned Fragment = 0;
    for (BinaryBasicBlock *const BB : llvm::make_range(Start, End))
      BB->setFragmentNum(FragmentNum(Fragment++));
  }
};
} // namespace

namespace llvm {
namespace bolt {

bool SplitFunctions::shouldOptimize(const BinaryFunction &BF) const {
  // Apply execution count threshold
  if (BF.getKnownExecutionCount() < opts::ExecutionCountThreshold)
    return false;

  return BinaryFunctionPass::shouldOptimize(BF);
}

Error SplitFunctions::runOnFunctions(BinaryContext &BC) {
  if (!opts::SplitFunctions)
    return Error::success();

  if (BC.IsLinuxKernel && BC.BOLTReserved.empty()) {
    BC.errs() << "BOLT-ERROR: split functions require reserved space in the "
                 "Linux kernel binary\n";
    exit(1);
  }

  // If split strategy is not CDSplit, then a second run of the pass is not
  // needed after function reordering.
  if (BC.HasFinalizedFunctionOrder &&
      opts::SplitStrategy != opts::SplitFunctionsStrategy::CDSplit)
    return Error::success();

  std::unique_ptr<SplitStrategy> Strategy;
  bool ForceSequential = false;

  switch (opts::SplitStrategy) {
  case opts::SplitFunctionsStrategy::CDSplit:
    // CDSplit runs two splitting passes: hot-cold splitting (SplitPrfoile2)
    // before function reordering and hot-warm-cold splitting
    // (SplitCacheDirected) after function reordering.
    if (BC.HasFinalizedFunctionOrder)
      Strategy = std::make_unique<SplitCacheDirected>(BC);
    else
      Strategy = std::make_unique<SplitProfile2>();
    opts::AggressiveSplitting = true;
    BC.HasWarmSection = true;
    break;
  case opts::SplitFunctionsStrategy::Profile2:
    Strategy = std::make_unique<SplitProfile2>();
    break;
  case opts::SplitFunctionsStrategy::Random2:
    Strategy = std::make_unique<SplitRandom2>();
    // If we split functions randomly, we need to ensure that across runs with
    // the same input, we generate random numbers for each function in the same
    // order.
    ForceSequential = true;
    break;
  case opts::SplitFunctionsStrategy::RandomN:
    Strategy = std::make_unique<SplitRandomN>();
    ForceSequential = true;
    break;
  case opts::SplitFunctionsStrategy::All:
    Strategy = std::make_unique<SplitAll>();
    break;
  }

  ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
    return !shouldOptimize(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_BB_LINEAR,
      [&](BinaryFunction &BF) { splitFunction(BF, *Strategy); }, SkipFunc,
      "SplitFunctions", ForceSequential);

  if (SplitBytesHot + SplitBytesCold > 0)
    BC.outs() << "BOLT-INFO: splitting separates " << SplitBytesHot
              << " hot bytes from " << SplitBytesCold << " cold bytes "
              << format("(%.2lf%% of split functions is hot).\n",
                        100.0 * SplitBytesHot /
                            (SplitBytesHot + SplitBytesCold));
  return Error::success();
}

void SplitFunctions::splitFunction(BinaryFunction &BF, SplitStrategy &S) {
  if (BF.empty())
    return;

  if (!S.canSplit(BF))
    return;

  FunctionLayout &Layout = BF.getLayout();
  BinaryFunction::BasicBlockOrderType PreSplitLayout(Layout.block_begin(),
                                                     Layout.block_end());

  BinaryContext &BC = BF.getBinaryContext();
  size_t OriginalHotSize;
  size_t HotSize;
  size_t ColdSize;
  if (BC.isX86()) {
    std::tie(OriginalHotSize, ColdSize) = BC.calculateEmittedSize(BF);
    LLVM_DEBUG(dbgs() << "Estimated size for function " << BF
                      << " pre-split is <0x"
                      << Twine::utohexstr(OriginalHotSize) << ", 0x"
                      << Twine::utohexstr(ColdSize) << ">\n");
  }

  BinaryFunction::BasicBlockOrderType NewLayout(Layout.block_begin(),
                                                Layout.block_end());
  // Never outline the first basic block.
  NewLayout.front()->setCanOutline(false);
  for (BinaryBasicBlock *const BB : NewLayout) {
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

    // Outlining blocks with dynamic branches is not supported yet.
    if (BC.IsLinuxKernel) {
      if (llvm::any_of(
              *BB, [&](MCInst &Inst) { return BC.MIB->isDynamicBranch(Inst); }))
        BB->setCanOutline(false);
    }
  }

  BF.getLayout().updateLayoutIndices();
  S.fragment(NewLayout.begin(), NewLayout.end());

  // Make sure all non-outlineable blocks are in the main-fragment.
  for (BinaryBasicBlock *const BB : NewLayout) {
    if (!BB->canOutline())
      BB->setFragmentNum(FragmentNum::main());
  }

  if (opts::AggressiveSplitting) {
    // All blocks with 0 count that we can move go to the end of the function.
    // Even if they were natural to cluster formation and were seen in-between
    // hot basic blocks.
    llvm::stable_sort(NewLayout, [&](const BinaryBasicBlock *const A,
                                     const BinaryBasicBlock *const B) {
      return A->getFragmentNum() < B->getFragmentNum();
    });
  } else if (BF.hasEHRanges() && !opts::SplitEH) {
    // Typically functions with exception handling have landing pads at the end.
    // We cannot move beginning of landing pads, but we can move 0-count blocks
    // comprising landing pads to the end and thus facilitate splitting.
    auto FirstLP = NewLayout.begin();
    while ((*FirstLP)->isLandingPad())
      ++FirstLP;

    std::stable_sort(FirstLP, NewLayout.end(),
                     [&](BinaryBasicBlock *A, BinaryBasicBlock *B) {
                       return A->getFragmentNum() < B->getFragmentNum();
                     });
  }

  // Make sure that fragments are increasing.
  FragmentNum CurrentFragment = NewLayout.back()->getFragmentNum();
  for (BinaryBasicBlock *const BB : reverse(NewLayout)) {
    if (BB->getFragmentNum() > CurrentFragment)
      BB->setFragmentNum(CurrentFragment);
    CurrentFragment = BB->getFragmentNum();
  }

  if (S.compactFragments()) {
    FragmentNum CurrentFragment = FragmentNum::main();
    FragmentNum NewFragment = FragmentNum::main();
    for (BinaryBasicBlock *const BB : NewLayout) {
      if (BB->getFragmentNum() > CurrentFragment) {
        CurrentFragment = BB->getFragmentNum();
        NewFragment = FragmentNum(NewFragment.get() + 1);
      }
      BB->setFragmentNum(NewFragment);
    }
  }

  const bool LayoutUpdated = BF.getLayout().update(NewLayout);

  // For shared objects, invoke instructions and corresponding landing pads
  // have to be placed in the same fragment. When we split them, create
  // trampoline landing pads that will redirect the execution to real LPs.
  TrampolineSetType Trampolines;
  if (BF.hasEHRanges() && BF.isSplit()) {
    // If all landing pads for this fragment are grouped in one (potentially
    // different) fragment, we can set LPStart to the start of that fragment
    // and avoid trampoline code.
    bool NeedsTrampolines = false;
    for (FunctionFragment &FF : BF.getLayout().fragments()) {
      // Vector of fragments that contain landing pads for this fragment.
      SmallVector<FragmentNum, 4> LandingPadFragments;
      for (const BinaryBasicBlock *BB : FF)
        for (const BinaryBasicBlock *LPB : BB->landing_pads())
          LandingPadFragments.push_back(LPB->getFragmentNum());

      // Eliminate duplicate entries from the vector.
      llvm::sort(LandingPadFragments);
      auto Last = llvm::unique(LandingPadFragments);
      LandingPadFragments.erase(Last, LandingPadFragments.end());

      if (LandingPadFragments.size() == 0) {
        // If the fragment has no landing pads, we can safely set itself as its
        // landing pad fragment.
        BF.setLPFragment(FF.getFragmentNum(), FF.getFragmentNum());
      } else if (LandingPadFragments.size() == 1) {
        BF.setLPFragment(FF.getFragmentNum(), LandingPadFragments.front());
      } else {
        if (!BC.HasFixedLoadAddress) {
          NeedsTrampolines = true;
          break;
        } else {
          BF.setLPFragment(FF.getFragmentNum(), std::nullopt);
        }
      }
    }

    // Trampolines guarantee that all landing pads for any given fragment will
    // be contained in the same fragment.
    if (NeedsTrampolines) {
      for (FunctionFragment &FF : BF.getLayout().fragments())
        BF.setLPFragment(FF.getFragmentNum(), FF.getFragmentNum());
      Trampolines = createEHTrampolines(BF);
    }
  }

  // Check the new size to see if it's worth splitting the function.
  if (BC.isX86() && LayoutUpdated) {
    std::tie(HotSize, ColdSize) = BC.calculateEmittedSize(BF);
    LLVM_DEBUG(dbgs() << "Estimated size for function " << BF
                      << " post-split is <0x" << Twine::utohexstr(HotSize)
                      << ", 0x" << Twine::utohexstr(ColdSize) << ">\n");
    if (alignTo(OriginalHotSize, opts::SplitAlignThreshold) <=
        alignTo(HotSize, opts::SplitAlignThreshold) + opts::SplitThreshold) {
      if (opts::Verbosity >= 2) {
        BC.outs() << "BOLT-INFO: Reversing splitting of function "
                  << formatv("{0}:\n  {1:x}, {2:x} -> {3:x}\n", BF, HotSize,
                             ColdSize, OriginalHotSize);
      }

      // Reverse the action of createEHTrampolines(). The trampolines will be
      // placed immediately before the matching destination resulting in no
      // extra code.
      if (PreSplitLayout.size() != BF.size())
        PreSplitLayout = mergeEHTrampolines(BF, PreSplitLayout, Trampolines);

      for (BinaryBasicBlock &BB : BF)
        BB.setFragmentNum(FragmentNum::main());
      BF.getLayout().update(PreSplitLayout);
    } else {
      SplitBytesHot += HotSize;
      SplitBytesCold += ColdSize;
    }
  }

  // Restore LP fragment for the main fragment if the splitting was undone.
  if (BF.hasEHRanges() && !BF.isSplit())
    BF.setLPFragment(FragmentNum::main(), FragmentNum::main());

  // Fix branches if the splitting decision of the pass after function
  // reordering is different from that of the pass before function reordering.
  if (LayoutUpdated && BC.HasFinalizedFunctionOrder)
    BF.fixBranches();
}

SplitFunctions::TrampolineSetType
SplitFunctions::createEHTrampolines(BinaryFunction &BF) const {
  const auto &MIB = BF.getBinaryContext().MIB;

  // Map real landing pads to the corresponding trampolines.
  TrampolineSetType LPTrampolines;

  // Iterate over the copy of basic blocks since we are adding new blocks to the
  // function which will invalidate its iterators.
  std::vector<BinaryBasicBlock *> Blocks(BF.pbegin(), BF.pend());
  for (BinaryBasicBlock *BB : Blocks) {
    for (MCInst &Instr : *BB) {
      const std::optional<MCPlus::MCLandingPad> EHInfo = MIB->getEHInfo(Instr);
      if (!EHInfo || !EHInfo->first)
        continue;

      const MCSymbol *LPLabel = EHInfo->first;
      BinaryBasicBlock *LPBlock = BF.getBasicBlockForLabel(LPLabel);
      if (BB->getFragmentNum() == LPBlock->getFragmentNum())
        continue;

      const MCSymbol *TrampolineLabel = nullptr;
      const TrampolineKey Key(BB->getFragmentNum(), LPLabel);
      auto Iter = LPTrampolines.find(Key);
      if (Iter != LPTrampolines.end()) {
        TrampolineLabel = Iter->second;
      } else {
        // Create a trampoline basic block in the same fragment as the thrower.
        // Note: there's no need to insert the jump instruction, it will be
        // added by fixBranches().
        BinaryBasicBlock *TrampolineBB = BF.addBasicBlock();
        TrampolineBB->setFragmentNum(BB->getFragmentNum());
        TrampolineBB->setExecutionCount(LPBlock->getExecutionCount());
        TrampolineBB->addSuccessor(LPBlock, TrampolineBB->getExecutionCount());
        TrampolineBB->setCFIState(LPBlock->getCFIState());
        TrampolineLabel = TrampolineBB->getLabel();
        LPTrampolines.insert(std::make_pair(Key, TrampolineLabel));
      }

      // Substitute the landing pad with the trampoline.
      MIB->updateEHInfo(Instr,
                        MCPlus::MCLandingPad(TrampolineLabel, EHInfo->second));
    }
  }

  if (LPTrampolines.empty())
    return LPTrampolines;

  // All trampoline blocks were added to the end of the function. Place them at
  // the end of corresponding fragments.
  BinaryFunction::BasicBlockOrderType NewLayout(BF.getLayout().block_begin(),
                                                BF.getLayout().block_end());
  stable_sort(NewLayout, [&](BinaryBasicBlock *A, BinaryBasicBlock *B) {
    return A->getFragmentNum() < B->getFragmentNum();
  });
  BF.getLayout().update(NewLayout);

  // Conservatively introduce branch instructions.
  BF.fixBranches();

  // Update exception-handling CFG for the function.
  BF.recomputeLandingPads();

  return LPTrampolines;
}

SplitFunctions::BasicBlockOrderType SplitFunctions::mergeEHTrampolines(
    BinaryFunction &BF, SplitFunctions::BasicBlockOrderType &Layout,
    const SplitFunctions::TrampolineSetType &Trampolines) const {
  DenseMap<const MCSymbol *, SmallVector<const MCSymbol *, 0>>
      IncomingTrampolines;
  for (const auto &Entry : Trampolines) {
    IncomingTrampolines[Entry.getFirst().Target].emplace_back(
        Entry.getSecond());
  }

  BasicBlockOrderType MergedLayout;
  for (BinaryBasicBlock *BB : Layout) {
    auto Iter = IncomingTrampolines.find(BB->getLabel());
    if (Iter != IncomingTrampolines.end()) {
      for (const MCSymbol *const Trampoline : Iter->getSecond()) {
        BinaryBasicBlock *LPBlock = BF.getBasicBlockForLabel(Trampoline);
        assert(LPBlock && "Could not find matching landing pad block.");
        MergedLayout.push_back(LPBlock);
      }
    }
    MergedLayout.push_back(BB);
  }

  return MergedLayout;
}

} // namespace bolt
} // namespace llvm
