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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
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

static cl::opt<SplitFunctionsStrategy> SplitStrategy(
    "split-strategy", cl::init(SplitFunctionsStrategy::Profile2),
    cl::values(clEnumValN(SplitFunctionsStrategy::Profile2, "profile2",
                          "split each function into a hot and cold fragment "
                          "using profiling information")),
    cl::values(clEnumValN(
        SplitFunctionsStrategy::Random2, "random2",
        "split each function into a hot and cold fragment at a randomly chosen "
        "split point (ignoring any available profiling information)")),
    cl::values(clEnumValN(
        SplitFunctionsStrategy::RandomN, "randomN",
        "split each function into N fragments at a randomly chosen split "
        "points (ignoring any available profiling information)")),
    cl::values(clEnumValN(
        SplitFunctionsStrategy::All, "all",
        "split all basic blocks of each function into fragments such that each "
        "fragment contains exactly a single basic block")),
    cl::desc("strategy used to partition blocks into fragments"),
    cl::cat(BoltOptCategory));
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

  bool keepEmpty() override { return false; }

  void fragment(const BlockIt Start, const BlockIt End) override {
    for (BinaryBasicBlock *const BB : llvm::make_range(Start, End)) {
      if (BB->getExecutionCount() == 0)
        BB->setFragmentNum(FragmentNum::cold());
    }
  }
};

struct SplitRandom2 final : public SplitStrategy {
  std::minstd_rand0 Gen;

  SplitRandom2() : Gen(opts::RandomSeed.getValue()) {}

  bool canSplit(const BinaryFunction &BF) override { return true; }

  bool keepEmpty() override { return false; }

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

  bool keepEmpty() override { return false; }

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

  bool keepEmpty() override {
    // Keeping empty fragments allows us to test, that empty fragments do not
    // generate symbols.
    return true;
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

void SplitFunctions::runOnFunctions(BinaryContext &BC) {
  if (!opts::SplitFunctions)
    return;

  std::unique_ptr<SplitStrategy> Strategy;
  bool ForceSequential = false;

  switch (opts::SplitStrategy) {
  case SplitFunctionsStrategy::Profile2:
    Strategy = std::make_unique<SplitProfile2>();
    break;
  case SplitFunctionsStrategy::Random2:
    Strategy = std::make_unique<SplitRandom2>();
    // If we split functions randomly, we need to ensure that across runs with
    // the same input, we generate random numbers for each function in the same
    // order.
    ForceSequential = true;
    break;
  case SplitFunctionsStrategy::RandomN:
    Strategy = std::make_unique<SplitRandomN>();
    ForceSequential = true;
    break;
  case SplitFunctionsStrategy::All:
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
    outs() << "BOLT-INFO: splitting separates " << SplitBytesHot
           << " hot bytes from " << SplitBytesCold << " cold bytes "
           << format("(%.2lf%% of split functions is hot).\n",
                     100.0 * SplitBytesHot / (SplitBytesHot + SplitBytesCold));
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

  if (!S.keepEmpty()) {
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

  BF.getLayout().update(NewLayout);

  // For shared objects, invoke instructions and corresponding landing pads
  // have to be placed in the same fragment. When we split them, create
  // trampoline landing pads that will redirect the execution to real LPs.
  TrampolineSetType Trampolines;
  if (!BC.HasFixedLoadAddress && BF.hasEHRanges() && BF.isSplit())
    Trampolines = createEHTrampolines(BF);

  // Check the new size to see if it's worth splitting the function.
  if (BC.isX86() && BF.isSplit()) {
    std::tie(HotSize, ColdSize) = BC.calculateEmittedSize(BF);
    LLVM_DEBUG(dbgs() << "Estimated size for function " << BF
                      << " post-split is <0x" << Twine::utohexstr(HotSize)
                      << ", 0x" << Twine::utohexstr(ColdSize) << ">\n");
    if (alignTo(OriginalHotSize, opts::SplitAlignThreshold) <=
        alignTo(HotSize, opts::SplitAlignThreshold) + opts::SplitThreshold) {
      LLVM_DEBUG(dbgs() << "Reversing splitting of function " << BF << ":\n  0x"
                        << Twine::utohexstr(HotSize) << ", 0x"
                        << Twine::utohexstr(ColdSize) << " -> 0x"
                        << Twine::utohexstr(OriginalHotSize) << '\n');

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
