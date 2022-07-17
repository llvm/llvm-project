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
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/ParallelUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
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
                   cl::desc("split functions into hot and cold regions"),
                   cl::cat(BoltOptCategory));

static cl::opt<unsigned> SplitThreshold(
    "split-threshold",
    cl::desc("split function only if its main size is reduced by more than "
             "given amount of bytes. Default value: 0, i.e. split iff the "
             "size is reduced. Note that on some architectures the size can "
             "increase after splitting."),
    cl::init(0), cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    RandomSplit("split-random",
                cl::desc("split functions randomly into hot/cold regions"),
                cl::Hidden);
} // namespace opts

namespace {
struct SplitCold {
  bool canSplit(const BinaryFunction &BF) {
    if (!BF.hasValidProfile())
      return false;

    bool AllCold = true;
    for (const BinaryBasicBlock &BB : BF) {
      const uint64_t ExecCount = BB.getExecutionCount();
      if (ExecCount == BinaryBasicBlock::COUNT_NO_PROFILE)
        return false;
      if (ExecCount != 0)
        AllCold = false;
    }

    return !AllCold;
  }

  bool canOutline(const BinaryBasicBlock &BB) {
    return BB.getExecutionCount() == 0;
  }

  template <typename It> void partition(const It Start, const It End) const {
    for (auto I = Start; I != End; ++I) {
      BinaryBasicBlock *BB = *I;
      if (!BB->canOutline())
        break;
      BB->setIsCold(true);
    }
  }
};

struct SplitRandom {
  std::minstd_rand0 *Gen;

  explicit SplitRandom(std::minstd_rand0 &Gen) : Gen(&Gen) {}

  bool canSplit(const BinaryFunction &BF) { return true; }
  bool canOutline(const BinaryBasicBlock &BB) { return true; }

  template <typename It> void partition(It Start, It End) const {
    using DiffT = typename It::difference_type;

    const It OutlineableBegin = Start;
    const It OutlineableEnd =
        std::find_if(OutlineableBegin, End, [](const BinaryBasicBlock *BB) {
          return !BB->canOutline();
        });
    const DiffT NumOutlineableBlocks = OutlineableEnd - OutlineableBegin;

    // We want to split at least one block unless there are not blocks that can
    // be outlined
    const auto MinimumSplit = std::min<DiffT>(NumOutlineableBlocks, 1);
    std::uniform_int_distribution<DiffT> Dist(MinimumSplit,
                                              NumOutlineableBlocks);
    const DiffT NumColdBlocks = Dist(*Gen);
    const It ColdEnd = OutlineableBegin + NumColdBlocks;

    LLVM_DEBUG(dbgs() << formatv("BOLT-DEBUG: randomly chose last {0} (out of "
                                 "{1} possible) blocks to split\n",
                                 ColdEnd - OutlineableBegin,
                                 OutlineableEnd - OutlineableBegin));

    std::for_each(OutlineableBegin, ColdEnd,
                  [](BinaryBasicBlock *BB) { BB->setIsCold(true); });
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

  ParallelUtilities::WorkFuncTy WorkFun;
  std::minstd_rand0 RandGen(opts::RandomSeed.getValue());
  if (opts::RandomSplit)
    WorkFun = [&](BinaryFunction &BF) {
      splitFunction(BF, SplitRandom(RandGen));
    };
  else
    WorkFun = [&](BinaryFunction &BF) { splitFunction<SplitCold>(BF); };

  ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
    return !shouldOptimize(BF);
  };

  // If we split functions randomly, we need to ensure that across runs with the
  // same input, we generate random numbers for each function in the same order.
  const bool ForceSequential = opts::RandomSplit;

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_BB_LINEAR, WorkFun, SkipFunc,
      "SplitFunctions", ForceSequential);

  if (SplitBytesHot + SplitBytesCold > 0)
    outs() << "BOLT-INFO: splitting separates " << SplitBytesHot
           << " hot bytes from " << SplitBytesCold << " cold bytes "
           << format("(%.2lf%% of split functions is hot).\n",
                     100.0 * SplitBytesHot / (SplitBytesHot + SplitBytesCold));
}

template <typename SplitStrategy>
void SplitFunctions::splitFunction(BinaryFunction &BF, SplitStrategy Strategy) {
  if (BF.empty())
    return;

  if (!Strategy.canSplit(BF))
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
    if (!Strategy.canOutline(*BB)) {
      BB->setCanOutline(false);
      continue;
    }
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

  if (opts::AggressiveSplitting) {
    // All blocks with 0 count that we can move go to the end of the function.
    // Even if they were natural to cluster formation and were seen in-between
    // hot basic blocks.
    stable_sort(NewLayout, [&](BinaryBasicBlock *A, BinaryBasicBlock *B) {
      return A->canOutline() < B->canOutline();
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
                       return A->canOutline() < B->canOutline();
                     });
  }

  // Separate hot from cold starting from the bottom.
  Strategy.partition(NewLayout.rbegin(), NewLayout.rend());
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
        BB.setIsCold(false);
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
      const Optional<MCPlus::MCLandingPad> EHInfo = MIB->getEHInfo(Instr);
      if (!EHInfo || !EHInfo->first)
        continue;

      const MCSymbol *LPLabel = EHInfo->first;
      BinaryBasicBlock *LPBlock = BF.getBasicBlockForLabel(LPLabel);
      if (BB->isCold() == LPBlock->isCold())
        continue;

      const MCSymbol *TrampolineLabel = nullptr;
      auto Iter = LPTrampolines.find(LPLabel);
      if (Iter != LPTrampolines.end()) {
        TrampolineLabel = Iter->second;
      } else {
        // Create a trampoline basic block in the same fragment as the thrower.
        // Note: there's no need to insert the jump instruction, it will be
        // added by fixBranches().
        BinaryBasicBlock *TrampolineBB = BF.addBasicBlock();
        TrampolineBB->setIsCold(BB->isCold());
        TrampolineBB->setExecutionCount(LPBlock->getExecutionCount());
        TrampolineBB->addSuccessor(LPBlock, TrampolineBB->getExecutionCount());
        TrampolineBB->setCFIState(LPBlock->getCFIState());
        TrampolineLabel = TrampolineBB->getLabel();
        LPTrampolines.insert(std::make_pair(LPLabel, TrampolineLabel));
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
    return A->isCold() < B->isCold();
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
  BasicBlockOrderType MergedLayout;
  for (BinaryBasicBlock *BB : Layout) {
    auto Iter = Trampolines.find(BB->getLabel());
    if (Iter != Trampolines.end()) {
      BinaryBasicBlock *LPBlock = BF.getBasicBlockForLabel(Iter->second);
      assert(LPBlock && "Could not find matching landing pad block.");
      MergedLayout.push_back(LPBlock);
    }
    MergedLayout.push_back(BB);
  }

  return MergedLayout;
}

} // namespace bolt
} // namespace llvm
