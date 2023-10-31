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
} // namespace opts

namespace llvm {
namespace bolt {

namespace {
/// Return true if the function should be considered for building call graph.
bool shouldConsider(const BinaryFunction &BF) {
  return BF.hasValidIndex() && BF.hasValidProfile() && !BF.empty();
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
}

/// Find the best index for splitting. The returned value is the index of the
/// last hot basic block. Hence, "no splitting" is equivalent to returning the
/// value which is one less than the size of the function.
size_t CDSplit::findSplitIndex(const BinaryFunction &BF,
                               const BasicBlockOrder &BlockOrder) {
  // Placeholder: hot-cold splitting.
  return BF.getLayout().getMainFragment().size() - 1;
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
