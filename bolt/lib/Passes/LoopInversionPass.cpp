//===- bolt/Passes/LoopInversionPass.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LoopInversionPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/LoopInversionPass.h"
#include "bolt/Core/ParallelUtilities.h"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltCategory;

extern cl::opt<bolt::ReorderBasicBlocks::LayoutType> ReorderBlocks;

static cl::opt<bool> LoopReorder(
    "loop-inversion-opt",
    cl::desc("reorder unconditional jump instructions in loops optimization"),
    cl::init(true), cl::cat(BoltCategory), cl::ReallyHidden);
} // namespace opts

namespace llvm {
namespace bolt {

bool LoopInversionPass::runOnFunction(BinaryFunction &BF) {
  bool IsChanged = false;
  if (BF.getLayout().block_size() < 3 || !BF.hasValidProfile())
    return false;

  BF.getLayout().updateLayoutIndices();
  for (BinaryBasicBlock *BB : BF.getLayout().blocks()) {
    if (BB->succ_size() != 1 || BB->pred_size() != 1)
      continue;

    BinaryBasicBlock *SuccBB = *BB->succ_begin();
    BinaryBasicBlock *PredBB = *BB->pred_begin();
    const unsigned BBIndex = BB->getLayoutIndex();
    const unsigned SuccBBIndex = SuccBB->getLayoutIndex();
    if (SuccBB == PredBB && BB != SuccBB && BBIndex != 0 && SuccBBIndex != 0 &&
        SuccBB->succ_size() == 2 && BB->isCold() == SuccBB->isCold()) {
      // Get the second successor (after loop BB)
      BinaryBasicBlock *SecondSucc = nullptr;
      for (BinaryBasicBlock *Succ : SuccBB->successors()) {
        if (Succ != &*BB) {
          SecondSucc = Succ;
          break;
        }
      }

      assert(SecondSucc != nullptr && "Unable to find a second BB successor");
      const uint64_t LoopCount = SuccBB->getBranchInfo(*BB).Count;
      const uint64_t ExitCount = SuccBB->getBranchInfo(*SecondSucc).Count;

      if (LoopCount < ExitCount) {
        if (BBIndex > SuccBBIndex)
          continue;
      } else if (BBIndex < SuccBBIndex) {
        continue;
      }

      IsChanged = true;
      BB->setLayoutIndex(SuccBBIndex);
      SuccBB->setLayoutIndex(BBIndex);
    }
  }

  if (IsChanged) {
    BinaryFunction::BasicBlockOrderType NewOrder(BF.getLayout().block_begin(),
                                                 BF.getLayout().block_end());
    llvm::sort(NewOrder, [&](BinaryBasicBlock *BB1, BinaryBasicBlock *BB2) {
      return BB1->getLayoutIndex() < BB2->getLayoutIndex();
    });
    BF.getLayout().update(NewOrder);
  }

  return IsChanged;
}

void LoopInversionPass::runOnFunctions(BinaryContext &BC) {
  std::atomic<uint64_t> ModifiedFuncCount{0};
  if (opts::ReorderBlocks == ReorderBasicBlocks::LT_NONE ||
      opts::LoopReorder == false)
    return;

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    if (runOnFunction(BF))
      ++ModifiedFuncCount;
  };

  ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
    return !shouldOptimize(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun, SkipFunc,
      "LoopInversionPass");

  outs() << "BOLT-INFO: " << ModifiedFuncCount
         << " Functions were reordered by LoopInversionPass\n";
}

} // end namespace bolt
} // end namespace llvm
