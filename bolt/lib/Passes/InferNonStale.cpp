//===- bolt/Passes/InferNonStale.cpp - Non-stale profile inference ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the InferNonStale pass that runs stale profile
// matching on functions with non-stale/non-inferred profile to improve
// profile quality.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/InferNonStale.h"

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Transforms/Utils/SampleProfileInference.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "infer-non-stale"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::opt<bool> TimeRewrite;
extern cl::OptionCategory BoltOptCategory;

cl::opt<bool>
    InferNonStaleProfile("infer-non-stale-profile",
                         cl::desc("Infer profile counts for functions with "
                                  "non-stale profile using profi"),
                         cl::init(false), cl::cat(BoltOptCategory));

// Reuse existing stale matching parameters
extern cl::opt<bool> StaleMatchingEvenFlowDistribution;
extern cl::opt<bool> StaleMatchingRebalanceUnknown;
extern cl::opt<bool> StaleMatchingJoinIslands;
extern cl::opt<unsigned> StaleMatchingCostBlockInc;
extern cl::opt<unsigned> StaleMatchingCostBlockDec;
extern cl::opt<unsigned> StaleMatchingCostJumpInc;
extern cl::opt<unsigned> StaleMatchingCostJumpDec;
extern cl::opt<unsigned> StaleMatchingCostBlockUnknownInc;
extern cl::opt<unsigned> StaleMatchingCostJumpUnknownInc;
extern cl::opt<unsigned> StaleMatchingCostJumpUnknownFTInc;

} // namespace opts

namespace llvm {
namespace bolt {

// Forward declarations of functions from StaleProfileMatching.cpp
FlowFunction
createFlowFunction(const BinaryFunction::BasicBlockOrderType &BlockOrder);
void preprocessUnreachableBlocks(FlowFunction &Func);
void assignProfile(BinaryFunction &BF,
                   const BinaryFunction::BasicBlockOrderType &BlockOrder,
                   FlowFunction &Func);

} // namespace bolt
} // namespace llvm

namespace llvm {
namespace bolt {

void InferNonStale::runOnFunction(BinaryFunction &BF) {
  NamedRegionTimer T("inferNonStale", "non-stale profile inference", "rewrite",
                     "Rewrite passes", opts::TimeRewrite);

  assert(BF.hasCFG() && "Function must have CFG");

  // Only process functions with profile that are not already inferred
  assert(BF.hasValidProfile() && "Function must have valid profile");

  assert(!BF.hasInferredProfile() && "Function must not have inferred profile");

  LLVM_DEBUG(dbgs() << "BOLT-INFO: applying non-stale profile inference for "
                    << "\"" << BF.getPrintName() << "\"\n");

  // Make sure that block hashes are up to date.
  BF.computeBlockHashes();

  const BinaryFunction::BasicBlockOrderType BlockOrder(
      BF.getLayout().block_begin(), BF.getLayout().block_end());

  // Create a wrapper flow function to use with the profile inference algorithm.
  FlowFunction Func = createFlowFunction(BlockOrder);

  // Assign existing profile counts to the flow function
  // This differs from stale matching - we use existing counts directly
  for (uint64_t I = 0; I < BlockOrder.size(); I++) {
    BinaryBasicBlock *BB = BlockOrder[I];
    FlowBlock &Block = Func.Blocks[I + 1]; // Skip dummy entry block

    // Set block weight from existing execution count
    Block.Weight = BB->getKnownExecutionCount();
    Block.HasUnknownWeight = (Block.Weight == 0);

    // Set jump weights from existing branch info
    for (FlowJump *Jump : Block.SuccJumps) {
      if (Jump->Target == Func.Blocks.size() - 1) // Skip artificial sink
        continue;

      BinaryBasicBlock *SuccBB = BlockOrder[Jump->Target - 1];
      if (BB->getSuccessor(SuccBB->getLabel())) {
        BinaryBasicBlock::BinaryBranchInfo &BI = BB->getBranchInfo(*SuccBB);
        Jump->Weight = BI.Count;
        Jump->HasUnknownWeight = (Jump->Weight == 0);
      }
    }
  }

  // Adjust the flow function by marking unreachable blocks Unlikely
  preprocessUnreachableBlocks(Func);

  // Set up inference parameters
  ProfiParams Params;
  Params.EvenFlowDistribution = opts::StaleMatchingEvenFlowDistribution;
  Params.RebalanceUnknown = opts::StaleMatchingRebalanceUnknown;
  Params.JoinIslands = opts::StaleMatchingJoinIslands;

  Params.CostBlockInc = opts::StaleMatchingCostBlockInc;
  Params.CostBlockEntryInc = opts::StaleMatchingCostBlockInc;
  Params.CostBlockDec = opts::StaleMatchingCostBlockDec;
  Params.CostBlockEntryDec = opts::StaleMatchingCostBlockDec;
  Params.CostBlockUnknownInc = opts::StaleMatchingCostBlockUnknownInc;

  Params.CostJumpInc = opts::StaleMatchingCostJumpInc;
  Params.CostJumpFTInc = opts::StaleMatchingCostJumpInc;
  Params.CostJumpDec = opts::StaleMatchingCostJumpDec;
  Params.CostJumpFTDec = opts::StaleMatchingCostJumpDec;
  Params.CostJumpUnknownInc = opts::StaleMatchingCostJumpUnknownInc;
  Params.CostJumpUnknownFTInc = opts::StaleMatchingCostJumpUnknownFTInc;

  // Apply the profile inference algorithm
  applyFlowInference(Params, Func);

  // Collect inferred counts and update function annotations
  assignProfile(BF, BlockOrder, Func);

  // Mark the function as having inferred profile
  BF.setHasInferredProfile(true);
}

Error InferNonStale::runOnFunctions(BinaryContext &BC) {
  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
    return !BF.hasValidProfile() || BF.hasInferredProfile() || !BF.hasCFG();
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_BB_QUADRATIC, WorkFun,
      SkipFunc, "InferNonStale");

  return Error::success();
}

} // namespace bolt
} // namespace llvm
