//===- bolt/Passes/ThreeWayBranch.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ThreeWayBranch class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ThreeWayBranch.h"

using namespace llvm;

namespace llvm {
namespace bolt {

bool ThreeWayBranch::shouldRunOnFunction(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  for (const BinaryBasicBlock &BB : Function)
    for (const MCInst &Inst : BB)
      if (BC.MIB->isPacked(Inst))
        return false;
  return true;
}

void ThreeWayBranch::runOnFunction(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  MCContext *Ctx = BC.Ctx.get();
  // New blocks will be added and layout will change,
  // so make a copy here to iterate over the original layout
  BinaryFunction::BasicBlockOrderType BlockLayout(
      Function.getLayout().block_begin(), Function.getLayout().block_end());
  for (BinaryBasicBlock *BB : BlockLayout) {
    // The block must be hot
    if (BB->getExecutionCount() == 0 ||
        BB->getExecutionCount() == BinaryBasicBlock::COUNT_NO_PROFILE)
      continue;
    // with two successors
    if (BB->succ_size() != 2)
      continue;
    // no jump table
    if (BB->hasJumpTable())
      continue;

    BinaryBasicBlock *FalseSucc = BB->getConditionalSuccessor(false);
    BinaryBasicBlock *TrueSucc = BB->getConditionalSuccessor(true);

    // One of BB's successors must have only one instruction that is a
    // conditional jump
    if ((FalseSucc->succ_size() != 2 || FalseSucc->size() != 1) &&
        (TrueSucc->succ_size() != 2 || TrueSucc->size() != 1))
      continue;

    // SecondBranch has the second conditional jump
    BinaryBasicBlock *SecondBranch = FalseSucc;
    BinaryBasicBlock *FirstEndpoint = TrueSucc;
    if (FalseSucc->succ_size() != 2) {
      SecondBranch = TrueSucc;
      FirstEndpoint = FalseSucc;
    }

    BinaryBasicBlock *SecondEndpoint =
        SecondBranch->getConditionalSuccessor(false);
    BinaryBasicBlock *ThirdEndpoint =
        SecondBranch->getConditionalSuccessor(true);

    // Make sure we can modify the jump in SecondBranch without disturbing any
    // other paths
    if (SecondBranch->pred_size() != 1)
      continue;

    // Get Jump Instructions
    MCInst *FirstJump = BB->getLastNonPseudoInstr();
    MCInst *SecondJump = SecondBranch->getLastNonPseudoInstr();

    // Get condition codes
    unsigned FirstCC = BC.MIB->getCondCode(*FirstJump);
    if (SecondBranch != FalseSucc)
      FirstCC = BC.MIB->getInvertedCondCode(FirstCC);
    // ThirdCC = ThirdCond && !FirstCC = !(!ThirdCond ||
    // !(!FirstCC)) = !(!ThirdCond || FirstCC)
    unsigned ThirdCC =
        BC.MIB->getInvertedCondCode(BC.MIB->getCondCodesLogicalOr(
            BC.MIB->getInvertedCondCode(BC.MIB->getCondCode(*SecondJump)),
            FirstCC));
    // SecondCC = !ThirdCond && !FirstCC = !(!(!ThirdCond) ||
    // !(!FirstCC)) = !(ThirdCond || FirstCC)
    unsigned SecondCC =
        BC.MIB->getInvertedCondCode(BC.MIB->getCondCodesLogicalOr(
            BC.MIB->getCondCode(*SecondJump), FirstCC));

    if (!BC.MIB->isValidCondCode(FirstCC) ||
        !BC.MIB->isValidCondCode(ThirdCC) || !BC.MIB->isValidCondCode(SecondCC))
      continue;

    std::vector<std::pair<BinaryBasicBlock *, unsigned>> Blocks;
    Blocks.push_back(std::make_pair(FirstEndpoint, FirstCC));
    Blocks.push_back(std::make_pair(SecondEndpoint, SecondCC));
    Blocks.push_back(std::make_pair(ThirdEndpoint, ThirdCC));

    llvm::sort(Blocks, [&](const std::pair<BinaryBasicBlock *, unsigned> A,
                           const std::pair<BinaryBasicBlock *, unsigned> B) {
      return A.first->getExecutionCount() < B.first->getExecutionCount();
    });

    uint64_t NewSecondBranchCount = Blocks[1].first->getExecutionCount() +
                                    Blocks[0].first->getExecutionCount();
    bool SecondBranchBigger =
        NewSecondBranchCount > Blocks[2].first->getExecutionCount();

    BB->removeAllSuccessors();
    if (SecondBranchBigger) {
      BB->addSuccessor(Blocks[2].first, Blocks[2].first->getExecutionCount());
      BB->addSuccessor(SecondBranch, NewSecondBranchCount);
    } else {
      BB->addSuccessor(SecondBranch, NewSecondBranchCount);
      BB->addSuccessor(Blocks[2].first, Blocks[2].first->getExecutionCount());
    }

    // Remove and add so there is no duplicate successors
    SecondBranch->removeAllSuccessors();
    SecondBranch->addSuccessor(Blocks[0].first,
                               Blocks[0].first->getExecutionCount());
    SecondBranch->addSuccessor(Blocks[1].first,
                               Blocks[1].first->getExecutionCount());

    SecondBranch->setExecutionCount(NewSecondBranchCount);

    // Replace the branch condition to fallthrough for the most common block
    if (SecondBranchBigger)
      BC.MIB->replaceBranchCondition(*FirstJump, Blocks[2].first->getLabel(),
                                     Ctx, Blocks[2].second);
    else
      BC.MIB->replaceBranchCondition(
          *FirstJump, SecondBranch->getLabel(), Ctx,
          BC.MIB->getInvertedCondCode(Blocks[2].second));

    // Replace the branch condition to fallthrough for the second most common
    // block
    BC.MIB->replaceBranchCondition(*SecondJump, Blocks[0].first->getLabel(),
                                   Ctx, Blocks[0].second);

    ++BranchesAltered;
  }
}

Error ThreeWayBranch::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    BinaryFunction &Function = It.second;
    if (!shouldRunOnFunction(Function))
      continue;
    runOnFunction(Function);
  }

  BC.outs() << "BOLT-INFO: number of three way branches order changed: "
            << BranchesAltered << "\n";
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
