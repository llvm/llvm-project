//===- bolt/Passes/TailDuplication.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TailDuplication class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/TailDuplication.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <numeric>

#define DEBUG_TYPE "taildup"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static cl::opt<bool> TailDuplicationAggressive(
    "tail-duplication-aggressive",
    cl::desc("tail duplication should act aggressively in duplicating multiple "
             "blocks per tail"),
    cl::ZeroOrMore, cl::ReallyHidden, cl::init(false),
    cl::cat(BoltOptCategory));

static cl::opt<unsigned>
    TailDuplicationMinimumOffset("tail-duplication-minimum-offset",
                                 cl::desc("minimum offset needed between block "
                                          "and successor to allow duplication"),
                                 cl::ZeroOrMore, cl::ReallyHidden, cl::init(64),
                                 cl::cat(BoltOptCategory));

static cl::opt<unsigned> TailDuplicationMaximumDuplication(
    "tail-duplication-maximum-duplication",
    cl::desc("maximum size of duplicated blocks (in bytes)"), cl::ZeroOrMore,
    cl::ReallyHidden, cl::init(64), cl::cat(BoltOptCategory));

static cl::opt<bool> TailDuplicationConstCopyPropagation(
    "tail-duplication-const-copy-propagation",
    cl::desc("enable const and copy propagation after tail duplication"),
    cl::ReallyHidden, cl::init(false), cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {
void TailDuplication::getCallerSavedRegs(const MCInst &Inst, BitVector &Regs,
                                         BinaryContext &BC) const {
  if (!BC.MIB->isCall(Inst))
    return;
  BitVector CallRegs = BitVector(BC.MRI->getNumRegs(), false);
  BC.MIB->getCalleeSavedRegs(CallRegs);
  CallRegs.flip();
  Regs |= CallRegs;
}

bool TailDuplication::regIsPossiblyOverwritten(const MCInst &Inst, unsigned Reg,
                                               BinaryContext &BC) const {
  BitVector WrittenRegs = BitVector(BC.MRI->getNumRegs(), false);
  BC.MIB->getWrittenRegs(Inst, WrittenRegs);
  getCallerSavedRegs(Inst, WrittenRegs, BC);
  if (BC.MIB->isRep(Inst))
    BC.MIB->getRepRegs(WrittenRegs);
  WrittenRegs &= BC.MIB->getAliases(Reg, false);
  return WrittenRegs.any();
}

bool TailDuplication::regIsDefinitelyOverwritten(const MCInst &Inst,
                                                 unsigned Reg,
                                                 BinaryContext &BC) const {
  BitVector WrittenRegs = BitVector(BC.MRI->getNumRegs(), false);
  BC.MIB->getWrittenRegs(Inst, WrittenRegs);
  getCallerSavedRegs(Inst, WrittenRegs, BC);
  if (BC.MIB->isRep(Inst))
    BC.MIB->getRepRegs(WrittenRegs);
  return (!regIsUsed(Inst, Reg, BC) && WrittenRegs.test(Reg) &&
          !BC.MIB->isConditionalMove(Inst));
}

bool TailDuplication::regIsUsed(const MCInst &Inst, unsigned Reg,
                                BinaryContext &BC) const {
  BitVector SrcRegs = BitVector(BC.MRI->getNumRegs(), false);
  BC.MIB->getSrcRegs(Inst, SrcRegs);
  SrcRegs &= BC.MIB->getAliases(Reg, true);
  return SrcRegs.any();
}

bool TailDuplication::isOverwrittenBeforeUsed(BinaryBasicBlock &StartBB,
                                              unsigned Reg) const {
  BinaryFunction *BF = StartBB.getFunction();
  BinaryContext &BC = BF->getBinaryContext();
  std::queue<BinaryBasicBlock *> Q;
  for (auto Itr = StartBB.succ_begin(); Itr != StartBB.succ_end(); ++Itr) {
    BinaryBasicBlock *NextBB = *Itr;
    Q.push(NextBB);
  }
  std::set<BinaryBasicBlock *> Visited;
  // Breadth first search through successive blocks and see if Reg is ever used
  // before its overwritten
  while (Q.size() > 0) {
    BinaryBasicBlock *CurrBB = Q.front();
    Q.pop();
    if (Visited.count(CurrBB))
      continue;
    Visited.insert(CurrBB);
    bool Overwritten = false;
    for (auto Itr = CurrBB->begin(); Itr != CurrBB->end(); ++Itr) {
      MCInst &Inst = *Itr;
      if (regIsUsed(Inst, Reg, BC))
        return false;
      if (regIsDefinitelyOverwritten(Inst, Reg, BC)) {
        Overwritten = true;
        break;
      }
    }
    if (Overwritten)
      continue;
    for (auto Itr = CurrBB->succ_begin(); Itr != CurrBB->succ_end(); ++Itr) {
      BinaryBasicBlock *NextBB = *Itr;
      Q.push(NextBB);
    }
  }
  return true;
}

void TailDuplication::constantAndCopyPropagate(
    BinaryBasicBlock &OriginalBB,
    std::vector<BinaryBasicBlock *> &BlocksToPropagate) {
  BinaryFunction *BF = OriginalBB.getFunction();
  BinaryContext &BC = BF->getBinaryContext();

  BlocksToPropagate.insert(BlocksToPropagate.begin(), &OriginalBB);
  // Iterate through the original instructions to find one to propagate
  for (auto Itr = OriginalBB.begin(); Itr != OriginalBB.end(); ++Itr) {
    MCInst &OriginalInst = *Itr;
    // It must be a non conditional
    if (BC.MIB->isConditionalMove(OriginalInst))
      continue;

    // Move immediate or move register
    if ((!BC.MII->get(OriginalInst.getOpcode()).isMoveImmediate() ||
         !OriginalInst.getOperand(1).isImm()) &&
        (!BC.MII->get(OriginalInst.getOpcode()).isMoveReg() ||
         !OriginalInst.getOperand(1).isReg()))
      continue;

    // True if this is constant propagation and not copy propagation
    bool ConstantProp = BC.MII->get(OriginalInst.getOpcode()).isMoveImmediate();
    // The Register to replaced
    unsigned Reg = OriginalInst.getOperand(0).getReg();
    // True if the register to replace was replaced everywhere it was used
    bool ReplacedEverywhere = true;
    // True if the register was definitely overwritten
    bool Overwritten = false;
    // True if the register to replace and the register to replace with (for
    // copy propagation) has not been overwritten and is still usable
    bool RegsActive = true;

    // Iterate through successor blocks and through their instructions
    for (BinaryBasicBlock *NextBB : BlocksToPropagate) {
      for (auto PropagateItr =
               ((NextBB == &OriginalBB) ? Itr + 1 : NextBB->begin());
           PropagateItr < NextBB->end(); ++PropagateItr) {
        MCInst &PropagateInst = *PropagateItr;
        if (regIsUsed(PropagateInst, Reg, BC)) {
          bool Replaced = false;
          // If both registers are active for copy propagation or the register
          // to replace is active for constant propagation
          if (RegsActive) {
            // Set Replaced and so ReplacedEverwhere to false if it cannot be
            // replaced (no replacing that opcode, Register is src and dest)
            if (ConstantProp)
              Replaced = BC.MIB->replaceRegWithImm(
                  PropagateInst, Reg, OriginalInst.getOperand(1).getImm());
            else
              Replaced = BC.MIB->replaceRegWithReg(
                  PropagateInst, Reg, OriginalInst.getOperand(1).getReg());
          }
          ReplacedEverywhere = ReplacedEverywhere && Replaced;
        }
        // For copy propagation, make sure no propagation happens after the
        // register to replace with is overwritten
        if (!ConstantProp &&
            regIsPossiblyOverwritten(PropagateInst,
                                     OriginalInst.getOperand(1).getReg(), BC))
          RegsActive = false;

        // Make sure no propagation happens after the register to replace is
        // overwritten
        if (regIsPossiblyOverwritten(PropagateInst, Reg, BC))
          RegsActive = false;

        // Record if the register to replace is overwritten
        if (regIsDefinitelyOverwritten(PropagateInst, Reg, BC)) {
          Overwritten = true;
          break;
        }
      }
      if (Overwritten)
        break;
    }

    // If the register was replaced everwhere and it was overwritten in either
    // one of the iterated through blocks or one of the successor blocks, delete
    // the original move instruction
    if (ReplacedEverywhere &&
        (Overwritten ||
         isOverwrittenBeforeUsed(
             *BlocksToPropagate[BlocksToPropagate.size() - 1], Reg))) {
      // If both registers are active for copy propagation or the register
      // to replace is active for constant propagation
      StaticInstructionDeletionCount++;
      DynamicInstructionDeletionCount += OriginalBB.getExecutionCount();
      Itr = std::prev(OriginalBB.eraseInstruction(Itr));
    }
  }
}

bool TailDuplication::isInCacheLine(const BinaryBasicBlock &BB,
                                    const BinaryBasicBlock &Succ) const {
  if (&BB == &Succ)
    return true;

  BinaryFunction::BasicBlockOrderType BlockLayout =
      BB.getFunction()->getLayout();
  uint64_t Distance = 0;
  int Direction = (Succ.getLayoutIndex() > BB.getLayoutIndex()) ? 1 : -1;

  for (unsigned I = BB.getLayoutIndex() + Direction; I != Succ.getLayoutIndex();
       I += Direction) {
    Distance += BlockLayout[I]->getOriginalSize();
    if (Distance > opts::TailDuplicationMinimumOffset)
      return false;
  }
  return true;
}

std::vector<BinaryBasicBlock *>
TailDuplication::moderateCodeToDuplicate(BinaryBasicBlock &BB) const {
  std::vector<BinaryBasicBlock *> BlocksToDuplicate;
  if (BB.hasJumpTable())
    return BlocksToDuplicate;
  if (BB.getOriginalSize() > opts::TailDuplicationMaximumDuplication)
    return BlocksToDuplicate;
  for (auto Itr = BB.succ_begin(); Itr != BB.succ_end(); ++Itr) {
    if ((*Itr)->getLayoutIndex() == BB.getLayoutIndex() + 1)
      // If duplicating would introduce a new branch, don't duplicate
      return BlocksToDuplicate;
  }
  BlocksToDuplicate.push_back(&BB);
  return BlocksToDuplicate;
}

std::vector<BinaryBasicBlock *>
TailDuplication::aggressiveCodeToDuplicate(BinaryBasicBlock &BB) const {
  std::vector<BinaryBasicBlock *> BlocksToDuplicate;
  BinaryBasicBlock *CurrBB = &BB;
  while (CurrBB) {
    LLVM_DEBUG(dbgs() << "Aggressive tail duplication: adding "
                      << CurrBB->getName() << " to duplication list\n";);
    BlocksToDuplicate.push_back(CurrBB);

    if (CurrBB->hasJumpTable()) {
      LLVM_DEBUG(dbgs() << "Aggressive tail duplication: clearing duplication "
                           "list due to a JT in "
                        << CurrBB->getName() << '\n';);
      BlocksToDuplicate.clear();
      break;
    }

    // With no successors, we've reached the end and should duplicate all of
    // BlocksToDuplicate
    if (CurrBB->succ_size() == 0)
      break;

    // With two successors, if they're both a jump, we should duplicate all
    // blocks in BlocksToDuplicate. Otherwise, we cannot find a simple stream of
    // blocks to copy
    if (CurrBB->succ_size() >= 2) {
      if (CurrBB->getConditionalSuccessor(false)->getLayoutIndex() ==
              CurrBB->getLayoutIndex() + 1 ||
          CurrBB->getConditionalSuccessor(true)->getLayoutIndex() ==
              CurrBB->getLayoutIndex() + 1) {
        LLVM_DEBUG(dbgs() << "Aggressive tail duplication: clearing "
                             "duplication list, can't find a simple stream at "
                          << CurrBB->getName() << '\n';);
        BlocksToDuplicate.clear();
      }
      break;
    }

    // With one successor, if its a jump, we should duplicate all blocks in
    // BlocksToDuplicate. Otherwise, we should keep going
    BinaryBasicBlock *Succ = CurrBB->getSuccessor();
    if (Succ->getLayoutIndex() != CurrBB->getLayoutIndex() + 1)
      break;
    CurrBB = Succ;
  }
  // Don't duplicate if its too much code
  unsigned DuplicationByteCount = std::accumulate(
      std::begin(BlocksToDuplicate), std::end(BlocksToDuplicate), 0,
      [](int value, BinaryBasicBlock *p) {
        return value + p->getOriginalSize();
      });
  if (DuplicationByteCount > opts::TailDuplicationMaximumDuplication) {
    LLVM_DEBUG(dbgs() << "Aggressive tail duplication: duplication byte count ("
                      << DuplicationByteCount << ") exceeds maximum "
                      << opts::TailDuplicationMaximumDuplication << '\n';);
    BlocksToDuplicate.clear();
  }
  LLVM_DEBUG(dbgs() << "Aggressive tail duplication: found "
                    << BlocksToDuplicate.size() << " blocks to duplicate\n";);
  return BlocksToDuplicate;
}

std::vector<BinaryBasicBlock *> TailDuplication::tailDuplicate(
    BinaryBasicBlock &BB,
    const std::vector<BinaryBasicBlock *> &BlocksToDuplicate) const {
  BinaryFunction *BF = BB.getFunction();
  BinaryContext &BC = BF->getBinaryContext();

  // Ratio of this new branches execution count to the total size of the
  // successor's execution count.  Used to set this new branches execution count
  // and lower the old successor's execution count
  double ExecutionCountRatio =
      BB.getExecutionCount() > BB.getSuccessor()->getExecutionCount()
          ? 1.0
          : (double)BB.getExecutionCount() /
                BB.getSuccessor()->getExecutionCount();

  // Use the last branch info when adding a successor to LastBB
  BinaryBasicBlock::BinaryBranchInfo &LastBI =
      BB.getBranchInfo(*(BB.getSuccessor()));

  BinaryBasicBlock *LastOriginalBB = &BB;
  BinaryBasicBlock *LastDuplicatedBB = &BB;
  assert(LastDuplicatedBB->succ_size() == 1 &&
         "tail duplication cannot act on a block with more than 1 successor");
  LastDuplicatedBB->removeSuccessor(LastDuplicatedBB->getSuccessor());

  std::vector<std::unique_ptr<BinaryBasicBlock>> DuplicatedBlocks;
  std::vector<BinaryBasicBlock *> DuplicatedBlocksToReturn;

  for (BinaryBasicBlock *CurrBB : BlocksToDuplicate) {
    DuplicatedBlocks.emplace_back(
        BF->createBasicBlock(0, (BC.Ctx)->createNamedTempSymbol("tail-dup")));
    BinaryBasicBlock *NewBB = DuplicatedBlocks.back().get();

    NewBB->addInstructions(CurrBB->begin(), CurrBB->end());
    // Set execution count as if it was just a copy of the original
    NewBB->setExecutionCount(
        std::max((uint64_t)1, CurrBB->getExecutionCount()));
    LastDuplicatedBB->addSuccessor(NewBB, LastBI);

    DuplicatedBlocksToReturn.push_back(NewBB);

    // As long as its not the first block, adjust both original and duplicated
    // to what they should be
    if (LastDuplicatedBB != &BB) {
      LastOriginalBB->adjustExecutionCount(1.0 - ExecutionCountRatio);
      LastDuplicatedBB->adjustExecutionCount(ExecutionCountRatio);
    }

    if (CurrBB->succ_size() == 1)
      LastBI = CurrBB->getBranchInfo(*(CurrBB->getSuccessor()));

    LastOriginalBB = CurrBB;
    LastDuplicatedBB = NewBB;
  }

  LastDuplicatedBB->addSuccessors(
      LastOriginalBB->succ_begin(), LastOriginalBB->succ_end(),
      LastOriginalBB->branch_info_begin(), LastOriginalBB->branch_info_end());

  LastOriginalBB->adjustExecutionCount(1.0 - ExecutionCountRatio);
  LastDuplicatedBB->adjustExecutionCount(ExecutionCountRatio);

  BF->insertBasicBlocks(&BB, std::move(DuplicatedBlocks));

  return DuplicatedBlocksToReturn;
}

void TailDuplication::runOnFunction(BinaryFunction &Function) {
  // New blocks will be added and layout will change,
  // so make a copy here to iterate over the original layout
  BinaryFunction::BasicBlockOrderType BlockLayout = Function.getLayout();
  for (BinaryBasicBlock *BB : BlockLayout) {
    if (BB->succ_size() == 1 &&
        BB->getSuccessor()->getLayoutIndex() != BB->getLayoutIndex() + 1)
      UnconditionalBranchDynamicCount += BB->getExecutionCount();
    if (BB->succ_size() == 2 &&
        BB->getFallthrough()->getLayoutIndex() != BB->getLayoutIndex() + 1)
      UnconditionalBranchDynamicCount += BB->getFallthroughBranchInfo().Count;
    AllBlocksDynamicCount += BB->getExecutionCount();

    // The block must be hot
    if (BB->getExecutionCount() == 0)
      continue;
    // with one successor
    if (BB->succ_size() != 1)
      continue;

    // no jump table
    if (BB->hasJumpTable())
      continue;

    // Skip not-in-layout, i.e. unreachable, blocks.
    if (BB->getLayoutIndex() >= BlockLayout.size())
      continue;

    // and we are estimating that this sucessor is not already in the same cache
    // line
    BinaryBasicBlock *Succ = BB->getSuccessor();
    if (isInCacheLine(*BB, *Succ))
      continue;
    std::vector<BinaryBasicBlock *> BlocksToDuplicate;
    if (opts::TailDuplicationAggressive)
      BlocksToDuplicate = aggressiveCodeToDuplicate(*Succ);
    else
      BlocksToDuplicate = moderateCodeToDuplicate(*Succ);

    if (BlocksToDuplicate.size() == 0)
      continue;
    PossibleDuplications++;
    PossibleDuplicationsDynamicCount += BB->getExecutionCount();
    std::vector<BinaryBasicBlock *> DuplicatedBlocks =
        tailDuplicate(*BB, BlocksToDuplicate);
    if (!opts::TailDuplicationConstCopyPropagation)
      continue;

    constantAndCopyPropagate(*BB, DuplicatedBlocks);
    BinaryBasicBlock *FirstBB = BlocksToDuplicate[0];
    if (FirstBB->pred_size() == 1) {
      BinaryBasicBlock *PredBB = *FirstBB->pred_begin();
      if (PredBB->succ_size() == 1)
        constantAndCopyPropagate(*PredBB, BlocksToDuplicate);
    }
  }
}

void TailDuplication::runOnFunctions(BinaryContext &BC) {
  for (auto &It : BC.getBinaryFunctions()) {
    BinaryFunction &Function = It.second;
    if (!shouldOptimize(Function))
      continue;
    runOnFunction(Function);
  }

  outs() << "BOLT-INFO: tail duplication possible duplications: "
         << PossibleDuplications << "\n";
  outs() << "BOLT-INFO: tail duplication possible dynamic reductions: "
         << PossibleDuplicationsDynamicCount << "\n";
  outs() << "BOLT-INFO: tail duplication possible dynamic reductions to "
            "unconditional branch execution : "
         << format("%.1f", ((float)PossibleDuplicationsDynamicCount * 100.0f) /
                               UnconditionalBranchDynamicCount)
         << "%\n";
  outs() << "BOLT-INFO: tail duplication possible dynamic reductions to all "
            "blocks execution : "
         << format("%.1f", ((float)PossibleDuplicationsDynamicCount * 100.0f) /
                               AllBlocksDynamicCount)
         << "%\n";
  outs() << "BOLT-INFO: tail duplication static propagation deletions: "
         << StaticInstructionDeletionCount << "\n";
  outs() << "BOLT-INFO: tail duplication dynamic propagation deletions: "
         << DynamicInstructionDeletionCount << "\n"; //
}

} // end namespace bolt
} // end namespace llvm
