//===- bolt/Passes/AllocCombiner.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AllocCombinerPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/AllocCombiner.h"

#define DEBUG_TYPE "alloccombiner"

using namespace llvm;

namespace opts {

extern cl::opt<bolt::FrameOptimizationType> FrameOptimization;

} // end namespace opts

namespace llvm {
namespace bolt {

static bool getStackAdjustmentSize(const BinaryContext &BC, const MCInst &Inst,
                                   int64_t &Adjustment) {
  return BC.MIB->evaluateStackOffsetExpr(
      Inst, Adjustment, std::make_pair(BC.MIB->getStackPointer(), 0LL),
      std::make_pair(0, 0LL));
}

static bool isIndifferentToSP(const MCInst &Inst, const BinaryContext &BC) {
  if (BC.MIB->isCFI(Inst))
    return true;

  const MCInstrDesc &II = BC.MII->get(Inst.getOpcode());
  if (BC.MIB->isTerminator(Inst) ||
      II.hasImplicitDefOfPhysReg(BC.MIB->getStackPointer(), BC.MRI.get()) ||
      II.hasImplicitUseOfPhysReg(BC.MIB->getStackPointer()))
    return false;

  for (const MCOperand &Operand : MCPlus::primeOperands(Inst))
    if (Operand.isReg() && Operand.getReg() == BC.MIB->getStackPointer())
      return false;
  return true;
}

static bool shouldProcess(const BinaryFunction &Function) {
  return Function.isSimple() && Function.hasCFG() && !Function.isIgnored();
}

static void runForAllWeCare(std::map<uint64_t, BinaryFunction> &BFs,
                            std::function<void(BinaryFunction &)> Task) {
  for (auto &It : BFs) {
    BinaryFunction &Function = It.second;
    if (shouldProcess(Function))
      Task(Function);
  }
}

void AllocCombinerPass::combineAdjustments(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock &BB : BF) {
    MCInst *Prev = nullptr;
    for (MCInst &Inst : llvm::reverse(BB)) {
      if (isIndifferentToSP(Inst, BC))
        continue; // Skip updating Prev

      int64_t Adjustment = 0LL;
      if (!Prev || !BC.MIB->isStackAdjustment(Inst) ||
          !BC.MIB->isStackAdjustment(*Prev) ||
          !getStackAdjustmentSize(BC, *Prev, Adjustment)) {
        Prev = &Inst;
        continue;
      }

      LLVM_DEBUG({
        dbgs() << "At \"" << BF.getPrintName() << "\", combining: \n";
        Inst.dump();
        Prev->dump();
        dbgs() << "Adjustment: " << Adjustment << "\n";
      });

      if (BC.MIB->isSUB(Inst))
        Adjustment = -Adjustment;

      BC.MIB->addToImm(Inst, Adjustment, BC.Ctx.get());

      LLVM_DEBUG({
        dbgs() << "After adjustment:\n";
        Inst.dump();
      });

      BB.eraseInstruction(BB.findInstruction(Prev));
      ++NumCombined;
      DynamicCountCombined += BB.getKnownExecutionCount();
      FuncsChanged.insert(&BF);
      Prev = &Inst;
    }
  }
}

Error AllocCombinerPass::runOnFunctions(BinaryContext &BC) {
  if (opts::FrameOptimization == FOP_NONE)
    return Error::success();

  runForAllWeCare(BC.getBinaryFunctions(), [&](BinaryFunction &Function) {
    combineAdjustments(Function);
  });

  BC.outs() << "BOLT-INFO: Allocation combiner: " << NumCombined
            << " empty spaces coalesced (dyn count: " << DynamicCountCombined
            << ").\n";
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
