//===- bolt/Passes/ADRRelaxationPass.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ADRRelaxationPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ADRRelaxationPass.h"
#include "bolt/Core/ParallelUtilities.h"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltCategory;

static cl::opt<bool>
    AdrPassOpt("adr-relaxation",
               cl::desc("Replace ARM non-local ADR instructions with ADRP"),
               cl::init(true), cl::cat(BoltCategory), cl::ReallyHidden);
} // namespace opts

namespace llvm {
namespace bolt {

void ADRRelaxationPass::runOnFunction(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock *BB : BF.layout()) {
    for (auto It = BB->begin(); It != BB->end(); ++It) {
      MCInst &Inst = *It;
      if (!BC.MIB->isADR(Inst))
        continue;

      const MCSymbol *Symbol = BC.MIB->getTargetSymbol(Inst);
      if (!Symbol)
        continue;

      if (BF.hasIslandsInfo()) {
        BinaryFunction::IslandInfo &Islands = BF.getIslandInfo();
        if (Islands.Symbols.count(Symbol) || Islands.ProxySymbols.count(Symbol))
          continue;
      }

      BinaryFunction *TargetBF = BC.getFunctionForSymbol(Symbol);
      if (TargetBF && TargetBF == &BF)
        continue;

      MCPhysReg Reg;
      BC.MIB->getADRReg(Inst, Reg);
      int64_t Addend = BC.MIB->getTargetAddend(Inst);
      InstructionListType Addr =
          BC.MIB->materializeAddress(Symbol, BC.Ctx.get(), Reg, Addend);
      It = BB->replaceInstruction(It, Addr);
    }
  }
}

void ADRRelaxationPass::runOnFunctions(BinaryContext &BC) {
  if (!opts::AdrPassOpt || !BC.HasRelocations)
    return;

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun, nullptr,
      "ADRRelaxationPass", /* ForceSequential */ true);
}

} // end namespace bolt
} // end namespace llvm
