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
#include "bolt/Utils/CommandLineOpts.h"
#include <iterator>

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

// We don't exit directly from runOnFunction since it would call ThreadPool
// destructor which might result in internal assert if we're not finished
// creating async jobs on the moment of exit. So we're finishing all parallel
// jobs and checking the exit flag after it.
static bool PassFailed = false;

void ADRRelaxationPass::runOnFunction(BinaryFunction &BF) {
  if (PassFailed)
    return;

  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock &BB : BF) {
    for (auto It = BB.begin(); It != BB.end(); ++It) {
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

      // Don't relax adr if it points to the same function and it is not split
      // and BF initial size is < 1MB.
      const unsigned OneMB = 0x100000;
      if (!BF.isSplit() && BF.getSize() < OneMB) {
        BinaryFunction *TargetBF = BC.getFunctionForSymbol(Symbol);
        if (TargetBF && TargetBF == &BF)
          continue;
      }

      MCPhysReg Reg;
      BC.MIB->getADRReg(Inst, Reg);
      int64_t Addend = BC.MIB->getTargetAddend(Inst);
      InstructionListType Addr;

      {
        auto L = BC.scopeLock();
        Addr = BC.MIB->materializeAddress(Symbol, BC.Ctx.get(), Reg, Addend);
      }

      if (It != BB.begin() && BC.MIB->isNoop(*std::prev(It))) {
        It = BB.eraseInstruction(std::prev(It));
      } else if (std::next(It) != BB.end() && BC.MIB->isNoop(*std::next(It))) {
        BB.eraseInstruction(std::next(It));
      } else if (!opts::StrictMode && !BF.isSimple()) {
        // If the function is not simple, it may contain a jump table undetected
        // by us. This jump table may use an offset from the branch instruction
        // to land in the desired place. If we add new instructions, we
        // invalidate this offset, so we have to rely on linker-inserted NOP to
        // replace it with ADRP, and abort if it is not present.
        auto L = BC.scopeLock();
        BC.errs() << formatv(
            "BOLT-ERROR: Cannot relax adr in non-simple function "
            "{0}. Use --strict option to override\n",
            BF.getOneName());
        PassFailed = true;
        return;
      }
      It = BB.replaceInstruction(It, Addr);
    }
  }
}

Error ADRRelaxationPass::runOnFunctions(BinaryContext &BC) {
  if (!opts::AdrPassOpt || !BC.HasRelocations)
    return Error::success();

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun, nullptr,
      "ADRRelaxationPass");

  if (PassFailed)
    return createFatalBOLTError("");
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
