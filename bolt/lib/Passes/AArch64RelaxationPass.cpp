//===- bolt/Passes/AArch64RelaxationPass.cpp
//----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64RelaxationPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/AArch64RelaxationPass.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Utils/CommandLineOpts.h"
#include <iterator>

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltCategory;

static cl::opt<bool> AArch64PassOpt(
    "adr-ldr-relaxation",
    cl::desc("Replace ARM non-local ADR/LDR instructions with ADRP"),
    cl::init(true), cl::cat(BoltCategory), cl::ReallyHidden);
} // namespace opts

namespace llvm {
namespace bolt {

// We don't exit directly from runOnFunction since it would call ThreadPool
// destructor which might result in internal assert if we're not finished
// creating async jobs on the moment of exit. So we're finishing all parallel
// jobs and checking the exit flag after it.
static bool PassFailed = false;

void AArch64RelaxationPass::runOnFunction(BinaryFunction &BF) {
  if (PassFailed)
    return;

  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock &BB : BF) {
    for (auto It = BB.begin(); It != BB.end(); ++It) {
      MCInst &Inst = *It;
      bool IsADR = BC.MIB->isADR(Inst);

      // TODO: Handlel other types of LDR (literal, PC-relative) instructions.
      if (!IsADR && !BC.MIB->isLDRXl(Inst) && !BC.MIB->isLDRWl(Inst))
        continue;

      const MCSymbol *Symbol = BC.MIB->getTargetSymbol(Inst, IsADR ? 0 : 1);
      if (!Symbol)
        continue;

      if (BF.hasIslandsInfo()) {
        BinaryFunction::IslandInfo &Islands = BF.getIslandInfo();
        if (Islands.Symbols.count(Symbol) || Islands.ProxySymbols.count(Symbol))
          continue;
      }

      // Don't relax ADR/LDR if it points to the same function and is in the
      // main fragment and BF initial size is < 1MB.
      const unsigned OneMB = 0x100000;
      if (BF.getSize() < OneMB) {
        BinaryFunction *TargetBF = BC.getFunctionForSymbol(Symbol);
        if (TargetBF == &BF && !BB.isSplit())
          continue;

        // No relaxation needed if ADR/LDR references a basic block in the same
        // fragment.
        if (BinaryBasicBlock *TargetBB = BF.getBasicBlockForLabel(Symbol))
          if (BB.getFragmentNum() == TargetBB->getFragmentNum())
            continue;
      }

      InstructionListType AdrpMaterialization;
      {
        auto L = BC.scopeLock();
        AdrpMaterialization =
            IsADR ? BC.MIB->undoAdrpAddRelaxation(Inst, BC.Ctx.get())
                  : BC.MIB->createAdrpLdr(Inst, BC.Ctx.get());
      }

      if (It != BB.begin() && BC.MIB->isNoop(*std::prev(It))) {
        It = BB.eraseInstruction(std::prev(It));
      } else if (std::next(It) != BB.end() && BC.MIB->isNoop(*std::next(It))) {
        BB.eraseInstruction(std::next(It));
      } else if (!BF.isSimple()) {
        // If the function is not simple, it may contain a jump table undetected
        // by us. This jump table may use an offset from the branch instruction
        // to land in the desired place. If we add new instructions, we
        // invalidate this offset, so we have to rely on linker-inserted NOP to
        // replace it with ADRP, and abort if it is not present.
        auto L = BC.scopeLock();
        BC.errs() << "BOLT-ERROR: cannot relax " << (IsADR ? "ADR" : "LDR")
                  << " in non-simple function " << BF << '\n';
        PassFailed = true;
        return;
      }
      It = BB.replaceInstruction(It, AdrpMaterialization);
    }
  }
}

Error AArch64RelaxationPass::runOnFunctions(BinaryContext &BC) {
  if (!opts::AArch64PassOpt || !BC.HasRelocations)
    return Error::success();

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun, nullptr,
      "AArch64RelaxationPass");

  if (PassFailed)
    return createFatalBOLTError("");
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
