//===- bolt/Passes/AArch64RelaxationPass.cpp ------------------------------===//
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
    "aarch64-relaxation",
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

bool AArch64RelaxationPass::redirectToIslandCopy(BinaryFunction &BF,
                                                 MCInst &Inst, unsigned OpNum,
                                                 const MCSymbol &Symbol) {
  assert(!opts::CloneConstantIsland && "Reference should already point to the "
                                       "cloned constant island if constant "
                                       "island clone is enabled");
  BinaryContext &BC = BF.getBinaryContext();
  const BinaryData *BD = BC.getBinaryDataByName(Symbol.getName());
  if (!BD)
    return false;

  const uint64_t Address = BD->getAddress();
  BinaryFunction *Owner = BC.getBinaryFunctionContainingAddress(
      Address, /*CheckPastEnd*/ false, /*UseMaxSize*/ true);
  if (!Owner || Owner == &BF || !Owner->isInConstantIsland(Address) ||
      Owner->hasDynamicRelocationAtIsland())
    return false;

  MCSymbol *Proxy = Owner->getOrCreateProxyIslandAccess(Address, BF);
  if (!Proxy)
    return false;

  if (!BF.hasIslandsInfo() || !BF.getIslandInfo().Dependency.contains(Owner)) {
    ++NumIslandCopies;
    IslandCopyBytes += Owner->estimateConstantIslandSize(&BF);
  }
  BF.createIslandDependency(Proxy, Owner);
  return BC.MIB->replaceMemOperandDisp(
      Inst, Proxy, BC.MIB->getTargetAddend(Inst, OpNum), BC.Ctx.get());
}

void AArch64RelaxationPass::runOnFunction(BinaryFunction &BF) {
  if (PassFailed)
    return;

  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock &BB : BF) {
    for (auto It = BB.begin(); It != BB.end(); ++It) {
      MCInst &Inst = *It;
      bool IsADR = BC.MIB->isADR(Inst);

      // TODO: Handle other types of LDR (literal, PC-relative) instructions.
      if (!IsADR && !BC.MIB->isLoadLiteralGPR(Inst))
        continue;

      const MCSymbol *Symbol = BC.MIB->getTargetSymbol(Inst, IsADR ? 0 : 1);
      if (!Symbol)
        continue;

      if (BF.hasIslandsInfo()) {
        BinaryFunction::IslandInfo &Islands = BF.getIslandInfo();
        if (Islands.Symbols.count(Symbol) || Islands.ProxySymbols.count(Symbol))
          continue;
      }

      // The layout of a non-simple function is preserved, so references within
      // the same fragment retain their original in-range displacement. For
      // simple functions, basic blocks can move, but an initial size below 1MiB
      // guarantees that internal references remain in range after reordering.
      const unsigned OneMB = 0x100000;
      if (!BF.isSimple() || BF.getSize() < OneMB) {
        BinaryFunction *TargetBF = BC.getFunctionForSymbol(Symbol);
        if (TargetBF == &BF && !BB.isSplit())
          continue;

        // No relaxation needed if ADR/LDR references a basic block in the same
        // fragment.
        if (BinaryBasicBlock *TargetBB = BF.getBasicBlockForLabel(Symbol))
          if (BB.getFragmentNum() == TargetBB->getFragmentNum())
            continue;
      }

      const bool NopBefore = It != BB.begin() && BC.MIB->isNoop(*std::prev(It));
      const bool NopAfter =
          std::next(It) != BB.end() && BC.MIB->isNoop(*std::next(It));
      if (!NopBefore && !NopAfter && !BF.isSimple()) {
        // If the function is not simple, it may contain a jump table undetected
        // by us. This jump table may use an offset from the branch instruction
        // to land in the desired place. If we add new instructions, we
        // invalidate this offset, so we have to rely on linker-inserted NOP to
        // replace it with ADRP, and abort if it is not present. With
        // --clone-constant-island=0, a reference to another function's
        // constant island can still use a copy of the island emitted with this
        // function. With cloning enabled, such references already use copies.
        auto L = BC.scopeLock();
        if (!opts::CloneConstantIsland &&
            redirectToIslandCopy(BF, Inst, IsADR ? 0 : 1, *Symbol))
          continue;
        BC.errs() << "BOLT-ERROR: cannot relax " << (IsADR ? "ADR" : "LDR")
                  << " in non-simple function " << BF << '\n';
        PassFailed = true;
        return;
      }

      InstructionListType AdrpMaterialization;
      {
        auto L = BC.scopeLock();
        AdrpMaterialization =
            IsADR ? BC.MIB->undoAdrpAddRelaxation(Inst, BC.Ctx.get())
                  : BC.MIB->createAdrpLdr(Inst, BC.Ctx.get());
      }

      if (NopBefore)
        It = BB.eraseInstruction(std::prev(It));
      else if (NopAfter)
        BB.eraseInstruction(std::next(It));
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

  if (NumIslandCopies)
    BC.outs() << "BOLT-INFO: copied " << NumIslandCopies
              << " constant island(s) totaling " << IslandCopyBytes
              << " bytes into non-simple functions\n";
  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
