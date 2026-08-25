//===- bolt/Passes/VeneerElimination.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements a pass that removes linker-inserted veneers from the
// code and redirects veneer callers to call to veneers destinations
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/VeneerElimination.h"
#define DEBUG_TYPE "veneer-elim"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

static llvm::cl::opt<bool>
    EliminateVeneers("elim-link-veneers",
                     cl::desc("run veneer elimination pass"), cl::init(true),
                     cl::Hidden, cl::cat(BoltOptCategory));

static llvm::cl::opt<bool> DropCortexA53843419Veneers(
    "drop-cortex-a53-843419-veneers",
    cl::desc("inline and drop Cortex-A53 erratum 843419 linker veneers; only "
             "use if the BOLTed binary will not run on Cortex-A53"),
    cl::init(false), cl::cat(BoltOptCategory));
} // namespace opts

namespace llvm {
namespace bolt {

Error VeneerElimination::runOnFunctions(BinaryContext &BC) {
  if (!opts::EliminateVeneers || !BC.isAArch64())
    return Error::success();

  std::unordered_map<const MCSymbol *, const MCSymbol *> VeneerDestinations;
  uint64_t NumEliminatedVeneers = 0;
  uint64_t NumE843419Inlined = 0;

  for (BinaryFunction &BF : llvm::make_second_range(BC.getBinaryFunctions())) {
    if (!BF.isPossibleVeneer())
      continue;

    if (BF.isIgnored())
      continue;

    // Cortex-A53 erratum 843419 veneers: inline the veneer body at each
    // branch site instead of redirecting, so LongJmp do not introduce
    // code that clobbers registers (e.g. x16) used by the caller.
    if (BC.MIB->matchE843419Veneer(BF)) {
      if (!opts::DropCortexA53843419Veneers) {
        BC.errs() << "BOLT-ERROR: binary contains Cortex-A53 erratum 843419 "
                     "workaround veneers; pass "
                     "--drop-cortex-a53-843419-veneers only if the BOLTed "
                     "binary will not run on Cortex-A53, or relink without "
                     "--fix-cortex-a53-843419\n";
        exit(1);
      }

      const MCInst &VeneerFirstInstr = BF.front().getInstructionAtIndex(0);
      const MCSymbol *ReturnTargetSym =
          BC.MIB->getTargetSymbol(BF.front().getInstructionAtIndex(1));

      // Check if this branch targets our e843419 veneer.
      auto BranchTargetsVeneer = [&BF, &BC](const MCSymbol *Target) {
        if (!Target)
          return false;
        if (BC.getFunctionForSymbol(Target) == &BF)
          return true;
        if (ErrorOr<uint64_t> Addr = BC.getSymbolValue(*Target))
          return BC.getBinaryFunctionContainingAddress(*Addr) == &BF;
        return false;
      };

      uint64_t CallSites = 0;

      // Find caller from veneer's branch-back target so we can limit the scan.
      BinaryFunction *CallerBF = nullptr;
      if (ReturnTargetSym) {
        if (ErrorOr<uint64_t> Addr = BC.getSymbolValue(*ReturnTargetSym))
          CallerBF = BC.getBinaryFunctionContainingAddress(*Addr);
      }

      auto ScanForBranchesToVeneer = [&](BinaryFunction &F) {
        for (BinaryBasicBlock &BB : F) {
          for (auto II = BB.begin(); II != BB.end();) {
            MCInst &Instr = *II;
            if (!BC.MIB->isBranch(Instr)) {
              ++II;
              continue;
            }
            const MCSymbol *Target = BC.MIB->getTargetSymbol(Instr);
            if (!BranchTargetsVeneer(Target)) {
              ++II;
              continue;
            }
            InstructionListType Repl;
            Repl.emplace_back(VeneerFirstInstr);
            II = BB.replaceInstruction(II, Repl);
            ++CallSites;
            ++NumE843419Inlined;
          }
        }
      };

      if (CallerBF && !CallerBF->isIgnored())
        ScanForBranchesToVeneer(*CallerBF);
      else
        LLVM_DEBUG(dbgs() << "BOLT: skipping e843419 veneer inline for "
                          << BF.getOneName()
                          << " (caller not resolved or ignored)\n");

      // Only mark pseudo when we actually inlined at least one branch site.
      if (CallSites > 0) {
        ++NumEliminatedVeneers;
        BF.setPseudo(true);
        LLVM_DEBUG(dbgs() << "BOLT-INFO: inlined e843419 veneer "
                          << BF.getOneName() << " at " << CallSites
                          << " branch sites\n");
      } else {
        LLVM_DEBUG(dbgs() << "BOLT: e843419 veneer " << BF.getOneName()
                          << " left unchanged (no branch sites inlined)\n");
      }
      continue;
    }

    MCInst &FirstInstruction = *(BF.begin()->begin());
    const MCSymbol *VeneerTargetSymbol = 0;
    uint64_t TargetAddress;
    if (BC.MIB->isTailCall(FirstInstruction)) {
      VeneerTargetSymbol = BC.MIB->getTargetSymbol(FirstInstruction);
    } else if (BC.MIB->matchAbsLongVeneer(BF, TargetAddress)) {
      if (BinaryFunction *TargetBF =
              BC.getBinaryFunctionAtAddress(TargetAddress))
        VeneerTargetSymbol = TargetBF->getSymbol();
    } else if (BC.MIB->hasAnnotation(FirstInstruction, "AArch64Veneer")) {
      VeneerTargetSymbol = BC.MIB->getTargetSymbol(FirstInstruction, 1);
    }

    if (!VeneerTargetSymbol)
      continue;

    for (const MCSymbol *Symbol : BF.getSymbols())
      VeneerDestinations[Symbol] = VeneerTargetSymbol;

    NumEliminatedVeneers++;
    BF.setPseudo(true);
  }

  BC.outs() << "BOLT-INFO: number of removed linker-inserted veneers: "
            << NumEliminatedVeneers << '\n';
  if (NumE843419Inlined)
    BC.outs() << "BOLT-INFO: e843419 veneer call sites inlined: "
              << NumE843419Inlined << '\n';

  // Handle veneers to veneers in case they occur
  for (auto &Entry : VeneerDestinations) {
    const MCSymbol *Src = Entry.first;
    const MCSymbol *Dest = Entry.second;
    while (VeneerDestinations.find(Dest) != VeneerDestinations.end())
      Dest = VeneerDestinations[Dest];

    VeneerDestinations[Src] = Dest;
  }

  uint64_t VeneerCallers = 0;
  for (BinaryFunction &BF : llvm::make_second_range(BC.getBinaryFunctions())) {
    for (BinaryBasicBlock &BB : BF) {
      for (MCInst &Instr : BB) {
        if (!BC.MIB->isCall(Instr) || BC.MIB->isIndirectCall(Instr))
          continue;

        const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Instr, 0);
        auto It = VeneerDestinations.find(TargetSymbol);
        if (It == VeneerDestinations.end())
          continue;

        VeneerCallers++;
        BC.MIB->replaceBranchTarget(Instr, It->second, BC.Ctx.get());
      }
    }
  }

  LLVM_DEBUG(
      dbgs() << "BOLT-INFO: number of linker-inserted veneers call sites: "
             << VeneerCallers << "\n");
  (void)VeneerCallers;
  return Error::success();
}

} // namespace bolt
} // namespace llvm
