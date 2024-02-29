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
} // namespace opts

namespace llvm {
namespace bolt {

Error VeneerElimination::runOnFunctions(BinaryContext &BC) {
  if (!opts::EliminateVeneers || !BC.isAArch64())
    return Error::success();

  std::map<uint64_t, BinaryFunction> &BFs = BC.getBinaryFunctions();
  std::unordered_map<const MCSymbol *, const MCSymbol *> VeneerDestinations;
  uint64_t VeneersCount = 0;
  for (auto &It : BFs) {
    BinaryFunction &VeneerFunction = It.second;
    if (!VeneerFunction.isAArch64Veneer())
      continue;

    VeneersCount++;
    VeneerFunction.setPseudo(true);
    MCInst &FirstInstruction = *(VeneerFunction.begin()->begin());
    const MCSymbol *VeneerTargetSymbol =
        BC.MIB->getTargetSymbol(FirstInstruction, 1);
    assert(VeneerTargetSymbol && "Expecting target symbol for instruction");
    for (const MCSymbol *Symbol : VeneerFunction.getSymbols())
      VeneerDestinations[Symbol] = VeneerTargetSymbol;
  }

  BC.outs() << "BOLT-INFO: number of removed linker-inserted veneers: "
            << VeneersCount << "\n";

  // Handle veneers to veneers in case they occur
  for (auto &Entry : VeneerDestinations) {
    const MCSymbol *Src = Entry.first;
    const MCSymbol *Dest = Entry.second;
    while (VeneerDestinations.find(Dest) != VeneerDestinations.end())
      Dest = VeneerDestinations[Dest];

    VeneerDestinations[Src] = Dest;
  }

  uint64_t VeneerCallers = 0;
  for (auto &It : BFs) {
    BinaryFunction &Function = It.second;
    for (BinaryBasicBlock &BB : Function) {
      for (MCInst &Instr : BB) {
        if (!BC.MIB->isCall(Instr) || BC.MIB->isIndirectCall(Instr))
          continue;

        const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Instr, 0);
        if (VeneerDestinations.find(TargetSymbol) == VeneerDestinations.end())
          continue;

        VeneerCallers++;
        if (!BC.MIB->replaceBranchTarget(
                Instr, VeneerDestinations[TargetSymbol], BC.Ctx.get())) {
          return createFatalBOLTError(
              "BOLT-ERROR: updating veneer call destination failed\n");
        }
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
