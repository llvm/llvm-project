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

  std::unordered_map<const MCSymbol *, const MCSymbol *> VeneerDestinations;
  uint64_t VeneersCount = 0;
  uint64_t NumAllVeneers = 0;
  for (BinaryFunction &BF : llvm::make_second_range(BC.getBinaryFunctions())) {
    if (!BF.isAArch64Veneer())
      continue;

    ++NumAllVeneers;

    if (BF.isIgnored())
      continue;

    MCInst &FirstInstruction = *(BF.begin()->begin());
    const MCSymbol *VeneerTargetSymbol;
    if (BC.MIB->isTailCall(FirstInstruction)) {
      VeneerTargetSymbol = BC.MIB->getTargetSymbol(FirstInstruction);
    } else {
      if (!BC.MIB->hasAnnotation(FirstInstruction, "AArch64Veneer"))
        continue;
      VeneerTargetSymbol = BC.MIB->getTargetSymbol(FirstInstruction, 1);
    }

    if (!VeneerTargetSymbol)
      continue;

    for (const MCSymbol *Symbol : BF.getSymbols())
      VeneerDestinations[Symbol] = VeneerTargetSymbol;

    VeneersCount++;
    BF.setPseudo(true);
  }

  BC.outs() << "BOLT-INFO: number of removed linker-inserted veneers: "
            << VeneersCount << ". Total veneers: " << NumAllVeneers << '\n';

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
        if (VeneerDestinations.find(TargetSymbol) == VeneerDestinations.end())
          continue;

        VeneerCallers++;
        BC.MIB->replaceBranchTarget(Instr, VeneerDestinations[TargetSymbol],
                                    BC.Ctx.get());
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
