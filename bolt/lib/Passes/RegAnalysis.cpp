//===- bolt/Passes/RegAnalysis.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RegAnalysis class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/RegAnalysis.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/CallGraphWalker.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "ra"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> Verbosity;
extern cl::OptionCategory BoltOptCategory;

cl::opt<bool> AssumeABI("assume-abi",
                        cl::desc("assume the ABI is never violated"),
                        cl::cat(BoltOptCategory));
}

namespace llvm {
namespace bolt {

RegAnalysis::RegAnalysis(BinaryContext &BC,
                         std::map<uint64_t, BinaryFunction> *BFs,
                         BinaryFunctionCallGraph *CG)
    : BC(BC), CS(opts::AssumeABI ? ConservativeStrategy::CLOBBERS_ABI
                                 : ConservativeStrategy::CLOBBERS_ALL) {
  if (!CG)
    return;

  CallGraphWalker CGWalker(*CG);

  CGWalker.registerVisitor([&](BinaryFunction *Func) -> bool {
    BitVector RegsKilled = getFunctionClobberList(Func);
    bool Updated = RegsKilledMap.find(Func) == RegsKilledMap.end() ||
                   RegsKilledMap[Func] != RegsKilled;
    if (Updated)
      RegsKilledMap[Func] = std::move(RegsKilled);
    return Updated;
  });

  CGWalker.registerVisitor([&](BinaryFunction *Func) -> bool {
    BitVector RegsGen = getFunctionUsedRegsList(Func);
    bool Updated = RegsGenMap.find(Func) == RegsGenMap.end() ||
                   RegsGenMap[Func] != RegsGen;
    if (Updated)
      RegsGenMap[Func] = std::move(RegsGen);
    return Updated;
  });

  CGWalker.walk();

  if (opts::Verbosity == 0) {
#ifndef NDEBUG
    if (!DebugFlag || !isCurrentDebugType(DEBUG_TYPE))
      return;
#else
    return;
#endif
  }

  if (!BFs)
    return;

  // This loop is for computing statistics only
  for (auto &MapEntry : *BFs) {
    BinaryFunction *Func = &MapEntry.second;
    auto Iter = RegsKilledMap.find(Func);
    assert(Iter != RegsKilledMap.end() &&
           "Failed to compute all clobbers list");
    if (Iter->second.all()) {
      uint64_t Count = Func->getExecutionCount();
      if (Count != BinaryFunction::COUNT_NO_PROFILE)
        CountFunctionsAllClobber += Count;
      ++NumFunctionsAllClobber;
    }
    DEBUG_WITH_TYPE("ra",
      dbgs() << "Killed regs set for func: " << Func->getPrintName() << "\n";
      const BitVector &RegsKilled = Iter->second;
      int RegIdx = RegsKilled.find_first();
      while (RegIdx != -1) {
        dbgs() << "\tREG" << RegIdx;
        RegIdx = RegsKilled.find_next(RegIdx);
      };
      dbgs() << "\nUsed regs set for func: " << Func->getPrintName() << "\n";
      const BitVector &RegsUsed = RegsGenMap.find(Func)->second;
      RegIdx = RegsUsed.find_first();
      while (RegIdx != -1) {
        dbgs() << "\tREG" << RegIdx;
        RegIdx = RegsUsed.find_next(RegIdx);
      };
      dbgs() << "\n";
    );
  }
}

void RegAnalysis::beConservative(BitVector &Result) const {
  switch (CS) {
  case ConservativeStrategy::CLOBBERS_ALL:
    Result.set();
    break;
  case ConservativeStrategy::CLOBBERS_ABI: {
    BitVector BV(BC.MRI->getNumRegs(), false);
    BC.MIB->getCalleeSavedRegs(BV);
    BV.flip();
    Result |= BV;
    break;
  }
  case ConservativeStrategy::CLOBBERS_NONE:
    Result.reset();
    break;
  }
}

bool RegAnalysis::isConservative(BitVector &Vec) const {
  switch (CS) {
  case ConservativeStrategy::CLOBBERS_ALL:
    return Vec.all();
  case ConservativeStrategy::CLOBBERS_ABI: {
    BitVector BV(BC.MRI->getNumRegs(), false);
    BC.MIB->getCalleeSavedRegs(BV);
    BV |= Vec;
    return BV.all();
  }
  case ConservativeStrategy::CLOBBERS_NONE:
    return Vec.none();
  }
  return false;
}

void RegAnalysis::getInstUsedRegsList(const MCInst &Inst, BitVector &RegSet,
                                      bool GetClobbers) const {
  if (!BC.MIB->isCall(Inst)) {
    if (GetClobbers)
      BC.MIB->getClobberedRegs(Inst, RegSet);
    else
      BC.MIB->getUsedRegs(Inst, RegSet);
    return;
  }

  // If no call graph supplied...
  if (RegsKilledMap.size() == 0) {
    beConservative(RegSet);
    return;
  }

  const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Inst);
  // If indirect call, we know nothing
  if (TargetSymbol == nullptr) {
    beConservative(RegSet);
    return;
  }

  const BinaryFunction *Function = BC.getFunctionForSymbol(TargetSymbol);
  if (Function == nullptr) {
    // Call to a function without a BinaryFunction object.
    // This should be a call to a PLT entry, and since it is a trampoline to
    // a DSO, we can't really know the code in advance.
    beConservative(RegSet);
    return;
  }
  if (GetClobbers) {
    auto BV = RegsKilledMap.find(Function);
    if (BV != RegsKilledMap.end()) {
      RegSet |= BV->second;
      return;
    }
    // Ignore calls to function whose clobber list wasn't yet calculated. This
    // instruction will be evaluated again once we have info for the callee.
    return;
  }
  auto BV = RegsGenMap.find(Function);
  if (BV != RegsGenMap.end()) {
    RegSet |= BV->second;
    return;
  }
}

void RegAnalysis::getInstClobberList(const MCInst &Inst,
                                     BitVector &KillSet) const {
  return getInstUsedRegsList(Inst, KillSet, /*GetClobbers*/ true);
}

BitVector RegAnalysis::getFunctionUsedRegsList(const BinaryFunction *Func) {
  BitVector UsedRegs = BitVector(BC.MRI->getNumRegs(), false);

  if (!Func->isSimple() || !Func->hasCFG()) {
    beConservative(UsedRegs);
    return UsedRegs;
  }

  for (const BinaryBasicBlock &BB : *Func) {
    for (const MCInst &Inst : BB) {
      getInstUsedRegsList(Inst, UsedRegs, /*GetClobbers*/ false);
      if (UsedRegs.all())
        return UsedRegs;
    }
  }

  return UsedRegs;
}

BitVector RegAnalysis::getFunctionClobberList(const BinaryFunction *Func) {
  BitVector RegsKilled = BitVector(BC.MRI->getNumRegs(), false);

  if (!Func->isSimple() || !Func->hasCFG()) {
    beConservative(RegsKilled);
    return RegsKilled;
  }

  for (const BinaryBasicBlock &BB : *Func) {
    for (const MCInst &Inst : BB) {
      getInstClobberList(Inst, RegsKilled);
      if (RegsKilled.all())
        return RegsKilled;
    }
  }

  return RegsKilled;
}

void RegAnalysis::printStats() {
  outs() << "BOLT-INFO REG ANALYSIS: Number of functions conservatively "
            "treated as clobbering all registers: "
         << NumFunctionsAllClobber
         << format(" (%.1lf%% dyn cov)\n",
                   (100.0 * CountFunctionsAllClobber / CountDenominator));
}

} // namespace bolt
} // namespace llvm
