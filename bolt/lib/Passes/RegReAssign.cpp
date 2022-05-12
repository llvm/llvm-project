//===- bolt/Passes/RegReAssign.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RegReAssign class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/RegReAssign.h"
#include "bolt/Core/MCPlus.h"
#include "bolt/Passes/BinaryFunctionCallGraph.h"
#include "bolt/Passes/DataflowAnalysis.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Utils/Utils.h"
#include <numeric>

#define DEBUG_TYPE "regreassign"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltOptCategory;
extern cl::opt<bool> UpdateDebugSections;

static cl::opt<bool>
AggressiveReAssign("use-aggr-reg-reassign",
  cl::desc("use register liveness analysis to try to find more opportunities "
           "for -reg-reassign optimization"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

}

namespace llvm {
namespace bolt {

void RegReAssign::swap(BinaryFunction &Function, MCPhysReg A, MCPhysReg B) {
  BinaryContext &BC = Function.getBinaryContext();
  const BitVector &AliasA = BC.MIB->getAliases(A, false);
  const BitVector &AliasB = BC.MIB->getAliases(B, false);

  // Regular instructions
  for (BinaryBasicBlock &BB : Function) {
    for (MCInst &Inst : BB) {
      for (MCOperand &Operand : MCPlus::primeOperands(Inst)) {
        if (!Operand.isReg())
          continue;

        unsigned Reg = Operand.getReg();
        if (AliasA.test(Reg)) {
          Operand.setReg(BC.MIB->getAliasSized(B, BC.MIB->getRegSize(Reg)));
          --StaticBytesSaved;
          DynBytesSaved -= BB.getKnownExecutionCount();
          continue;
        }
        if (!AliasB.test(Reg))
          continue;
        Operand.setReg(BC.MIB->getAliasSized(A, BC.MIB->getRegSize(Reg)));
        ++StaticBytesSaved;
        DynBytesSaved += BB.getKnownExecutionCount();
      }
    }
  }

  // CFI
  DenseSet<const MCCFIInstruction *> Changed;
  for (BinaryBasicBlock &BB : Function) {
    for (MCInst &Inst : BB) {
      if (!BC.MIB->isCFI(Inst))
        continue;
      const MCCFIInstruction *CFI = Function.getCFIFor(Inst);
      if (Changed.count(CFI))
        continue;
      Changed.insert(CFI);

      switch (CFI->getOperation()) {
      case MCCFIInstruction::OpRegister: {
        const unsigned CFIReg2 = CFI->getRegister2();
        const MCPhysReg Reg2 = *BC.MRI->getLLVMRegNum(CFIReg2, /*isEH=*/false);
        if (AliasA.test(Reg2)) {
          Function.setCFIFor(
              Inst, MCCFIInstruction::createRegister(
                        nullptr, CFI->getRegister(),
                        BC.MRI->getDwarfRegNum(
                            BC.MIB->getAliasSized(B, BC.MIB->getRegSize(Reg2)),
                            false)));
        } else if (AliasB.test(Reg2)) {
          Function.setCFIFor(
              Inst, MCCFIInstruction::createRegister(
                        nullptr, CFI->getRegister(),
                        BC.MRI->getDwarfRegNum(
                            BC.MIB->getAliasSized(A, BC.MIB->getRegSize(Reg2)),
                            false)));
        }
      }
      LLVM_FALLTHROUGH;
      case MCCFIInstruction::OpUndefined:
      case MCCFIInstruction::OpDefCfa:
      case MCCFIInstruction::OpOffset:
      case MCCFIInstruction::OpRestore:
      case MCCFIInstruction::OpSameValue:
      case MCCFIInstruction::OpDefCfaRegister:
      case MCCFIInstruction::OpRelOffset:
      case MCCFIInstruction::OpEscape: {
        unsigned CFIReg;
        if (CFI->getOperation() != MCCFIInstruction::OpEscape) {
          CFIReg = CFI->getRegister();
        } else {
          Optional<uint8_t> Reg =
              readDWARFExpressionTargetReg(CFI->getValues());
          // Handle DW_CFA_def_cfa_expression
          if (!Reg)
            break;
          CFIReg = *Reg;
        }
        const MCPhysReg Reg = *BC.MRI->getLLVMRegNum(CFIReg, /*isEH=*/false);
        if (AliasA.test(Reg))
          Function.mutateCFIRegisterFor(
              Inst,
              BC.MRI->getDwarfRegNum(
                  BC.MIB->getAliasSized(B, BC.MIB->getRegSize(Reg)), false));
        else if (AliasB.test(Reg))
          Function.mutateCFIRegisterFor(
              Inst,
              BC.MRI->getDwarfRegNum(
                  BC.MIB->getAliasSized(A, BC.MIB->getRegSize(Reg)), false));
        break;
      }
      default:
        break;
      }
    }
  }
}

void RegReAssign::rankRegisters(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  std::fill(RegScore.begin(), RegScore.end(), 0);
  std::fill(RankedRegs.begin(), RankedRegs.end(), 0);

  for (BinaryBasicBlock &BB : Function) {
    for (MCInst &Inst : BB) {
      const bool CannotUseREX = BC.MIB->cannotUseREX(Inst);
      const MCInstrDesc &Desc = BC.MII->get(Inst.getOpcode());

      // Disallow substituitions involving regs in implicit uses lists
      const MCPhysReg *ImplicitUses = Desc.getImplicitUses();
      while (ImplicitUses && *ImplicitUses) {
        const size_t RegEC =
            BC.MIB->getAliases(*ImplicitUses, false).find_first();
        RegScore[RegEC] =
            std::numeric_limits<decltype(RegScore)::value_type>::min();
        ++ImplicitUses;
      }

      // Disallow substituitions involving regs in implicit defs lists
      const MCPhysReg *ImplicitDefs = Desc.getImplicitDefs();
      while (ImplicitDefs && *ImplicitDefs) {
        const size_t RegEC =
            BC.MIB->getAliases(*ImplicitDefs, false).find_first();
        RegScore[RegEC] =
            std::numeric_limits<decltype(RegScore)::value_type>::min();
        ++ImplicitDefs;
      }

      for (int I = 0, E = MCPlus::getNumPrimeOperands(Inst); I != E; ++I) {
        const MCOperand &Operand = Inst.getOperand(I);
        if (!Operand.isReg())
          continue;

        if (Desc.getOperandConstraint(I, MCOI::TIED_TO) != -1)
          continue;

        unsigned Reg = Operand.getReg();
        size_t RegEC = BC.MIB->getAliases(Reg, false).find_first();
        if (RegEC == 0)
          continue;

        // Disallow substituitions involving regs in instrs that cannot use REX
        if (CannotUseREX) {
          RegScore[RegEC] =
              std::numeric_limits<decltype(RegScore)::value_type>::min();
          continue;
        }

        // Unsupported substitution, cannot swap BH with R* regs, bail
        if (BC.MIB->isUpper8BitReg(Reg) && ClassicCSR.test(Reg)) {
          RegScore[RegEC] =
              std::numeric_limits<decltype(RegScore)::value_type>::min();
          continue;
        }

        RegScore[RegEC] += BB.getKnownExecutionCount();
      }
    }
  }
  std::iota(RankedRegs.begin(), RankedRegs.end(), 0); // 0, 1, 2, 3...
  std::sort(RankedRegs.begin(), RankedRegs.end(),
            [&](size_t A, size_t B) { return RegScore[A] > RegScore[B]; });

  LLVM_DEBUG({
    for (size_t Reg : RankedRegs) {
      if (RegScore[Reg] == 0)
        continue;
      dbgs() << Reg << " ";
      if (RegScore[Reg] > 0)
        dbgs() << BC.MRI->getName(Reg) << ": " << RegScore[Reg] << "\n";
      else
        dbgs() << BC.MRI->getName(Reg) << ": (blacklisted)\n";
    }
  });
}

void RegReAssign::aggressivePassOverFunction(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  rankRegisters(Function);

  // Bail early if our registers are all black listed, before running expensive
  // analysis passes
  bool Bail = true;
  int64_t LowScoreClassic = std::numeric_limits<int64_t>::max();
  for (int J = ClassicRegs.find_first(); J != -1;
       J = ClassicRegs.find_next(J)) {
    if (RegScore[J] <= 0)
      continue;
    Bail = false;
    if (RegScore[J] < LowScoreClassic)
      LowScoreClassic = RegScore[J];
  }
  if (Bail)
    return;
  BitVector Extended = ClassicRegs;
  Extended.flip();
  Extended &= GPRegs;
  Bail = true;
  int64_t HighScoreExtended = 0;
  for (int J = Extended.find_first(); J != -1; J = Extended.find_next(J)) {
    if (RegScore[J] <= 0)
      continue;
    Bail = false;
    if (RegScore[J] > HighScoreExtended)
      HighScoreExtended = RegScore[J];
  }
  // Also bail early if there is no profitable substitution even if we assume
  // all registers can be exchanged
  if (Bail || (LowScoreClassic << 1) >= HighScoreExtended)
    return;

  // -- expensive pass -- determine all regs alive during func start
  DataflowInfoManager Info(Function, RA.get(), nullptr);
  BitVector AliveAtStart = *Info.getLivenessAnalysis().getStateAt(
      ProgramPoint::getFirstPointAt(*Function.begin()));
  for (BinaryBasicBlock &BB : Function)
    if (BB.pred_size() == 0)
      AliveAtStart |= *Info.getLivenessAnalysis().getStateAt(
          ProgramPoint::getFirstPointAt(BB));

  // Mark frame pointer alive because of CFI
  AliveAtStart |= BC.MIB->getAliases(BC.MIB->getFramePointer(), false);
  // Never touch return registers
  BC.MIB->getDefaultLiveOut(AliveAtStart);

  // Try swapping more profitable options first
  auto Begin = RankedRegs.begin();
  auto End = std::prev(RankedRegs.end());
  while (Begin != End) {
    MCPhysReg ClassicReg = *End;
    if (!ClassicRegs[ClassicReg] || RegScore[ClassicReg] <= 0) {
      --End;
      continue;
    }

    MCPhysReg ExtReg = *Begin;
    if (!Extended[ExtReg] || RegScore[ExtReg] <= 0) {
      ++Begin;
      continue;
    }

    if (RegScore[ClassicReg] << 1 >= RegScore[ExtReg]) {
      LLVM_DEBUG(dbgs() << " Ending at " << BC.MRI->getName(ClassicReg)
                        << " with " << BC.MRI->getName(ExtReg)
                        << " because exchange is not profitable\n");
      break;
    }

    BitVector AnyAliasAlive = AliveAtStart;
    AnyAliasAlive &= BC.MIB->getAliases(ClassicReg);
    if (AnyAliasAlive.any()) {
      LLVM_DEBUG(dbgs() << " Bailed on " << BC.MRI->getName(ClassicReg)
                        << " with " << BC.MRI->getName(ExtReg)
                        << " because classic reg is alive\n");
      --End;
      continue;
    }
    AnyAliasAlive = AliveAtStart;
    AnyAliasAlive &= BC.MIB->getAliases(ExtReg);
    if (AnyAliasAlive.any()) {
      LLVM_DEBUG(dbgs() << " Bailed on " << BC.MRI->getName(ClassicReg)
                        << " with " << BC.MRI->getName(ExtReg)
                        << " because extended reg is alive\n");
      ++Begin;
      continue;
    }

    // Opportunity detected. Swap.
    LLVM_DEBUG(dbgs() << "\n ** Swapping " << BC.MRI->getName(ClassicReg)
                      << " with " << BC.MRI->getName(ExtReg) << "\n\n");
    swap(Function, ClassicReg, ExtReg);
    FuncsChanged.insert(&Function);
    ++Begin;
    if (Begin == End)
      break;
    --End;
  }
}

bool RegReAssign::conservativePassOverFunction(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  rankRegisters(Function);

  // Try swapping R12, R13, R14 or R15 with RBX (we work with all callee-saved
  // regs except RBP)
  MCPhysReg Candidate = 0;
  for (int J = ExtendedCSR.find_first(); J != -1;
       J = ExtendedCSR.find_next(J))
    if (RegScore[J] > RegScore[Candidate])
      Candidate = J;

  if (!Candidate || RegScore[Candidate] < 0)
    return false;

  // Check if our classic callee-saved reg (RBX is the only one) has lower
  // score / utilization rate
  MCPhysReg RBX = 0;
  for (int I = ClassicCSR.find_first(); I != -1; I = ClassicCSR.find_next(I)) {
    int64_t ScoreRBX = RegScore[I];
    if (ScoreRBX <= 0)
      continue;

    if (RegScore[Candidate] > (ScoreRBX + 10))
      RBX = I;
  }

  if (!RBX)
    return false;

  LLVM_DEBUG(dbgs() << "\n ** Swapping " << BC.MRI->getName(RBX) << " with "
                    << BC.MRI->getName(Candidate) << "\n\n");
  swap(Function, RBX, Candidate);
  FuncsChanged.insert(&Function);
  return true;
}

void RegReAssign::setupAggressivePass(BinaryContext &BC,
                                      std::map<uint64_t, BinaryFunction> &BFs) {
  setupConservativePass(BC, BFs);
  CG.reset(new BinaryFunctionCallGraph(buildCallGraph(BC)));
  RA.reset(new RegAnalysis(BC, &BFs, &*CG));

  GPRegs = BitVector(BC.MRI->getNumRegs(), false);
  BC.MIB->getGPRegs(GPRegs);
}

void RegReAssign::setupConservativePass(
    BinaryContext &BC, std::map<uint64_t, BinaryFunction> &BFs) {
  // Set up constant bitvectors used throughout this analysis
  ClassicRegs = BitVector(BC.MRI->getNumRegs(), false);
  CalleeSaved = BitVector(BC.MRI->getNumRegs(), false);
  ClassicCSR = BitVector(BC.MRI->getNumRegs(), false);
  ExtendedCSR = BitVector(BC.MRI->getNumRegs(), false);
  // Never consider the frame pointer
  BC.MIB->getClassicGPRegs(ClassicRegs);
  ClassicRegs.flip();
  ClassicRegs |= BC.MIB->getAliases(BC.MIB->getFramePointer(), false);
  ClassicRegs.flip();
  BC.MIB->getCalleeSavedRegs(CalleeSaved);
  ClassicCSR |= ClassicRegs;
  ClassicCSR &= CalleeSaved;
  BC.MIB->getClassicGPRegs(ClassicRegs);
  ExtendedCSR |= ClassicRegs;
  ExtendedCSR.flip();
  ExtendedCSR &= CalleeSaved;

  LLVM_DEBUG({
    RegStatePrinter P(BC);
    dbgs() << "Starting register reassignment\nClassicRegs: ";
    P.print(dbgs(), ClassicRegs);
    dbgs() << "\nCalleeSaved: ";
    P.print(dbgs(), CalleeSaved);
    dbgs() << "\nClassicCSR: ";
    P.print(dbgs(), ClassicCSR);
    dbgs() << "\nExtendedCSR: ";
    P.print(dbgs(), ExtendedCSR);
    dbgs() << "\n";
  });
}

void RegReAssign::runOnFunctions(BinaryContext &BC) {
  RegScore = std::vector<int64_t>(BC.MRI->getNumRegs(), 0);
  RankedRegs = std::vector<size_t>(BC.MRI->getNumRegs(), 0);

  if (opts::AggressiveReAssign)
    setupAggressivePass(BC, BC.getBinaryFunctions());
  else
    setupConservativePass(BC, BC.getBinaryFunctions());

  for (auto &I : BC.getBinaryFunctions()) {
    BinaryFunction &Function = I.second;

    if (!Function.isSimple() || Function.isIgnored())
      continue;

    LLVM_DEBUG(dbgs() << "====================================\n");
    LLVM_DEBUG(dbgs() << " - " << Function.getPrintName() << "\n");
    if (!conservativePassOverFunction(Function) && opts::AggressiveReAssign) {
      aggressivePassOverFunction(Function);
      LLVM_DEBUG({
        if (FuncsChanged.count(&Function))
          dbgs() << "Aggressive pass successful on " << Function.getPrintName()
                 << "\n";
      });
    }
  }

  if (FuncsChanged.empty()) {
    outs() << "BOLT-INFO: Reg Reassignment Pass: no changes were made.\n";
    return;
  }
  if (opts::UpdateDebugSections)
    outs() << "BOLT-WARNING: You used -reg-reassign and -update-debug-sections."
           << " Some registers were changed but associated AT_LOCATION for "
           << "impacted variables were NOT updated! This operation is "
           << "currently unsupported by BOLT.\n";
  outs() << "BOLT-INFO: Reg Reassignment Pass Stats:\n";
  outs() << "\t   " << FuncsChanged.size() << " functions affected.\n";
  outs() << "\t   " << StaticBytesSaved << " static bytes saved.\n";
  outs() << "\t   " << DynBytesSaved << " dynamic bytes saved.\n";
}

} // namespace bolt
} // namespace llvm
