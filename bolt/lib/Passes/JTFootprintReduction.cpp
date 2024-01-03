//===- bolt/Passes/JTFootprintReduction.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements JTFootprintReduction class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/JTFootprintReduction.h"
#include "bolt/Passes/BinaryFunctionCallGraph.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "JT"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<unsigned> Verbosity;

extern cl::opt<JumpTableSupportLevel> JumpTables;

static cl::opt<bool> JTFootprintOnlyPIC(
    "jt-footprint-optimize-for-icache",
    cl::desc("with jt-footprint-reduction, only process PIC jumptables and turn"
             " off other transformations that increase code size"),
    cl::init(false), cl::ZeroOrMore, cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

void JTFootprintReduction::checkOpportunities(BinaryFunction &Function,
                                              DataflowInfoManager &Info) {
  BinaryContext &BC = Function.getBinaryContext();
  std::map<JumpTable *, uint64_t> AllJTs;

  for (BinaryBasicBlock &BB : Function) {
    for (MCInst &Inst : BB) {
      JumpTable *JumpTable = Function.getJumpTable(Inst);
      if (!JumpTable)
        continue;

      AllJTs[JumpTable] += BB.getKnownExecutionCount();
      ++IndJmps;

      if (BlacklistedJTs.count(JumpTable)) {
        ++IndJmpsDenied;
        continue;
      }

      uint64_t Scale;
      // Try a standard indirect jump matcher
      std::unique_ptr<MCPlusBuilder::MCInstMatcher> IndJmpMatcher =
          BC.MIB->matchIndJmp(BC.MIB->matchAnyOperand(),
                              BC.MIB->matchImm(Scale), BC.MIB->matchReg(),
                              BC.MIB->matchAnyOperand());
      if (!opts::JTFootprintOnlyPIC &&
          IndJmpMatcher->match(*BC.MRI, *BC.MIB,
                               MutableArrayRef<MCInst>(&*BB.begin(), &Inst + 1),
                               -1) &&
          Scale == 8) {
        if (Info.getLivenessAnalysis().scavengeRegAfter(&Inst))
          continue;
        BlacklistedJTs.insert(JumpTable);
        ++IndJmpsDenied;
        ++NumJTsNoReg;
        continue;
      }

      // Try a PIC matcher. The pattern we are looking for is a PIC JT ind jmp:
      //    addq    %rdx, %rsi
      //    addq    %rdx, %rdi
      //    leaq    DATAat0x402450(%rip), %r11
      //    movslq  (%r11,%rdx,4), %rcx
      //    addq    %r11, %rcx
      //    jmpq    *%rcx # JUMPTABLE @0x402450
      MCPhysReg BaseReg1;
      MCPhysReg BaseReg2;
      uint64_t Offset;
      std::unique_ptr<MCPlusBuilder::MCInstMatcher> PICIndJmpMatcher =
          BC.MIB->matchIndJmp(BC.MIB->matchAdd(
              BC.MIB->matchReg(BaseReg1),
              BC.MIB->matchLoad(BC.MIB->matchReg(BaseReg2),
                                BC.MIB->matchImm(Scale), BC.MIB->matchReg(),
                                BC.MIB->matchImm(Offset))));
      std::unique_ptr<MCPlusBuilder::MCInstMatcher> PICBaseAddrMatcher =
          BC.MIB->matchIndJmp(
              BC.MIB->matchAdd(BC.MIB->matchLoadAddr(BC.MIB->matchSymbol()),
                               BC.MIB->matchAnyOperand()));
      if (!PICIndJmpMatcher->match(
              *BC.MRI, *BC.MIB,
              MutableArrayRef<MCInst>(&*BB.begin(), &Inst + 1), -1) ||
          Scale != 4 || BaseReg1 != BaseReg2 || Offset != 0 ||
          !PICBaseAddrMatcher->match(
              *BC.MRI, *BC.MIB,
              MutableArrayRef<MCInst>(&*BB.begin(), &Inst + 1), -1)) {
        BlacklistedJTs.insert(JumpTable);
        ++IndJmpsDenied;
        ++NumJTsBadMatch;
        continue;
      }
    }
  }

  // Statistics only
  for (const auto &JTFreq : AllJTs) {
    JumpTable *JT = JTFreq.first;
    uint64_t CurScore = JTFreq.second;
    TotalJTScore += CurScore;
    if (!BlacklistedJTs.count(JT)) {
      OptimizedScore += CurScore;
      if (JT->EntrySize == 8)
        BytesSaved += JT->getSize() >> 1;
    }
  }
  TotalJTs += AllJTs.size();
  TotalJTsDenied += BlacklistedJTs.size();
}

bool JTFootprintReduction::tryOptimizeNonPIC(
    BinaryContext &BC, BinaryBasicBlock &BB, BinaryBasicBlock::iterator Inst,
    uint64_t JTAddr, JumpTable *JumpTable, DataflowInfoManager &Info) {
  if (opts::JTFootprintOnlyPIC)
    return false;

  MCOperand Base;
  uint64_t Scale;
  MCPhysReg Index;
  MCOperand Offset;
  std::unique_ptr<MCPlusBuilder::MCInstMatcher> IndJmpMatcher =
      BC.MIB->matchIndJmp(BC.MIB->matchAnyOperand(Base),
                          BC.MIB->matchImm(Scale), BC.MIB->matchReg(Index),
                          BC.MIB->matchAnyOperand(Offset));
  if (!IndJmpMatcher->match(*BC.MRI, *BC.MIB,
                            MutableArrayRef<MCInst>(&*BB.begin(), &*Inst + 1),
                            -1))
    return false;

  assert(Scale == 8 && "Wrong scale");

  Scale = 4;
  IndJmpMatcher->annotate(*BC.MIB, "DeleteMe");

  LivenessAnalysis &LA = Info.getLivenessAnalysis();
  MCPhysReg Reg = LA.scavengeRegAfter(&*Inst);
  assert(Reg != 0 && "Register scavenger failed!");
  MCOperand RegOp = MCOperand::createReg(Reg);
  SmallVector<MCInst, 4> NewFrag;

  BC.MIB->createIJmp32Frag(NewFrag, Base, MCOperand::createImm(Scale),
                           MCOperand::createReg(Index), Offset, RegOp);
  BC.MIB->setJumpTable(NewFrag.back(), JTAddr, Index);

  JumpTable->OutputEntrySize = 4;

  BB.replaceInstruction(Inst, NewFrag.begin(), NewFrag.end());
  return true;
}

bool JTFootprintReduction::tryOptimizePIC(BinaryContext &BC,
                                          BinaryBasicBlock &BB,
                                          BinaryBasicBlock::iterator Inst,
                                          uint64_t JTAddr, JumpTable *JumpTable,
                                          DataflowInfoManager &Info) {
  MCPhysReg BaseReg;
  uint64_t Scale;
  MCPhysReg Index;
  MCOperand Offset;
  MCOperand JumpTableRef;
  std::unique_ptr<MCPlusBuilder::MCInstMatcher> PICIndJmpMatcher =
      BC.MIB->matchIndJmp(BC.MIB->matchAdd(
          BC.MIB->matchLoadAddr(BC.MIB->matchAnyOperand(JumpTableRef)),
          BC.MIB->matchLoad(BC.MIB->matchReg(BaseReg), BC.MIB->matchImm(Scale),
                            BC.MIB->matchReg(Index),
                            BC.MIB->matchAnyOperand())));
  if (!PICIndJmpMatcher->match(
          *BC.MRI, *BC.MIB, MutableArrayRef<MCInst>(&*BB.begin(), &*Inst + 1),
          -1))
    return false;

  assert(Scale == 4 && "Wrong scale");

  PICIndJmpMatcher->annotate(*BC.MIB, "DeleteMe");

  MCOperand RegOp = MCOperand::createReg(BaseReg);
  SmallVector<MCInst, 4> NewFrag;

  BC.MIB->createIJmp32Frag(NewFrag, MCOperand::createReg(0),
                           MCOperand::createImm(Scale),
                           MCOperand::createReg(Index), JumpTableRef, RegOp);
  BC.MIB->setJumpTable(NewFrag.back(), JTAddr, Index);

  JumpTable->OutputEntrySize = 4;
  // DePICify
  JumpTable->Type = JumpTable::JTT_NORMAL;

  BB.replaceInstruction(Inst, NewFrag.begin(), NewFrag.end());
  return true;
}

void JTFootprintReduction::optimizeFunction(BinaryFunction &Function,
                                            DataflowInfoManager &Info) {
  BinaryContext &BC = Function.getBinaryContext();
  for (BinaryBasicBlock &BB : Function) {
    if (!BB.getNumNonPseudos())
      continue;

    auto IndJmpRI = BB.getLastNonPseudo();
    auto IndJmp = std::prev(IndJmpRI.base());
    const uint64_t JTAddr = BC.MIB->getJumpTable(*IndJmp);

    if (!JTAddr)
      continue;

    JumpTable *JumpTable = Function.getJumpTable(*IndJmp);
    if (BlacklistedJTs.count(JumpTable))
      continue;

    if (tryOptimizeNonPIC(BC, BB, IndJmp, JTAddr, JumpTable, Info) ||
        tryOptimizePIC(BC, BB, IndJmp, JTAddr, JumpTable, Info)) {
      Modified.insert(&Function);
      continue;
    }

    llvm_unreachable("Should either optimize PIC or NonPIC successfully");
  }

  if (!Modified.count(&Function))
    return;

  for (BinaryBasicBlock &BB : Function)
    for (auto I = BB.begin(); I != BB.end();)
      if (BC.MIB->hasAnnotation(*I, "DeleteMe"))
        I = BB.eraseInstruction(I);
      else
        ++I;
}

void JTFootprintReduction::runOnFunctions(BinaryContext &BC) {
  if (opts::JumpTables == JTS_BASIC && BC.HasRelocations)
    return;

  std::unique_ptr<RegAnalysis> RA;
  std::unique_ptr<BinaryFunctionCallGraph> CG;
  if (!opts::JTFootprintOnlyPIC) {
    CG.reset(new BinaryFunctionCallGraph(buildCallGraph(BC)));
    RA.reset(new RegAnalysis(BC, &BC.getBinaryFunctions(), &*CG));
  }
  for (auto &BFIt : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFIt.second;

    if (!Function.isSimple() || Function.isIgnored())
      continue;

    if (Function.getKnownExecutionCount() == 0)
      continue;

    DataflowInfoManager Info(Function, RA.get(), nullptr);
    BlacklistedJTs.clear();
    checkOpportunities(Function, Info);
    optimizeFunction(Function, Info);
  }

  if (TotalJTs == TotalJTsDenied) {
    outs() << "BOLT-INFO: JT Footprint reduction: no changes were made.\n";
    return;
  }

  outs() << "BOLT-INFO: JT Footprint reduction stats (simple funcs only):\n";
  if (OptimizedScore)
    outs() << format("\t   %.2lf%%", (OptimizedScore * 100.0 / TotalJTScore))
           << " of dynamic JT entries were reduced.\n";
  outs() << "\t   " << TotalJTs - TotalJTsDenied << " of " << TotalJTs
         << " jump tables affected.\n";
  outs() << "\t   " << IndJmps - IndJmpsDenied << " of " << IndJmps
         << " indirect jumps to JTs affected.\n";
  outs() << "\t   " << NumJTsBadMatch
         << " JTs discarded due to unsupported jump pattern.\n";
  outs() << "\t   " << NumJTsNoReg
         << " JTs discarded due to register unavailability.\n";
  outs() << "\t   " << BytesSaved << " bytes saved.\n";
}

} // namespace bolt
} // namespace llvm
