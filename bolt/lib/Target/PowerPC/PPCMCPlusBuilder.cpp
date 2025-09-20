//===- bolt/Target/PowerPC/PPCMCPlusBuilder.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides PowerPC-specific MCPlus builder.
//
//===----------------------------------------------------------------------===//

#include "bolt/Target/PowerPC/PPCMCPlusBuilder.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"

using namespace llvm;
using namespace bolt;

// Create instructions to push two registers onto the stack
void PPCMCPlusBuilder::createPushRegisters(MCInst &Inst1, MCInst &Inst2,
                                           MCPhysReg Reg1, MCPhysReg /*Reg2*/) {

  Inst1.clear();
  Inst1.setOpcode(PPC::STDU);
  Inst1.addOperand(MCOperand::createReg(PPC::R1)); // destination (SP)
  Inst1.addOperand(MCOperand::createReg(PPC::R1)); // base (SP)
  Inst1.addOperand(MCOperand::createImm(-16));     // offset

  Inst2.clear();
  Inst2.setOpcode(PPC::STD);
  Inst2.addOperand(MCOperand::createReg(Reg1));    // source register
  Inst2.addOperand(MCOperand::createReg(PPC::R1)); // base (SP)
  Inst2.addOperand(MCOperand::createImm(0));       // offset
}

bool PPCMCPlusBuilder::shouldRecordCodeRelocation(unsigned Type) const {
  switch (Type) {
  case ELF::R_PPC64_REL24:
  case ELF::R_PPC64_REL14:
    return true;
  default:
    return false;
  }
}

IndirectBranchType PPCMCPlusBuilder::analyzeIndirectBranch(
    MCInst &Instruction, InstructionIterator Begin, InstructionIterator End,
    const unsigned PtrSize, MCInst *&MemLocInstrOut, unsigned &BaseRegNumOut,
    unsigned &IndexRegNumOut, int64_t &DispValueOut, const MCExpr *&DispExprOut,
    MCInst *&PCRelBaseOut, MCInst *&FixedEntryLoadInstr) const {
  (void)Instruction;
  MemLocInstrOut = nullptr;
  BaseRegNumOut = 0;
  IndexRegNumOut = 0;
  DispValueOut = 0;
  DispExprOut = nullptr;
  PCRelBaseOut = nullptr;
  FixedEntryLoadInstr = nullptr;
  return IndirectBranchType::UNKNOWN;
}

namespace llvm {
namespace bolt {

MCPlusBuilder *createPowerPCMCPlusBuilder(const MCInstrAnalysis *Analysis,
                                          const MCInstrInfo *Info,
                                          const MCRegisterInfo *RegInfo,
                                          const MCSubtargetInfo *STI) {
  return new PPCMCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm