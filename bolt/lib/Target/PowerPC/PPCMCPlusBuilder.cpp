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

static inline unsigned opc(const MCInst &I) { return I.getOpcode(); }

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

bool PPCMCPlusBuilder::hasPCRelOperand(const MCInst &I) const {
  switch (opc(I)) {
  case PPC::BL:
  case PPC::BLA:
  case PPC::B:
  case PPC::BA:
  case PPC::BC:
    return true;
  default:
    return false;
  }
}

int PPCMCPlusBuilder::getPCRelOperandNum(const MCInst &I) const {
  return hasPCRelOperand(I) ? 0 : -1;
}

int PPCMCPlusBuilder::getMemoryOperandNo(const MCInst & /*Inst*/) const {
  return -1;
}

void PPCMCPlusBuilder::replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                                           MCContext *Ctx) const {
  // TODO: Implement PPC branch target replacement
  (void)Inst;
  (void)TBB;
  (void)Ctx;
}

const MCSymbol *PPCMCPlusBuilder::getTargetSymbol(const MCInst &Inst,
                                                  unsigned OpNum) const {
  (void)Inst;
  (void)OpNum;
  return nullptr;
}

bool PPCMCPlusBuilder::convertJmpToTailCall(MCInst &Inst) {
  (void)Inst;
  return false;
}

bool PPCMCPlusBuilder::isCall(const MCInst &I) const {
  switch (opc(I)) {
  case PPC::BL:    // branch with link (relative)
  case PPC::BLA:   // absolute with link
  case PPC::BCL:   // conditional with link (rare for calls, but safe)
  case PPC::BCTRL: // branch to CTR with link (indirect call)
    return true;
  default:
    return false;
  }
}

bool PPCMCPlusBuilder::isTailCall(const MCInst &I) const {
  (void)I;
  return false;
}

bool PPCMCPlusBuilder::isReturn(const MCInst & /*Inst*/) const { return false; }

bool PPCMCPlusBuilder::isConditionalBranch(const MCInst &I) const {
  switch (opc(I)) {
  case PPC::BC: // branch conditional
    return true;
  default:
    return false;
  }
}

bool PPCMCPlusBuilder::isUnconditionalBranch(const MCInst &I) const {
  switch (opc(I)) {
  case PPC::B:    // branch
  case PPC::BA:   // absolute branch
  case PPC::BCTR: // branch to CTR (no link) – often tail call
  case PPC::BCLR: // branch to LR  (no link)
    return true;
  default:
    return false;
  }
}

// Disable “conditional tail call” path for now.
const MCInst *PPCMCPlusBuilder::getConditionalTailCall(const MCInst &) const {
  return nullptr;
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