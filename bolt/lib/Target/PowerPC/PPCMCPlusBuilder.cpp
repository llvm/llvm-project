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

static const MCSymbol *getBranchTargetSymbol(const MCInst &I) {
  // For B/BC the last operand is a branch target (expr)
  if (I.getNumOperands() == 0)
    return nullptr;
  const MCOperand &Op = I.getOperand(I.getNumOperands() - 1);
  if (!Op.isExpr())
    return nullptr;
  if (auto *SymRef = dyn_cast<MCSymbolRefExpr>(Op.getExpr()))
    return &SymRef->getSymbol();
  return nullptr;
}

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
  switch (Inst.getOpcode()) {
  // Uncoditional direct branch -> add link bit
  case PPC::B: // relative
    Inst.setOpcode(PPC::BL);
    return true;
  case PPC::BA: // absolute
    Inst.setOpcode(PPC::BLA);
    return true;

  // Indirect branch via CTR -> add link bit
  case PPC::BCTR:
    Inst.setOpcode(PPC::BCTRL);
    return true;
  // Contitional branches
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

bool PPCMCPlusBuilder::isNoop(const MCInst &Inst) const {
  // NOP on PPC is encoded as "ori r0, r0, 0"
  return Inst.getOpcode() == PPC::ORI && Inst.getOperand(0).isReg() &&
         Inst.getOperand(0).getReg() == PPC::R0 && Inst.getOperand(1).isReg() &&
         Inst.getOperand(1).getReg() == PPC::R0 && Inst.getOperand(2).isImm() &&
         Inst.getOperand(2).getImm() == 0;
}

bool PPCMCPlusBuilder::analyzeBranch(InstructionIterator Begin,
                                     InstructionIterator End,
                                     const MCSymbol *&Tgt,
                                     const MCSymbol *&Fallthrough,
                                     MCInst *&CondBr, MCInst *&UncondBr) const {
  Tgt = nullptr;
  Fallthrough = nullptr;
  CondBr = nullptr;
  UncondBr = nullptr;

  if (Begin == End)
    return false;

  // Look at the last instruction (canonical BOLT pattern)
  InstructionIterator I = End;
  --I;
  const MCInst &Last = *I;

  // Return (blr) → no branch terminator
  if (Last.getOpcode() == PPC::BLR) {
    return false;
  }

  if (isUnconditionalBranch(Last)) {
    UncondBr = const_cast<MCInst *>(&Last);
    Tgt = getBranchTargetSymbol(Last);
    // with an unconditional branch, there's no fall-through
    return false;
  }

  if (isConditionalBranch(Last)) {
    CondBr = const_cast<MCInst *>(&Last);
    Tgt = getBranchTargetSymbol(Last);
    // Assume the block has a fallthrough if no following unconditional branch.
    // (BOLT will compute actual fallthrough later once CFG is built.)
    return false;
  }

  // Otherwise: not a branch terminator (let caller treat as fallthrough/ret)
  return false;
}

bool PPCMCPlusBuilder::lowerTailCall(MCInst &Inst) { return false; }

uint64_t PPCMCPlusBuilder::analyzePLTEntry(MCInst &Instruction,
                                           InstructionIterator Begin,
                                           InstructionIterator End,
                                           uint64_t BeginPC) const {
  (void)Instruction;
  (void)Begin;
  (void)End;
  (void)BeginPC;
  return 0;
}

void PPCMCPlusBuilder::createLongTailCall(std::vector<MCInst> &Seq,
                                          const MCSymbol *Target,
                                          MCContext *Ctx) {
  (void)Seq;
  (void)Target;
  (void)Ctx;
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