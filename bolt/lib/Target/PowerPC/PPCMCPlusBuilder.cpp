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
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#define GET_INSTRINFO_ENUM
#include "llvm/Target/PowerPC/PPCGenInstrInfo.inc"
#define GET_REGINFO_ENUM
#include "llvm/Target/PowerPC/PPCGenRegisterInfo.inc"

namespace llvm {
namespace bolt {

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

MCPlusBuilder *createPowerPCMCPlusBuilder(const MCInstrAnalysis *Analysis,
                                          const MCInstrInfo *Info,
                                          const MCRegisterInfo *RegInfo,
                                          const MCSubtargetInfo *STI) {
  return new PPCMCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm
