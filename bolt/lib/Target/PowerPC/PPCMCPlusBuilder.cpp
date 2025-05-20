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

#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCPhysReg.h"
#include "llvm/Target/PowerPC/PPCInstrInfo.h"
#include "llvm/Target/PowerPC/PPCRegisterInfo.h"

namespace llvm {
namespace bolt {

class PPCMCPlusBuilder : public MCPlusBuilder{
public:
    using MCPlusBuilder::MCPlusBuilder;

    // Create instructions to push two registers onto the stack
    static void createPushRegisters(MCInst &Inst1, MCInst &Inst2, MCPhysReg Reg1, MCPhysReg /*Reg2*/){

        Inst1.clear();
        Inst1.setOpcode(PPC::STDU);
        Inst1.addOperand(MCOperand::createReg(PPC::R1)); // destination (SP)
        Inst1.addOperand(MCOperand::createReg(PPC::R1)); // base (SP)
        Inst1.addOperand(MCOperand::createImm(-16));     // offset

        Inst2.clear();
        Inst2.setOpcode(PPC::STD);
        Inst2.addOperand(MCOperand::createReg(Reg1));     // source register
        Inst2.addOperand(MCOperand::createReg(PPC::R1));  // base (SP)
        Inst2.addOperand(MCOperand::createImm(0));        // offset
    }
};

} // namespace bolt
} // namespace llvm