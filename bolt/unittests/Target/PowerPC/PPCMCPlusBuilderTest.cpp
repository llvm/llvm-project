//===- bolt/unittest/Target/PowerPC/PPCMCPlusBuilderTest.cpp
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Target/PowerPC/PPCMCPlusBuilder.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/MC/MCInst.h"
#include "gtest/gtest.h"
#define GET_INSTRINFO_ENUM
#include "llvm/Target/PowerPC/PPCGenInstrInfo.inc"
#define GET_REGINFO_ENUM
#include "llvm/Target/PowerPC/PPCGenRegisterInfo.inc"

using namespace llvm;
using namespace bolt;

namespace {

TEST(PPCMCPlusBuilderTest, CreatePushRegisters) {
  // Set up dummy input registers
  MCInst Inst1, Inst2;
  MCPhysReg Reg1 = PPC::R3; // Arbitary register

  // Call the method under test
  PPCMCPlusBuilder::createPushRegisters(Inst1, Inst2, Reg1, /*Reg2=*/PPC::R4);

  // Check Inst1 is STDU R1, R1, -16
  EXPECT_EQ(Inst1.getOpcode(), PPC::STDU);
  ASSERT_EQ(Inst1.getNumOperands(), 3u);
  EXPECT_TRUE(Inst1.getOperand(0).isReg());
  EXPECT_EQ(Inst1.getOperand(0).getReg(), PPC::R1);
  EXPECT_TRUE(Inst1.getOperand(1).isReg());
  EXPECT_EQ(Inst1.getOperand(1).getReg(), PPC::R1);
  EXPECT_TRUE(Inst1.getOperand(2).isImm());
  EXPECT_EQ(Inst1.getOperand(2).getImm(), -16);

  // Check Inst2 is STD Reg1, R1, 0
  EXPECT_EQ(Inst2.getOpcode(), PPC::STD);
  ASSERT_EQ(Inst2.getNumOperands(), 3u);
  EXPECT_TRUE(Inst2.getOperand(0).isReg());
  EXPECT_EQ(Inst2.getOperand(0).getReg(), Reg1);
  EXPECT_TRUE(Inst2.getOperand(1).isReg());
  EXPECT_EQ(Inst2.getOperand(1).getReg(), PPC::R1);
  EXPECT_TRUE(Inst2.getOperand(2).isImm());
  EXPECT_EQ(Inst2.getOperand(2).getImm(), 0);
}

} // end anonymous namespace