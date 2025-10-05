//===- bolt/unittest/Target/PowerPC/PPCMCPlusBuilderTest.cpp
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Target/PowerPC/PPCMCPlusBuilder.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/MC/MCInst.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace bolt;

namespace {

TEST(PPCMCPlusBuilderTest, CreatePushRegisters) {

  MCInst Inst1, Inst2;
  MCPhysReg Reg1 = PPC::R3;

  PPCMCPlusBuilder::createPushRegisters(Inst1, Inst2, Reg1, /*Reg2=*/PPC::R4);

  // Check Inst is ORI R0, R0, 0
  auto ExpectNop = [](const MCInst &Inst) {
    EXPECT_EQ(Inst.getOpcode(), PPC::ORI);
    ASSERT_EQ(Inst.getNumOperands(), 3u);

    ASSERT_TRUE(Inst.getOperand(0).isReg());
    ASSERT_TRUE(Inst.getOperand(1).isReg());
    ASSERT_TRUE(Inst.getOperand(2).isImm());

    EXPECT_EQ(Inst.getOperand(0).getReg(), PPC::R0);
    EXPECT_EQ(Inst.getOperand(1).getReg(), PPC::R0);
    EXPECT_EQ(Inst.getOperand(2).getImm(), 0);
  };
  ExpectNop(Inst1);
  ExpectNop(Inst2);
}

} // end anonymous namespace