//===- StopTest.cpp -- Stop runtime builder unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genExitTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value status = firBuilder->createIntegerConstant(loc, i32Ty, 0);
  fir::runtime::genExit(*firBuilder, loc, status);
  mlir::Block *block = firBuilder->getBlock();
  EXPECT_TRUE(block) << "Failed to retrieve the block!";
  checkBlockForCallOp(block, "_FortranAExit", 1);
}
