//===- CommandTest.cpp -- command line runtime builder unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Command.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genCommandArgumentCountTest) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value result = fir::runtime::genCommandArgumentCount(*firBuilder, loc);
  checkCallOp(result.getDefiningOp(), "_FortranAArgumentCount", /*nbArgs=*/0,
      /*addLocArgs=*/false);
}
