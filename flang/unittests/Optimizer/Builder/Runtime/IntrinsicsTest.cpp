//===-- unittests/Optimizer/Builder/Runtime/IntrinsicsTest.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Intrinsics.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

TEST_F(RuntimeCallTest, genGetGID) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value result = fir::runtime::genGetGID(*firBuilder, loc);
  checkCallOp(result.getDefiningOp(), "_FortranAGetGID", /*nbArgs=*/0,
      /*addLocArgs=*/false);
}

TEST_F(RuntimeCallTest, genGetUID) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value result = fir::runtime::genGetUID(*firBuilder, loc);
  checkCallOp(result.getDefiningOp(), "_FortranAGetUID", /*nbArgs=*/0,
      /*addLocArgs=*/false);
}
