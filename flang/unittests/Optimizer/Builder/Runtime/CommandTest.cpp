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

TEST_F(RuntimeCallTest, genGetCommandArgument) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Type intTy = firBuilder->getDefaultIntegerType();
  mlir::Type charTy = fir::BoxType::get(firBuilder->getNoneType());
  mlir::Value number = firBuilder->create<fir::UndefOp>(loc, intTy);
  mlir::Value value = firBuilder->create<fir::UndefOp>(loc, charTy);
  mlir::Value errmsg = firBuilder->create<fir::UndefOp>(loc, charTy);
  // genGetCommandArgument expects `length` and `status` to be memory references
  mlir::Value length = firBuilder->create<fir::AllocaOp>(loc, intTy);
  mlir::Value status = firBuilder->create<fir::AllocaOp>(loc, intTy);

  fir::runtime::genGetCommandArgument(
      *firBuilder, loc, number, value, length, status, errmsg);
  checkCallOpFromResultBox(
      value, "_FortranAArgumentValue", /*nbArgs=*/3, /*addLocArgs=*/false);
  mlir::Block *block = firBuilder->getBlock();
  EXPECT_TRUE(block) << "Failed to retrieve the block!";
  checkBlockForCallOp(block, "_FortranAArgumentLength", /*nbArgs=*/1);
}

TEST_F(RuntimeCallTest, genGetEnvironmentVariable) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Type intTy = firBuilder->getDefaultIntegerType();
  mlir::Type charTy = fir::BoxType::get(firBuilder->getNoneType());
  mlir::Value number = firBuilder->create<fir::UndefOp>(loc, intTy);
  mlir::Value value = firBuilder->create<fir::UndefOp>(loc, charTy);
  mlir::Value trimName = firBuilder->create<fir::UndefOp>(loc, i1Ty);
  mlir::Value errmsg = firBuilder->create<fir::UndefOp>(loc, charTy);
  // genGetCommandArgument expects `length` and `status` to be memory references
  mlir::Value length = firBuilder->create<fir::AllocaOp>(loc, intTy);
  mlir::Value status = firBuilder->create<fir::AllocaOp>(loc, intTy);

  fir::runtime::genGetEnvironmentVariable(
      *firBuilder, loc, number, value, length, status, trimName, errmsg);
  checkCallOpFromResultBox(
      value, "_FortranAEnvVariableValue", /*nbArgs=*/6, /*addLocArgs=*/false);
  mlir::Block *block = firBuilder->getBlock();
  EXPECT_TRUE(block) << "Failed to retrieve the block!";
  checkBlockForCallOp(block, "_FortranAEnvVariableLength", /*nbArgs=*/4);
}
