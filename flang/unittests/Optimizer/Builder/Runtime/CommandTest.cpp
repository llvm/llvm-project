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
  mlir::Type boxTy = fir::BoxType::get(firBuilder->getNoneType());
  mlir::Value number = fir::UndefOp::create(*firBuilder, loc, intTy);
  mlir::Value value = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value length = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value errmsg = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value result = fir::runtime::genGetCommandArgument(
      *firBuilder, loc, number, value, length, errmsg);
  checkCallOp(result.getDefiningOp(), "_FortranAGetCommandArgument",
      /*nbArgs=*/4,
      /*addLocArgs=*/true);
}

TEST_F(RuntimeCallTest, genGetEnvVariable) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value name = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value value = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value length = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value trimName = fir::UndefOp::create(*firBuilder, loc, i1Ty);
  mlir::Value errmsg = fir::UndefOp::create(*firBuilder, loc, boxTy);
  mlir::Value result = fir::runtime::genGetEnvVariable(
      *firBuilder, loc, name, value, length, trimName, errmsg);
  checkCallOp(result.getDefiningOp(), "_FortranAGetEnvVariable", /*nbArgs=*/5,
      /*addLocArgs=*/true);
}

TEST_F(RuntimeCallTest, genGetPID) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Value result = fir::runtime::genGetPID(*firBuilder, loc);
  checkCallOp(result.getDefiningOp(), "_FortranAGetPID", /*nbArgs=*/0,
      /*addLocArgs=*/false);
}
