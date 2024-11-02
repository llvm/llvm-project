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
  mlir::Value number = firBuilder->create<fir::UndefOp>(loc, intTy);
  mlir::Value value = firBuilder->create<fir::UndefOp>(loc, boxTy);
  mlir::Value length = firBuilder->create<fir::UndefOp>(loc, boxTy);
  mlir::Value errmsg = firBuilder->create<fir::UndefOp>(loc, boxTy);
  mlir::Value result = fir::runtime::genGetCommandArgument(
      *firBuilder, loc, number, value, length, errmsg);
  checkCallOp(result.getDefiningOp(), "_FortranAGetCommandArgument",
      /*nbArgs=*/4,
      /*addLocArgs=*/true);
}

TEST_F(RuntimeCallTest, genEnvVariableValue) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Type charTy = fir::BoxType::get(firBuilder->getNoneType());
  mlir::Value name = firBuilder->create<fir::UndefOp>(loc, charTy);
  mlir::Value value = firBuilder->create<fir::UndefOp>(loc, charTy);
  mlir::Value trimName = firBuilder->create<fir::UndefOp>(loc, i1Ty);
  mlir::Value errmsg = firBuilder->create<fir::UndefOp>(loc, charTy);
  mlir::Value result = fir::runtime::genEnvVariableValue(
      *firBuilder, loc, name, value, trimName, errmsg);
  checkCallOp(result.getDefiningOp(), "_FortranAEnvVariableValue", /*nbArgs=*/4,
      /*addLocArgs=*/true);
}

TEST_F(RuntimeCallTest, genEnvVariableLength) {
  mlir::Location loc = firBuilder->getUnknownLoc();
  mlir::Type charTy = fir::BoxType::get(firBuilder->getNoneType());
  mlir::Value name = firBuilder->create<fir::UndefOp>(loc, charTy);
  mlir::Value trimName = firBuilder->create<fir::UndefOp>(loc, i1Ty);
  mlir::Value result =
      fir::runtime::genEnvVariableLength(*firBuilder, loc, name, trimName);
  checkCallOp(result.getDefiningOp(), "_FortranAEnvVariableLength",
      /*nbArgs=*/2, /*addLocArgs=*/true);
}
