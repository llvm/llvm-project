//===- NumericTest.cpp -- Numeric intrinsic runtime builder unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Numeric.h"
#include "RuntimeCallTestBase.h"
#include "gtest/gtest.h"

void testGenExponent(fir::FirOpBuilder &builder, mlir::Type resultType,
    mlir::Type xType, llvm::StringRef fctName) {
  auto loc = builder.getUnknownLoc();
  mlir::Value x = builder.create<fir::UndefOp>(loc, xType);
  mlir::Value exp = fir::runtime::genExponent(builder, loc, resultType, x);
  checkCallOp(exp.getDefiningOp(), fctName, 1, /*addLocArg=*/false);
}

TEST_F(RuntimeCallTest, genExponentTest) {
  testGenExponent(*firBuilder, i32Ty, f32Ty, "_FortranAExponent4_4");
  testGenExponent(*firBuilder, i64Ty, f32Ty, "_FortranAExponent4_8");
  testGenExponent(*firBuilder, i32Ty, f64Ty, "_FortranAExponent8_4");
  testGenExponent(*firBuilder, i64Ty, f64Ty, "_FortranAExponent8_8");
  testGenExponent(*firBuilder, i32Ty, f80Ty, "_FortranAExponent10_4");
  testGenExponent(*firBuilder, i64Ty, f80Ty, "_FortranAExponent10_8");
  testGenExponent(*firBuilder, i32Ty, f128Ty, "_FortranAExponent16_4");
  testGenExponent(*firBuilder, i64Ty, f128Ty, "_FortranAExponent16_8");
}

void testGenX(fir::FirOpBuilder &builder, mlir::Type xType,
    mlir::Value (*genFct)(fir::FirOpBuilder &, Location, mlir::Value),
    llvm::StringRef fctName) {
  auto loc = builder.getUnknownLoc();
  mlir::Value x = builder.create<fir::UndefOp>(loc, xType);
  mlir::Value val = genFct(builder, loc, x);
  checkCallOp(val.getDefiningOp(), fctName, 1, /*addLocArg=*/false);
}

TEST_F(RuntimeCallTest, genFractionTest) {
  testGenX(*firBuilder, f32Ty, fir::runtime::genFraction, "_FortranAFraction4");
  testGenX(*firBuilder, f64Ty, fir::runtime::genFraction, "_FortranAFraction8");
  testGenX(
      *firBuilder, f80Ty, fir::runtime::genFraction, "_FortranAFraction10");
  testGenX(
      *firBuilder, f128Ty, fir::runtime::genFraction, "_FortranAFraction16");
}

void testGenNearest(fir::FirOpBuilder &builder, mlir::Type xType,
    mlir::Type sType, llvm::StringRef fctName) {
  auto loc = builder.getUnknownLoc();
  mlir::Value x = builder.create<fir::UndefOp>(loc, xType);
  mlir::Value s = builder.create<fir::UndefOp>(loc, sType);
  mlir::Value nearest = fir::runtime::genNearest(builder, loc, x, s);
  checkCallOp(nearest.getDefiningOp(), fctName, 2, /*addLocArg=*/false);
  auto callOp = mlir::dyn_cast<fir::CallOp>(nearest.getDefiningOp());
  mlir::Value select = callOp.getOperands()[1];
  EXPECT_TRUE(mlir::isa<mlir::arith::SelectOp>(select.getDefiningOp()));
  auto selectOp = mlir::dyn_cast<mlir::arith::SelectOp>(select.getDefiningOp());
  mlir::Value cmp = selectOp.getCondition();
  EXPECT_TRUE(mlir::isa<mlir::arith::CmpFOp>(cmp.getDefiningOp()));
  auto cmpOp = mlir::dyn_cast<mlir::arith::CmpFOp>(cmp.getDefiningOp());
  EXPECT_EQ(s, cmpOp.getLhs());
}

TEST_F(RuntimeCallTest, genNearestTest) {
  testGenNearest(*firBuilder, f32Ty, f32Ty, "_FortranANearest4");
  testGenNearest(*firBuilder, f64Ty, f32Ty, "_FortranANearest8");
  testGenNearest(*firBuilder, f80Ty, f32Ty, "_FortranANearest10");
  testGenNearest(*firBuilder, f128Ty, f32Ty, "_FortranANearest16");
}

TEST_F(RuntimeCallTest, genRRSpacingTest) {
  testGenX(
      *firBuilder, f32Ty, fir::runtime::genRRSpacing, "_FortranARRSpacing4");
  testGenX(
      *firBuilder, f64Ty, fir::runtime::genRRSpacing, "_FortranARRSpacing8");
  testGenX(
      *firBuilder, f80Ty, fir::runtime::genRRSpacing, "_FortranARRSpacing10");
  testGenX(
      *firBuilder, f128Ty, fir::runtime::genRRSpacing, "_FortranARRSpacing16");
}

void testGenXI(fir::FirOpBuilder &builder, mlir::Type xType, mlir::Type iType,
    mlir::Value (*genFct)(
        fir::FirOpBuilder &, Location, mlir::Value, mlir::Value),
    llvm::StringRef fctName) {
  auto loc = builder.getUnknownLoc();
  mlir::Value x = builder.create<fir::UndefOp>(loc, xType);
  mlir::Value i = builder.create<fir::UndefOp>(loc, iType);
  mlir::Value val = genFct(builder, loc, x, i);
  checkCallOp(val.getDefiningOp(), fctName, 2, /*addLocArg=*/false);
}

TEST_F(RuntimeCallTest, genScaleTest) {
  testGenXI(
      *firBuilder, f32Ty, f32Ty, fir::runtime::genScale, "_FortranAScale4");
  testGenXI(
      *firBuilder, f64Ty, f32Ty, fir::runtime::genScale, "_FortranAScale8");
  testGenXI(
      *firBuilder, f80Ty, f32Ty, fir::runtime::genScale, "_FortranAScale10");
  testGenXI(
      *firBuilder, f128Ty, f32Ty, fir::runtime::genScale, "_FortranAScale16");
}

TEST_F(RuntimeCallTest, genSetExponentTest) {
  testGenXI(*firBuilder, f32Ty, f32Ty, fir::runtime::genSetExponent,
      "_FortranASetExponent4");
  testGenXI(*firBuilder, f64Ty, f32Ty, fir::runtime::genSetExponent,
      "_FortranASetExponent8");
  testGenXI(*firBuilder, f80Ty, f32Ty, fir::runtime::genSetExponent,
      "_FortranASetExponent10");
  testGenXI(*firBuilder, f128Ty, f32Ty, fir::runtime::genSetExponent,
      "_FortranASetExponent16");
}

TEST_F(RuntimeCallTest, genSpacingTest) {
  testGenX(*firBuilder, f32Ty, fir::runtime::genSpacing, "_FortranASpacing4");
  testGenX(*firBuilder, f64Ty, fir::runtime::genSpacing, "_FortranASpacing8");
  testGenX(*firBuilder, f80Ty, fir::runtime::genSpacing, "_FortranASpacing10");
  testGenX(*firBuilder, f128Ty, fir::runtime::genSpacing, "_FortranASpacing16");
}
