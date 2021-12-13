//===- DoLoopHelper.cpp -- DoLoopHelper unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/DoLoopHelper.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include <string>

struct DoLoopHelperTest : public testing::Test {
public:
  void SetUp() {
    kindMap = std::make_unique<fir::KindMapping>(&context);
    mlir::OpBuilder builder(&context);
    firBuilder = new fir::FirOpBuilder(builder, *kindMap);
    fir::support::loadDialects(context);
  }
  void TearDown() { delete firBuilder; }

  fir::FirOpBuilder &getBuilder() { return *firBuilder; }

  mlir::MLIRContext context;
  std::unique_ptr<fir::KindMapping> kindMap;
  fir::FirOpBuilder *firBuilder;
};

void checkConstantValue(const mlir::Value &value, int64_t v) {
  EXPECT_TRUE(mlir::isa<ConstantOp>(value.getDefiningOp()));
  auto cstOp = dyn_cast<ConstantOp>(value.getDefiningOp());
  auto valueAttr = cstOp.getValue().dyn_cast_or_null<IntegerAttr>();
  EXPECT_EQ(v, valueAttr.getInt());
}

TEST_F(DoLoopHelperTest, createLoopWithCountTest) {
  auto firBuilder = getBuilder();
  fir::factory::DoLoopHelper helper(firBuilder, firBuilder.getUnknownLoc());

  auto c10 = firBuilder.createIntegerConstant(
      firBuilder.getUnknownLoc(), firBuilder.getIndexType(), 10);
  auto loop =
      helper.createLoop(c10, [&](fir::FirOpBuilder &, mlir::Value index) {});
  checkConstantValue(loop.lowerBound(), 0);
  EXPECT_TRUE(mlir::isa<arith::SubIOp>(loop.upperBound().getDefiningOp()));
  auto subOp = dyn_cast<arith::SubIOp>(loop.upperBound().getDefiningOp());
  EXPECT_EQ(c10, subOp.getLhs());
  checkConstantValue(subOp.getRhs(), 1);
  checkConstantValue(loop.getStep(), 1);
}

TEST_F(DoLoopHelperTest, createLoopWithLowerAndUpperBound) {
  auto firBuilder = getBuilder();
  fir::factory::DoLoopHelper helper(firBuilder, firBuilder.getUnknownLoc());

  auto lb = firBuilder.createIntegerConstant(
      firBuilder.getUnknownLoc(), firBuilder.getIndexType(), 1);
  auto ub = firBuilder.createIntegerConstant(
      firBuilder.getUnknownLoc(), firBuilder.getIndexType(), 20);
  auto loop =
      helper.createLoop(lb, ub, [&](fir::FirOpBuilder &, mlir::Value index) {});
  checkConstantValue(loop.getLowerBound(), 1);
  checkConstantValue(loop.getUpperBound(), 20);
  checkConstantValue(loop.getStep(), 1);
}

TEST_F(DoLoopHelperTest, createLoopWithStep) {
  auto firBuilder = getBuilder();
  fir::factory::DoLoopHelper helper(firBuilder, firBuilder.getUnknownLoc());

  auto lb = firBuilder.createIntegerConstant(
      firBuilder.getUnknownLoc(), firBuilder.getIndexType(), 1);
  auto ub = firBuilder.createIntegerConstant(
      firBuilder.getUnknownLoc(), firBuilder.getIndexType(), 20);
  auto step = firBuilder.createIntegerConstant(
      firBuilder.getUnknownLoc(), firBuilder.getIndexType(), 2);
  auto loop = helper.createLoop(
      lb, ub, step, [&](fir::FirOpBuilder &, mlir::Value index) {});
  checkConstantValue(loop.getLowerBound(), 1);
  checkConstantValue(loop.getUpperBound(), 20);
  checkConstantValue(loop.getStep(), 2);
}
