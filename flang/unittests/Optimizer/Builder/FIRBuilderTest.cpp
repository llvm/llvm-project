//===- FIRBuilderTest.cpp -- FIRBuilder unit tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"

struct FIRBuilderTest : public testing::Test {
public:
  void SetUp() override {
    fir::KindMapping kindMap(&context);
    mlir::OpBuilder builder(&context);
    firBuilder = std::make_unique<fir::FirOpBuilder>(builder, kindMap);
    fir::support::loadDialects(context);
  }

  fir::FirOpBuilder &getBuilder() { return *firBuilder; }

  mlir::MLIRContext context;
  std::unique_ptr<fir::FirOpBuilder> firBuilder;
};

static arith::CmpIOp createCondition(fir::FirOpBuilder &builder) {
  auto loc = builder.getUnknownLoc();
  auto zero1 = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto zero2 = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  return builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, zero1, zero2);
}

//===----------------------------------------------------------------------===//
// IfBuilder tests
//===----------------------------------------------------------------------===//

TEST_F(FIRBuilderTest, genIfThen) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfThen(loc, cdt);
  EXPECT_FALSE(ifBuilder.getIfOp().thenRegion().empty());
  EXPECT_TRUE(ifBuilder.getIfOp().elseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfThenElse) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfThenElse(loc, cdt);
  EXPECT_FALSE(ifBuilder.getIfOp().thenRegion().empty());
  EXPECT_FALSE(ifBuilder.getIfOp().elseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfWithThen) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfOp(loc, {}, cdt, false);
  EXPECT_FALSE(ifBuilder.getIfOp().thenRegion().empty());
  EXPECT_TRUE(ifBuilder.getIfOp().elseRegion().empty());
}

TEST_F(FIRBuilderTest, genIfWithThenAndElse) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto cdt = createCondition(builder);
  auto ifBuilder = builder.genIfOp(loc, {}, cdt, true);
  EXPECT_FALSE(ifBuilder.getIfOp().thenRegion().empty());
  EXPECT_FALSE(ifBuilder.getIfOp().elseRegion().empty());
}

//===----------------------------------------------------------------------===//
// Helper functions tests
//===----------------------------------------------------------------------===//

TEST_F(FIRBuilderTest, genIsNotNull) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto dummyValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto res = builder.genIsNotNull(loc, dummyValue);
  EXPECT_TRUE(mlir::isa<arith::CmpIOp>(res.getDefiningOp()));
  auto cmpOp = dyn_cast<arith::CmpIOp>(res.getDefiningOp());
  EXPECT_EQ(arith::CmpIPredicate::ne, cmpOp.predicate());
}

TEST_F(FIRBuilderTest, genIsNull) {
  auto builder = getBuilder();
  auto loc = builder.getUnknownLoc();
  auto dummyValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), 0);
  auto res = builder.genIsNull(loc, dummyValue);
  EXPECT_TRUE(mlir::isa<arith::CmpIOp>(res.getDefiningOp()));
  auto cmpOp = dyn_cast<arith::CmpIOp>(res.getDefiningOp());
  EXPECT_EQ(arith::CmpIPredicate::eq, cmpOp.predicate());
}
