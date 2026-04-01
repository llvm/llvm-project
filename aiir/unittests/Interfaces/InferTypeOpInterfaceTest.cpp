//===- InferTypeOpInterfaceTest.cpp - Unit Test for type interface --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/IR/ImplicitLocOpBuilder.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Parser/Parser.h"

#include <gtest/gtest.h>

using namespace aiir;

class ValueShapeRangeTest : public testing::Test {
protected:
  void SetUp() override {
    const char *ir = R"AIIR(
      func.func @map(%arg : tensor<1xi64>) {
        %0 = arith.constant dense<[10]> : tensor<1xi64>
        %1 = arith.addi %arg, %0 : tensor<1xi64>
        return
      }
    )AIIR";

    registry.insert<func::FuncDialect, arith::ArithDialect>();
    ctx.appendDialectRegistry(registry);
    module = parseSourceString<ModuleOp>(ir, &ctx);
    assert(module);
    mapFn = cast<func::FuncOp>(module->front());
  }

  // Create ValueShapeRange on the arith.addi operation.
  ValueShapeRange addiRange() {
    auto &fnBody = mapFn.getBody();
    return std::next(fnBody.front().begin())->getOperands();
  }

  DialectRegistry registry;
  AIIRContext ctx;
  OwningOpRef<ModuleOp> module;
  func::FuncOp mapFn;
};

TEST_F(ValueShapeRangeTest, ShapesFromValues) {
  ValueShapeRange range = addiRange();

  EXPECT_FALSE(range.getValueAsShape(0));
  ASSERT_TRUE(range.getValueAsShape(1));
  EXPECT_TRUE(range.getValueAsShape(1).hasRank());
  EXPECT_EQ(range.getValueAsShape(1).getRank(), 1);
  EXPECT_EQ(range.getValueAsShape(1).getDimSize(0), 10);
  EXPECT_EQ(range.getShape(1).getRank(), 1);
  EXPECT_EQ(range.getShape(1).getDimSize(0), 1);
}

TEST_F(ValueShapeRangeTest, MapValuesToShapes) {
  ValueShapeRange range = addiRange();
  ShapedTypeComponents fixed(SmallVector<int64_t>{30});
  auto mapping = [&](Value val) -> ShapeAdaptor {
    if (val == mapFn.getArgument(0))
      return &fixed;
    return nullptr;
  };
  range.setValueToShapeMapping(mapping);

  ASSERT_TRUE(range.getValueAsShape(0));
  EXPECT_TRUE(range.getValueAsShape(0).hasRank());
  EXPECT_EQ(range.getValueAsShape(0).getRank(), 1);
  EXPECT_EQ(range.getValueAsShape(0).getDimSize(0), 30);
  ASSERT_TRUE(range.getValueAsShape(1));
  EXPECT_TRUE(range.getValueAsShape(1).hasRank());
  EXPECT_EQ(range.getValueAsShape(1).getRank(), 1);
  EXPECT_EQ(range.getValueAsShape(1).getDimSize(0), 10);
}

TEST_F(ValueShapeRangeTest, SettingShapes) {
  ShapedTypeComponents shape(SmallVector<int64_t>{10, 20});
  ValueShapeRange range = addiRange();
  auto mapping = [&](Value val) -> ShapeAdaptor {
    if (val == mapFn.getArgument(0))
      return &shape;
    return nullptr;
  };
  range.setOperandShapeMapping(mapping);

  ASSERT_TRUE(range.getShape(0));
  EXPECT_EQ(range.getShape(0).getRank(), 2);
  EXPECT_EQ(range.getShape(0).getDimSize(0), 10);
  EXPECT_EQ(range.getShape(0).getDimSize(1), 20);
  ASSERT_TRUE(range.getShape(1));
  EXPECT_EQ(range.getShape(1).getRank(), 1);
  EXPECT_EQ(range.getShape(1).getDimSize(0), 1);
  EXPECT_FALSE(range.getShape(2));
}
