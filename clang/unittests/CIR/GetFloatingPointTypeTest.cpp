//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for cir::getFloatingPointType.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/APFloat.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

namespace {

class GetFloatingPointTypeTest : public ::testing::Test {
protected:
  GetFloatingPointTypeTest() { context.loadDialect<cir::CIRDialect>(); }

  MLIRContext context;
};

// Every floating-point semantics CIR has a type for maps to that type.
TEST_F(GetFloatingPointTypeTest, MapsSupportedSemantics) {
  EXPECT_TRUE(mlir::isa<cir::FP16Type>(
      getFloatingPointType(llvm::APFloat::IEEEhalf(), &context)));
  EXPECT_TRUE(mlir::isa<cir::BF16Type>(
      getFloatingPointType(llvm::APFloat::BFloat(), &context)));
  EXPECT_TRUE(mlir::isa<cir::SingleType>(
      getFloatingPointType(llvm::APFloat::IEEEsingle(), &context)));
  EXPECT_TRUE(mlir::isa<cir::DoubleType>(
      getFloatingPointType(llvm::APFloat::IEEEdouble(), &context)));
  EXPECT_TRUE(mlir::isa<cir::FP80Type>(
      getFloatingPointType(llvm::APFloat::x87DoubleExtended(), &context)));
  EXPECT_TRUE(mlir::isa<cir::FP128Type>(
      getFloatingPointType(llvm::APFloat::IEEEquad(), &context)));
}

// A semantics CIR has no type for returns a null type rather than asserting.
TEST_F(GetFloatingPointTypeTest, ReturnsNullForUnsupportedSemantics) {
  EXPECT_FALSE(
      getFloatingPointType(llvm::APFloat::PPCDoubleDouble(), &context));
}

} // namespace
