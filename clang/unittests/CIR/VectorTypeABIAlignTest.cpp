//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for VectorType::getABIAlignment.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace cir;

namespace {

class VectorTypeABIAlignTest : public ::testing::Test {
protected:
  VectorTypeABIAlignTest() { context.loadDialect<cir::CIRDialect>(); }

  MLIRContext context;

  uint64_t abiAlign(mlir::Type elementType, uint64_t size) {
    cir::VectorType ty = cir::VectorType::get(elementType, size);
    OpBuilder builder(&context);
    auto module = ModuleOp::create(builder.getUnknownLoc());
    mlir::DataLayout dl(module);
    uint64_t align = dl.getTypeABIAlignment(ty);
    module->erase();
    return align;
  }

  mlir::Type f32() { return cir::SingleType::get(&context); }
  mlir::Type f64() { return cir::DoubleType::get(&context); }
};

// The hook answers in bytes, so a vector holding one double aligns to 8 and
// not to its 64-bit size.
TEST_F(VectorTypeABIAlignTest, AlignmentIsInBytes) {
  EXPECT_EQ(abiAlign(f64(), 1), 8u);
  EXPECT_EQ(abiAlign(f32(), 1), 4u);
}

// A size that is already a power of two keeps it rather than rounding to the
// next one up.
TEST_F(VectorTypeABIAlignTest, ExactPowerOfTwoSizeIsUnchanged) {
  EXPECT_EQ(abiAlign(f32(), 4), 16u);
  EXPECT_EQ(abiAlign(f32(), 8), 32u);
  EXPECT_EQ(abiAlign(f32(), 16), 64u);
}

// An element count that is not a power of two rounds the byte size up.
TEST_F(VectorTypeABIAlignTest, NonPowerOfTwoElementCountRoundsUp) {
  EXPECT_EQ(abiAlign(f32(), 3), 16u);
  EXPECT_EQ(abiAlign(f64(), 3), 32u);
}

} // namespace
